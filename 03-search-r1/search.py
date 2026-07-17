"""提供搜索工具后端，当前实现为知乎全局搜索 API。"""

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv


SEARCH_ENDPOINT = "https://developer.zhihu.com/api/v1/content/global_search"


@dataclass(frozen=True)
class SearchItem:
    """保存一条精简后的知乎搜索结果。"""

    title: str
    content: str
    url: str
    source: str


@dataclass(frozen=True)
class SearchResult:
    """保存一次搜索的结果或错误信息。"""

    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None


@dataclass
class SearchStats:
    """累计搜索请求的运行指标。"""

    requests: int = 0
    successes: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    credential_failovers: int = 0
    errors: int = 0
    latency_total: float = 0.0

    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的比例。"""
        denominator = max(self.requests, 1)
        return {
            "search/success_rate": self.successes / denominator,
            "search/timeout_rate": self.timeouts / denominator,
            "search/429_rate": self.rate_limits / denominator,
            "search/credential_failover_rate": self.credential_failovers / denominator,
            "search/error_rate": self.errors / denominator,
            "search/latency": self.latency_total / denominator,
        }


@dataclass
class ZhihuSearchClient:
    """轮转使用多组凭证，并通过有限重试执行知乎搜索。"""

    access_secrets: str | list[str]
    timeout: float = 15.0
    max_retries: int = 2
    retry_delay: float = 1.0
    stats: SearchStats = field(default_factory=SearchStats)
    _next_secret_index: int = field(default=0, init=False, repr=False)
    _rate_limited_secret_indices: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """清洗、去重凭证，并兼容直接传入单个字符串。"""
        raw_secrets = (
            [self.access_secrets]
            if isinstance(self.access_secrets, str)
            else self.access_secrets
        )
        secrets = list(dict.fromkeys(secret.strip() for secret in raw_secrets if secret.strip()))
        if not secrets:
            raise ValueError("至少需要一个知乎搜索 key")
        self.access_secrets = secrets

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "ZhihuSearchClient":
        """从逗号或换行分隔的环境变量读取一组搜索凭证。"""
        if env_path:
            load_dotenv(env_path)
        raw_secrets = (
            os.getenv("ZHIHU_SEARCH_KEYS")
            or os.getenv("ZHIHU_SEARCH_KEY")
            or os.getenv("ZHIHU_ACCESS_SECRET")
        )
        if not raw_secrets:
            raise ValueError(
                "请设置 ZHIHU_SEARCH_KEYS、ZHIHU_SEARCH_KEY 或 ZHIHU_ACCESS_SECRET"
            )
        secrets = [item.strip() for item in re.split(r"[,\n]", raw_secrets) if item.strip()]
        return cls(access_secrets=secrets, **kwargs)

    def search(self, query: str) -> SearchResult:
        """轮转 key 搜索一个 query；429 切换 key，超时和 5xx 有限重试。"""
        started = time.perf_counter()
        self.stats.requests += 1
        saw_timeout = False
        saw_rate_limit = False
        credential = self._next_credential()
        if credential is None:
            return self._error_result(started, "all search keys are rate limited", 429)
        secret_index, access_secret = credential
        attempt = 0
        while True:
            try:
                result = self._request(query, started, access_secret)
                self.stats.successes += 1
                self.stats.latency_total += result.latency
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429 and not saw_rate_limit:
                    self.stats.rate_limits += 1
                    saw_rate_limit = True
                if error.code == 429:
                    self._rate_limited_secret_indices.add(secret_index)
                    credential = self._next_credential()
                    if credential is None:
                        return self._error_result(
                            started, "all search keys are rate limited", error.code
                        )
                    secret_index, access_secret = credential
                    self.stats.credential_failovers += 1
                    continue
                if error.code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return self._error_result(started, type(error).__name__)

    def _next_credential(self) -> tuple[int, str] | None:
        """按 round-robin 顺序取下一组尚未被 429 停用的凭证。"""
        secrets = cast(list[str], self.access_secrets)
        for _ in range(len(secrets)):
            index = self._next_secret_index
            self._next_secret_index = (self._next_secret_index + 1) % len(secrets)
            if index not in self._rate_limited_secret_indices:
                return index, secrets[index]
        return None

    def _request(self, query: str, started: float, access_secret: str) -> SearchResult:
        """发出一次知乎 API 请求并解析真实响应结构。"""
        params = urllib.parse.urlencode({"Query": query, "Count": 3, "SearchDB": "all"})
        request = urllib.request.Request(
            f"{SEARCH_ENDPOINT}?{params}",
            headers={
                "Authorization": f"Bearer {access_secret}",
                "X-Request-Timestamp": str(int(time.time())),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            items = [self._parse_item(item) for item in payload["Data"]["Items"]]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    def _parse_item(self, item: dict[str, Any]) -> SearchItem:
        """从一条 API 结果中保留标题、摘要、来源和链接。"""
        source_parts = [str(item.get("ContentType") or "Zhihu")]
        if item.get("AuthorName"):
            source_parts.append(str(item["AuthorName"]))
        return SearchItem(
            title=str(item.get("Title") or "Untitled").strip(),
            content=str(item.get("ContentText") or "").strip()[:1200],
            url=str(item.get("Url") or "").strip(),
            source=" / ".join(source_parts),
        )

    def _error_result(self, started: float, message: str, status: int | None = None) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        latency = time.perf_counter() - started
        self.stats.errors += 1
        self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


def format_item(item: SearchItem, index: int) -> str:
    """把一条搜索结果格式化成完整的工具文本块。"""
    return (
        f"[{index}] Title: {item.title}\n"
        f"    Content: {item.content}\n"
        f"    Source: {item.source}\n"
        f"    URL: {item.url}"
    )
