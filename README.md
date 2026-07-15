<div align="center">

<a href="https://github.com/KMnO4-zx/llm-agent-rl-lab">
  <img src="images/llm-agent-rl-lab.png" alt="LLM Agent RL Lab" width=100% />
</a>

<h1><i>LLM Agent RL Lab</i></h1>

<p>
  复现和拆解前沿 LLM 强化学习算法，用更简单的代码和更低的 GPU 门槛，把 GRPO、OPD、GSPO、DAPO、Search-R1、Slime 等方法跑起来，方便复现。
</p>

<p>
  <img alt="Series" src="https://img.shields.io/badge/Series-10%2B%20RL%20Papers-ff6b5f?style=flat-square" />
  <img alt="LLM RL" src="https://img.shields.io/badge/LLM-RL-111111?style=flat-square" />
  <a href="https://pytrio.cn/"><img alt="PyTRIO" src="https://img.shields.io/badge/PyTRIO-Remote%20Training-d94a45?style=flat-square" /></a>
  <a href="https://swanlab.cn/"><img alt="SwanLab" src="https://img.shields.io/badge/SwanLab-Experiment%20Tracking-258f4b?style=flat-square" /></a>
  <a href="https://swanlab.cn/@kmno4/llm-agent-rl-lab/overview"><img alt="SwanLab Experiments" src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg" /></a>
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6" /></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49" /></a>
  <a href="https://github.com/KMnO4-zx/llm-agent-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-llm-agent-rl-lab&amp;label=visitors&amp;color=1283c3&amp;style=flat" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13%2B-306998?style=flat-square" />
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-cb8d00?style=flat-square" />
</p>

</div>

## 这个仓库是什么？

这是一个偏实验记录和教程的仓库。我会用 PyTRIO 复现一组和 LLM / Agent RL 相关的强化学习算法，主要做三件事：

1. 先把算法讲明白：它从哪篇论文来，解决什么问题，核心变量是什么。
2. 再用可运行代码复现：数据、reward、loss、训练循环、SwanLab 记录都放在仓库里。
3. 可能未来会做一个更友好和轻量的 Agent RL 训练框架～

我选择 PyTRIO 的原因很简单：研究这些算法时，我更想比较 loss、reward、group size、学习率和采样参数，而不是先维护一套 8 卡训练服务。PyTRIO 把训练、采样、LoRA 权重保存和远端资源管理托管掉，本地代码就可以专注在实验逻辑上。

## 文章目录

| 篇章 | 主题 | 内容 |
| --- | --- | --- |
| [第 0 篇](./00-loss-function/readme.md) | Loss Function | 用直觉解释 `importance_sampling`、`ppo`、`cispo` 分别在优化什么 |
| [第 1 篇](./01-grpo/readme.md) | GRPO | 复现 GSM8K 上的 GRPO，并比较 `importance_sampling` / `ppo` / `cispo` 三个 loss |
| [第 2 篇](./02-opd/general-opd/readme.md) | General OPD | 用 DeepMath-103K 跑通 Student 采样、Teacher 打分与 reverse KL 的最小闭环 |
| [第 2 篇](./02-opd/readme.md) | Medical OPD | 从 Medical SFT 出发，用 SAR-OPD 和 IDT-OPD 增强医疗能力，同时保持通用能力 |
| [第 3 篇]() |  | GSPO 还是 Search-R1 呢？ |


## 快速启动

如果是直接 clone 这个仓库：

```bash
git clone https://github.com/KMnO4-zx/llm-agent-rl-lab.git
cd llm-agent-rl-lab
uv sync
```

如果只想把某个 demo 脚本拎到自己的项目里跑：

```bash
uv add "datasets>=5.0.0" "matplotlib>=3.11.0" "numpy>=2.5.1" "openai>=2.44.0" "python-dotenv>=1.2.2" "pytrio==0.2.2" "swanlab>=0.8.4" "torch>=2.12.1" "tqdm>=4.68.3"
```

## 目录结构

```text
├── 00-loss-function/
│   ├── readme.md
│   └── images/
├── 01-grpo/
│   ├── 01-demo-sync.py
│   ├── 02-demo-async.py
│   ├── readme.md
│   └── images/
├── 02-opd/
│   ├── general-opd/
│   │   ├── 01-demo-sync.py
│   │   ├── 02-demo-async.py
│   │   ├── readme.md
│   │   └── images/
│   ├── 00-download-dataset.py
│   ├── 01-eval-ceval.py
│   ├── 01-eval-medical.py
│   ├── 02-medical-sft.py
│   ├── 03-medical-opd-sync.py
│   ├── 03-medical-opd-async.py
│   ├── 04-ceval-opd-async.py
│   ├── 05-interleaved-multi-teacher-opd.py
│   ├── readme.md
│   └── images/
├── images/
│   └── llm-agent-rl-lab.png
├── pyproject.toml
└── README.md
```

## License

See [LICENSE](./LICENSE).
