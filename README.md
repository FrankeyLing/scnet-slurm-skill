# 超算互联网 Slurm 提交技能

把 PyTorch 训练任务配置成可审查、可验证的 Slurm 作业，提交到超算互联网指定的数据中心、集群和队列，使用海光 DCU（数据中心加速器，Data Center Unit，用于训练等并行计算的加速卡）。

代码负责选择候选资源、计算分段时限、生成作业脚本、检查提交环境和核实训练产物。技能入口很短，既可供 Codex 调用，也能从命令行独立使用。训练入口及其参数由项目显式提供。

## 安装

克隆仓库，把 `skill/scnet-slurm` 目录放进 Codex 的技能目录，然后调用 `$scnet-slurm`。[技能入口](skill/scnet-slurm/SKILL.md)

```bash
git clone https://github.com/FrankeyLing/scnet-slurm-skill.git
cd scnet-slurm-skill
mkdir -p ~/.codex/skills
cp -R skill/scnet-slurm ~/.codex/skills/
```

## 命令行

在候选集群的 Linux 登录节点执行 `inspect`，按现场结果填写队列和运行环境。以下昆山演示使用合成数据。请替换 `CLUSTER`、`PARTITION`，按实机规格设置时限、核数与主机内存。检查调用提交工具的 Python 至少是 3.7。作业通过 `--setup-file` 初始化自己的运行环境。哈尔滨的环境与内存配置见 [实机记录](docs/live-smoke.md)。

```bash
CLUSTER=example-cluster
PARTITION=example-partition
SETUP_SCRIPT=/absolute/path/to/site-setup.sh
source "$SETUP_SCRIPT"
python --version
python skill/scnet-slurm/scripts/scnet.py inspect --partition "$PARTITION"
python examples/make_demo.py --center Kunshan --cluster "$CLUSTER" --partition "$PARTITION" \
  --python python --runtime-id example-vendor-module --cards 1 --wall-minutes 120 \
  --cpus-per-task 4 --memory-mib 12288 --startup-minutes 4 --probe-timeout-seconds 150 \
  --setup-file "$SETUP_SCRIPT" --output demo
python skill/scnet-slurm/scripts/scnet.py decide demo/spec.json
python skill/scnet-slurm/scripts/scnet.py render demo/spec.json demo/bundle
python skill/scnet-slurm/scripts/scnet.py preflight demo/bundle
python skill/scnet-slurm/scripts/scnet.py submit demo/bundle --segment 1 --execute
```

规划和生成只依赖 Python 3.7 或更高版本的标准库，可以离线运行。演示配置使用生成时的 Linux 路径，从预检开始也应在所选集群的登录节点操作。确认首段和续训的短跑通过后，可通过 `--chain` 提交按成功状态衔接的整条链。用 `status <作业号>` 查询调度状态，再检查输出、错误日志和本次训练生成的凭据。[配置、提交和排障](skill/scnet-slurm/references/submit.md)

运行 `python -m unittest discover -s tests -v` 可复核资源决策、断点衔接、产物凭据和提交行为。两地的合成短测记录包含调度状态、计算节点探针、训练凭据和主机内存排障。[实机验证记录](docs/live-smoke.md)

调度器响应较慢时，可在 `inspect`、`preflight`、`submit`、`status` 中明确指定
`--scheduler-timeout-seconds 120`。它控制每条调度命令的等待时间，范围为 1 至
600 秒，原默认值仍为 30 秒。程序照常完成真实预检和提交意图登记，不自动重试；
包装器的整个阶段预算需另行设置。[等待时限与升级方法](skill/scnet-slurm/references/submit.md)

## 资源与框架选择

挑选昆山和哈尔滨的候选队列时，先核实当前账号的可用性、计费方式、卡型、软件环境及墙钟时限，再用同一份数据、权重与参数做短跑。将冷启动、保存、验证、排队及跨中心传输都算进预计交付时间。小探针和易复现的短对照适合快速队列。无法安全切分的任务，或者分段排队与启动开销更高的任务，才值得优先考虑长队列。

成熟框架优先评估 DDP（分布式数据并行，Distributed Data Parallel，多张卡分别计算后同步梯度），检查官方训练入口、分布式后端、显存和真实吞吐。自研框架正式训练默认单卡。采用多卡前需要通过当前训练配置的数值一致性验收。通信探针通过只证实通信可用。使用短队列分段时，成熟框架使用它的完整状态恢复接口，自研框架先通过中断续训和连续运行的对照验收。检查点必须能恢复训练状态和原来的学习率进度。

## 许可证

MIT（麻省理工学院许可证，Massachusetts Institute of Technology License，保留版权与许可声明即可使用、修改和分发）。[完整许可证](LICENSE)
