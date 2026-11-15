# 超算互联网实机短测

2026 年 11 月 15 日在昆山和哈尔滨测试了公开仓库的 `17c837a` 版本。使用随仓库发布的 `make_demo.py` 和 `train_demo.py`，固定随机种子、全局批量及超参数，用合成数据执行两次优化器更新。初始作业与昆山复测按 6 分钟申请。[受测版本](https://github.com/FrankeyLing/scnet-slurm-skill/tree/17c837a0a607c32903ca7f1fe868442c1b14ff77)

每个作业都走工具的计划、脚本生成、调度器预检和正式提交。通过需要 `sacct` 父作业最终状态为 `COMPLETED 0:0`，计算节点探针和训练凭据都与作业号一致，且训练凭据为 `progress=2`。探针在分配的每张卡上实际执行了矩阵乘法和反向传播。双卡测试还要求两个进程都通过跨卡求和。

## 结果

| 中心 / 分区 | 卡数 | 处理器核心 / 主机内存 | 作业运行时间 | 验收记录 |
| --- | ---: | --- | ---: | --- |
| 昆山 `kshdAI` | 1 | 4 / 12 GiB | 1 分 22 秒 | `COMPLETED 0:0`，一张卡通过前后向探针，实际训练 2 步 |
| 昆山 `kshdAI` | 2 | 8 / 16 GiB | 2 分 34 秒 | `COMPLETED 0:0`，两张卡通过前后向探针，两个进程均通过跨卡求和，实际训练 2 步 |
| 哈尔滨 `hebhdnormal` | 1 | 8 / 24 GiB | 38 秒 | `COMPLETED 0:0`，一张卡通过前后向探针，实际训练 2 步 |

表中时间包含软件启动、前后向探针及两步合成训练。

## 环境复现

从对应中心的登录节点运行下面的模块命令。同一套命令也放入 `--setup-file` 指定的脚本。裸登录环境可能缺少适用的解释器。在本次会话中，昆山的裸环境为 Python 3.6，哈尔滨还遇到了默认运行时模块冲突。

```bash
# 昆山 cancon / kshdAI
source /etc/profile.d/modules.sh
module load apps/PyTorch/1.10/dtk-22.04-py37
python --version
```

昆山实测解释器为 Python 3.7.13，PyTorch 版本为 `1.10.0a0+git450cdd1-dtk22.04`，设备名报告为 `Device 66a1`，单张卡可见显存约 15.98 GiB。

```bash
# 哈尔滨 cluster_m301n1 / hebhdnormal
source /etc/profile.d/modules.sh
module purge
module load apps/PyTorch/chatglm2_py38/pytorch1.13-dtk23.04-py38
python --version
```

哈尔滨实测解释器为 Python 3.8.17，PyTorch 版本为 `1.13.1`，设备名报告为 `Vega 20`，可见显存约 15.98 GiB。成功的单卡作业申请了 8 分钟，复现时用 `--cpus-per-task 8 --memory-mib 24576 --startup-minutes 6 --probe-timeout-seconds 240`。

哈尔滨在继承提交端已经加载的训练模块时，曾在作业内再次初始化模块并报 `Cannot initialize TCL`。该次作业未进入设备探针，随后取消。重提时用绝对路径调用控制端 Python，只保留调度器和模块发现需要的环境。训练模块由作业的 `setup_file` 加载。以下路径为当时的现场值，复用前先在目标登录节点确认。

```bash
env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" SHELL=/bin/bash \
  PATH=/opt/gridview/slurm/bin:/usr/bin:/bin MODULEPATH="$MODULEPATH" \
  /public/software/apps/DeepLearning/PyTorch/chatglm2/bin/python \
  demo/bundle/scnet.py submit demo/bundle --segment 1 --execute
```

## 主机内存排障

两地最初都用 1 张卡、4 个处理器核心和 4 GiB 主机内存。昆山在运行设备探针时被调度器判为 `OUT_OF_MEMORY`。哈尔滨完成了两步并写出训练凭据，但调度器仍报告同一内存错误。判断通过时需要一起查父作业账本和产物。

哈尔滨的调度服务拒绝了 4 核配 16 GiB 的请求。分区显示 `DefMemPerCPU=3701`，调度器预检也明确报出了内存与核心的比例限制。4 核配 12 GiB 通过了预检，但作业写出两步训练凭据后仍被判为 `OUT_OF_MEMORY`。最后通过的配置同时调整了核数、主机内存和提交环境。12 GiB 的失败和 24 GiB 的成功不能用于推断该环境的最低主机内存需求。

本次检查覆盖两步合成训练和指定的设备及通信路径。项目级多卡训练数值一致性、跨作业断点续训和长跑稳定性仍需使用各自的验收方案。
