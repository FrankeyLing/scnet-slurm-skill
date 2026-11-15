# 计算节点与训练入口的契约

预检使用作业实际加载的 Python 环境。在已分配的计算节点上检查 PyTorch 的 HIP（异构计算可移植接口，Heterogeneous-compute Interface for Portability，海光兼容运行时向 PyTorch 暴露的加速接口）构建标识、恰好可见的卡数和每张卡的矩阵乘法及反向梯度。环境里的设备名仍使用 PyTorch 的 `cuda` 命名。可按实测填写 `--arch gfxNNN` 来断言芯片架构，架构与设备名都不能单独用来确定商品型号。

分配多卡时再用同一 Python 解释器启动单机多进程，对每个进程绑定本地设备并检验跨卡求和。这个检查覆盖设备和通信路径。正式使用 DDP（分布式数据并行，Distributed Data Parallel，多进程计算各自的数据后同步梯度）仍须满足训练框架的验收条件。整个探针有限时，且每次使用独立会合地址。测过的 PyTorch 旧版本中 `--standalone` 可能绑定固定端口。

~~~bash
python -u runtime.py probe --cards 2 --timeout-seconds 60 --out /shared/run/stage1.probe.json
~~~

训练命令由作业清单固定。把训练框架自带的完成回调或一个检查结果的适配器接到运行器，按本次实际完成的轮次或优化步写标记。起始、分段和目标进度必须使用同一计数单位。单卡或多卡的全局第零个进程保存检查点和完成标记。在使用自带启动器的成熟框架中，也可在启动器成功退出后由适配器核实产物与进度，然后写标记。

~~~python
import os
from runtime import mark_complete  # 从技能的 scripts 目录导入

# 由适配器读取实际计数和训练状态，先断言达到本段目标。
# 如约定了检查点输出，训练代码先将完整状态保存到指定位置并原子发布。
mark_complete(actual_progress, os.environ.get("SCNET_CHECKPOINT_OUT"))
~~~

同样可由适配器执行 `python runtime.py mark --progress ACTUAL`，保存检查点时再传 `--checkpoint ACTUAL_PATH`。运行器提供 `SCNET_COMPLETION_FILE`、`SCNET_COMPLETION_TOKEN` 和 `SCNET_EXPECTED_PROGRESS`，供标记工具核对。数据已保存且进度已核实之后才调用标记工具。

运行器要求训练退出码为零、写出了本次运行的新标记，并且实际进度等于本段目标。要求保存检查点时，还会复核文件非空且内容摘要吻合；检查点绝对路径、字节数、内容摘要、进度和 Slurm 作业号进入凭据。使用未占用的标记、凭据和输出检查点路径。遇到复用、校验失败或训练非零退出时，作业失败而且没有通过凭据。

续训时同时传入 `--checkpoint-in`、`--previous-receipt` 和 `--previous-progress`。运行器在启动之前核对前段通过凭据和检查点的路径、进度与内容。适配器从 `SCNET_CHECKPOINT_IN` 读取路径，用框架的恢复入口加载优化器及调度器等训练状态，并验证实际恢复的进度。预留写入检查点和计算摘要的时间；训练适配器应在时限前完成本段，作业时限由 Slurm 执行。

关于设备命名和会合参数，可核对 [PyTorch 兼容运行时文档](https://docs.pytorch.org/docs/stable/notes/hip.html)、[单机多进程启动文档](https://docs.pytorch.org/docs/stable/elastic/run.html)和 [PyTorch 1.10 启动器源码](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/distributed/run.py)。
