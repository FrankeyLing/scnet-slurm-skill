# 从计划到作业

## 1. 在各中心的登录节点查实际配置

`python3 scripts/scnet.py inspect --partition PARTITION` 返回 `ClusterName`、分区配置、每节点通用资源、队列原因和公平份额。还没有分区名时先运行 `sinfo -h -o '%R|%l|%G|%c|%m|%t|%D'`。从平台控制台进入各中心命令行，或通过自己已经配置的远程登录方式连接；不要假设一个登录节点能提交另一中心。先在平台查开通、费用和专项适用范围；`--test-only` 只检查调度器请求，不能证明会用哪笔余额。用实际短测确认扣费和算子可用性。

昆山的历史候选是 `kshdAI`（两小时）和 `kshdtest`（三天）；哈尔滨历史上曾在 `hebhdnormal` 分到单张海光卡。名字、分区时限、节点数、软件模块和账单权限都要现场确认。软件模块可以用 `module avail` / `module show` 检查，选择该中心维护的海光兼容 PyTorch；使用模块时把具体版本与激活命令放进经核对的 `setup_file`。保留调度器提供的设备可见性，不要把主机卡编号写成全局可见列表。先测项目真正用的算子，再决定精度、混合精度和第三方扩展。

在登录节点用 Python 3.7 或更高版本调用提交工具。裸登录环境的解释器有时低于要求或根本没有 `python3`。可用单独的解释器调用提交工具，让 `setup_file` 在非交互 Bash 中初始化作业环境。有些站点的模块可以在提交与作业中用同一套环境，也要先验证重复加载是否正常。生成的作业会继承提交环境。如果模块系统报 `Cannot initialize TCL`，先从未加载训练模块的提交环境复现和预检，避免把已有的运行时库路径传到再次初始化模块的作业中。

## 2. 填计划与工作负载文件

两个文件均用 UTF-8（八位统一码转换格式，8-bit Unicode Transformation Format，保留中文且不在脚本开头插入标记的编码），禁止重复键。使用 `sha256sum` 或 PowerShell `Get-FileHash -Algorithm SHA256` 计算文件摘要；摘要填小写。先编辑 `workload.json`，最后计算摘要，任一内容变化都重新计算并匹配对应验收。示例生成器见仓库 `examples/make_demo.py`。

`spec.json`：`schema_version: 1`、`framework: "mature"` 或 `"custom"`、`purpose: "smoke"` 或 `"formal"`、`workload_manifest`（相对本文件或绝对路径）、`workload_sha256`、`total_steps`、显式 `supports_resume`、`candidates`、`evidence`（没有就 `[]`）。每个候选必须给出 `id`、`center`、`cluster`、`partition`、`cards`、`wall_minutes`、`queue_minutes`、`setup_minutes`、`startup_minutes`、`seconds_per_step`、`checkpoint_margin_minutes`。证据的 `artifact` 填可读取的本地文件路径，允许相对计划文件；规划时会核对摘要。预算和证据表见 [policy.md](policy.md)。

`workload.json` 包含 `identity`（显式给出 `revision`、`data`、`initial_weights`、`training_recipe`、`runtime` 五个身份字符串）、与计划一致的 `total_steps`，及 `jobs`。每个 `jobs` 键与候选 `id` 一致，其值包含 `launch` 和 `files`。`files` 是远程绝对路径到小写内容摘要的映射，至少锁定入口与环境脚本。需要固定的模型配置或数据索引也纳入校验；大数据使用经检验的内容清单，整文件校验的时间要计入启动预算。工作负载文件应随代码、数据和环境变化更新。

| `launch` 字段 | 内容 |
| --- | --- |
| `route` | 该候选的 `center`、`cluster`、`partition`、`cards` |
| `workdir`, `entrypoint` | 训练工作目录、适配器脚本的远程绝对路径 |
| `python`, `setup_file` | 解释器绝对路径或激活后的可执行名；环境脚本绝对路径，或显式 `null` |
| `args` | 参数逐个写为数组元素，包括独立的 `{end_step}`、`{total_steps}`，可使用 `{start_step}` |
| `resume_args`, `checkpoint_args` | 不使用时填 `[]`；否则包含独立的 `{checkpoint_in}`、`{checkpoint_out}`。允许自选实际参数名 |
| `gres`, `cpus_per_task`, `memory_mib`, `threads_per_rank` | `dcu` 或 `dcu:类型`（不要带数量）、整节点单任务核心数、内存申请、每训练进程线程数 |
| `probe_timeout_seconds`, `env` | 计算节点探针限时、显式环境变量对象（不需要额外变量就填 `{}`） |
| `account`, `qos`, `arch` | 可选。确知后填写账户、调度服务级别或经验证的设备架构 |

工具固定单卡直接启动、单节点多卡一张卡一个进程，由 `torch.distributed.run` 创建。适配器必须识别启动器提供的进程配置、在正确设备训练和分片，调用已经选定框架的原生分布式入口。不要在已启动的每个进程里再次启动一整组进程。不支持该启动器或需要多节点的环境不能直接用这套模板。必要时从实际解释器探测 `-m torch.distributed.run --help`；旧模块不保证提供同一接口。

## 3. 生成、预检和提交

```bash
python3 scripts/scnet.py decide spec.json
python3 scripts/scnet.py render spec.json work/bundle
python3 work/bundle/scnet.py preflight work/bundle
python3 work/bundle/scnet.py submit work/bundle --segment 1 --execute
python3 work/bundle/scnet.py status JOB_ID
```

前两步可本地执行。上传的是整个新 `bundle` 目录，至少保留 `logs/`、`artifacts/`、`submissions/` 三个空目录，并在计算节点可见的共享文件系统上使用。代码、数据和训练环境独立部署。后面三步在对应中心的登录节点执行。预检校验集群名、分区 `MaxTime`、每节点形状、训练文件摘要和脚本的 Bash 语法，使用 `sbatch --test-only` 核对当前申请。预检不会提交作业，不能保证后续启动时间和节点状态。如果站点拒绝 `--test-only`，停止并向站点核实，不把实际提交当预检用。

`inspect`、`preflight`、`submit`、`status` 可显式传入 `--scheduler-timeout-seconds 120`。
这是每一条调度命令的等待上限，须为 1 至 600 秒的整数；未传时保留原来的 30 秒。
集群、分区、资源、队列、份额、父作业核对和真正的提交都使用该值；本地 Bash
语法检查仍限 30 秒。预检与提交登记会记录实际选择的等待上限。它不会重试命令。

```bash
python3 work/bundle/scnet.py preflight work/bundle --scheduler-timeout-seconds 120
python3 work/bundle/scnet.py submit work/bundle --segment 1 --execute --scheduler-timeout-seconds 120
python3 work/bundle/scnet.py status JOB_ID --scheduler-timeout-seconds 120
```

自动化包装器应另设整个阶段的总等待预算。一次 `inspect` 顺序执行五条调度命令，
一次 `preflight` 执行三条查询，再为每段执行一次 `sbatch --test-only`；文件摘要核验
还需要额外时间。单条设为 120 秒时，不能仍用 120 秒作为整个阶段的总限时。
延长等待不能保证站点及时响应；超时后先对账。真正的 `sbatch` 超时仍保留提交意图，
禁止据此自动重交。升级后用新版本重新生成新的 bundle，保留旧版本及失败现场，
不要改写已有 bundle 中受摘要保护的工具脚本。

通过后使用回执中的作业号。在一个新目录中，`submit ... --chain --execute` 会按成功依赖预先提交所有分段。普通 `--segment 2` 只有等前段账本完成、凭据与检查点一致后才放行。链路任何一段非零退出时，成功依赖会拦住后继；仍需查看 `status` 核对现场状态。运行器在每一段真正开始训练前重新检查检查点。监控耗时、退出码、真实进度、卡数和显存，主机内存超限与显存超限分别排查。不要根据台账里 `billing` 的数值推算金额。

如果到达时限、运行异常或成功凭据缺失，不把该段标记为完成。找最后一份已验证的完整检查点。要从计划外的检查点续跑，创建新计划，指定 `resume_existing: true`、实际 `start_step` 和 `initial_checkpoint: {"path": "/absolute/checkpoint.pt", "sha256": "..."}`；保持原 `total_steps`、同一个工作负载及适用验收。适配器必须检查恢复后的真实进度。提交调用结果不明时，`submissions/` 的 `intent` 是防重记录；先在平台按时段和作业名辨明是否已提交，再决定下一步。不要手动清空意图记录并盲重试。

短测通过要求 `sacct` 的父作业最终为 `COMPLETED 0:0`，且探针、训练凭据均匹配该作业号和预期进度。实测中有一次训练已写出凭据，调度器仍因主机内存超限将作业判为 `OUT_OF_MEMORY`。以两项检查共同判定。初始化软件栈和运行时探针也要预留主机内存。若 `sbatch --test-only` 拒绝内存与核心的比例，按目标队列现场规则调整申请，再运行预检。

调度器参数语义参见 [Slurm sbatch](https://slurm.schedmd.com/sbatch.html)；站点拓扑查询参见 [超算互联网计算服务文档](https://www.scnet.cn/doc/1.0.6/30000/general-handbook/sys-info-new.html)。
