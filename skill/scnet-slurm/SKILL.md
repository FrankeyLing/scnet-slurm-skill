---
name: scnet-slurm
description: Plan, render, validate and submit single-node PyTorch jobs to SCNet (超算互联网) Slurm partitions using Hygon accelerators. Use for 昆山 or 哈尔滨 queue choice, two-hour checkpoint and resume, single versus distributed training, and hardware or submission verification.
---

# 超算互联网训练作业

用 `scripts/scnet.py` 完成选择、脚本生成、核查、提交和查询。脚本只支持单节点，卡数和框架启动方式要与实际训练入口一致。执行位置是目标集群的 Linux 登录节点；本地只能做规划和生成。

1. 每个候选先在各自登录节点运行 `inspect --partition <分区>`。核对账号可用性、专项余额适用性、实际时限和卡资源；没有现场依据就先安排有界短测。读 [资源选择与验收](references/policy.md) 决定是否值得迁移或排长队列。历史队列数据只用于定位。
2. 读 [提交手册](references/submit.md)，固定代码、数据、起始状态、超参数、环境和适配器。用 `decide <spec.json>` 显示选择与拒绝理由，再用 `render <spec.json> <bundle>` 生成新目录。成熟框架优先评估原生多卡；自研框架正式多卡必须有同配方的真实数值验收。自研框架续训有独立的对照验收。仅通过硬件或通信检查时可提交短测，不能当作训练验收。
3. 在选中的节点 `preflight <bundle>`，通过后才按本次授权使用 `submit <bundle> --segment 1 --execute`。`--chain` 提交整条成功依赖链，适合首段及续训路径已短测的任务。生成和预检不会启动训练。
4. `status <job_id>` 看排队和父作业状态，同时查 `logs/` 的标准输出及错误、`artifacts/` 的探针和训练凭据。作业成功需要本次运行的有效凭据。用 [运行契约](references/runtime.md) 适配训练入口及保存检查点。对时限杀死、训练异常或提交结果不明，先对账并找最后通过核查的检查点，不自动重交。
