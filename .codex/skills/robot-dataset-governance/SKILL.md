---
name: robot-dataset-governance
description: Enforce the AUBO-RobotTraj project data standard whenever creating, modifying, reviewing, or deleting robot-dataset logic in this repository. Use for trajectory collection, observation/action/reward schemas, sensor recording, serialization, annotations, dataset splits, statistics, validation, replay, migrations, loaders, model training inputs, and RLDS, LeRobot, Robomimic, or other dataset converters.
---

# 机器人数据集治理

## 必读规范

1. 先定位项目根目录。
2. 在对任何数据集逻辑采取操作前，完整阅读 `docs/机器人轨迹数据集规范.md`。
3. 同时应用 `project-engineering-guardrails`。
4. 若规范文档缺失、无法读取或与现有数据冲突，停止数据写入或 schema 修改，报告冲突和所需决策。

将该文档视为数据语义、存储布局、版本和验收规则的单一事实来源。不在代码或本技能中另行发明冲突规则。

## 修改前

1. 运行 `git status --short`，阅读目标文件的当前内容和 diff，保留用户已有改动。
2. 将改动归类为采集、schema、传感器流、存储、标注、划分、统计、加载、校验、回放、迁移或格式导出。
3. 列出受影响的 Frame、Transition、Episode、时间戳、单位、坐标系和下游消费者。
4. 区分无损主数据和模型特定派生数据。
5. 判断改动是否需要 schema `PATCH`、`MINOR` 或 `MAJOR` 升级。

## 强制不变量

- 保持 `T + 1` 个 Frame 和 `T` 个 Transition，能构造 `(S_t, A_t, R_t, S_{t+1}, delta_t, done_t)`。
- 将部署时可获得的观测与仿真/教师特权状态分开。
- 对动作记录原始策略输出、物理命令、控制器目标和执行结果。
- 对每种动作声明 representation、frame、unit 和 normalization。
- 将关节位置/速度测量归入 observation；只将目标、指令或增量归入 action。
- 对奖励记录 raw value、weight、time scale、contribution 和 total。
- 分开 `terminated`、`truncated`、`success` 和 `invalid`。
- 保留原始多频率时间戳，将对齐、插值和降采样作为可追溯的派生步骤。
- 保存代码、配置、资产、checkpoint、shard 和 manifest 哈希。
- 不原地改写已发布数据，不对字段做静默改义。

## 实施

1. 若字段、时序、单位、坐标系、存储结构或终止语义变化，先更新规范文档的版本和迁移策略。
2. 保持“配置 -> 数据逻辑 -> Isaac 适配/执行 -> 数据写入”的单向依赖。
3. 不让数据写入模块反向控制环境任务逻辑。
4. 使用具名字段或受版本管理的 schema 解析器，不在下游散落匿名向量切片常量。
5. 将图像转码、归一化、窗口化和 action chunk 写入派生数据，记录源 manifest 和转换参数。
6. 格式转换时显式报告无法映射或有损的字段，不静默丢弃。

## 验证

按改动范围执行并报告：

1. Schema 必填字段、dtype、shape、枚举和外键校验。
2. `num_frames == num_transitions + 1` 及 Frame/Transition 索引校验。
3. 时间戳单调性、`delta_t` 和丢帧校验。
4. NaN/Inf、四元数、物理范围和坐标变换校验。
5. Reward contribution 求和与 total 一致性校验。
6. 最小 Episode 采集或 fixture 的往返序列化与回放校验。
7. 受影响的 RLDS、LeRobot、Robomimic 或其他导出器校验。
8. 修改后的 `git diff`、UTF-8 和意外文件校验。

如因 Isaac 运行环境、硬件或缺失旧版数据而无法执行某项验证，明确说明未验证项、风险和可执行的后续命令，不将其声明为已通过。

## 交付

说明：

- 更改的数据逻辑和字段。
- Schema 版本及向后兼容性。
- 主数据与派生数据的边界。
- 迁移或外部格式影响。
- 已执行和未执行的验证。
