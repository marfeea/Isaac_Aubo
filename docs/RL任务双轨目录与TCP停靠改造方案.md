# RL 任务双轨目录与夹爪 TCP 停靠改造方案

> 状态：阶段四已完成待验收
> 范围：旧版无夹爪 Flange reach 与新版带夹爪 TCP parking 的目录隔离、接口边界、坐标约定和验证计划

## 1. 本轮结论

项目后续保留两条可独立训练、验证的任务链：

- `WithoutClaw`：保留旧目标，即无夹爪 AUBO 的 `Flange` 靠近 target。
- `WithClaw`：新增带夹爪 AUBO 的 TCP 停靠任务，即 TCP 在所选 `preposition` 周围低速持续停留，同时法兰姿态仍朝向 target 本体。

两条任务只复用与具体任务目标无关的数学、Isaac API 适配和通用训练运行设施；观测、奖励、终止、reset 缓存和任务资产配置不得交叉导入。

当前工作区的 `configs/RLcfg.py`、`configs/asset.py`、`configs/collision_cfg.py`、`configs/place_cfg.py`、`configs/scene_cfg.py`、`scripts/test.py`、`scripts/train.py`、`tools/logic.py`、`tools/scene.py` 已有用户修改。施工时把它们视为用户基线，不回退、不覆盖，也不通过整文件格式化制造无关差异。

## 2. 建议目录

采用 `tasks/` 作为任务边界，新建两个对称任务包。`scripts/test.py`、`scripts/train.py` 和 `scripts/eval.py` 暂不搬移；新入口直接放进各自任务包，避免修改现有入口造成冲突。

```text
Test/
|-- tasks/
|   |-- __init__.py
|   |-- common/
|   |   |-- __init__.py
|   |   |-- buffers.py          # reset-safe 张量缓存、连续步计数
|   |   |-- frames.py           # 点、向量、速度的坐标变换
|   |   |-- collision.py        # 通用接触张量解析，不包含任务判定
|   |   `-- runtime.py          # 训练/验证入口共用的参数与启动封装
|   |-- WithoutClaw/
|   |   |-- __init__.py
|   |   |-- asset_cfg.py        # AUBO_E5.usd 与无夹爪专用 prim/body 配置
|   |   |-- scene_cfg.py
|   |   |-- env_cfg.py          # manager-based 环境组装
|   |   |-- observations.py
|   |   |-- rewards.py          # 旧 Flange reach 奖励
|   |   |-- terminations.py     # 旧 reach 成功/失败/timeout
|   |   |-- events.py           # 旧 reset 行为
|   |   |-- train.py            # 可直接运行的薄训练入口
|   |   |-- eval.py             # 可直接运行的薄验证入口
|   |   `-- tests/
|   |       `-- test_logic.py
|   `-- WithClaw/
|       |-- __init__.py
|       |-- asset_cfg.py        # AUBO_E5_Withclaw.usd、Flange、TCP 偏置
|       |-- task_cfg.py         # 五状态、阈值、工作空间、奖励权重的单一事实源
|       |-- scene_cfg.py
|       |-- env_cfg.py          # manager-based 环境组装
|       |-- tcp.py              # TCP 位姿/速度及根坐标系表达
|       |-- observations.py
|       |-- rewards.py          # tcp_progress/proximity/parking/dwell
|       |-- terminations.py     # 成功、碰撞、越界和 target 扰动失败
|       |-- events.py           # 离散状态 reset 与缓存
|       |-- train.py            # 可直接运行的薄训练入口
|       |-- eval.py             # 可直接运行的薄验证入口
|       `-- tests/
|           |-- test_tcp_math.py
|           |-- test_task_logic.py
|           `-- test_five_states.py
|-- configs/                    # 现有场景背景配置；不再承载两种任务的奖励/终止定义
|-- tools/                      # 现有通用 API 封装，按需兼容，不放新任务判定
`-- scripts/                    # 保留现有用户入口和调试脚本
```

### 2.1 依赖方向

```text
configs/place_cfg.py、现有场景背景
                 |
                 v
          tasks/common
           /          \
          v            v
WithoutClaw 任务包    WithClaw 任务包
          |            |
          v            v
      各自 train.py / eval.py
```

约束如下：

1. `WithoutClaw` 与 `WithClaw` 不互相导入。
2. `common` 不导入任何具体任务包。
3. `task_cfg.py` 或各任务的配置文件是阈值、权重、资产和状态表的唯一事实源，运行逻辑中不重复硬编码。
4. 增量式法兰 IK 继续使用当前三维动作和 `AuboTaskSpaceIKAction` 行为契约；新版仅将任务度量点从 Flange 原点改成 TCP，不把 TCP 位置误当成 IK body。
5. 现有脚本保留。等新旧两条链均通过验证后，再单独决定是否把根目录旧入口改为兼容转发器；本次施工不默认执行该迁移。

## 3. 旧任务整理原则

`WithoutClaw` 不是从当前文件盲目复制一份，而是冻结“无夹爪 Flange reach”的有效行为：

- 机器人资产固定为 `AUBO_E5.usd`，不能继续引用当前已切换的 `AUBO_E5_Withclaw.usd`。
- 任务点为 `Flange`，观测、距离奖励和成功判定都基于法兰原点。
- 旧奖励、工作空间、碰撞和 timeout 语义按当前有效实现迁入后做回归测试。
- 训练和验证入口显式导入 `tasks.WithoutClaw.env_cfg`，模型输出目录带 `WithoutClaw` 标识，避免 checkpoint 混用。
- 若旧逻辑依赖当前工作区中尚未提交的修复，则逐项迁入新包并保留原文件，不用 Git 历史版本覆盖用户改动。

## 4. 新任务坐标约定

统一采用 Isaac Lab 的 `wxyz` 四元数和右手坐标系。

- `W`：世界坐标系。
- `B`：机器人 articulation root 坐标系。
- `F`：`Flange` body 局部坐标系。
- `T`：夹爪 TCP 点。第一阶段只标定 TCP 原点，不把未标定的 TCP 姿态用于控制或奖励。
- `P`：当前离散状态对应的 `preposition` 点。

### 4.1 TCP 平移偏置的必要确认

运行时需要的是“TCP 相对 Flange 原点、用 Flange 局部轴表达”的向量：

```text
r_FT_F = position(T) - position(F), expressed in F
```

若现有记录 `(0, 0.12, -0.102)` 的准确含义是：

```text
position(F) - position(T), expressed in F
```

则反向偏置确定为：

```text
r_FT_F = (0, -0.12, 0.102) m
```

如果该记录是用 TCP 局部轴表达，则由于 `R_FT` 尚未标定，不能只逐项取负；必须先得到 Flange 与 TCP 的旋转关系。施工前必须通过 USD prim 层级/测量脚本确认记录向量的表达坐标系。未确认前，新任务配置中的偏置只能标记为待标定，不能作为最终训练参数。

### 4.2 TCP 世界位置

确认 `r_FT_F` 后：

```text
p_T_W = p_F_W + R_WF * r_FT_F
```

因此 TCP 会随法兰旋转正确运动，不允许直接在世界坐标上加固定三元组。

### 4.3 TCP 世界速度

TCP 线速度必须包含法兰角速度产生的切向项：

```text
r_FT_W = R_WF * r_FT_F
v_T_W  = v_F_W + omega_F_W x r_FT_W
```

`v_F_W` 和 `omega_F_W` 分别读取 `Flange` body 的世界线速度和世界角速度。纯逻辑测试必须覆盖“法兰原点线速度为零但角速度非零时，TCP 速度非零”的情况。

### 4.4 根坐标系观测

位置和相对向量：

```text
p_T_B = R_BW * (p_T_W - p_B_W)
p_P_B = R_BW * (p_P_W - p_B_W)
d_TP_B = p_P_B - p_T_B
```

速度按完整相对速度定义，避免未来机器人根节点可动时语义变化：

```text
v_T_rel_W = v_T_W - v_B_W - omega_B_W x (p_T_W - p_B_W)
v_T_B     = R_BW * v_T_rel_W
```

当前机器人底座固定时，根速度项理论上为零，但实现仍保留完整公式。

新版低维策略观测顺序固定为：关节相对位置、关节速度、`tcp_pos_b`、`preposition_b`、`tcp_to_preposition_b`、`tcp_vel_b`。所有项均保持批量形状 `(num_envs, N)`，不把世界绝对坐标直接输入策略。

## 5. Reset 数据流

`TARGET_INITIAL_STATES` 的五个具名状态迁入 `tasks/WithClaw/task_cfg.py`，每项继续包含 `name`、`pos`、`rot`、`preposition`。target 场景键固定为 `ws_interactive_reagent_01_sample_bottle`。

一次 reset 对指定 `env_ids` 原子完成以下操作：

1. 选择或按测试参数固定一个 state。
2. 写入 target 位姿，并将 target 刚体线速度、角速度清零。
3. 缓存 `selected_state_ids: LongTensor[num_envs]`。
4. 缓存 `selected_state_names`；运行数值逻辑只依赖 id，名称用于日志与评估输出。
5. 缓存 `preposition_w: Tensor[num_envs, 3]`，由 state 的 env 局部坐标加 `env_origins` 得到。
6. 从写入后的 target 实际根位置缓存 `target_initial_pos_w: Tensor[num_envs, 3]`，避免只缓存命令值而漏掉初始化误差。
7. 清零前一步 TCP 距离、停车区域状态和 dwell 计数器，防止跨 episode 污染。

五个缓存必须支持部分环境 reset，未 reset 的环境数据不得改变。

## 6. 新版奖励

奖励项独立放在 `tasks/WithClaw/rewards.py`，权重集中在 `task_cfg.py`。第一版先保持公式透明，权重在 smoke test 后只通过配置调整。

| 奖励项 | 建议定义 | 目的 |
|---|---|---|
| `tcp_progress` | `previous_distance - current_distance`，允许负值，reset 首步为 0 | 奖励向 preposition 的真实进展并惩罚后退 |
| `tcp_proximity` | `exp(-(distance / sigma_d)^2)` | 提供连续的近距离吸引信号 |
| `tcp_parking` | 停车区域有效时乘 `exp(-(speed / sigma_v)^2)` | 同时鼓励接近和减速，避免高速穿越刷奖励 |
| `tcp_dwell` | `min(consecutive_low_speed_steps / 3, 1)` | 奖励连续停留，不把单步命中当成功 |

继续保留：

- `action_l2` 动作幅值惩罚。
- `action_rate_l2` 动作变化惩罚。
- `step_penalty` 时间惩罚。
- TCP 根坐标系工作空间越界惩罚。
- 非法碰撞惩罚。

不再使用旧任务的 Flange-to-target 距离奖励或旧 `success` 奖励。法兰朝向 target 仅由现有 IK 姿态逻辑负责，不改变三维增量动作接口。

## 7. 成功、迟滞与失败

### 7.1 成功状态机

按以下迟滞解释实现用户给出的 `0.03 / 0.045 m`：

```text
未进入停车区 -- distance < 0.03 m --> 已进入停车区
已进入停车区 -- distance > 0.045 m --> 退出并清零 dwell
已进入停车区 -- TCP speed < 0.02 m/s --> dwell + 1
已进入停车区 -- TCP speed >= 0.02 m/s --> dwell 清零，但距离未越过退出半径时保留区域状态
dwell >= 3 个控制步 --> success，terminated=True
```

这是一种 Schmitt trigger 迟滞：进入后允许距离在 `0.03–0.045 m` 间小幅波动，避免边界噪声反复切换。若期望“连续三步每一步都必须严格小于 `0.03 m`”，则退出半径不会参与 dwell，需在确认时明确改成严格模式。

### 7.2 失败终止

以下任一条件触发 `terminated=True`：

- 非法碰撞。
- TCP 超出机器人根坐标系下的任务工作空间。
- `norm(target_pos_w - target_initial_pos_w) > 0.03 m`。
- target 世界线速度模长 `> 0.05 m/s`。

新版任务以无接触停靠为目标，因此建议任何机器人 body 与 target 的接触都视为非法碰撞；不沿用旧任务“Flange 与 target 轻触可接受”的特例。若实际任务允许指爪轻触瓶体，需要在确认后明确允许 body 名称和力阈值。

### 7.3 Timeout

`time_out` 的 `DoneTerm` 必须设置 `time_out=True`。因此时间耗尽只进入 Isaac Lab 的 truncated 路径，不得混入任务失败的 terminated 路径。纯逻辑测试和短程环境测试都检查该标志。

## 8. IK 接口保持不变

新版动作仍是三维归一化增量：

```text
action = (dx, dy, dz)
```

- ActionTerm 仍对 `Flange` 下发增量式位置目标。
- `joint_names`、动作维数、位置缩放、单步位移上限和 Lula 控制器调用接口不变。
- 姿态目标仍由 Flange 到 target 本体中心的方向生成，不改成朝向 `preposition`。
- TCP 只用于观测、奖励、成功/失败和工作空间判断。
- 若后续实测发现 TCP 轴向与夹爪视觉朝向不一致，再单独标定 TCP 旋转；本阶段不把未知旋转偷偷写入控制逻辑。

## 9. 四阶段施工与状态门禁

### 9.1 状态总表

状态只允许使用：`待用户确认`、`未开始`、`进行中`、`已完成待验收`、`已验收`、`阻塞`。

| 阶段 | 任务 | 当前状态 | 启动条件 |
|---|---|---|---|
| 阶段一 | 双任务骨架与 `WithoutClaw` 基线 | 已验收 | 用户确认四阶段拆分及目录方案 |
| 阶段二 | TCP 坐标标定、几何计算与 reset 缓存 | 已验收 | 阶段一已验收 |
| 阶段三 | `WithClaw` 观测、奖励、终止与环境组装 | 已验收 | 阶段二已验收且 TCP 偏置结论明确 |
| 阶段四 | 物理稳定性、短程训练和文档总验收 | 已完成待验收 | 阶段三已验收 |

### 9.2 每阶段执行协议

每次施工只执行一个阶段，并严格遵守以下门禁：

1. 开始前重新读取本文件第 9 节状态总表、当前阶段范围、上一阶段验收记录和未决事项。
2. 运行 `git status --short` 并检查拟修改文件的 diff，确认不会覆盖用户工作区修改。
3. 若上一阶段不是 `已验收`，或当前阶段仍存在会改变实现方向的未决事项，则不得开始写代码。
4. 向用户简要报告读取到的当前状态、该阶段输入和预计修改范围，然后把当前阶段标记为 `进行中`。
5. 只完成本阶段列出的内容，不提前施工下一阶段。
6. 完成本阶段验证后，将状态改成 `已完成待验收`，在本文件记录实际文件、命令、结果、偏差和遗留问题，并停止施工。
7. 用户明确验收后，才把状态改成 `已验收`；下一次执行再进入后续阶段。
8. 若验证失败但仍可在本阶段范围内修复，保持 `进行中`；若依赖用户决策或外部条件，标记为 `阻塞` 并记录原因。

### 9.3 阶段一：双任务骨架与 WithoutClaw 基线

目标：先完成目录隔离，并证明旧的无夹爪 Flange reach 任务仍能独立训练和验证。

工作范围：

1. 建立 `tasks/common`、`tasks/WithoutClaw`、`tasks/WithClaw` 包骨架；`WithClaw` 本阶段只建空包和接口占位，不实现任务逻辑。
2. 将无夹爪资产、场景、观测、动作、奖励、终止和 reset 行为整理进 `WithoutClaw`。
3. 无夹爪资产固定为 `AUBO_E5.usd`，任务度量点固定为 `Flange`。
4. 新建 `WithoutClaw/train.py` 与 `WithoutClaw/eval.py` 薄入口，使用独立 checkpoint/log 路径。
5. 提取本阶段确实需要的通用运行和数学设施到 `tasks/common`，不为未来需求预先抽象。
6. 不修改现有 `scripts/test.py`、`scripts/train.py` 和 `scripts/eval.py`。

阶段验收：

- 新增/修改 Python 文件通过语法检查。
- 旧任务纯逻辑回归测试通过。
- `WithoutClaw` 环境可创建、reset、step。
- 训练与验证入口能启动，短 rollout 不出现非有限观测或奖励。
- 现有用户修改未被覆盖，`WithClaw` 逻辑尚未提前实现。

### 9.4 阶段二：TCP 坐标标定、几何计算与 reset 缓存

目标：冻结新版任务的坐标语义，并建立可独立验证的 TCP 与五状态基础数据层。

工作范围：

1. 检查 `AUBO_E5_Withclaw.usd` 的 Flange、夹爪和 TCP 相关 prim 层级，确认现有偏置记录使用的表达坐标系。
2. 仅当证据确认 `(0, 0.12, -0.102)` 是用 Flange 局部轴表达的 TCP→Flange 向量时，才将 `r_FT_F` 配置为 `(0, -0.12, 0.102)`；否则保持阻塞并给出所需标定数据。
3. 实现 TCP 世界位置、含角速度切向项的世界速度，以及机器人根坐标系位置/速度变换。
4. 迁入五个 `TARGET_INITIAL_STATES`，实现随机状态、固定状态和部分环境 reset。
5. 缓存 `selected_state_ids`、`selected_state_names`、`preposition_w` 和 `target_initial_pos_w`，并清理跨 episode 临时状态。
6. 实现纯逻辑测试和必要的单环境 Isaac 读取检查；本阶段不接奖励和终止配置。

阶段验收：

- 偏置方向和表达坐标系有明确证据，不依赖猜测。
- TCP 位置、纯旋转切向速度、根坐标变换的批量测试通过。
- 五状态 id/name/preposition 映射和部分 reset 测试通过。
- 实际 Flange 旋转时 TCP 轨迹方向与预期一致。
- 新增/修改 Python 文件通过语法检查。

### 9.5 阶段三：WithClaw 完整 MDP 组装

目标：形成可运行的带夹爪 TCP 低速持续停靠任务，但暂不做最终长链路验收。

工作范围：

1. 接入根坐标系下的 TCP、preposition、相对向量和 TCP 速度观测。
2. 实现 `tcp_progress`、`tcp_proximity`、`tcp_parking`、`tcp_dwell`，并保留动作、时间、工作空间和碰撞惩罚。
3. 实现 `0.03 m` 进入、`0.045 m` 退出、`0.02 m/s` 速度阈值和连续 3 个控制步的成功状态机。
4. 实现非法碰撞、TCP 越界、target 位移 `> 0.03 m`、target 速度 `> 0.05 m/s` 的失败终止。
5. 明确 `time_out=True`，保证 timeout 只进入 truncated。
6. 接入 `AUBO_E5_Withclaw.usd` 和实际 articulation/夹爪碰撞传感器层级。
7. 保持三维增量式 Flange IK 接口不变，姿态继续朝向 target 本体。
8. 新建 `WithClaw/train.py`、`WithClaw/eval.py` 和独立输出路径。

阶段验收：

- 全部奖励项、迟滞 dwell、四类失败和 timeout 语义的纯逻辑测试通过。
- `WithClaw` 环境可创建、reset，并以零动作和随机动作完成短步进。
- 观测形状固定且全部 finite，reward breakdown 只包含新版任务项。
- IK 的动作维数、调用接口和朝向目标没有发生行为性变更。
- 新增/修改 Python 文件通过语法检查。

### 9.6 阶段四：物理、训练与文档总验收

目标：完成用户要求的仿真和训练验证，确认两条任务链均可交付。

工作范围：

1. 运行五状态物理稳定性测试，记录 target 初始误差、最大单步位移、累计漂移、最大速度、TCP 初始距离和接触情况。
2. 运行 `WithoutClaw` 与 `WithClaw` 的短程训练 smoke test，至少完成一个 PPO rollout 和一次参数更新。
3. 分别运行两条任务的验证入口，检查 checkpoint 隔离、reward breakdown、terminated/truncated 统计和日志路径。
4. 对全部新增/修改 Python 文件执行最终语法检查，并运行全量纯逻辑测试。
5. 审查最终 diff、编码、临时诊断输出和任务间反向依赖。
6. 更新本文件的实际落地文件、验证命令和结果，并按最终目录同步 README 项目结构。

阶段验收：

- 语法检查、纯逻辑测试、五状态物理稳定性测试和两条短程训练 smoke test 全部有可复核结果。
- timeout 被记录为 truncated，任务成功和四类失败被记录为 terminated。
- 两条任务可以独立训练、验证，且配置、奖励和 checkpoint 不混用。
- 工作区原有无关改动完整保留。
- 文档与最终代码目录一致。

### 9.7 阶段执行记录

各阶段完成时在此追加，不改写历史记录：

| 阶段 | 日期 | 状态变化 | 实际修改 | 验证结果 | 遗留/偏差 |
|---|---|---|---|---|---|
| 阶段一 | 2026-07-05 | 待用户确认 → 进行中 → 已完成待验收 → 已验收 | 新增 `tasks/common`、完整 `tasks/WithoutClaw` 和 `tasks/WithClaw` 占位包；现有入口未修改 | 语法、4 项纯逻辑测试、8 timestep PPO 更新、4 step checkpoint 回放均通过 | 用户已验收；回放受 `max_steps=4` 限制，未收集完整 episode |
| 阶段二 | 2026-07-05 | 未开始 → 进行中 → 阻塞 → 进行中 → 已完成待验收 → 已验收 | 已实现固定 TCP 偏置、TCP 几何/速度、articulation 读取适配、五状态 reset 与缓存及标定检查 | 语法、9 项纯逻辑测试、USD 几何检查、2 环境部分 reset、实际 Flange 旋转轨迹验证通过 | 用户指示继续阶段三，视为阶段二验收；直接反向局部偏置的已知标定风险继续保留 |
| 阶段三 | 2026-07-05 | 未开始 → 进行中 → 已完成待验收 → 已验收 | 已实现 WithClaw 独立资产/场景、根坐标观测、四项 TCP 奖励、迟滞 dwell、四类失败、timeout、三维 Flange IK 和独立入口 | 语法、13 项纯逻辑/坐标测试、2 环境 reset 及零动作/随机动作短步进通过 | 用户指示继续阶段四，视为阶段三验收 |
| 阶段四 | 2026-07-05 | 未开始 → 进行中 → 已完成待验收 | 新增五状态并行物理验收；训练支持安全 run label；回放输出 terminated/truncated 和活动项；同步中英文 README | 五状态、18 项纯逻辑、reset/TCP/env 集成、双任务各 1 次 PPO 更新和独立 checkpoint 回放全部通过 | TCP 偏置仍采用用户确认的第一版直接反向值；Isaac Sim 既有用户配置与 shader cache 警告不影响结果 |

#### 阶段一实际落地与验证

实际落地：

- `tasks/common/paths.py`：按任务名隔离日志与 checkpoint 路径。
- `tasks/common/sb3_runtime.py`：任务无关的精简 SB3 训练与回放运行时，并在 reset/step 检查非有限观测和奖励。
- `tasks/WithoutClaw/`：无夹爪资产、contact、场景、事件、观测/动作、奖励、终止、环境组装和独立 train/eval 入口。
- `tasks/WithoutClaw/task_cfg.py`：`AUBO_E5.usd`、`Flange`、固定 target 场景键、仿真周期和工作空间的单一事实源。
- `tasks/WithClaw/__init__.py`：只有阶段边界说明，没有提前实现 TCP 逻辑。

验证记录：

```text
python -m py_compile <tasks 下全部 Python 文件>
结果：通过

python -m unittest tasks.WithoutClaw.tests.test_logic -v
结果：4 tests passed

D:\Anaconda\envs\isaaclab\python.exe tasks\WithoutClaw\train.py \
  --headless --device cpu --num_envs 1 --total_timesteps 8 --n_steps 4 --batch_size 4
结果：环境创建成功；观测 shape=(24,)，动作 shape=(3,)；完成 8 timestep 和 PPO 参数更新；
      输出 checkpoints/WithoutClaw/sb3_aubo/ppo_WithoutClaw_final.zip

D:\Anaconda\envs\isaaclab\python.exe tasks\WithoutClaw\eval.py \
  --headless --device cpu --weight ppo_WithoutClaw_final \
  --num_envs 1 --episodes 1 --max_steps 4 --deterministic
结果：checkpoint 加载成功并完成 4 个有限 step；因 max_steps=4，预期未完成完整 episode。
```

环境说明：`C:\isaac-sim\kit\python.bat` 和当前 base 环境均未安装 `isaaclab`；本机有效解释器为 `D:\Anaconda\envs\isaaclab\python.exe`。Isaac Sim 报告用户配置文件不可写、shader cache/Kit 锁等警告，但本阶段环境创建、训练和回放均以退出码 0 完成。

#### 阶段二偏置决策、落地与验证

已完成：

- `tasks/WithClaw/task_cfg.py`：迁入五个具名状态；记录原始向量，并将正式 `FLANGE_TO_TCP_TRANSLATION_F` 冻结为直接反向值 `(0,-0.12,0.102)`。
- `tasks/WithClaw/tcp.py`：实现 wxyz 批量旋转、TCP 世界位置、`v_F + omega_F × r_FT_W` 世界速度、完整根相对速度和 articulation data 读取适配。
- `tasks/WithClaw/reset_state.py`：实现状态校验、随机/固定 state id、world/env 原点换算和名称部分更新。
- `tasks/WithClaw/events.py`：实现 target 位姿写入、速度清零、`selected_state_ids`、`selected_state_names`、`preposition_w`、`target_initial_pos_w` 缓存及 episode 临时缓存清理。
- `tasks/WithClaw/inspect_asset.py`：不修改 USD 的 Flange/夹爪几何检查。
- `tasks/WithClaw/inspect_runtime_pose.py`：读取初始 Flange 世界姿态并反算记录向量在 Flange 坐标系的表达。

验证结果：

```text
D:\Anaconda\envs\isaaclab\python.exe -m unittest \
  tasks.WithClaw.tests.test_tcp_math tasks.WithClaw.tests.test_reset_state -v
结果：9 tests passed

D:\Anaconda\envs\isaaclab\python.exe tasks\WithClaw\inspect_asset.py
结果：Flange body=/Root/AUBO_E5/Flange
      claw origin in Flange=(-0.068636, 0, 0)
      claw mesh bounds in Flange:
        min=(-0.211236, -0.035912, -0.064)
        max=(-0.068636,  0.035912,  0.1205)

D:\Anaconda\envs\isaaclab\python.exe \
  tasks\WithClaw\tests\test_reset_integration.py --headless --device cpu
结果：2 环境全 reset 后仅重置 env 1，最终 ids=[0,4]、名称与缓存均正确；
      未 reset 的 env 0 缓存保持不变，target 初始位置和速度检查通过。

D:\Anaconda\envs\isaaclab\python.exe \
  tasks\WithClaw\tests\test_tcp_runtime.py --headless --device cpu
结果：实际 Flange 关节旋转 90° 后，TCP 世界位置位移 0.192135 m；
      旋转前后从世界位置反算的 Flange 局部偏置均为 (0,-0.12,0.102)。
```

偏置阻塞证据：

1. 在同一坐标系内，向量 `TCP→Flange=(0,0.12,-0.102)` 的数学反向确实是 `Flange→TCP=(0,-0.12,0.102)`。
2. 但 USD 中实体夹爪相对 Flange 沿局部 `-X` 延伸，局部 `Y` 范围只有约 `±0.036 m`；因此 `(0,-0.12,0.102)` 不可能直接表示当前 USD Flange 坐标系内的实体 TCP 点。
3. 若把直接反向向量解释为当前初始姿态下的世界向量，使用运行时 Flange 四元数反算得到局部约 `(0.120000,-0.000001,0.102000)`，其 `+X` 方向仍与位于 `-X` 的夹爪实体相反。
4. USD 内没有具名 TCP prim，无法从资产自动获得最终 TCP 原点；仅凭当前记录不能判定其表达坐标系。

用户随后明确指示继续阶段二，因此本阶段按用户决策采用直接反向值作为第一版 Flange 局部偏置。USD 几何冲突作为已知标定风险保留，不再阻塞实现；后续若获得实测 TCP 点，只修改 `tasks/WithClaw/task_cfg.py::FLANGE_TO_TCP_TRANSLATION_F`。该段记录阶段二完成时的状态，阶段三启动后阶段二已转为 `已验收`。

#### 阶段三落地与验证

已完成：

- `tasks/WithClaw/task_cfg.py`：集中管理工作空间、停车阈值、失败阈值和奖励权重。
- `tasks/WithClaw/asset_cfg.py`、`collision_cfg.py`、`scene_cfg.py`：固定带夹爪 USD、8 个关节驱动和 `/AUBObot/AUBO_E5/.*` 接触传感器层级。
- `tasks/WithClaw/observations.py`、`runtime.py`：按固定顺序输出关节状态、`tcp_pos_b`、`preposition_b`、相对向量和 `tcp_vel_b`；停车状态在同一控制步内幂等更新。
- `tasks/WithClaw/mdp_logic.py`、`rewards.py`、`terminations.py`：实现四项 TCP 奖励、迟滞 dwell、成功、非法碰撞、TCP 越界、target 位移/速度失败和 timeout 截断。
- `tasks/WithClaw/env_cfg.py`、`train.py`、`eval.py`：完成独立环境组装和按 `WithClaw` 隔离的训练/验证入口。
- `tasks/WithClaw/tests/test_mdp_logic.py`、`test_env_smoke.py`：覆盖公式、状态机、四类失败、配置语义和短步进。

验证结果：

```text
D:\Anaconda\envs\isaaclab\python.exe -m unittest \
  tasks.WithClaw.tests.test_mdp_logic \
  tasks.WithClaw.tests.test_tcp_math \
  tasks.WithClaw.tests.test_reset_state -v
结果：13 tests passed

D:\Anaconda\envs\isaaclab\python.exe tasks\WithClaw\tests\test_env_smoke.py \
  --headless --num_envs 2 --steps_per_mode 2
结果：环境创建和 reset 通过；零动作、随机动作各 2 步通过；
      policy observation shape=(2,28)，全部 finite；action_dim=3；
      reward breakdown 仅含 9 个 WithClaw 奖励项；time_out=True。
```

运行环境仍报告既有的用户配置文件不可写、shader cache/Kit 锁和集成显卡跳过警告；这些警告未阻止本次双环境测试通过。PPO 参数更新、五状态物理稳定性和完整 terminated/truncated 统计属于阶段四，未提前执行。

#### 阶段四最终验收

新增与调整：

- `tasks/WithClaw/tests/test_five_states.py`：在五个并行完整场景中一次性验证五个状态，记录初始化误差、最大单步位移、最大/累计漂移、最大速度、TCP 初始距离和机器人接触力。
- `tasks/common/paths.py`、`sb3_runtime.py`：增加安全的 `run_label` checkpoint 命名，并在回放结果中分别统计 `terminated`、`truncated`、活动奖励项和终止项。
- 两条任务的 `train.py`、`eval.py`：接入 `--run_label` 和用于 timeout 验收的 `--episode_length_s`。
- `README.md`、`README_CN.md`：同步双任务目录、职责、运行入口和隔离输出路径。

五状态物理结果（每个状态 120 个物理步）：

| 状态 | 初始误差 m | 最大单步 m | 最大漂移 m | 累计漂移 m | 最大速度 m/s | TCP 初始距离 m | 最大机器人接触 N |
|---|---:|---:|---:|---:|---:|---:|---:|
| state_01 | 0.000000 | 0.000394 | 0.004006 | 0.004291 | 0.021245 | 0.805379 | 0.000000 |
| state_02 | 0.000000 | 0.000756 | 0.001897 | 0.002586 | 0.003059 | 0.631474 | 0.000000 |
| state_03 | 0.000000 | 0.000697 | 0.000808 | 0.001510 | 0.003209 | 0.600425 | 0.000000 |
| state_04 | 0.000000 | 0.000910 | 0.001063 | 0.001771 | 0.017028 | 0.592605 | 0.000000 |
| state_05 | 0.000000 | 0.001497 | 0.001940 | 0.002395 | 0.014744 | 0.604302 | 0.000000 |

首次使用单环境串行切换五状态时，state_03 受前一状态的 PhysX 历史接触状态影响出现 `0.245613 m/s` 假峰值。改为五环境并行、每个环境只设置一次状态后，state_03 最大速度为 `0.003209 m/s`，其余指标也稳定；最终验收采用无跨状态历史污染的并行结果。

训练与回放：

```text
D:\Anaconda\envs\isaaclab\python.exe tasks\WithoutClaw\train.py \
  --headless --num_envs 2 --n_steps 2 --batch_size 4 --total_timesteps 4 \
  --run_label stage4_smoke
结果：iterations=1，total_timesteps=4；
      checkpoints/WithoutClaw/sb3_aubo/ppo_WithoutClaw_stage4_smoke.zip

D:\Anaconda\envs\isaaclab\python.exe tasks\WithClaw\train.py \
  --headless --num_envs 2 --n_steps 2 --batch_size 4 --total_timesteps 4 \
  --run_label stage4_smoke --fixed_state_name sample_bottle_state_01
结果：iterations=1，total_timesteps=4；
      checkpoints/WithClaw/sb3_aubo/ppo_WithClaw_stage4_smoke.zip

两条 eval 均使用 --episode_length_s 0.5 --episodes 2：
WithoutClaw：episodes=2, steps=2, terminated=0, truncated=2
WithClaw：   episodes=2, steps=2, terminated=0, truncated=2
结果：timeout 只进入 truncated；两条任务的 reward/termination 项和 checkpoint 路径互不混用。
```

最终回归：18 项纯逻辑/坐标测试通过；部分 reset、实际 TCP 旋转、双环境零动作/随机动作短步进通过；全部 `tasks/` Python 文件通过 `py_compile`。成功和四类失败的纯逻辑掩码均通过测试，且对应 `DoneTerm.time_out=False`；只有 timeout 项为 `time_out=True`。

## 10. 验证矩阵

### 10.1 静态与纯逻辑验证

1. 对全部新增/修改 Python 文件执行 `py_compile`。
2. TCP 位置：单位旋转、90 度旋转、批量环境三种用例。
3. TCP 速度：纯平移、纯角速度切向项、平移加旋转三种用例。
4. 根坐标变换：非零 root 平移和旋转用例。
5. reset：五状态映射、名称/id 一致性、world/env 原点换算、部分 reset。
6. 奖励：progress 正负号、reset 首步为零、停车速度门控、dwell 连续性。
7. 终止：进入/退出迟滞、3 控制步成功、四类失败、timeout 只 truncated。

### 10.2 五状态物理稳定性测试

对五个 state 逐一固定 reset，至少记录：

- 初始化位置/姿态误差。
- target 最大单步位移、累计漂移和最大速度。
- TCP 初始世界位置、根坐标位置及到 preposition 的距离。
- 仿真期间是否出现接触、NaN、target 位移失败或速度失败。

测试脚本使用新版 `tasks/WithClaw/tests/test_five_states.py`，不覆盖现有 `scripts/test_target_states.py`。

### 10.3 短程训练 smoke test

分别运行 `WithoutClaw` 和 `WithClaw` 的短程训练，至少检查：

- 环境可以创建、reset 和 step。
- 观测维数固定且全部 finite。
- PPO 可以完成一个短 rollout 和一次参数更新。
- reward breakdown 包含正确任务的项，不出现跨任务项。
- `terminated` 与 `truncated` 统计分离。
- checkpoint/log 输出路径包含任务名。

物理稳定性和训练 smoke test 必须使用 Isaac Sim 自带 Python；普通 Python 只用于不依赖 Isaac Lab 的纯逻辑测试。

## 11. 已采用的决策

四阶段施工实际采用以下约定：

1. 目录采用本方案的 `tasks/WithoutClaw`、`tasks/WithClaw` 和 `tasks/common`。
2. 根据用户继续施工的指示，第一版按 Flange 局部轴直接反向采用 `(0, -0.12, 0.102)`；USD 几何证据与该解释存在冲突，因此该值仍是后续实测标定的唯一待替换项。
3. 成功采用本文件描述的 `0.03 m` 进入、`0.045 m` 退出的迟滞 dwell，而不是三步都严格小于 `0.03 m`。
4. 新版无接触停靠中，任何机器人 body 与 target 接触均作为非法碰撞。

以上约定已进入实现与验收结果；若获得实际 TCP 标定，只修改 `FLANGE_TO_TCP_TRANSLATION_F`，不改变其余 MDP 接口。
