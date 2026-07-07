# Isaac Lab 项目模板

[English](README.md)

## 概述

本项目/仓库是一个用于构建 Isaac Lab 项目或扩展的模板。
它允许你在 Isaac Lab 核心仓库之外，以独立环境开发自己的代码。

**主要特性：**

- `隔离性` 在 Isaac Lab 核心仓库之外工作，保证开发内容保持自包含。
- `灵活性` 该模板已配置为支持代码以 Omniverse 扩展形式运行。

**关键词：** extension, template, isaaclab

## 项目结构

本仓库围绕项目配置、可运行脚本、可复用工具、扩展源码和项目级工具组织。

```text
Test/
|-- README.md
|-- README_CN.md
|-- pyproject.toml
|-- .pre-commit-config.yaml
|-- .vscode/
|   `-- tools/
|       `-- setup_vscode.py
|-- configs/
|   |-- asset.py
|   |-- collision_cfg.py
|   |-- RenderCfg.py
|   |-- RLcfg.py
|   `-- Testcfg.py
|-- scripts/
|   |-- _bootstrap.py
|   |-- aubo.py
|   |-- eval.py
|   |-- test.py
|   |-- train.py
|   |-- list_envs.py
|   |-- zero_agent.py
|   |-- random_agent.py
|   |-- sb3/
|   |   |-- train.py
|   |   `-- play.py
|   |-- skrl/
|   |   |-- train.py
|   |   `-- play.py
|   |-- rl_games/
|   |   |-- train.py
|   |   `-- play.py
|   `-- rsl_rl/
|       |-- train.py
|       |-- play.py
|       `-- cli_args.py
|-- tasks/
|   |-- common/              # 通用 SB3 运行时与任务输出路径
|   |-- WithoutClaw/         # AUBO_E5.usd 法兰到达任务
|   `-- WithClaw/            # AUBO_E5_Withclaw.usd TCP 停靠任务
|-- docs/
|   `-- RL任务双轨目录与TCP停靠改造方案.md
|-- tools/
|   |-- camera.py
|   |-- contact.py
|   |-- logic.py
|   |-- rl.py
|   |-- scene.py
|   `-- tool.py
`-- source/
    `-- Test/
        |-- setup.py
        |-- pyproject.toml
        |-- config/
        |   `-- extension.toml
        |-- docs/
        |   `-- CHANGELOG.rst
        `-- Test/
            |-- __init__.py
            |-- ui_extension_example.py
            `-- tasks/
                |-- __init__.py
                |-- manager_based/
                |   `-- test/
                |       |-- test_env_cfg.py
                |       |-- mdp/
                |       |   `-- rewards.py
                |       `-- agents/
                |           |-- rsl_rl_ppo_cfg.py
                |           |-- sb3_ppo_cfg.yaml
                |           |-- rl_games_ppo_cfg.yaml
                |           `-- skrl_*.yaml
                `-- direct/
                    `-- test_marl/
                        |-- test_marl_env.py
                        |-- test_marl_env_cfg.py
                        `-- agents/
                            |-- rsl_rl_ppo_cfg.py
                            |-- sb3_ppo_cfg.yaml
                            |-- rl_games_ppo_cfg.yaml
                            `-- skrl_*.yaml
```

### 根目录

- `README.md`：英文主文档，包含安装、使用和开发说明。
- `README_CN.md`：主文档的中文翻译。
- `pyproject.toml`：仓库级 Python 工具配置，主要用于模板中的构建系统元信息。
- `.pre-commit-config.yaml`：格式化和静态检查钩子配置，用于保持代码风格一致。
- `.vscode/`：VSCode 相关辅助文件。`tools/setup_vscode.py` 用于生成本地 IDE 的 Python 路径配置，方便索引 Isaac Sim 和 Omniverse 包。

### `configs/`

该目录存放项目专用配置模块，通常由可运行脚本和任务逻辑引用。

- `asset.py`：集中管理本地 USD 资源根目录、场景 key、机器人位姿、工作站位姿和交互物体摆放信息。
- `collision_cfg.py`：描述工作站台面资源分组，并提供拆分 USD 场景加载辅助函数。
- `RenderCfg.py`：定义视觉测试运行使用的渲染配置和 RTX 运行时设置。
- `RLcfg.py`：定义 AUBO manager-based 强化学习环境、场景、观测、动作、奖励、事件和终止配置。
- `Testcfg.py`：定义视觉检查/测试场景，包括工作站资源、双 AUBO 机器人和相机传感器配置。

### `scripts/`

该目录存放可直接运行的入口脚本，是训练、评估、测试和检查任务注册时最常用的一层。

- `_bootstrap.py`：将仓库根目录加入 `sys.path`，使直接运行 `scripts/*.py` 时可以导入 `configs` 和 `tools`。
- `aubo.py`：用于在 Isaac Sim 中检查 AUBO 运动的交互式 IK 仿真脚本。
- `test.py`：视觉检查以及工作站/相机诊断脚本。
- `train.py`：基于本地 AUBO RL 配置的 SB3 PPO 训练入口。
- `eval.py`：基于本地 AUBO RL 配置的 SB3 checkpoint 评估入口。
- `list_envs.py`：列出该扩展注册的全部环境，用于确认任务注册和命名是否正确。
- `zero_agent.py` 和 `random_agent.py`：使用零动作或随机动作运行环境的健康检查脚本，适合正式训练前验证环境配置。
- `scripts/sb3/`、`scripts/skrl/`、`scripts/rl_games/`、`scripts/rsl_rl/`：不同强化学习后端的训练和回放入口。
- `rsl_rl/cli_args.py`：集中管理 `rsl_rl` 工作流专用命令行参数。

### `tools/`

该目录存放可复用项目逻辑和辅助模块，不作为直接启动入口。

- `scene.py`：场景/实体查找、位姿写入、工作空间检查和几何辅助函数。
- `rl.py`：RL 逻辑使用的动作和逐环境缓存辅助函数。
- `contact.py`：接触传感器兼容性辅助函数。
- `camera.py`：相机位姿设置和 PNG 图像保存辅助函数。
- `logic.py`：`configs/RLcfg.py` 使用的 AUBO 事件、奖励和终止函数。
- `tool.py`：兼容性门面，从 `tools` 包重新导出常用工具类。

### 项目内 AUBO 双任务链

根目录 `tasks/` 下的两条 AUBO 任务链相互隔离：

- `tasks/WithoutClaw/`：使用 `AUBO_E5.usd` 的旧版六关节无夹爪 Flange reach 任务。
- `tasks/WithClaw/`：使用带夹爪八关节场景和三维 Flange IK 动作的 TCP 低速持续停靠任务；TCP 只参与观测、奖励、工作空间和终止判断。
- `tasks/common/`：任务无关的 SB3 训练/验证与输出路径辅助逻辑。checkpoint 和 TensorBoard 日志分别写入 `checkpoints/<任务名>/sb3_aubo/` 与 `logs/<任务名>/sb3_aubo/`。

使用 Isaac Lab Python 环境运行：

```bash
python tasks/WithoutClaw/train.py --headless
python tasks/WithoutClaw/eval.py --headless --weight <checkpoint.zip>
python tasks/WithClaw/train.py --headless
python tasks/WithClaw/eval.py --headless --weight <checkpoint.zip>
```

WithClaw 离散状态物理验收入口为 `tasks/WithClaw/tests/test_five_states.py`。TCP 标定风险及完整阶段验证记录见 `docs/RL任务双轨目录与TCP停靠改造方案.md`。

### `source/Test/`

这是可安装扩展包的根目录，包含打包元数据、Omniverse 扩展配置，以及 Isaac Lab 运行时使用的实际 Python 模块。

- `setup.py`：定义 `Test` 包的安装方式，包括依赖项以及从 `config/extension.toml` 读取的元信息。
- `pyproject.toml`：扩展包的构建后端定义。
- `config/extension.toml`：核心扩展清单，定义包元数据、Isaac Lab 依赖和 Omniverse/Isaac Lab 需要加载的 Python 模块。
- `docs/CHANGELOG.rst`：预留的版本历史和变更日志位置。

### `source/Test/Test/`

该目录是真正会在运行时导入的 Python 包。

- `__init__.py`：标记 Python 包，通常会触发任务注册相关导入。
- `ui_extension_example.py`：Omniverse UI 扩展示例，展示如何将自定义界面接入扩展生命周期。
- `tasks/`：项目的任务库核心目录，包含环境、任务配置、奖励逻辑和 agent 配置。

### `tasks/` 任务组织

任务代码按环境实现风格分为两类：

- `manager_based/`：基于 Isaac Lab manager-based 工作流构建的任务，观测、奖励、事件和终止项由配置类组装。
- `direct/`：直接编写环境类的任务，环境逻辑更显式地写在 Python 代码中。

每个任务目录中常见文件的作用如下：

- `*_env_cfg.py`：环境配置文件，描述场景资源、观测项、动作空间、奖励项、重置规则和仿真参数。
- `*_env.py`：direct 风格任务的环境实现文件，定义场景创建、动作施加、观测拼接、奖励计算、终止判断和 reset 行为。
- `mdp/`：模块化任务逻辑，例如奖励项或其他被配置类复用的 MDP 辅助函数。
- `agents/`：算法专用训练配置。它们将同一个环境映射到不同 RL 库，使任务可以用 `sb3`、`skrl`、`rl_games` 或 `rsl_rl` 训练。

### 建议阅读顺序

如果你想高效理解或修改项目，建议按以下顺序阅读：

1. 从 `scripts/<backend>/train.py` 开始，了解任务如何被启动。
2. 然后打开 `source/Test/Test/tasks/` 下对应的任务目录。
3. 先阅读 `*_env_cfg.py`，理解环境定义和面向训练的配置。
4. 对 direct 任务，继续阅读 `*_env.py`，检查运行时逻辑。
5. 最后查看 `agents/`，了解不同后端的 PPO 或 MARL 超参数。

## 安装

- 按照 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab。
  推荐使用 conda 或 uv 安装方式，因为这样更方便从终端调用 Python 脚本。

- 将本项目/仓库单独克隆或复制到 Isaac Lab 安装目录之外，也就是不要放在 `IsaacLab` 目录内部：

- 使用已安装 Isaac Lab 的 Python 解释器，以可编辑模式安装该库：

    ```bash
    # 如果 Isaac Lab 未安装到 Python venv 或 conda 中，请用 'PATH_TO_isaaclab.sh|bat -p' 代替 'python'
    python -m pip install -e source/Test
    ```

- 验证扩展是否已正确安装：

    - 列出可用任务：

        注意：如果任务名称发生变化，可能需要更新 `scripts/list_envs.py` 中的搜索模式 `"Template-"`，这样任务才能被列出。

        ```bash
        # 如果 Isaac Lab 未安装到 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
        python scripts/list_envs.py
        ```

    - 运行任务：

        ```bash
        # 如果 Isaac Lab 未安装到 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - 使用 dummy agents 运行任务：

        这些脚本输出零动作或随机动作，可用于确认环境配置是否正确。

        - 零动作 agent

            ```bash
            # 如果 Isaac Lab 未安装到 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```

        - 随机动作 agent

            ```bash
            # 如果 Isaac Lab 未安装到 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### 设置 IDE（可选）

如需设置 IDE，请按以下步骤操作：

- 打开 VSCode 命令面板，按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后运行下拉菜单中的 `setup_python_env`。
  运行该任务时，会提示你输入 Isaac Sim 安装目录的绝对路径。

如果执行成功，它会在 `.vscode` 目录下创建 `.python.env` 文件。
该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展 Python 路径。
这有助于编辑代码时索引 Python 模块并获得智能提示。

### 作为 Omniverse 扩展设置（可选）

项目提供了一个 UI 扩展示例，启用 `source/Test/Test/ui_extension_example.py` 中定义的扩展后会加载它。

启用扩展的步骤如下：

1. **将本项目/仓库的搜索路径添加到扩展管理器**：
    - 通过 `Window` -> `Extensions` 打开扩展管理器。
    - 点击 **Hamburger Icon**，进入 `Settings`。
    - 在 `Extension Search Paths` 中输入本项目/仓库 `source` 目录的绝对路径。
    - 如果尚未添加，也在 `Extension Search Paths` 中输入 Isaac Lab 扩展目录路径（`IsaacLab/source`）。
    - 点击 **Hamburger Icon**，然后点击 `Refresh`。

2. **搜索并启用扩展**：
    - 在 `Third Party` 分类下找到你的扩展。
    - 切换开关以启用扩展。

## 代码格式化

模板提供了 pre-commit 配置，用于自动格式化代码。
安装 pre-commit：

```bash
pip install pre-commit
```

然后运行：

```bash
pre-commit run --all-files
```

## 故障排查

### Pylance 未索引扩展

在某些 VSCode 版本中，部分扩展可能未被索引。
这种情况下，可以在 `.vscode/settings.json` 的 `"python.analysis.extraPaths"` 下加入扩展路径。

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Test"
    ]
}
```

### Pylance 崩溃

如果遇到 `pylance` 崩溃，可能是索引文件过多导致内存不足。
一种解决方式是排除项目中没有使用到的部分 Omniverse 包。
为此，可以修改 `.vscode/settings.json`，注释掉 `"python.analysis.extraPaths"` 下的部分包路径。
以下是一些通常可以排除的包示例：

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // 动画包
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI 工具
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI 工具
"<path-to-isaac-sim>/extscache/omni.services.*"     // 服务工具
...
```
