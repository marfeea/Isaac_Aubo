# Isaac Lab 项目模板

## 概述

本项目/仓库是一个用于构建基于 Isaac Lab 的项目或扩展的模板。
它允许你在独立环境中进行开发，而无需直接在 Isaac Lab 核心仓库内工作。

**主要特性：**

- `隔离性` 在 Isaac Lab 核心仓库之外开展工作，确保你的开发内容保持独立、自包含。
- `灵活性` 该模板已配置为支持你的代码以 Omniverse 扩展的形式运行。

**关键词：** extension，template，isaaclab

## 项目结构说明

本仓库主要由三部分组成：可执行脚本、扩展源码，以及项目级配置/工具文件。

```text
Test/
|-- README.md
|-- READMECN.md
|-- pyproject.toml
|-- .pre-commit-config.yaml
|-- .vscode/
|   `-- tools/
|       `-- setup_vscode.py
|-- scripts/
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

### 根目录文件

- `README.md`：英文主文档，包含安装、运行和开发说明。
- `READMECN.md`：中文文档，用于补充或同步主文档内容。
- `pyproject.toml`：仓库级 Python 构建配置，当前主要用于声明构建系统元信息。
- `.pre-commit-config.yaml`：代码格式化与静态检查钩子配置。
- `.vscode/`：VSCode 相关辅助文件，其中 `tools/setup_vscode.py` 用于生成本地 IDE 所需的 Python 路径配置，方便索引 Isaac Sim 和 Omniverse 包。

### `scripts/`

该目录存放项目的命令行入口脚本，是训练、测试、回放和排查环境注册问题时最常用的一层。

- `list_envs.py`：列出当前扩展注册到 Isaac Lab 的所有环境，用于确认任务是否注册成功、任务名是否正确。
- `zero_agent.py`、`random_agent.py`：使用零动作或随机动作运行环境，适合在正式训练前做基础联通性验证。
- `scripts/sb3/`、`scripts/skrl/`、`scripts/rl_games/`、`scripts/rsl_rl/`：针对不同强化学习后端提供的训练与回放脚本目录。
- `train.py`：对应后端的训练入口，负责加载任务并启动训练流程。
- `play.py`：对应后端的回放/推理入口，用于加载策略并观察运行效果。
- `rsl_rl/cli_args.py`：封装 `rsl_rl` 训练流程使用的命令行参数定义。

### `source/Test/`

该目录是可安装扩展的根目录，包含打包配置、Omniverse 扩展清单，以及真正的 Python 包源码。

- `setup.py`：定义 `Test` 包的安装方式，包括依赖项以及从 `config/extension.toml` 读取的包元信息。
- `pyproject.toml`：定义扩展包的构建后端。
- `config/extension.toml`：扩展清单文件，声明包名称、版本、依赖的 Isaac Lab 模块，以及需要被加载的 Python 模块。
- `docs/CHANGELOG.rst`：版本记录与变更日志位置。

### `source/Test/Test/`

这里是运行时真正被导入的 Python 包，也是项目核心代码所在目录。

- `__init__.py`：用于标记 Python 包，并通常承担任务注册相关的导入逻辑。
- `ui_extension_example.py`：Omniverse UI 扩展示例，演示如何在扩展启用后挂载自定义界面。
- `tasks/`：项目最核心的任务库目录，环境定义、任务配置、奖励函数、训练配置基本都放在这里。

### `tasks/` 任务目录划分

任务代码按环境实现风格分为两类：

- `manager_based/`：基于 Isaac Lab manager 机制构建的任务，通常通过配置类组合 observation、reward、event 和 termination。
- `direct/`：直接编写环境类的任务，环境逻辑、动作应用、奖励计算等更显式地写在 Python 代码中。

每个任务子目录中常见文件的作用如下：

- `*_env_cfg.py`：环境配置文件，定义场景资源、观测项、动作空间、奖励项、重置规则和仿真参数。
- `*_env.py`：直接式任务的环境实现文件，负责场景创建、动作施加、观测拼接、奖励计算、终止判断和 reset 行为。
- `mdp/`：存放 MDP 相关的辅助逻辑，例如奖励函数、状态计算函数或其他可复用任务组件。
- `agents/`：按强化学习框架划分的训练配置文件。相同任务可以在这里分别配置 `sb3`、`skrl`、`rl_games`、`rsl_rl` 等后端的超参数。

### 推荐阅读顺序

如果你准备理解或修改这个项目，建议按下面顺序阅读：

1. 先看 `scripts/<backend>/train.py`，理解任务是如何被启动的。
2. 再进入 `source/Test/Test/tasks/` 中对应的任务目录。
3. 优先阅读 `*_env_cfg.py`，先建立对环境配置和训练接口的整体认识。
4. 如果是 direct 类型任务，再继续阅读 `*_env.py`，理解运行时逻辑。
5. 最后查看 `agents/` 目录，确认不同训练框架下使用的超参数配置。

## 安装

- 按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab。
  推荐使用 conda 或 uv 方式安装，这样可以更方便地从终端调用 Python 脚本。

- 在 Isaac Lab 安装目录之外，单独克隆或复制本项目/仓库（也就是放在 `IsaacLab` 目录外）：

- 使用一个已经安装 Isaac Lab 的 Python 解释器，以可编辑模式安装该库：

    ```bash
    # 如果 Isaac Lab 不是安装在 Python venv 或 conda 中，请使用 'PATH_TO_isaaclab.sh|bat -p' 代替 'python'
    python -m pip install -e source/Test
    ```

- 通过以下方式验证扩展是否已正确安装：

    - 列出可用任务：

        注意：如果任务名称发生变化，可能需要更新搜索模式 `"Template-"`
        （位于 `scripts/list_envs.py` 文件中），这样任务才能被列出。

        ```bash
        # 如果 Isaac Lab 不是安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
        python scripts/list_envs.py
        ```

    - 运行任务：

        ```bash
        # 如果 Isaac Lab 不是安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - 使用虚拟智能体运行任务：

        这里包含了一些输出全零动作或随机动作的虚拟智能体，可用于确认环境是否配置正确。

        - 零动作智能体

            ```bash
            # 如果 Isaac Lab 不是安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```

        - 随机动作智能体

            ```bash
            # 如果 Isaac Lab 不是安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### 配置 IDE（可选）

如需配置 IDE，请按照以下说明进行操作：

- 运行 VSCode Tasks：按下 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后在下拉菜单中运行 `setup_python_env`。
  运行该任务时，系统会提示你填写 Isaac Sim 安装路径的绝对路径。

如果一切执行正常，它会在 `.vscode` 目录下创建一个 `.python.env` 文件。
该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展的 Python 路径。
这有助于在编写代码时为所有 Python 模块建立索引并提供智能提示。

### 配置为 Omniverse 扩展（可选）

我们提供了一个示例 UI 扩展，在启用 `source/Test/Test/ui_extension_example.py` 中定义的扩展后会自动加载。

要启用你的扩展，请按以下步骤操作：

1. **将当前项目/仓库的搜索路径添加到扩展管理器**：
    - 通过 `Window` -> `Extensions` 打开扩展管理器。
    - 点击 **Hamburger Icon**，然后进入 `Settings`。
    - 在 `Extension Search Paths` 中，输入本项目/仓库 `source` 目录的绝对路径。
    - 如果尚未添加，在 `Extension Search Paths` 中再输入 Isaac Lab 扩展目录所在路径（`IsaacLab/source`）。
    - 点击 **Hamburger Icon**，然后点击 `Refresh`。

2. **搜索并启用你的扩展**：
    - 在 `Third Party` 分类下找到你的扩展。
    - 打开开关以启用该扩展。

## 代码格式化

我们提供了一个 pre-commit 模板，用于自动格式化代码。
安装 pre-commit：

```bash
pip install pre-commit
```

然后可以通过以下命令运行 pre-commit：

```bash
pre-commit run --all-files
```

## 故障排查

### Pylance 未索引到扩展

在某些 VSCode 版本中，部分扩展可能不会被正确索引。
这种情况下，请在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 键下添加你的扩展路径。

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Test"
    ]
}
```

### Pylance 崩溃

如果你遇到 `pylance` 崩溃，通常意味着被索引的文件过多，导致内存不足。
一种可行的解决办法是排除一些项目中未使用的 Omniverse 包。
为此，请修改 `.vscode/settings.json`，并在 `"python.analysis.extraPaths"` 键下将部分包注释掉。
以下是一些通常可以排除的包示例：

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // 动画相关包
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI 工具
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI 工具
"<path-to-isaac-sim>/extscache/omni.services.*"     // 服务工具
...
```
