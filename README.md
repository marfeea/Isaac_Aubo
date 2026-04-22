# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Project Structure

The repository is organized around three concerns: runnable scripts, extension source code, and project-level tooling/configuration.

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

### Root directory

- `README.md`: Main project documentation, including installation, usage, and development notes.
- `READMECN.md`: Chinese translation of the main documentation.
- `pyproject.toml`: Repository-level Python tooling configuration, mainly used for build-system metadata in this template.
- `.pre-commit-config.yaml`: Formatting and linting hooks used to keep the codebase consistent.
- `.vscode/`: VSCode-related helper files. `tools/setup_vscode.py` is used to generate local IDE Python path configuration for Isaac Sim and Omniverse packages.

### `scripts/`

This directory stores runnable entry scripts. These files are the command-line layer of the project and are usually the first place to start when training, evaluating, or inspecting tasks.

- `list_envs.py`: Lists all registered environments exposed by this extension so you can confirm task registration and naming.
- `zero_agent.py` and `random_agent.py`: Sanity-check scripts that run environments with trivial actions to verify environment setup before training.
- `scripts/sb3/`, `scripts/skrl/`, `scripts/rl_games/`, `scripts/rsl_rl/`: Training and playback entry points for different RL backends.
- `train.py`: Starts training for the selected backend and task.
- `play.py`: Loads a trained policy or runs inference/visual evaluation for the selected backend.
- `rsl_rl/cli_args.py`: Centralizes command-line arguments specific to the `rsl_rl` workflow.

### `source/Test/`

This is the installable extension package root. It contains packaging metadata, Omniverse extension configuration, and the actual Python module used by Isaac Lab.

- `setup.py`: Defines how the `Test` package is installed, including dependencies and metadata loaded from `config/extension.toml`.
- `pyproject.toml`: Build backend definition for packaging the extension.
- `config/extension.toml`: Core extension manifest. It defines package metadata, Isaac Lab dependencies, and the Python module that Omniverse/Isaac Lab should load.
- `docs/CHANGELOG.rst`: Reserved location for version history and release notes.

### `source/Test/Test/`

This directory is the real Python package imported at runtime.

- `__init__.py`: Marks the package and usually triggers task registration imports.
- `ui_extension_example.py`: Example Omniverse UI extension showing how to attach a custom UI component to the extension lifecycle.
- `tasks/`: The core task library of the project. This is where environments, task configs, reward logic, and agent configs live.

### `tasks/` task organization

The task code is split by environment style:

- `manager_based/`: Tasks built with Isaac Lab's manager-based workflow, where observations, rewards, events, and terminations are assembled from config classes.
- `direct/`: Tasks implemented with direct environment classes, where environment logic is written more explicitly in Python.

Inside each task folder:

- `*_env_cfg.py`: Environment configuration files. They describe scene assets, observation terms, action spaces, reward terms, reset rules, and simulation parameters.
- `*_env.py`: Environment implementation files for direct-style tasks. They define scene setup, action application, observation assembly, reward computation, termination logic, and reset behavior.
- `mdp/`: Modular task logic such as reward terms or other MDP helper functions reused by configuration classes.
- `agents/`: Algorithm-specific training configs. These files map one environment to different RL libraries so the same task can be trained with `sb3`, `skrl`, `rl_games`, or `rsl_rl`.

### Practical reading order

If you want to understand or modify the project efficiently, read files in this order:

1. Start from `scripts/<backend>/train.py` to see how a task is launched.
2. Then open the corresponding task folder under `source/Test/Test/tasks/`.
3. Read `*_env_cfg.py` first to understand the environment definition and training-facing configuration.
4. For direct tasks, continue into `*_env.py` to inspect runtime logic.
5. Finally, check `agents/` to see backend-specific PPO or MARL hyperparameters.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Test

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/Test/Test/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Test"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
