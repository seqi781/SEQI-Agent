# Agent

这是一个使用 `LangGraph` 构建智能体的 Python 项目，使用 `uv` 管理解释器、依赖和虚拟环境，并支持直接在 `JupyterLab` 中开发。

## 环境要求

- `uv`
- Python `3.12`

## 初始化环境

```bash
uv sync
```

这会创建 `.venv` 并安装：

- 运行时依赖：`langgraph`、`langchain`、`langchain-core`、`langsmith`
- 开发依赖：`jupyterlab`、`ipykernel`、`python-dotenv`

## 启动 Jupyter

```bash
uv run jupyter lab
```

如果你希望在 Jupyter 中固定选择当前项目内核，先注册一次：

```bash
uv run python -m ipykernel install --user --name agent --display-name "Python (agent)"
```

之后在 Notebook 里选择 `Python (agent)` 即可。

## 常用命令

```bash
uv run python main.py
uv run python
uv tree
```
