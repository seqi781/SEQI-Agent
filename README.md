# Agent

这是一个面向 `terminal-bench` / `Harbor` 的 `LangGraph` 智能体项目。它不是 notebook demo，而是可以直接被 Harbor 通过 `import_path` 加载的 agent 实现。

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

## Agent 入口

主 agent 类：

```text
src.langgraph_terminal_agent:LangGraphTerminalBenchAgent
```

兼容旧配置的别名入口：

```text
src.patch_verify_agent:PatchVerifyTerminalAgent
src.multi_step_agent:MultiStepTerminalAgent
src.action_agent:ActionTerminalAgent
src.llm_single_loop_agent:LLMSingleLoopAgent
```

命令行查看 import path：

```bash
uv run python main.py --print-import-path
```

## Harbor / terminal-bench 配置示例

```json
{
  "name": null,
  "import_path": "src.langgraph_terminal_agent:LangGraphTerminalBenchAgent",
  "model_name": "openai:gpt-4.1-mini",
  "kwargs": {
    "max_steps": 24,
    "tool_timeout_sec": 180,
    "test_timeout_sec": 600,
    "working_dir": "/app"
  },
  "env": {}
}
```

## 内置工具

这个 agent 在 LangGraph 里挂了多种工具，适合 `terminal-bench` 这类代码修复/验证任务。

按职责分组如下：

- Inspect:
  `list_files`, `find_files`, `search_text`, `file_info`, `read_file`, `read_many_files`, `read_json`, `inspect_env`
- Edit:
  `write_file`, `write_json`, `append_file`, `replace_in_file`, `make_directory`, `copy_file`, `move_file`, `delete_file`, `apply_unified_diff`
- Verify:
  `exec_shell`, `check_command_available`, `run_program_with_input`, `compare_output`, `run_tests`, `list_processes`, `list_ports`, `wait_for_port`, `inspect_services`, `extract_test_signals`, `summarize_failures`, `propose_next_actions`
- Repo:
  `git_diff`, `git_status`
- Web:
  `brave_web_search`, `fetch_url`, `download_url`
- Extension:
  `create_helper_tool`, `create_python_tool`, `create_shell_tool`, `create_command_shim`, `install_helper_tool`

## 设计目标与原则

这个 agent 的默认目标不是“解释 benchmark”，而是“尽快产出真实可通过的结果”。

期望目标：

- 解当前任务实例，而不是修 benchmark harness
- 产出真实可通过的工件，而不是看起来合理的解释
- 用这次运行里的正证据验证成功，而不是把“没报错”当成“通过”
- 在工具不足时自补能力，但默认不去修改 verifier / `/tests/*`

运行原则：

- 先收集局部证据，再下判断
- 先做小实验，再做大改动
- 优先验证真实产物和真实输出
- 对 pytest / verifier 这类验证脚本，要求真实 runner 或正向成功信号
- 默认把 `/tests/*` 视为受保护的 benchmark 侧路径，不靠修测试来过题

当前实现还会把运行中的证据沉淀到结构化 state 里，包括：

- `evidence_log`
- `verification_state`
- `verification_summary`
- `helper_roles`
- `blocked_verifiers`
- `verified_failures` / `verified_successes`
- `rejected_solution_patterns`

验证类 helper 建议输出机器可读标记，agent 会把这些标记抽成结构化证据并自动生成 `next_actions`：

- `VERIFICATION_RESULT=PASS|FAIL|BLOCKED`
- `ALERT_PRESENT=1|0` 或 `ALERT_PRESENT=true|false`

例如一个自建浏览器 verifier 输出 `VERIFICATION_RESULT=FAIL` 和 `ALERT_PRESENT=0` 后，state 会进入 `negatively_verified`，后续规划会优先要求换方案家族并复用该 verifier helper。

## 代码结构

当前实现已经按职责拆分到包内：

```text
src/
  langgraph_terminal_agent.py        # 兼容导出层
  model_config.py                    # 模型与 LangSmith 配置
  terminal_agent/
    agent.py                         # 主 agent 类
    tools.py                         # 兼容导出层
    formatting.py                    # 输出格式化与文本处理
    streaming.py                     # 实时日志输出
    constants.py                     # 常量
    types.py                         # LangGraph state 类型
    toolkit/
      build.py                       # 工具装配入口
      inspect.py                     # 检查/读取类工具
      edit.py                        # 编辑类工具
      verify.py                      # 执行/验证/环境观测类工具
      repo.py                        # git 相关工具
      web.py                         # 网络搜索 / 抓取 / 下载工具
      extension.py                   # 自扩展 helper 工具
```

## 流式输出

这个 agent 会把执行过程实时写到两个地方：

- Harbor 的 `trial.log`
- agent 目录下的 `stream.log`

流式内容包括：

- `setup.start` / `setup.finish`
- `model.start` / `model.chunk` / `model.finish`
- `tool.start` / `tool.command` / `tool.finish` / `tool.result`
- `run.final_response`

跑测评时你可以直接看：

```bash
tail -f jobs/<job目录>/<trial目录>/trial.log
```

或者：

```bash
tail -f jobs/<job目录>/<trial目录>/agent/stream.log
```

## 模型

模型配置集中在：

```text
src/model_config.py
```

现在模型定义是最简单的单文件形式，直接写在 [model_config.py](/Users/seqi/Agent/src/model_config.py)：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5.4-nano",
)
```

如果你要换别的 OpenAI 官方模型，直接改这个文件里的 `model=` 即可。

## Brave Search

项目现在支持把 Brave Search 当作 agent 的一个可选工具，而不是默认依赖。

如果你要启用 `brave_web_search`，在 `.env` 里加入：

```bash
BRAVE_SEARCH_API_KEY=...
```

推荐策略：

- 先查本地文件和环境
- 本地证据不足时，再调用 `brave_web_search`
- 如果需要公开脚本或文档，再用 `fetch_url` / `download_url`
- 如果还是缺能力，就用 `create_python_tool` / `create_shell_tool` / `create_helper_tool` 在任务环境里生成一个临时 helper

当前临时 helper 默认会集中写到：

```text
/app/.agent-tools
```

agent 运行时会把已创建/已下载的 helper 记录到状态和日志里，后续步骤会优先复用这些 helper，而不是重复创建。

另外，`/app/.agent-tools` 会自动加入每次 shell 执行的 `PATH`。这意味着 agent 可以直接创建一个同名 shim，例如 `python`、`jq`、`file`，然后后续命令就能像系统命令一样调用它。

## LangSmith

项目已经接入 LangSmith。

当前行为：

- 启动时会自动加载项目根目录的 `.env`
- 默认开启 `LANGSMITH_TRACING=true`
- 默认项目名是 `agent`
- agent 的模型调用和 graph 运行会带 `run_name`、`tags`、`metadata`

你需要在 `.env` 里至少放这些变量：

```bash
LANGSMITH_API_KEY=...
OPENAI_API_KEY=...
```

可选：

```bash
LANGSMITH_PROJECT=agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## 启动 Jupyter

```bash
uv run jupyter lab
```

如果你希望在 Jupyter 中固定选择当前项目内核，先注册一次：

```bash
uv run python -m ipykernel install --user --name agent --display-name "Python (agent)"
```

之后在 Notebook 里选择 `Python (agent)` 即可。

## 本地验证

```bash
uv run python -m py_compile $(find src tests -name '*.py' -print) main.py
uv run python -m unittest discover -s tests -v
```

## 常用命令

```bash
uv run python main.py
uv run python main.py --print-import-path
uv run python
uv tree
```
