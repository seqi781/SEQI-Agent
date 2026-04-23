from pydantic import BaseModel, ConfigDict, Field


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListFilesInput(ToolInput):
    path: str = Field(default=".", description="Directory path to inspect.")
    max_entries: int = Field(default=200, description="Maximum number of paths to return.")
    include_hidden: bool = Field(default=False, description="Whether to include hidden files.")


class FindFilesInput(ToolInput):
    pattern: str = Field(description="Glob-style filename pattern such as *.py or *test*.")
    path: str = Field(default=".", description="Directory path to search under.")
    max_entries: int = Field(default=200, description="Maximum number of matching files to return.")


class SearchTextInput(ToolInput):
    pattern: str = Field(description="Text or regex pattern to search for.")
    path: str = Field(default=".", description="Directory path to search under.")
    max_matches: int = Field(default=200, description="Maximum number of matches to return.")


class FileInfoInput(ToolInput):
    path: str = Field(description="File or directory path to inspect.")


class ReadFileInput(ToolInput):
    path: str = Field(description="File path to read.")
    start_line: int = Field(default=1, description="First line number to include.")
    end_line: int = Field(default=200, description="Last line number to include.")


class ReadManyFilesInput(ToolInput):
    paths: list[str] = Field(description="List of file paths to read.")
    start_line: int = Field(default=1, description="First line number to include.")
    end_line: int = Field(default=200, description="Last line number to include.")


class ReadJsonInput(ToolInput):
    path: str = Field(description="JSON file path to inspect.")


class InspectEnvInput(ToolInput):
    max_env_vars: int = Field(default=80, description="Maximum number of env vars to display.")


class InspectFileBytesInput(ToolInput):
    path: str = Field(description="Binary file path to inspect.")
    offset: int = Field(default=0, description="Byte offset to start reading from.")
    length: int = Field(default=256, description="Number of bytes to inspect.")


class ScanStringsInput(ToolInput):
    path: str = Field(description="File path to scan for printable strings.")
    pattern: str = Field(default="", description="Optional filter pattern for extracted strings.")
    max_matches: int = Field(default=80, description="Maximum number of extracted strings to return.")
    min_length: int = Field(default=4, description="Minimum printable string length.")


class WriteFileInput(ToolInput):
    path: str = Field(description="Destination file path.")
    content: str = Field(description="Full file content to write.")


class WriteJsonInput(ToolInput):
    path: str = Field(description="Destination JSON file path.")
    content: str = Field(description="JSON string to validate and write.")


class MakeDirectoryInput(ToolInput):
    path: str = Field(description="Directory path to create.")


class AppendFileInput(ToolInput):
    path: str = Field(description="Destination file path.")
    content: str = Field(description="Text to append.")


class ReplaceInFileInput(ToolInput):
    path: str = Field(description="Target file path.")
    old_text: str = Field(description="Text to replace.")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of only the first.")


class CopyFileInput(ToolInput):
    source: str = Field(description="Source file or directory path.")
    destination: str = Field(description="Destination path.")


class MoveFileInput(ToolInput):
    source: str = Field(description="Source file or directory path.")
    destination: str = Field(description="Destination path.")


class DeleteFileInput(ToolInput):
    path: str = Field(description="Path to remove.")
    recursive: bool = Field(default=False, description="Allow directory deletion.")
    force: bool = Field(default=True, description="Ignore missing files.")


class ApplyUnifiedDiffInput(ToolInput):
    diff: str = Field(description="Unified diff text to apply.")
    cwd: str | None = Field(default=None, description="Optional working directory override.")


class ExecShellInput(ToolInput):
    command: str = Field(description="Shell command to execute in the task environment.")
    cwd: str | None = Field(default=None, description="Optional working directory override.")
    timeout_sec: int = Field(default=30, description="Execution timeout in seconds.")


class CheckCommandAvailableInput(ToolInput):
    command_name: str = Field(description="Command name to probe with command -v.")


class RunProgramWithInputInput(ToolInput):
    command: str = Field(description="Command to execute.")
    stdin_text: str = Field(default="", description="Text piped to stdin.")
    cwd: str | None = Field(default=None, description="Optional working directory override.")
    timeout_sec: int = Field(default=30, description="Execution timeout in seconds.")


class CompareOutputInput(ToolInput):
    actual: str = Field(description="Actual text output.")
    expected: str = Field(description="Expected text output.")
    mode: str = Field(default="exact", description="Comparison mode: exact, contains, or diff.")


class RunTestsInput(ToolInput):
    command: str = Field(description="Test, verifier, or build command to execute.")
    cwd: str | None = Field(default=None, description="Optional working directory override.")
    timeout_sec: int = Field(default=30, description="Execution timeout in seconds.")


class ListProcessesInput(ToolInput):
    max_entries: int = Field(default=80, description="Maximum number of processes to display.")


class ListPortsInput(ToolInput):
    max_entries: int = Field(default=80, description="Maximum number of listening ports to display.")


class WaitForPortInput(ToolInput):
    host: str = Field(default="127.0.0.1", description="Host to probe.")
    port: int = Field(default=8000, description="Port number to probe.")
    timeout_sec: int = Field(default=30, description="Maximum wait time in seconds.")
    interval_sec: int = Field(default=1, description="Polling interval in seconds.")


class InspectServicesInput(ToolInput):
    max_entries: int = Field(default=50, description="Maximum number of processes and ports to display.")


class ExtractTestSignalsInput(ToolInput):
    text: str = Field(description="Raw test or verifier output.")
    max_lines: int = Field(default=40, description="Maximum number of interesting lines to keep.")


class SummarizeFailuresInput(ToolInput):
    text: str = Field(description="Raw failure text or logs.")
    max_lines: int = Field(default=12, description="Maximum number of failure categories to summarize.")


class ProposeNextActionsInput(ToolInput):
    text: str = Field(description="Failure text or observations to analyze.")
    max_actions: int = Field(default=8, description="Maximum number of proposed next actions.")


class GitDiffInput(ToolInput):
    cwd: str | None = Field(default=None, description="Optional working directory override.")


class GitStatusInput(ToolInput):
    cwd: str | None = Field(default=None, description="Optional working directory override.")


class BraveWebSearchInput(ToolInput):
    query: str = Field(description="Search query.")
    count: int = Field(default=5, description="Maximum number of results to return.")
    country: str = Field(default="us", description="Two-letter country code.")
    search_lang: str = Field(default="en", description="Preferred search language.")


class FetchUrlInput(ToolInput):
    url: str = Field(description="Public URL to fetch.")
    max_chars: int = Field(default=12000, description="Maximum number of characters to return.")


class DownloadUrlInput(ToolInput):
    url: str = Field(description="Public URL to download.")
    destination: str = Field(description="Destination path in the task environment.")
    executable: bool = Field(default=False, description="Whether to chmod +x the downloaded file.")


class CreateHelperToolInput(ToolInput):
    path: str = Field(description="Helper script path, relative paths land in the helper directory.")
    content: str = Field(description="Helper script content. If this is a verifier helper, print VERIFICATION_RESULT=PASS|FAIL|BLOCKED and task-specific facts when possible.")
    executable: bool = Field(default=True, description="Whether to mark the helper executable.")
    usage_note: str = Field(default="", description="Short usage note shown after creation.")


class CreatePythonToolInput(ToolInput):
    path: str = Field(description="Helper script path, relative paths land in the helper directory.")
    content: str = Field(description="Python helper content. If this is a verifier helper, print VERIFICATION_RESULT=PASS|FAIL|BLOCKED and task-specific facts when possible.")
    executable: bool = Field(default=True, description="Whether to mark the helper executable.")
    usage_note: str = Field(default="", description="Short usage note shown after creation.")


class CreateShellToolInput(ToolInput):
    path: str = Field(description="Helper script path, relative paths land in the helper directory.")
    content: str = Field(description="Shell helper content. If this is a verifier helper, print VERIFICATION_RESULT=PASS|FAIL|BLOCKED and task-specific facts when possible.")
    executable: bool = Field(default=True, description="Whether to mark the helper executable.")
    usage_note: str = Field(default="", description="Short usage note shown after creation.")


class InstallHelperToolInput(ToolInput):
    source_url: str = Field(description="Public URL for the helper script.")
    destination: str = Field(description="Destination path, relative paths land in the helper directory.")
    executable: bool = Field(default=True, description="Whether to chmod +x the installed helper.")


class CreateCommandShimInput(ToolInput):
    command_name: str = Field(description="Command name to create in the helper directory.")
    content: str = Field(description="Shell or Python script body for the shim.")
    kind: str = Field(default="shell", description="Shim implementation type: shell or python.")
    usage_note: str = Field(default="", description="Short usage note shown after creation.")
