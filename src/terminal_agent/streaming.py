from datetime import datetime
from pathlib import Path
from typing import Protocol


class LoggerLike(Protocol):
    def info(self, msg: str) -> None: ...


class StreamEmitter:
    def __init__(self, logger: LoggerLike, logs_dir: Path, stream_log_path: Path):
        self.logger = logger
        self.logs_dir = logs_dir
        self.stream_log_path = stream_log_path

    def emit(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.logger.info(line)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.stream_log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def emit_block(self, title: str, content: str) -> None:
        self.emit(f"{title}\n{content}")
