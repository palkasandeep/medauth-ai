from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional, Set

from server.utils.config import (
    get_rules_config,
    get_workflow_dev_config,
    resolve_project_path,
)


class MonitorAgent:
    def __init__(
        self,
        watch_dir: Optional[str | Path] = None,
        processed_log_path: Optional[str | Path] = None,
        supported_extensions: Optional[List[str]] = None,
        on_new_file: Optional[Callable[[Path], None]] = None,
    ) -> None:
        rules = get_rules_config()
        self.watch_dir = (
            Path(watch_dir).resolve()
            if watch_dir
            else resolve_project_path(rules["reporting"]["incoming_dir"])
        )
        self.processed_log_path = (
            Path(processed_log_path).resolve()
            if processed_log_path
            else self.watch_dir / rules["reporting"]["processed_log_filename"]
        )
        exts = supported_extensions or rules["supported_extensions"]
        self.supported_extensions: Set[str] = {e.lower() for e in exts}
        self.on_new_file = on_new_file
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_processed(self) -> Set[str]:
        if not self.processed_log_path.exists():
            return set()
        return {
            line.strip()
            for line in self.processed_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def _append_processed(self, filename: str) -> None:
        self.processed_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.processed_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{filename}\n")

    def poll(self, callback: Optional[Callable[[Path], None]] = None) -> List[Path]:
        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True, exist_ok=True)
            return []

        processed = self._read_processed()
        cb = callback or self.on_new_file
        new_files: List[Path] = []

        for entry in sorted(self.watch_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            if entry.resolve() == self.processed_log_path.resolve():
                continue
            if entry.suffix.lower() not in self.supported_extensions:
                continue
            if entry.name in processed:
                continue

            self._append_processed(entry.name)
            processed.add(entry.name)
            new_files.append(entry)
            if cb:
                cb(entry)

        return new_files

    def start_polling(self, interval_ms: Optional[int] = None) -> None:
        ms = interval_ms or get_workflow_dev_config()["poll_interval_ms"]
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.wait(ms / 1000.0):
                self.poll()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        self._stop_event.set()


def self_test() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="medauth-monitor-"))
    try:
        (tmp_dir / ".hidden.txt").write_text("ignore", encoding="utf-8")
        (tmp_dir / "bad.exe").write_text("ignore", encoding="utf-8")
        (tmp_dir / "pa_form.TXT").write_text("PRIOR AUTHORIZATION REQUEST", encoding="utf-8")
        (tmp_dir / "labs.JSON").write_text('{"doc_type":"lab_results"}', encoding="utf-8")

        fired: List[str] = []
        agent = MonitorAgent(
            watch_dir=tmp_dir,
            on_new_file=lambda p: fired.append(p.name),
        )
        first = agent.poll()
        assert len(first) == 2
        assert sorted(fired) == ["labs.JSON", "pa_form.TXT"]
        assert (tmp_dir / "processed.log").exists()

        second = agent.poll()
        assert len(second) == 0
        assert len(fired) == 2
        print("[PASS] MonitorAgent self_test")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    self_test()
