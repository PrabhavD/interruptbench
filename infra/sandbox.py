from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class DockerSandbox:
    """
    Thin Python wrapper around `docker run` for executing generated code inside
    a constrained local container.

    Safety defaults:
      - no network access (`--network none`)
      - CPU limit (`--cpus`)
      - memory limit (`--memory`)
      - process limit (`--pids-limit`)
      - read-only root filesystem (`--read-only`)
      - tmpfs-backed writable /tmp (`--tmpfs /tmp`)
      - non-root user inherited from the image
    """

    def __init__(
        self,
        image: str = "interruptbench:base",
        cpus: float = 1.0,
        memory: str = "512m",
        timeout_sec: int = 30,
        pids_limit: int = 64,
        workdir: str = "/workspace",
    ) -> None:
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.timeout_sec = timeout_sec
        self.pids_limit = pids_limit
        self.workdir = workdir

    def run_python(self, code: str) -> SandboxResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "sandbox_task.py"
            script_path.write_text(textwrap.dedent(code), encoding="utf-8")
            return self._run(
                ["python", f"{self.workdir}/sandbox_task.py"],
                mount_dir=tmpdir,
            )

    def run_script(
        self,
        local_script_path: str,
        extra_args: Optional[list[str]] = None,
    ) -> SandboxResult:
        host_path = Path(local_script_path).resolve()
        if not host_path.exists():
            raise FileNotFoundError(f"Script not found: {host_path}")
        args = extra_args or []
        return self._run(
            ["python", f"{self.workdir}/{host_path.name}", *args],
            mount_dir=str(host_path.parent),
        )

    def _run(self, container_cmd: list[str], mount_dir: str) -> SandboxResult:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--pids-limit",
            str(self.pids_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            f"{Path(mount_dir).resolve()}:{self.workdir}:ro",
            "-w",
            self.workdir,
            self.image,
            *container_cmd,
        ]
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                timed_out=False,
                command=docker_cmd,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                returncode=124,
                timed_out=True,
                command=docker_cmd,
            )


if __name__ == "__main__":
    sandbox = DockerSandbox()
    result = sandbox.run_python("""
print('hello from sandbox')
""")
    print(json.dumps({
        "ok": result.ok,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": result.command,
    }, indent=2))