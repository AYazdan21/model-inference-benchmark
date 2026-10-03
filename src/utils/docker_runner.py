import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger("DockerRunner")

def is_docker_installed() -> bool:
    return shutil.which("docker") is not None

def is_docker_daemon_running() -> bool:
    if not is_docker_installed():
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return res.returncode == 0
    except Exception:
        return False

def build_docker_run_command(
    target: str,
    script_name: str,
    script_args: List[str],
    project_root: Path,
    max_ram_mb: Optional[int] = None,
    container_name: Optional[str] = None
) -> List[str]:
    """
    Constructs the docker compose command to execute the script
    inside the simulated hardware container.
    """
    compose_file = project_root / "docker" / "docker-compose.yml"
    
    cmd = [
        "docker", "compose",
        "-f", str(compose_file),
        "run", "--rm", "-T"  # -T: no pseudo-TTY, so output can be piped/captured (web app, CI)
    ]

    # In docker compose run, pass environment variables via -e if needed
    if max_ram_mb:
        cmd.extend(["-e", f"MAX_RAM_MB={max_ram_mb}"])

    if container_name:
        cmd.extend(["--name", container_name])

    # Map target to service name (matching targets.yaml to compose service)
    service_name = target
    cmd.append(service_name)

    # Internal container command
    cmd.extend(["python", f"scripts/{script_name}"])
    cmd.extend(script_args)

    return cmd
