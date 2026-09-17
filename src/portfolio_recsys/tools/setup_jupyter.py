from __future__ import annotations

import subprocess
import sys


def main() -> None:
    kernel_name = "portfolio_recsys"
    display_name = "Portfolio Recsys Environment"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            kernel_name,
            "--display-name",
            display_name,
        ],
        check=True,
    )

    print(f"✅ Kernel '{display_name}' instalado correctamente.")
