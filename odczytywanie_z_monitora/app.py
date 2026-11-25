"""Main application launcher: starts GUI by default, or CLI when --cli is passed."""

from __future__ import annotations

import sys

import cli
from monitor_reader.gui import run_gui


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        return cli.main(sys.argv[2:])
    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
