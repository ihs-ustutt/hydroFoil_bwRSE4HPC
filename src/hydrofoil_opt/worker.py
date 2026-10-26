"""Command-line dispatcher for hydrofoil evaluation stages."""

import argparse
from pathlib import Path

from hydrofoil_opt.runtime import configure_turbulent, finalize, prepare


def main(argv: list[str] | None = None) -> int:
    """Execute one stage; let the backend record failures."""
    parser = argparse.ArgumentParser(prog="hydrofoil-opt-worker")
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "configure-turbulent", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("request", type=Path)
        if name == "finalize":
            command.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepare(args.request)
    elif args.stage == "configure-turbulent":
        configure_turbulent(args.request)
    else:
        finalize(args.request, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
