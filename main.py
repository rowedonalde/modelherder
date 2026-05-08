"""Entry point for `uv run main.py` — delegates to the modelherder CLI."""

from modelherder.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
