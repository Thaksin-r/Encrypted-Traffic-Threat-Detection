"""Allow `python -m ettd.pipeline_extract` via package helpers."""

from ettd.pipeline_extract import main

if __name__ == "__main__":
    raise SystemExit(main())
