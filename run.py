#!/usr/bin/env python3
import sys
from pathlib import Path
import traceback

# Insert src/ at front of sys.path so modules inside it can use relative imports
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

# Now import the pipeline module (the module lives in src/)
from pipeline import PipelineRunner  # imports src/pipeline.py as module 'pipeline'


def main(config_path: str = "./config.yaml"):
    """
    Initialize and run the pipeline using the specified config file.
    Returns 0 on success, 1 on failure.
    """
    runner = PipelineRunner(config_path)
    try:
        success = runner.run()
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "./config.yaml"
    sys.exit(main(cfg))
