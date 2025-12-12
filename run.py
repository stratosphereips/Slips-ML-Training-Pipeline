#!/usr/bin/env python3
import sys
from pathlib import Path

# Insert src/ at front of sys.path so modules inside it can use relative imports
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

# Now import the pipeline module (the module lives in src/)
from pipeline import PipelineRunner  # imports src/pipeline.py as module 'pipeline'

def main(config_path: str = "./config.yaml"):
    runner = PipelineRunner(config_path)
    success = runner.run()
    return 0 if success else 1

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "./config.yaml"
    sys.exit(main(cfg))
