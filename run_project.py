"""One command reproduces the data, models, metrics and written artifacts."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/default.json")
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    commands=[["scripts/get_data.py"],["-m","src.train","--config",args.config],
              ["-m","src.evaluate"],["scripts/build_deliverables.py"]]
    for command in commands:
        subprocess.run([sys.executable,*command],cwd=root,check=True)

if __name__=="__main__":
    main()
