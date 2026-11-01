"""Package submission artifacts without datasets, weights, environments or course PDFs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
NAME="ec417-miniproject-P22-623147-623138"

def main():
    files=[ROOT/p for p in ["README.md","requirements.txt","LICENSE",".gitignore","run_project.py"]]
    for folder in ["src","scripts","configs","notebooks","results","report","tests","docs"]:
        files.extend(p for p in (ROOT/folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in [".pyc",".npz"])
    files.extend([ROOT/"data/README.md",ROOT/"data/source_manifest.json"])
    with ZipFile(ROOT/f"{NAME}.zip","w",ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p,Path(NAME)/p.relative_to(ROOT))
    print(ROOT/f"{NAME}.zip")

if __name__=="__main__":
    main()
