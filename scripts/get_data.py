"""Download the documented TensorFlow mirror of the MPI Jena 2009-2016 data."""
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
EXPECTED_SHA256 = "63d757501e92284a7de7cdbef0337f03b24e13ead7ac2b5b8c86a18d8e38ba5b"

def main():
    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "jena_climate_2009_2016.csv.zip"
    csv = folder / "jena_climate_2009_2016.csv"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        print("Downloading", URL, flush=True)
        with urlopen(URL, timeout=120) as source, partial.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive checksum differs from the recorded experiment. Do not silently change dataset versions.")
    with zipfile.ZipFile(archive) as z:
        # Extract only the expected member, never arbitrary archive paths.
        with z.open(csv.name) as source, csv.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    manifest = ROOT / "data/source_manifest.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    metadata = {
        "dataset": "Jena Climate 2009-2016", "download_url": URL,
        "official_source": "https://www.bgc-jena.mpg.de/wetter/",
        "accessed_utc": old.get("accessed_utc", datetime.now(timezone.utc).isoformat()),
        "archive_sha256": digest,
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "terms_url": "https://www.bgc-jena.mpg.de/wetter/weather_data.html"
    }
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(csv, flush=True)

if __name__ == "__main__":
    main()
