"""Execute both notebooks with this Python interpreter and workspace-local caches."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for name in ["IPYTHONDIR","JUPYTER_CONFIG_DIR","JUPYTER_RUNTIME_DIR","MPLCONFIGDIR"]:
    cache=ROOT/"tmp"/name.lower()
    cache.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(cache)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

for filename in ["01_eda.ipynb","02_experiments.ipynb"]:
    path=ROOT/"notebooks"/filename
    nb=nbformat.read(path,as_version=4)
    km=KernelManager(kernel_name="python3")
    km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    NotebookClient(nb,km=km,timeout=120,resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(nb,path)
    print("Executed",filename,flush=True)
