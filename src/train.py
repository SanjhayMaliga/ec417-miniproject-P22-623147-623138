"""Train all models; select checkpoints on validation 24h MAE only."""
import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from .data import ROOT, prepare
from .models import Forecaster, linear_model

def predict(model, dataset, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size):
            predictions.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(predictions) * dataset.target_std + dataset.target_mean

def run(config_path="configs/default.json"):
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    output = ROOT / "checkpoints"
    output.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    datasets, _, audit = prepare(config)
    (ROOT / "results/data_audit.json").write_text(json.dumps(audit, indent=2))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    log, selections = [], []
    train, val = datasets["train"], datasets["validation"]
    # Highly correlated weather features need double precision in the ridge solve.
    x, xv = train.flattened().astype(np.float64), val.flattened().astype(np.float64)
    start = time.perf_counter()
    best, best_alpha, chosen = float("inf"), None, None
    primary = config["horizons"].index(24)
    with threadpool_limits(limits=config["threads"]):
        for alpha in config["ridge_alphas"]:
            model = linear_model(alpha).fit(x, train.targets())
            score = float(np.abs(model.predict(xv)[:,primary] - val.targets()[:,primary]).mean())
            selections.append({"model":"Ridge", "alpha":alpha, "validation_mae_24h":score})
            if score < best:
                best, best_alpha, chosen = score, alpha, model
    joblib.dump(chosen, output / "ridge.joblib")
    ridge_time = time.perf_counter() - start
    del x, xv
    run_info = {"config": config, "device": str(device), "platform": platform.platform(),
                "processor": platform.processor(), "torch": torch.__version__,
                "ridge_alpha": best_alpha, "ridge_selection_seconds": ridge_time}
    for cyclical, name in [(False, "GRU"), (True, "GRU_calendar")]:
        ds, _, _ = prepare(config, cyclical)
        for seed in config["seeds"]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = Forecaster(ds["train"].x.shape[1], config["hidden_size"], len(config["horizons"])).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            loader = DataLoader(ds["train"], batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            best, stale, best_state, best_epoch = float("inf"), 0, None, 0
            start = time.perf_counter()
            for epoch in range(1, config["epochs"]+1):
                model.train()
                total, count = 0.0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.l1_loss(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += loss.item()*len(x); count += len(x)
                pred = predict(model, ds["validation"], config["batch_size"], device)
                score = float(np.abs(pred[:,primary]-ds["validation"].targets()[:,primary]).mean())
                log.append({"model":name,"seed":seed,"epoch":epoch,
                            "train_mae_all_horizons":total/count*ds["train"].target_std,"validation_mae_24h":score})
                print(f"{name} seed={seed} epoch={epoch}: validation 24h MAE={score:.3f} C", flush=True)
                if score < best:
                    best, stale, best_state, best_epoch = score, 0, copy.deepcopy(model.state_dict()), epoch
                else:
                    stale += 1
                if stale >= config["patience"]:
                    break
            seconds = time.perf_counter()-start
            torch.save({"state_dict":best_state,"inputs":ds["train"].x.shape[1],"seed":seed}, output/f"{name}_{seed}.pt")
            selections.append({"model":name,"seed":seed,"validation_mae_24h":best,
                               "best_epoch":best_epoch,"train_seconds":seconds,
                               "parameters":sum(p.numel() for p in model.parameters())})
            pd.DataFrame(log).to_csv(ROOT/"results/training_history.csv",index=False)
    (ROOT/"results/run_info.json").write_text(json.dumps(run_info,indent=2))
    pd.DataFrame(selections).to_csv(ROOT/"results/validation_selection.csv",index=False)
    print("Selection complete. Test targets were not used for selection.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    run(parser.parse_args().config)
