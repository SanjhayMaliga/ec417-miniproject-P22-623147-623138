"""One final evaluation after all validation-based model selection is complete."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ROOT, TARGET, prepare
from .models import Forecaster, persistence
from .train import predict

COLORS = {"Persistence":"#8795a6", "Ridge":"#2879c0", "GRU":"#e49428", "GRU_calendar":"#159b85"}

def savefig(name):
    plt.tight_layout()
    plt.savefig(ROOT/f"results/figures/{name}.png", dpi=160, bbox_inches="tight")
    plt.close()

def run():
    config = json.loads((ROOT/"checkpoints/config.json").read_text())
    info = json.loads((ROOT/"results/run_info.json").read_text())
    torch.set_num_threads(config["threads"])
    ds, parts, audit = prepare(config)
    test = ds["test"]
    truth = test.targets()
    primary = config["horizons"].index(24)
    selections = pd.read_csv(ROOT/"results/validation_selection.csv")
    predictions = {"Persistence": [persistence(test)]}
    predictions["Ridge"] = [joblib.load(ROOT/"checkpoints/ridge.joblib").predict(test.flattened())]
    rows = []
    for calendar, name in [(False,"GRU"),(True,"GRU_calendar")]:
        current, _, _ = prepare(config, calendar)
        predictions[name] = []
        for seed in config["seeds"]:
            checkpoint = torch.load(ROOT/f"checkpoints/{name}_{seed}.pt", weights_only=True)
            model = Forecaster(checkpoint["inputs"],config["hidden_size"],len(config["horizons"]))
            model.load_state_dict(checkpoint["state_dict"])
            predictions[name].append(predict(model,current["test"],config["batch_size"],torch.device("cpu")))
    summary = []
    for name, preds in predictions.items():
        # Deterministic baselines are evaluated once; repeated rows are not independent runs.
        for i, pred in enumerate(preds):
            seed = config["seeds"][i] if len(preds)>1 else "deterministic"
            for j, h in enumerate(config["horizons"]):
                error = pred[:,j]-truth[:,j]
                for metric, value in [("MAE",np.abs(error).mean()),("RMSE",np.sqrt(np.mean(error**2)))]:
                    rows.append({"model":name,"seed":seed,"horizon_hours":h,"metric":metric,"value":float(value)})
        maes = [np.abs(p[:,primary]-truth[:,primary]).mean() for p in preds]
        rmses = [np.sqrt(np.mean((p[:,primary]-truth[:,primary])**2)) for p in preds]
        times = selections.loc[selections.model == name,"train_seconds"].dropna().tolist()
        duration = np.mean(times) if times else (info["ridge_selection_seconds"] if name=="Ridge" else 0)
        summary.append({"model":name,"mae_mean":np.mean(maes),"mae_std":np.std(maes,ddof=1) if len(maes)>1 else 0,
                        "rmse_mean":np.mean(rmses),"rmse_std":np.std(rmses,ddof=1) if len(rmses)>1 else 0,
                        "train_seconds_mean":duration,"runs":len(preds)})
    pd.DataFrame(rows).to_csv(ROOT/"results/metrics.csv",index=False)
    pd.DataFrame(summary).to_csv(ROOT/"results/summary.csv",index=False)
    (ROOT/"results/figures").mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10,"figure.figsize":(9,4)})
    plt.figure()
    for name, preds in predictions.items():
        errors = np.array([np.abs(p-truth).mean(axis=0) for p in preds])
        spread = errors.std(axis=0,ddof=1) if len(preds)>1 else np.zeros(errors.shape[1])
        plt.errorbar(config["horizons"],errors.mean(axis=0),yerr=spread,label=name,color=COLORS[name],marker="o",capsize=3)
    plt.xlabel("Forecast horizon (hours)"); plt.ylabel("Test MAE (degrees C)")
    plt.title("Forecast error by horizon | mean +/- seed standard deviation"); plt.legend(ncol=2)
    savefig("horizon_mae")
    # Predetermined first two complete test weeks; seed 17, not cherry-picked.
    timestamps = test.index[test.origins+24]
    mask = timestamps < timestamps[0]+pd.Timedelta(days=14)
    plt.figure(figsize=(11,4))
    plt.plot(timestamps[mask],truth[mask,primary],color="#172b4d",label="Observed",linewidth=2)
    for name, preds in predictions.items():
        plt.plot(timestamps[mask],preds[0][mask,primary],label=name,color=COLORS[name],alpha=.85,linewidth=1)
    plt.ylabel("Temperature (degrees C)"); plt.xlabel("Target time in 2016")
    plt.title("24-hour forecasts | first two eligible test weeks | seed 17"); plt.legend(ncol=3,fontsize=8)
    savefig("two_test_weeks")
    plt.figure()
    for name, frame in parts.items():
        weekly = frame[TARGET].resample("7D").mean()
        plt.plot(weekly.index,weekly,label=name)
    plt.ylabel("Weekly mean temperature (degrees C)"); plt.title("Chronological split"); plt.legend()
    savefig("temperature_split")
    plt.figure(figsize=(9,4))
    hist = pd.read_csv(ROOT/"results/training_history.csv")
    for (name,seed), group in hist.groupby(["model","seed"]):
        plt.plot(group.epoch, group.validation_mae_24h,label=f"{name} / {seed}",alpha=.8)
    plt.xlabel("Epoch"); plt.ylabel("Validation 24h MAE (degrees C)"); plt.legend(fontsize=8,ncol=2)
    plt.title("Checkpoint selection uses validation data only"); savefig("learning_curves")
    plt.figure()
    for name in ["GRU","GRU_calendar"]:
        group=hist[(hist.model==name)&(hist.seed==17)]
        plt.plot(group.epoch,group.train_mae_all_horizons,label=f"{name}: train, all horizons")
        plt.plot(group.epoch,group.validation_mae_24h,linestyle="--",label=f"{name}: validation, 24h")
    plt.xlabel("Epoch"); plt.ylabel("MAE (degrees C)"); plt.legend(fontsize=8)
    plt.title("Training objective and primary validation metric (different horizons)"); savefig("training_curves")
    monthly=[]
    for name,preds in predictions.items():
        err=np.mean([np.abs(p[:,primary]-truth[:,primary]) for p in preds],axis=0)
        for month in range(1,13):
            selected=timestamps.month==month
            monthly.append({"model":name,"month":month,"n":int(selected.sum()),"mae":float(err[selected].mean())})
    pd.DataFrame(monthly).to_csv(ROOT/"results/monthly_errors.csv",index=False)
    plt.figure()
    for name,g in pd.DataFrame(monthly).groupby("model",sort=False):
        plt.plot(g.month,g.mae,label=name,color=COLORS[name],marker=".")
    plt.xlabel("Test month (2016)"); plt.ylabel("24h MAE (degrees C)"); plt.legend(ncol=2)
    plt.title("Seasonal error analysis"); savefig("monthly_errors")
    detail = pd.DataFrame({"target_time":timestamps,"actual":truth[:,primary]})
    for name,preds in predictions.items():
        detail[name]=preds[0][:,primary]
    detail["absolute_error_calendar"]=np.abs(detail.GRU_calendar-detail.actual)
    detail.nlargest(10,"absolute_error_calendar").to_csv(ROOT/"results/largest_errors.csv",index=False)
    np.savez_compressed(ROOT/"results/predictions.npz", truth=truth, **{k:np.array(v) for k,v in predictions.items()})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__ == "__main__":
    run()
