"""Demonstrate a saved model at an eligible historical test forecast origin."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from .data import ROOT, prepare
from .models import Forecaster

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["GRU","GRU_calendar"],default="GRU_calendar")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--origin",help="Historical 2016 origin, e.g. 2016-07-01T12:00:00; defaults to first eligible origin")
    args=parser.parse_args()
    config=json.loads((ROOT/"checkpoints/config.json").read_text())
    torch.set_num_threads(config["threads"])
    ds,_,_=prepare(config,args.model=="GRU_calendar")
    test=ds["test"]
    valid_times=test.index[test.origins]
    if args.origin:
        locations=np.flatnonzero(valid_times==pd.Timestamp(args.origin))
        if not len(locations):
            parser.error("Origin is outside eligible test windows or has missing target/reference readings.")
        i=int(locations[0])
    else:
        i=0
    saved=torch.load(ROOT/f"checkpoints/{args.model}_{args.seed}.pt",weights_only=True)
    model=Forecaster(saved["inputs"],config["hidden_size"],len(config["horizons"]))
    model.load_state_dict(saved["state_dict"]);model.eval()
    x,_=test[i]
    with torch.no_grad():
        prediction=model(torch.from_numpy(x[None])).numpy()[0]*test.target_std+test.target_mean
    origin=valid_times[i]
    print(f"Historical demonstration: {args.model}, seed {args.seed}, origin {origin}")
    print(pd.DataFrame({"horizon_hours":config["horizons"],"target_time":[origin+pd.Timedelta(hours=h) for h in config["horizons"]],
                        "prediction_C":prediction,"observed_C":test.targets()[i]}).to_string(index=False))

if __name__=="__main__":
    main()
