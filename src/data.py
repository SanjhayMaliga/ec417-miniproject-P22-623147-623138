"""Chronological preprocessing; no window or imputation crosses a split."""
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
TARGET = "T (degC)"

def load_hourly(path=None):
    raw = pd.read_csv(path or ROOT / "data/raw/jena_climate_2009_2016.csv")
    dates = pd.to_datetime(raw.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    raw.index = dates
    raw = raw.sort_index()
    audit = {"raw_rows": len(raw), "duplicate_timestamps": int(raw.index.duplicated().sum()),
             "sentinel_counts": (raw == -9999).sum().astype(int).to_dict()}
    raw = raw[~raw.index.duplicated(keep="first")].replace(-9999, np.nan)
    # Split raw observations first. Hour labels are the END of their averaging bin.
    periods = {"train": ("2009-01-01", "2015-01-01"),
               "validation": ("2015-01-01", "2016-01-01"),
               "test": ("2016-01-01", "2017-01-01")}
    parts, observed = {}, {}
    for name, (start, end) in periods.items():
        part = raw.loc[(raw.index >= start) & (raw.index < end)].copy()
        # Circular wind direction becomes Cartesian components before averaging.
        angle = np.deg2rad(part.pop("wd (deg)"))
        speed = part.pop("wv (m/s)")
        part["wind_x"] = speed * np.cos(angle)
        part["wind_y"] = speed * np.sin(angle)
        hourly = part.resample("1h", label="right", closed="right").mean()
        hourly = hourly.loc[(hourly.index >= start) & (hourly.index < end)]
        observed[name] = hourly[TARGET].notna().to_numpy()
        parts[name] = hourly
    # Medians and normalization are estimated exclusively from the training period.
    medians = parts["train"].median()
    audit["missing_hourly_cells_before_fill"] = {k: int(v.isna().sum().sum()) for k,v in parts.items()}
    parts = {k: v.ffill().fillna(medians) for k,v in parts.items()}
    if any(not np.isfinite(v.to_numpy()).all() for v in parts.values()):
        raise ValueError("Nonfinite data remains after training-only imputation")
    mean, std = parts["train"].mean(), parts["train"].std(ddof=0).replace(0, 1)
    audit["splits"] = {k: {"start": str(v.index[0]), "end": str(v.index[-1]), "hours": len(v)} for k,v in parts.items()}
    audit["features"] = list(mean.index)
    audit["mean"] = mean.to_dict()
    audit["std"] = std.to_dict()
    audit["medians"] = medians.to_dict()
    return parts, observed, mean, std, audit

def calendar_features(index):
    daily = 2 * np.pi * index.hour.to_numpy() / 24
    days_in_year = np.where(index.is_leap_year, 366, 365)
    annual = 2 * np.pi * (index.day_of_year.to_numpy() - 1 + index.hour.to_numpy()/24) / days_in_year
    return np.column_stack([np.sin(daily), np.cos(daily), np.sin(annual), np.cos(annual)]).astype("float32")

class Windows(Dataset):
    def __init__(self, frame, observed, mean, std, lookback=120, horizons=(1,6,12,24), stride=1, cyclical=False):
        self.x = ((frame - mean) / std).to_numpy(dtype="float32")
        if cyclical:
            self.x = np.column_stack([self.x, calendar_features(frame.index)])
        self.temperature = frame[TARGET].to_numpy(dtype="float32")
        self.target_mean, self.target_std = float(mean[TARGET]), float(std[TARGET])
        self.index = frame.index
        self.lookback, self.horizons = lookback, np.asarray(horizons)
        origins = np.arange(lookback-1, len(frame)-max(horizons), stride)
        # Never score an imputed target or use an imputed persistence reference.
        valid = observed[origins] & observed[origins[:, None] + self.horizons].all(axis=1)
        self.origins = origins[valid]
        if not len(self.origins):
            raise ValueError("Period is too short for the requested lookback and horizons")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        t = self.origins[i]
        y = (self.temperature[t+self.horizons] - self.target_mean) / self.target_std
        return self.x[t-self.lookback+1:t+1], y.astype("float32")

    def targets(self):
        return self.temperature[self.origins[:, None] + self.horizons]

    def flattened(self):
        return np.stack([self[i][0].reshape(-1) for i in range(len(self))])

def prepare(config, cyclical=False):
    parts, observed, mean, std, audit = load_hourly()
    datasets = {k: Windows(v, observed[k], mean, std, config["lookback"], config["horizons"],
                           config["train_stride"] if k == "train" else 1, cyclical)
                for k,v in parts.items()}
    audit["windows"] = {k: len(v) for k,v in datasets.items()}
    return datasets, parts, audit
