# Jena Climate data

- Dataset: Jena Climate, 2009-2016 archive, 14 original weather variables at approximately ten-minute intervals.
- Original provider: Max Planck Institute for Biogeochemistry, Jena, Germany.
- Official source: https://www.bgc-jena.mpg.de/wetter/
- Terms: the institute's download page states CC BY 4.0: https://www.bgc-jena.mpg.de/wetter/weather_data.html
- Download mirror: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
- Mirror provenance: the TensorFlow time-series tutorial documents this archive, prepared by Francois Chollet for *Deep Learning with Python*: https://www.tensorflow.org/tutorials/structured_data/time_series
- Access/byte-level version: see `source_manifest.json`, including access timestamp and SHA-256 hashes. Source pages checked 29 October 2026.
- Attribution: Max Planck Institute for Biogeochemistry, weather station observations, Jena; 2009-2016 subset distributed through the TensorFlow/Keras dataset mirror. Cite both provider and mirror when sharing derived work.
- Obtain locally: `python scripts/get_data.py`. Raw files and checkpoints are excluded from Git.

## Exact processing protocol

Parse timestamps with `%d.%m.%Y %H:%M:%S`, sort, keep the first duplicate timestamp, and replace numeric -9999 sentinel readings with missing values. Split raw observations first: training [2009-01-01, 2015-01-01), validation [2015-01-01, 2016-01-01), test [2016-01-01, 2017-01-01). Data outside these intervals is excluded.

Replace wind direction and speed with wind x/y components before hourly averaging; retain maximum wind speed. Each hourly bin is right-closed and labelled by its ending time, so it contains no future readings relative to that label. Bins outside their period are discarded. Forward-fill within each split only, with training-period feature medians for initial missing values. Never back-fill from future observations. Fit means and population standard deviations using the training period only.

Create windows independently inside each period: 120 hourly inputs ending at time t, with targets at t+1, t+6, t+12 and t+24 hours. The primary score is t+24. Auxiliary horizons support the required horizon-error comparison and are identical for both GRUs and the linear model. Exclude windows with missing original hourly target observations or a missing current-temperature reference. Other missing inputs may be imputed. Stride 3 for training, stride 1 for validation/test, identical across learned models. No window borrows context from another split. This intentionally omits the first five days of validation and test forecasts.

Calendar improvement: append sine/cosine of hour and within-year position to the input window, including leap-year length. These four features are known from timestamps and require no fitting. Both models use identical training objectives, hidden width, optimizer, seeds and selection rules. The input layer consequently has a few more parameters, reported separately.

Seeds: 17, 42, 147. No class labels or class balance apply to this regression problem. Risks: overlapping windows are correlated, the station represents one site, timestamps have no explicit timezone, and historical observations may not represent present-day climate. Seed standard deviations measure training variability, not independent weather-sample uncertainty.
