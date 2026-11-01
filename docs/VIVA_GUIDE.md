# Five-minute demo and viva preparation

1. State the task: use the last 120 hourly observations to predict temperature 24 hours later.
2. Explain persistence: tomorrow's temperature equals the latest observed temperature.
3. Show the time split and explain why randomly shuffling before splitting leaks weather patterns.
4. Explain a ridge model: flatten the weather history, fit a linear mapping, penalize large coefficients; choose the penalty with 2015 data.
5. Explain a GRU: recurrent hidden state with gates that control how past information is retained and updated.
6. Explain the improvement: sine/cosine features preserve proximity at midnight and year boundaries.
7. Read the actual MAE table and say whether the proposed improvement helped. A negative result is valid.
8. Show the two-week plot and one large-error example. Separate observed errors from hypotheses about weather causes.
9. Explain three-seed mean and sample standard deviation. They describe initialization/training variability, not confidence across future climates.
10. State limitations, disclose AI assistance, and demonstrate the reproduction command.

## Questions to practise

- Why MAE? It measures typical absolute temperature error in degrees Celsius; RMSE additionally emphasizes large errors.
- Why 2015 validation and 2016 test? This mirrors prediction into future time. Test outcomes never choose epochs or ridge penalties.
- What is a window boundary? For an origin at row 119, the 120 inputs are rows 0-119 and the 24-hour target is row 143.
- Why right-labelled resampling? A bin labelled t includes observations up to t, so the input cannot see past its forecast origin.
- Why not interpolate across the whole timeline? That could use future measurements or bridge the train/test boundary.
- Why four outputs? They enable the required horizon plot. Both GRUs and ridge share these targets, so the isolated improvement is calendar inputs.
- Why do deterministic baselines have zero seed spread? They do not depend on random initialization. That does not imply zero sampling uncertainty.
- Does the method apply to another city? That has not been tested.
