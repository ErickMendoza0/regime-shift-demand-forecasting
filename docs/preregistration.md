# Analysis plan, fixed before the forecasting runs

Written on 3 October 2026, before the forecasting campaign of the
monthly-origin protocol was run. While the code was being written, each model
was run from a single origin per dataset with default settings to check that it
worked and how long it took, and the change-point analysis of the Ecuadorian
series was run once; those smoke runs were discarded. Anything below that
changes later is recorded at the end with its date and reason.

## Question

Does a more elaborate forecaster hold up better than a simple one when the
series it forecasts goes through a real regime shift, and how much of its
in-regime accuracy survives the shift?

## Data

- Ecuador: monthly billed electricity consumption (ARCONEL), 116 province by
  consumer-group series summed to the national total, January 2014 to December
  2024. In late 2024 this is served consumption under scheduled load shedding,
  not unconstrained demand.
- Brazil: monthly electricity consumption of the five regions (Eletrobras/EPE via
  Ipeadata), from January 1990, summed to the national total. Evaluated around
  the 2001 rationing (origins 1999-01 to 2003-12) and around COVID-19 (origins
  2018-01 to 2022-12).
- Europe: monthly electricity available to the internal market for the 27 EU
  countries (Eurostat `nrg_cb_em`), from January 2008; each country is its own
  target. Origins from 2018-01 to the last common month.
- ONI (NOAA CPC), Ecuador only.

## Protocol

- Monthly forecast origins, expanding training window, horizons 1 to 12.
- Everything a model sees is cut at the origin: series admissibility, gap
  filling, scaling and the ONI values (published with a two-month lag).
- ONI tiers: x1 persists the last published value, x2 uses an ARIMA forecast of
  ONI made at the origin, x3 uses the realised values. x3 is a hindsight bound
  and never enters a ranking, a test or a combination.
- Tuned models get 50 Optuna trials each, scored by MASE on two validation
  origins 12 and 6 months before the tuning origin. Ecuador re-tunes every
  January; Brazil and Europe tune once at the start of each block of origins.
  The default configuration is always the first trial.
- Seeded models run with 5 seeds (Ecuador) or 3 (Brazil, Europe); the median of
  the seeds' forecasts represents the model.

## Regimes

Year-on-year growth of each target is segmented in mean (dynamic programming,
number of breaks by BIC). A segment with mean growth at or below -2 % is a
shift. A segment starting within 13 months of the start of a shift, with mean
above the median segment mean, is a rebound. Everything else is stable. PELT
and BOCPD are reported as cross-checks; the segmentation decides.

For the Ecuadorian drought, results are also reported for four fixed windows:
all of 2024; the detected break to December 2024; September to December 2024;
October 2023 to December 2024.

## Measures

MASE (scale: in-sample seasonal-naive MAE of the target before the origin) is
the primary measure; MAPE, RMSE, MAE and mean error are also reported. Losses
are pooled per target month over all forecasts of that month.

## Hypotheses

- H1. No model has the lowest error in every regime.
- H2. The ratio of shift to stable MASE is larger for learned models (boosting,
  recurrent, deep, foundation) than for the simple and statistical references.
- H3. Stable-regime accuracy does not predict shift-regime accuracy: the
  Spearman correlation across models is not significantly positive.
- H4. A deployable ONI tier (x1 or x2) improves on the same model without ONI.
- H5. At least one combination or switching rule has lower shift-regime MASE
  than the best single deployable model chosen on the selection window, without
  being worse in the stable regime.

## Tests

- Model confidence set at 90 % (stationary bootstrap, block 6) per regime.
- All pairwise Diebold-Mariano tests on target-month losses with a Newey-West
  variance (lag 11) and the HLN correction, Holm-adjusted within each regime;
  Benjamini-Yekutieli reported alongside.
- Giacomini-White test of each model against seasonal naive with a shift dummy
  as instrument (Holm-adjusted), and Giacomini-Rossi fluctuation paths.
- Across independent shift episodes (target by episode): Friedman with Nemenyi
  critical difference, and the Bayesian signed-rank test with a 1 % ROPE on
  relative MASE.
- Significance level 0.05 throughout. Every result is reported whether it
  supports a hypothesis or not.

## Changes after this date

None yet.
