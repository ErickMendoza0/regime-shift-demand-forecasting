"""Regime-adaptive switching (RAS) ensemble, computed post-hoc from the stored
predictions.

Within each test year the default forecaster (LSTM) is monitored month by
month. When its trailing SWITCH_WINDOW-month MAPE exceeds SWITCH_FACTOR times
its earlier trailing MAPE, the remaining months of the horizon are served by
the seasonal statistical fallback (SARIMA). The decision at month t uses only
information observable up to t, so this mimics an operational deployment.

    python -m src.ensemble
"""
import numpy as np
import pandas as pd

import config as C
from src import common


def switching_series(df_def: pd.DataFrame, df_fb: pd.DataFrame) -> pd.Series:
    d = df_def.sort_values("date").reset_index(drop=True)
    f = df_fb.sort_values("date").reset_index(drop=True)
    ape = (np.abs(d.y_true - d.y_pred) / d.y_true).values
    pred, switched = [], False
    for i in range(len(d)):
        pred.append(f.y_pred[i] if switched else d.y_pred[i])
        if not switched and i + 1 >= C.SWITCH_WINDOW:
            recent = np.mean(ape[i + 1 - C.SWITCH_WINDOW: i + 1])
            base = np.mean(ape[: max(i + 1 - C.SWITCH_WINDOW, 1)]) if i + 1 > C.SWITCH_WINDOW else np.inf
            if recent > C.SWITCH_FACTOR * base:
                switched = True
    return pd.Series(pred, index=d.date)


def main() -> None:
    allp = common.load_all_predictions()
    for fold in C.FOLD_YEARS:
        d = allp[(allp.model == C.SWITCH_DEFAULT) & (allp.fold == fold)]
        f = allp[(allp.model == C.SWITCH_FALLBACK) & (allp.fold == fold)]
        if d.empty or f.empty:
            print(f"skipping fold {fold}: missing base predictions")
            continue
        y_pred = switching_series(d, f)
        y_true = d.sort_values("date").set_index("date").y_true
        common.save_predictions("ras_switch", fold, y_true, y_pred)


if __name__ == "__main__":
    main()
