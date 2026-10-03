"""Hyper-parameter search spaces, shared by every dataset.

Each tuned model gets the same Optuna budget (config.TUNING_TRIALS) and its
default configuration is always evaluated first, so tuning can only improve on
the defaults as judged on the validation window.

    python -m src.spaces     # print the spaces as a Markdown table
"""
from src.models import boosting, prophet_model, recurrent

NEURAL_DEFAULTS = {"input_size": 24, "learning_rate": 1e-3, "max_steps": 1000,
                   "batch_size": 32, "scaler_type": "robust"}


def _neural_common(t):
    return {
        "input_size": t.suggest_categorical("input_size", [12, 24]),
        "learning_rate": t.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "max_steps": t.suggest_categorical("max_steps", [300, 500, 1000, 2000]),
        "batch_size": t.suggest_categorical("batch_size", [16, 32, 64]),
        "scaler_type": t.suggest_categorical("scaler_type", ["robust", "standard"]),
    }


def _recurrent(t):
    return {
        "window": t.suggest_categorical("window", [12, 18, 24]),
        "units_1": t.suggest_categorical("units_1", [32, 64, 128]),
        "units_2": t.suggest_categorical("units_2", [16, 32, 64]),
        "dropout": t.suggest_float("dropout", 0.0, 0.4),
        "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": t.suggest_categorical("batch_size", [32, 64, 128]),
    }


SPACES = {
    "prophet": lambda t: {
        "changepoint_prior_scale": t.suggest_float("changepoint_prior_scale", 1e-3, 0.5, log=True),
        "seasonality_prior_scale": t.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
        "seasonality_mode": t.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        "changepoint_range": t.suggest_float("changepoint_range", 0.8, 0.95),
    },
    "lgbm": lambda t: {
        "n_lags": t.suggest_categorical("n_lags", [12, 24]),
        "num_leaves": t.suggest_int("num_leaves", 8, 64, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": t.suggest_int("n_estimators", 100, 1500, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 50),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "gru": _recurrent,
    "lstm": _recurrent,
    "bilstm": _recurrent,
    "nbeats": lambda t: {
        **_neural_common(t),
        "n_blocks": t.suggest_categorical("n_blocks", [1, 2, 3]),
        "width": t.suggest_categorical("width", [128, 256, 512]),
        "dropout_prob_theta": t.suggest_float("dropout_prob_theta", 0.0, 0.3),
    },
    "nhits": lambda t: {
        **_neural_common(t),
        "pooling": t.suggest_categorical("pooling", ["2-2-1", "4-2-1", "8-4-1"]),
        "downsample": t.suggest_categorical("downsample", ["4-2-1", "12-4-1", "1-1-1"]),
        "width": t.suggest_categorical("width", [128, 256, 512]),
        "dropout_prob_theta": t.suggest_float("dropout_prob_theta", 0.0, 0.3),
    },
    "patchtst": lambda t: {
        **_neural_common(t),
        "patch_len": t.suggest_categorical("patch_len", [4, 6, 8, 12]),
        "overlap": t.suggest_categorical("overlap", [True, False]),
        "hidden_size": t.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "n_heads": t.suggest_categorical("n_heads", [4, 8, 16]),
        "encoder_layers": t.suggest_int("encoder_layers", 1, 3),
        "dropout": t.suggest_float("dropout", 0.0, 0.3),
    },
    "dlinear": lambda t: {
        **_neural_common(t),
        "moving_avg_window": t.suggest_categorical("moving_avg_window", [5, 13, 25]),
    },
    "tide": lambda t: {
        **_neural_common(t),
        "hidden_size": t.suggest_categorical("hidden_size", [64, 128, 256, 512]),
        "decoder_output_dim": t.suggest_categorical("decoder_output_dim", [4, 8, 16, 32]),
        "temporal_decoder_dim": t.suggest_categorical("temporal_decoder_dim", [16, 32, 64, 128]),
        "num_encoder_layers": t.suggest_int("num_encoder_layers", 1, 3),
        "num_decoder_layers": t.suggest_int("num_decoder_layers", 1, 3),
        "dropout": t.suggest_float("dropout", 0.0, 0.5),
    },
}
# The ONI variants are tuned with the same space as their base model.
SPACES["lgbm_oni"] = SPACES["lgbm"]
SPACES["lstm_oni"] = SPACES["lstm"]

DEFAULTS = {
    "prophet": prophet_model.DEFAULTS,
    "lgbm": boosting.DEFAULTS,
    "lgbm_oni": boosting.DEFAULTS,
    **recurrent.DEFAULTS,
    "nbeats": {**NEURAL_DEFAULTS, "n_blocks": 1, "width": 512, "dropout_prob_theta": 0.0},
    "nhits": {**NEURAL_DEFAULTS, "pooling": "2-2-1", "downsample": "4-2-1", "width": 512,
              "dropout_prob_theta": 0.0},
    "patchtst": {**NEURAL_DEFAULTS, "patch_len": 8, "overlap": False, "hidden_size": 128,
                 "n_heads": 16, "encoder_layers": 3, "dropout": 0.2},
    "dlinear": {**NEURAL_DEFAULTS, "moving_avg_window": 25},
    "tide": {**NEURAL_DEFAULTS, "hidden_size": 512, "decoder_output_dim": 32,
             "temporal_decoder_dim": 128, "num_encoder_layers": 1,
             "num_decoder_layers": 1, "dropout": 0.3},
}


def markdown() -> str:
    """Search space of every tuned model, read off a recording trial."""
    class Recorder:
        def __init__(self):
            self.rows = []

        def suggest_categorical(self, name, choices):
            self.rows.append((name, ", ".join(map(str, choices))))
            return choices[0]

        def suggest_float(self, name, low, high, log=False):
            self.rows.append((name, f"[{low}, {high}]" + (" log" if log else "")))
            return low

        def suggest_int(self, name, low, high, log=False):
            self.rows.append((name, f"{low}..{high}" + (" log" if log else "")))
            return low

    lines = ["| model | parameter | range | default |", "|---|---|---|---|"]
    for model, space in SPACES.items():
        rec = Recorder()
        space(rec)
        for name, rng in rec.rows:
            lines.append(f"| {model} | {name} | {rng} | {DEFAULTS[model].get(name, '')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown())
