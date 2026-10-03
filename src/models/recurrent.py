"""GRU, LSTM and BiLSTM in PyTorch, forecasting recursively one month at a time.

The default configuration is the thesis architecture: two stacked recurrent
layers (128 and 64 units, dropout 0.2) over a 12-month window (18 for the
BiLSTM), a 4-dimensional embedding of the consumer group fused into every time
step, and a small dense head. Tuning starts from it and searches around it.

Series are min-max scaled with their own training history. The last 12 months
of history are held out for early stopping.
"""
import numpy as np
import pandas as pd

from src import oni as O
from src.models import frame

DEFAULTS = {
    "lstm": {"window": 12, "units_1": 128, "units_2": 64, "dropout": 0.2,
             "lr": 1e-3, "batch_size": 64},
    "gru": {"window": 12, "units_1": 128, "units_2": 64, "dropout": 0.2,
            "lr": 1e-3, "batch_size": 64},
    "bilstm": {"window": 18, "units_1": 128, "units_2": 64, "dropout": 0.2,
               "lr": 1e-3, "batch_size": 64},
}
DEFAULTS["lstm_oni"] = DEFAULTS["lstm"]

MAX_EPOCHS = 200
PATIENCE = 20
EMBED = 4


def _net(cell, n_in, n_cat, units_1, units_2, dropout, bidirectional):
    import torch
    from torch import nn

    rnn = nn.GRU if cell == "gru" else nn.LSTM
    k = 2 if bidirectional else 1

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(n_cat, EMBED)
            self.rnn_1 = rnn(n_in + EMBED, units_1, batch_first=True, bidirectional=bidirectional)
            self.drop = nn.Dropout(dropout)
            self.rnn_2 = rnn(units_1 * k, units_2, batch_first=True, bidirectional=bidirectional)
            if bidirectional:
                self.head = nn.Sequential(nn.Linear(units_2 * k, 64), nn.LeakyReLU(),
                                          nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
            else:
                self.head = nn.Sequential(nn.Linear(units_2, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, cat):
            e = self.embed(cat).unsqueeze(1).expand(-1, x.size(1), -1)
            z, _ = self.rnn_1(torch.cat([x, e], dim=-1))
            z, _ = self.rnn_2(self.drop(z))
            return self.head(z[:, -1]).squeeze(-1)

    return Net()


def _inputs(static):
    """Category used by the embedding and the static channels of each series."""
    s = static.set_index("series_id")
    if "group_code" in s:
        cat = s["group_code"].astype(int)
        geo = s[["latitude", "longitude"]]
        geo = (geo - geo.mean()) / geo.std()
        return cat, geo
    return pd.Series(np.arange(len(s)), index=s.index), None


def _rows(scaled, dates, geo_row, oni_values):
    month = np.asarray([d.month for d in dates])
    cols = [scaled, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
    if geo_row is not None:
        cols += [np.full(len(dates), geo_row.iloc[0]), np.full(len(dates), geo_row.iloc[1])]
    if oni_values is not None:
        cols.append(oni_values)
    return np.column_stack(cols).astype("float32")


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    import torch

    p = {**DEFAULTS[name], **ctx.params}
    window = int(p["window"])
    use_oni = name == "lstm_oni"
    torch.manual_seed(ctx.seed)
    np.random.seed(ctx.seed)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cat, geo = _inputs(static)
    cutoff = ctx.origin - pd.DateOffset(months=12)
    X_tr, c_tr, y_tr, X_va, c_va, y_va = [], [], [], [], [], []
    state = {}
    for sid, g in hist.groupby("series_id"):
        g = g.sort_values("date")
        y = g["y"].to_numpy(float)
        lo, hi = y.min(), y.max()
        hi = hi if hi > lo else lo + 1.0
        scaled = (y - lo) / (hi - lo)
        dates = pd.DatetimeIndex(g["date"])
        oni_in = O.feature(ctx.oni, dates) if use_oni else None
        geo_row = geo.loc[sid] if geo is not None else None
        rows = _rows(scaled, dates, geo_row, oni_in)
        for i in range(window, len(rows)):
            into = (X_va, c_va, y_va) if dates[i] >= cutoff else (X_tr, c_tr, y_tr)
            into[0].append(rows[i - window:i])
            into[1].append(cat[sid])
            into[2].append(scaled[i])
        state[sid] = (rows, lo, hi, geo_row)

    net = _net("gru" if name == "gru" else "lstm", X_tr[0].shape[1], int(cat.max()) + 1,
               int(p["units_1"]), int(p["units_2"]), float(p["dropout"]),
               bidirectional=name == "bilstm").to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(p["lr"]))
    loss_fn = torch.nn.MSELoss()

    def tensors(X, c, y):
        return (torch.tensor(np.stack(X), device=device),
                torch.tensor(np.asarray(c), device=device),
                torch.tensor(np.asarray(y, dtype="float32"), device=device))

    Xt, ct, yt = tensors(X_tr, c_tr, y_tr)
    Xv, cv, yv = tensors(X_va, c_va, y_va)
    gen = torch.Generator(device="cpu").manual_seed(ctx.seed)
    best, best_state, waited = np.inf, None, 0
    bs = int(p["batch_size"])
    for _ in range(MAX_EPOCHS):
        net.train()
        order = torch.randperm(len(yt), generator=gen).to(device)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            opt.zero_grad()
            loss_fn(net(Xt[idx], ct[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(Xv, cv), yv))
        if val < best - 1e-7:
            best, waited = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()

    # Recursive forecast, all series advanced together one month at a time.
    oni_path = O.horizon(ctx.oni, ctx.origin, ctx.dates, ctx.tier) if use_oni else None
    ids = list(state)
    buffers = {sid: state[sid][0].copy() for sid in ids}
    cats = torch.tensor([cat[sid] for sid in ids], device=device)
    preds = {sid: [] for sid in ids}
    for step, d in enumerate(ctx.dates):
        x = torch.tensor(np.stack([buffers[sid][-window:] for sid in ids]), device=device)
        with torch.no_grad():
            yhat = net(x, cats).cpu().numpy()
        for sid, v in zip(ids, yhat):
            _, lo, hi, geo_row = state[sid]
            preds[sid].append(v * (hi - lo) + lo)
            new = _rows(np.array([v]), [d], geo_row,
                        np.array([oni_path[step]]) if use_oni else None)
            buffers[sid] = np.vstack([buffers[sid], new])
    return pd.concat([frame(sid, ctx.dates, preds[sid]) for sid in ids], ignore_index=True)
