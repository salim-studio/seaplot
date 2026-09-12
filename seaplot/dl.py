"""Deep learning lite: NumPy MLP + optional PyTorch helper.

No heavy dependency required::

    import seaplot as sp
    mlp = sp.MLP(hidden=(64, 32), task="classification", epochs=50)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)

If torch is installed, ``sp.torch_mlp(...)`` builds a torch nn.Module
with the same hyper-params.
"""
from __future__ import annotations

import numpy as np


def _relu(z):
    return np.maximum(z, 0)


def _relu_d(z):
    return (z > 0).astype(float)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """Small fully-connected net in pure NumPy (Adam, early-stopping-lite)."""

    def __init__(self, hidden=(64, 32), task="classification", epochs=100,
                 lr=1e-3, batch=256, seed=42, verbose=False, val_size=0.15):
        self.hidden = tuple(hidden)
        self.task = task
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch = int(batch)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.val_size = float(val_size)
        self.params_ = None
        self.history_ = {"loss": [], "val_loss": []}

    # -- init -------------------------------------------------------
    def _init(self, d_in, n_out):
        rng = np.random.default_rng(self.seed)
        sizes = [d_in, *self.hidden, n_out]
        P = {}
        for i in range(len(sizes) - 1):
            P[f"W{i}"] = rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
            P[f"b{i}"] = np.zeros(sizes[i + 1])
            P[f"mW{i}"] = np.zeros_like(P[f"W{i}"])
            P[f"vW{i}"] = np.zeros_like(P[f"W{i}"])
            P[f"mb{i}"] = np.zeros_like(P[f"b{i}"])
            P[f"vb{i}"] = np.zeros_like(P[f"b{i}"])
        return P

    def _forward(self, X, P):
        A, Zs = [X], []
        L = len(self.hidden) + 1
        for i in range(L):
            Z = A[-1] @ P[f"W{i}"] + P[f"b{i}"]
            Zs.append(Z)
            A.append(_relu(Z) if i < L - 1 else Z)
        return A, Zs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            classes = np.unique(y)
            self.classes_ = classes
            k = len(classes)
            cmap = {c: i for i, c in enumerate(classes)}
            Y = np.zeros((len(y), max(k, 2) if k > 2 else 1))
            idx = np.array([cmap[v] for v in y])
            if k == 2:
                Y[:, 0] = idx
            else:
                Y[np.arange(len(y)), idx] = 1
            n_out = 1 if k == 2 else k
        else:
            Y = y.astype(float).reshape(-1, 1)
            n_out = 1
        # fill nan median
        col_med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), col_med, X)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Xn = (X - mu) / sd
        self.mu_, self.sd_ = mu, sd
        # split val
        n = len(Xn)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        nv = int(n * self.val_size)
        vi, ti = perm[:nv], perm[nv:]
        Xv, Yv = Xn[vi], Y[vi]
        Xt, Yt = Xn[ti], Y[ti]
        P = self._init(X.shape[1], n_out)
        L = len(self.hidden) + 1
        t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(self.epochs):
            rng.shuffle(ti)
            for s in range(0, len(ti), self.batch):
                t += 1
                bi = ti[s: s + self.batch]
                Xb, Yb = Xn[bi], Y[bi]
                A, Zs = self._forward(Xb, P)
                logits = A[-1]
                if self.task == "classification":
                    if n_out == 1:
                        p = 1 / (1 + np.exp(-logits))
                        dlog = (p - Yb.reshape(-1, 1)) / len(Xb)
                    else:
                        p = _softmax(logits)
                        dlog = (p - Yb) / len(Xb)
                else:
                    dlog = (logits - Yb.reshape(-1, 1)) / len(Xb)
                dA = dlog
                for i in reversed(range(L)):
                    dW = A[i].T @ dA
                    db = dA.sum(0)
                    if i > 0:
                        dA = (dA @ P[f"W{i}"].T) * _relu_d(Zs[i - 1])
                    # adam update
                    for key, g in ((f"W{i}", dW), (f"b{i}", db)):
                        m = P["m" + key] if key.startswith("W") else P["m" + key]
                        v = P["v" + key] if key.startswith("W") else P["v" + key]
                        m[:] = b1 * m + (1 - b1) * g
                        v[:] = b2 * v + (1 - b2) * g * g
                        mh = m / (1 - b1 ** t)
                        vh = v / (1 - b2 ** t)
                        P[key] -= self.lr * mh / (np.sqrt(vh) + eps)
            # log losses
            tr_loss = float(self._loss(Xt, Yt, P, n_out))
            va_loss = float(self._loss(Xv, Yv, P, n_out)) if nv else float("nan")
            self.history_["loss"].append(tr_loss)
            self.history_["val_loss"].append(va_loss)
            if self.verbose and ep % max(1, self.epochs // 10) == 0:
                print(f"epoch {ep}: loss={tr_loss:.4f} val={va_loss:.4f}")
        self.params_ = P
        self.n_out_ = n_out
        return self

    def _loss(self, X, Y, P, n_out):
        A, _ = self._forward(X, P)
        logits = A[-1]
        if self.task == "classification":
            if n_out == 1:
                p = 1 / (1 + np.exp(-logits.reshape(-1)))
                p = np.clip(p, 1e-7, 1 - 1e-7)
                yb = Y.reshape(-1)
                return -(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean()
            p = np.clip(_softmax(logits), 1e-7, 1)
            return -(Y * np.log(p)).sum(1).mean()
        return ((logits.reshape(-1) - Y.reshape(-1)) ** 2).mean()

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        col_med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), col_med, X)
        Xn = (X - self.mu_) / self.sd_
        A, _ = self._forward(Xn, self.params_)
        logits = A[-1]
        if self.task == "classification":
            if self.n_out_ == 1:
                p = 1 / (1 + np.exp(-logits.reshape(-1)))
                return np.array([self.classes_[int(v > 0.5)] for v in p])
            return self.classes_[np.argmax(_softmax(logits), axis=1)]
        return logits.reshape(-1)

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        X = np.asarray(X, dtype=float)
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        Xn = (X - self.mu_) / self.sd_
        A, _ = self._forward(Xn, self.params_)
        if self.n_out_ == 1:
            p = 1 / (1 + np.exp(-A[-1].reshape(-1)))
            return np.column_stack([1 - p, p])
        return _softmax(A[-1])


def torch_mlp(input_dim, output_dim, hidden=(64, 32), task="classification"):
    """Build torch nn.Module mirror of MLP. Requires torch."""
    try:
        import torch.nn as nn
    except Exception as e:
        raise ImportError("torch_mlp needs torch: pip install torch") from e
    layers = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
