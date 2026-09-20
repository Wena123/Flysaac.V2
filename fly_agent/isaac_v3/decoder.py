from pathlib import Path

import numpy as np

from . import config


class ActionDecoder:
    """
    FlyIsaac V3.4 temporal decoder.

    Shared temporal DN context -> hidden trunk, then two mutually-exclusive softmax heads:
      movement: none + four cardinals + four diagonals (9 classes)
      shooting: none + four cardinals (5 classes)

    The MaleCNS connectome remains frozen.
    """

    VERSION = 34

    def __init__(
        self,
        full_input_dim,
        hidden_dim=config.DECODER_HIDDEN,
        feature_count=config.DECODER_FEATURES,
        seed=config.DECODER_SEED,
        dn_dim=None,
        context_frames=config.TEMPORAL_CONTEXT_FRAMES,
        sample_hz=config.TEMPORAL_SAMPLE_HZ,
        lag_samples=0,
    ):
        self.full_input_dim = int(full_input_dim)
        self.context_frames = max(1, int(context_frames))
        self.dn_dim = int(dn_dim) if dn_dim is not None else int(self.full_input_dim // self.context_frames)
        self.sample_hz = float(sample_hz)
        self.lag_samples = int(lag_samples)
        if self.dn_dim * self.context_frames != self.full_input_dim:
            raise ValueError(
                f"full_input_dim={self.full_input_dim} must equal dn_dim({self.dn_dim}) * "
                f"context_frames({self.context_frames})"
            )
        self.hidden_dim = int(hidden_dim)
        self.feature_count = min(int(feature_count), self.full_input_dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.feature_indices = np.arange(self.feature_count, dtype=np.int64)
        self.mean = np.zeros(self.feature_count, dtype=np.float32)
        self.std = np.ones(self.feature_count, dtype=np.float32)
        self._init_weights(self.feature_count)

    def _init_weights(self, input_dim):
        scale1 = np.sqrt(2.0 / max(1, input_dim))
        scale2 = np.sqrt(2.0 / max(1, self.hidden_dim))
        self.w1 = (self.rng.standard_normal((input_dim, self.hidden_dim)) * scale1).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        self.w_move = (
            self.rng.standard_normal((self.hidden_dim, len(config.MOVEMENT_CLASSES))) * scale2
        ).astype(np.float32)
        self.b_move = np.zeros(len(config.MOVEMENT_CLASSES), dtype=np.float32)

        self.w_shoot = (
            self.rng.standard_normal((self.hidden_dim, len(config.SHOOT_CLASSES))) * scale2
        ).astype(np.float32)
        self.b_shoot = np.zeros(len(config.SHOOT_CLASSES), dtype=np.float32)

    @staticmethod
    def _softmax(logits):
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)

    def fit_preprocessor(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.full_input_dim:
            raise ValueError(f"Expected X shape (*, {self.full_input_dim}), got {x.shape}")

        variance = np.var(x, axis=0)
        k = min(self.feature_count, x.shape[1])
        if k < x.shape[1]:
            idx = np.argpartition(variance, -k)[-k:]
            idx = idx[np.argsort(variance[idx])[::-1]]
        else:
            idx = np.arange(x.shape[1])

        self.feature_indices = idx.astype(np.int64)
        selected = x[:, self.feature_indices]
        self.mean = selected.mean(axis=0).astype(np.float32)
        self.std = selected.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-5] = 1.0
        self._init_weights(len(self.feature_indices))

    def _prepare(self, x):
        x = np.asarray(x, dtype=np.float32)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.full_input_dim:
            raise ValueError(f"Decoder expects {self.full_input_dim} DN values, got {x.shape}")
        z = x[:, self.feature_indices]
        z = (z - self.mean) / self.std
        return z, one

    def _forward_prepared(self, z):
        h = np.tanh(z @ self.w1 + self.b1)
        move = self._softmax(h @ self.w_move + self.b_move)
        shoot = self._softmax(h @ self.w_shoot + self.b_shoot)
        return h, move, shoot

    def predict_proba(self, x):
        z, one = self._prepare(x)
        _, move, shoot = self._forward_prepared(z)
        if one:
            return move[0], shoot[0]
        return move, shoot

    def predict_classes(self, x):
        move, shoot = self.predict_proba(x)
        if move.ndim != 1 or shoot.ndim != 1:
            raise ValueError("predict_classes expects one DN feature vector")
        return int(np.argmax(move)), int(np.argmax(shoot)), move, shoot

    def predict_actions(self, x, threshold=None):
        # threshold is accepted only for backwards call compatibility; V3.4
        # uses mutually-exclusive softmax classes instead of 8 thresholds.
        move_idx, shoot_idx, move_p, shoot_p = self.predict_classes(x)
        move_name = config.MOVEMENT_CLASSES[move_idx]
        shoot_name = config.SHOOT_CLASSES[shoot_idx]
        actions = set(config.MOVEMENT_ACTIONS[move_name])
        actions.update(config.SHOOT_ACTIONS[shoot_name])
        info = {
            "move_index": move_idx,
            "shoot_index": shoot_idx,
            "move_name": move_name,
            "shoot_name": shoot_name,
            "move_confidence": float(move_p[move_idx]),
            "shoot_confidence": float(shoot_p[shoot_idx]),
            "move_probs": move_p,
            "shoot_probs": shoot_p,
        }
        return actions, info

    @staticmethod
    def _class_weights(targets, n_classes, max_weight=3.0):
        targets = np.asarray(targets, dtype=np.int64)
        counts = np.bincount(targets, minlength=int(n_classes)).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        inv = np.sqrt(float(len(targets)) / counts)
        inv /= max(1e-6, float(inv.mean()))
        return np.clip(inv, 0.5, float(max_weight)).astype(np.float32)

    @staticmethod
    def _ce(targets, probs, class_weights=None, sample_weights=None):
        targets = np.asarray(targets, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float32)
        p = np.clip(probs[np.arange(len(targets)), targets], 1e-7, 1.0)
        w = np.ones(len(targets), dtype=np.float32)
        if class_weights is not None:
            w *= np.asarray(class_weights, dtype=np.float32)[targets]
        if sample_weights is not None:
            w *= np.asarray(sample_weights, dtype=np.float32)
        return float(np.sum(-np.log(p) * w) / max(1e-8, float(w.sum())))

    def _snapshot(self):
        return tuple(
            p.copy()
            for p in (
                self.w1, self.b1,
                self.w_move, self.b_move,
                self.w_shoot, self.b_shoot,
            )
        )

    def _restore(self, snap):
        (
            self.w1, self.b1,
            self.w_move, self.b_move,
            self.w_shoot, self.b_shoot,
        ) = tuple(np.asarray(p, dtype=np.float32).copy() for p in snap)

    def train(
        self,
        x,
        move_targets,
        shoot_targets,
        sample_weights=None,
        epochs=40,
        batch_size=256,
        learning_rate=1e-3,
        validation=None,
        fit_preprocessor=True,
        patience=config.EARLY_STOP_PATIENCE,
        min_delta=config.EARLY_STOP_MIN_DELTA,
        verbose=True,
    ):
        x = np.asarray(x, dtype=np.float32)
        move_targets = np.asarray(move_targets, dtype=np.int64)
        shoot_targets = np.asarray(shoot_targets, dtype=np.int64)
        if len(x) != len(move_targets) or len(x) != len(shoot_targets):
            raise ValueError("X / movement / shooting sample counts differ")
        if len(x) < 2:
            raise ValueError("Need at least two samples")

        if sample_weights is None:
            sample_weights = np.ones(len(x), dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32)
            if len(sample_weights) != len(x):
                raise ValueError("sample_weights length mismatch")

        if fit_preprocessor:
            self.fit_preprocessor(x)
        z, _ = self._prepare(x)

        move_cw = self._class_weights(move_targets, len(config.MOVEMENT_CLASSES))
        shoot_cw = self._class_weights(shoot_targets, len(config.SHOOT_CLASSES))

        params = [
            self.w1, self.b1,
            self.w_move, self.b_move,
            self.w_shoot, self.b_shoot,
        ]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        step = 0
        history = []
        best_val = np.inf
        best_epoch = 0
        best_snapshot = None
        bad_epochs = 0

        for epoch in range(1, int(epochs) + 1):
            order = self.rng.permutation(len(z))
            batch_losses = []

            for start in range(0, len(z), int(batch_size)):
                idx = order[start:start + int(batch_size)]
                xb = z[idx]
                mb = move_targets[idx]
                sb = shoot_targets[idx]
                sw = sample_weights[idx]

                h, move_p, shoot_p = self._forward_prepared(xb)
                move_loss = self._ce(mb, move_p, move_cw, sw)
                shoot_loss = self._ce(sb, shoot_p, shoot_cw, sw)
                batch_losses.append(move_loss + shoot_loss)

                dm = move_p.copy()
                dm[np.arange(len(idx)), mb] -= 1.0
                wm = sw * move_cw[mb]
                dm *= (wm / max(1e-8, float(wm.sum())))[:, None]

                ds = shoot_p.copy()
                ds[np.arange(len(idx)), sb] -= 1.0
                ws = sw * shoot_cw[sb]
                ds *= (ws / max(1e-8, float(ws.sum())))[:, None]

                gw_move = h.T @ dm
                gb_move = dm.sum(axis=0)
                gw_shoot = h.T @ ds
                gb_shoot = ds.sum(axis=0)

                dh = dm @ self.w_move.T + ds @ self.w_shoot.T
                dh_pre = dh * (1.0 - h * h)
                gw1 = xb.T @ dh_pre
                gb1 = dh_pre.sum(axis=0)

                grads = [gw1, gb1, gw_move, gb_move, gw_shoot, gb_shoot]
                step += 1
                for i, (param, grad) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (grad * grad)
                    mhat = m[i] / (1.0 - beta1 ** step)
                    vhat = v[i] / (1.0 - beta2 ** step)
                    param -= float(learning_rate) * mhat / (np.sqrt(vhat) + eps)

            train_loss = float(np.mean(batch_losses))
            record = {"epoch": epoch, "train_loss": train_loss}

            if validation is not None:
                vx, vm, vs = validation
                vmove, vshoot = self.predict_proba(vx)
                vloss = self._ce(vm, vmove) + self._ce(vs, vshoot)
                mpred = np.argmax(vmove, axis=1)
                spred = np.argmax(vshoot, axis=1)
                move_acc = float((mpred == vm).mean())
                shoot_acc = float((spred == vs).mean())
                joint_acc = float(((mpred == vm) & (spred == vs)).mean())
                record.update(
                    val_loss=vloss,
                    move_accuracy=move_acc,
                    shoot_accuracy=shoot_acc,
                    joint_accuracy=joint_acc,
                )

                if vloss < best_val - float(min_delta):
                    best_val = vloss
                    best_epoch = epoch
                    best_snapshot = self._snapshot()
                    bad_epochs = 0
                else:
                    bad_epochs += 1

            history.append(record)
            if verbose:
                text = f"epoch {epoch:02d} | train={train_loss:.4f}"
                if validation is not None:
                    text += (
                        f" | val={record['val_loss']:.4f}"
                        f" | move={record['move_accuracy']:.3f}"
                        f" | shoot={record['shoot_accuracy']:.3f}"
                        f" | joint={record['joint_accuracy']:.3f}"
                    )
                    if best_epoch:
                        text += f" | best={best_epoch:02d}"
                print(text)

            if validation is not None and bad_epochs >= int(patience):
                if verbose:
                    print(
                        f"EARLY STOP at epoch {epoch}; restoring best epoch {best_epoch} "
                        f"(val={best_val:.4f})"
                    )
                break

        if best_snapshot is not None:
            self._restore(best_snapshot)

        self.best_epoch = int(best_epoch if best_epoch else history[-1]["epoch"])
        self.best_val_loss = float(best_val) if np.isfinite(best_val) else np.nan
        return history

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=np.asarray([self.VERSION], dtype=np.int32),
            full_input_dim=np.asarray([self.full_input_dim], dtype=np.int32),
            dn_dim=np.asarray([self.dn_dim], dtype=np.int32),
            context_frames=np.asarray([self.context_frames], dtype=np.int32),
            sample_hz=np.asarray([self.sample_hz], dtype=np.float32),
            lag_samples=np.asarray([self.lag_samples], dtype=np.int32),
            hidden_dim=np.asarray([self.hidden_dim], dtype=np.int32),
            feature_count=np.asarray([len(self.feature_indices)], dtype=np.int32),
            feature_indices=self.feature_indices,
            mean=self.mean,
            std=self.std,
            w1=self.w1,
            b1=self.b1,
            w_move=self.w_move,
            b_move=self.b_move,
            w_shoot=self.w_shoot,
            b_shoot=self.b_shoot,
            best_epoch=np.asarray([getattr(self, "best_epoch", 0)], dtype=np.int32),
        )
        print("Decoder V3.4 saved:", path)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        version = int(data["version"][0]) if "version" in data else 0
        if version != cls.VERSION:
            raise RuntimeError(
                f"Decoder {path} is format v{version}, but FlyIsaac V3.4 needs v{cls.VERSION}. "
                "Run: python train_decoder_v3.py"
            )
        obj = cls(
            full_input_dim=int(data["full_input_dim"][0]),
            hidden_dim=int(data["hidden_dim"][0]),
            feature_count=int(data["feature_count"][0]),
            dn_dim=int(data["dn_dim"][0]) if "dn_dim" in data else None,
            context_frames=int(data["context_frames"][0]) if "context_frames" in data else 1,
            sample_hz=float(data["sample_hz"][0]) if "sample_hz" in data else config.TEMPORAL_SAMPLE_HZ,
            lag_samples=int(data["lag_samples"][0]) if "lag_samples" in data else 0,
        )
        obj.feature_indices = data["feature_indices"].astype(np.int64)
        obj.mean = data["mean"].astype(np.float32)
        obj.std = data["std"].astype(np.float32)
        obj.w1 = data["w1"].astype(np.float32)
        obj.b1 = data["b1"].astype(np.float32)
        obj.w_move = data["w_move"].astype(np.float32)
        obj.b_move = data["b_move"].astype(np.float32)
        obj.w_shoot = data["w_shoot"].astype(np.float32)
        obj.b_shoot = data["b_shoot"].astype(np.float32)
        obj.best_epoch = int(data["best_epoch"][0]) if "best_epoch" in data else 0
        return obj
