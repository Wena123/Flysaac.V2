import argparse
import shutil
from pathlib import Path

import numpy as np

from isaac_v3 import config
from isaac_v3.dataset import blocked_split, convert_legacy_labels
from isaac_v3.decoder_v35 import BranchDecoderV35
from isaac_v3.grid_features import GridFeatureExtractor
from isaac_v3.multimodal import load_v35_dataset, load_v34_visual_dataset, source_matrix
from isaac_v3.temporal import build_temporal_examples

try:
    from isaac_v3.decoder import ActionDecoder as LegacyV34Decoder
except Exception:
    LegacyV34Decoder = None

HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets"
CK = HERE / "checkpoints"
BEST = CK / "isaac_decoder_v37_best.npz"
BEST_MODE = CK / config.V37_BEST_MODE_FILE
ASSIST = CK / "isaac_decoder_v37_best_assisted.npz"
ASSIST_MODE = CK / config.V37_BEST_ASSISTED_MODE_FILE


def discover(patterns):
    if patterns:
        files = []
        for p in patterns:
            q = Path(p)
            files += [q] if q.exists() else list(Path().glob(p))
        assisted, v34 = [], []
        for path in sorted(set(files)):
            name = path.name.lower()
            if "v37" in name or "v36" in name or "v35" in name:
                assisted.append(path)
            elif "v34" in name:
                v34.append(path)
        return assisted, v34
    assisted = (
        sorted(DATA.glob("isaac_v37_demo_*.npz")) + sorted(DATA.glob("isaac_v37_dagger_*.npz")) +
        sorted(DATA.glob("isaac_v36_demo_*.npz")) + sorted(DATA.glob("isaac_v36_dagger_*.npz")) +
        sorted(DATA.glob("isaac_v35_demo_*.npz")) + sorted(DATA.glob("isaac_v35_dagger_*.npz"))
    )
    v34 = sorted(DATA.glob("isaac_v34_demo_*.npz")) + sorted(DATA.glob("isaac_v34_dagger_*.npz"))
    return assisted, v34


def load_records(v35, v34):
    records = []
    dn_dim = None
    vis_dim = None
    grid_dim = GridFeatureExtractor().dim
    print("FLYISAAC V3.7 FAST TACTICAL-SENSORY DATASETS")
    print("=======================================")

    for path in v35:
        dn, vis, grid, y = load_v35_dataset(path)
        move, shoot, st = convert_legacy_labels(y)
        if dn_dim is None:
            dn_dim, vis_dim = dn.shape[1], vis.shape[1]
        if (dn.shape[1], vis.shape[1], grid.shape[1]) != (dn_dim, vis_dim, grid_dim):
            raise RuntimeError(f"Dimension mismatch: {path.name}")
        records.append(dict(
            path=path, dn=dn, visual=vis, grid=grid, move=move, shoot=shoot,
            is_dagger="dagger" in path.name.lower(), has_grid=True,
            sensor_version=(37 if "v37" in path.name.lower() else (36 if "v36" in path.name.lower() else 35)),
        ))
        print(
            f"{'DAGGER' if 'dagger' in path.name.lower() else 'DEMO':6s} {path.name}: "
            f"samples={len(dn)} DN={dn.shape[1]} VIS={vis.shape[1]} GRID={grid.shape[1]} "
            f"conflicts={st['movement_conflicts']} multi_shoot={st['shoot_ambiguous']}"
        )

    legacy_count = 0
    for path in v34:
        try:
            dn, vis, y = load_v34_visual_dataset(path)
        except Exception as exc:
            print(f"SKIP V3.4 {path.name}: {exc}")
            continue
        if dn_dim is None:
            dn_dim, vis_dim = dn.shape[1], vis.shape[1]
        if dn.shape[1] != dn_dim or vis.shape[1] != vis_dim:
            print(f"SKIP V3.4 {path.name}: width mismatch DN={dn.shape[1]} VIS={vis.shape[1]}")
            continue
        move, shoot, _ = convert_legacy_labels(y)
        records.append(dict(
            path=path, dn=dn, visual=vis, grid=None, move=move, shoot=shoot,
            is_dagger="dagger" in path.name.lower(), has_grid=False, sensor_version=34,
        ))
        legacy_count += 1
        print(
            f"LEGACY {'DAGGER' if 'dagger' in path.name.lower() else 'DEMO':6s} {path.name}: "
            f"samples={len(dn)} DN={dn.shape[1]} VIS={vis.shape[1]} -> PURE VISUAL transfer/training"
        )

    if not records:
        raise RuntimeError(
            "No compatible V3.4/V3.5 datasets.\n"
            "If you only have an old V3.4 checkpoint, PLAY can use it directly, but\n"
            "V3.5 transfer training still needs frames. Record V3.5 with:\n"
            "  python collect_isaac_v3.py --minutes 15"
        )

    has_grid = any(r["has_grid"] for r in records)
    print(f"\nLegacy V3.4 sessions available: {legacy_count}")
    print("assisted GRID sessions available:", sum(1 for r in records if r["has_grid"]))
    print("V3.7 tactical/projectile sessions:", sum(1 for r in records if r.get("sensor_version") == 37))
    print("V3.6 brain-grid/dopamine sessions:", sum(1 for r in records if r.get("sensor_version") == 36))
    if not has_grid:
        print("NOTE: no V3.5 grid dataset yet -> this run can train/migrate PURE VISUAL only.")
    return records, int(dn_dim), int(vis_dim), int(grid_dim), has_grid


def eligible(records, mode):
    if mode == "visual":
        return list(records)
    grid_records = [r for r in records if r["has_grid"]]
    if mode == "visual_grid_dn":
        v37 = [r for r in grid_records if r.get("sensor_version") == 37]
        if v37:
            return v37
        v36 = [r for r in grid_records if r.get("sensor_version") == 36]
        return v36 if v36 else grid_records
    return grid_records


def build_split(records, mode, lag, ctx, dagger_weight):
    tx, tm, ts, tw, vx, vm, vs = [], [], [], [], [], [], []
    for r in eligible(records, mode):
        grid = r["grid"] if r["grid"] is not None else np.zeros((len(r["dn"]), 1), np.float32)
        raw = source_matrix(mode, r["dn"], r["visual"], grid)
        x, m, s, _, _ = build_temporal_examples(
            raw, r["move"], r["shoot"], context_frames=ctx, lag_samples=lag
        )
        if len(x) < 4:
            continue
        tr, va = blocked_split(len(x))
        w = float(dagger_weight) if r["is_dagger"] else 1.0
        tx.append(x[tr]); tm.append(m[tr]); ts.append(s[tr]); tw.append(np.full(len(tr), w, np.float32))
        vx.append(x[va]); vm.append(m[va]); vs.append(s[va])
    if not tx or not vx:
        raise RuntimeError(f"Not enough data for {mode}")
    return tuple(map(np.concatenate, (tx, tm, ts, tw, vx, vm, vs)))


def balanced(y, p, n):
    vals = []
    for c in range(n):
        q = y == c
        if np.any(q):
            vals.append(float((p[q] == c).mean()))
    return float(np.mean(vals)) if vals else 0.0


def eval_dec(d, x, m, s):
    mp, sp = d.predict_proba(x)
    mh, sh = np.argmax(mp, 1), np.argmax(sp, 1)
    return dict(
        move_acc=float((mh == m).mean()),
        shoot_acc=float((sh == s).mean()),
        joint=float(((mh == m) & (sh == s)).mean()),
        move_bal=balanced(m, mh, len(config.MOVEMENT_CLASSES)),
        shoot_bal=balanced(s, sh, len(config.SHOOT_CLASSES)),
    )


def probe(xtr, mtr, str_, wtr, xv, mv, sv, k=256, ridge=1.0):
    var = np.var(xtr, 0)
    k = min(k, xtr.shape[1])
    idx = np.argpartition(var, -k)[-k:] if k < xtr.shape[1] else np.arange(xtr.shape[1])
    a, b = xtr[:, idx], xv[:, idx]
    mean, std = a.mean(0), a.std(0)
    std[std < 1e-5] = 1
    a, b = (a - mean) / std, (b - mean) / std
    a = np.c_[a, np.ones(len(a), np.float32)]
    b = np.c_[b, np.ones(len(b), np.float32)]
    nm, ns = len(config.MOVEMENT_CLASSES), len(config.SHOOT_CLASSES)
    y = np.zeros((len(a), nm + ns), np.float32)
    y[np.arange(len(a)), mtr] = 1
    y[np.arange(len(a)), nm + str_] = 1
    sw = np.sqrt(np.maximum(wtr, 1e-6))
    aw, yw = a * sw[:, None], y * sw[:, None]
    gram = aw.T @ aw
    gram.flat[::gram.shape[0] + 1] += ridge
    W = np.linalg.solve(gram.astype(np.float64), (aw.T @ yw).astype(np.float64))
    pr = b @ W
    mh, sh = np.argmax(pr[:, :nm], 1), np.argmax(pr[:, nm:], 1)
    mba, sba = balanced(mv, mh, nm), balanced(sv, sh, ns)
    joint = float(((mh == mv) & (sh == sv)).mean())
    return mba, sba, joint, .5 * (mba + sba) + .1 * joint


def select_lag(records, mode, dw):
    print(f"\nLAG SWEEP ({mode.upper()})")
    print("=" * (12 + len(mode)))
    best = None
    for lag in range(config.LAG_SWEEP_MIN, config.LAG_SWEEP_MAX + 1):
        x, m, s, w, xv, mv, sv = build_split(records, mode, lag, 1, dw)
        mba, sba, j, score = probe(x, m, s, w, xv, mv, sv, config.LAG_PROBE_FEATURES)
        causal = lag >= 0 or config.ALLOW_NEGATIVE_LAG_DEFAULT
        mark = "*" if causal else " "
        print(
            f"{mark} lag={lag:+3d} ({1000*lag/config.TEMPORAL_SAMPLE_HZ:+5.0f} ms) | "
            f"move_bal={mba:.3f} shoot_bal={sba:.3f} joint={j:.3f} score={score:.3f}"
        )
        if causal and (best is None or score > best[1]):
            best = (lag, score)
    print(f"Selected lag: {best[0]:+d} samples ({1000*best[0]/config.TEMPORAL_SAMPLE_HZ:+.0f} ms)")
    return int(best[0])


def find_v34_visual_teacher():
    if LegacyV34Decoder is None:
        return None, None
    direct = CK / "isaac_decoder_v34_visual.npz"
    if direct.exists():
        try:
            return LegacyV34Decoder.load(direct), direct
        except Exception as exc:
            print("V3.4 teacher load warning:", exc)
    mode_file = CK / getattr(config, "V34_BEST_MODE_FILE", "isaac_decoder_v34_best_mode.txt")
    best = CK / "isaac_decoder_v34_best.npz"
    if mode_file.exists() and best.exists():
        mode = mode_file.read_text(encoding="utf-8").strip().lower()
        if mode == "visual":
            try:
                return LegacyV34Decoder.load(best), best
            except Exception as exc:
                print("V3.4 best teacher load warning:", exc)
    return None, None


def build_distill_training(records, teacher, context_frames):
    """Only the TRAIN block is distilled; blocked validation remains untouched."""
    xs = []
    for r in records:
        visual = np.asarray(r["visual"], np.float32)
        dummy = np.zeros(len(visual), dtype=np.int64)
        x, _, _, _, _ = build_temporal_examples(
            visual, dummy, dummy, context_frames=context_frames, lag_samples=0
        )
        if len(x) < 4:
            continue
        tr, _ = blocked_split(len(x))
        xs.append(x[tr])
    if not xs:
        return None
    x = np.concatenate(xs)
    if x.shape[1] != teacher.full_input_dim:
        print(
            f"TRANSFER SKIPPED: teacher expects {teacher.full_input_dim} values, "
            f"current visual context has {x.shape[1]}."
        )
        return None
    mp, sp = teacher.predict_proba(x)
    return x, np.asarray(mp, np.float32), np.asarray(sp, np.float32)


def main():
    p = argparse.ArgumentParser(description="Train FlyIsaac V3.7 with V3.4 transfer + tactical/projectile-state DN.")
    p.add_argument("patterns", nargs="*")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--context-frames", type=int, default=config.TEMPORAL_CONTEXT_FRAMES)
    p.add_argument("--dagger-weight", type=float, default=config.DAGGER_WEIGHT)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=7e-4)
    p.add_argument("--no-transfer", action="store_true", help="Disable V3.4 teacher distillation/warm start.")
    args = p.parse_args()

    v35, v34 = discover(args.patterns)
    records, dn_dim, vis_dim, grid_dim, has_grid = load_records(v35, v34)
    modes = list(config.V35_INPUT_MODES if has_grid else ("visual",))
    lags = {m: select_lag(records, m, args.dagger_weight) for m in modes}
    CK.mkdir(parents=True, exist_ok=True)

    teacher = teacher_path = None
    if not args.no_transfer:
        teacher, teacher_path = find_v34_visual_teacher()
    if teacher is not None:
        print("\nV3.4 KNOWLEDGE TRANSFER")
        print("=======================")
        print("Teacher:", teacher_path)
        print("Teacher format: V3.4")
        print("Teacher context:", teacher.context_frames)
        print("Teacher visual width:", teacher.full_input_dim // teacher.context_frames)
    else:
        print("\nV3.4 KNOWLEDGE TRANSFER: no compatible V3.4 VISUAL checkpoint found.")
        print("Supervised reuse of compatible V3.4 datasets is still ON.")

    results, paths = {}, {}
    visual_student = None

    for mode in modes:
        x, m, s, w, xv, mv, sv = build_split(
            records, mode, lags[mode], args.context_frames, args.dagger_weight
        )
        print(f"\nTRAINING {mode.upper()}")
        print("=" * (9 + len(mode)))
        print("train:", x.shape, "validation:", xv.shape)

        d = BranchDecoderV35(
            mode, dn_dim, vis_dim, grid_dim,
            context_frames=args.context_frames,
            sample_hz=config.TEMPORAL_SAMPLE_HZ,
            lag_samples=lags[mode],
        )

        fit_pre = True
        preserve = ()
        if mode == "visual" and teacher is not None:
            if teacher.context_frames != args.context_frames:
                print("TRANSFER SKIPPED: teacher/student context frame mismatch.")
            else:
                pack = build_distill_training(records, teacher, args.context_frames)
                if pack is not None:
                    dx, dmp, dsp = pack
                    print(f"Distilling V3.4 teacher -> V3.5 visual branch on {len(dx)} TRAIN-only examples...")
                    d.transfer_teacher = str(teacher_path.name)
                    d.distill_soft(
                        dx, dmp, dsp,
                        epochs=args.distill_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.distill_lr,
                        verbose=True,
                    )
                    print("Transferred from V3.4 teacher: YES")
                    fit_pre = False

        elif mode != "visual" and visual_student is not None and "visual" in d.branches:
            d.warm_start_from_visual(visual_student)
            print("Warm-started from learned V3.5 VISUAL branch: YES")
            preserve = ("visual",)

        d.train(
            x, m, s,
            sample_weights=w,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation=(xv, mv, sv),
            patience=config.EARLY_STOP_PATIENCE,
            verbose=True,
            fit_preprocessor=fit_pre,
            preserve_branches=preserve,
        )
        met = eval_dec(d, xv, mv, sv)
        met["score"] = .5 * (met["move_bal"] + met["shoot_bal"]) + .1 * met["joint"]
        path = CK / f"isaac_decoder_v37_{mode}.npz"
        d.save(path)
        results[mode], paths[mode] = met, path
        if mode == "visual":
            visual_student = d

    print("\nV3.7 TRANSFER + TACTICAL/PROJECTILE ABLATION")
    print("================================")
    print("mode            lag move_acc move_bal shoot_acc shoot_bal joint  score")
    for mode in modes:
        q = results[mode]
        print(
            f"{mode:15s} {lags[mode]:+3d}  {q['move_acc']:.3f}    {q['move_bal']:.3f}    "
            f"{q['shoot_acc']:.3f}     {q['shoot_bal']:.3f}    {q['joint']:.3f}  {q['score']:.3f}"
        )

    best = max(modes, key=lambda n: results[n]["score"])
    shutil.copy2(paths[best], BEST)
    BEST_MODE.write_text(best + "\n", encoding="utf-8")
    print("\nBEST OVERALL:", best.upper())

    assisted_modes = [m for m in modes if "grid" in m]
    if assisted_modes:
        assisted = max(assisted_modes, key=lambda n: results[n]["score"])
        shutil.copy2(paths[assisted], ASSIST)
        ASSIST_MODE.write_text(assisted + "\n", encoding="utf-8")
        print("BEST GRID-ASSISTED:", assisted.upper())
    else:
        print("BEST GRID-ASSISTED: not trained yet (record V3.5 grid data first)")

    if teacher is not None:
        print("Transferred from V3.4 teacher: YES ->", teacher_path.name)
    else:
        print("Transferred from V3.4 teacher: NO compatible teacher found")
    print("PURE VISION V3.7 model:", paths["visual"])
    print("Next: python play_isaac_v3.py")


if __name__ == "__main__":
    main()
