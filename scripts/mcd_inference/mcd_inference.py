"""MC-dropout inference with dropout active, plus the dropout-placement/rate ablation (Table S2).

Why this exists. The MCD models were saved as Keras 3 files with `training=True` recorded on
the dropout layer(s) meant to be active at inference. Keras 3's `model.predict()` does NOT
honour that recorded argument: it runs every layer with training=False, so 100 "Monte Carlo"
passes through `predict()` are 100 deterministic forward passes, differing only by
floating-point noise (SD ~1e-8, 60% exactly zero). `model(x, training=True)` activates
dropout as intended (SD ~1e-2, 4% zero). Every MCD result in the paper uses the latter.

What it does. For every seed it loads the trained MCD model, rebuilds the same architecture
with the dropout RATES set per variant (rate 0 = layer inactive; Dropout has no weights, so
the trained weights load unchanged), and runs n_iter passes with training=True. Output goes
to a separate tree so nothing in the original results is overwritten:

    <out>/<variant>/<dataset_setting_name>/iter_preds_mcd_v_<v>_s_<seed>_<n>.pickle

Variants, sequence tasks:
    baseline    last dropout layer only, rate 0.25   (what the paper describes)
    all_layers  all three layers, rate 0.25
    rate05      last layer only, rate 0.5
    both        all three layers, rate 0.5
Variants, single-cell MLP (two dropout layers, trained at 0.3):
    baseline    both layers, rate 0.3                (used in the paper)
    last_layer  last layer only, rate 0.3
    rate015     both layers, rate 0.15
    last015     last layer only, rate 0.15

Run inside a Slurm allocation, never on the login node. See scripts/mcd_inference/run_ablation_tf.sbatch.
"""
import argparse, configparser, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "experiments"))

SEQ_VARIANTS = {  # name: (rate_drop1, rate_drop2, rate_drop3)
    "baseline":   (0.0, 0.0, 0.25),
    "all_layers": (0.25, 0.25, 0.25),
    "rate05":     (0.0, 0.0, 0.5),
    "both":       (0.5, 0.5, 0.5),
}
GEX_VARIANTS = {  # name: (rate_drop1, rate_drop2)
    "baseline":   (0.3, 0.3),
    "last_layer": (0.0, 0.3),
    "rate015":    (0.15, 0.15),
    "last015":    (0.0, 0.15),
}


def build_gex_mlp(rates):
    """The single-cell MC-dropout classifier (models.build_mcd_gex_mlp) with the dropout rates
    as arguments; same layer names, so the trained weights load unchanged."""
    import tensorflow as tf
    inp = tf.keras.layers.Input(shape=(20,), name="input")
    h = tf.keras.layers.Dense(10, activation="relu", name="dense1")(inp)
    h = tf.keras.layers.Dropout(rates[0], name="drop1")(h)
    h = tf.keras.layers.Dense(10, activation="relu", name="dense2")(h)
    h = tf.keras.layers.Dropout(rates[1], name="drop2")(h)
    out = tf.keras.layers.Dense(4, activation="softmax", name="output")(h)
    return tf.keras.Model(inputs=inp, outputs=out)


def mc_passes(model, x, n_iter, batch=1024):
    import numpy as np
    out = []
    for _ in range(n_iter):
        parts = [model(x[i:i + batch], training=True).numpy() for i in range(0, len(x), batch)]
        out.append(np.concatenate(parts))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=["rbp", "tf", "gex"])
    ap.add_argument("--species", required=True)
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--balanced", type=int, default=1)
    ap.add_argument("--multi-label", type=int, default=0)
    ap.add_argument("--noise", default="none")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import tensorflow as tf
    import utils as U
    import models as M

    cfg = configparser.ConfigParser()
    cfg.read(a.config)
    cfg["input_data_selection"]["classification_task"] = a.task
    cfg["input_data_selection"]["data_type_species"] = a.species
    cfg["input_data_selection"]["n_classes"] = str(a.n_classes)
    cfg["input_data_selection"]["balanced_dataset"] = str(bool(a.balanced))
    cfg["input_data_selection"]["multi_label"] = str(bool(a.multi_label))
    cfg["input_data_selection"]["noise"] = a.noise
    cfg["model_selection"]["model_type"] = "mcd"
    cfg["general"]["verbose"] = "False"
    for i in (1, 2, 3):   # the flags no longer matter (training=True is passed at call time)
        cfg["model_selection"][f"mcd_dropout_at_ineference_{i}"] = "True"
    variants = a.variants or (list(GEX_VARIANTS) if a.task == "gex" else list(SEQ_VARIANTS))

    for seed in a.seeds:
        cfg["general"]["random_seed"] = str(seed)
        np.random.seed(seed); tf.random.set_seed(seed)
        x_tr, y_tr, x_va, y_va, x_te, y_te, setting = U.read_splitted_dataset(cfg, print_plot_imbalance=False)
        version_rand = f"v_{cfg['general']['version']}_s_{seed}"
        saved = os.path.join(cfg["general"]["path_models"], setting, f"mcd_model_{version_rand}.keras")
        if not os.path.exists(saved):
            print(f"[seed {seed}] no saved MCD model at {saved} -- skipped"); continue
        trained = tf.keras.models.load_model(saved)
        weights = trained.get_weights()
        for name in variants:
            if a.task == "gex":
                model = build_gex_mlp(GEX_VARIANTS[name])
            else:
                for i, r in enumerate(SEQ_VARIANTS[name], 1):
                    cfg["model_selection"][f"dropout_rate_{i}"] = str(r)
                model = M.build_mcd_deepripe(cfg)
            model.set_weights(weights)
            preds = mc_passes(model, x_te, a.n_iter)
            sd = preds.std(0); mean = preds.mean(0); pred = mean.argmax(1)
            sdp = sd[np.arange(len(pred)), pred]
            acc = (float(np.mean((mean > 0.5) == y_te)) if a.multi_label else float((pred == y_te.argmax(1)).mean()))
            d = os.path.join(a.out, name, setting)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"iter_preds_mcd_{version_rand}_{preds.shape[0]}.pickle"), "wb") as fh:
                pickle.dump(preds, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[seed {seed}] {name:11s} acc {acc:.3f}  median SD@pred {np.median(sdp):.2e}  "
                  f"zero-SD {100*(sdp == 0).mean():5.1f}%  -> {d}", flush=True)


if __name__ == "__main__":
    main()
