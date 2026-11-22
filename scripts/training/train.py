"""Train one model on one dataset setting and write its 100 inference passes.

    python3 scripts/training/train.py --setting tf_mislabeled --model mcd --seed 7 \
        --config config/config.ini --datasets $D/datasets_v2/ --models $D/models_v20/ --results $D/results_v20/

Settings: tf_mislabeled | tf_mislabeled15 | tf_mislabeled30 | rbp_imbal_single | rbp_imbal_multi | gex_imbal | gex_bal (used with the leave-donor-out tree). Models: det | mcd | bnn | ens.
Hyper-parameters: det/ens/mcd 50 epochs, patience 5 (ens on TF: 100/10);
bnn 1000 epochs, patience 50 (single-label) / 100 (multi-label); learning rate 0.005 for bnn, 0.002 otherwise (override with UQ_LR).
BNN needs the legacy Keras (TF_USE_LEGACY_KERAS=1, set here before TensorFlow is imported); the others
run on Keras 3. MC-dropout passes are generated with dropout active (eval_functions.infer_probabilistic_model).
Used by scripts/training/run_train_perturbed.sbatch (job array). Never on the login node.
"""
import argparse, configparser, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--setting", required=True, choices=["tf_mislabeled", "tf_mislabeled15", "tf_mislabeled30", "rbp_imbal_single", "rbp_imbal_multi", "gex_imbal", "gex_bal", "tf_clean", "rbp_bal_single", "rbp_bal_multi", "rbp_app27"])
ap.add_argument("--model", required=True, choices=["det", "mcd", "bnn", "ens"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--datasets", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--version", default="20")
ap.add_argument("--run-mode", default="all", choices=["all", "training", "generate_evaluation"])
a = ap.parse_args()

if a.model == "bnn":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
else:
    os.environ.pop("TF_USE_LEGACY_KERAS", None)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "experiments"))
import main_run as MR  # noqa: E402

SETTINGS = {
    "tf_mislabeled":    dict(classification_task="tf",  data_type_species="simulated_human", n_classes="6", balanced_dataset="True",  multi_label="False", noise="mislabeled"),
    "tf_mislabeled15":  dict(classification_task="tf",  data_type_species="simulated_human", n_classes="6", balanced_dataset="True",  multi_label="False", noise="mislabeled15"),
    "tf_mislabeled30":  dict(classification_task="tf",  data_type_species="simulated_human", n_classes="6", balanced_dataset="True",  multi_label="False", noise="mislabeled30"),
    "rbp_imbal_single": dict(classification_task="rbp", data_type_species="real_human",      n_classes="3", balanced_dataset="False", multi_label="False", noise="none"),
    "rbp_imbal_multi":  dict(classification_task="rbp", data_type_species="real_human",      n_classes="3", balanced_dataset="False", multi_label="True",  noise="none"),
    "gex_imbal":        dict(classification_task="gex", data_type_species="real_human",      n_classes="4", balanced_dataset="False", multi_label="False", noise="none"),
    "gex_bal":          dict(classification_task="gex", data_type_species="real_human",      n_classes="4", balanced_dataset="True",  multi_label="False", noise="none"),
    "tf_clean":         dict(classification_task="tf",  data_type_species="simulated_human", n_classes="6", balanced_dataset="True",  multi_label="False", noise="none"),
    "rbp_bal_single":   dict(classification_task="rbp", data_type_species="real_human",      n_classes="3", balanced_dataset="True",  multi_label="False", noise="none"),
    "rbp_bal_multi":    dict(classification_task="rbp", data_type_species="real_human",      n_classes="3", balanced_dataset="True",  multi_label="True",  noise="none"),
    # 27-class application dataset, placed by scripts/splits/prep_rbp27_dataset.py under
    # rbp/real_human/train_imbalanced_test_balanced/single_label/27class_split<seed>.pickle
    "rbp_app27":        dict(classification_task="rbp", data_type_species="real_human",      n_classes="27", balanced_dataset="False", multi_label="False", noise="none"),
}

cfg = configparser.ConfigParser()
cfg.read(a.config)
cfg["general"]["path_datasets"] = a.datasets.rstrip("/") + "/"
cfg["general"]["path_models"] = a.models.rstrip("/") + "/"
cfg["general"]["path_results"] = a.results.rstrip("/") + "/"
cfg["general"]["version"] = a.version
cfg["general"]["random_seed"] = str(a.seed)
cfg["general"]["run_mode"] = a.run_mode
cfg["general"]["verbose"] = "True"
cfg["general"]["neptune_token"] = ""
cfg["general"]["neptune_project"] = ""
for k, v in SETTINGS[a.setting].items():
    cfg["input_data_selection"][k] = v
cfg["input_data_selection"]["application_mode"] = "False"
cfg["model_selection"]["model_type"] = a.model
cfg["model_selection"]["batch_size"] = "128"
cfg["model_selection"]["early_stopping"] = "True"
multi = a.setting in ("rbp_imbal_multi", "rbp_bal_multi")
# Training protocol (Section S1.1.3 of the paper): BNN at most 1000 epochs, patience 50
# (single-label) / 100 (multi-label); det/mcd/ens 50 epochs, patience 5, except ENS on the TF
# data 100 / 10. With fewer BNN epochs or a higher learning rate the BNNs do not converge
# (TF stuck at chance, GEX at 0.46 after 100 epochs).
if a.model == "bnn":
    cfg["model_selection"]["epochs"] = os.environ.get("UQ_BNN_EPOCHS", "1000")   # step cap; UQ_BNN_EPOCHS overrides
    cfg["model_selection"]["patience"] = "100" if multi else "50"
else:
    tf_ens = (a.model == "ens" and a.setting in ("tf_mislabeled", "tf_mislabeled15", "tf_mislabeled30", "tf_clean"))
    cfg["model_selection"]["epochs"] = "100" if tf_ens else "50"
    cfg["model_selection"]["patience"] = "10" if tf_ens else "5"
cfg["model_selection"]["learning_rate"] = os.environ.get("UQ_LR", "0.005" if a.model == "bnn" else "0.002")
for i in (1, 2, 3):
    cfg["model_selection"][f"dropout_rate_{i}"] = "0.25"
    cfg["model_selection"][f"mcd_dropout_at_ineference_{i}"] = "True"
cfg["evaluation"]["n_iterations"] = "100"

print(f"== {a.setting} / {a.model} / seed {a.seed} / v{a.version} / {a.run_mode}", flush=True)
MR.main_run_function(cfg, MR._NullRun())
print("== done", flush=True)
