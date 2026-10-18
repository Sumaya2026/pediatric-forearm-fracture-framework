#coding:utf-8
"""
Framework 3: predict the patient's AGE and GENDER from the X-rays + fracture label.

    inputs  : front X-ray, side X-ray, fracture category (Proximal / Midshaft / Distal)
    outputs : gender  -> SVM classifier            (male / female)
              age     -> Linear Regression          (years, continuous)

Pipeline
    1. A ResNet34 backbone turns each view into a 512-d feature (front + side -> 1024-d).
       By default this is the ImageNet ResNet34; pass --backbone <URFNet.pth / URFFusionNet.pth>
       to reuse the encoder you already trained for fracture classification.
    2. The fracture label is appended as a one-hot vector (3-d).
    3. Features are cached to .npz, so the (slow) CNN step runs only once.
    4. scikit-learn models are fit on the cached features:
         gender : StandardScaler -> PCA -> SVC(rbf, class_weight='balanced'), C/gamma by 5-fold CV
         age    : StandardScaler -> PCA -> LinearRegression
    5. Test metrics, per-image predictions (CSV) and the fitted models (joblib) are saved.

Usage
    python age_gender.py \
        --train_json /root/autodl-tmp/train.json --test_json /root/autodl-tmp/test.json \
        --train_csv  /root/autodl-tmp/train.csv  --test_csv  /root/autodl-tmp/test.csv \
        --out_dir    /root/autodl-tmp/age_gender
    # optional: --backbone /root/autodl-tmp/pro/URFNet.pth
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)

CLASS_NAMES = ["Proximal", "Midshaft", "Distal"]
GENDER_NAMES = ["female", "male"]  # 0 = female, 1 = male


# --------------------------------------------------------------------------- #
#  Labels / metadata
# --------------------------------------------------------------------------- #
def fracture_label(path: str) -> int:
    """Same rule as dataset.Clsdata."""
    if 'Proximal ulna and radius fractures' in path:
        return 0
    if 'Midshaft ulna and radius fractures' in path:
        return 1
    return 2


def parse_age(s: str) -> float:
    """'Y10' -> 10.0 years, 'M8' -> 0.667 years."""
    s = str(s).strip().upper()
    if s.startswith("Y"):
        return float(s[1:])
    if s.startswith("M"):
        return float(s[1:]) / 12.0
    return float(s)


def load_raw_meta(csv_path: str) -> dict:
    """Raw csv (Category, Separate folders labeled, Gender, Age) -> {folder_id: (age_years, gender)}.

    Columns are taken by position, so a broken header (e.g. test.csv's first
    column is named '63478201302170280') does not matter.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df = df.iloc[:, :4]
    df.columns = ["category", "folder_id", "gender", "age"]
    df["folder_id"] = df["folder_id"].str.strip()
    meta = {}
    for _, r in df.iterrows():
        g = str(r["gender"]).strip().lower()
        if g not in ("male", "female"):
            continue
        meta[r["folder_id"]] = (parse_age(r["age"]), 1 if g == "male" else 0)
    return meta


def folder_id(img_path: str, meta: dict):
    for part in reversed(Path(img_path).parent.parts):
        if part in meta:
            return part
    return None


def build_index(json_path: str, csv_path: str):
    """Front-image paths that have a metadata row, with their targets."""
    with open(json_path) as f:
        imgs = json.load(f)
    # keep one entry per study (front view); side is derived from it
    imgs = sorted({p for p in imgs if Path(p).name == "front.jpg"})
    meta = load_raw_meta(csv_path)

    paths, fids, ages, genders, labels = [], [], [], [], []
    missing = 0
    for p in imgs:
        fid = folder_id(p, meta)
        if fid is None:
            missing += 1
            continue
        age, g = meta[fid]
        paths.append(p); fids.append(fid); ages.append(age); genders.append(g)
        labels.append(fracture_label(p))
    print(f"[{Path(json_path).name}] {len(paths)} studies with metadata "
          f"({missing} images had no row in {Path(csv_path).name} and were skipped)")
    return (paths, fids, np.array(ages, dtype=np.float32),
            np.array(genders, dtype=np.int64), np.array(labels, dtype=np.int64))


# --------------------------------------------------------------------------- #
#  CNN feature extraction (torch is only imported here)
# --------------------------------------------------------------------------- #
def extract_features(paths, backbone_path=None, img_size=224, batch_size=32, num_workers=4):
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image, ImageOps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone_path:
        import models  # noqa: F401  (lets torch.load unpickle URFNet / URFFusionNet)
        full = torch.load(backbone_path, weights_only=False, map_location=device)
        encoder = full.resnet
        print(f"backbone: ResNet34 encoder from {backbone_path}")
    else:
        encoder = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
        encoder.fc = nn.Identity()
        print("backbone: ImageNet ResNet34")
    encoder = encoder.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class PairDS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            front = paths[i]
            side = front.replace("front.jpg", "side.jpg")
            a = ImageOps.exif_transpose(Image.open(front)).convert("RGB")
            b = ImageOps.exif_transpose(Image.open(side)).convert("RGB")
            return tf(a), tf(b)

    dl = DataLoader(PairDS(), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    with torch.no_grad():
        for i, (x1, x2) in enumerate(dl):
            f1 = encoder(x1.to(device))
            f2 = encoder(x2.to(device))
            feats.append(torch.cat([f1, f2], dim=1).cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  {min((i + 1) * batch_size, len(paths))}/{len(paths)}")
    return np.concatenate(feats, axis=0).astype(np.float32)


def get_split(name, json_path, csv_path, out_dir, args):
    paths, fids, ages, genders, labels = build_index(json_path, csv_path)
    tag = "finetuned" if args.backbone else "imagenet"
    cache = Path(out_dir) / f"feats_{name}_{tag}.npz"
    if cache.exists() and not args.recompute:
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == paths:
            print(f"loaded cached features: {cache}")
            return paths, fids, d["X"], ages, genders, labels
    X = extract_features(paths, args.backbone, args.img_size, args.batch_size, args.num_workers)
    np.savez_compressed(cache, X=X, paths=np.array(paths))
    print(f"saved features: {cache}  shape={X.shape}")
    return paths, fids, X, ages, genders, labels


def add_label(X_img, labels, use_label=True):
    """Concatenate image features with the one-hot fracture label."""
    if not use_label:
        return X_img
    onehot = np.eye(3, dtype=np.float32)[labels]
    return np.hstack([X_img, onehot])


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #
def gender_model(n_pca, seed):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=seed)),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
    ])
    grid = {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 1e-3]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)


def age_model(n_pca, seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=seed)),
        ("lr", LinearRegression()),
    ])


def eval_gender(y, pred, prob):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
    }


def eval_age(y, pred):
    err = np.abs(y - pred)
    return {
        "MAE_years": mean_absolute_error(y, pred),
        "RMSE_years": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred),
        "within_1yr": float((err <= 1).mean()),
        "within_2yr": float((err <= 2).mean()),
    }


def fmt(d):
    return ", ".join(f"{k} {v:.4f}" for k, v in d.items())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="/root/autodl-tmp/train.json")
    ap.add_argument("--test_json",  default="/root/autodl-tmp/test.json")
    ap.add_argument("--train_csv",  default="/root/autodl-tmp/train.csv", help="raw csv with Age like Y10 / M8")
    ap.add_argument("--test_csv",   default="/root/autodl-tmp/test.csv")
    ap.add_argument("--out_dir",    default="/root/autodl-tmp/age_gender")
    ap.add_argument("--backbone",   default=None,
                    help="optional trained URFNet/URFFusionNet .pth; its ResNet34 is used as feature extractor")
    ap.add_argument("--no_label", action="store_true", help="ablation: drop the fracture label from the inputs")
    ap.add_argument("--pca", type=int, default=128, help="PCA components before SVM / LR")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("-b", "--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--recompute", action="store_true", help="ignore cached features")
    ap.add_argument("-s", "--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    tr_paths, _, Xtr_img, age_tr, gen_tr, lab_tr = get_split("train", args.train_json, args.train_csv, args.out_dir, args)
    te_paths, te_fids, Xte_img, age_te, gen_te, lab_te = get_split("test", args.test_json, args.test_csv, args.out_dir, args)

    use_label = not args.no_label
    Xtr = add_label(Xtr_img, lab_tr, use_label)
    Xte = add_label(Xte_img, lab_te, use_label)
    n_pca = min(args.pca, Xtr.shape[1], Xtr.shape[0] - 1)
    print(f"\nfeatures: train {Xtr.shape}, test {Xte.shape}, label input: {use_label}, PCA: {n_pca}")
    print(f"train: {int(gen_tr.sum())} male / {int((1 - gen_tr).sum())} female, "
          f"age {age_tr.mean():.2f} ± {age_tr.std():.2f} years")

    # ---------------- gender: SVM ----------------
    print("\n=== Gender (SVM) ===")
    svm = gender_model(n_pca, args.seed).fit(Xtr, gen_tr)
    print(f"best params: {svm.best_params_}  (CV balanced acc {svm.best_score_:.4f})")
    g_pred = svm.predict(Xte)
    g_prob = svm.decision_function(Xte)  # >0 means male; used for ROC-AUC
    g_metrics = eval_gender(gen_te, g_pred, g_prob)
    print("test:", fmt(g_metrics))
    print("confusion matrix (rows=true, cols=pred; 0=female, 1=male):")
    print(confusion_matrix(gen_te, g_pred, labels=[0, 1]))
    print(classification_report(gen_te, g_pred, target_names=GENDER_NAMES, zero_division=0))
    base_g = eval_gender(gen_te, np.ones_like(gen_te), np.full(len(gen_te), 0.5))
    print(f"baseline (always 'male'): accuracy {base_g['accuracy']:.4f}, "
          f"balanced accuracy {base_g['balanced_accuracy']:.4f}")

    # ---------------- age: Linear Regression ----------------
    print("\n=== Age (Linear Regression) ===")
    lr = age_model(n_pca, args.seed).fit(Xtr, age_tr)
    a_pred = np.clip(lr.predict(Xte), 0, None)
    a_metrics = eval_age(age_te, a_pred)
    print("test:", fmt(a_metrics))
    base_a = eval_age(age_te, np.full_like(age_te, age_tr.mean()))
    print(f"baseline (predict train mean): MAE {base_a['MAE_years']:.4f}, R2 {base_a['R2']:.4f}")
    print("MAE by fracture class:")
    for c in range(3):
        m = lab_te == c
        if m.any():
            print(f"  {CLASS_NAMES[c]:9s} n={m.sum():4d}  MAE {mean_absolute_error(age_te[m], a_pred[m]):.3f}")

    # ---------------- save ----------------
    joblib.dump(svm.best_estimator_, Path(args.out_dir) / "gender_svm.joblib")
    joblib.dump(lr, Path(args.out_dir) / "age_linreg.joblib")
    pd.DataFrame({
        "folder_id": te_fids, "front_path": te_paths,
        "fracture": [CLASS_NAMES[c] for c in lab_te],
        "gender_true": [GENDER_NAMES[g] for g in gen_te],
        "gender_pred": [GENDER_NAMES[g] for g in g_pred],
        "svm_score_male": np.round(g_prob, 4),
        "age_true": np.round(age_te, 3), "age_pred": np.round(a_pred, 3),
    }).to_csv(Path(args.out_dir) / "test_predictions.csv", index=False)
    with open(Path(args.out_dir) / "metrics.json", "w") as f:
        json.dump({"gender_svm": g_metrics, "gender_baseline_always_male": base_g,
                   "age_linear_regression": a_metrics, "age_baseline_train_mean": base_a,
                   "svm_best_params": {k: str(v) for k, v in svm.best_params_.items()},
                   "use_label": use_label, "pca": n_pca, "backbone": args.backbone or "imagenet"},
                  f, indent=2)
    print(f"\nsaved models, predictions and metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
