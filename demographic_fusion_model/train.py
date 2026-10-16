#coding:utf-8
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save plots to file; no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
import json
import time
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from dataset import Clsdata
from torchsummary import summary
import argparse
import random
from torch.optim.lr_scheduler import StepLR
import timm
from sklearn.model_selection import train_test_split
from models import FusionModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed: int = 123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CLASS_NAMES = ["Proximal", "Midshaft", "Distal"]

def label_of(path):
    # Same rule as Clsdata.__getitem__ in dataset.py
    if 'Proximal ulna and radius fractures' in path:
        return 0
    if 'Midshaft ulna and radius fractures' in path:
        return 1
    return 2

def class_weights(img_paths, power=1.0, num_classes=3):
    counts = np.bincount([label_of(p) for p in img_paths], minlength=num_classes)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))   # inverse frequency
    w = w ** power                                              # power=0.5 -> softened weights
    print("Train counts:", dict(zip(CLASS_NAMES, counts.tolist())))
    print("Class weights:", dict(zip(CLASS_NAMES, np.round(w, 3).tolist())))
    return torch.tensor(w, dtype=torch.float32)

def save_history(history, prefix):
    with open(prefix + ".json", "w") as f:
        json.dump(history, f, indent=2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    a1.plot(history["epoch"], history["train_loss"])
    a1.set_title("Training loss"); a1.set_xlabel("epoch")
    a2.plot(history["epoch"], [x * 100 for x in history["val_acc"]], label="accuracy")
    a2.plot(history["epoch"], [x * 100 for x in history["val_f1"]], label="macro-F1")
    a2.set_title("Validation metrics (%)"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(prefix + ".png", dpi=120); plt.close(fig)

def train(model, train_root, batch_size, lr, epochs, weight_power=1.0, val_ratio=0.15, seed=123, meta_csv=None):
    # step1: data — split train.json into train / validation (test.json is NOT used here)
    with open(train_root, 'r') as fr:
        all_imgs = json.load(fr)
    labels = [label_of(p) for p in all_imgs]
    train_imgs, val_imgs = train_test_split(all_imgs, test_size=val_ratio,
                                            stratify=labels, random_state=seed)
    print(f"Split: {len(train_imgs)} train / {len(val_imgs)} val (stratified, seed {seed})")

    train_data = Clsdata(train_imgs, train=True, shuffle=True, meta_csv=meta_csv)
    val_data   = Clsdata(val_imgs, train=False, shuffle=False, test=True, meta_csv=meta_csv)   # no augmentation
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    train_length = len(train_data)

    # step2: weighted criterion and optimizer (weights from the training part only)
    cls_w = class_weights(train_imgs, power=weight_power).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight=cls_w)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    best_path = f'./FusionModel.pth'
    hist_prefix = os.path.splitext(best_path)[0] + "_history"
    with open(os.path.splitext(best_path)[0] + "_split.json", "w") as f:   # record the split
        json.dump({"train": train_imgs, "val": val_imgs}, f, indent=2)
    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_acc, bestEpoch = 0.0, 0.0, -1

    # step3: training
    for epoch in range(epochs):
        epochStart = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()
        epoch_loss = 0.0
        for i, (img1, img2, meta, tgt_cls) in enumerate(train_dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            meta = meta.to(device)
            tgt_cls = tgt_cls.to(device)

            out_cls = model(img1, img2, meta)
            loss_cls = criterion_cls(out_cls, tgt_cls)
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            epoch_loss += loss_cls.item() * img1.size(0)
        scheduler.step()
        epoch_loss /= train_length
        epochEnd = time.time()

        # step4: validate
        acc_cls, f1_cls = val(model, val_dataloader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(epoch_loss))
        history["val_acc"].append(float(acc_cls))
        history["val_f1"].append(float(f1_cls))

        print("Epoch : {:03d}, Loss: {:.4f}, Val Acc: {:.2f}%, Val F1: {:.2f}%, Time: {:.1f}s".format(
            epoch + 1, epoch_loss, acc_cls * 100, f1_cls * 100, epochEnd - epochStart))

        if f1_cls > best_f1:                          # select by macro-F1, not accuracy
            best_f1, best_acc, bestEpoch = f1_cls, acc_cls, epoch + 1
            torch.save(model, best_path)
            print(f"  -> New best model saved: {best_path} (Epoch {bestEpoch})")
        print("* best Macro-F1 {:.2f}% (Acc {:.2f}%, epoch {})".format(best_f1 * 100, best_acc * 100, bestEpoch))
        
        save_history(history, hist_prefix)

@torch.no_grad()
def val(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    for img1, img2, meta, tgt_cls in dataloader:
        out = model(img1.to(device), img2.to(device), meta.to(device))
        y_pred.extend(out.argmax(1).cpu().tolist())
        y_true.extend(tgt_cls.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2], zero_division=0)
    return acc, float(f1)

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-trd", "--train_root", default='/root/autodl-tmp/train.json', help="train json path")
    ap.add_argument("-ted", "--test_root",  default='/root/autodl-tmp/test.json', help="test json path")
    ap.add_argument("-b", "--batch_size", type=int, default=32, help="training batch size")
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    ap.add_argument("-e", "--epochs", type=int, default=200, help="no. of epochs")
    ap.add_argument("-im","--img-size", type=int, default=224, help="imagesize")
    ap.add_argument("-s","--seed", type=int, default=123, help="seed")
    ap.add_argument("-wp", "--weight-power", type=float, default=1.0,
                  help="1.0 = raw inverse-frequency weights, 0.5 = softened (sqrt)")
    ap.add_argument("-vr", "--val-ratio", type=float, default=0.15,
                    help="fraction of train.json held out for validation")
    ap.add_argument("--train_meta", default='/root/autodl-tmp/train_meta_processed.csv')
    args = vars(ap.parse_args())

    seed_all(args['seed'])
    train_root = args["train_root"]
    test_root  = args["test_root"]

    model = FusionModel(num_classes=3)
    model = model.to(device)
    print(model)

    train(model, train_root=train_root,
          batch_size=args["batch_size"], lr=args["learning_rate"], epochs=args["epochs"],
          weight_power=args["weight_power"], val_ratio=args["val_ratio"], seed=args["seed"], meta_csv=args["train_meta"])
