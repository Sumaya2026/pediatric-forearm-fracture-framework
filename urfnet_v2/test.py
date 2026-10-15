import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import argparse
import json
from dataset import Clsdata
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import random


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_root,  batch_size):
    
    with open(test_root, 'r') as fr:
         test_imgs = json.load(fr)
    test_data  = Clsdata(test_imgs,  train=False, shuffle=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_length = len(test_data)
    # Set to evaluation mode
    model.eval()
    test_acc_cls=0.0
    y_true=[]
    y_pred=[]

    # Validation loop
    for i, (img1, img2, tgt_cls) in enumerate(test_dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        tgt_cls = tgt_cls.to(device)
        y_true.extend(tgt_cls.cpu().numpy().tolist())

        # Forward pass - compute outputs on input data using the model
        pred_cls = model(img1,img2)
        # Calculate validation accuracy
        pred, predictions = torch.max(pred_cls.data, 1)

        y_pred.extend(predictions.cpu().numpy().tolist())
        correct_counts = predictions.eq(tgt_cls.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc_cls = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to valid_acc
        test_acc_cls += acc_cls.item() * img1.size(0)

    labels = [0, 1, 2]
    names = ["Proximal", "Midshaft", "Distal"]
    Pre = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    Rec = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    F1  = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)   # true macro-F1

    print(classification_report(y_true, y_pred, labels=labels, target_names=names,
                                digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=names, columns=names))



    return Acc, Pre, Rec, F1


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ted", "--test_root", default = '/root/autodl-tmp/test.json',
		help = "test image path")
    ap.add_argument("-b", "--batch_size", type=int, default=32,
		help="training batch size")
    args = vars(ap.parse_args())

    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test_root = args["test_root"]
    model_dir = "/root/autodl-tmp/pro"  
    model_path = os.path.join(model_dir, "URFNet.pth")
    model = torch.load(model_path, weights_only=False) 
    # Test the model
    Acc, Pre, Rec, F1 = test(model, test_root=test_root,  batch_size=args["batch_size"])
    print("URFNet: Acc {:.2f}%,Pre {:.2f}%, Rec: {:.2f}%, F1: {:.2f}%".format(Acc * 100, Pre * 100, Rec * 100, F1 * 100))



