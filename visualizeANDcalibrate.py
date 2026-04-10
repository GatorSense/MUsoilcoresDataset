import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Load the GT data
df_gt = pd.read_csv("data_info/MNHT_rootmorphology_GT_v2.csv")
df_gt_val = pd.read_csv("data_info/CoresGT_val.csv")


# Load the prediction data
model_name = "unet100k"
df_pred = pd.read_csv(f"data_info/prediction_{model_name}.csv")
df_pred_val = pd.read_csv(f"data_info/prediction_{model_name}_val.csv")

# Keep only the cores that are in both datasets
df_gt = df_gt[df_gt["label ID"].isin(df_pred["soil_core"])].reset_index(drop=True)
df_gt_val = df_gt_val[df_gt_val["corename"].isin(df_pred_val["soil_core"])].reset_index(drop=True)
df_pred = df_pred[df_pred["soil_core"].isin(df_gt["label ID"])].reset_index(drop=True)
df_pred_val = df_pred_val[df_pred_val["soil_core"].isin(df_gt_val["corename"])].reset_index(drop=True)

feats_gt = ["Length(cm)", "ProjArea(cm2)", "SurfArea(cm2)", "AvgDiam(mm)", 
            "L00", "L10", "L20", "L30", "L40"]

feats_pred = ["L00", "L10", "L20", "L30", "L40"]

nrows = len(feats_pred)
ncols = len(feats_gt)

# Visualizations
plt.figure(figsize=(10, 10))
for j, feat_pred in enumerate(feats_pred):
    for i, feat_gt in enumerate(feats_gt):
        plt.subplot(nrows, ncols, j*ncols + i + 1)
        x, y = df_gt[feat_gt], df_pred[feat_pred]
        x_val, y_val = df_gt_val[feat_gt], df_pred_val[feat_pred]
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        x_val = (x_val - x_val.mean()) / x_val.std()
        y_val = (y_val - y_val.mean()) / y_val.std()
        plt.scatter(x, y, alpha=0.5, label="Test")
        plt.scatter(x_val, y_val, alpha=0.5, label="Validation")
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel(f"{feat_pred} Pred")
        if j == nrows - 1:
            plt.xlabel(f"{feat_gt} GT")
        # plt.legend()
        # plt.xlabel(f"{feat_gt} GT")
        # plt.ylabel(f"{feat_pred} Pred")
        # plt.title(f"{feat_gt} GT vs {feat_pred} Pred")


feat_gt = "Length(cm)"
feat_pred = "L00"
x, y = df_gt[feat_gt], df_pred[feat_pred]
x_val, y_val = df_gt_val[feat_gt], df_pred_val[feat_pred]
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()
x_val = (x_val - x_val.mean()) / x_val.std()
y_val = (y_val - y_val.mean()) / y_val.std()

plt.figure(figsize=(5, 5))
plt.scatter(x, y, alpha=0.5, label="Test")
plt.scatter(x_val, y_val, alpha=0.5, label="Validation")
plt.xlabel(f"{feat_gt} GT")
plt.ylabel(f"{feat_pred} Pred")
plt.title(f"{feat_gt} GT vs {feat_pred} Pred")
plt.legend()
plt.show()

