import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def corr_str(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return "r=nan, ρ=nan"
    r = np.corrcoef(a, b)[0, 1]
    rho = np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1]
    return f"r={r:.2f}, ρ={rho:.2f}"

# prev_file = "unetr_soil_cores_summary_vertical.csv"
# cur_file = "prediction_unetr_val.csv"
prev_file = "segresnet_soil_cores_summary_vertical.csv"
cur_file = "prediction_segresnet_val.csv"


GT_VAL_PATH = "data_info/CoresGT_val.csv"


df_prev = pd.read_csv(os.path.join("data_info", "results_v1", prev_file))
df_cur = pd.read_csv(os.path.join("data_info", cur_file))
df_gt = pd.read_csv(GT_VAL_PATH)

feats_list = ["Root Length Diameter Range 1 (px)", "Root Length Diameter Range 2 (px)",
               "Root Length Diameter Range 3 (px)", "Root Length Diameter Range 4 (px)",
                 "Root Length Diameter Range 5 (px)"]

# Corresponding GT diameter-range columns, in the same order as feats_list
gt_feats_list = [
    "Sum of 0<.L.<=1.000000",
    "Sum of 1.0000000<.L.<=2.0000000",
    "Sum of 2.0000000<.L.<=3.0000000",
    "Sum of 3.0000000<.L.<=4.0000000",
    "Sum of .L.>4.0000000",
]

df_prev.set_index("soil_core", inplace=True)
df_cur.set_index("soil_core", inplace=True)
df_gt.set_index("corename", inplace=True)

l00 = "L00"

for feat, gt_feat in zip(feats_list, gt_feats_list):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(feat)

    # --- Prev vs Cur ---
    ax = axes[0, 0]
    common_pc = df_prev.index.intersection(df_cur.index)
    ax.scatter(df_prev.loc[common_pc, feat], df_cur.loc[common_pc, feat], alpha=0.5)
    for sc in common_pc:
        ax.annotate(sc, (df_prev.loc[sc, feat], df_cur.loc[sc, feat]), fontsize=7, alpha=0.7)
    lo, hi = df_prev.loc[common_pc, feat].min(), df_prev.loc[common_pc, feat].max()
    ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_xlabel("Previous")
    ax.set_ylabel("Current")
    ax.set_title(f"Previous vs Current  ({corr_str(df_prev.loc[common_pc, feat], df_cur.loc[common_pc, feat])})")
    ax.grid()

    # --- Prev vs GT ---
    ax = axes[0, 1]
    common_pg = df_prev.index.intersection(df_gt.index)
    ax.scatter(df_gt.loc[common_pg, gt_feat], df_prev.loc[common_pg, feat], alpha=0.5)
    for sc in common_pg:
        ax.annotate(sc, (df_gt.loc[sc, gt_feat], df_prev.loc[sc, feat]), fontsize=7, alpha=0.7)
    lo, hi = df_gt.loc[common_pg, gt_feat].min(), df_gt.loc[common_pg, gt_feat].max()
    ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_xlabel(f"GT ({gt_feat})")
    ax.set_ylabel("Previous")
    ax.set_title(f"Previous vs GT  ({corr_str(df_gt.loc[common_pg, gt_feat], df_prev.loc[common_pg, feat])})")
    ax.grid()

    # --- Cur vs GT ---
    ax = axes[0, 2]
    common_cg = df_cur.index.intersection(df_gt.index)
    ax.scatter(df_gt.loc[common_cg, gt_feat], df_cur.loc[common_cg, feat], alpha=0.5)
    for sc in common_cg:
        ax.annotate(sc, (df_gt.loc[sc, gt_feat], df_cur.loc[sc, feat]), fontsize=7, alpha=0.7)
    lo, hi = df_gt.loc[common_cg, gt_feat].min(), df_gt.loc[common_cg, gt_feat].max()
    ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_xlabel(f"GT ({gt_feat})")
    ax.set_ylabel("Current")
    ax.set_title(f"Current vs GT  ({corr_str(df_gt.loc[common_cg, gt_feat], df_cur.loc[common_cg, feat])})")
    ax.grid()

    # --- Prev vs GT L00 ---
    axes[1, 0].set_visible(False)

    ax = axes[1, 1]
    common_pg_l00 = df_prev.index.intersection(df_gt.index)
    ax.scatter(df_gt.loc[common_pg_l00, l00], df_prev.loc[common_pg_l00, feat], alpha=0.5)
    for sc in common_pg_l00:
        ax.annotate(sc, (df_gt.loc[sc, l00], df_prev.loc[sc, feat]), fontsize=7, alpha=0.7)
    lo, hi = df_gt.loc[common_pg_l00, l00].min(), df_gt.loc[common_pg_l00, l00].max()
    ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_xlabel("GT L00")
    ax.set_ylabel("Previous")
    ax.set_title(f"Previous vs GT L00  ({corr_str(df_gt.loc[common_pg_l00, l00], df_prev.loc[common_pg_l00, feat])})")
    ax.grid()

    # --- Cur vs GT L00 ---
    ax = axes[1, 2]
    common_cg_l00 = df_cur.index.intersection(df_gt.index)
    ax.scatter(df_gt.loc[common_cg_l00, l00], df_cur.loc[common_cg_l00, feat], alpha=0.5)
    for sc in common_cg_l00:
        ax.annotate(sc, (df_gt.loc[sc, l00], df_cur.loc[sc, feat]), fontsize=7, alpha=0.7)
    lo, hi = df_gt.loc[common_cg_l00, l00].min(), df_gt.loc[common_cg_l00, l00].max()
    ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_xlabel("GT L00")
    ax.set_ylabel("Current")
    ax.set_title(f"Current vs GT L00  ({corr_str(df_gt.loc[common_cg_l00, l00], df_cur.loc[common_cg_l00, feat])})")
    ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.5)

plt.show()