
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from ngs_toolkit.general import signed_max
from scipy.stats import gaussian_kde
from scipy.stats import zscore
import matplotlib.pyplot as plt


# Set settings
pd.set_option("date_dayfirst", True)
sns.set(context="paper", style="white", palette="pastel", color_codes=True)
sns.set_palette(sns.color_palette("colorblind"))
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('text', usetex=False)


df = pd.read_csv(os.path.join("metadata", "pharmacoscopy.toxicity_vs_sensitivity.res.joined.2.a.compare.csv"), index_col=1)
df['diff_tcn'] = scipy.stats.zscore(df['diff_tcn'])
df['diff'] = scipy.stats.zscore(df['diff_gfp']) - df['diff_tcn']


df[['diff_tcn', "diff_gfp"]].corr()

n_top = 5
to_label = [
    "Bortezomib",
    "Carfilzomib",
    "BI 2536",
    "ABT-263 = Navitoclax",
    "Everolimus",
    "Volasertib"
]

fig, axis = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3), sharex=False, sharey=False)
kernel = gaussian_kde(df[['diff_tcn', "diff_gfp"]].T, bw_method='silverman')
density = kernel(df[['diff_tcn', "diff_gfp"]].T)
sc = axis[0].scatter(
    df['diff_tcn'], df["diff_gfp"],
    s=5, alpha=0.8, c=density, cmap="copper")
# plt.colorbar(sc, ax=axis[2])
already = list()
for variable in ['diff_tcn', 'diff_gfp']:
    for t in df.sort_values(variable).head(n_top).index:
        if t not in already:
            axis[0].text(df.loc[t, 'diff_tcn'], df.loc[t, "diff_gfp"], s=t, fontsize=4)
            already.append(t)
    for t in df.sort_values(variable).tail(n_top).index:
        if t not in already:
            axis[0].text(df.loc[t, 'diff_tcn'], df.loc[t, "diff_gfp"], s=t, fontsize=4)
            already.append(t)
    for t in to_label:
        if t not in already:
            axis[0].text(df.loc[t, 'diff_tcn'], df.loc[t, "diff_gfp"], s=t, fontsize=4)
            already.append(t)

sc = axis[1].scatter(
    df['diff'].rank(ascending=False), df['diff'],
    s=5, alpha=0.8, c=df['diff'], cmap='coolwarm', vmin=-df['diff'].abs().max(), vmax=df['diff'].abs().max())
# plt.colorbar(sc, ax=axis[2])
already = list()
for t in df['diff'].sort_values().head(n_top).index:
    if t not in already:
        axis[1].text(df['diff'].rank(ascending=False).loc[t], df['diff'].loc[t], s=t, fontsize=4, ha="right")
        already.append(t)
for t in df['diff'].sort_values().tail(n_top).index:
    if t not in already:
        axis[1].text(df['diff'].rank(ascending=False).loc[t], df['diff'].loc[t], s=t, fontsize=4, ha="left")
        already.append(t)
for t in to_label:
    if t not in already:
        axis[1].text(df['diff'].rank(ascending=False).loc[t], df['diff'].loc[t], s=t, fontsize=4, ha="left")
        already.append(t)
axis[0].set_xlabel("Change in toxicity upon ibrutinib (Z-score of total cell killing)")
axis[0].set_ylabel("Change in specificity (differential CLL cell killing) upon ibrutinib \n")
axis[1].set_xlabel("Specificity / Toxicity (rank)")
axis[1].set_ylabel("Specificity / Toxicity")
axis[0].axhline(0, color="black", linestyle="--", alpha=0.5, zorder=-100)
axis[1].axhline(0, color="black", linestyle="--", alpha=0.5, zorder=-100)
sns.despine(fig)
fig.tight_layout()
fig.savefig(
    os.path.join(
        "results",
        "pharmacoscopy.toxicity_vs_sensitivity.scatter_rank.20180811.svg"),
    dpi=300, bbox_inches="tight")
