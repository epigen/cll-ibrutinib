
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# Set settings
pd.set_option("date_dayfirst", True)
sns.set(context="paper", style="white", palette="pastel", color_codes=True)
sns.set_palette(sns.color_palette("colorblind"))
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('text', usetex=False)

df = pd.read_csv(os.path.join("data", "culture_method_comparison.melted.csv"))
df = df.rename(columns={"value": "Apoptosis"})

df2 = df[df['variable'].str.endswith("/Ax+")]

# c = len(df2['culture'].unique())
# fig, axis = plt.subplots(c, 1, figsize=(8, c * 4))
# for i, culture in enumerate(df2['culture'].unique()):
#     sns.barplot(data=df2[df2['culture'] == culture], x="treatment", y="Apoptosis", hue="variable", ax=axis[i])
#     axis[i].set_xticklabels(axis[i].get_xticklabels(), rotation=90)
#     axis[i].set_ylim((0, 100))
#     axis[i].set_title(culture)
# sns.despine(fig)
# fig.tight_layout()
# fig.savefig(os.path.join("results", "culture_method_comparison.barplot.svg"), dpi=300, bbox_inches="tight")

# c = len(df2['culture'].unique())
# fig, axis = plt.subplots(c, 1, figsize=(8, c * 4))
# for i, culture in enumerate(df2['culture'].unique()):
#     sns.barplot(data=df2[df2['culture'] == culture], x="variable", y="Apoptosis", hue="treatment", ax=axis[i])
#     axis[i].set_xticklabels(axis[i].get_xticklabels(), rotation=90)
#     axis[i].set_ylim((0, 100))
#     axis[i].set_title(culture)
# sns.despine(fig)
# fig.tight_layout()
# fig.savefig(os.path.join("results", "culture_method_comparison.treat.barplot.svg"), dpi=300, bbox_inches="tight")


grid = sns.factorplot(data=df2, x="treatment", y="Apoptosis", hue="variable", row="culture", sharex=False)
for ax in grid.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim((0, 100))
sns.despine(grid.fig)
grid.fig.tight_layout()
grid.fig.savefig(
    os.path.join("results", "culture_method_comparison.factorplot.svg"),
    dpi=300, bbox_inches="tight")

grid = sns.factorplot(
    data=df[(~df['culture'].str.contains("alone")) & (df['variable'] == "All Ax+")],
    x="treatment", y="Apoptosis", hue="culture", sharex=False)
for ax in grid.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim((0, 100))
sns.despine(grid.fig)
grid.fig.tight_layout()
grid.fig.savefig(
    os.path.join("results", "culture_method_comparison.killing.factorplot.svg"),
    dpi=300, bbox_inches="tight")


grid = sns.factorplot(
    data=df[df['culture'].str.contains("alone")],
    x="treatment", y="Apoptosis", hue="variable", row="culture", sharex=False)
for ax in grid.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim((0, 100))
sns.despine(grid.fig)
grid.fig.tight_layout()
grid.fig.savefig(
    os.path.join("results", "culture_method_comparison.alone.factorplot.svg"),
    dpi=300, bbox_inches="tight")

grid = sns.factorplot(
    data=df[(df['culture'].str.contains("alone")) & (~df['variable'].str.endswith("-"))],
    x="treatment", y="Apoptosis", hue="variable", row="culture", sharex=False)
for ax in grid.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim((0, 20))
sns.despine(grid.fig)
grid.fig.tight_layout()
grid.fig.savefig(
    os.path.join("results", "culture_method_comparison.alone.killing.factorplot.svg"),
    dpi=300, bbox_inches="tight")


fig, axis = plt.subplots(1, 1, figsize=(4, 1 * 4))
sns.violinplot(
    data=df[(~df['culture'].str.contains("alone")) & (df['treatment'] == "Control")],
    x="variable", y="Apoptosis", hue="culture", ax=axis)
sns.despine(fig)
fig.tight_layout()
fig.savefig(
    os.path.join("results", "culture_method_comparison.control_only.violinplot.svg"),
    dpi=300, bbox_inches="tight")

df3 = df[
    (~df['culture'].str.contains("alone")) &
    (~df['treatment'].str.contains("1|10|100|1000")) &
    (~df['treatment'].str.contains(r"\+"))]

# grid = sns.FacetGrid(data=df3, row='variable', col=None, hue="treatment", col_wrap=None, sharex=True, sharey=False, hue_order=df3['treatment'].unique())
# grid.map(sns.violinplot, "culture", "Apoptosis")
# grid.map(sns.swarmplot, "culture", "Apoptosis")
grid = sns.factorplot(
    data=df3, x="culture", y="Apoptosis", hue="treatment", row="variable",
    sharey=False, order=['suspension', 'NKTert co-culture', 'pBMSC co-culture'], hue_order=df3['treatment'].unique())
for ax in grid.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
grid.axes.flatten()[-1].set_ylim((0, 100))
#grid.add_legend(title="Treatment")
sns.despine(grid.fig)
grid.fig.tight_layout()
grid.fig.savefig(
    os.path.join("results", "culture_method_comparison.suspension&alone.factorplot.svg"),
    dpi=300, bbox_inches="tight")
