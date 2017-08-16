
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from looper.models import Project


# Set settings
pd.set_option("date_dayfirst", True)
sns.set(context="paper", style="white", palette="pastel", color_codes=True)
sns.set_palette(sns.color_palette("colorblind"))
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('text', usetex=False)

import sys
sys.setrecursionlimit(10000)

c = {
    "drug_alone": 0,
    "drug_plus_10nM_Ibrut": 10,
    "drug_plus_100nM_Ibrut": 100,
    "drug_plus_500nM_Ibrut": 500,
    "drug_plus_1µM_Ibrut": 1000,}

df = pd.read_csv(os.path.join("metadata", "drug_combinations.csv"))
df = df.sort_values(['patient_id', "drug_name", "exposure_time", "drug_concentration"])

# Melt dataframe
df = pd.melt(df, id_vars=['patient_id', 'exposure_time', 'drug_concentration_txt', 'drug_concentration', 'drug_name', 'drug_alias'], value_name="viability")
df['ibrutinib_concentration'] = [c[x] for x in df['variable']]
df.loc[df['drug_concentration_txt'] == "Control", "ibrutinib_concentration"] = 0
df = df[df['drug_concentration_txt'] != "Control"]
df = df.drop(['variable', 'drug_concentration_txt'], axis=1)
df = df.sort_values(['patient_id', "drug_name", "exposure_time", "ibrutinib_concentration", "drug_concentration", "viability"])
df.to_csv(os.path.join("metadata", "drug_combinations.melted.csv"), encoding="utf-8", index=False)


# Let's cap the drug concentrations to the maximum based on known potency
df = df[df['ibrutinib_concentration'] < 1000] # Ibrutinib < 1000uM
df = df[~((df['drug_name'] == "Navitoclax") & (df['drug_concentration'] > 500))] # Navitoclax < 500uM
df = df[~((df['drug_name'] == "Venetoclax") & (df['drug_concentration'] > 100))] # Navitoclax < 500uM


# Plot concentration curves
ibrutinib_concentrations = df['ibrutinib_concentration'].drop_duplicates()
patients = df['patient_id'].drop_duplicates()
drugs = df['drug_name'].drop_duplicates()
drugs = [
    "Bortezomib",
    "Carfilzomib",
    "MG132",
    "Venetoclax",
    "Navitoclax",
    "Dactolisib",
    "Idelalisib",
    "Fludarabine",
    "Zoledronic acid (Zometa)"]

for patient in patients:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
    fig.text(0.5, 1.02, "Patient {}".format(patient), fontsize=14, ha="center")
    for i, drug_name in enumerate(drugs):
        ax = axis.flatten()[i]
        for ibrutinib_concentration in ibrutinib_concentrations:
            df2 = df.loc[
                (df["patient_id"] == patient) &
                (df["drug_name"] == drug_name) &
                (df["ibrutinib_concentration"] == ibrutinib_concentration), :]
            ax.plot(df2['drug_concentration'], df2["viability"],
            label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
            linestyle="-", marker="o", zorder=i)

        if df2['drug_concentration'].max() >= 100:
            ax.set_xscale("symlog", basex=10, linthreshx=5)

        # Ibrutinib alone
        df3 = df.loc[
            (df["patient_id"] == patient) &
            (df["drug_name"] == drug_name) &
            (df["drug_concentration"] == 0), :]
        df3 = df3[df3['ibrutinib_concentration'] <= df2['drug_concentration'].max()]
        ax.plot(df3['ibrutinib_concentration'], df3["viability"], label="Ibrutinib alone",
            linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)

        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("% Viability")
        # ax.set_xlim((0, ax.get_xlim()[1]))
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_title(drug_name)
        ax.set_title(drug_name)
        if i == (n ** 2) - 1:
            ax.legend()

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_synergies.curves.patient_{}.svg".format(patient)), dpi=300, bbox_inches="tight")

# Concentration curves with patient means
df2 = df.groupby(['drug_name', 'drug_concentration', 'ibrutinib_concentration'])["viability"].mean().reset_index()
n = int(np.ceil(np.sqrt(len(drugs))))
fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
fig.text(0.5, 1.02, "Patient means", fontsize=14, ha="center")
for i, drug_name in enumerate(drugs):
    ax = axis.flatten()[i]
    for ibrutinib_concentration in ibrutinib_concentrations:
        df3 = df2.loc[
            (df2["drug_name"] == drug_name) &
            (df2["ibrutinib_concentration"] == ibrutinib_concentration), :]
        ax.plot(df3['drug_concentration'], df3["viability"],
            label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
            linestyle="-", marker="o", zorder=i)

    # Ibrutinib alone
    df4 = df2.loc[
        (df2["drug_name"] == drug_name) &
        (df2["drug_concentration"] == 0), :]
    df4 = df4[df4['ibrutinib_concentration'] <= df3['drug_concentration'].max()]
    ax.plot(df4['ibrutinib_concentration'], df4["viability"], label="Ibrutinib alone",
        linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)

    if df3['drug_concentration'].max() >= 100:
        ax.set_xscale("symlog", basex=10, linthreshx=5)
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("% Viability")
    ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_title(drug_name)

    if i == (n ** 2) - 1:
        ax.legend()
sns.despine(fig)
fig.tight_layout()
fig.savefig(os.path.join("results", "drug_synergies.curves.all_patients_mean.svg"), dpi=300, bbox_inches="tight")


# Synnergy plots
df["phenotype"] = 100 - df['viability']

for patient in patients:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
    fig.text(0.5, 1.02, "Patient {}".format(patient), fontsize=14, ha="center")
    fig.text(0.5, -.02, "Drug concentration (nM)", fontsize=14, ha="center")
    fig.text(-.02, 0.5, "Ibrutinib concentration (nM)", fontsize=14, ha="center", rotation=90)
    for i, drug_name in enumerate(drugs):
        ax = axis.flatten()[i]
        df2 = pd.pivot_table(data=df.loc[
            (df["patient_id"] == patient) &
            (df["drug_name"] == drug_name), :],
            index="ibrutinib_concentration", columns="drug_concentration", values="phenotype")
        sns.heatmap(data=df2, ax=ax, cmap="RdBu_r", cbar_kws={"label": "% killing"},
            vmin=-100, vmax=100,
            label="drug + {}nM Ibrutinib".format(ibrutinib_concentration))
        ax.set_title(drug_name)
        ax.set_xlabel(ax.get_xlabel(), visible=False)
        ax.set_ylabel(ax.get_ylabel(), visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_synergies.heatmaps.patient_{}.svg".format(patient)), dpi=300, bbox_inches="tight")

# Heatmaps with patient means
fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
fig.text(0.5, 1.02, "Patient means", fontsize=14, ha="center")
fig.text(0.5, -.02, "Drug concentration (nM)", fontsize=14, ha="center")
fig.text(-.02, 0.5, "Ibrutinib concentration (nM)", fontsize=14, ha="center", rotation=90)
for i, drug_name in enumerate(drugs):
    ax = axis.flatten()[i]
    df2 = pd.pivot_table(data=df.loc[
        (df["drug_name"] == drug_name), :],
        index="ibrutinib_concentration", columns="drug_concentration", values="phenotype")
    sns.heatmap(data=df2, ax=ax, cmap="RdBu_r", cbar_kws={"label": "% killing"},
        vmin=-100, vmax=100,
        label="drug + {}nM Ibrutinib".format(ibrutinib_concentration))
    ax.set_title(drug_name)
    ax.set_xlabel(ax.get_xlabel(), visible=False)
    ax.set_ylabel(ax.get_ylabel(), visible=False)
fig.tight_layout()
fig.savefig(os.path.join("results", "drug_synergies.heatmaps.all_patients_mean.svg"), dpi=300, bbox_inches="tight")


# Calculate AUCs
auc = (
        df
        .groupby(['patient_id', "drug_name", "exposure_time", "ibrutinib_concentration"])
        .apply(
            lambda x: 
                np.trapz(x["viability"], x=x["drug_concentration"]))).to_frame(name="auc").reset_index()
auc.to_csv(os.path.join("results", "drug_synergies.AUCs.csv"), encoding="utf-8", index=False)

# Calculate AUC fold-change
f =  (
        auc
        .groupby(['patient_id', "drug_name", "exposure_time"])
        .apply(
            lambda x: 
                np.log2(x['auc'] / x.loc[x['ibrutinib_concentration'] == 0, 'auc'].squeeze()).squeeze()
        )
    ).to_frame(name="auc_fold_change").reset_index().set_index('level_3')[['auc_fold_change']]
auc = auc.join(f)
auc.to_csv(os.path.join("results", "drug_synergies.AUCs.csv"), encoding="utf-8", index=False)


# Plot AUC fold-change heatmaps for the various Ibrutinib concentrations for each patient and mean of patients
fig, axis = plt.subplots(1, 4, sharey=True, figsize=(4, 1.25))
for i, patient in enumerate(patients):
    ax = axis[i]
    df2 = auc[auc["patient_id"] == patient]
    p = pd.pivot_table(df2, index="drug_name", columns="ibrutinib_concentration", values="auc_fold_change")
    sns.heatmap(p, ax=ax, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
        cbar_kws={"label": "log2 fold-change of AUC\nover Control (Ibrutinib alone)"},
        xticklabels=(p.columns.astype(str) + "uM").tolist())
    ax.set_title(patient)
df2 = auc.groupby(['drug_name', 'exposure_time', 'ibrutinib_concentration'])['auc_fold_change'].mean().reset_index()
p = pd.pivot_table(df2, index="drug_name", columns="ibrutinib_concentration", values="auc_fold_change")
sns.heatmap(p, ax=axis[3], cmap="RdBu_r", vmin=-0.5, vmax=0.5,
        cbar_kws={"label": "log2 fold-change of AUC\nover Control (Ibrutinib alone)"},
        xticklabels=(p.columns.astype(str) + "uM").tolist())
axis[3].set_title("Mean")
for ax in axis:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
fig.savefig(os.path.join("results", "drug_synergies.AUC_fold_change.heatmap.svg"), dpi=300, bbox_inches="tight")


# Plot AUC fold-change heatmaps for 500uM Ibrutinib for each patient and mean of patients
df2 = auc[auc["ibrutinib_concentration"] == 500]
p = pd.pivot_table(df2, index="drug_name", columns="patient_id", values="auc_fold_change")
p = p.join(df2.groupby(['drug_name'])['auc_fold_change'].mean().to_frame(name="mean"))
g = sns.clustermap(p.T, cmap="RdBu_r", row_cluster=False, col_cluster=True,
    cbar_kws={"label": "log2 fold-change of AUC\nover Control (Ibrutinib alone)"},
    vmin=-0.5, vmax=0.5, figsize=(8, 4), square=True)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, ha="left")
g.savefig(os.path.join("results", "drug_synergies.AUC_fold_change.500uM_ibrutinib.clustermap.svg"), dpi=300, bbox_inches="tight")




library("reshape")
library("synergyfinder")
df = read.csv("drug_combinations.melted.csv", sep=",")

df = df[df$ibrutinib_concentration < 1000, ]

all_scores = data.frame()
for (patient in unique(df$patient_id)){
    for (drug in unique(df$drug_name)){
        print(c(patient, drug))
        df2 = df[
            (df$patient_id == patient) & 
            (df$drug_name == drug),
        ]
        # pivot
        df2 = as.matrix(cast(df2, ibrutinib_concentration ~ drug_concentration, value="viability"))

        # Calculate all scores
        res = try(Bliss(100 - df2))
        if (inherits(res, "try-error")) {
            print(c("FAILED:", patient, drug))
        } else {
            bliss = melt(as.data.frame(res))
            bliss$patient = patient
            bliss$drug = drug
            bliss$test = "Bliss"
            all_scores = rbind(all_scores, bliss)
        }
        res = try(ZIP(100 - df2))
        if (inherits(res, "try-error")) {
            print(c("FAILED:", patient, drug))
        } else {
            zip = as.data.frame(res)
            zip$ibrutinib_concentration = rownames(zip)
            zip = melt(zip)
            zip$patient = patient
            zip$drug = drug
            zip$test = "ZIP"
            colnames(zip)[2] <- "drug_concentration"
            all_scores = rbind(all_scores, zip)
        }
        res = try(Loewe(100 - df2))
        if (inherits(res, "try-error")) {
            print(c("FAILED:", patient, drug))
        } else {
            # loewe = melt(as.data.frame(res))
            # loewe$patient = patient
            # loewe$drug = drug
            # loewe$test = "Loewe"
            # all_scores = rbind(all_scores, loewe)
        }
        res = try(HSA(100 - df2))
        if (inherits(res, "try-error")) {
            print(c("FAILED:", patient, drug))
        } else {
            hsa = melt(as.data.frame(res))
            hsa$patient = patient
            hsa$drug = drug
            hsa$test = "HSA"
            all_scores = rbind(all_scores, hsa)
        }
    }
}
write.csv(all_scores, "drug_combinations.synergy_scores.csv", sep=",", row.names=FALSE)


from scipy.interpolate import griddata
scores = pd.read_csv(os.path.join("results", "drug_combinations.synergy_scores.csv"))

for score in scores['test'].drop_duplicates():
    for patient in patients:
        n = int(np.ceil(np.sqrt(len(drugs))))
        fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
        fig.text(0.5, 1.02, "Patient {}".format(patient), fontsize=14, ha="center")
        fig.text(0.5, -.02, "Drug concentration (nM)", fontsize=14, ha="center")
        fig.text(-.02, 0.5, "Ibrutinib concentration (nM)", fontsize=14, ha="center", rotation=90)
        for i, drug_name in enumerate(drugs):
            ax = axis.flatten()[i]
            df2 = pd.pivot_table(data=scores.loc[
                (scores["test"] == score) &
                (scores["patient"] == patient) &
                (scores["drug"] == drug_name), :],
                index="ibrutinib_concentration", columns="drug_concentration", values="value")

            # Interpolate unstructured D-dimensional data.
            x1 = np.linspace(df2.index.min(), df2.index.max(), 100)
            y1 = np.linspace(df2.columns.min(), df2.columns.max(), 100)
            x2, y2 = np.meshgrid(x1, y1)
            points = np.array([[x, y] for y in df2.columns.tolist() for x in df2.index.tolist()])
            z2 = griddata(points, df2.values.flatten(), (x2, y2), method='cubic')
            ax.imshow(z2, interpolation="bilinear", cmap="RdBu_r")
            # sns.heatmap(data=df2, ax=ax, cbar_kws={"label": "{} score".format(score)})
            s = np.round(df2.values.mean(), 3)
            ax.set_title(drug_name + "; score={}".format(s))
            ax.set_xlabel(ax.get_xlabel(), visible=False)
            ax.set_ylabel(ax.get_ylabel(), visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join("results", "drug_synergies.{}_score.patient_{}.heatmaps.svg".format(score, patient)), dpi=300, bbox_inches="tight")

# Heatmap with patient interaction scores
s = scores.groupby(['patient', 'drug', 'test'])['value'].mean().reset_index()

fig, axis = plt.subplots(1, 3, figsize=(6, 2), sharex=True, sharey=True)
for i, test in enumerate(s['test'].drop_duplicates()):
    p = pd.pivot_table(s[s['test'] == test], index="patient", columns="drug", values="value")
    p = p.T.join(p.mean(axis=0).to_frame(name="mean")).T

    ax = axis[i]
    sns.heatmap(p, square=True, ax=ax)
    ax.set_title(test + " score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.savefig(os.path.join("results", "drug_synergies.mean_synergy_scores.all_patients_mean.svg"), dpi=300, bbox_inches="tight")
