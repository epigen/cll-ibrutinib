
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

df = pd.read_csv(os.path.join("data", "drug_combinations.coculture.melted.csv"))
df = df.sort_values(['drug', "patient_id", "ibrutinib_concentration_nM", "drug_concentration_nM"])

# Let's cap the drug concentrations to the maximum based on known potency
df = df[df['ibrutinib_concentration_nM'] < 1000] # Ibrutinib < 1000uM
# df = df[~((df['drug'] == "Navitoclax") & (df['drug_concentration_nM'] > 500))] # Navitoclax < 500uM
# df = df[~((df['drug'] == "Venetoclax") & (df['drug_concentration_nM'] > 100))] # Navitoclax < 500uM


# Plot concentration curves
ibrutinib_concentrations = df['ibrutinib_concentration_nM'].drop_duplicates()
patients = df['patient_id'].drop_duplicates()
drugs = df['drug'].drop_duplicates()
# drugs = [
#     "Bortezomib",
#     "Carfilzomib",
#     "MG132",
#     "Venetoclax",
#     "Navitoclax",
#     "Dactolisib",
#     "Idelalisib",
#     "Fludarabine",
#     "Zoledronic acid (Zometa)"]

for patient in patients:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
    fig.text(0.5, 1.02, "Patient {}".format(patient), fontsize=14, ha="center")
    for i, drug in enumerate(drugs):
        ax = axis.flatten()[i]
        for ibrutinib_concentration in ibrutinib_concentrations:
            df2 = df.loc[
                (df["patient_id"] == patient) &
                (df["drug"] == drug) &
                (df["ibrutinib_concentration_nM"] == ibrutinib_concentration), :]
            ax.plot(df2['drug_concentration_nM'], df2["viability"],
            label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
            linestyle="-", marker="o", zorder=i)

        if df2['drug_concentration_nM'].max() >= 100:
            ax.set_xscale("symlog", basex=10, linthreshx=5)

        # Ibrutinib alone
        df3 = df.loc[
            (df["patient_id"] == patient) &
            (df["drug"] == drug) &
            (df["drug_concentration_nM"] == 0), :]
        df3 = df3[df3['ibrutinib_concentration_nM'] <= df2['drug_concentration_nM'].max()]
        ax.plot(df3['ibrutinib_concentration_nM'], df3["viability"], label="Ibrutinib alone",
            linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)

        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("% Viability")
        # ax.set_xlim((0, ax.get_xlim()[1]))
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_title(drug)
        ax.set_title(drug)
        if i == (n ** 2) - 1:
            ax.legend()

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_combinations.curves.patient_{}.svg".format(patient)), dpi=300, bbox_inches="tight")

# Concentration curves with patient means
df2 = df.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'])["viability"].mean().reset_index()
df2 = df2.join(df.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'])["viability"].count().to_frame(name="n").reset_index()[['n']])
df2 = df2.join(df.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'])["viability"].apply(scipy.stats.sem).reset_index()['viability'], rsuffix="_sem")

df2['upper_sem'] = df2['viability'] + df2['viability_sem'] / 2.
df2['lower_sem'] = df2['viability'] - df2['viability_sem'] / 2.

# df2 = pd.melt(df, id_vars=['patient_id', 'drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'], value_vars="viability", value_name='viability')
# g = sns.factorplot(data=df2, x="drug_concentration_nM", y="viability", col="drug", col_wrap=3, sharex=False)


for log, scale in [(True, "log"), (False, "linear")]:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False)
    fig.text(0.5, 1.02, "Patient mean + std", fontsize=14, ha="center")
    for i, drug in enumerate(drugs):
        ax = axis.flatten()[i]
        for ibrutinib_concentration in ibrutinib_concentrations:
            df3 = df2.loc[
                (df2["drug"] == drug) &
                (df2["ibrutinib_concentration_nM"] == ibrutinib_concentration), :]

            # mean
            ax.plot(df3['drug_concentration_nM'], df3["viability"],
                label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
                linestyle="-", marker="o", zorder=i)
            # stderror bars
            for _, row in df3.iterrows():
                ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                    linewidth=1, color="black")

        # Ibrutinib alone
        df4 = df2.loc[
            (df2["drug"] == drug) &
            (df2["drug_concentration_nM"] == 0), :]
        df4 = df4[df4['ibrutinib_concentration_nM'] <= df3['drug_concentration_nM'].max()]
        ax.plot(df4['ibrutinib_concentration_nM'], df4["viability"], label="Ibrutinib alone",
            linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)
        # stderror bars
        for _, row in df4.iterrows():
            ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                linewidth=1, color="black")

        if log:
            ax.set_xscale("symlog", basex=10, linthreshx=5)
        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("% Viability")
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_title(drug + "\n(n = {})".format(list(set((df4['n'])))[0]), ha="center")
        ax.legend()

        if i == (n ** 2) - 1:
            ax.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_combinations.curves.all_patients_mean.{}.svg".format(scale)), dpi=300, bbox_inches="tight")



# Drugs from 1st submission
c = {
    "drug_alone": 0,
    "drug_plus_10nM_Ibrut": 10,
    "drug_plus_100nM_Ibrut": 100,
    "drug_plus_500nM_Ibrut": 500,
    "drug_plus_1µM_Ibrut": 1000,}

df = pd.read_csv(os.path.join("metadata", "drug_combinations.csv"))
df = df.rename(columns=c).rename(columns={"drug_name": "drug", "drug_concentration": "drug_concentration_nM"})
df = pd.melt(df, id_vars=[i for i in df.columns if i not in c.values()], var_name="ibrutinib_concentration_nM", value_name="viability")
df = df.sort_values(['drug', "patient_id", "ibrutinib_concentration_nM", "drug_concentration_nM"])

# Let's cap the drug concentrations to the maximum based on known potency
df = df[df['ibrutinib_concentration_nM'] < 1000] # Ibrutinib < 1000uM
# df = df[~((df['drug'] == "Navitoclax") & (df['drug_concentration_nM'] > 500))] # Navitoclax < 500uM
# df = df[~((df['drug'] == "Venetoclax") & (df['drug_concentration_nM'] > 100))] # Navitoclax < 500uM


# Concentration curves with patient means
df2 = df.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'])["viability"].mean().reset_index()
df2 = df2.join(df.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'])["viability"].count().to_frame(name="n").reset_index()[['n']])
df2 = df2.join(df.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM'])["viability"].apply(scipy.stats.sem).reset_index()['viability'], rsuffix="_sem")

df2['upper_sem'] = df2['viability'] + df2['viability_sem'] / 2.
df2['lower_sem'] = df2['viability'] - df2['viability_sem'] / 2.


# Plot concentration curves
ibrutinib_concentrations = df2['ibrutinib_concentration_nM'].drop_duplicates()
patients = df['patient_id'].drop_duplicates()
drugs =  ["Fludarabine"] # df['drug'].drop_duplicates()


for log, scale in [(True, "log"), (False, "linear")]:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False, squeeze=False)
    fig.text(0.5, 1.02, "Patient mean + std", fontsize=14, ha="center")
    for i, drug in enumerate(drugs):
        ax = axis.flatten()[i]
        for ibrutinib_concentration in ibrutinib_concentrations:
            df3 = df2.loc[
                (df2["drug"] == drug) &
                (df2["ibrutinib_concentration_nM"] == ibrutinib_concentration), :]

            # mean
            ax.plot(df3['drug_concentration_nM'], df3["viability"],
                label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
                linestyle="-", marker="o", zorder=i)
            # stderror bars
            for _, row in df3.iterrows():
                ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                    linewidth=1, color="black")

        # Ibrutinib alone
        df4 = df2.loc[
            (df2["drug"] == drug) &
            (df2["drug_concentration_nM"] == 0), :]
        df4 = df4[df4['ibrutinib_concentration_nM'] <= df3['drug_concentration_nM'].max()]
        ax.plot(df4['ibrutinib_concentration_nM'], df4["viability"], label="Ibrutinib alone",
            linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)
        # stderror bars
        for _, row in df4.iterrows():
            ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                linewidth=1, color="black")

        if log:
            ax.set_xscale("symlog", basex=10, linthreshx=5)
        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("% Viability")
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_title(drug + "\n(n = {})".format(list(set((df4['n'])))[0]), ha="center")
        ax.legend()

        if i == (n ** 2) - 1:
            ax.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_combinations.20170816.curves.all_patients_mean.{}.fludarabine_only.svg".format(scale)), dpi=300, bbox_inches="tight")

# values normalized to ibrutinib
df3 = df2.copy()
for group, i in df3.groupby(['drug', 'drug_concentration_nM', 'ibrutinib_concentration_nM']).groups.items():
    d = (
        df2.loc[i, ['viability', 'upper_sem', 'lower_sem']].values -
        df2.loc[
            (df2['drug'] == group[0]) &
            (df2['drug_concentration_nM'] == 0) &
            (df2['ibrutinib_concentration_nM'] == group[2]).values,
            ['viability', 'upper_sem', 'lower_sem']].values)
    df3.loc[i, ['viability', 'upper_sem', 'lower_sem']] = 100 + d
to_plot = df3

ibrutinib_concentrations = to_plot['ibrutinib_concentration_nM'].drop_duplicates()
drugs =  ["Fludarabine"] # to_plot['drug'].drop_duplicates()

for log, scale in [(True, "log"), (False, "linear")]:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False, squeeze=False)
    fig.text(0.5, 1.02, "Patient mean + std", fontsize=14, ha="center")
    for i, drug in enumerate(drugs):
        ax = axis.flatten()[i]
        for ibrutinib_concentration in ibrutinib_concentrations:
            p = to_plot.loc[
                (to_plot["drug"] == drug) &
                (to_plot["ibrutinib_concentration_nM"] == ibrutinib_concentration), :]

            # mean
            ax.plot(p['drug_concentration_nM'], p["viability"],
                label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
                linestyle="-", marker="o", zorder=i)
            # stderror bars
            for _, row in p.iterrows():
                ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                    linewidth=1, color="black")

        # Ibrutinib alone
        pp = to_plot.loc[
            (to_plot["drug"] == drug) &
            (to_plot["drug_concentration_nM"] == 0), :]
        pp = pp[pp['ibrutinib_concentration_nM'] <= p['drug_concentration_nM'].max()]
        ax.plot(pp['ibrutinib_concentration_nM'], pp["viability"], label="Ibrutinib alone",
            linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)
        # stderror bars
        for _, row in pp.iterrows():
            ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                linewidth=1, color="black")

        if log:
            ax.set_xscale("symlog", basex=10, linthreshx=5)
        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("% Viability")
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_title(drug + "\n(n = {})".format(list(set((pp['n'])))[0]), ha="center")
        ax.legend()

        if i == (n ** 2) - 1:
            ax.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_combinations.20170816.curves.all_patients_mean.ibrutinib_normalized.{}.fludarabine_only.svg".format(scale)), dpi=300, bbox_inches="tight")


# New values for Fludarabine, normalized to ibrutinib by Medhat

df = pd.read_csv(os.path.join("data", "drug_combinations.coculture.only_fludarabine_ibrut_normalized.melted.csv"))

to_plot = df.groupby(df.columns[1:-1].tolist()).mean().reset_index()
to_plot = to_plot.join(df.groupby(df.columns[1:-1].tolist())["viability"].count().to_frame(name="n").reset_index()[['n']])
to_plot = to_plot.join(df.groupby(df.columns[1:-1].tolist())["viability"].apply(scipy.stats.sem).reset_index()['viability'], rsuffix="_sem")

to_plot['upper_sem'] = to_plot['viability'] + to_plot['viability_sem'] / 2.
to_plot['lower_sem'] = to_plot['viability'] - to_plot['viability_sem'] / 2.
ibrutinib_concentrations = to_plot['ibrutinib_concentration_nM'].drop_duplicates()
drugs =  ["Fludarabine"] # to_plot['drug'].drop_duplicates()

for log, scale in [(True, "log"), (False, "linear")]:
    n = int(np.ceil(np.sqrt(len(drugs))))
    fig, axis = plt.subplots(n, n, figsize=(n * 3.5, n * 3.5), sharex=False, sharey=False, squeeze=False)
    fig.text(0.5, 1.02, "Patient mean + std", fontsize=14, ha="center")
    for i, drug in enumerate(drugs):
        ax = axis.flatten()[i]
        for ibrutinib_concentration in ibrutinib_concentrations:
            p = to_plot.loc[
                (to_plot["drug"] == drug) &
                (to_plot["ibrutinib_concentration_nM"] == ibrutinib_concentration), :]

            # mean
            ax.plot(p['drug_concentration_nM'], p["viability"],
                label="drug + {}nM Ibrutinib".format(ibrutinib_concentration),
                linestyle="-", marker="o", zorder=i)
            # stderror bars
            for _, row in p.iterrows():
                ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                    linewidth=1, color="black")

        # Ibrutinib alone
        pp = to_plot.loc[
            (to_plot["drug"] == drug) &
            (to_plot["drug_concentration_nM"] == 0), :]
        pp = pp[pp['ibrutinib_concentration_nM'] <= p['drug_concentration_nM'].max()]
        ax.plot(pp['ibrutinib_concentration_nM'], pp["viability"], label="Ibrutinib alone",
            linestyle="--", marker="o", color="black", alpha=0.75, zorder=0)
        # stderror bars
        for _, row in pp.iterrows():
            ax.plot((row['drug_concentration_nM'], row['drug_concentration_nM']), (row["lower_sem"], row["upper_sem"]),
                linewidth=1, color="black")

        if log:
            ax.set_xscale("symlog", basex=10, linthreshx=5)
        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("% Viability")
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_title(drug + "\n(n = {})".format(list(set((pp['n'])))[0]), ha="center")
        ax.legend()

        if i == (n ** 2) - 1:
            ax.legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "drug_combinations.curves.all_patients_mean.ibrutinib_normalized.{}.fludarabine_only.medhat_values.20180817.svg".format(scale)), dpi=300, bbox_inches="tight")

