
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from looper.models import Project


# Set settings
pd.set_option("date_dayfirst", True)
sns.set(context="paper", style="white", palette="pastel", color_codes=True)
sns.set_palette(sns.color_palette("colorblind"))
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('text', usetex=False)


df = pd.read_csv(os.path.join("data", "proteasome_assay.csv"))
df_melt = pd.melt(df, id_vars=["patient_id", "sample_id", "time", "ibrutinib_concentration", "bortezomib_concentration", "condition"])

g = sns.factorplot(data=df_melt, x="condition", y="value", col="time", hue="variable", estimator=np.mean, legend_out=True, sharey=True)
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
g.savefig(os.path.join("results", "proteasome_assay.all_data.svg"), dpi=300, bbox_inches="tight")


g = sns.factorplot(data=df_melt, x="condition", y="value", col="time", row="patient_id", hue="variable", estimator=np.mean, legend_out=True, sharey=True)
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
g.savefig(os.path.join("results", "proteasome_assay.per_patient.svg"), dpi=300, bbox_inches="tight")
