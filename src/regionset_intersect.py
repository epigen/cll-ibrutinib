import os
import pandas as pd
import pybedtools
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


cmd1 = "wget -O GSE100672_cll-ibrutinib.coverage.csv.gz \
      https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE100672&format=file&file=GSE100672_cll-ibrutinib.coverage.csv.gz"
cmd2 = "wget -O cll_peaks.coverage_qnorm.log2.annotated.tsv\
      http://www.medical-epigenomics.org/papers/rendeiro2016/data/cll_peaks.coverage_qnorm.log2.annotated.tsv"

for cmd in [cmd1, cmd2]:
    os.system(cmd)

new = pd.read_csv("GSE100672_cll-ibrutinib.coverage.csv.gz", index_col=0)
old = pd.read_csv("cll_peaks.coverage_qnorm.log2.annotated.tsv", sep="\t")

new['chrom'] = new.index.str.split(":").str[0]
tmp = new.index.str.split(":").str[1]
new['start'] = tmp.str.split("-").str[0]
new['end'] = tmp.str.split("-").str[1]
oldb = pybedtools.BedTool.from_dataframe(old[['chrom', 'start', 'end']])
newb = pybedtools.BedTool.from_dataframe(new[['chrom', 'start', 'end']])

intersection = oldb.intersect(newb).count()
a = oldb.intersect(newb, v=True).count()
b = newb.intersect(oldb, v=True).count()
print(intersection)
print(a)
print(b)


fig, axis = plt.subplots(1, figsize=(4, 4))
venn2(
    subsets=(b, a, intersection),
    set_labels=('This study', 'Rendeiro et al, 2016'), ax=axis)
fig.savefig("regionset_intersection.venn.svg", dpi=300, bbox_inches="tight")
