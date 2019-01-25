#!/usr/bin/env python

import os
import pandas as pd
import pybedtools
import re


def get_tss(x):
    if x['strand'] == "+":
        x['end'] = x['start'] + 1
    elif x['strand'] == "-":
        x['start'] = x['end'] - 1
    return x


def get_promoter(x, radius=2500):
    if x['strand'] == "+":
        x['start'] = x['start'] - radius if x['start'] - radius > 0 else 0
        x['end'] = x['start'] + (radius * 2)
        return x
    elif x['strand'] == "-":
        x['end'] += radius
        x['start'] = x['end'] - (radius * 2) if x['end'] - (radius * 2) > 0 else 0
        return x


def get_promoter_and_genebody(x, radius=2500):
    if x['strand'] == "+":
        x['start'] = x['start'] - radius if x['start'] - radius > 0 else 0
        return x
    elif x['strand'] == "-":
        x['end'] += radius
        return x


def bowtie2Map(inputFastq1, outputBam, log, metrics, genomeIndex, maxInsert, cpus, inputFastq2=None):
    import re

    outputBam = re.sub("\.bam$", "", outputBam)
    # Admits 2000bp-long fragments (--maxins option)
    cmd = "bowtie2 --very-sensitive -p {0}".format(cpus)
    cmd += " -x {0}".format(genomeIndex)
    cmd += " --met-file {0}".format(metrics)
    if inputFastq2 is None:
        cmd += " {0} ".format(inputFastq1)
    else:
        cmd += " --maxins {0}".format(maxInsert)
        cmd += " -1 {0}".format(inputFastq1)
        cmd += " -2 {0}".format(inputFastq2)
    cmd += " 2> {0} | samtools view -S -b - | samtools sort - {1}".format(log, outputBam)

    return cmd


# Get encode blacklisted regions
blacklist = "wgEncodeDacMapabilityConsensusExcludable.bed.gz"

os.system("wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/{0}".format(blacklist))
os.system("gzip -d {0}".format(blacklist))
os.system("mv {} ../data/{}".format(re.sub(".gz", "", blacklist)))


# Get RefSeq in refflatt format exported as bed file from UCSC
df = pd.read_csv(os.path.join("data", "external", "refseq.refflat.bed.gz"), sep="\t", header=None)
# groupby chromossome, gene name and strand, get wider gene borders across all isoforms
df = df.groupby([0, 3, 5])[[1, 2]].apply(lambda x: pd.Series([min(x[1]), max(x[2])])).drop_duplicates()
df.columns = ["start", "end"]
df.index.names = ["chrom", "gene_name", "strand"]

# Get TSSs
tsss = df.reset_index()[["chrom", "start", "end", "gene_name", "strand"]].apply(get_tss, axis=1)
tsss.to_csv(os.path.join("data", "external", "refseq.refflat.tss.bed"), sep="\t", index=False, header=False)
# sort bed file
tsss = pybedtools.BedTool(os.path.join("data", "external", "refseq.refflat.tss.bed"))
tsss.sort().saveas(os.path.join("data", "external", "refseq.refflat.tss.bed"))


#


# # Get ensembl genes and transcripts from grch37 and hg19 chromosomes (to get TSSs)
# ensembl_genes = "ensGene.txt.gz"
# os.system("wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/{0}".format(ensembl_genes))
# os.system("gzip -d {0}".format(ensembl_genes))
# os.system("mv ensGene.txt ../data/ensGene.txt")
# # get names
# ensembl_names = "ensemblToGeneName.txt.gz"
# os.system("wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/{0}".format(ensembl_names))
# os.system("gzip -d {0}".format(ensembl_names))
# os.system("mv ensemblToGeneName.txt ../data/ensemblToGeneName.txt")

# Get ensembl genes from grch37 and hg19 chromosomes from UCSC (you get introns this way)
regions = [
    "data/external/ensembl_tss.bed",
    "data/external/ensembl_tss2kb.bed",
    "data/external/ensembl_utr5.bed",
    "data/external/ensembl_exons.bed",
    "data/external/ensembl_introns.bed",
    "data/external/ensembl_utr3.bed",
]

# remove annotation after Ensembl transcriptID
for region in regions:
    r = pybedtools.BedTool(region).sort()
    r = pd.read_csv(region, sep="\t", header=None)
    r[3] = r[3].apply(lambda x: x.split("_")[0])
    r.to_csv(region, sep="\t", header=None, index=False)

for i, region in enumerate(regions[1:]):
    r = pybedtools.BedTool(region)
    if i == 0:
        genes = r
    else:
        genes.cat(r)
genes.sort().saveas("data/external/ensembl_genes.bed")

# Make bed file
genes = pd.read_csv("data/external/ensGene.txt", sep="\t", header=None)
genes = genes[[2, 4, 5, 12, 1, 3]]
genes.columns = ['chrom', 'start', 'end', 'gene', 'transcript', 'strand']

# Annotate with gene names
names = pd.read_csv("data/external/ensemblToGeneName.txt", sep="\t", header=None)
names.columns = ['transcript', 'name']
annotation = pd.merge(genes, names)
annotation.to_csv("data/external/GRCh37_hg19_ensembl_genes.bed", sep="\t", index=False, header=False)

# Get TSSs
tsss = annotation.apply(get_tss, axis=1)
tsss.to_csv(".data/external/GRCh37_hg19_ensembl_genes.tss.bed", sep="\t", index=False, header=False)
# sort bed file
tsss = pybedtools.BedTool("../data/GRCh37_hg19_ensembl_genes.tss.bed")
tsss.sort().saveas("../data/GRCh37_hg19_ensembl_genes.tss.bed")


# Get refseq bed file from UCSC, add 1 bp upstream, name as hg19.refSeq.TSS.bed
"sed 's/_up_1_.*//' hg19.refSeq.TSS.bed > t"
# Filter out ncRNAs
"grep NM t > hg19.refSeq.TSS.mRNA.bed"


# Get roadmap HMM state annotation
# read more about it here http://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html#core_15state
roadmap_url = "http://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/"
roadmap_15statesHMM = "all.mnemonics.bedFiles.tgz"
os.system("wget {0}{1}".format(roadmap_url, roadmap_15statesHMM))

# Get all cell's HMM state annotation
os.system("gzip -d {0}".format(roadmap_15statesHMM))

# concatenate all files
all_states = "all_states_all_lines.bed"
os.system("cat *.bed > {0}".format(all_states))

# Get CD19 perypheral blood HMM state annotation
roadmap_15statesHMM_CD19 = "E032_15_coreMarks_mnemonics.bed.gz"
os.system("tar zxvf {0} {1}".format(roadmap_15statesHMM, roadmap_15statesHMM_CD19))
os.system("gzip -d {0}".format(roadmap_15statesHMM_CD19))
os.system("mv E032_15_coreMarks_mnemonics.bed ../data/E032_15_coreMarks_mnemonics.bed")
