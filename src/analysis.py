#!/usr/bin/env python

"""
This is the main script of the cll-ibrutinib project.
"""

# %logstart  # log ipython session
from argparse import ArgumentParser
import os
import sys
from looper.models import Project
import pybedtools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
import multiprocessing
import parmap
import pysam
import numpy as np
import pandas as pd
import cPickle as pickle
from collections import Counter


# Set settings
pd.set_option("date_dayfirst", True)
sns.set(context="paper", style="white", palette="pastel", color_codes=True)
sns.set_palette(sns.color_palette("colorblind"))
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rc('text', usetex=False)

sys.setdefaultencoding('utf8')


def pickle_me(function):
    """
    Decorator for some methods of Analysis class.
    """
    def wrapper(obj, *args):
        function(obj, *args)
        pickle.dump(obj, open(obj.pickle_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return wrapper


class Analysis(object):
    """
    Class to hold functions and data from analysis.
    """

    def __init__(self, data_dir, results_dir, samples, pickle_file):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.samples = samples
        self.pickle_file = pickle_file

    @pickle_me
    def to_pickle(self):
        pass

    def from_pickle(self):
        return pickle.load(open(self.pickle_file, 'rb'))

    @pickle_me
    def get_consensus_sites(self, samples):
        """Get consensus (union) sites across samples"""

        for i, sample in enumerate(samples):
            print(sample.name)
            # Get peaks
            peaks = pybedtools.BedTool(sample.peaks)
            # Merge overlaping peaks within a sample
            peaks = peaks.merge()
            if i == 0:
                sites = peaks
            else:
                # Concatenate all peaks
                sites = sites.cat(peaks)

        # Merge overlaping peaks across samples
        sites = sites.merge()

        # Filter
        # remove blacklist regions
        blacklist = pybedtools.BedTool(os.path.join(self.data_dir, "external", "wgEncodeDacMapabilityConsensusExcludable.bed"))
        # remove chrM peaks and save
        sites.intersect(v=True, b=blacklist).filter(lambda x: x.chrom != 'chrM').saveas(os.path.join(self.results_dir, "cll-ibrutinib_peaks.bed"))

        # Read up again
        self.sites = pybedtools.BedTool(os.path.join(self.results_dir, "cll-ibrutinib_peaks.bed"))

    @pickle_me
    def calculate_peak_support(self, samples):
        # calculate support (number of samples overlaping each merged peak)
        for i, sample in enumerate(samples):
            print(sample.name)
            if i == 0:
                support = self.sites.intersect(sample.peaks, wa=True, c=True)
            else:
                support = support.intersect(sample.peaks, wa=True, c=True)
            print(support.head())
        support = support.to_dataframe()
        # support = support.reset_index()
        support.columns = ["chrom", "start", "end"] + [sample.name for sample in samples]
        support.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.binary_overlap_support.csv"), index=False)

        # get % of total consensus regions found per sample
        m = pd.melt(support, ["chrom", "start", "end"], var_name="sample_name")
        # groupby
        n = m.groupby("sample_name").apply(lambda x: len(x[x["value"] == 1]))

        # divide sum (of unique overlaps) by total to get support value between 0 and 1
        support["support"] = support[range(len(samples))].apply(lambda x: sum([i if i <= 1 else 1 for i in x]) / float(len(self.samples)), axis=1)
        # save
        support.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.support.csv"), index=False)

        self.support = support

    @pickle_me
    def measure_coverage(self, samples):
        # Count reads with pysam
        # make strings with intervals
        sites_str = [str(i.chrom) + ":" + str(i.start) + "-" + str(i.stop) for i in self.sites]
        # count, create dataframe
        self.coverage = pd.DataFrame(
            map(
                lambda x:
                    pd.Series(x),
                    parmap.map(
                        count_reads_in_intervals,
                        [sample.filtered for sample in samples],
                        sites_str,
                        parallel=True
                    )
            ),
            index=[sample.name for sample in samples]
        ).T

        # Add interval description to df
        ints = map(
            lambda x: (
                x.split(":")[0],
                x.split(":")[1].split("-")[0],
                x.split(":")[1].split("-")[1]
            ),
            self.coverage.index
        )
        self.coverage["chrom"] = [x[0] for x in ints]
        self.coverage["start"] = [int(x[1]) for x in ints]
        self.coverage["end"] = [int(x[2]) for x in ints]

        # save to disk
        self.coverage.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.raw_coverage.tsv"), sep="\t", index=True)

    @pickle_me
    def normalize_coverage_quantiles(self, samples):
        # Normalize by quantiles
        to_norm = self.coverage.iloc[:, :len(samples)]
        self.coverage_qnorm = pd.DataFrame(
            normalize_quantiles_r(np.array(to_norm)),
            index=to_norm.index,
            columns=to_norm.columns
        )
        # Log2 transform
        self.coverage_qnorm = np.log2(1 + self.coverage_qnorm)

        self.coverage_qnorm = self.coverage_qnorm.join(self.coverage[['chrom', 'start', 'end']])
        self.coverage_qnorm.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.coverage_qnorm.log2.tsv"), sep="\t", index=True)

    def get_peak_gene_annotation(self):
        """
        Annotates peaks with closest gene.
        Needs files downloaded by prepare_external_files.py
        """
        # create bedtool with hg19 TSS positions
        hg19_ensembl_tss = pybedtools.BedTool(os.path.join(self.data_dir, "external", "ensembl_tss.bed"))
        # get closest TSS of each cll peak
        closest = self.sites.closest(hg19_ensembl_tss, d=True).to_dataframe()[['chrom', 'start', 'end', 'thickStart', 'blockCount']]
        closest.columns = ['chrom', 'start', 'end', 'ensembl_transcript_id', 'distance']

        # add gene name and ensemble_gene_id
        ensembl_gtn = pd.read_table(os.path.join(self.data_dir, "external", "ensemblToGeneName.txt"), header=None)
        ensembl_gtn.columns = ['ensembl_transcript_id', 'gene_name']
        ensembl_gtp = pd.read_table(os.path.join(self.data_dir, "external", "ensGtp.txt"), header=None)[[0, 1]]
        ensembl_gtp.columns = ['ensembl_gene_id', 'ensembl_transcript_id']
        ensembl = pd.merge(ensembl_gtn, ensembl_gtp)

        gene_annotation = pd.merge(closest, ensembl, how="left")

        # aggregate annotation per peak, concatenate various genes (comma-separated)
        self.gene_annotation = gene_annotation.groupby(['chrom', 'start', 'end']).aggregate(lambda x: ",".join(set([str(i) for i in x]))).reset_index()

        # save to disk
        self.gene_annotation.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.gene_annotation.csv"), index=False)

        # save distances to all TSSs (for plotting)
        self.closest_tss_distances = closest['distance'].tolist()
        pickle.dump(self.closest_tss_distances, open(os.path.join(self.results_dir, "cll-ibrutinib_peaks.closest_tss_distances.pickle"), 'wb'))

    def get_peak_genomic_location(self):
        """
        Annotates peaks with its type of genomic location.
        Needs files downloaded by prepare_external_files.py
        """
        regions = [
            "ensembl_genes.bed", "ensembl_tss2kb.bed",
            "ensembl_utr5.bed", "ensembl_exons.bed", "ensembl_introns.bed", "ensembl_utr3.bed"]

        # create background
        # shuffle regions in genome to create background (keep them in the same chromossome)
        background = self.sites.shuffle(genome='hg19', chrom=True)

        for i, region in enumerate(regions):
            region_name = region.replace(".bed", "").replace("ensembl_", "")
            r = pybedtools.BedTool(os.path.join(self.data_dir, "external", region))
            if region_name == "genes":
                region_name = "intergenic"
                df = self.sites.intersect(r, wa=True, f=0.2, v=True).to_dataframe()
                dfb = background.intersect(r, wa=True, f=0.2, v=True).to_dataframe()
            else:
                df = self.sites.intersect(r, wa=True, u=True, f=0.2).to_dataframe()
                dfb = background.intersect(r, wa=True, u=True, f=0.2).to_dataframe()
            df['genomic_region'] = region_name
            dfb['genomic_region'] = region_name
            if i == 0:
                region_annotation = df
                region_annotation_b = dfb
            else:
                region_annotation = pd.concat([region_annotation, df])
                region_annotation_b = pd.concat([region_annotation_b, dfb])

        # sort
        region_annotation.sort(['chrom', 'start', 'end'], inplace=True)
        region_annotation_b.sort(['chrom', 'start', 'end'], inplace=True)
        # remove duplicates (there shouldn't be anyway)
        region_annotation = region_annotation.reset_index(drop=True).drop_duplicates()
        region_annotation_b = region_annotation_b.reset_index(drop=True).drop_duplicates()
        # join various regions per peak
        self.region_annotation = region_annotation.groupby(['chrom', 'start', 'end']).aggregate(lambda x: ",".join(set([str(i) for i in x]))).reset_index()
        self.region_annotation_b = region_annotation_b.groupby(['chrom', 'start', 'end']).aggregate(lambda x: ",".join(set([str(i) for i in x]))).reset_index()

        # save to disk
        self.region_annotation.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.region_annotation.csv"), index=False)
        self.region_annotation_b.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.region_annotation_background.csv"), index=False)

    def get_peak_chromatin_state(self):
        """
        Annotates peaks with chromatin states.
        (For now states are from CD19+ cells).
        Needs files downloaded by prepare_external_files.py
        """
        # create bedtool with CD19 chromatin states
        states_cd19 = pybedtools.BedTool(os.path.join(self.data_dir, "external", "E032_15_coreMarks_mnemonics.bed"))

        # create background
        # shuffle regions in genome to create background (keep them in the same chromossome)
        background = self.sites.shuffle(genome='hg19', chrom=True)

        # intersect with cll peaks, to create annotation, get original peaks
        chrom_state_annotation = self.sites.intersect(states_cd19, wa=True, wb=True, f=0.2).to_dataframe()[['chrom', 'start', 'end', 'thickStart']]
        chrom_state_annotation_b = background.intersect(states_cd19, wa=True, wb=True, f=0.2).to_dataframe()[['chrom', 'start', 'end', 'thickStart']]

        # aggregate annotation per peak, concatenate various annotations (comma-separated)
        self.chrom_state_annotation = chrom_state_annotation.groupby(['chrom', 'start', 'end']).aggregate(lambda x: ",".join(x)).reset_index()
        self.chrom_state_annotation.columns = ['chrom', 'start', 'end', 'chromatin_state']

        self.chrom_state_annotation_b = chrom_state_annotation_b.groupby(['chrom', 'start', 'end']).aggregate(lambda x: ",".join(x)).reset_index()
        self.chrom_state_annotation_b.columns = ['chrom', 'start', 'end', 'chromatin_state']

        # save to disk
        self.chrom_state_annotation.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.chromatin_state.csv"), index=False)
        self.chrom_state_annotation_b.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.chromatin_state_background.csv"), index=False)

    @pickle_me
    def annotate(self, samples):
        # add closest gene
        self.coverage_qnorm_annotated = pd.merge(
            self.coverage_qnorm,
            self.gene_annotation, on=['chrom', 'start', 'end'], how="left")
        # add genomic location
        self.coverage_qnorm_annotated = pd.merge(
            self.coverage_qnorm_annotated,
            self.region_annotation[['chrom', 'start', 'end', 'genomic_region']], on=['chrom', 'start', 'end'], how="left")
        # add chromatin state
        self.coverage_qnorm_annotated = pd.merge(
            self.coverage_qnorm_annotated,
            self.chrom_state_annotation[['chrom', 'start', 'end', 'chromatin_state']], on=['chrom', 'start', 'end'], how="left")

        # add support
        self.coverage_qnorm_annotated = pd.merge(
            self.coverage_qnorm_annotated,
            self.support[['chrom', 'start', 'end', 'support']], on=['chrom', 'start', 'end'], how="left")

        # calculate mean coverage
        self.coverage_qnorm_annotated['mean'] = self.coverage_qnorm_annotated[[sample.name for sample in samples]].mean(axis=1)
        # calculate coverage variance
        self.coverage_qnorm_annotated['variance'] = self.coverage_qnorm_annotated[[sample.name for sample in samples]].var(axis=1)
        # calculate std deviation (sqrt(variance))
        self.coverage_qnorm_annotated['std_deviation'] = np.sqrt(self.coverage_qnorm_annotated['variance'])
        # calculate dispersion (variance / mean)
        self.coverage_qnorm_annotated['dispersion'] = self.coverage_qnorm_annotated['variance'] / self.coverage_qnorm_annotated['mean']
        # calculate qv2 (std / mean) ** 2
        self.coverage_qnorm_annotated['qv2'] = (self.coverage_qnorm_annotated['std_deviation'] / self.coverage_qnorm_annotated['mean']) ** 2

        # calculate "amplitude" (max - min)
        self.coverage_qnorm_annotated['amplitude'] = (
            self.coverage_qnorm_annotated[[sample.name for sample in samples]].max(axis=1) -
            self.coverage_qnorm_annotated[[sample.name for sample in samples]].min(axis=1)
        )

        # Pair indexes
        assert self.coverage.shape[0] == self.coverage_qnorm_annotated.shape[0]
        self.coverage_qnorm_annotated.index = self.coverage.index

        # Save
        self.coverage_qnorm_annotated.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.coverage_qnorm.log2.annotated.tsv"), sep="\t", index=True)

    def plot_peak_characteristics(self):
        # Loop at summary statistics:
        # interval lengths
        fig, axis = plt.subplots()
        sns.distplot([interval.length for interval in self.sites], bins=300, kde=False, ax=axis)
        axis.set_xlim(0, 2000)  # cut it at 2kb
        axis.set_xlabel("peak width (bp)")
        axis.set_ylabel("frequency")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.lengths.svg"), bbox_inches="tight")
        plt.close("all")

        # plot support
        fig, axis = plt.subplots()
        sns.distplot(self.support["support"], bins=40, ax=axis)
        axis.set_ylabel("frequency")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.support.svg"), bbox_inches="tight")
        plt.close("all")

        # Plot distance to nearest TSS
        fig, axis = plt.subplots()
        sns.distplot(self.closest_tss_distances, bins=200, ax=axis)
        axis.set_xlim(0, 100000)  # cut it at 100kb
        axis.set_xlabel("distance to nearest TSS (bp)")
        axis.set_ylabel("frequency")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.tss_distance.svg"), bbox_inches="tight")
        plt.close("all")

        # Plot genomic regions
        # these are just long lists with genomic regions
        all_region_annotation = [item for sublist in self.region_annotation['genomic_region'].apply(lambda x: x.split(",")) for item in sublist]
        all_region_annotation_b = [item for sublist in self.region_annotation_b['genomic_region'].apply(lambda x: x.split(",")) for item in sublist]

        # count region frequency
        count = Counter(all_region_annotation)
        data = pd.DataFrame([count.keys(), count.values()]).T
        data = data.sort([1], ascending=False)
        # also for background
        background = Counter(all_region_annotation_b)
        background = pd.DataFrame([background.keys(), background.values()]).T
        background = background.ix[data.index]  # same sort order as in the real data

        # plot individually
        fig, axis = plt.subplots(3, sharex=True, sharey=False)
        sns.barplot(x=0, y=1, data=data, ax=axis[0])
        sns.barplot(x=0, y=1, data=background, ax=axis[1])
        sns.barplot(x=0, y=1, data=pd.DataFrame([data[0], np.log2((data[1] / background[1]).astype(float))]).T, ax=axis[2])
        axis[0].set_title("ATAC-seq peaks")
        axis[1].set_title("genome background")
        axis[2].set_title("peaks over background")
        axis[1].set_xlabel("genomic region")
        axis[2].set_xlabel("genomic region")
        axis[0].set_ylabel("frequency")
        axis[1].set_ylabel("frequency")
        axis[2].set_ylabel("fold-change")
        fig.autofmt_xdate()
        fig.tight_layout()
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.genomic_regions.svg"), bbox_inches="tight")
        plt.close("all")

        # # Plot chromatin states
        # get long list of chromatin states (for plotting)
        all_chrom_state_annotation = [item for sublist in self.chrom_state_annotation['chromatin_state'].apply(lambda x: x.split(",")) for item in sublist]
        all_chrom_state_annotation_b = [item for sublist in self.chrom_state_annotation_b['chromatin_state'].apply(lambda x: x.split(",")) for item in sublist]

        # count region frequency
        count = Counter(all_chrom_state_annotation)
        data = pd.DataFrame([count.keys(), count.values()]).T
        data = data.sort([1], ascending=False)
        # also for background
        background = Counter(all_chrom_state_annotation_b)
        background = pd.DataFrame([background.keys(), background.values()]).T
        background = background.ix[data.index]  # same sort order as in the real data

        fig, axis = plt.subplots(3, sharex=True, sharey=False)
        sns.barplot(x=0, y=1, data=data, ax=axis[0])
        sns.barplot(x=0, y=1, data=background, ax=axis[1])
        sns.barplot(x=0, y=1, data=pd.DataFrame([data[0], np.log2((data[1] / background[1]).astype(float))]).T, ax=axis[2])
        axis[0].set_title("ATAC-seq peaks")
        axis[1].set_title("genome background")
        axis[2].set_title("peaks over background")
        axis[1].set_xlabel("chromatin state")
        axis[2].set_xlabel("chromatin state")
        axis[0].set_ylabel("frequency")
        axis[1].set_ylabel("frequency")
        axis[2].set_ylabel("fold-change")
        fig.autofmt_xdate()
        fig.tight_layout()
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.chromatin_states.svg"), bbox_inches="tight")

        # distribution of count attributes
        data = self.coverage_qnorm_annotated.copy()

        fig, axis = plt.subplots(1)
        sns.distplot(data["mean"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.mean.distplot.svg"), bbox_inches="tight")

        fig, axis = plt.subplots(1)
        sns.distplot(data["qv2"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.qv2.distplot.svg"), bbox_inches="tight")

        fig, axis = plt.subplots(1)
        sns.distplot(data["dispersion"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.dispersion.distplot.svg"), bbox_inches="tight")

        # this is loaded now
        df = pd.read_csv(os.path.join(self.data_dir, "cll-ibrutinib_peaks.support.csv"))
        fig, axis = plt.subplots(1)
        sns.distplot(df["support"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.support.distplot.svg"), bbox_inches="tight")

        plt.close("all")

    def plot_coverage(self):
        data = self.coverage_qnorm_annotated.copy()
        # (rewrite to avoid putting them there in the first place)
        variables = ['gene_name', 'genomic_region', 'chromatin_state']

        for variable in variables:
            d = data[variable].str.split(',').apply(pd.Series).stack()  # separate comma-delimited fields
            d.index = d.index.droplevel(1)  # returned a multiindex Series, so get rid of second index level (first is from original row)
            data = data.drop([variable], axis=1)  # drop original column so there are no conflicts
            d.name = variable
            data = data.join(d)  # joins on index

        variables = [
            'chrom', 'start', 'end',
            'ensembl_transcript_id', 'distance', 'ensembl_gene_id', 'support',
            'mean', 'variance', 'std_deviation', 'dispersion', 'qv2',
            'amplitude', 'gene_name', 'genomic_region', 'chromatin_state']
        # Plot
        data_melted = pd.melt(
            data,
            id_vars=variables, var_name="sample", value_name="norm_counts")

        # transform dispersion
        data_melted['dispersion'] = np.log2(1 + data_melted['dispersion'])

        # Together in same violin plot
        fig, axis = plt.subplots(1)
        sns.violinplot("genomic_region", "norm_counts", data=data_melted, ax=axis)
        fig.savefig(os.path.join(self.results_dir, "norm_counts.per_genomic_region.violinplot.png"), bbox_inches="tight", dpi=300)

        # dispersion
        fig, axis = plt.subplots(1)
        sns.violinplot("genomic_region", "dispersion", data=data_melted, ax=axis)
        fig.savefig(os.path.join(self.results_dir, "norm_counts.dispersion.per_genomic_region.violinplot.png"), bbox_inches="tight", dpi=300)

        # dispersion
        fig, axis = plt.subplots(1)
        sns.violinplot("genomic_region", "qv2", data=data_melted, ax=axis)
        fig.savefig(os.path.join(self.results_dir, "norm_counts.qv2.per_genomic_region.violinplot.png"), bbox_inches="tight", dpi=300)

        fig, axis = plt.subplots(1)
        sns.violinplot("chromatin_state", "norm_counts", data=data_melted, ax=axis)
        fig.savefig(os.path.join(self.results_dir, "norm_counts.chromatin_state.violinplot.png"), bbox_inches="tight", dpi=300)

        fig, axis = plt.subplots(1)
        sns.violinplot("chromatin_state", "dispersion", data=data_melted, ax=axis)
        fig.savefig(os.path.join(self.results_dir, "norm_counts.dispersion.chromatin_state.violinplot.png"), bbox_inches="tight", dpi=300)

        fig, axis = plt.subplots(1)
        sns.violinplot("chromatin_state", "qv2", data=data_melted, ax=axis)
        fig.savefig(os.path.join(self.results_dir, "norm_counts.qv2.chromatin_state.violinplot.png"), bbox_inches="tight", dpi=300)

        # separated by variable in one grid
        g = sns.FacetGrid(data_melted, col="genomic_region", col_wrap=3)
        g.map(sns.distplot, "mean", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.mean.per_genomic_region.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="genomic_region", col_wrap=3)
        g.map(sns.distplot, "dispersion", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.dispersion.per_genomic_region.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="genomic_region", col_wrap=3)
        g.map(sns.distplot, "qv2", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.qv2.per_genomic_region.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="genomic_region", col_wrap=3)
        g.map(sns.distplot, "support", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.support.per_genomic_region.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="chromatin_state", col_wrap=3)
        g.map(sns.distplot, "mean", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.mean.chromatin_state.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="chromatin_state", col_wrap=3)
        g.map(sns.distplot, "dispersion", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.dispersion.chromatin_state.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="chromatin_state", col_wrap=3)
        g.map(sns.distplot, "qv2", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.qv2.chromatin_state.distplot.png"), bbox_inches="tight", dpi=300)

        g = sns.FacetGrid(data_melted, col="chromatin_state", col_wrap=3)
        g.map(sns.distplot, "support", hist=False, rug=False)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts.support.chromatin_state.distplot.png"), bbox_inches="tight", dpi=300)
        plt.close("all")

    def plot_variance(self, samples):

        g = sns.jointplot('mean', "dispersion", data=self.coverage_qnorm_annotated, kind="kde")
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.dispersion.png"), bbox_inches="tight", dpi=300)

        g = sns.jointplot('mean', "qv2", data=self.coverage_qnorm_annotated)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.qv2_vs_mean.png"), bbox_inches="tight", dpi=300)

        g = sns.jointplot('support', "qv2", data=self.coverage_qnorm_annotated)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.support_vs_qv2.png"), bbox_inches="tight", dpi=300)

        # Filter out regions which the maximum across all samples is below a treshold
        filtered = self.coverage_qnorm_annotated[self.coverage_qnorm_annotated[[sample.name for sample in samples]].max(axis=1) > 3]

        sns.jointplot('mean', "dispersion", data=filtered)
        plt.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.dispersion.filtered.png"), bbox_inches="tight", dpi=300)
        plt.close('all')
        sns.jointplot('mean', "qv2", data=filtered)
        plt.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.support_vs_qv2.filtered.png"), bbox_inches="tight", dpi=300)

    def plot_sample_correlations(self):
        pass

    def unsupervised(self, samples):
        """
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS
        import matplotlib.patches as mpatches

        traits = ["patient_id", "timepoint_name", "treatment_response.1", "gender", "ighv_mutation_status", "CD38_positive", "diagnosis_stage_rai", "atac_seq_batch"]
        pallete = sns.color_palette("colorblind") + sns.color_palette("Set2", 10) + sns.color_palette("Set2")[::-1]

        # Approach 1: all regions
        X = self.coverage_qnorm_annotated[[s.name for s in samples]]

        # Pairwise correlations
        g = sns.clustermap(X.corr(), xticklabels=False, annot=True)
        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        g.fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.all_sites.corr.clustermap.svg"), bbox_inches='tight')

        # MDS
        mds = MDS(n_jobs=-1)
        x_new = mds.fit_transform(X.T)
        # transform again
        x = pd.DataFrame(x_new)
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)

        fig, axis = plt.subplots(1, len(traits), figsize=(4 * len(traits), 4))
        axis = axis.flatten()
        for i, trait in enumerate(traits):
            # make unique dict
            color_dict = dict(zip(list(set([getattr(s, trait) for s in samples])), pallete))
            colors = [color_dict[getattr(s, trait)] for s in samples]

            for j in range(xx.shape[0]):
                axis[i].scatter(xx.ix[j][0], xx.ix[j][1], s=50, color=colors[j], label=getattr(samples[j], trait))
            axis[i].set_title(traits[i])
            axis[i].legend(
                handles=[mpatches.Patch(color=v, label=k) for k, v in color_dict.items()],
                ncol=1,
                loc='center left',
                bbox_to_anchor=(1, 0.5))

        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.all_sites.mds.svg"), bbox_inches="tight")

        # PCA
        pca = PCA()
        x_new = pca.fit_transform(X.T)
        # transform again
        x = pd.DataFrame(x_new)
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)

        # plot % explained variance per PC
        fig, axis = plt.subplots(1)
        axis.plot(
            range(1, len(pca.explained_variance_) + 1),  # all PCs
            (pca.explained_variance_ / pca.explained_variance_.sum()) * 100, 'o-')  # % of total variance
        axis.set_xlabel("PC")
        axis.set_ylabel("% variance")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.all_sites.pca.explained_variance.svg"), bbox_inches='tight')

        # plot
        fig, axis = plt.subplots(min(xx.shape[0] - 1, 10), len(traits), figsize=(4 * len(traits), 4 * min(xx.shape[0] - 1, 10)))
        for pc in range(min(xx.shape[0] - 1, 10)):
            for i, trait in enumerate(traits):
                # make unique dict
                color_dict = dict(zip(list(set([getattr(s, trait) for s in samples])), pallete))
                colors = [color_dict[getattr(s, trait)] for s in samples]

                for j in range(len(xx)):
                    axis[pc][i].scatter(xx.ix[j][pc], xx.ix[j][pc + 1], s=50, color=colors[j], label=getattr(samples[j], trait))
                axis[pc][i].set_title(traits[i])
                axis[pc][i].set_xlabel("PC {}".format(pc + 1))
                axis[pc][i].set_ylabel("PC {}".format(pc + 2))
                axis[pc][i].legend(
                    handles=[mpatches.Patch(color=v, label=k) for k, v in color_dict.items()],
                    ncol=1,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.all_sites.pca.svg"), bbox_inches="tight")

        #

        # # Test association of PCs with variables
        import itertools
        from scipy.stats import kruskal
        from scipy.stats import pearsonr

        associations = list()
        for pc in range(xx.shape[0]):
            for trait in traits:
                print("Trait %s." % trait)
                sel_samples = [s for s in samples if not pd.isnull(getattr(s, trait))]

                # Get all values of samples for this trait
                groups = set([getattr(s, trait) for s in sel_samples])

                # Determine if trait is categorical or continuous
                if all([type(i) is str for i in groups]) or len(groups) == 2:
                    variable_type = "categorical"
                elif all([type(i) in [int, float] for i in groups]):
                    variable_type = "numerical"
                else:
                    print("Trait %s cannot be tested." % trait)
                    associations.append([pc, trait, variable_type, np.nan, np.nan, np.nan])
                    continue

                if variable_type == "categorical":
                    # It categorical, test pairwise combinations of traits
                    for group1, group2 in itertools.combinations(groups, 2):
                        g1_indexes = [i for i, s in enumerate(samples) if getattr(s, trait) == group1]
                        g2_indexes = [i for i, s in enumerate(samples) if getattr(s, trait) == group2]

                        g1_values = xx.loc[g1_indexes, pc]
                        g2_values = xx.loc[g2_indexes, pc]

                        # Test ANOVA (or Kruskal-Wallis H-test)
                        p = kruskal(g1_values, g2_values)[1]

                        # Append
                        associations.append([pc, trait, variable_type, group1, group2, p])

                elif variable_type == "numerical":
                    # It numerical, calculate pearson correlation
                    pc_values = xx.loc[:, pc]
                    trait_values = [getattr(s, trait) for s in samples]
                    p = pearsonr(pc_values, trait_values)[1]

                    associations.append([pc, trait, variable_type, np.nan, np.nan, p])

        associations = pd.DataFrame(associations, columns=["pc", "trait", "variable_type", "group_1", "group_2", "p_value"])

        # write
        associations.to_csv(os.path.join(self.results_dir, "cll-ibrutinib_peaks.pca.variable_principle_components_association.csv"), index=False)

        # Plot
        # associations[associations['p_value'] < 0.05].drop(['group_1', 'group_2'], axis=1).drop_duplicates()
        # associations.drop(['group_1', 'group_2'], axis=1).drop_duplicates().pivot(index="pc", columns="trait", values="p_value")
        pivot = associations.groupby(["pc", "trait"]).min()['p_value'].reset_index().pivot(index="pc", columns="trait", values="p_value").dropna(axis=1)

        # heatmap of -log p-values
        sns.clustermap(-np.log10(pivot), row_cluster=False, annot=True)
        plt.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.pca.variable_principle_components_association.svg"), bbox_inches="tight")

        # heatmap of masked significant
        sns.clustermap((pivot < 0.05).astype(int), row_cluster=False)
        plt.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.pca.variable_principle_components_association.masked.svg"), bbox_inches="tight")

        sns.clustermap((pivot < 0.05).astype(int).drop(["patient_id", "batch", "diagnosis_stage_rai"], axis=1), row_cluster=False)
        plt.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.pca.variable_principle_components_association.masked.filtered.svg"), bbox_inches="tight")

        # plot PCs associated with treatment
        combs = [(1, 4), (1, 17), (4, 17), (4, 5)]

        fig, axis = plt.subplots(len(combs), len(traits), figsize=(4 * len(traits), 4 * len(combs)))
        for z, (pc1, pc2) in enumerate(combs):
            for i, trait in enumerate(traits):
                # make unique dict
                color_dict = dict(zip(list(set([getattr(s, trait) for s in samples])), pallete))
                colors = [color_dict[getattr(s, trait)] for s in samples]

                for j in range(len(xx)):
                    axis[z][i].scatter(xx.ix[j][pc1], xx.ix[j][pc2], s=50, color=colors[j], label=getattr(samples[j], trait))
                axis[z][i].set_title(traits[i])
                axis[z][i].set_xlabel("PC {}".format(pc1 + 1))
                axis[z][i].set_ylabel("PC {}".format(pc2 + 1))
                axis[z][i].legend(
                    handles=[mpatches.Patch(color=v, label=k) for k, v in color_dict.items()],
                    ncol=1,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib_peaks.all_sites.pca.selected_treatment.svg"), bbox_inches="tight")

    def pathways(self):

        gene_lists = {
            "BCR": [
                "LYN", "VAV3", "CD72", "SYK", "CD81", "PRKCB", "DAPP1",
                "NFATC1", "FOS", "PIK3R1", "MALT1", "PIK3CG", "VAV2",
                "RASGRP3", "INPP5D", "BLNK", "AKT1", "MAPK1", "PTPN6",
                "KRAS", "SOS1", "PIK3AP1", "SOS2"] + [
                "CD81", "PIK3CB", "PIK3R1", "PIK3CG", "RASGRP3",
                "IKBKB", "CD79A", "PPP3CC", "PPP3R2", "AKT2", "AKT3",
                "PLCG2", "RAC3", "MAPK3", "LYN", "VAV3", "MAP2K2",
                "PRKCB", "NFATC3", "NFATC2", "NFATC1", "VAV2", "NFKBIE",
                "SOS1", "PIK3AP1", "CD22", "CARD11"],
            "FOXO": [
                "PRKAA1", "CDKN1B", "AGAP2", "PTEN", "PIK3R1", "FOXO3",
                "FOXO1", "PIK3CG", "IGF1R", "S1PR1", "AKT1", "MAPK1",
                "SMAD4", "GABARAPL1", "CDKN2B", "SMAD3", "HOMER1", "HOMER2",
                "GADD45A", "SOD2", "TGFBR1", "TGFBR2", "BCL6", "CCNG2", "MDM2",
                "SGK3", "KRAS", "SGK1", "SOS1", "SOS2"] + [
                "ROCK2", "NPR1", "ITPR1", "ATP2A3", "ITPR2", "ATP2A2", "IRS2",
                "CACNA1D", "ATP1A1", "PIK3R1", "PIK3CB", "ADRB2", "MYLK3", "PIK3CG",
                "MYLK", "PPP1CC", "PPP3R2", "PPP3CC", "CREB3L1", "AKT2", "AKT3",
                "KCNMB2", "CACNA1S", "MAPK3", "MEF2A", "TRPC6", "MAP2K2", "PRKCE",
                "NFATC3", "ATP2B4", "NFATC2", "NFATC1", "ATP2B2", "ATP1B1", "ADRA2B",
                "ADRA2A", "PPIF", "VDAC3", "VDAC2", "GTF2IRD1", "MRVI1", "PDE5A", "CREB5"],
            "Apoptosis": [
                "HRK", "PARP4", "GADD45A", "ITPR1", "ITPR2", "PIK3R1", "CFLAR",
                "FOS", "PIK3CG", "LMNB1", "CASP7", "CASP8", "LMNA", "CAPN2",
                "FAS", "AKT1", "PMAIP1", "MAPK1", "KRAS", "BIRC2", "CTSC",
                "MAP3K5", "BIRC3", "CTSB"],
            "Focal Adhesion": [
                "FLT1", "PIK3CB", "ARHGAP5", "MYLK3", "PIK3CG", "MYLK", "IGF1R",
                "PPP1CC", "AKT2", "AKT3", "RAC3", "ITGB8", "VAV3", "PRKCB", "ITGA2",
                "ITGA1", "ACTN4", "PGF", "VAV2", "MYL5", "COL4A2", "COL4A1", "MYL2",
                "ITGA7", "ITGA6", "TLN2", "SOS1", "BIRC2", "ROCK2", "SRC", "LAMA3",
                "PIK3R1", "EGFR", "PAK1", "SPP1", "PAK6", "FLNB", "FYN", "FLNC",
                "MAPK3", "EGF", "LAMB4", "FN1", "VEGFC", "PARVB", "PTK2", "VEGFA",
                "DIAPH1", "COL1A2", "PARVG", "BCL2", "CTNNB1"
            ]
        }

        df = self.coverage_qnorm_annotated

        for name, genes in gene_lists.items():
            # get regions
            path_regions = df[df["gene_name"].str.contains("|".join(list(set(genes))))][["gene_name"] + [s.name for s in self.samples]]
            # reduce per gene
            path_genes = path_regions.groupby(["gene_name"]).mean()

            g = sns.clustermap(
                path_regions.drop("gene_name", axis=1),
                xticklabels=[" - ".join([s.patient_id, str(getattr(s, "timepoint_name")), str(getattr(s, "treatment_response.1"))]) for s in self.samples],
                z_score=0,
                figsize=(10, 15))
            plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
            plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
            g.savefig(os.path.join(self.results_dir, "gene_lists.regions.%s.regions.png" % name), bbox_inches="tight", dpi=300)

            g = sns.clustermap(
                path_genes,
                xticklabels=[" - ".join([s.patient_id, str(getattr(s, "timepoint_name")), str(getattr(s, "treatment_response.1"))]) for s in self.samples],
                z_score=0,
                figsize=(10, 15))
            plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
            plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
            g.savefig(os.path.join(self.results_dir, "gene_lists.genes.%s.genes.png" % name), bbox_inches="tight", dpi=300)

        g = sns.clustermap(
            path_regions.drop("gene_name", axis=1),
            xticklabels=[" - ".join([s.patient_id, str(getattr(s, "timepoint_name")), str(getattr(s, "treatment_response.1"))]) for s in self.samples],
            z_score=0,
            figsize=(10, 15))
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        g.savefig(os.path.join(self.results_dir, "gene_lists.regions.all.regions.png"), bbox_inches="tight", dpi=300)

        # get regions
        path_regions = df[df["gene_name"].str.contains("|".join(list(set(sum(gene_lists.values(), [])))))][["gene_name"] + [s.name for s in self.samples]]
        # reduce per gene
        path_genes = path_regions.groupby(["gene_name"]).mean()
        g = sns.clustermap(
            path_genes,
            xticklabels=[" - ".join([s.patient_id, str(getattr(s, "timepoint_name")), str(getattr(s, "treatment_response.1"))]) for s in self.samples],
            z_score=0,
            figsize=(10, 15))
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        g.savefig(os.path.join(self.results_dir, "gene_lists.genes.all.genes.png"), bbox_inches="tight", dpi=300)

    @pickle_me
    def differential_analysis(self, samples, trait):
        """
        Discover differnetial regions across samples that are associated with a certain trait.
        """
        import string
        import itertools

        sel_samples = [s for s in samples if not pd.isnull(getattr(s, trait))]

        # Get matrix of counts
        counts_matrix = self.coverage[[s.name for s in sel_samples]]

        # Get experiment matrix
        experiment_matrix = pd.DataFrame([sample.as_series() for sample in sel_samples], index=[sample.name for sample in sel_samples])

        # Make output dir
        output_dir = os.path.join(self.results_dir, "..", "deseq")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Two independent models asking different questions:
        # all variables:
        # variables = ["atac_seq_batch", "patient_id", "patient_gender", "del11q", "del13q", "del17p", "tri12", "treatment_response", "ibrutinib_treatment"]

        # 1) for each patient, how was the impact of treatment?
        a = [
            "./results/deseq/deseq.treatment_per_patient",
            "ibrutinib_treatment",
            ["patient_id"]]

        # 2) across patients, what determines the response?
        b = [
            "./results/deseq/deseq.response",
            "treatment_response",
            ["atac_seq_batch", "patient_gender", "del11q", "del13q", "del17p", "tri12", "ibrutinib_treatment"]]

        for output_prefix, variable, covariates in [a, b]:

            # # Run DESeq2 analysis
            # deseq_table = DESeq_analysis(
            #     counts_matrix, experiment_matrix, variable,
            #     covariates=covariates,
            #     output_prefix=output_prefix,
            #     alpha=0.05)

            # to just read in
            deseq_table = pd.read_csv(os.path.join(output_prefix + ".%s.csv" % variable), index_col=0)

            df = self.coverage_qnorm.join(deseq_table)

            # Extract significant based on p-value and fold-change
            diff = df[(df["padj"] < 0.05)]

            groups = pd.Series([getattr(s, variable) for s in sel_samples]).dropna().unique().tolist()
            comparisons = pd.Series(df['comparison'].unique())

            # # Statistics of differential regions
            # # direction-dependent
            # diff["direction"] = diff["log2FoldChange"].apply(lambda x: "up" if x >= 0 else "down")

            # split_diff = diff.groupby(["direction"]).apply(len).sort_values(ascending=False)
            # fig, axis = plt.subplots(1, figsize=(12, 8))
            # sns.barplot(
            #     split_diff.values,
            #     split_diff.reset_index()[['direction']].apply(string.join, axis=1),
            #     orient="h", ax=axis)
            # for t in axis.get_xticklabels():
            #     t.set_rotation(0)
            # sns.despine(fig)
            # fig.savefig(os.path.join("%s.%s.number_differential.split.svg" % (output_prefix, variable)), bbox_inches="tight")

            # #

            # # Pairwise scatter plots
            # fig, axis = plt.subplots(1, figsize=(6, 6), sharex=True, sharey=True)
            # for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            #     # get comparison
            #     comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()
            #     if type(comparison) is pd.Series:
            #         if len(comparison) > 1:
            #             comparison = comparison.iloc[0]

            #     df2 = df[df["comparison"] == comparison]
            #     if df2.shape[0] == 0:
            #         continue

            #     # Hexbin plot
            #     axis.hexbin(np.log2(1 + df2[cond1]), np.log2(1 + df2[cond2]), bins="log", alpha=.75)
            #     axis.set_xlabel(cond1)

            #     diff2 = diff[diff["comparison"] == comparison]
            #     if diff2.shape[0] > 0:
            #         # Scatter plot
            #         axis.scatter(np.log2(1 + diff2[cond1]), np.log2(1 + diff2[cond2]), alpha=0.5, color="red", s=2)
            #     axis.set_ylabel(cond2)
            # sns.despine(fig)
            # fig.savefig(os.path.join("%s.%s.scatter_plots.png" % (output_prefix, variable)), bbox_inches="tight", dpi=300)

            # # Volcano plots
            # fig, axis = plt.subplots(1, figsize=(6, 6), sharex=True, sharey=True)
            # for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            #     # get comparison
            #     comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()
            #     if type(comparison) is pd.Series:
            #         if len(comparison) > 1:
            #             comparison = comparison.iloc[0]

            #     df2 = df[df["comparison"] == comparison]
            #     if df2.shape[0] == 0:
            #         continue

            #     # hexbin
            #     axis.hexbin(df2["log2FoldChange"], -np.log10(df2['padj']), alpha=0.75, color="black", bins='log', mincnt=1)

            #     diff2 = diff[diff["comparison"] == comparison]
            #     if diff2.shape[0] > 0:
            #         # significant scatter
            #         axis.scatter(diff2["log2FoldChange"], -np.log10(diff2['padj']), alpha=0.5, color="red", s=2)
            #     axis.set_title(comparison)
            # sns.despine(fig)
            # fig.savefig(os.path.join("%s.%s.volcano_plots.png" % (output_prefix, variable)), bbox_inches="tight", dpi=300)

            # # MA plots
            # fig, axis = plt.subplots(1, figsize=(6, 6), sharex=True, sharey=True)
            # for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            #     # get comparison
            #     comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()
            #     if type(comparison) is pd.Series:
            #         if len(comparison) > 1:
            #             comparison = comparison.iloc[0]

            #     df2 = df[df["comparison"] == comparison]
            #     if df2.shape[0] == 0:
            #         continue

            #     # hexbin
            #     axis.hexbin(np.log2(df2["baseMean"]), df2["log2FoldChange"], alpha=0.75, color="black", bins='log', mincnt=1)

            #     diff2 = diff[diff["comparison"] == comparison]
            #     if diff2.shape[0] > 0:
            #         # significant scatter
            #         axis.scatter(np.log2(diff2["baseMean"]), diff2["log2FoldChange"], alpha=0.5, color="red", s=2)
            #     axis.set_title(comparison)
            # sns.despine(fig)
            # fig.savefig(os.path.join("%s.%s.ma_plots.png" % (output_prefix, variable)), bbox_inches="tight", dpi=300)

            # Exploration of differential regions

            # Get rankings
            # get effect size
            diff["effect_size"] = diff[groups[1]] - diff[groups[0]]

            # rank effect size, fold change and p-value
            diff["effect_size_rank"] = diff["effect_size"].rank(ascending=False)
            diff["log2FoldChange_rank"] = diff["log2FoldChange"].rank(ascending=False)
            diff["padj_rank"] = diff["padj"].rank(ascending=True)
            # max (worst ranking) of those
            diff["max_rank"] = diff[["effect_size_rank", "log2FoldChange_rank", "padj_rank"]].max(1)
            diff = diff.sort_values("max_rank")

            # Plot 4000 top ranked regions
            g = sns.clustermap(
                diff[[s.name for s in samples]].head(4000),
                yticklabels=False,
                xticklabels=[" - ".join([s.patient_id, str(getattr(s, variable))]) for s in samples],
                standard_scale=0)
            plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
            g.savefig(os.path.join("%s.clustermap.std0.top_4000_ranked.png" % output_prefix), bbox_inches="tight", dpi=300)
            plt.close('all')

            # Examine each region cluster
            region_enr = pd.DataFrame()
            lola_enr = pd.DataFrame()
            motif_enr = pd.DataFrame()
            enrichr_enr = pd.DataFrame()
            for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
                # get comparison
                comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()

                if type(comparison) is pd.Series:
                    if len(comparison) > 1:
                        comparison = comparison.iloc[0]

                # Separate in up/down-regulated regions
                for f, direction in [(np.less, "down"), (np.greater, "up")]:
                    comparison_df = self.coverage_qnorm_annotated.ix[diff[
                        (diff["comparison"] == comparison) &
                        (f(diff["log2FoldChange"], 0))
                    ].index]

                    # Characterize regions
                    prefix = ".".join([output_prefix, variable, comparison, direction])
                    name = ".".join([variable, comparison, direction])
                    if not os.path.exists(prefix):
                        os.makedirs(prefix)

                    print("Doing regions of comparison %s, with prefix %s" % (comparison, prefix))

                    # region's structure
                    if not os.path.exists(os.path.join(prefix, prefix + "_regions.region_enrichment.csv")):
                        characterize_regions_structure(df=comparison_df, prefix=os.path.basename(prefix), output_dir=prefix)
                    # region's function
                    if not os.path.exists(os.path.join(prefix, prefix + "_regions.enrichr.csv")):
                        characterize_regions_function(df=comparison_df, prefix=os.path.basename(prefix), output_dir=prefix)

                    # Read/parse enrichment outputs and add to DFs
                    enr = pd.read_csv(os.path.join(prefix, os.path.basename(prefix) + "_regions.region_enrichment.csv"))
                    enr.columns = ["region"] + enr.columns[1:].tolist()
                    enr["comparison"] = name
                    region_enr = region_enr.append(enr, ignore_index=True)

                    enr = pd.read_csv(os.path.join(prefix, "allEnrichments.txt"), sep="\t")
                    enr["comparison"] = name
                    lola_enr = lola_enr.append(enr, ignore_index=True)

                    enr = parse_ame(prefix).reset_index()
                    enr["comparison"] = name
                    motif_enr = motif_enr.append(enr, ignore_index=True)

                    enr = pd.read_csv(os.path.join(prefix, os.path.basename(prefix) + "_regions.enrichr.csv"))
                    enr["comparison"] = name
                    enrichr_enr = enrichr_enr.append(enr, ignore_index=True)

            motif_enr.columns = ["motifs", "p_value", "comparison"]

            # write combined enrichemnts
            region_enr.to_csv(
                os.path.join("%s.%s.diff_regions.regions.csv" % (output_prefix, variable)), index=False)
            lola_enr.to_csv(
                os.path.join("%s.%s.diff_regions.lola.csv" % (output_prefix, variable)), index=False)
            motif_enr.to_csv(
                os.path.join("%s.%s.diff_regions.motifs.csv" % (output_prefix, variable)), index=True)
            enrichr_enr.to_csv(
                os.path.join("%s.%s.diff_regions.enrichr.csv" % (output_prefix, variable)), index=False)

            # Get unique ids
            lola_enr["label"] = lola_enr.apply(lambda x: "-".join([str(i) for i in x[["description", "cellType", "tissue", "antibody", "treatment", "dataSource"]].tolist()]), axis=1)

            # Get significant
            lola_sig = lola_enr[(lola_enr["pValueLog"] > 1.3)]
            motif_sig = motif_enr[(motif_enr["p_value"] < 0.05)]
            enrichr_sig = enrichr_enr[(enrichr_enr["adjusted_p_value"] < 0.05) & (enrichr_enr["gene_set_library"] != "ChEA_2015")]

            # Show the top 30 most enriched LOLA sets per region set
            for i, comparison in enumerate(sorted(lola_sig["comparison"].unique())):
                lola_comparison = lola_sig[lola_sig["comparison"] == comparison]
                lola_comparison = lola_comparison.sort_values('pValueLog', ascending=False)

                lola_comparison2 = lola_comparison.head(30)

                fig, axis = plt.subplots(1)
                g = sns.barplot(x='label', y='pValueLog', hue="collection", data=lola_comparison2, estimator=max, ax=axis)
                for item in g.get_xticklabels():
                    item.set_rotation(90)
                sns.despine(fig)
                fig.savefig(os.path.join("%s.%s.diff_regions.lola.%s.svg" % (output_prefix, variable, comparison)), bbox_inches="tight")

            # Motifs - show the top 30 most enriched
            for i, comparison in enumerate(sorted(motif_sig["comparison"].unique())):
                motif_comparison = motif_sig[motif_sig["comparison"] == comparison]
                motif_comparison["p_value"] = -np.log10(motif_comparison["p_value"])
                motif_comparison = motif_comparison.sort_values('p_value', ascending=False)

                motif_comparison2 = motif_comparison.head(30)

                fig, axis = plt.subplots(1)
                g = sns.barplot(x='motifs', y='p_value', data=motif_comparison2, estimator=max, ax=axis)
                for item in g.get_xticklabels():
                    item.set_rotation(90)
                sns.despine(fig)
                fig.savefig(os.path.join("%s.%s.diff_regions.motifs.%s.svg" % (output_prefix, variable, comparison)), bbox_inches="tight")

            # Enrichr - show the top 30 most enriched per region set
            for i, comparison in enumerate(sorted(enrichr_sig["comparison"].unique())):
                enrichr_comparison = enrichr_sig[enrichr_sig["comparison"] == comparison]
                enrichr_comparison = enrichr_comparison.sort_values('adjusted_p_value', ascending=True)

                for i, gene_set_library in enumerate(sorted(enrichr_comparison["gene_set_library"].unique())):
                    enrichr_comparison2 = enrichr_comparison[enrichr_comparison["gene_set_library"] == gene_set_library].head(30)
                    enrichr_comparison2["adjusted_p_value"] = -np.log10(enrichr_comparison2["adjusted_p_value"])

                    fig, axis = plt.subplots(1)
                    g = sns.barplot(x='description', y='adjusted_p_value', data=enrichr_comparison2, estimator=max, ax=axis)
                    for item in g.get_xticklabels():
                        item.set_rotation(90)
                    sns.despine(fig)
                    fig.savefig(os.path.join("%s.%s.diff_regions.enrichr.%s.%s.svg" % (output_prefix, variable, comparison, gene_set_library)), bbox_inches="tight")


def add_args(parser):
    """
    Options for project and pipelines.
    """
    # Behaviour
    parser.add_argument("-g", "--generate", dest="generate", action="store_true",
                        help="Should we generate data and plots? Default=False")

    return parser


def count_reads_in_intervals(bam, intervals):
    """
    Counts reads in a iterable holding strings
    representing genomic intervals of the type chrom:start-end.
    """
    counts = dict()

    bam = pysam.Samfile(bam, 'rb')

    chroms = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]

    for interval in intervals:
        if interval.split(":")[0] not in chroms:
            continue
        counts[interval] = bam.count(region=interval)
    bam.close()

    return counts


def normalize_quantiles_r(array):
    # install package
    # R
    # source('http://bioconductor.org/biocLite.R')
    # biocLite('preprocessCore')

    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    robjects.r('require("preprocessCore")')
    normq = robjects.r('normalize.quantiles')
    return np.array(normq(array))


def name_to_repr(name):
    return "_".join([name.split("_")[0]] + [name.split("_")[2]] + name.split("_")[3:4])


def name_to_id(name):
    """This returns joined patient and sample IDs"""
    return "_".join([name.split("_")[2]] + name.split("_")[3:4])


def name_to_patient_id(name):
    return name.split("_")[2]


def name_to_sample_id(name):
    return name.split("_")[3]


def samples_to_color(samples, trait="IGHV"):
    # unique color per patient
    if trait == "patient":
        patients = set([sample.patient_id for sample in samples])
        color_dict = cm.Paired(np.linspace(0, 1, len(patients)))
        color_dict = dict(zip(patients, color_dict))
        return [color_dict[sample.patient_id] for sample in samples]
    # rainbow (unique color per sample)
    elif trait == "unique_sample":
        return cm.Paired(np.linspace(0, 1, len(samples)))
        # gender
    elif trait == "gender":
        colors = list()
        for sample in samples:
            if sample.patient_gender == "F":
                colors.append('red')
            elif sample.patient_gender == "M":
                colors.append('blue')
            else:
                colors.append('gray')
        return colors
    # disease at diagnosis time
    elif trait == "disease":
        colors = list()
        for sample in samples:
            if sample.diagnosis_disease == "CLL":
                colors.append('#A6CEE3')
            elif sample.diagnosis_disease == "MBL":
                colors.append('#F17047')
            elif sample.diagnosis_disease == "SLL":
                colors.append('#482115')
            else:
                colors.append('grey')
        return colors
    # dependent on trait threshold
    elif trait in ["IGHV", "under_treatment"]:
        # This uses sns colorblind pallete
        colors = list()
        for sample in samples:
            if getattr(sample, trait) == 1:
                colors.append(sns.color_palette("colorblind")[0])  # blue #0072b2
            elif getattr(sample, trait) == 0:
                colors.append(sns.color_palette("colorblind")[2])  # vermillion #d55e00
            else:
                colors.append('gray')
        return colors
    # unique color per patient
    if trait in ["timepoint_name"]:
        uniques = set([getattr(sample, trait) for sample in samples])
        color_dict = cm.Paired(np.linspace(0, 1, len(uniques)))
        color_dict = dict(zip(uniques, color_dict))
        return [color_dict[getattr(sample, trait)] for sample in samples]
    # IGHV homology color scale from min to max
    if trait in ["ighv_homology"]:
        vmin = min([getattr(s, trait) for s in samples])
        # This uses sns summer colormap
        cmap = plt.get_cmap('summer')
        # scale colormap to min and max ighv homology
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=100)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        # get colors acordingly
        colors = list()
        for sample in samples:
            if vmin <= getattr(sample, trait) <= 100:
                colors.append(m.to_rgba(getattr(sample, trait)))
            else:
                colors.append('gray')
        return colors
    if trait in ["batch"]:
        import re
        # get set of experiemnt batches
        n = set([re.sub(r"(ATAC\d+)[s-].*", r"\1", s.experiment_name) for s in samples])
        # get mapping of batch:color
        cmap = dict(zip(n, sns.color_palette("cubehelix", len(n))))
        colors = list()
        for s in samples:
            colors.append(cmap[re.sub(r"(ATAC\d+)[s-].*", r"\1", s.experiment_name)])
        return colors
    else:
        raise ValueError("trait %s is not valid" % trait)


def all_sample_colors(samples, order=""):
    return [
        samples_to_color(samples, "patient"),
        samples_to_color(samples, "gender"),
        samples_to_color(samples, "disease"),
        samples_to_color(samples, "IGHV"),
        samples_to_color(samples, "under_treatment"),
        samples_to_color(samples, "timepoint_name"),
        samples_to_color(samples, "ighv_homology"),
        samples_to_color(samples, "batch"),
    ]


def samples_to_symbol(samples, method="unique"):
    from itertools import cycle
    valid = ['D', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']
    c = cycle([x for x in matplotlib.markers.MarkerStyle.markers.items() if x[0] in valid])

    # unique color per patient
    if method == "unique":
        # per patient
        patients = set(sample.patient_id for sample in samples)
        symbol_dict = [c.next()[0] for _ in range(len(patients))]
        symbol_dict = dict(zip(patients, symbol_dict))
        return [symbol_dict[sample.patient_id] for sample in samples]
    # rainbow (unique color per sample)
    elif method == "unique_sample":
        return [c.next()[0] for sample in samples]
    else:
        raise ValueError("Method %s is not valid" % method)


def annotate_clinical_traits(samples):
    # Annotate traits
    chemo_drugs = ["Chlor", "Chlor R", "B Of", "BR", "CHOPR"]  # Chemotherapy
    target_drugs = ["Alemtuz", "Ibrutinib"]  # targeted treatments
    muts = ["del13", "del11", "tri12", "del17"]  # chrom abnorms
    muts += ["SF3B1", "ATM", "NOTCH1", "BIRC3", "BCL2", "TP53", "MYD88", "CHD2", "NFKIE"]  # mutations
    for s in samples:
        # Gender
        s.gender = 1 if s.patient_gender == "M" else 0 if s.patient_gender == "F" else pd.np.nan
        # IGHV mutation status
        s.IGHV = s.ighv_mutation_status

    # Annotate samples which are under treament but with different types
    for sample in samples:
        if not sample.under_treatment:
            sample.chemo_treated = pd.np.nan
            sample.target_treated = pd.np.nan
        else:
            sample.chemo_treated = 1 if sample.treatment_regimen in chemo_drugs else 0
            sample.target_treated = 1 if sample.treatment_regimen in target_drugs else 0
        for mut in muts:
            setattr(sample, mut, 1 if sample.mutations is not pd.np.nan and mut in str(sample.mutations) else 0)

    return samples


def annotate_disease_treatments(samples):
    """
    Annotate samples with timepoint, treatment_status, treatment_regimen
    """
    def string_to_date(string):
        if type(string) is str:
            if len(string) == 10:
                return pd.to_datetime(string, format="%d/%m/%Y")
            if len(string) == 7:
                return pd.to_datetime(string, format="%m/%Y")
            if len(string) == 4:
                return pd.to_datetime(string, format="%Y")
        return pd.NaT

    new_samples = list()

    for sample in samples:
        if sample.cell_line == "CLL":
            # Get sample collection date
            sample.collection_date = string_to_date(sample.sample_collection_date)
            # Get diagnosis date
            sample.diagnosis_date = string_to_date(sample.diagnosis_date)
            # Get diagnosis disease
            sample.primary_CLL = 1 if sample.diagnosis_disease == "CLL" else 0  # binary label useful for later

            # Get time since diagnosis
            sample.time_since_diagnosis = sample.collection_date - sample.diagnosis_date

            # Annotate treatment type, time since treatment
            if sample.under_treatment:
                sample.time_since_treatment = sample.collection_date - string_to_date(sample.treatment_date)

        # Append sample
        new_samples.append(sample)
    return new_samples


def annotate_samples(samples, attrs):
    new_samples = list()
    for sample in samples:
        # If any attribute is not set, set to NaN
        for attr in attrs:
            if not hasattr(sample, attr):
                setattr(sample, attr, pd.np.nan)
        new_samples.append(sample)

    # read in file with IGHV group of samples selected for ChIPmentation
    selected = pd.read_csv(os.path.join("metadata", "selected_samples.tsv"), sep="\t").astype(str)
    # annotate samples with the respective IGHV group
    for sample in samples:
        group = selected[
            (selected["patient_id"].astype(str) == str(sample.patient_id)) &
            (selected["sample_id"].astype(str) == str(sample.sample_id))
        ]["sample_cluster"]
        if len(group) == 1:
            sample.ighv_group = group.squeeze()
        else:
            sample.ighv_group = pd.np.nan

    return annotate_clinical_traits(annotate_disease_treatments(new_samples))


def state_enrichment_overlap(n=100):
    all_states = "all_states_all_lines.bed"

    # states of interest:
    # get names of all states
    states = pd.read_csv(all_states, sep="\t", header=None)[3].unique().tolist()

    # loop through states, merge intervals, count number intersepting CLL peaks, and not intersepting
    cll_ints = pybedtools.BedTool(os.path.join("data", "cll-ibrutinib_peaks.bed"))

    df = pd.DataFrame()
    for state in states[-3:]:
        state_bed = "{0}.bed".format(state)
        os.system("grep {0} {1} > {2}".format(state, all_states, state_bed))

        # merge all intervals (of the same type across cell types)
        state_ints = pybedtools.BedTool(state_bed).sort().merge()

        total = len(state_ints)
        pos = len(state_ints.intersect(cll_ints))

        # get mean of `n` shuffled cll sites
        background = list()
        for i in range(n):
            background.append(len(state_ints.intersect(cll_ints.shuffle(genome='hg19', chrom=True))))

        # append to df
        df = df.append(pd.Series([total, pos, np.round(np.mean(background))]), ignore_index=True)
    df.index = states
    df.columns = ['total', 'pos', 'background']

    df['state'] = df.index

    df.to_csv("chrom_state_overlap_all.csv", index=False)

    df2 = pd.melt(df, id_vars='state')

    df2.sort(['variable', 'value'], inplace=True)

    fig, axis = plt.subplots(1)
    sns.barplot(data=df2, x='state', y='value', hue='variable', ax=axis)
    fig.savefig("chrom_state_overlap_all.svg", bbox_inches='tight')

    # fraction of total
    df['posF'] = df['pos'] / df['total']
    df['backgroundF'] = df['background'] / df['total']
    df3 = pd.melt(df[["state", "posF", "backgroundF"]], id_vars='state')
    df3.sort(['variable', 'value'], inplace=True)

    fig, axis = plt.subplots(1)
    sns.barplot(data=df3, x='state', y='value', hue='variable', ax=axis)
    fig.savefig("chrom_state_overlap_all.fraction_total.svg", bbox_inches='tight')

    # fraction of total enriched over background
    df['foldF'] = df['posF'] / df['backgroundF']
    df4 = pd.melt(df[["state", "foldF"]], id_vars='state')
    df4.sort(['variable', 'value'], inplace=True)

    fig, axis = plt.subplots(1)
    sns.barplot(data=df4, x='state', y='value', hue='variable', ax=axis)
    fig.savefig("chrom_state_overlap_all.fraction_total.enriched.svg", bbox_inches='tight')

    # same with log2
    df['foldFlog'] = np.log2(df['foldF'])
    df5 = pd.melt(df[["state", "foldFlog"]], id_vars='state')
    df5.sort(['variable', 'value'], inplace=True)

    fig, axis = plt.subplots(1)
    sns.barplot(data=df5, x='state', y='value', hue='variable', ax=axis)
    fig.savefig("chrom_state_overlap_all.fraction_total.enriched.log.svg", bbox_inches='tight')


def DESeq_analysis(counts_matrix, experiment_matrix, variable, covariates, output_prefix, alpha=0.05):
    """
    """
    import rpy2.robjects as robj
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    run = robj.r("""
        run = function(countData, colData, variable, covariates, output_prefix, alpha) {
            library(DESeq2)

            colData$replicate = as.factor(colData$replicate)

            design = as.formula((paste("~", covariates, variable)))
            print(design)
            dds <- DESeqDataSetFromMatrix(
                countData = countData, colData = colData,
                design)

            dds <- DESeq(dds)
            save(dds, file=paste0("results/", output_prefix, ".deseq_dds_object.Rdata"))
            # load("deseq_gene_expresion.deseq_dds_object.Rdata")

            # Get group means
            # get groups with one sample and set mean to the value of that sample
            single_levels = names(table(colData[, variable])[table(colData[, variable]) == 1])
            single_levels_values = sapply(
                single_levels,
                function(lvl) counts(dds, normalized=TRUE)[, dds[, variable] == lvl]
            )
            # all others, get sample means
            multiple_levels = names(table(colData[, variable])[table(colData[, variable]) > 1])
            multiple_levels_values = sapply(
                multiple_levels,
                function(lvl) rowMeans(counts(dds, normalized=TRUE)[, colData[, variable] == lvl])
            )
            group_means = cbind(single_levels_values, multiple_levels_values)
            write.table(group_means, paste0("results/", output_prefix, ".", variable, ".group_means.csv"), sep=",")

            # pairwise combinations
            combs = combn(sort(unique(colData[, variable]), descending=FALSE), 2)

            # keep track of output files
            result_files = list()

            for (i in 1:ncol(combs)) {

                cond1 = as.character(combs[1, i])
                cond2 = as.character(combs[2, i])
                contrast = c(variable, cond1, cond2)
                print(contrast)

                # get results
                res <- results(dds, contrast=contrast, alpha=alpha, independentFiltering=FALSE)
                res <- as.data.frame(res)

                # append group means
                res <- cbind(group_means, res)

                # append to results
                comparison_name = paste(cond1, cond2, sep="-")
                output_name = paste0("results/", output_prefix, ".", variable, ".", comparison_name, ".csv")
                res["comparison"] = comparison_name

                # coherce to character
                res = data.frame(lapply(res, as.character), stringsAsFactors=FALSE)

                # add index
                rownames(res) = rownames(countData)

                write.table(res, output_name, sep=",")
                result_files[i] = output_name
            }
        return(result_files)
        }

    """)

    # replace names
    counts_matrix.columns = ["S" + str(i) for i in range(len(counts_matrix.columns))]
    experiment_matrix.index = ["S" + str(i) for i in range(len(experiment_matrix.index))]
    experiment_matrix.index.name = "sample"

    # save to disk just in case
    counts_matrix.to_csv(os.path.join(os.path.dirname(output_prefix), "counts_matrix.csv"), index=True)
    experiment_matrix.to_csv(os.path.join(os.path.dirname(output_prefix), "experiment_matrix.csv"), index=True)

    result_files = run(counts_matrix, experiment_matrix, variable, " + ".join(covariates) + " + " if len(covariates) > 0 else "", output_prefix, alpha)

    # concatenate all files
    results = pd.DataFrame()
    for result_file in result_files:
        df = pd.read_csv(result_file)
        df.index = counts_matrix.index

        results = results.append(df)

    # save all
    results.to_csv(os.path.join(output_prefix + ".%s.csv" % variable), index=True)

    # return
    return results


def lola(bed_files, universe_file, output_folder):
    """
    Performs location overlap analysis (LOLA) on bedfiles with regions sets.
    """
    import rpy2.robjects as robj

    run = robj.r("""
        function(bedFiles, universeFile, outputFolder) {
            library("LOLA")

            userUniverse  <- LOLA::readBed(universeFile)

            dbPath1 = "/data/groups/lab_bock/shared/resources/regions/LOLACore/hg19/"
            dbPath2 = "/data/groups/lab_bock/shared/resources/regions/customRegionDB/hg19/"
            regionDB = loadRegionDB(c(dbPath1, dbPath2))

            if (typeof(bedFiles) == "character") {
                userSet <- LOLA::readBed(bedFiles)
                lolaResults = runLOLA(list(userSet), userUniverse, regionDB, cores=12)
                lolaResults[order(support, decreasing=TRUE), ]
                writeCombinedEnrichment(lolaResults, outFolder=outputFolder)
            } else if (typeof(bedFiles) == "double") {
                for (bedFile in bedFiles) {
                    userSet <- LOLA::readBed(bedFile)
                    lolaResults = runLOLA(list(userSet), userUniverse, regionDB, cores=12)
                    lolaResults[order(support, decreasing=TRUE), ]
                    writeCombinedEnrichment(lolaResults, outFolder=outputFolder)
                }
            }
        }
    """)

    # convert the pandas dataframe to an R dataframe
    run(bed_files, universe_file, output_folder)

    # for F in `find . -iname *_regions.bed`
    # do
    #     if  [ ! -f `dirname $F`/allEnrichments.txt ]; then
    #         echo $F
    #         sbatch -J LOLA_${F} -o ${F/_regions.bed/_lola.log} ~/run_LOLA.sh $F ~/projects/cll-ibrutinib/results/cll-ibrutinib.bed hg19 `dirname $F`
    #     fi
    # done


def bed_to_fasta(bed_file, fasta_file):
    # write name column
    bed = pd.read_csv(bed_file, sep='\t', header=None)
    bed['name'] = bed[0] + ":" + bed[1].astype(str) + "-" + bed[2].astype(str)
    bed[1] = bed[1].astype(int)
    bed[2] = bed[2].astype(int)
    bed.to_csv(bed_file + ".tmp.bed", sep='\t', header=None, index=False)

    # do enrichment
    cmd = "twoBitToFa ~/resources/genomes/hg19/hg19.2bit -bed={0} {1}".format(bed_file + ".tmp.bed", fasta_file)

    os.system(cmd)
    # os.system("rm %s" % bed_file + ".tmp.bed")


def meme_ame(input_fasta, output_dir, background_fasta=None):
    # shuffle input in no background is provided
    if background_fasta is None:
        shuffled = input_fasta + ".shuffled"
        cmd = """
        fasta-dinucleotide-shuffle -c 1 -f {0} > {1}
        """.format(input_fasta, shuffled)
        os.system(cmd)

    cmd = """
    ame --bgformat 1 --scoring avg --method ranksum --pvalue-report-threshold 0.05 \\
    --control {0} -o {1} {2} ~/resources/motifs/motif_databases/HUMAN/HOCOMOCOv9.meme
    """.format(background_fasta if background_fasta is not None else shuffled, output_dir, input_fasta)
    os.system(cmd)

    os.system("rm %s" % shuffled)

    # for F in `find . -iname *fa`
    # do
    #     if  [ ! -f `dirname $F`/ame.txt ]; then
    #         echo $F
    #         sbatch -J MEME-AME_${F} -o ${F/fa/ame.log} ~/run_AME.sh $F human
    #     fi
    # done


def parse_ame(ame_dir):

    with open(os.path.join(ame_dir, "ame.txt"), 'r') as handle:
        lines = handle.readlines()

    output = list()
    for line in lines:
        # skip header lines
        if line[0] not in [str(i) for i in range(10)]:
            continue

        # get motif string and the first half of it (simple name)
        motif = line.strip().split(" ")[5].split("_")[0]
        # get corrected p-value
        q_value = float(line.strip().split(" ")[-2])
        # append
        output.append((motif, q_value))

    return pd.Series(dict(output))


def enrichr(dataframe, gene_set_libraries=None, kind="genes"):
    """
    Use Enrichr on a list of genes (currently only genes supported through the API).
    """
    import json
    import requests

    ENRICHR_ADD = 'http://amp.pharm.mssm.edu/Enrichr/addList'
    ENRICHR_RETRIEVE = 'http://amp.pharm.mssm.edu/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'

    if gene_set_libraries is None:
        gene_set_libraries = [
            "GO_Biological_Process_2015",
            "GO_Molecular_Function_2015",
            "GO_Cellular_Component_2015",
            # "ChEA_2015",
            "KEGG_2016",
            # "ESCAPE",
            # "Epigenomics_Roadmap_HM_ChIP-seq",
            # "ENCODE_TF_ChIP-seq_2015",
            # "ENCODE_Histone_Modifications_2015",
            "OMIM_Expanded",
            "TF-LOF_Expression_from_GEO",
            "Single_Gene_Perturbations_from_GEO_down",
            "Single_Gene_Perturbations_from_GEO_up",
            "Disease_Perturbations_from_GEO_down",
            "Disease_Perturbations_from_GEO_up",
            "Drug_Perturbations_from_GEO_down",
            "Drug_Perturbations_from_GEO_up",
            "WikiPathways_2016",
            "Reactome_2016",
            "BioCarta_2016",
            "NCI-Nature_2016"
        ]
    results = pd.DataFrame()
    for gene_set_library in gene_set_libraries:
        print("Using enricher on %s gene set library." % gene_set_library)

        if kind == "genes":
            # Build payload with bed file
            attr = "\n".join(dataframe["gene_name"].dropna().tolist())
        elif kind == "regions":
            # Build payload with bed file
            attr = "\n".join(dataframe[['chrom', 'start', 'end']].apply(lambda x: "\t".join([str(i) for i in x]), axis=1).tolist())

        payload = {
            'list': (None, attr),
            'description': (None, gene_set_library)
        }
        # Request adding gene set
        response = requests.post(ENRICHR_ADD, files=payload)
        if not response.ok:
            raise Exception('Error adding gene list')

        # Track gene set ID
        user_list_id = json.loads(response.text)['userListId']

        # Request enriched sets in gene set
        response = requests.get(
            ENRICHR_RETRIEVE + query_string % (user_list_id, gene_set_library)
        )
        if not response.ok:
            raise Exception('Error fetching enrichment results')

        # Get enriched sets in gene set
        res = json.loads(response.text)
        # If there's no enrichemnt, continue
        if len(res) < 0:
            continue

        # Put in dataframe
        res = pd.DataFrame([pd.Series(s) for s in res[gene_set_library]])
        res.columns = ["rank", "description", "p_value", "z_score", "combined_score", "genes", "adjusted_p_value"]

        # Remember gene set library used
        res["gene_set_library"] = gene_set_library

        # Append to master dataframe
        results = results.append(res, ignore_index=True)

    return results


def characterize_regions_structure(df, prefix, output_dir, universe_df=None):
    # use all sites as universe
    if universe_df is None:
        universe_df = pd.read_csv(os.path.join("results", "cll-ibrutinib.coverage_qnorm.log2.annotated.tsv"), sep="\t", index_col=0)

    # make output dirs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # compare genomic regions and chromatin_states
    enrichments = pd.DataFrame()
    for i, var in enumerate(['genomic_region', 'chromatin_state']):
        # prepare:
        # separate comma-delimited fields:
        df_count = Counter(df[var].str.split(',').apply(pd.Series).stack().tolist())
        df_universe_count = Counter(universe_df[var].str.split(',').apply(pd.Series).stack().tolist())

        # divide by total:
        df_count = {k: v / float(len(df)) for k, v in df_count.items()}
        df_universe_count = {k: v / float(len(universe_df)) for k, v in df_universe_count.items()}

        # join data, sort by subset data
        both = pd.DataFrame([df_count, df_universe_count], index=['subset', 'all']).T
        both = both.sort("subset")
        both['region'] = both.index
        data = pd.melt(both, var_name="set", id_vars=['region']).replace(np.nan, 0)

        # sort for same order
        data.sort('region', inplace=True)

        # g = sns.FacetGrid(col="region", data=data, col_wrap=3, sharey=True)
        # g.map(sns.barplot, "set", "value")
        # plt.savefig(os.path.join(output_dir, "%s_regions.%s.svg" % (prefix, var)), bbox_inches="tight")

        fc = pd.DataFrame(np.log2(both['subset'] / both['all']), columns=['value'])
        fc['variable'] = var

        # append
        enrichments = enrichments.append(fc)

    # save
    enrichments.to_csv(os.path.join(output_dir, "%s_regions.region_enrichment.csv" % prefix), index=True)


def characterize_regions_function(df, output_dir, prefix, data_dir="data", universe_file=None):
    # use all sites as universe
    if universe_file is None:
        universe_file = os.path.join(data_dir, "cll-ibrutinib.bed")

    # make output dirs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save to bed
    bed_file = os.path.join(output_dir, "%s_regions.bed" % prefix)
    df[['chrom', 'start', 'end']].to_csv(bed_file, sep="\t", header=False, index=False)
    # save as tsv
    tsv_file = os.path.join(output_dir, "%s_regions.tsv" % prefix)
    df[['chrom', 'start', 'end']].reset_index().to_csv(tsv_file, sep="\t", header=False, index=False)

    # export ensembl gene names
    df['gene_name'].str.split(",").apply(pd.Series, 1).stack().drop_duplicates().to_csv(os.path.join(output_dir, "%s_genes.symbols.txt" % prefix), index=False)
    # export ensembl gene names
    df['ensembl_gene_id'].str.split(",").apply(pd.Series, 1).stack().drop_duplicates().to_csv(os.path.join(output_dir, "%s_genes.ensembl.txt" % prefix), index=False)

    # Motifs
    # de novo motif finding - enrichment
    fasta_file = os.path.join(output_dir, "%s_regions.fa" % prefix)
    bed_to_fasta(bed_file, fasta_file)

    meme_ame(fasta_file, output_dir)

    # Lola
    try:
        lola(bed_file, universe_file, output_dir)
    except:
        print("LOLA analysis for %s failed!" % prefix)

    # Enrichr
    results = enrichr(df[['chrom', 'start', 'end', "gene_name"]])

    # Save
    results.to_csv(os.path.join(output_dir, "%s_regions.enrichr.csv" % prefix), index=False, encoding='utf-8')


def loci_plots(self, samples):

    #
    bed = os.path.join(self.data_dir, "deseq", "deseq.ibrutinib_treatment.diff_regions", "deseq.ibrutinib_treatment.diff_regions_regions.bed")
    cmd = """fluff bandplot -S -f {bed} -d {bams} -o {output}""".format(
        bed=bed,
        bams=" ".join([s.filtered for s in samples]),
        output="differential_regions.fluff_bandplot.png")
    os.system(cmd)

    #
    bed = os.path.join(self.data_dir, "deseq", "deseq.ibrutinib_treatment.diff_regions", "deseq.ibrutinib_treatment.diff_regions_regions.bed")
    cmd = """fluff bandplot -f {bed} -d {bams} -o {output}""".format(
        bed=bed,
        bams=" ".join([s.filtered for s in samples]),
        output="differential_regions.fluff_bandplot.png")
    os.system(cmd)

    #
    cmd = """
    fluff heatmap -f {bed} -d {bams} -C k -k 5 -g -M Pearson -o {output}
    """.format(
        bed=bed,
        bams=" ".join([s.filtered for s in samples]),
        output="differential_regions.fluff_heatmap.png")
    os.system(cmd)

    # A few loci
    cmd = """fluff profile -i chr1:23876028-23894677 -d {bams} -o profile_chr1_23876028_23894677
    """.format(bams=" ".join([s.filtered for s in samples]))
    os.system(cmd)

    cmd = """fluff profile -S {scale} -i chr1:23857378-23913327 -d {bams} -o profile_chr1_23857378_23913327-S
    """.format(bams=" ".join([s.filtered for s in samples]), scale=",".join(["200"] * len(samples)))
    os.system(cmd)

    cmd = """fluff profile -s -i chr1:23857378-23913327 -d {bams} -o profile_chr1_23857378_23913327-s
    """.format(bams=" ".join([s.filtered for s in samples]))
    os.system(cmd)


def gene_level_oppeness(self, samples):
    """
    Plot openness of regulatory elements of relevant genes.
    """
    sns.set(style="white", palette="pastel", color_codes=True)

    # Read in genes from supplementary table
    genes = pd.read_csv(os.path.join("metadata", "Han_et_al_supplement.csv"))
    genes["log2_fold_change"] = np.log2(genes["fold_change"])
    # map to ensembl ids
    # add gene name and ensemble_gene_id
    ensembl_gtn = pd.read_table(os.path.join(self.data_dir, "external", "ensemblToGeneName.txt"), header=None)
    ensembl_gtn.columns = ['ensembl_transcript_id', 'gene_symbol']
    ensembl_gtp = pd.read_table(os.path.join(self.data_dir, "external", "ensGtp.txt"), header=None)[[0, 1]]
    ensembl_gtp.columns = ['ensembl_gene_id', 'ensembl_transcript_id']
    ensembl = pd.merge(ensembl_gtn, ensembl_gtp)

    genes = pd.merge(genes, ensembl[['gene_symbol', 'ensembl_gene_id']].drop_duplicates(), how="left")

    # Get gene-level measurements of accessibility (simply the mean of all regulatory elements across all samples)
    a = self.coverage_qnorm_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples]].mean(axis=1)

    # Get gene-level measurements of accessibility dependent on treatment
    u = self.coverage_qnorm_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples if not s.under_treatment]].mean(axis=1)
    t = self.coverage_qnorm_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples if s.under_treatment]].mean(axis=1)

    # get indices of genes with fold > 1
    i = a[(abs(u - t) > 1)].index
    g = genes["ensembl_gene_id"]
    gu = genes[genes["log2_fold_change"] > 0]["ensembl_gene_id"]
    gd = genes[genes["log2_fold_change"] < 0]["ensembl_gene_id"]

    # Plot
    fig, axis = plt.subplots(2)
    # MA plot
    axis[0].scatter(a, (u - t))
    axis[0].scatter(a.ix[gu], (u.ix[gu] - t.ix[gu]), color="red")
    axis[0].scatter(a.ix[gd], (u.ix[gd] - t.ix[gd]), color="blue")
    axis[0].scatter(a.ix[i], (u.ix[i] - t.ix[i]), color="green")

    # Scatter
    axis[1].scatter(u, t)
    axis[1].scatter(u.ix[gu], t.ix[gu], color="red")
    axis[1].scatter(u.ix[gd], t.ix[gd], color="blue")
    axis[1].scatter(u.ix[i], t.ix[i], color="green")
    fig.savefig(os.path.join(self.results_dir, "gene_level.accessibility.svg"), bbox_inches="tight")

    # plot expression vs accessibility
    fig, axis = plt.subplots(1)
    axis.scatter(genes["log2_fold_change"], (u.ix[g] - t.ix[g]))
    axis.set_xlabel("Fold change gene expression (log2)")
    axis.set_ylabel("Difference in mean accessibility")
    fig.savefig(os.path.join(self.results_dir, "EGCG_targets.expression_vs_accessibility.svg"), bbox_inches="tight")

    # plot expression vs accessibility
    t2 = self.coverage_qnorm_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples if s.timepoint_name == 'EGCG_100uM']].mean(axis=1)

    fig, axis = plt.subplots(1)
    axis.scatter(genes["log2_fold_change"], (u.ix[g] - t2.ix[g]))
    axis.set_xlabel("Fold change gene expression (log2)")
    axis.set_ylabel("Difference in mean accessibility")
    fig.savefig(os.path.join(self.results_dir, "EGCG_targets.expression_vs_accessibility.untreated_EGCG_100uM.svg"), bbox_inches="tight")

    # Patient-specific
    n = set([s.patient_id for s in samples])
    fig, axis = plt.subplots(len(n) / 2, 2, figsize=(15, 15), sharex=True, sharey=True)
    axis = axis.flatten()
    for i, p in enumerate(n):
        patient = [s for s in samples if s.patient_id == p]
        u2 = self.coverage_qnorm_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in patient if s.timepoint_name == 'control']].mean(axis=1)
        t2 = self.coverage_qnorm_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in patient if s.timepoint_name == 'EGCG_100uM']].mean(axis=1)
        axis[i].scatter(genes["log2_fold_change"], (u2.ix[g] - t2.ix[g]))
        axis[i].set_title(p)
        axis[i].set_xlabel("Fold change gene expression (log2)")
        axis[i].set_ylabel("Difference in mean accessibility")
    fig.savefig(os.path.join(self.results_dir, "EGCG_targets.expression_vs_accessibility.untreated_EGCG_100uM.patient_specific.svg"), bbox_inches="tight")


def main():
    # Parse arguments
    parser = ArgumentParser(
        prog="pipelines",
        description="pipelines. Project management and sample loop."
    )
    parser = add_args(parser)
    args = parser.parse_args()

    # Start project
    prj = Project("metadata/project_config.yaml")
    prj.add_sample_sheet()
    # temporary:
    for sample in prj.samples:
        sample.peaks = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_peaks.narrowPeak")
        sample.summits = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_summits.bed")
        sample.mapped = os.path.join(sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.bam")
        sample.filtered = os.path.join(sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.filtered.bam")
        sample.coverage = os.path.join(sample.paths.sample_root, "coverage", sample.name + ".cov")

    # Only pass_qc samples
    prj.samples = [s for s in prj.samples if s.pass_qc and s.ibrutinib_cohort]

    # annotated samples with a few more things:
    prj.samples = annotate_samples(prj.samples, prj.sheet.df.columns.tolist())

    # Start analysis object
    analysis = Analysis(
        data_dir=os.path.join(".", "data"),
        results_dir=os.path.join(".", "results"),
        samples=prj.samples,
        pickle_file=os.path.join(".", "data", "analysis.pickle")
    )
    # pair analysis and Project
    analysis.prj = prj

    # GET CONSENSUS PEAK SET, ANNOTATE IT, PLOT FEATURES
    if args.generate:
        # Get consensus peak set from all samples
        analysis.get_consensus_sites(analysis.samples)
        analysis.calculate_peak_support(analysis.samples)

        # GET CHROMATIN OPENNESS MEASUREMENTS, PLOT
        # Get coverage values for each peak in each sample of ATAC-seq and ChIPmentation
        analysis.measure_coverage(analysis.samples)
        # normalize coverage values
        analysis.normalize_coverage_quantiles(analysis.samples)
        # Annotate peaks with closest gene
        analysis.get_peak_gene_annotation()
        # Annotate peaks with genomic regions
        analysis.get_peak_genomic_location()
        # Annotate peaks with ChromHMM state from CD19+ cells
        analysis.get_peak_chromatin_state()
        # Annotate peaks with closest gene, chromatin state,
        # genomic location, mean and variance measurements across samples
        analysis.annotate(analysis.samples)
    else:
        analysis.sites = pybedtools.BedTool(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.bed"))
        analysis.gene_annotation = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.gene_annotation.csv"))
        analysis.closest_tss_distances = pickle.load(open(os.path.join(analysis.results_dir, "cll-ibrutinib_peaks.closest_tss_distances.pickle"), "rb"))
        analysis.region_annotation = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.region_annotation.csv"))
        analysis.region_annotation_b = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.region_annotation_background.csv"))
        analysis.chrom_state_annotation = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.chromatin_state.csv"))
        analysis.chrom_state_annotation_b = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.chromatin_state_background.csv"))
        analysis.coverage = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.raw_coverage.tsv"), sep="\t", index_col=0)
        analysis.coverage_qnorm = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.coverage_qnorm.log2.tsv"), sep="\t", index_col=0)
        analysis.coverage_qnorm_annotated = pd.read_csv(os.path.join(analysis.data_dir, "cll-ibrutinib_peaks.coverage_qnorm.log2.annotated.tsv"), sep="\t", index_col=0)

    # Plots
    # plot general peak set features
    analysis.plot_peak_characteristics()
    # Plot rpkm features across peaks/samples
    analysis.plot_coverage()
    analysis.plot_variance()

    # Unsupervised analysis
    analysis.unsupervised(analysis.samples)

    # Differential analysis
    analysis.differential_analysis([s for s in analysis.samples if s.ibrutinib_cohort == 1], "ibrutinib_treatment")
    analysis.differential_analysis(analysis.samples, "under_treatment")


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Program canceled by user!")
        sys.exit(1)
