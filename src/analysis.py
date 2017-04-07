#!/usr/bin/env python

"""
This is the main script of the cll-ibrutinib project.
"""
import matplotlib
matplotlib.use('Agg')

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
    def __init__(
            self,
            name="analysis",
            data_dir=os.path.join(".", "data"),
            results_dir=os.path.join(".", "results"),
            pickle_file=None,
            samples=None,
            prj=None,
            from_pickle=False,
            **kwargs):
        # parse kwargs with default
        self.name = name
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.samples = samples
        if pickle_file is None:
            pickle_file = os.path.join(results_dir, "analysis.{}.pickle".format(name))
        self.pickle_file = pickle_file

        for directory in [self.data_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # parse remaining kwargs
        self.__dict__.update(kwargs)

        # reload itself if required
        if from_pickle:
            self.__dict__.update(self.from_pickle().__dict__)

    @pickle_me
    def to_pickle(self):
        pass

    def from_pickle(self):
        return pickle.load(open(self.pickle_file, 'rb'))

    @pickle_me
    def get_consensus_sites(self, samples, region_type="peaks", extension=250):
        """Get consensus (union) sites across samples"""
        import re

        for i, sample in enumerate(samples):
            print(sample.name)
            # Get peaks
            try:
                if region_type == "summits":
                    peaks = pybedtools.BedTool(re.sub("_peaks.narrowPeak", "_summits.bed", sample.peaks)).slop(b=extension, genome=sample.genome)
                else:
                    peaks = pybedtools.BedTool(sample.peaks)
            except ValueError:
                print("Peaks for sample {} not found!".format(sample))
                continue
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
        sites.intersect(v=True, b=blacklist).filter(lambda x: x.chrom != 'chrM').saveas(os.path.join(self.results_dir, self.name + "_peak_set.bed"))

        # Read up again
        self.sites = pybedtools.BedTool(os.path.join(self.results_dir, self.name + "_peak_set.bed"))

    @pickle_me
    def set_consensus_sites(self, bed_file, overwrite=True):
        """Get consensus (union) sites across samples"""
        self.sites = pybedtools.BedTool(bed_file)
        if overwrite:
            self.sites.saveas(os.path.join(self.results_dir, self.name + "_peak_set.bed"))

    @pickle_me
    def calculate_peak_support(self, samples, region_type="peaks"):
        import re

        # calculate support (number of samples overlaping each merged peak)
        for i, sample in enumerate(samples):
            print(sample.name)
            if region_type == "summits":
                peaks = re.sub("_peaks.narrowPeak", "_summits.bed", sample.peaks)
            else:
                peaks = sample.peaks

            if i == 0:
                support = self.sites.intersect(peaks, wa=True, c=True)
            else:
                support = support.intersect(peaks, wa=True, c=True)

        try:
            support = support.to_dataframe()
        except:
            support.saveas("_tmp.peaks.bed")
            support = pd.read_csv("_tmp.peaks.bed", sep="\t", header=None)

        support.columns = ["chrom", "start", "end"] + [sample.name for sample in samples]
        support.to_csv(os.path.join(self.results_dir, self.name + "_peaks.binary_overlap_support.csv"), index=False)

        # get % of total consensus regions found per sample
        m = pd.melt(support, ["chrom", "start", "end"], var_name="sample_name")
        # groupby
        n = m.groupby("sample_name").apply(lambda x: len(x[x["value"] == 1]))

        # divide sum (of unique overlaps) by total to get support value between 0 and 1
        support["support"] = support[range(len(samples))].apply(lambda x: sum([i if i <= 1 else 1 for i in x]) / float(len(self.samples)), axis=1)
        # save
        support.to_csv(os.path.join(self.results_dir, self.name + "_peaks.support.csv"), index=False)

        self.support = support

    def get_supported_peaks(self, samples):
        """
        Mask peaks with 0 support in the given samples.
        Returns boolean pd.Series of length `peaks`.
        """
        # calculate support (number of samples overlaping each merged peak)
        return self.support[[s.name for s in samples]].sum(1) != 0

    @pickle_me
    def measure_coverage(self, samples):
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

        missing = [s for s in samples if not os.path.exists(s.filtered)]
        if len(missing) > 0:
            print("Samples have missing BAM file: {}".format(missing))
            samples = [s for s in samples if s not in missing]

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
        self.coverage.to_csv(os.path.join(self.results_dir, self.name + "_peaks.raw_coverage.csv"), index=True)

    @pickle_me
    def normalize_coverage_quantiles(self, samples):
        def normalize_quantiles_r(array):
            # install R package
            # source('http://bioconductor.org/biocLite.R')
            # biocLite('preprocessCore')

            import rpy2.robjects as robjects
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()

            robjects.r('require("preprocessCore")')
            normq = robjects.r('normalize.quantiles')
            return np.array(normq(array))

        # Quantifle normalization
        to_norm = self.coverage[[s.name for s in samples]]

        self.coverage_qnorm = pd.DataFrame(
            normalize_quantiles_r(to_norm.values),
            index=to_norm.index,
            columns=to_norm.columns
        )
        self.coverage_qnorm = self.coverage_qnorm.join(self.coverage[['chrom', 'start', 'end']])
        self.coverage_qnorm.to_csv(os.path.join(self.results_dir, self.name + "_peaks.coverage_qnorm.csv"), index=True)

    @pickle_me
    def get_peak_gccontent_length(self, bed_file=None, genome="hg19", fasta_file="/home/arendeiro/resources/genomes/{g}/{g}.fa"):
        """
        Bed file must be a 3-column BED!
        """
        if bed_file is None:
            sites = self.sites
        else:
            sites = pybedtools.BedTool(bed_file)

        nuc = sites.nucleotide_content(fi=fasta_file.format(g=genome)).to_dataframe(comment="#")[["score", "blockStarts"]]
        nuc.columns = ["gc_content", "length"]
        nuc.index = [str(i.chrom) + ":" + str(i.start) + "-" + str(i.stop) for i in sites]

        # get only the sites matching the coverage (not overlapping blacklist)
        self.nuc = nuc.ix[self.coverage.index]

        self.nuc.to_csv(os.path.join(self.results_dir, self.name + "_peaks.gccontent_length.csv"), index=True)

    @pickle_me
    def normalize_gc_content(self, samples):
        def cqn(cov, gc_content, lengths):
            # install R package
            # source('http://bioconductor.org/biocLite.R')
            # biocLite('cqn')
            import rpy2
            rpy2.robjects.numpy2ri.deactivate()

            import rpy2.robjects as robjects
            import rpy2.robjects.pandas2ri
            rpy2.robjects.pandas2ri.activate()

            robjects.r('require("cqn")')
            cqn = robjects.r('cqn')

            cqn_out = cqn(cov, x=gc_content, lengths=lengths)

            y_r = cqn_out[list(cqn_out.names).index('y')]
            y = pd.DataFrame(
                np.array(y_r),
                index=cov.index,
                columns=cov.columns)
            offset_r = cqn_out[list(cqn_out.names).index('offset')]
            offset = pd.DataFrame(
                np.array(offset_r),
                index=cov.index,
                columns=cov.columns)

            return y + offset

        if not hasattr(self, "nuc"):
            self.normalize_coverage_quantiles(samples)

        if not hasattr(self, "nuc"):
            self.get_peak_gccontent_length()

        to_norm = self.coverage_qnorm[[s.name for s in samples]]

        self.coverage_gc_corrected = (
            cqn(cov=to_norm, gc_content=self.nuc["gc_content"], lengths=self.nuc["length"])
            .join(self.coverage[['chrom', 'start', 'end']]))

        self.coverage_gc_corrected.to_csv(os.path.join(self.results_dir, self.name + "_peaks.coverage_gc_corrected.csv"), index=True)

    def get_peak_gene_annotation(self):
        """
        Annotates peaks with closest gene.
        Needs files downloaded by prepare_external_files.py
        """
        # create bedtool with hg19 TSS positions
        hg19_refseq_tss = pybedtools.BedTool(os.path.join(self.data_dir, "external", "refseq.refflat.tss.bed"))
        # get closest TSS of each cll peak
        gene_annotation = self.sites.closest(hg19_refseq_tss, d=True).to_dataframe()
        gene_annotation = gene_annotation[['chrom', 'start', 'end'] + gene_annotation.columns[-3:].tolist()]  # TODO: check this
        gene_annotation.columns = ['chrom', 'start', 'end', 'gene_name', "strand", 'distance']

        # aggregate annotation per peak, concatenate various genes (comma-separated)
        self.gene_annotation = gene_annotation.groupby(['chrom', 'start', 'end']).aggregate(lambda x: ",".join(set([str(i) for i in x]))).reset_index()

        # save to disk
        self.gene_annotation.to_csv(os.path.join(self.results_dir, self.name + "_peaks.gene_annotation.csv"), index=False)

        # save distances to all TSSs (for plotting)
        self.closest_tss_distances = gene_annotation['distance'].tolist()
        pickle.dump(self.closest_tss_distances, open(os.path.join(self.results_dir, self.name + "_peaks.closest_tss_distances.pickle"), 'wb'))

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
        self.region_annotation.to_csv(os.path.join(self.results_dir, self.name + "_peaks.region_annotation.csv"), index=False)
        self.region_annotation_b.to_csv(os.path.join(self.results_dir, self.name + "_peaks.region_annotation_background.csv"), index=False)

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
        self.chrom_state_annotation.to_csv(os.path.join(self.results_dir, self.name + "_peaks.chromatin_state.csv"), index=False)
        self.chrom_state_annotation_b.to_csv(os.path.join(self.results_dir, self.name + "_peaks.chromatin_state_background.csv"), index=False)

    @pickle_me
    def annotate(self, samples):
        # add closest gene
        self.coverage_annotated = pd.merge(
            self.coverage_gc_corrected,
            self.gene_annotation, on=['chrom', 'start', 'end'], how="left")
        # add genomic location
        self.coverage_annotated = pd.merge(
            self.coverage_annotated,
            self.region_annotation[['chrom', 'start', 'end', 'genomic_region']], on=['chrom', 'start', 'end'], how="left")
        # add chromatin state
        self.coverage_annotated = pd.merge(
            self.coverage_annotated,
            self.chrom_state_annotation[['chrom', 'start', 'end', 'chromatin_state']], on=['chrom', 'start', 'end'], how="left")

        # add support
        self.coverage_annotated = pd.merge(
            self.coverage_annotated,
            self.support[['chrom', 'start', 'end', 'support']], on=['chrom', 'start', 'end'], how="left")

        # calculate mean coverage
        self.coverage_annotated['mean'] = self.coverage_annotated[[sample.name for sample in samples]].mean(axis=1)
        # calculate coverage variance
        self.coverage_annotated['variance'] = self.coverage_annotated[[sample.name for sample in samples]].var(axis=1)
        # calculate std deviation (sqrt(variance))
        self.coverage_annotated['std_deviation'] = np.sqrt(self.coverage_annotated['variance'])
        # calculate dispersion (variance / mean)
        self.coverage_annotated['dispersion'] = self.coverage_annotated['variance'] / self.coverage_annotated['mean']
        # calculate qv2 (std / mean) ** 2
        self.coverage_annotated['qv2'] = (self.coverage_annotated['std_deviation'] / self.coverage_annotated['mean']) ** 2

        # calculate "amplitude" (max - min)
        self.coverage_annotated['amplitude'] = (
            self.coverage_annotated[[sample.name for sample in samples]].max(axis=1) -
            self.coverage_annotated[[sample.name for sample in samples]].min(axis=1)
        )

        # Pair indexes
        assert self.coverage.shape[0] == self.coverage_annotated.shape[0]
        self.coverage_annotated.index = self.coverage.index

        # Save
        self.coverage_annotated.to_csv(os.path.join(self.results_dir, self.name + "_peaks.coverage_qnorm.log2.annotated.csv"), index=True)

    @pickle_me
    def annotate_with_sample_metadata(
            self,
            attributes=[
                "sample_name", "cell_type", "patient_id", "clinical_centre", "timepoint_name", "patient_gender", "patient_age_at_collection",
                "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
                "del11q", "del13q", "del17p", "tri12", "p53",
                "time_since_treatment", "treatment_response"]):

        samples = [s for s in self.samples if s.name in self.coverage_annotated.columns]

        attrs = list()
        for attr in attributes:
            l = list()
            for sample in samples:  # keep order of samples in matrix
                try:
                    l.append(getattr(sample, attr))
                except AttributeError:
                    l.append(np.nan)
            attrs.append(l)

        # Generate multiindex columns
        index = pd.MultiIndex.from_arrays(attrs, names=attributes)
        self.accessibility = self.coverage_annotated[[s.name for s in samples]]
        self.accessibility.columns = index

        # Save
        self.accessibility.to_csv(os.path.join(self.results_dir, self.name + ".accessibility.annotated_metadata.csv"), index=True)

    def get_level_colors(self, index=None, levels=None, pallete="Paired", cmap="RdBu_r", nan_color=(0.662745, 0.662745, 0.662745, 0.5)):
        if index is None:
            index = self.accessibility.columns

        if levels is not None:
            index = index.droplevel([l.name for l in index.levels if l.name not in levels])

        _cmap = plt.get_cmap(cmap)
        _pallete = plt.get_cmap(pallete)

        colors = list()
        for level in index.levels:
            # determine the type of data in each level
            most_common = Counter([type(x) for x in level]).most_common()[0][0]
            print(level.name, most_common)

            # Add either colors based on categories or numerical scale
            if most_common in [int, float, np.float32, np.float64, np.int32, np.int64]:
                values = index.get_level_values(level.name)
                # Create a range of either 0-100 if only positive values are found
                # or symmetrically from the maximum absolute value found
                if not any(values.dropna() < 0):
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
                else:
                    r = max(abs(values.min()), abs(values.max()))
                    norm = matplotlib.colors.Normalize(vmin=-r, vmax=r)

                col = _cmap(norm(values))
                # replace color for nan cases
                col[np.where(index.get_level_values(level.name).to_series().isnull().tolist())] = nan_color
                colors.append(col.tolist())
            else:
                n = len(set(index.get_level_values(level.name)))
                # get n equidistant colors
                p = [_pallete(1. * i / n) for i in range(n)]
                color_dict = dict(zip(list(set(index.get_level_values(level.name))), p))
                # color for nan cases
                color_dict[np.nan] = nan_color
                col = [color_dict[x] for x in index.get_level_values(level.name)]
                colors.append(col)

        return colors

    # c = analysis.accessibility.corr()
    # c.index = analysis.accessibility.columns.get_level_values("sample_name")
    # c.columns = analysis.accessibility.columns.get_level_values("sample_name")
    # g = sns.clustermap(c, row_colors=get_level_colors(analysis), cmap='inferno', xticklabels=False)
    # for item in g.ax_heatmap.get_yticklabels():
    #     item.set_rotation(0)
    # g.ax_row_colors.text(0, -2, "asd", rotation=45)
    # g.ax_row_colors.text(1, -2, "asd", rotation=45)
    # g.ax_row_colors.text(2, -2, "asd", rotation=45)

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
        data = self.coverage_annotated.copy()

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
        df = pd.read_csv(os.path.join(self.data_dir, self.name + "_peaks.support.csv"))
        fig, axis = plt.subplots(1)
        sns.distplot(df["support"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "cll-ibrutinib.support.distplot.svg"), bbox_inches="tight")

        plt.close("all")

    def plot_coverage(self):
        data = self.coverage_annotated.copy()
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
            'ensembl_transcript_id', 'distance', 'support',
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

        g = sns.jointplot('mean', "dispersion", data=self.coverage_annotated, kind="kde")
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.dispersion.png"), bbox_inches="tight", dpi=300)

        g = sns.jointplot('mean', "qv2", data=self.coverage_annotated)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.qv2_vs_mean.png"), bbox_inches="tight", dpi=300)

        g = sns.jointplot('support', "qv2", data=self.coverage_annotated)
        g.fig.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.support_vs_qv2.png"), bbox_inches="tight", dpi=300)

        # Filter out regions which the maximum across all samples is below a treshold
        filtered = self.coverage_annotated[self.coverage_annotated[[sample.name for sample in samples]].max(axis=1) > 3]

        sns.jointplot('mean', "dispersion", data=filtered)
        plt.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.dispersion.filtered.png"), bbox_inches="tight", dpi=300)
        plt.close('all')
        sns.jointplot('mean', "qv2", data=filtered)
        plt.savefig(os.path.join(self.results_dir, "norm_counts_per_sample.support_vs_qv2.filtered.png"), bbox_inches="tight", dpi=300)

    def plot_sample_correlations(self):
        pass

    def unsupervised(
            self, samples, attributes=[
                "sample_name", "cell_type", "patient_id", "clinical_centre", "timepoint_name", "patient_gender", "patient_age_at_collection",
                "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
                "del11q", "del13q", "del17p", "tri12", "p53",
                "time_since_treatment", "treatment_response"],
            exclude=[]):
        """
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS
        from collections import OrderedDict
        import re
        import itertools
        from scipy.stats import kruskal
        from scipy.stats import pearsonr

        color_dataframe = pd.DataFrame(self.get_level_colors(levels=attributes), index=attributes, columns=[s.name for s in self.samples])

        # exclude samples if needed
        samples = [s for s in samples if s.name not in exclude]
        color_dataframe = color_dataframe[[s.name for s in samples]]
        sample_display_names = color_dataframe.columns.str.replace("_ATAC-seq", "").str.replace("_hg19", "")

        # exclude attributes if needed
        to_plot = attributes[:]
        to_exclude = ["patient_age_at_collection", "sample_name"]
        for attr in to_exclude:
            try:
                to_plot.pop(to_plot.index(attr))
            except:
                continue

        color_dataframe = color_dataframe.ix[to_plot]

        # All regions
        X = self.accessibility[[s.name for s in samples if s.name not in exclude]]

        # Pairwise correlations
        g = sns.clustermap(
            X.corr(), xticklabels=False, yticklabels=sample_display_names, annot=True,
            cmap="Spectral_r", figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        g.ax_heatmap.set_xlabel(None)
        g.ax_heatmap.set_ylabel(None)
        g.fig.savefig(os.path.join(self.results_dir, "{}.all_sites.corr.clustermap.svg".format(self.name)), bbox_inches='tight')

        # MDS
        mds = MDS(n_jobs=-1)
        x_new = mds.fit_transform(X.T)
        # transform again
        x = pd.DataFrame(x_new, index=X.columns.get_level_values("sample_name"), columns=["D{}".format(i) for i in range(x_new.shape[1])])
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)

        fig, axis = plt.subplots(1, len(to_plot), figsize=(4 * len(to_plot), 4 * 1))
        axis = axis.flatten()
        for i, attr in enumerate(to_plot):
            for j in range(len(xx)):
                try:
                    label = getattr(samples[j], to_plot[i])
                except AttributeError:
                    label = np.nan
                axis[i].scatter(xx.ix[j][0], xx.ix[j][1], s=50, color=color_dataframe.ix[attr][j], label=label)
            axis[i].set_title(to_plot[i])
            axis[i].set_xlabel("MDS 1")
            axis[i].set_ylabel("MDS 2")
            axis[i].set_xticklabels(axis[i].get_xticklabels(), visible=False)
            axis[i].set_yticklabels(axis[i].get_yticklabels(), visible=False)

            # Unique legend labels
            handles, labels = axis[i].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            if any([type(c) in [str, unicode] for c in by_label.keys()]) and len(by_label) <= 20:
                if not any([re.match("^\d", c) for c in by_label.keys()]):
                    axis[i].legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(self.results_dir, "{}.all_sites.mds.svg".format(self.name)), bbox_inches="tight")

        # PCA
        pca = PCA()
        x_new = pca.fit_transform(X.T)
        # transform again
        x = pd.DataFrame(x_new, index=X.columns, columns=["PC{}".format(i) for i in range(1, 1 + x_new.shape[1])])
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)

        # plot % explained variance per PC
        fig, axis = plt.subplots(1)
        axis.plot(
            range(1, len(pca.explained_variance_) + 1),  # all PCs
            (pca.explained_variance_ / pca.explained_variance_.sum()) * 100, 'o-')  # % of total variance
        axis.axvline(len(to_plot), linestyle='--')
        axis.set_xlabel("PC")
        axis.set_ylabel("% variance")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.explained_variance.svg".format(self.name)), bbox_inches='tight')

        # plot
        pcs = min(xx.shape[0] - 1, 10)
        fig, axis = plt.subplots(pcs, len(to_plot), figsize=(4 * len(to_plot), 4 * pcs))
        for pc in range(pcs):
            for i, attr in enumerate(to_plot):
                for j in range(len(xx)):
                    try:
                        label = getattr(samples[j], to_plot[i])
                    except AttributeError:
                        label = np.nan
                    axis[pc, i].scatter(xx.loc[j, pc], xx.loc[j, pc + 1], s=50, color=color_dataframe.loc[attr, j], label=label)
                axis[pc, i].set_title(to_plot[i])
                axis[pc, i].set_xlabel("PC {}".format(pc + 1))
                axis[pc, i].set_ylabel("PC {}".format(pc + 2))
                axis[pc, i].set_xticklabels(axis[pc, i].get_xticklabels(), visible=False)
                axis[pc, i].set_yticklabels(axis[pc, i].get_yticklabels(), visible=False)

                # Unique legend labels
                handles, labels = axis[pc, i].get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                if any([type(c) in [str, unicode] for c in by_label.keys()]) and len(by_label) <= 20:
                    if not any([re.match("^\d", c) for c in by_label.keys()]):
                        axis[pc, i].legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.svg".format(self.name)), bbox_inches="tight")

        # plot Figure 1c
        # patients sorted by leftmost point between timepoints
        x = pd.DataFrame(x_new, index=X.columns, columns=["PC{}".format(i) for i in range(1, 1 + x_new.shape[1])])
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)
        xx = xx[xx.index.get_level_values("patient_id") != "CLL16"]

        order = xx.groupby(level=['patient_id']).apply(lambda x: min(x.loc[x.index.get_level_values("timepoint_name") == "after_Ibrutinib", "PC3"].squeeze(), x.loc[x.index.get_level_values("timepoint_name") == "before_Ibrutinib", "PC3"].squeeze()))
        order = order[[type(i) is np.float64 for i in order]].sort_values()
        order.name = "patient_change"
        order = order.to_frame()
        order['patient_change_order'] = order['patient_change'].rank()
        xx = pd.merge(xx.reset_index(), order.reset_index()).set_index(xx.index.names + ["patient_change_order"]).sort_index(axis=0, level=["patient_change_order", "timepoint_name"])

        g = sns.PairGrid(xx.reset_index(), x_vars=xx.columns[:4], y_vars=["sample_name"], hue="timepoint_name", size=6, aspect=.5)
        g.map(sns.stripplot, size=10, orient="h")
        g.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.PC1-4.stripplot.min_sorted.svg".format(self.name)), bbox_inches="tight")

        g = sns.PairGrid(xx.reset_index(), x_vars=xx.columns[:4], y_vars=["patient_id"], hue="timepoint_name", size=6, aspect=.5)
        g.map(sns.stripplot, size=10, orient="h")
        g.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.PC1-4.stripplot.patient_centric.min_sorted.svg".format(self.name)), bbox_inches="tight")

        # patients sorted by ammount changed between timepoints
        x = pd.DataFrame(x_new, index=X.columns, columns=["PC{}".format(i) for i in range(1, 1 + x_new.shape[1])])
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)
        xx = xx[xx.index.get_level_values("patient_id") != "CLL16"]

        order = xx.groupby(level=['patient_id']).apply(lambda x: abs(x.loc[x.index.get_level_values("timepoint_name") == "after_Ibrutinib", "PC3"].squeeze() - x.loc[x.index.get_level_values("timepoint_name") == "before_Ibrutinib", "PC3"].squeeze()))
        order = order[[type(i) is np.float64 for i in order]].sort_values()
        order.name = "patient_change"
        order = order.to_frame()
        order['patient_change_order'] = order['patient_change'].rank()
        xx = pd.merge(xx.reset_index(), order.reset_index()).set_index(xx.index.names + ["patient_change_order"]).sort_index(axis=0, level=["patient_change_order", "timepoint_name"])

        g = sns.PairGrid(xx.reset_index(), x_vars=xx.columns[:4], y_vars=["sample_name"], hue="timepoint_name", size=6, aspect=.5)
        g.map(sns.stripplot, size=10, orient="h")
        g.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.PC1-4.stripplot.diff_sorted.svg".format(self.name)), bbox_inches="tight")

        g = sns.PairGrid(xx.reset_index(), x_vars=xx.columns[:4], y_vars=["patient_id"], hue="timepoint_name", size=6, aspect=.5)
        g.map(sns.stripplot, size=10, orient="h")
        g.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.PC1-4.stripplot.patient_centric.diff_sorted.svg".format(self.name)), bbox_inches="tight")

        # PCA
        fig, axis = plt.subplots(1, len(to_plot), figsize=(4 * len(to_plot), 4 * 1))
        for i, attr in enumerate(to_plot):
            for j in range(len(xx)):
                try:
                    label = getattr(samples[j], to_plot[i])
                except AttributeError:
                    label = np.nan
                axis[i].scatter(xx.ix[j][0], xx.ix[j][2], s=50, color=color_dataframe.ix[attr][j], label=label)
            axis[i].set_title(to_plot[i])
            axis[i].set_xlabel("PC {}".format(1))
            axis[i].set_ylabel("PC {}".format(3))
            axis[i].set_xticklabels(axis[i].get_xticklabels(), visible=False)
            axis[i].set_yticklabels(axis[i].get_yticklabels(), visible=False)

            # Unique legend labels
            handles, labels = axis[i].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            if any([type(c) in [str, unicode] for c in by_label.keys()]) and len(by_label) <= 20:
                if not any([re.match("^\d", c) for c in by_label.keys()]):
                    axis[i].legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.PC1vs3.svg".format(self.name)), bbox_inches="tight")

        # plot PC 1 vs 3
        fig, axis = plt.subplots(1, len(to_plot), figsize=(4 * len(to_plot), 4 * 1))
        for i, attr in enumerate(to_plot):
            for j in range(len(xx)):
                try:
                    label = getattr(samples[j], to_plot[i])
                except AttributeError:
                    label = np.nan
                axis[i].scatter(xx.ix[j][0], xx.ix[j][2], s=50, color=color_dataframe.ix[attr][j], label=label)
            axis[i].set_title(to_plot[i])
            axis[i].set_xlabel("PC {}".format(1))
            axis[i].set_ylabel("PC {}".format(3))
            axis[i].set_xticklabels(axis[i].get_xticklabels(), visible=False)
            axis[i].set_yticklabels(axis[i].get_yticklabels(), visible=False)

            # Unique legend labels
            handles, labels = axis[i].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            if any([type(c) in [str, unicode] for c in by_label.keys()]) and len(by_label) <= 20:
                if not any([re.match("^\d", c) for c in by_label.keys()]):
                    axis[i].legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.PC1vs3.svg".format(self.name)), bbox_inches="tight")

        #

        # # Test association of PCs with attributes
        associations = list()
        for pc in range(pcs):
            for attr in attributes[1:]:
                print("PC {}; Attribute {}.".format(pc + 1, attr))
                sel_samples = [s for s in samples if hasattr(s, attr)]
                sel_samples = [s for s in sel_samples if not pd.isnull(getattr(s, attr))]

                # Get all values of samples for this attr
                groups = set([getattr(s, attr) for s in sel_samples])

                # Determine if attr is categorical or continuous
                if all([type(i) in [str, bool] for i in groups]) or len(groups) == 2:
                    variable_type = "categorical"
                elif all([type(i) in [int, float, np.int64, np.float64] for i in groups]):
                    variable_type = "numerical"
                else:
                    print("attr %s cannot be tested." % attr)
                    associations.append([pc + 1, attr, variable_type, np.nan, np.nan, np.nan])
                    continue

                if variable_type == "categorical":
                    # It categorical, test pairwise combinations of attributes
                    for group1, group2 in itertools.combinations(groups, 2):
                        g1_indexes = [i for i, s in enumerate(sel_samples) if getattr(s, attr) == group1]
                        g2_indexes = [i for i, s in enumerate(sel_samples) if getattr(s, attr) == group2]

                        g1_values = xx.loc[g1_indexes, pc]
                        g2_values = xx.loc[g2_indexes, pc]

                        # Test ANOVA (or Kruskal-Wallis H-test)
                        p = kruskal(g1_values, g2_values)[1]

                        # Append
                        associations.append([pc + 1, attr, variable_type, group1, group2, p])

                elif variable_type == "numerical":
                    # It numerical, calculate pearson correlation
                    indexes = [i for i, s in enumerate(samples) if s in sel_samples]
                    pc_values = xx.loc[indexes, pc]
                    trait_values = [getattr(s, attr) for s in sel_samples]
                    p = pearsonr(pc_values, trait_values)[1]

                    associations.append([pc + 1, attr, variable_type, np.nan, np.nan, p])

        associations = pd.DataFrame(associations, columns=["pc", "attribute", "variable_type", "group_1", "group_2", "p_value"])

        # write
        associations.to_csv(os.path.join(self.results_dir, "{}.all_sites.pca.variable_principle_components_association.csv".format(self.name)), index=False)

        # Plot
        # associations[associations['p_value'] < 0.05].drop(['group_1', 'group_2'], axis=1).drop_duplicates()
        # associations.drop(['group_1', 'group_2'], axis=1).drop_duplicates().pivot(index="pc", columns="attribute", values="p_value")
        pivot = associations.groupby(["pc", "attribute"]).min()['p_value'].reset_index().pivot(index="pc", columns="attribute", values="p_value").dropna(axis=1)

        # heatmap of -log p-values
        g = sns.clustermap(-np.log10(pivot), row_cluster=False, annot=True, cbar_kws={"label": "-log10(p_value) of association"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        g.fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.variable_principle_components_association.svg".format(self.name)), bbox_inches="tight")

        # heatmap of masked significant
        g = sns.clustermap((pivot < 0.05).astype(int), row_cluster=False, cbar_kws={"label": "significant association"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        g.fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.variable_principle_components_association.masked.svg".format(self.name)), bbox_inches="tight")

    def unsupervised_enrichment(self, samples):
        """
        """
        from sklearn.decomposition import PCA
        import itertools
        from statsmodels.sandbox.stats.multicomp import multipletests

        def jackstraw(data, pcs, n_vars, n_iter=100):
            """
            """
            import rpy2.robjects as robj
            from rpy2.robjects.vectors import IntVector
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()

            run = robj.r("""
                run = function(data, pcs, n_vars, B) {
                    library(jackstraw)
                    out <- jackstraw.PCA(data, r1=pcs, r=n_vars, B=B)
                    return(out$p.value)
                }
            """)
            # save to disk just in case
            data.to_csv("_tmp_matrix.jackstraw.csv", index=True)

            if type(pcs) is not int:
                pcs = IntVector(pcs)
            return run(data.values, pcs, n_vars, n_iter)

        def lola(bed_files, universe_file, output_folder):
            """
            Performs location overlap analysis (LOLA) on bedfiles with regions sets.
            """
            import rpy2.robjects as robj

            run = robj.r("""
                function(bedFiles, universeFile, outputFolder) {
                    library("LOLA")

                    # universeFile = "~/cll-ibrutinib/results/cll-ibrutinib_AKH_peak_set.bed"
                    # bedFiles = "~/cll-ibrutinib/results/cll-ibrutinib_AKH.ibrutinib_treatment/cll-ibrutinib_AKH.ibrutinib_treatment.timepoint_name.diff_regions.comparison_after_Ibrutinib-before_Ibrutinib.up/cll-ibrutinib_AKH.ibrutinib_treatment.timepoint_name.diff_regions.comparison_after_Ibrutinib-before_Ibrutinib.up_regions.bed"
                    # outputFolder = "~/cll-ibrutinib/results/cll-ibrutinib_AKH.ibrutinib_treatment/cll-ibrutinib_AKH.ibrutinib_treatment.timepoint_name.diff_regions.comparison_after_Ibrutinib-before_Ibrutinib.up/"

                    userUniverse  <- LOLA::readBed(universeFile)

                    dbPath1 = "/data/groups/lab_bock/shared/resources/regions/LOLACore/hg19/"
                    dbPath2 = "/data/groups/lab_bock/shared/resources/regions/customRegionDB/hg19/"
                    regionDB = loadRegionDB(c(dbPath1, dbPath2))

                    if (typeof(bedFiles) == "character") {
                        userSet <- LOLA::readBed(bedFiles)
                        lolaResults = runLOLA(list(userSet), userUniverse, regionDB, cores=12)
                        writeCombinedEnrichment(lolaResults, outFolder=outputFolder, includeSplits=FALSE)
                    } else if (typeof(bedFiles) == "double") {
                        for (bedFile in bedFiles) {
                            userSet <- LOLA::readBed(bedFile)
                            lolaResults = runLOLA(list(userSet), userUniverse, regionDB, cores=12)
                            writeCombinedEnrichment(lolaResults, outFolder=outputFolder, includeSplits=FALSE)
                        }
                    }
                }
            """)

            # convert the pandas dataframe to an R dataframe
            run(bed_files, universe_file, output_folder)

        def vertical_line(x, **kwargs):
            plt.axvline(x.mean(), **kwargs)

        # Get accessibility matrix excluding sex chroms
        X = self.accessibility[[s.name for s in samples]]
        X = X.ix[X.index[~X.index.str.contains("chrX|chrY")]]

        # Now perform association analysis (jackstraw)
        n_vars = 20
        max_sig = 1000
        alpha = 0.01
        pcs = range(1, n_vars + 1)
        pcs += list(itertools.combinations(pcs, 2))

        p_values = pd.DataFrame(index=X.index)
        for pc in pcs:
            print(pc)
            out = jackstraw(X, pc, n_vars, 10).flatten()
            if type(pc) is int:
                p_values[pc] = out
            else:
                p_values["+".join([str(x) for x in pc])] = out
        q_values = p_values.apply(lambda x: multipletests(x, method="fdr_bh")[1])
        p_values.to_csv(os.path.join("results", "{}.PCA.PC_pvalues.csv".format(self.name)), index=True)

        # Get enrichments of each PC-regions
        lola_enrichments = pd.DataFrame()
        enrichr_enrichments = pd.DataFrame()
        for pc in p_values.columns:
            p = p_values[pc].sort_values()
            sig = p[p < alpha].index

            # Cap to a maximum number of regions
            if len(sig) > max_sig:
                sig = p.head(max_sig).index

            # Run LOLA
            # save to bed
            output_folder = os.path.join("results", "{}.PCA.PC{}_regions".format(self.name, pc))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            universe_file = os.path.join(output_folder, "universe_sites.bed")
            self.sites.saveas(universe_file)
            bed_file = os.path.join(output_folder, "PCA.PC{}_regions.bed".format(pc))
            self.coverage[['chrom', 'start', 'end']].ix[sig].to_csv(bed_file, sep="\t", header=False, index=False)
            # lola(bed_file, universe_file, output_folder)

            # read lola
            lol = pd.read_csv(os.path.join(output_folder, "allEnrichments.txt"), sep="\t")
            lol["PC"] = pc
            lola_enrichments = lola_enrichments.append(lol)

            # Get genes, run enrichr
            sig_genes = self.coverage_annotated['gene_name'].ix[sig]
            sig_genes = [x for g in sig_genes.dropna().astype(str).tolist() for x in g.split(',')]

            enr = enrichr(pd.DataFrame(sig_genes, columns=["gene_name"]))
            enr["PC"] = pc
            enrichr_enrichments = enrichr_enrichments.append(enr)
        enrichr_enrichments.to_csv(os.path.join("results", "{}.PCA.enrichr.csv".format(self.name)), index=False, encoding="utf-8")
        enrichr_enrichments = pd.read_csv(os.path.join("results", "{}.PCA.enrichr.csv".format(self.name)))
        lola_enrichments.to_csv(os.path.join("results", "{}.PCA.lola.csv".format(self.name)), index=False, encoding="utf-8")

        # Plots

        # p-value distributions
        g = sns.FacetGrid(data=pd.melt(-np.log10(p_values), var_name="PC", value_name="-log10(p-value)"), col="PC", col_wrap=5)
        g.map(sns.distplot, "-log10(p-value)", kde=False)
        g.map(plt.axvline, x=-np.log10(alpha), linestyle="--")
        g.add_legend()
        g.fig.savefig(os.path.join("results", "{}.PCA.PC_pvalues.distplot.svg".format(self.name)), bbox_inches="tight")

        # Volcano plots (loading vs p-value)
        # get PCA loadings
        pca = PCA()
        pca.fit(X.T)
        loadings = pd.DataFrame(pca.components_.T, index=X.index, columns=range(1, X.shape[1] + 1))
        loadings.to_csv(os.path.join("results", "{}.PCA.loadings.csv".format(self.name)), index=True, encoding="utf-8")

        melted_loadings = pd.melt(loadings.reset_index(), var_name="PC", value_name="loading", id_vars=["index"]).set_index(["index", "PC"])
        melted_p_values = pd.melt((-np.log10(p_values)).reset_index(), var_name="PC", value_name="-log10(p-value)", id_vars=["index"]).set_index(["index", "PC"])
        melted = melted_loadings.join(melted_p_values)

        g = sns.FacetGrid(data=melted.dropna().reset_index(), col="PC", col_wrap=5, sharey=False, sharex=False)
        g.map(plt.scatter, "loading", "-log10(p-value)", s=2, alpha=0.5)
        g.map(plt.axhline, y=-np.log10(alpha), linestyle="--")
        g.add_legend()
        g.fig.savefig(os.path.join("results", "{}.PCA.PC_pvalues_vs_loading.scatter.png".format(self.name)), bbox_inches="tight", dpi=300)

        # Plot enrichments
        # LOLA
        # take top n per PC
        import string
        lola_enrichments["set_id"] = lola_enrichments[
            ["collection", "description", "cellType", "tissue", "antibody", "treatment"]].astype(str).apply(string.join, axis=1)

        top = lola_enrichments.set_index('set_id').groupby("PC")['pValueLog'].nlargest(25)
        top_ids = top.index.get_level_values('set_id').unique()

        pivot = pd.pivot_table(
            lola_enrichments,
            index="set_id", columns="PC", values="pValueLog").fillna(0)
        pivot.index = pivot.index.str.replace(" nan", "").str.replace("blueprint blueprint", "blueprint").str.replace("None", "")
        top_ids = top_ids.str.replace(" nan", "").str.replace("blueprint blueprint", "blueprint").str.replace("None", "")

        g = sns.clustermap(
            pivot.ix[top_ids],
            cbar_kws={"label": "p-value z-score"},
            col_cluster=True, z_score=0)
        for tick in g.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in g.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        g.fig.savefig(os.path.join("results", "{}.PCA.PC_pvalues.lola_enrichments.svg".format(self.name)), bbox_inches="tight", dpi=300)

        # Enrichr
        for gene_set_library in enrichr_enrichments["gene_set_library"].drop_duplicates():
            enr = enrichr_enrichments[enrichr_enrichments["gene_set_library"] == gene_set_library]

            top = enr.set_index('description').groupby("PC")['p_value'].nsmallest(20)
            top_ids = top.index.get_level_values('description').unique()

            pivot = pd.pivot_table(enr, index="description", columns="PC", values="p_value").fillna(1)
            pivot.index = pivot.index.str.extract("(.*)[,\_\(].*").str.replace("_Homo sapiens", "")
            top_ids = top_ids.str.extract("(.*)[,\_\(].*").str.replace("_Homo sapiens", "")

            g = sns.clustermap(
                -np.log10(pivot.ix[top_ids]), cmap='BuGn',
                cbar_kws={"label": "-log10(p-value)"}, col_cluster=True, figsize=(6, 15))
            for tick in g.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in g.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            g.fig.savefig(os.path.join("results", "{}.PCA.PC_pvalues.enrichr_enrichments.{}.svg".format(self.name, gene_set_library)), bbox_inches="tight", dpi=300)

            g = sns.clustermap(
                -np.log10(pivot.ix[top_ids]),
                cbar_kws={"label": "-log10(p-value) z-score"}, col_cluster=True, z_score=0, figsize=(6, 15))
            for tick in g.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in g.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            g.fig.savefig(os.path.join("results", "{}.PCA.PC_pvalues.enrichr_enrichments.{}.z_score.svg".format(self.name, gene_set_library)), bbox_inches="tight", dpi=300)

    def differential_region_analysis(
            self, samples, trait="ibrutinib_treatment",
            variables=["atac_seq_batch", "patient_gender", "ighv_mutation_status", "timepoint_name", "ibrutinib_treatment"],
            output_suffix="ibrutinib_treatment"):
        """
        Discover differential regions across samples that are associated with a certain trait.
        """
        import itertools

        sel_samples = [s for s in samples if not pd.isnull(getattr(s, trait))]

        # Get matrix of counts
        counts_matrix = self.coverage[[s.name for s in sel_samples]]

        # Get experiment matrix
        experiment_matrix = pd.DataFrame([sample.as_series() for sample in sel_samples], index=[sample.name for sample in sel_samples])
        # keep only variables
        experiment_matrix = experiment_matrix[["sample_name"] + variables].fillna("Unknown")

        # Make output dir
        output_dir = os.path.join(self.results_dir, output_suffix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Run DESeq2 analysis
        deseq_table = DESeq_analysis(
            counts_matrix, experiment_matrix, trait, covariates=[x for x in variables if x != trait], output_prefix=os.path.join(output_dir, output_suffix), alpha=0.05)
        deseq_table.columns = deseq_table.columns.str.replace(".", " ")

        # to just read in
        # deseq_table = pd.read_csv(os.path.join(output_dir, output_suffix) + ".%s.csv" % trait, index_col=0)
        # self.accessibility = pd.read_csv(os.path.join(self.results_dir, "breg_peaks.coverage_qnorm.log2.annotated.csv"), index_col=0)

        df = self.accessibility.join(deseq_table)
        df.to_csv(os.path.join(output_dir, output_suffix) + ".%s.annotated.csv" % trait)

        # Extract significant based on p-value and fold-change
        diff = df[(df["padj"] < 0.01) & (abs(df["log2FoldChange"]) > 0.5)]
        # diff = df[(df["padj"] < 0.05) & (abs(df["log2FoldChange"]) > 1)]

        if diff.shape[0] < 1:
            print("No significantly different regions found.")
            return

        groups = list(set([getattr(s, trait) for s in sel_samples]))
        comparisons = pd.Series(df['comparison'].unique())

        # Statistics of differential regions
        import string
        total_sites = float(len(self.sites))

        total_diff = diff.groupby(["comparison"])['stat'].count().sort_values(ascending=False)
        fig, axis = plt.subplots(1)
        sns.barplot(total_diff.values, total_diff.index, orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.total.svg" % (output_suffix, trait)), bbox_inches="tight")
        # percentage of total
        fig, axis = plt.subplots(1)
        sns.barplot(
            (total_diff.values / total_sites) * 100,
            total_diff.index,
            orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.total_percentage.svg" % (output_suffix, trait)), bbox_inches="tight")

        # direction-dependent
        diff["direction"] = diff["log2FoldChange"].apply(lambda x: "up" if x >= 0 else "down")

        split_diff = diff.groupby(["comparison", "direction"])['stat'].count().sort_values(ascending=False)
        fig, axis = plt.subplots(1, figsize=(12, 8))
        sns.barplot(
            split_diff.values,
            split_diff.reset_index()[['comparison', 'direction']].apply(string.join, axis=1),
            orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.split.svg" % (output_suffix, trait)), bbox_inches="tight")
        # percentage of total
        fig, axis = plt.subplots(1, figsize=(12, 8))
        sns.barplot(
            (split_diff.values / total_sites) * 100,
            split_diff.reset_index()[['comparison', 'direction']].apply(string.join, axis=1),
            orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.split_percentage.svg" % (output_suffix, trait)), bbox_inches="tight")

        # Pairwise scatter plots
        n_rows = n_cols = int(np.ceil(len(comparisons) / 2.))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
        if n_rows > 1 or n_cols > 1:
            axes = iter(axes.flatten())
        else:
            axes = iter([axes])
        for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()
            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            df2 = df[df["comparison"] == comparison]
            if df2.shape[0] == 0:
                continue
            axis = axes.next()

            # Hexbin plot
            axis.hexbin(np.log2(1 + df2[cond1]), np.log2(1 + df2[cond2]), bins="log", alpha=.85)
            axis.set_xlabel(cond1)

            diff2 = diff[diff["comparison"] == comparison]
            if diff2.shape[0] > 0:
                # Scatter plot
                axis.scatter(np.log2(1 + diff2[cond1]), np.log2(1 + diff2[cond2]), alpha=0.1, color="red", s=2)
            m = max(np.log2(1 + df2[cond1]).max(), np.log2(1 + df2[cond2]).max())
            axis.plot([0, m], [0, m], color="black", alpha=0.8, linestyle="--")
            axis.set_ylabel(cond1)
            axis.set_ylabel(cond2)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.scatter_plots.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # Volcano plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
        if n_rows > 1 or n_cols > 1:
            axes = iter(axes.flatten())
        else:
            axes = iter([axes])
        for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()
            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            df2 = df[df["comparison"] == comparison]
            if df2.shape[0] == 0:
                continue
            axis = axes.next()

            # hexbin
            axis.hexbin(df2["log2FoldChange"], -np.log10(df2['pvalue']), alpha=0.85, color="black", edgecolors="white", linewidths=0, bins='log', mincnt=1)

            diff2 = diff[diff["comparison"] == comparison]
            if diff2.shape[0] > 0:
                # significant scatter
                axis.scatter(diff2["log2FoldChange"], -np.log10(diff2['pvalue']), alpha=0.2, color="red", s=2)
            axis.axvline(0, linestyle="--", color="black", alpha=0.8)
            axis.set_title(comparison)
            axis.set_xlabel("log2(fold change)")
            axis.set_ylabel("-log10(p-value)")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.volcano_plots.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # MA plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
        if n_rows > 1 or n_cols > 1:
            axes = iter(axes.flatten())
        else:
            axes = iter([axes])
        for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()
            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            df2 = df[df["comparison"] == comparison]
            if df2.shape[0] == 0:
                continue
            axis = axes.next()

            # hexbin
            axis.hexbin(np.log2(df2["baseMean"]), df2["log2FoldChange"], alpha=0.85, color="black", edgecolors="white", linewidths=0, bins='log', mincnt=1)

            diff2 = diff[diff["comparison"] == comparison]
            if diff2.shape[0] > 0:
                # significant scatter
                axis.scatter(np.log2(diff2["baseMean"]), diff2["log2FoldChange"], alpha=0.2, color="red", s=2)
            axis.axhline(0, linestyle="--", color="black", alpha=0.8)
            axis.set_title(comparison)
            axis.set_xlabel("Intensity")
            axis.set_ylabel("log2(fold change)")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.ma_plots.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # save unique differential regions
        diff2 = diff[groups].ix[diff.index.unique()].drop_duplicates()
        diff2.to_csv(os.path.join(output_dir, "%s.%s.differential_regions.csv" % (output_suffix, trait)))

        # Exploration of differential regions
        # Characterize regions
        prefix = "%s.%s.diff_regions" % (output_suffix, trait)
        # region's structure
        # characterize_regions_structure(df=diff2, prefix=prefix, output_dir=output_dir)
        # region's function
        # characterize_regions_function(df=diff2, prefix=prefix, output_dir=output_dir)

        # Heatmaps
        # Comparison level
        g = sns.clustermap(np.log2(1 + diff2[groups]).corr(), xticklabels=False)
        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.groups.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(np.log2(1 + diff2[groups]).T, xticklabels=False)
        for item in g.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.groups.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(np.log2(1 + diff2[groups]).T, xticklabels=False, z_score=1)
        for item in g.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.groups.clustermap.z0.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # Sample level
        attributes = [
            "patient_id", "timepoint_name", "patient_gender", "patient_age_at_collection",
            "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
            "del11q", "del13q", "del17p", "tri12", "p53",
            "time_since_treatment", "treatment_response"]
        color_dataframe = pd.DataFrame(self.get_level_colors(levels=attributes), index=attributes, columns=[s.name for s in self.samples])

        # exclude samples if needed
        color_dataframe = color_dataframe[[s.name for s in samples]]
        sample_display_names = color_dataframe.columns.str.replace("_ATAC-seq", "").str.replace("_hg19", "")

        g = sns.clustermap(
            self.accessibility.ix[diff2.index][[s.name for s in sel_samples]].corr(),
            xticklabels=False, yticklabels=sample_display_names, annot=True, vmin=0, vmax=1,
            cmap="Spectral_r", figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.samples.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(
            self.accessibility.ix[diff2.index][[s.name for s in sel_samples]].T,
            xticklabels=False, yticklabels=sample_display_names, annot=True,
            figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.samples.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(
            self.accessibility.ix[diff2.index][[s.name for s in sel_samples]].T, z_score=1,
            xticklabels=False, yticklabels=sample_display_names, annot=True,
            figsize=(15, 15), cbar_kws={"label": "Z-score of Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.samples.clustermap.z0.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # Examine each region cluster
        region_enr = pd.DataFrame()
        lola_enr = pd.DataFrame()
        motif_enr = pd.DataFrame()
        pathway_enr = pd.DataFrame()
        for cond1, cond2 in sorted(list(itertools.combinations(groups, 2)), key=lambda x: len(x[0])):
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1) & comparisons.str.contains(cond2)].squeeze()

            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            # Separate in up/down-regulated regions
            for f, direction in [(np.less, "down"), (np.greater, "up")]:
                comparison_df = self.coverage_annotated.ix[diff[
                    (diff["comparison"] == comparison) &
                    (f(diff["log2FoldChange"], 0))
                ].index].join(diff[['log2FoldChange', 'padj']])
                if comparison_df.shape[0] < 1:
                    continue
                # Characterize regions
                prefix = "%s.%s.diff_regions.comparison_%s.%s" % (output_suffix, trait, comparison, direction)

                comparison_dir = os.path.join(output_dir, prefix)

                print("Doing regions of comparison %s, with prefix %s" % (comparison, prefix))
                characterize_regions_function(df=comparison_df, prefix=prefix, output_dir=comparison_dir)

                # region's structure
                if not os.path.exists(os.path.join(comparison_dir, prefix + "_regions.region_enrichment.csv")):
                    print(prefix)
                    characterize_regions_structure(df=comparison_df, prefix=prefix, output_dir=comparison_dir)
                # region's function
                if not os.path.exists(os.path.join(comparison_dir, prefix + "_regions.enrichr.csv")):
                    print(prefix)
                    characterize_regions_function(df=comparison_df, prefix=prefix, output_dir=comparison_dir)

                # Read/parse enrichment outputs and add to DFs
                enr = pd.read_csv(os.path.join(comparison_dir, prefix + "_regions.region_enrichment.csv"))
                enr.columns = ["region"] + enr.columns[1:].tolist()
                enr["comparison"] = prefix
                region_enr = region_enr.append(enr, ignore_index=True)

                enr = pd.read_csv(os.path.join(comparison_dir, "allEnrichments.txt"), sep="\t")
                enr["comparison"] = prefix
                lola_enr = lola_enr.append(enr, ignore_index=True)

                enr = parse_ame(comparison_dir).reset_index()
                enr["comparison"] = prefix
                motif_enr = motif_enr.append(enr, ignore_index=True)

                enr = pd.read_csv(os.path.join(comparison_dir, prefix + "_regions.enrichr.csv"))
                enr["comparison"] = prefix
                pathway_enr = pathway_enr.append(enr, ignore_index=True)

        # write combined enrichments
        region_enr.to_csv(
            os.path.join(output_dir, "%s.%s.diff_regions.regions.csv" % (output_suffix, trait)), index=False)
        lola_enr.to_csv(
            os.path.join(output_dir, "%s.%s.diff_regions.lola.csv" % (output_suffix, trait)), index=False)
        motif_enr.columns = ["motif", "p_value", "comparison"]
        motif_enr.to_csv(
            os.path.join(output_dir, "%s.%s.diff_regions.motifs.csv" % (output_suffix, trait)), index=False)
        pathway_enr.to_csv(
            os.path.join(output_dir, "%s.%s.diff_regions.enrichr.csv" % (output_suffix, trait)), index=False)

    def investigate_differential_regions(self, trait="condition", output_suffix="deseq", n=50):
        import string
        from scipy.cluster.hierarchy import fcluster

        output_dir = os.path.join(self.results_dir, output_suffix)

        # REGION TYPES
        # read in
        regions = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_regions.regions.csv" % (output_suffix, trait)))
        # pretty names
        regions["comparison"] = regions["comparison"].str.extract("%s.%s.diff_regions.comparison_(.*)" % (output_suffix, trait), expand=True)

        # pivot table
        regions_pivot = pd.pivot_table(regions, values="value", columns="region", index="comparison")

        # fillna
        regions_pivot = regions_pivot.fillna(0)

        # plot correlation
        fig = sns.clustermap(regions_pivot)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "region_type_enrichment.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "region_type_enrichment.png"), bbox_inches="tight", dpi=300)

        #

        # LOLA
        # read in
        lola = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_regions.lola.csv" % (output_suffix, trait)))
        # pretty names
        lola["comparison"] = lola["comparison"].str.extract("%s.%s.diff_regions.comparison_(.*)" % (output_suffix, trait), expand=True)

        # unique ids for lola sets
        cols = ['description', u'cellType', u'tissue', u'antibody', u'treatment', u'dataSource', u'filename']
        lola['label'] = lola[cols].astype(str).apply(string.join, axis=1)

        # pivot table
        lola_pivot = pd.pivot_table(lola, values="pValueLog", columns="label", index="comparison")
        lola_pivot.columns = lola_pivot.columns.str.decode("utf-8")

        # plot correlation
        fig = sns.clustermap(lola_pivot.T.corr())
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "lola.correlation.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "lola.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(fig.dendrogram_col.linkage, 3, criterion="maxclust")

        # Get top n terms which are more in each cluster compared with all others
        top_terms = list()
        cluster_means = pd.DataFrame()
        for cluster in set(cluster_assignment):
            cluster_comparisons = lola_pivot.index[cluster_assignment == cluster].tolist()
            other_comparisons = lola_pivot.index[cluster_assignment != cluster].tolist()

            terms = (lola_pivot.ix[cluster_comparisons].mean() - lola_pivot.ix[other_comparisons].mean()).sort_values()

            top_terms += terms.dropna().head(n).index.tolist()

            # additionallly, get mean of cluster
            cluster_means[cluster] = lola_pivot.ix[cluster_comparisons].mean()

        # plot clustered heatmap
        fig = sns.clustermap(lola_pivot[list(set(top_terms))].replace({np.inf: 50}), z_score=0, figsize=(20, 12))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "lola.cluster_specific.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "lola.cluster_specific.png"), bbox_inches="tight", dpi=300)

        #

        # MOTIFS
        # read in
        motifs = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_regions.motifs.csv" % (output_suffix, trait)))
        # pretty names
        motifs["comparison"] = motifs["comparison"].str.extract("%s.%s.diff_regions.comparison_(.*)" % (output_suffix, trait), expand=True)

        # pivot table
        motifs_pivot = pd.pivot_table(motifs, values="p_value", columns="motif", index="comparison")

        # transform p-values
        motifs_pivot = -np.log10(motifs_pivot.fillna(1))
        motifs_pivot = motifs_pivot.replace({np.inf: 300})

        # plot correlation
        fig = sns.clustermap(motifs_pivot.T.corr())
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "motifs.correlation.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "motifs.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(fig.dendrogram_col.linkage, 5, criterion="maxclust")

        # Get top n terms which are more in each cluster compared with all others
        top_terms = list()
        cluster_means = pd.DataFrame()
        for cluster in set(cluster_assignment):
            cluster_comparisons = motifs_pivot.index[cluster_assignment == cluster].tolist()
            other_comparisons = motifs_pivot.index[cluster_assignment != cluster].tolist()

            terms = (motifs_pivot.ix[cluster_comparisons].mean() - motifs_pivot.ix[other_comparisons].mean()).sort_values()

            top_terms += terms.dropna().head(n).index.tolist()

        # plot clustered heatmap
        fig = sns.clustermap(motifs_pivot[list(set(top_terms))], figsize=(20, 12))  # .apply(lambda x: (x - x.mean()) / x.std())
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.png"), bbox_inches="tight", dpi=300)

        df = motifs_pivot[list(set(top_terms))]  # .apply(lambda x: (x - x.mean()) / x.std())

        fig = sns.clustermap(df[df.mean(1) > -0.5], figsize=(20, 12))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.png"), bbox_inches="tight", dpi=300)

        #

        # ENRICHR
        # read in
        enrichr = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_regions.enrichr.csv" % (output_suffix, trait)))
        # pretty names
        enrichr["comparison"] = enrichr["comparison"].str.extract("%s.%s.diff_regions.comparison_(.*)" % (output_suffix, trait), expand=True)

        for gene_set_library in enrichr["gene_set_library"].unique():
            print(gene_set_library)
            if gene_set_library == "Epigenomics_Roadmap_HM_ChIP-seq":
                continue

            # pivot table
            enrichr_pivot = pd.pivot_table(
                enrichr[enrichr["gene_set_library"] == gene_set_library],
                values="adjusted_p_value", columns="description", index="comparison")
            enrichr_pivot.columns = enrichr_pivot.columns.str.decode("utf-8")

            # transform p-values
            enrichr_pivot = -np.log10(enrichr_pivot.fillna(1))
            enrichr_pivot = enrichr_pivot.replace({np.inf: 300})

            # plot correlation
            fig = sns.clustermap(enrichr_pivot.T.corr())
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "enrichr.%s.correlation.svg" % gene_set_library), bbox_inches="tight")
            fig.savefig(os.path.join(output_dir, "enrichr.%s.correlation.png" % gene_set_library), bbox_inches="tight", dpi=300)

            cluster_assignment = fcluster(fig.dendrogram_col.linkage, 4, criterion="maxclust")

            # Get top n terms which are more in each cluster compared with all others
            top_terms = list()
            cluster_means = pd.DataFrame()
            for cluster in set(cluster_assignment):
                cluster_comparisons = enrichr_pivot.index[cluster_assignment == cluster].tolist()
                other_comparisons = enrichr_pivot.index[cluster_assignment != cluster].tolist()

                terms = (enrichr_pivot.ix[cluster_comparisons].mean() - enrichr_pivot.ix[other_comparisons].mean()).sort_values()

                top_terms += terms.dropna().head(n).index.tolist()

            # plot clustered heatmap
            fig = sns.clustermap(enrichr_pivot[list(set(top_terms))], figsize=(20, 12))  # .apply(lambda x: (x - x.mean()) / x.std())
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.svg" % gene_set_library), bbox_inches="tight")
            fig.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.png" % gene_set_library), bbox_inches="tight", dpi=300)

    # using genes weighted by score

    def treatment_response_score(
            self, samples=None, trait="timepoint_name",
            output_suffix="ibrutinib_treatment", attributes=[
                "sample_name", "cell_type", "patient_id", "clinical_centre", "timepoint_name", "patient_gender", "patient_age_at_collection",
                "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
                "del11q", "del13q", "del17p", "tri12", "p53",
                "time_since_treatment", "treatment_response"]):
        """
        Discover differential regions for a condition with only one replicate.
        """
        def z_score(x):
            return (x - x.mean()) / x.std()

        if samples is None:
            samples = [s for s in self.samples if s.patient_id != "CLL16"]

        index = self.accessibility.columns[self.accessibility.columns.get_level_values("sample_name").isin([s.name for s in samples])]

        color_dataframe = pd.DataFrame(self.get_level_colors(index=index, levels=attributes), index=attributes, columns=[s.name for s in samples])
        sample_display_names = color_dataframe.columns.str.replace("_ATAC-seq", "").str.replace("_hg19", "")

        # exclude attributes if needed
        to_plot = attributes[:]
        to_exclude = ["sample_name"]
        for attr in to_exclude:
            try:
                to_plot.pop(to_plot.index(attr))
            except:
                continue
        color_dataframe = color_dataframe.ix[to_plot]

        # read in differential regions
        output_suffix = "{}.ibrutinib_treatment".format(self.name)
        output_dir = os.path.join(self.results_dir, output_suffix)
        diff = pd.read_csv(os.path.join(output_dir, "%s.%s.differential_regions.csv" % (output_suffix, trait)), index_col=0).index

        X = self.accessibility.ix[diff][[s.name for s in samples]].apply(z_score, axis=1)
        X.columns = X.columns.get_level_values("sample_name")

        # 1. Compute score based on intensities of up- or down-regulated regions
        # 1.1 get up and down
        cond1, cond2 = sorted(set([getattr(s, trait) for s in samples]))
        u1 = X[[s.name for s in samples if getattr(s, trait) == cond1]].mean(axis=1)
        u2 = X[[s.name for s in samples if getattr(s, trait) == cond2]].mean(axis=1)
        extremes = pd.DataFrame([u1, u2], index=[cond1, cond2]).T
        up = extremes[extremes['after_Ibrutinib'] > extremes['before_Ibrutinib']].index
        down = extremes[extremes['after_Ibrutinib'] < extremes['before_Ibrutinib']].index

        # 1.2 Make score
        # get sum/mean intensities in either
        # weighted by each side contribution to the signature
        # sum the value of each side
        scores = (
            -(X.ix[up].mean(axis=0) * (float(up.size) / X.shape[0])) +
            (X.ix[down].mean(axis=0) * (float(down.size) / X.shape[0]))
        )
        # reverse for post samples (give positive values for depletion in downregulated regions, negative for depletion in upregulated)
        scores.loc[scores.index.str.contains("post")] = -scores.loc[scores.index.str.contains("post")]

        # 2. Visualize
        cmap = plt.get_cmap("RdBu_r")
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        g = sns.clustermap(
            X.T,
            xticklabels=False, yticklabels=sample_display_names.tolist(), annot=True,
            figsize=(15, 15), cbar_kws={"label": "Z-score of accessibility"}, row_colors=cmap(norm(scores.ix[X.columns])))
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.intensity_score_per_condition.clustermap.z0.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # joint response per patient
        # mean of both timepoints
        p = scores.index.str.extract(".*(CLL\d+)_.*", expand=False).tolist()
        t = scores.index.str.extract(".*_(.*)$", expand=False).tolist()
        s = pd.DataFrame([scores.tolist(), p, t], index=["score", "patient_id", "timepoint"], columns=scores.index).T
        s['score'] = s['score'].astype(float)
        m_s = s.groupby("patient_id")['score'].mean()
        m_s.name = "combined_score"
        scores = pd.merge(s.reset_index(), m_s.reset_index())
        scores = scores.sort_values(['combined_score', 'timepoint'])
        scores.to_csv(os.path.join(output_dir, "%s.%s.diff_regions.intensity_scores.csv" % (output_suffix, trait)))

        # complete color dataframe with score color
        cd = color_dataframe.T
        cd.index.name = "sample_name"
        cd = cd.join(scores.set_index("sample_name")['score'])
        cd['score'] = cd['score'].apply(norm).apply(cmap)

        cmap = plt.get_cmap("RdBu_r")
        norm = matplotlib.colors.Normalize(vmin=-0.2, vmax=1)

        g = sns.clustermap(
            X[scores.sort_values('score')['sample_name']].T, row_cluster=False,
            xticklabels=False, annot=True,
            figsize=(15, 15), cbar_kws={"label": "Z-score of accessibility"}, row_colors=cd.ix[scores.sort_values('score')['sample_name']].T.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.intensity_score_per_condition.clustermap.sorted.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        for tp in ['pre', 'post']:
            g = sns.clustermap(
                X[scores[scores['sample_name'].str.contains(tp)].sort_values('score')['sample_name']].T, row_cluster=False,
                xticklabels=False, annot=True,
                figsize=(15, 7.5), cbar_kws={"label": "Z-score of accessibility"}, row_colors=cd.ix[scores[scores['sample_name'].str.contains(tp)].sort_values('score')['sample_name']].T.values.tolist())
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
            g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
            g.fig.savefig(os.path.join(output_dir, "{}.{}.diff_regions.intensity_score_per_condition.clustermap.sorted.{}_samples_only.svg".format(output_suffix, trait, tp)), bbox_inches="tight", dpi=300)

        # complete color dataframe with combined score color
        cd = color_dataframe.T
        cd.index.name = "sample_name"
        cd = cd.join(scores.set_index("sample_name")['combined_score'])

        cd['combined_score'] = cd['combined_score'].apply(norm).apply(cmap)

        cmap = plt.get_cmap("RdBu_r")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        g = sns.clustermap(
            X[scores['sample_name']].T, row_cluster=False,
            xticklabels=False, annot=True,
            figsize=(15, 15), cbar_kws={"label": "Z-score of accessibility"}, row_colors=cd.ix[scores['sample_name']].T.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "{}.{}.diff_regions.intensity_score_combined.clustermap.sorted.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        cmap = plt.get_cmap("RdBu_r")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        g = sns.clustermap(
            X[scores['sample_name']].T, row_cluster=True,
            xticklabels=False, annot=True,
            figsize=(15, 15), cbar_kws={"label": "Z-score of accessibility"}, row_colors=cd.ix[scores['sample_name']].T.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "{}.{}.diff_regions.intensity_score_combined.clustermap.cluster.svg".format(output_suffix, trait)), bbox_inches="tight", dpi=300)

        # 3. Correlate score with all variables

        # # Test association of PCs with attributes
        scores = scores.set_index('sample_name').ix[[s.name for s in samples]]

        import itertools
        from scipy.stats import kruskal
        from scipy.stats import pearsonr
        associations = list()

        for measurement in ["score", "combined_score"]:
            for attr in attributes[5:]:
                print("Attribute {}.".format(attr))
                sel_samples = [s for s in samples if hasattr(s, attr)]
                sel_samples = [s for s in sel_samples if not pd.isnull(getattr(s, attr))]

                # Get all values of samples for this attr
                groups = set([getattr(s, attr) for s in sel_samples])

                # Determine if attr is categorical or continuous
                if all([type(i) in [str, bool] for i in groups]) or len(groups) == 2:
                    variable_type = "categorical"
                elif all([type(i) in [int, float, np.int64, np.float64] for i in groups]):
                    variable_type = "numerical"
                else:
                    print("attr %s cannot be tested." % attr)
                    associations.append([measurement, attr, variable_type, np.nan, np.nan, np.nan, np.nan])
                    continue

                if variable_type == "categorical":
                    # It categorical, test pairwise combinations of attributes
                    for group1, group2 in itertools.combinations(groups, 2):
                        g1_values = scores.loc[scores.index.isin([s.name for s in sel_samples if getattr(s, attr) == group1]), measurement]
                        g2_values = scores.loc[scores.index.isin([s.name for s in sel_samples if getattr(s, attr) == group2]), measurement]

                        # Test ANOVA (or Kruskal-Wallis H-test)
                        st, p = kruskal(g1_values, g2_values)

                        # Append
                        associations.append([measurement, attr, variable_type, group1, group2, p, st])

                elif variable_type == "numerical":
                    # It numerical, calculate pearson correlation
                    trait_values = [getattr(s, attr) for s in sel_samples]
                    st, p = pearsonr(scores[measurement].ix[[s.name for s in sel_samples]], trait_values)

                    associations.append([measurement, attr, variable_type, np.nan, np.nan, p, st])

        associations = pd.DataFrame(associations, columns=["score", "attribute", "variable_type", "group_1", "group_2", "p_value", "stat"])

        # write
        associations.to_csv(os.path.join(self.results_dir, "{}.{}.diff_regions.intensity_score_combined.associations.csv".format(output_suffix, trait)), index=False)

        # Plot
        # associations[associations['p_value'] < 0.05].drop(['group_1', 'group_2'], axis=1).drop_duplicates()
        # associations.drop(['group_1', 'group_2'], axis=1).drop_duplicates().pivot(index="score", columns="attribute", values="p_value")
        pivot = associations.groupby(["score", "attribute"]).min()['p_value'].reset_index().pivot(index="score", columns="attribute", values="p_value").dropna(axis=1)

        # heatmap of -log p-values
        g = sns.clustermap(-np.log10(pivot), row_cluster=False, annot=True, cbar_kws={"label": "-log10(p_value) of association"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        g.fig.savefig(os.path.join(self.results_dir, "{}.{}.diff_regions.intensity_score_combined.associations.svg".format(output_suffix, trait)), bbox_inches="tight")

        # heatmap of masked significant
        g = sns.clustermap((pivot < 0.05).astype(int), row_cluster=False, cbar_kws={"label": "significant association"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        g.fig.savefig(os.path.join(self.results_dir, "{}.{}.diff_regions.intensity_score_combined.associations.masked.svg".format(output_suffix, trait)), bbox_inches="tight")

        #

    def interaction_differential_analysis(
            self, samples, formula="~patient_id * timepoint_name",
            output_suffix="interaction"):
        """
        Discover differential regions for the interaction of patient and treatment.
        """
        def diff_score(x):
            s = np.log2(x["baseMean"]) * abs(x["log2FoldChange"]) * -np.log10(x["pvalue"])
            return s if s >= 0 else 0

        # Get matrix of counts
        counts_matrix = self.coverage[[s.name for s in self.samples]]

        # Get experiment matrix
        experiment_matrix = pd.DataFrame([s.as_series() for sample in self.samples], index=[s.name for sample in self.samples]).fillna("Unknown")

        # Make output dir
        output_dir = os.path.join(self.results_dir, output_suffix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Run DESeq2 analysis
        deseq_table = DESeq_interaction(
            counts_matrix, experiment_matrix, formula=formula, output_prefix=os.path.join(output_dir, output_suffix), alpha=0.05)
        deseq_table.columns = deseq_table.columns.str.replace(".", " ")
        # deseq_table = pd.read_csv(os.path.join(output_dir, output_suffix + ".all_patients.csv"), index_col=0)

        # Compute score
        deseq_table['score'] = np.log2(deseq_table["baseMean"]) * abs(deseq_table["log2FoldChange"]) * -np.log10(deseq_table["pvalue"])
        deseq_table.loc[deseq_table['score'] < 0, 'score'] = 0

        # Compute rankings
        for i, patient in enumerate(deseq_table['patient'].unique()):
            deseq_table.loc[deseq_table["patient"] == patient, "baseMean_rank"] = deseq_table.loc[
                deseq_table["patient"] == patient, "baseMean"].rank(ascending=False)
            deseq_table.loc[deseq_table["patient"] == patient, "log2FoldChange_rank"] = abs(deseq_table.loc[
                deseq_table["patient"] == patient, "log2FoldChange"]).rank(ascending=False)
            deseq_table.loc[deseq_table["patient"] == patient, "pvalue_rank"] = deseq_table.loc[
                deseq_table["patient"] == patient, "pvalue"].rank(ascending=True)
            deseq_table.loc[deseq_table["patient"] == patient, "combined_rank"] = deseq_table.loc[
                deseq_table["patient"] == patient, ["baseMean_rank", "log2FoldChange_rank", "pvalue_rank"]].max(axis=1)

        deseq_table = deseq_table.sort_values(['patient', "combined_rank"])

        df = self.coverage_annotated.join(deseq_table)
        df.to_csv(os.path.join(output_dir, output_suffix) + ".all_patients.annotated.csv")
        df = pd.read_csv(os.path.join(output_dir, output_suffix) + ".all_patients.annotated.csv")

        # Extract significant based on score
        diff = df[(df["score"] > 2 ** 4)]
        # Extract significant based on p-value, fold-change, mean accessibility
        diff = df[
            (df["baseMean"] > 10) &
            (abs(df["log2FoldChange"]) > 1) &
            (df['pvalue'] < 0.05)
        ]
        # Extract significant based on rank and p-value
        diff = pd.merge(df.reset_index(), df[df['pvalue'] < 0.05].groupby('patient')['combined_rank'].nsmallest(500).reset_index(0).reset_index()).set_index('index')

        if diff.shape[0] < 1:
            print("No significantly different regions found.")
            return

        # Statistics of differential regions
        import string
        total_sites = float(len(df.index.unique()))

        total_diff = diff.groupby(["patient"])['stat'].count().sort_values(ascending=False)
        fig, axis = plt.subplots(1)
        sns.barplot(total_diff.values, total_diff.index, orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.number_differential.total.svg" % output_suffix), bbox_inches="tight")
        # percentage of total
        fig, axis = plt.subplots(1)
        sns.barplot(
            (total_diff.values / total_sites) * 100,
            total_diff.index,
            orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.number_differential.total_percentage.svg" % output_suffix), bbox_inches="tight")

        # direction-dependent
        diff["direction"] = diff["log2FoldChange"].apply(lambda x: "up" if x >= 0 else "down")

        split_diff = diff.groupby(["patient", "direction"])['stat'].count().sort_values(ascending=False)
        fig, axis = plt.subplots(1, figsize=(12, 8))
        sns.barplot(
            split_diff.values,
            split_diff.reset_index()[['patient', 'direction']].apply(string.join, axis=1),
            orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.number_differential.split.svg" % output_suffix), bbox_inches="tight")
        # percentage of total
        fig, axis = plt.subplots(1, figsize=(12, 8))
        sns.barplot(
            (split_diff.values / total_sites) * 100,
            split_diff.reset_index()[['patient', 'direction']].apply(string.join, axis=1),
            orient="h", ax=axis)
        for t in axis.get_xticklabels():
            t.set_rotation(0)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.number_differential.split_percentage.svg" % output_suffix), bbox_inches="tight")

        # # Pyupset
        # import pyupset as pyu
        # # Build dict
        # diff["comparison_direction"] = diff[["comparison", "direction"]].apply(string.join, axis=1)
        # df_dict = {group: diff[diff["comparison_direction"] == group].reset_index()[['index']] for group in set(diff["comparison_direction"])}
        # # Plot
        # plot = pyu.plot(df_dict, unique_keys=['index'], inters_size_bounds=(10, np.inf))
        # plot['figure'].set_size_inches(20, 8)
        # plot['figure'].savefig(os.path.join(output_dir, "%s.%s.number_differential.upset.svg" % (output_suffix, trait)), bbox_inched="tight")

        groups = pd.Series(df['patient'].unique()).sort_values()

        # Score distribution per patient
        fig, axis = plt.subplots(int(len(groups) / 3.), (3), figsize=(3 * (len(groups) / 3.), 3 * (3)), sharex=True, sharey=True)
        axis = axis.flatten()
        for i, patient in enumerate(groups):
            sns.distplot(np.log2(1 + df[df['patient'] == patient]['score']), kde=False, ax=axis[i])
            axis[i].set_xlabel(patient)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.score_distribution.svg" % output_suffix), bbox_inches="tight")

        # Pairwise scatter plots per patient
        # norm_counts = pd.read_csv(os.path.join(output_dir, "%s.normalized_counts.csv" % output_suffix))
        # norm_counts = np.log2(1 + norm_counts)

        fig, axis = plt.subplots(int(len(groups) / 3.), (3), figsize=(2 * (len(groups) / 3.), 4 * (3)), sharex=True, sharey=True)
        axis = axis.flatten()
        for i, patient in enumerate(groups):
            # get samples from patient
            cond1 = [s.name for s in samples if s.patient_id == patient and s.timepoint_name == "after Ibrutinib"][0]
            cond2 = [s.name for s in samples if s.patient_id == patient and s.timepoint_name == "before Ibrutinib"][0]

            # Hexbin plot
            df2 = df[df['patient'] == patient]
            axis[i].hexbin(df2[cond1], df2[cond2], bins="log", alpha=.85)
            axis[i].set_xlabel(cond1)
            # Scatter plot
            diff2 = diff[diff['patient'] == patient]
            axis[i].scatter(diff2[cond1], diff2[cond2], alpha=0.1, color="red", s=2)
            axis[i].set_ylabel(cond2)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.scatter_plots.png" % output_suffix), bbox_inches="tight", dpi=300)

        # Volcano plots
        fig, axis = plt.subplots(int(len(groups) / 3.), (3), figsize=(3 * (len(groups) / 3.), 3 * (3)), sharex=True, sharey=True)
        axis = axis.flatten()
        for i, patient in enumerate(groups):
            # get samples from patient
            cond1 = [s.name for s in samples if s.patient_id == patient and s.timepoint_name == "after Ibrutinib"][0]
            cond2 = [s.name for s in samples if s.patient_id == patient and s.timepoint_name == "before Ibrutinib"][0]

            # Hexbin plot
            df2 = df[df['patient'] == patient]
            axis[i].hexbin(df2["log2FoldChange"], -np.log10(df2["pvalue"]), bins="log", alpha=.85)
            axis[i].set_xlabel(cond1)
            # Scatter plot
            diff2 = diff[diff['patient'] == patient]
            axis[i].scatter(diff2["log2FoldChange"], -np.log10(diff2["pvalue"]), alpha=0.1, color="red", s=2)
            axis[i].set_ylabel(cond2)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.volcano_plots.png" % output_suffix), bbox_inches="tight", dpi=300)

        # MA plots
        fig, axis = plt.subplots(int(len(groups) / 3.), (3), figsize=(3 * (len(groups) / 3.), 3 * (3)), sharex=True, sharey=True)
        axis = axis.flatten()
        for i, patient in enumerate(groups):
            # get samples from patient
            cond1 = [s.name for s in samples if s.patient_id == patient and s.timepoint_name == "after Ibrutinib"][0]
            cond2 = [s.name for s in samples if s.patient_id == patient and s.timepoint_name == "before Ibrutinib"][0]

            # Hexbin plot
            df2 = df[df['patient'] == patient]
            axis[i].hexbin(np.log2(df2["baseMean"]), df2["log2FoldChange"], bins="log", alpha=.85)
            axis[i].set_xlabel(cond1)
            # Scatter plot
            diff2 = diff[diff['patient'] == patient]
            axis[i].scatter(np.log2(diff2["baseMean"]), diff2["log2FoldChange"], alpha=0.1, color="red", s=2)
            axis[i].set_ylabel(cond2)
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.ma_plots.png" % output_suffix), bbox_inches="tight", dpi=300)

        # save unique differential regions
        diff2 = diff.ix[diff.index.unique()].drop_duplicates()
        diff2.to_csv(os.path.join(output_dir, "%s.differential_regions.csv" % output_suffix))

        # Exploration of differential regions
        # get unique differential regions
        df2 = pd.merge(diff2, self.coverage_annotated)

        # Characterize regions
        prefix = "%s.diff_regions" % output_suffix
        # region's structure
        # characterize_regions_structure(df=df2, prefix=prefix, output_dir=output_dir)
        # region's function
        # characterize_regions_function(df=df2, prefix=prefix, output_dir=output_dir)

        # Heatmaps
        # Sample level
        ax = sns.clustermap(df2[[s.name for s in samples]].corr(), xticklabels=False, metric="correlation")
        for item in ax.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        plt.savefig(os.path.join(output_dir, "%s.diff_regions.samples.clustermap.corr.png" % output_suffix), bbox_inches="tight", dpi=300)
        plt.close('all')

        ax = sns.clustermap(df2[[s.name for s in samples]], yticklabels=False, metric="correlation")
        for item in ax.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
        plt.savefig(os.path.join(output_dir, "%s.diff_regions.samples.clustermap.png" % output_suffix), bbox_inches="tight", dpi=300)
        plt.close('all')

        ax = sns.clustermap(df2[[s.name for s in samples]], yticklabels=False, z_score=0, metric="correlation")
        for item in ax.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
        plt.savefig(os.path.join(output_dir, "%s.diff_regions.samples.clustermap.z0.png" % output_suffix), bbox_inches="tight", dpi=300)
        plt.close('all')

        ax = sns.clustermap(df2[[s.name for s in samples]], yticklabels=False, z_score=0, metric="correlation")
        for item in ax.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
        plt.savefig(os.path.join(output_dir, "%s.diff_regions.samples.clustermap.z0.png" % output_suffix), bbox_inches="tight", dpi=300)
        plt.close('all')

        # Examine each region cluster
        region_enr = pd.DataFrame()
        lola_enr = pd.DataFrame()
        motif_enr = pd.DataFrame()
        pathway_enr = pd.DataFrame()
        for i, patient in enumerate(groups):
            # Separate in up/down-regulated regions
            for direction in diff["direction"].unique():
                comparison_df = self.coverage_annotated.ix[diff[
                    (diff["patient"] == patient) &
                    (diff["direction"] == direction)
                ].index]
                if comparison_df.shape[0] < 1:
                    continue
                # Characterize regions
                prefix = "%s.diff_regions.patient_%s.%s" % (output_suffix, patient, direction)

                patient_dir = os.path.join(output_dir, prefix)

                print("Doing regions of patient %s, with prefix %s" % (patient, prefix))

                # region's structure
                if not os.path.exists(os.path.join(patient_dir, prefix + "_regions.region_enrichment.csv")):
                    print(prefix)
                    characterize_regions_structure(df=comparison_df, prefix=prefix, output_dir=patient_dir)
                # region's function
                if not os.path.exists(os.path.join(patient_dir, prefix + "_genes.enrichr.csv")):
                    print(prefix)
                    characterize_regions_function(df=comparison_df, prefix=prefix, output_dir=patient_dir)

                # Read/parse enrichment outputs and add to DFs
                enr = pd.read_csv(os.path.join(patient_dir, prefix + "_regions.region_enrichment.csv"))
                enr.columns = ["region"] + enr.columns[1:].tolist()
                enr["patient"] = prefix
                region_enr = region_enr.append(enr, ignore_index=True)

                enr = pd.read_csv(os.path.join(patient_dir, "allEnrichments.txt"), sep="\t")
                enr["patient"] = prefix
                lola_enr = lola_enr.append(enr, ignore_index=True)

                enr = parse_ame(patient_dir).reset_index()
                enr["patient"] = prefix
                motif_enr = motif_enr.append(enr, ignore_index=True)

                enr = pd.read_csv(os.path.join(patient_dir, prefix + "_genes.enrichr.csv"))
                enr["patient"] = prefix
                pathway_enr = pathway_enr.append(enr, ignore_index=True)

        # write combined enrichments
        region_enr.to_csv(
            os.path.join(output_dir, "%s.diff_regions.regions.csv" % (output_suffix)), index=False)
        lola_enr.to_csv(
            os.path.join(output_dir, "%s.diff_regions.lola.csv" % (output_suffix)), index=False)
        motif_enr.columns = ["motif", "p_value", "comparison"]
        motif_enr.to_csv(
            os.path.join(output_dir, "%s.diff_regions.motifs.csv" % (output_suffix)), index=False)
        pathway_enr.to_csv(
            os.path.join(output_dir, "%s.diff_regions.enrichr.csv" % (output_suffix)), index=False)

    def investigate_interaction_regions(self, output_suffix="interaction", n=50):
        import string
        from scipy.cluster.hierarchy import fcluster

        output_dir = os.path.join(self.results_dir, output_suffix)

        # REGION TYPES
        # read in
        regions = pd.read_csv(os.path.join(output_dir, "%s.diff_regions.regions.csv" % output_suffix))
        # pretty names
        regions["patient"] = regions["patient"].str.extract("%s.diff_regions.patient_(.*)" % output_suffix, expand=True)

        # pivot table
        regions_pivot = pd.pivot_table(regions, values="value", columns="region", index="patient")

        # fillna
        regions_pivot = regions_pivot.fillna(0)

        # plot correlation
        fig = sns.clustermap(regions_pivot)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "region_type_enrichment.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "region_type_enrichment.png"), bbox_inches="tight", dpi=300)

        #

        # LOLA
        # read in
        lola = pd.read_csv(os.path.join(output_dir, "%s.diff_regions.lola.csv" % output_suffix))
        # pretty names
        lola["patient"] = lola["patient"].str.extract("%s.diff_regions.patient_(.*)" % output_suffix, expand=True)

        # unique ids for lola sets
        cols = ['description', u'cellType', u'tissue', u'antibody', u'treatment', u'dataSource', u'filename']
        lola['label'] = lola[cols].astype(str).apply(string.join, axis=1)

        # pivot table
        lola_pivot = pd.pivot_table(lola, values="pValueLog", columns="label", index="patient")
        lola_pivot.columns = lola_pivot.columns.str.decode("utf-8")

        # plot correlation
        fig = sns.clustermap(lola_pivot.T.corr())
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "lola.correlation.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "lola.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(fig.dendrogram_col.linkage, 3, criterion="maxclust")

        # Get top n terms which are more in each cluster compared with all others
        top_terms = list()
        cluster_means = pd.DataFrame()
        for cluster in set(cluster_assignment):
            cluster_patients = lola_pivot.index[cluster_assignment == cluster].tolist()
            other_patients = lola_pivot.index[cluster_assignment != cluster].tolist()

            terms = (lola_pivot.ix[cluster_patients].mean() - lola_pivot.ix[other_patients].mean()).sort_values()

            top_terms += terms.dropna().head(n).index.tolist()

            # additionallly, get mean of cluster
            cluster_means[cluster] = lola_pivot.ix[cluster_patients].mean()

        # plot clustered heatmap
        fig = sns.clustermap(lola_pivot[list(set(top_terms))].replace({np.inf: 50}), z_score=0, figsize=(20, 12))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "lola.cluster_specific.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "lola.cluster_specific.png"), bbox_inches="tight", dpi=300)

        #

        # MOTIFS
        # read in
        motifs = pd.read_csv(os.path.join(output_dir, "%s.diff_regions.motifs.csv" % output_suffix))
        # pretty names
        motifs["patient"] = motifs["comparison"].str.extract("%s.diff_regions.patient_(.*)" % output_suffix, expand=True)

        # pivot table
        motifs_pivot = pd.pivot_table(motifs, values="p_value", columns="motif", index="patient")

        # transform p-values
        motifs_pivot = -np.log10(motifs_pivot.fillna(1))
        motifs_pivot = motifs_pivot.replace({np.inf: 300})

        # plot correlation
        fig = sns.clustermap(motifs_pivot.T.corr())
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "motifs.correlation.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "motifs.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(fig.dendrogram_col.linkage, 5, criterion="maxclust")

        # Get top n terms which are more in each cluster compared with all others
        top_terms = list()
        cluster_means = pd.DataFrame()
        for cluster in set(cluster_assignment):
            cluster_patients = motifs_pivot.index[cluster_assignment == cluster].tolist()
            other_patients = motifs_pivot.index[cluster_assignment != cluster].tolist()

            terms = (motifs_pivot.ix[cluster_patients].mean() - motifs_pivot.ix[other_patients].mean()).sort_values()

            top_terms += terms.dropna().head(n).index.tolist()

        # plot clustered heatmap
        fig = sns.clustermap(motifs_pivot[list(set(top_terms))].apply(lambda x: (x - x.mean()) / x.std()), figsize=(20, 12))  #
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.png"), bbox_inches="tight", dpi=300)

        df = motifs_pivot[list(set(top_terms))].apply(lambda x: (x - x.mean()) / x.std())

        fig = sns.clustermap(df[df.mean(1) > -0.5], figsize=(20, 12))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.png"), bbox_inches="tight", dpi=300)

        #

        # ENRICHR
        # read in
        enrichr = pd.read_csv(os.path.join(output_dir, "%s.diff_regions.enrichr.csv" % output_suffix))
        # pretty names
        enrichr["patient"] = enrichr["patient"].str.extract("%s.diff_regions.patient_(.*)" % output_suffix, expand=True)

        for gene_set_library in enrichr["gene_set_library"].unique():
            print(gene_set_library)
            if gene_set_library == "Epigenomics_Roadmap_HM_ChIP-seq":
                continue

            # pivot table
            enrichr_pivot = pd.pivot_table(
                enrichr[enrichr["gene_set_library"] == gene_set_library],
                values="adjusted_p_value", columns="description", index="patient")
            enrichr_pivot.columns = enrichr_pivot.columns.str.decode("utf-8")

            # transform p-values
            enrichr_pivot = -np.log10(enrichr_pivot.fillna(1))
            enrichr_pivot = enrichr_pivot.replace({np.inf: 300})

            # plot correlation
            fig = sns.clustermap(enrichr_pivot.T.corr())
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "enrichr.%s.correlation.svg" % gene_set_library), bbox_inches="tight")
            fig.savefig(os.path.join(output_dir, "enrichr.%s.correlation.png" % gene_set_library), bbox_inches="tight", dpi=300)

            cluster_assignment = fcluster(fig.dendrogram_col.linkage, 4, criterion="maxclust")

            # Get top n terms which are more in each cluster compared with all others
            top_terms = list()
            cluster_means = pd.DataFrame()
            for cluster in set(cluster_assignment):
                cluster_comparisons = enrichr_pivot.index[cluster_assignment == cluster].tolist()
                other_comparisons = enrichr_pivot.index[cluster_assignment != cluster].tolist()

                terms = (enrichr_pivot.ix[cluster_comparisons].mean() - enrichr_pivot.ix[other_comparisons].mean()).sort_values()

                top_terms += terms.dropna().head(n).index.tolist()

            # plot clustered heatmap
            fig = sns.clustermap(enrichr_pivot[list(set(top_terms))].apply(lambda x: (x - x.mean()) / x.std()), figsize=(20, 12))  #
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.svg" % gene_set_library), bbox_inches="tight")
            fig.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.png" % gene_set_library), bbox_inches="tight", dpi=300)


def annotate_drugs(analysis, sensitivity):
    """
    """
    from bioservices.kegg import KEGG
    from collections import defaultdict
    import string
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

    output_dir = os.path.join(analysis.results_dir, "pharmacoscopy")

    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    rename = {
        "ABT-199 = Venetoclax": "ABT-199",
        "ABT-263 = Navitoclax": "ABT-263",
        "ABT-869 = Linifanib": "ABT-869",
        "AC220 = Quizartinib": "AC220",
        "Buparlisib (BKM120)": "BKM120",
        "EGCG = Epigallocatechin gallate": "Epigallocatechin gallate",
        "JQ1": "(+)-JQ1",
        "MLN-518 = Tandutinib": "MLN-518",
        "Selinexor (KPT-330)": "KPT-330"}
    sensitivity["proper_name"] = sensitivity["drug"]
    for p, n in rename.items():
        sensitivity["proper_name"] = sensitivity["proper_name"].replace(p, n)

    # CLOUD id/ SMILES
    cloud = pd.read_csv(os.path.join("metadata", "CLOUD_simple_annotation.csv"))
    cloud.loc[:, "name_lower"] = cloud["drug_name"].str.lower()
    sensitivity.loc[:, "name_lower"] = sensitivity["proper_name"].str.lower()
    annot = pd.merge(sensitivity[['drug', 'proper_name', 'name_lower']].drop_duplicates(), cloud, on="name_lower", how="left")

    # DGIdb: drug -> genes
    interact = pd.read_csv("http://dgidb.genome.wustl.edu/downloads/interactions.tsv", sep="\t")
    interact.loc[:, "name_lower"] = interact["drug_primary_name"].str.lower()
    cats = pd.read_csv("http://dgidb.genome.wustl.edu/downloads/categories.tsv", sep="\t")
    dgidb = pd.merge(interact, cats, how="left")
    dgidb.to_csv(os.path.join(output_dir, "dgidb.interactions_categories.csv"), index=False)
    # tight match
    annot = pd.merge(annot, dgidb, on="name_lower", how="left")
    annot.to_csv(os.path.join("metadata", "drugs_annotated.csv"), index=False)
    annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.csv"))

    # KEGG: gene -> pathways
    k = KEGG()
    k.organism = "hsa"
    paths = pd.DataFrame()
    for gene in annot['entrez_gene_symbol'].drop_duplicates().dropna().sort_values():
        print(gene)
        try:
            s = pd.Series(k.get_pathway_by_gene(gene, "hsa"))
        except AttributeError:
            continue
        # if not s.isnull().all():
        s = s.reset_index()
        s['gene'] = gene
        paths = paths.append(s)
    paths.columns = ['kegg_pathway_id', 'kegg_pathway_name', 'entrez_gene_symbol']
    paths = paths.sort_values(['entrez_gene_symbol', 'kegg_pathway_name'])
    # match
    annot = pd.merge(annot, paths, on="entrez_gene_symbol", how="left")
    annot.to_csv(os.path.join("metadata", "drugs_annotated.csv"), index=False)
    annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.csv"))

    # Cleanup (reduce redudancy of some)
    annot = annot.replace("n/a", pd.np.nan)
    annot = annot.replace("other/unknown", pd.np.nan)

    # Score interactions
    # values as direction of interaction
    score_map = defaultdict(lambda: 0)
    score_map.update({
        'inducer': 1,
        'partial agonist': 1,
        'agonist': 1,
        'ligand': 0,
        'binder': 0,
        'multitarget': 0,
        'adduct': 0,
        'inhibitor': -1,
        'antagonist': -1,
        'negative modulator': -1,
        'competitive,inhibitor': -1,
        'inhibitor,competitive': -1,
        'multitarget,antagonist': -1,
        'blocker': -1,
    })
    # score interactions
    annot['interaction_score'] = annot['interaction_types'].apply(lambda x: score_map[x])
    annot.to_csv(os.path.join("metadata", "drugs_annotated.csv"), index=False)
    annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.csv"))
    analysis.drug_annotation = annot

    # Create directed Drug -> Gene network
    gene_net = annot[['drug', 'interaction_types', 'entrez_gene_symbol']].drop_duplicates()
    gene_net = (
        gene_net.groupby(['drug', 'entrez_gene_symbol'])
        ['interaction_types']
        .apply(lambda x: pd.Series([i for i in x if not pd.isnull(i)] if not pd.isnull(x).all() else pd.np.nan))
        .reset_index()
    )[['drug', 'interaction_types', 'entrez_gene_symbol']]
    # score interactions
    gene_net['interaction_score'] = gene_net['interaction_types'].apply(lambda x: score_map[x])
    gene_net.to_csv(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.tsv"), sep='\t', index=False)
    # collapse annotation to one row per drug (one classification per drug)
    interactions = (
        gene_net.groupby(['drug'])['interaction_types']
        .apply(lambda x: x.value_counts().argmax() if not pd.isnull(x).all() else pd.np.nan))
    genes = gene_net.groupby(['drug'])['entrez_gene_symbol'].aggregate(string.join, sep=";")
    collapsed_net = pd.DataFrame([genes, interactions]).T.reset_index()
    collapsed_net[['drug', 'interaction_types', 'entrez_gene_symbol']].to_csv(
        os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.reduced_to_drug.tsv"), sep='\t', index=False)

    # Create directed Drug -> Pathway network
    path_net = annot[['drug', 'interaction_types', 'kegg_pathway_name']].drop_duplicates()
    path_net = (
        path_net.groupby(['drug', 'kegg_pathway_name'])
        ['interaction_types']
        .apply(lambda x: pd.Series([i for i in x if not pd.isnull(i)] if not pd.isnull(x).all() else pd.np.nan))
        .reset_index()
    )[['drug', 'interaction_types', 'kegg_pathway_name']]
    # score interactions
    path_net['interaction_score'] = path_net['interaction_types'].apply(lambda x: score_map[x])
    path_net.to_csv(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.tsv"), sep='\t', index=False)
    # collapse annotation to one row per drug (one classification per drug)
    interactions = (
        path_net.groupby(['drug'])['interaction_types']
        .apply(lambda x: x.value_counts().argmax() if not pd.isnull(x).all() else pd.np.nan))
    genes = path_net.groupby(['drug'])['kegg_pathway_name'].aggregate(string.join, sep=";")
    collapsed_net = pd.DataFrame([genes, interactions]).T.reset_index()
    collapsed_net[['drug', 'interaction_types', 'kegg_pathway_name']].to_csv(
        os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.reduced_to_drug.tsv"), sep='\t', index=False)

    # Plot annotations

    # distributions of drug-gene annotations
    fig, axis = plt.subplots(2, 3, figsize=(12, 8))
    axis = axis.flatten()

    # how many genes per drug?
    axis[0].set_title("Genes per drug")
    mats0 = annot.groupby(["proper_name"])['entrez_gene_symbol'].nunique()
    mats0_a = dgidb.groupby(["drug_primary_name"])['entrez_gene_symbol'].nunique()
    axis[0].hist([mats0, mats0_a], max(mats0.tolist() + mats0_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[0].set_xlabel("Genes")

    ax = zoomed_inset_axes(axis[0], 3, loc=1, axes_kwargs={"xlim": (0, 30), "ylim": (0, .20)})
    ax.hist([mats0, mats0_a], max(mats0.tolist() + mats0_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)

    # support of each drug-> gene assignment
    axis[1].set_title("Support of each interaction")
    mats1 = annot.groupby(["proper_name", "entrez_gene_symbol"])['interaction_claim_source'].nunique()
    mats1_a = dgidb.groupby(["drug_primary_name", "entrez_gene_symbol"])['interaction_claim_source'].nunique()
    axis[1].hist([mats1, mats1_a], max(mats1.tolist() + mats1_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[1].set_xlabel("Sources")

    # types of interactions per drug (across genes)
    axis[2].set_title("Interaction types per drug")
    mats2 = annot.groupby(["proper_name"])['interaction_types'].nunique()
    mats2_a = dgidb.groupby(["drug_primary_name"])['interaction_types'].nunique()
    axis[2].hist([mats2, mats2_a], max(mats2.tolist() + mats2_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[2].set_xlabel("Interaction types")

    # types of interactions per drug-> assignemnt
    axis[3].set_title("Interactions types per drug->gene interaction")
    mats3 = annot.groupby(["proper_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    mats3_a = dgidb.groupby(["drug_primary_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    axis[3].hist([mats3, mats3_a], max(mats3.tolist() + mats3_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[3].set_xlabel("Interaction types")

    # types of categories per drug (across genes)
    axis[4].set_title("Categories per drug")
    mats4 = annot.groupby(["proper_name"])['interaction_types'].nunique()
    mats4_a = dgidb.groupby(["drug_primary_name"])['interaction_types'].nunique()
    axis[4].hist([mats4, mats4_a], max(mats4.tolist() + mats4_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[4].set_xlabel("Categories")

    # types of categories per drug-> assignemnt
    axis[5].set_title("Categories per drug->gene interaction")
    mats5 = annot.groupby(["proper_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    mats5_a = dgidb.groupby(["drug_primary_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    axis[5].hist([mats5, mats5_a], max(mats5.tolist() + mats5_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[5].set_xlabel("Categories")

    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene.annotations.svg"), bbox_inches="tight")

    # distributions of drug-pathway annotations
    fig, axis = plt.subplots(2, 3, figsize=(12, 8))
    axis = axis.flatten()

    # how many genes per drug?
    axis[0].set_title("Pathways per drug")
    mats0 = annot.groupby(["proper_name"])['kegg_pathway_name'].nunique()
    sns.distplot(mats0, hist_kws={"normed": True}, kde=False, bins=50, ax=axis[0])
    axis[0].set_xlabel("pathways")

    # support of each drug-> pathway assignment
    axis[1].set_title("Support of each interaction")
    mats1 = annot.groupby(["proper_name", "kegg_pathway_name"])['interaction_claim_source'].nunique()
    sns.distplot(mats1, hist_kws={"normed": True}, kde=False, ax=axis[1])
    axis[1].set_xlabel("Sources")

    # types of interactions per drug (across pathways)
    axis[2].set_title("Interaction types per drug")
    mats2 = annot.groupby(["proper_name"])['interaction_types'].nunique()
    sns.distplot(mats2, hist_kws={"normed": True}, kde=False, ax=axis[2])
    axis[2].set_xlabel("Interaction types")

    # types of interactions per drug-> assignemnt
    axis[3].set_title("Interactions types per drug->pathway interaction")
    mats3 = annot.groupby(["proper_name", "kegg_pathway_name"])['interaction_types'].nunique()
    sns.distplot(mats3, hist_kws={"normed": True}, kde=False, ax=axis[3])
    axis[3].set_xlabel("Interaction types")

    # scores per drug (across pathways)
    axis[4].set_title("Score per drug")
    mats4 = annot[["proper_name", 'kegg_pathway_name', 'interaction_score']].drop_duplicates()['interaction_score']
    sns.distplot(mats4, hist_kws={"normed": True}, kde=False, ax=axis[4])
    axis[4].set_xlabel("Score")

    # score per drug-> assignemnt
    axis[5].set_title("Mean score per drug->pathway interaction")
    mats5 = annot.groupby(["proper_name", "kegg_pathway_name"])['interaction_score'].mean()
    sns.distplot(mats5, hist_kws={"normed": True}, kde=False, ax=axis[5])
    axis[5].set_xlabel("Score")

    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway.annotations.svg"), bbox_inches="tight")

    # Drug vs Category
    annot['intercept'] = 1
    cat_pivot = pd.pivot_table(annot, index="drug", columns='category', values="intercept")
    # get drugs with not gene annotated and add them to matrix
    for drug in annot['drug'].drop_duplicates()[~annot['drug'].drop_duplicates().isin(cat_pivot.index)]:
        cat_pivot.loc[drug, :] = pd.np.nan
    # heatmap
    fig = sns.clustermap(cat_pivot.fillna(0), figsize=(7, 12))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-categories.binary.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-categories.binary.svg"), bbox_inches="tight")

    # Drug vs Gene matrix heatmap
    gene_pivot = pd.pivot_table(gene_net, index="drug", columns='entrez_gene_symbol', values="interaction_score", aggfunc=sum)
    # get drugs with not gene annotated and add them to matrix
    for drug in annot['drug'].drop_duplicates()[~annot['drug'].drop_duplicates().isin(gene_pivot.index)]:
        gene_pivot.loc[drug, :] = pd.np.nan
    # heatmap
    fig = sns.clustermap(gene_pivot.fillna(0), figsize=(20, 10))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.svg"), bbox_inches="tight")
    # binary
    path_pivot_binary = (~gene_pivot.isnull()).astype(int)
    fig = sns.clustermap(path_pivot_binary, figsize=(20, 10))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.binary.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.binary.svg"), bbox_inches="tight")

    # Drug vs Gene matrix heatmap
    path_pivot = pd.pivot_table(path_net, index="drug", columns='kegg_pathway_name', values="interaction_score", aggfunc=sum)
    # get drugs with not gene annotated and add them to matrix
    for drug in annot['drug'].drop_duplicates()[~annot['drug'].drop_duplicates().isin(path_pivot.index)]:
        path_pivot.loc[drug, :] = pd.np.nan
    # heatmap
    fig = sns.clustermap(path_pivot.fillna(0), figsize=(20, 10))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.svg"), bbox_inches="tight")
    # binary
    path_pivot_binary = (~path_pivot.isnull()).astype(int)
    fig = sns.clustermap(path_pivot_binary, figsize=(20, 10))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.binary.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.binary.svg"), bbox_inches="tight")

    # cluster, label drugs with function


def pharmacoscopy(analysis):
    """
    """
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind
    from statsmodels.sandbox.stats.multicomp import multipletests
    import scipy
    from statsmodels.nonparametric.smoothers_lowess import lowess

    def z_score(x):
        return (x - x.mean()) / x.std()

    def standard_score(x):
        return (x - x.min()) / (x.max() - x.min())

    def mean_standard_score(x):
        re_x = (x - x.min()) / (x.max() - x.min())
        return re_x / re_x.mean()

    def drug_to_pathway_space(drug_vector, drug_annotation, plot=False, plot_label=None):
        """
        Convert a vector of measurements for each drug (total cell numbers, sensitivities)
        to a vector of pathway scores based on known drug -> pathway interactions (weighted or unweighted).
        """
        def nan_prod(a, b):
            return np.nanprod(np.array([a, b]))

        def multiply(vector, matrix):
            res = pd.DataFrame(np.empty(matrix.shape), index=matrix.index, columns=matrix.columns)
            for col in matrix.columns:
                res.loc[:, col] = nan_prod(vector, matrix[col])
            return res

        # reduce drug->pathway interaction by number of interactions
        interaction_number = drug_annotation.groupby(["drug", "kegg_pathway_name"])['intercept'].sum().reset_index()

        pathway_matrix = pd.pivot_table(
            # this would reduce interaction scores to 1 per drug->pathway interaction (by mean):
            # drug_annotation.groupby(["drug", "kegg_pathway_name"])['interaction_score'].mean().reset_index(),
            # however, I now prefer to use the total count of interactions, normalized across pathways
            interaction_number,
            index="drug", columns="kegg_pathway_name", values="intercept")

        # normalize across pathways (to compensate for more investigated pathways)
        dist = interaction_number.groupby("kegg_pathway_name")['intercept'].count().sort_values()
        pathway_matrix_norm = pathway_matrix / dist

        pathway_scores = multiply(vector=drug_vector, matrix=pathway_matrix_norm) / pathway_matrix_norm

        if plot:
            if 'master_fig' not in globals():
                global master_fig
                master_fig = sns.clustermap(np.log2(1 + pathway_matrix.fillna(0)), figsize=(30, 15))
                for tick in master_fig.ax_heatmap.get_xticklabels():
                    tick.set_rotation(90)
                for tick in master_fig.ax_heatmap.get_yticklabels():
                    tick.set_rotation(0)
                master_fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.score.png"), bbox_inches="tight", dpi=300)
                # fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.score.svg"), bbox_inches="tight")

                fig = sns.clustermap(np.log2(1 + pathway_matrix_norm.fillna(0)), figsize=(30, 15), row_linkage=master_fig.dendrogram_row.linkage, col_linkage=master_fig.dendrogram_col.linkage)
                for tick in fig.ax_heatmap.get_xticklabels():
                    tick.set_rotation(90)
                for tick in fig.ax_heatmap.get_yticklabels():
                    tick.set_rotation(0)
                fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.weighted_score.png"), bbox_inches="tight", dpi=300)
                # fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.weighted_score.svg"), bbox_inches="tight")

            fig = sns.clustermap(np.log2(1 + pathway_scores.fillna(0)), figsize=(30, 15), row_linkage=master_fig.dendrogram_row.linkage, col_linkage=master_fig.dendrogram_col.linkage)
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.{}.sample_scores.png".format(plot_label)), bbox_inches="tight", dpi=300)
            # fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.{}.sample_scores.svg".format(plot_label)), bbox_inches="tight")

        pathway_vector = pathway_scores.mean(axis=0)
        new_drug_space = pathway_scores.mean(axis=1)

        return pathway_vector, new_drug_space

    output_dir = os.path.join(analysis.results_dir, "pharmacoscopy")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    auc = pd.read_csv(os.path.join("metadata", "pharmacoscopy_AUC_v3.csv")).dropna()

    # Read up drug annotation
    # annot = analysis.drug_annotation
    # annot = pd.read_csv(os.path.join(analysis.data_dir, "drugs_annotated.csv"))

    # transform AUC to inverse
    auc["AUC"] *= -1

    # demonstrate the scaling
    fig, axis = plt.subplots(1, 2, figsize=(4 * 2, 4 * 1), sharex=False, sharey=False)
    axis = axis.flatten()
    # axis[0].set_title("original sensitivity")
    sns.distplot(sensitivity['score'], bins=100, kde=False, ax=axis[0], label="original score")
    # ax = zoomed_inset_axes(axis[0], zoom=6, loc=1, axes_kwargs={"aspect": "auto", "xlim": (-2, 5), "ylim": (0, 100000)})
    # sns.distplot(sensitivity['score'], bins=300, kde=False, ax=ax)
    axis[0].legend()
    # axis[2].set_title("original AUC")
    sns.distplot(auc['AUC'].dropna(), bins=100, kde=False, ax=axis[1], label="original AUC")
    # ax = zoomed_inset_axes(axis[1], zoom=6, loc=1, axes_kwargs={"aspect": "auto", "xlim": (-20, 20), "ylim": (0, 300)})
    # sns.distplot(auc['AUC'].dropna(), bins=300, kde=False, ax=ax)
    axis[1].legend()
    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "scaling_demo.svg"), bbox_inches="tight")

    # merge in one table
    # (
    #     pd.merge(pd.merge(auc, cd19, how="outer"), sensitivity, how="outer")
    #     .sort_values(['patient_id', 'timepoint', 'drug', 'concentration'])
    #     .to_csv(os.path.join(analysis.data_dir, "pharmacoscopy_all.csv"), index=False))

    # Plot distributions
    # auc & sensitivity
    for df, name in [(sensitivity, "score"), (auc, "AUC")]:
        df['p_id'] = df['patient_id'].astype(str) + " " + df['timepoint_name'].astype(str)

        # Mean across patients
        df_mean = df.groupby(['drug', 'timepoint_name'])[name].mean().reset_index()
        t = len(df_mean['timepoint_name'].unique())
        fig, axis = plt.subplots(1, t, figsize=(4 * t, 4))
        for i, timepoint_name in enumerate(df_mean['timepoint_name'].unique()):
            sns.distplot(df_mean[df_mean['timepoint_name'] == timepoint_name][name].dropna(), kde=False, ax=axis[i])
            axis[i].set_title("Mean {} across patients in {}".format(name, timepoint_name))
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "{}.timepoints.distplot.svg".format(name)), bbox_inches="tight")

        # Heatmaps
        # clustered
        df_pivot = pd.pivot_table(df, index="drug", columns="p_id", values=name)
        fig = sns.clustermap(df_pivot, figsize=(8, 20))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.svg".format(name)), bbox_inches="tight")

        # sorted
        p = df_pivot.ix[df_pivot.sum(axis=1).sort_values().index]  # sort drugs
        p = p[p.sum(axis=0).sort_values().index]  # sort samples
        fig = sns.clustermap(p, figsize=(8, 20), col_cluster=False, row_cluster=False)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.both_axis_sorted.svg".format(name)), bbox_inches="tight")\

        # across samples in each timepoint - sorted
        a = df_pivot[df_pivot.columns[df_pivot.columns.str.contains("after")]].mean(axis=1)
        b = df_pivot[df_pivot.columns[df_pivot.columns.str.contains("before")]].mean(axis=1)
        a.name = "after_ibrutinib"
        b.name = "before_ibrutinib"

        p = pd.DataFrame([a, b]).T
        p = p.ix[df_pivot.sum(axis=1).sort_values().index]  # sort drugs
        p = p[p.sum(axis=0).sort_values().index]  # sort samples
        fig = sns.clustermap(p, figsize=(8, 20), col_cluster=False, row_cluster=False)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.across_patients.both_axis_sorted.svg".format(name)), bbox_inches="tight")

        # scatter
        from statsmodels.nonparametric.smoothers_lowess import lowess
        fit = lowess(b, a, return_sorted=False)
        dist = abs(b - fit)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=dist.max())

        fig, axis = plt.subplots(1, figsize=(4, 4))
        axis.scatter(a, b, color=plt.cm.inferno(norm(dist)))
        for l in (fit - b).sort_values().tail(10).index:
            axis.text(a.ix[l], b.ix[l], l, fontsize=7.5)
        for l in (b - fit).sort_values().tail(10).index:
            axis.text(a.ix[l], b.ix[l], l, fontsize=7.5)
        axis.set_xlabel("After ibrutinib")
        axis.set_ylabel("Before ibrutinib")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "{}.across_patients.scatter.svg".format(name)), bbox_inches="tight")

        # Difference between timepoints
        df_pivot2 = (
            df.groupby(['patient_id', 'drug', 'timepoint_name'])
            [name].mean()
            .reset_index())
        a = df_pivot2[df_pivot2['timepoint_name'] == "after_Ibrutinib"].pivot_table(index="patient_id", columns="drug", values=name)
        b = df_pivot2[df_pivot2['timepoint_name'] == "before_Ibrutinib"].pivot_table(index="patient_id", columns="drug", values=name)

        abs_diff = a - b

        # clustered
        fig = sns.clustermap(abs_diff.dropna(), figsize=(20, 8))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.abs_diff.svg".format(name)), bbox_inches="tight")

        # clustered
        p = abs_diff.dropna().ix[abs_diff.dropna().sum(axis=1).sort_values().index]  # sort drugs
        p = p[p.sum(axis=0).sort_values().index]  # sort samples
        fig = sns.clustermap(p, figsize=(20, 8), col_cluster=False, row_cluster=False)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.abs_diff.both_axis_sorted.svg".format(name)), bbox_inches="tight")

    # Unsupervised analysis on pharmacoscopy data
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from collections import OrderedDict
    import re
    import itertools
    from scipy.stats import kruskal
    from scipy.stats import pearsonr

    for name, score, matrix in [("sensitivity", "score", sensitivity), ("auc", "AUC", auc)]:

        # Make drug - sample pivot table
        matrix['id'] = matrix['patient_id'] + " - " + matrix['timepoint_name']
        matrix = pd.pivot_table(data=matrix, index="drug", columns=['id', 'patient_id', 'sample_id', 'pharmacoscopy_id', 'timepoint_name'], values=score)

        color_dataframe = pd.DataFrame(
            analysis.get_level_colors(index=matrix.columns, levels=matrix.columns.names),
            index=matrix.columns.names,
            columns=matrix.columns.get_level_values("id"))

        # Pairwise correlations
        g = sns.clustermap(
            matrix.corr(), xticklabels=False, yticklabels=matrix.columns.get_level_values("id"), annot=True,
            cmap="Spectral_r", figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        g.ax_heatmap.set_xlabel(None)
        g.ax_heatmap.set_ylabel(None)
        g.fig.savefig(os.path.join(output_dir, "{}.corr.clustermap.svg".format(name)), bbox_inches='tight')

        # MDS
        to_plot = ["patient_id", "timepoint_name"]
        mds = MDS(n_jobs=-1)
        x_new = mds.fit_transform(matrix.T)
        # transform again
        x = pd.DataFrame(x_new)
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)

        fig, axis = plt.subplots(1, len(to_plot), figsize=(4 * len(to_plot), 4 * 1))
        axis = axis.flatten()
        for i, attr in enumerate(to_plot):
            for j in range(len(xx)):
                try:
                    label = matrix.columns.get_level_values(attr)[j]
                except AttributeError:
                    label = np.nan
                axis[i].scatter(xx.ix[j][0], xx.ix[j][1], s=50, color=color_dataframe.ix[attr][j], label=label)
            axis[i].set_title(to_plot[i])
            axis[i].set_xlabel("MDS 1")
            axis[i].set_ylabel("MDS 2")
            axis[i].set_xticklabels(axis[i].get_xticklabels(), visible=False)
            axis[i].set_yticklabels(axis[i].get_yticklabels(), visible=False)

            # Unique legend labels
            handles, labels = axis[i].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            if any([type(c) in [str, unicode] for c in by_label.keys()]) and len(by_label) <= 20:
                if not any([re.match("^\d", c) for c in by_label.keys()]):
                    axis[i].legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(output_dir, "{}.mds.svg".format(name)), bbox_inches="tight")

        # PCA
        pca = PCA()
        x_new = pca.fit_transform(matrix.T)
        # transform again
        x = pd.DataFrame(x_new)
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)

        # plot % explained variance per PC
        fig, axis = plt.subplots(1)
        axis.plot(
            range(1, len(pca.explained_variance_) + 1),  # all PCs
            (pca.explained_variance_ / pca.explained_variance_.sum()) * 100, 'o-')  # % of total variance
        axis.axvline(len(to_plot), linestyle='--')
        axis.set_xlabel("PC")
        axis.set_ylabel("% variance")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "{}.pca.explained_variance.svg".format(name)), bbox_inches='tight')

        # plot
        pcs = min(xx.shape[0] - 1, 10)
        fig, axis = plt.subplots(pcs, len(to_plot), figsize=(4 * len(to_plot), 4 * pcs))
        for pc in range(pcs):
            for i, attr in enumerate(to_plot):
                for j in range(len(xx)):
                    try:
                        label = matrix.columns.get_level_values(attr)[j]
                    except AttributeError:
                        label = np.nan
                    axis[pc, i].scatter(xx.ix[j][pc], xx.ix[j][pc + 1], s=50, color=color_dataframe.ix[attr][j], label=label)
                axis[pc, i].set_title(to_plot[i])
                axis[pc, i].set_xlabel("PC {}".format(pc + 1))
                axis[pc, i].set_ylabel("PC {}".format(pc + 2))
                axis[pc, i].set_xticklabels(axis[pc, i].get_xticklabels(), visible=False)
                axis[pc, i].set_yticklabels(axis[pc, i].get_yticklabels(), visible=False)

                # Unique legend labels
                handles, labels = axis[pc, i].get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                if any([type(c) in [str, unicode] for c in by_label.keys()]) and len(by_label) <= 20:
                    if not any([re.match("^\d", c) for c in by_label.keys()]):
                        axis[pc, i].legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(output_dir, "{}.pca.svg".format(name)), bbox_inches="tight")

    #

    # Convert each sample to pathway-space
    pathway_space = pd.DataFrame()
    new_drugs = pd.DataFrame()
    for patient_id in sensitivity['patient_id'].drop_duplicates().sort_values():
        for timepoint in sensitivity['timepoint_name'].drop_duplicates().sort_values(ascending=False):

                sample_id = "-".join([patient_id, timepoint])
                print(sample_id)
                # get respective data and reduce replicates (by mean)
                drug_vector = sensitivity.loc[
                    (
                        (sensitivity['patient_id'] == patient_id) &
                        (sensitivity['timepoint_name'] == timepoint)),
                    ['drug', 'score']].groupby('drug').mean().squeeze()

                # covert between spaces
                pathway_vector, new_drug_space = drug_to_pathway_space(
                    drug_vector=drug_vector,
                    drug_annotation=analysis.drug_annotation,
                    plot=False,
                    plot_label=sample_id)

                # save
                pathway_space[sample_id] = pathway_vector
                new_drugs[sample_id] = new_drug_space
    pathway_space.to_csv(os.path.join(output_dir, "pharmacoscopy.score.pathway_space.csv"))
    new_drugs.to_csv(os.path.join(output_dir, "pharmacoscopy.score.new_drug_space.csv"))
    pathway_space = pd.read_csv(os.path.join(output_dir, "pharmacoscopy.score.pathway_space.csv"), index_col=0)

    # Plots

    # evaluate "performance"
    # (similarity between measured and new drug sensitivities)
    s = pd.Series(new_drugs.columns).apply(lambda x: pd.Series(x.split("-")))
    s.index = new_drugs.columns
    s.columns = [['patient_id', 'timepoint_name']]
    new_drug_space = new_drugs.T.join(s)

    fig, axis = plt.subplots(
        6, 4,
        # len(sensitivity['timepoint'].drop_duplicates()),
        # len(sensitivity['patient_id'].drop_duplicates()),
        figsize=(11, 14),
        sharex=True, sharey=True
    )
    axis = axis.flatten()
    i = 0
    for patient_id in sensitivity['patient_id'].drop_duplicates().sort_values():
        for timepoint in sensitivity['timepoint_name'].drop_duplicates().sort_values(ascending=False):
            print(patient_id, timepoint)

            original_drug = sensitivity.loc[
                (
                    (sensitivity['patient_id'] == patient_id) &
                    (sensitivity['timepoint_name'] == timepoint)), ["drug", "score"]].groupby('drug').mean().squeeze()
            new_drug = new_drug_space.loc[
                (
                    (new_drug_space['patient_id'] == patient_id) &
                    (new_drug_space['timepoint_name'] == timepoint)), :].squeeze().drop(['patient_id', 'timepoint_name'])
            p = pd.DataFrame([original_drug, new_drug], index=['original', 'engineered']).T.dropna()
            axis[i].scatter(p['original'], p['engineered'], alpha=0.3)
            axis[i].set_title(" ".join([patient_id, timepoint]))
            i += 1
    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.measured_vs_predicted.scatter.svg"), bbox_inches='tight')

    # aggregated by patient
    # rank vs cross-patient sensitivity
    p = pathway_space.mean(1).sort_values()
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    fig, axis = plt.subplots(1)
    axis.scatter(p.rank(), p, alpha=0.8, color=plt.cm.inferno(norm(p)))
    for l in p.tail(10).index:
        axis.text(p.rank().ix[l], p.ix[l], l, fontsize=7.5, ha="right")
    for l in p.head(10).index:
        axis.text(p.rank().ix[l], p.ix[l], l, fontsize=7.5)
    axis.axhline(0, linestyle="--", color='black', alpha=0.8)
    axis.set_ylabel("Cross-patient pathway sensitivity")
    axis.set_xlabel("Pathway sensitivity rank")
    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.across_patients.rank.scatter.svg"), bbox_inches='tight')

    # vizualize in heatmaps
    fig = sns.clustermap(pathway_space.T.dropna(), figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.svg"), bbox_inches='tight')
    fig = sns.clustermap(pathway_space.T.dropna(), figsize=(20, 8), z_score=1)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.z_score.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.z_score.svg"), bbox_inches='tight')
    fig = sns.clustermap(new_drugs.T.dropna(), figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.svg"), bbox_inches='tight')
    fig = sns.clustermap(new_drugs.T.dropna(), figsize=(20, 8), z_score=0)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.z_score.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.z_score.svg"), bbox_inches='tight')

    # correlate samples in pathway space
    g = sns.clustermap(pathway_space.T.dropna().T.corr(), figsize=(8, 8), xticklabels=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.pearson.png"), dpi=300, bbox_inches='tight')

    # correlate pathways
    g = sns.clustermap(pathway_space.T.dropna().corr(), figsize=(20, 20), xticklabels=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.pathways.pearson.png"), dpi=300, bbox_inches='tight')

    #

    # Differential
    a = pathway_space[pathway_space.columns[pathway_space.columns.str.contains("after")]].mean(axis=1)
    b = pathway_space[pathway_space.columns[pathway_space.columns.str.contains("before")]].mean(axis=1)
    a.name = "after_Ibrutinib"
    b.name = "before_Ibrutinib"
    abs_diff = (a - b).sort_values()
    abs_diff.name = 'fold_change'

    # add mean
    changes = pd.DataFrame([a, b, abs_diff]).T
    changes["mean"] = changes[[a.name, b.name]].mean(axis=1)

    # add number of drugs per pathway
    n = analysis.drug_annotation.groupby("kegg_pathway_name")["drug"].nunique()
    n.name = "n_drugs"
    changes = changes.join(n)
    changes.to_csv(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"))
    changes = pd.read_csv(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"), index_col=0)

    # scatter
    fit = lowess(b, a, return_sorted=False)
    dist = abs(b - fit)
    normalizer = matplotlib.colors.Normalize(vmin=0, vmax=dist.max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(a, b, color=plt.cm.inferno(normalizer(dist)), alpha=0.8, s=4 + (3 ** z_score(changes["n_drugs"])))
    for l in (fit - b).sort_values().tail(15).index:
        axis.text(a.ix[l], b.ix[l], l, fontsize=7.5, ha="left")
    for l in (b - fit).sort_values().tail(15).index:
        axis.text(a.ix[l], b.ix[l], l, fontsize=7.5, ha="right")
    lims = [
        np.min([axis.get_xlim(), axis.get_ylim()]),  # min of both axes
        np.max([axis.get_xlim(), axis.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    axis.plot(lims, lims, '--', alpha=0.75, color="black", zorder=0)
    axis.set_aspect('equal')
    axis.set_xlim(lims)
    axis.set_ylim(lims)
    axis.set_xlabel("After ibrutinib")
    axis.set_ylabel("Before ibrutinib")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pharmacoscopy.sensitivity.pathway_space.differential.scatter.svg"), bbox_inches="tight")

    # maplot
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['mean'], changes['fold_change'], color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (3 ** z_score(changes["n_drugs"])))
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['mean'].ix[l], changes['fold_change'].ix[l], l, fontsize=7.5, ha="right")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['mean'].ix[l], changes['fold_change'].ix[l], l, fontsize=7.5, ha="right")
    axis.axhline(0, linestyle="--", color="black", alpha=0.5)
    axis.set_xlabel("Mean pathway accessibilty between timepoints")
    axis.set_ylabel("Change in pathway sensitivity (after / before)")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pharmacoscopy.sensitivity.pathway_space.differential.maplot.svg"), bbox_inches="tight")

    # rank vs cross-patient sensitivity
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['fold_change'].rank(method="dense"), changes['fold_change'], color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (3 ** z_score(changes["n_drugs"])))
    axis.axhline(0, color="black", alpha=0.8, linestyle="--")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['fold_change'].rank(method="dense").ix[l], changes['fold_change'].ix[l], l, fontsize=5, ha="left")
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['fold_change'].rank(method="dense").ix[l], changes['fold_change'].ix[l], l, fontsize=5, ha="right")
    axis.set_xlabel("Rank in change pathway accessibility")
    axis.set_ylabel("Change in pathway sensitivity (after / before)")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pharmacoscopy.sensitivity.pathway_space.differential.rank.svg"), bbox_inches="tight")

    # To confirm, plot a few specific examples
    # Proteasome
    of_interest = ["Bortezomib", "Carfilzomib"]

    d = sensitivity[
        (sensitivity["drug"].isin(of_interest)) &
        (sensitivity["patient_id"] != "CLL2")
    ]

    # difference of conditions
    f = d.groupby(['patient_id', 'drug']).apply(
        lambda x:
            x[x['timepoint_name'] == 'after_Ibrutinib']["score"].squeeze() -
            x[x['timepoint_name'] == 'before_Ibrutinib']["score"].squeeze())
    f.name = "diff"
    f = f.reset_index()
    d = pd.merge(d, f)

    # mean of conditions
    f = d.groupby(['patient_id', 'drug'])["score"].mean()
    f.name = "mean"
    f = f.reset_index()
    d = pd.merge(d, f)

    fig, axis = plt.subplots(len(of_interest), 2, figsize=(2 * 6, len(of_interest) * 4))
    d1 = d[d['drug'] == of_interest[0]]
    sns.stripplot(x="patient_id", y="score", data=d1, hue="timepoint_name", order=d1.sort_values("diff")['patient_id'].drop_duplicates(), ax=axis[0][0])
    sns.stripplot(x="patient_id", y="score", data=d1, hue="timepoint_name", order=d1.sort_values("mean")['patient_id'].drop_duplicates(), ax=axis[1][0])
    axis[0][0].set_title(of_interest[0])
    axis[1][0].set_title(of_interest[0])
    axis[0][0].axhline(0, color="black", linestyle="--", alpha=0.6)
    axis[1][0].axhline(0, color="black", linestyle="--", alpha=0.6)
    d2 = d[d['drug'] == of_interest[1]]
    sns.stripplot(x="patient_id", y="score", data=d2, hue="timepoint_name", order=d2.sort_values("diff")['patient_id'].drop_duplicates(), ax=axis[0][1])
    sns.stripplot(x="patient_id", y="score", data=d2, hue="timepoint_name", order=d2.sort_values("mean")['patient_id'].drop_duplicates(), ax=axis[1][1])
    axis[0][1].set_title(of_interest[1])
    axis[1][1].set_title(of_interest[1])
    axis[0][1].axhline(0, color="black", linestyle="--", alpha=0.6)
    axis[1][1].axhline(0, color="black", linestyle="--", alpha=0.6)
    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.proteasome_inhibitors.stripplot.svg"), bbox_inches='tight', dpi=300)

    fig, axis = plt.subplots(1, figsize=(4, 4))
    sns.stripplot(data=d[["patient_id", "diff", "drug"]].drop_duplicates(), x="drug", y="diff", ax=axis)
    sns.violinplot(data=d[["patient_id", "diff", "drug"]].drop_duplicates(), x="drug", y="diff", ax=axis, alpha=0.6)
    axis.axhline(0, color="black", linestyle="--", alpha=0.6)
    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.proteasome_inhibitors.diff_violin.svg"), bbox_inches='tight', dpi=300)

    #

    # Differential, patient-specific
    pathway_space2 = pathway_space.T
    # pathway_space2["patient"] = pd.Series(pathway_space2.index.str.split("-"), index=pathway_space2.index).apply(lambda x: x[0]).astype("category")
    pathway_space2["timepoint_name"] = pd.Series(pathway_space2.index.str.split("-"), index=pathway_space2.index).apply(lambda x: x[1]).astype("category")

    a = pathway_space2[pathway_space2['timepoint_name'] == "after_Ibrutinib"].drop("timepoint_name", axis=1).sort_index().reset_index(drop=True)
    b = pathway_space2[pathway_space2['timepoint_name'] == "before_Ibrutinib"].drop("timepoint_name", axis=1).sort_index().reset_index(drop=True)

    abs_diff = a - b
    abs_diff.index = pathway_space2.index.str.extract("(.*)-.*", expand=False).drop_duplicates()
    abs_diff.T.to_csv(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.differential.patient_specific.abs_diff.csv"))

    # vizualize in heatmaps
    g = sns.clustermap(abs_diff.dropna(), figsize=(20, 8))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.differential.abs_diff.svg"), bbox_inches='tight', dpi=300)

    # correlate patients in pathway space
    g = sns.clustermap(abs_diff.dropna().T.corr(), figsize=(8, 8), xticklabels=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.differential.abs_diff.patient_correlation.svg"), bbox_inches='tight', dpi=300)

    # correlate pathways across patients
    g = sns.clustermap(abs_diff.dropna().corr(), figsize=(20, 20), xticklabels=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.differential.abs_diff.pathway_correlation.svg"), bbox_inches='tight', dpi=300)


def atac_to_pathway(analysis, samples=None):
    """
    Quantify the activity of each pathway by the accessibility of the regulatory elements of its genes.
    """

    def z_score(x):
        return (x - x.mean()) / x.std()

    from bioservices.kegg import KEGG
    import scipy
    from scipy.stats import norm
    from statsmodels.sandbox.stats.multicomp import multipletests
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if samples is None:
        samples = analysis.samples

    # Query KEGG for genes member of pathways with drugs annotated
    k = KEGG()
    k.organism = "hsa"

    drug_annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.csv"))
    pathway_genes = dict()
    for pathway in drug_annot['kegg_pathway_id'].dropna().drop_duplicates():
        print(pathway)
        res = k.parse(k.get(pathway))
        if type(res) is dict and 'GENE' in res:
            print(len(res['GENE']))
            pathway_genes[res['PATHWAY_MAP'][pathway]] = list(set([x.split(";")[0] for x in res['GENE'].values()]))

    pickle.dump(pathway_genes, open(os.path.join("metadata", "pathway_gene_annotation_kegg.pickle"), "wb"))
    pathway_genes = pickle.load(open(os.path.join("metadata", "pathway_gene_annotation_kegg.pickle"), "rb"))

    # Get accessibility for each regulatory element assigned to each gene in each pathway
    # Reduce values per gene
    # Reduce values per pathway
    pc_samples = [s for s in samples if hasattr(s, "pharmacoscopy_id")]
    cov = analysis.accessibility[[s.name for s in pc_samples]]
    chrom_annot = analysis.coverage_annotated
    path_cov = pd.DataFrame()
    path_cov.columns.name = 'kegg_pathway_name'
    for pathway in pathway_genes.keys():
        print(pathway)
        # mean of all reg. elements of all genes
        index = chrom_annot.loc[(chrom_annot['gene_name'].isin(pathway_genes[pathway])), 'gene_name'].index
        q = cov.ix[index]
        path_cov[pathway] = (q[[s.name for s in pc_samples]].mean(axis=0) / cov[[s.name for s in pc_samples]].sum(axis=0)) * 1e6
    path_cov = path_cov.T.dropna().T

    path_cov_z = path_cov.apply(z_score, axis=0)
    path_cov.T.to_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.csv"))
    path_cov_z.T.to_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.z_score.csv"))
    path_cov = pd.read_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.csv"), index_col=0, header=range(24)).T
    path_cov_z = pd.read_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.z_score.csv"), index_col=0, header=range(24)).T

    # Get null distribution from permutations
    n_perm = 100
    pathway_sizes = pd.Series({k: len(v) for k, v in pathway_genes.items()}).sort_values()
    pathway_sizes.name = "pathway_size"
    all_genes = [j for i in pathway_genes.values() for j in i]

    r_fcs = list()
    for i in range(n_perm):
        r_path_cov = pd.DataFrame()
        r_path_cov.columns.name = 'kegg_pathway_name'
        for pathway in pathway_genes.keys():
            print(i, pathway)
            # choose same number of random genes from all pathways
            r = np.random.choice(all_genes, pathway_sizes.ix[pathway])
            index = chrom_annot.loc[(chrom_annot['gene_name'].isin(r)), 'gene_name'].index
            q = cov.ix[index]
            r_path_cov[pathway] = (q[[s.name for s in samples]].mean(axis=0) / cov[[s.name for s in samples]].sum(axis=0)) * 1e6

        r_path_cov = r_path_cov.T.dropna().T
        a = r_path_cov[r_path_cov.index.get_level_values("timepoint_name") == "after_Ibrutinib"].mean()
        b = r_path_cov[r_path_cov.index.get_level_values("timepoint_name") == "before_Ibrutinib"].mean()
        r_fcs.append(a - b)
    pickle.dump(r_fcs, open(os.path.join("metadata", "pathway.sample_accessibility.random.fold_changes.pickle"), "wb"))
    r_fcs = pickle.load(open(os.path.join("metadata", "pathway.sample_accessibility.random.fold_changes.pickle"), "rb"))

    [sns.distplot(x) for x in r_fcs]

    # Visualize
    # clustered
    g = sns.clustermap(path_cov_z, figsize=(30, 8), yticklabels=path_cov_z.index.get_level_values("sample_name"))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.svg"), bbox_inches="tight", dpi=300)

    # sorted
    p = path_cov[path_cov.sum(axis=0).sort_values().index]
    p = p.ix[p.sum(axis=1).sort_values().index]
    g = sns.clustermap(p, figsize=(30, 8), yticklabels=p.index.get_level_values("sample_name"), col_cluster=False, row_cluster=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.sorted.svg"), bbox_inches="tight", dpi=300)

    #

    # Differential
    # change
    a = path_cov[path_cov.index.get_level_values("timepoint_name") == "after_Ibrutinib"].mean()
    a.name = "after_Ibrutinib"
    b = path_cov[path_cov.index.get_level_values("timepoint_name") == "before_Ibrutinib"].mean()
    b.name = "before_Ibrutinib"
    fc = (a - b)
    fc.name = "fold_change"

    # p-values & q-values
    params = norm.fit([j for i in r_fcs for j in i])
    p_values = pd.Series(norm.sf(abs(fc), *params) * 2, index=fc.index, name="p_value")
    q_values = pd.Series(multipletests(p_values, method="bonferroni")[1], index=fc.index, name="q_value")
    # add number of genes per pathway
    changes = pd.DataFrame([pathway_sizes, a, b, fc, p_values, q_values]).T
    changes.index.name = "pathway"
    # mean of both timepoints
    changes["mean"] = changes[["after_Ibrutinib", "before_Ibrutinib"]].mean(axis=1)

    # save
    changes.sort_values("p_value").to_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.size-log2_fold_change-p_value.q_value.csv"))
    changes = pd.read_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.size-log2_fold_change-p_value.q_value.csv"), index_col=0)

    # scatter
    fit = lowess(b, a, return_sorted=False)
    dist = abs(b - fit)
    normalizer = matplotlib.colors.Normalize(vmin=0, vmax=dist.max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(a, b, color=plt.cm.inferno(normalizer(dist)), alpha=0.8, s=4 + (5 ** z_score(changes["mean"])))
    for l in (fit - b).sort_values().tail(15).index:
        axis.text(a.ix[l], b.ix[l], l, fontsize=7.5, ha="left")
    for l in (b - fit).sort_values().tail(15).index:
        axis.text(a.ix[l], b.ix[l], l, fontsize=7.5, ha="right")
    lims = [
        np.min([axis.get_xlim(), axis.get_ylim()]),  # min of both axes
        np.max([axis.get_xlim(), axis.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    axis.plot(lims, lims, '--', alpha=0.75, color="black", zorder=0)
    axis.set_aspect('equal')
    axis.set_xlim(lims)
    axis.set_ylim(lims)
    axis.set_xlabel("After ibrutinib")
    axis.set_ylabel("Before ibrutinib")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.scatter.svg"), bbox_inches="tight")

    # volcano
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())
    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['fold_change'], -np.log10(changes['q_value']), color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (5 ** z_score(changes["mean"])))
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['fold_change'].ix[l], -np.log10(changes['q_value']).ix[l], l, fontsize=7.5, ha="left")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['fold_change'].ix[l], -np.log10(changes['q_value']).ix[l], l, fontsize=7.5, ha="right")
    axis.set_xlabel("Change in pathway accessibilty (after / before)")
    axis.set_ylabel("-log10(p-value)")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.volcano.svg"), bbox_inches="tight")

    # maplot
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['mean'], changes['fold_change'], color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (5 ** z_score(changes["mean"])))
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['mean'].ix[l], changes['fold_change'].ix[l], l, fontsize=7.5, ha="left")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['mean'].ix[l], changes['fold_change'].ix[l], l, fontsize=7.5, ha="right")
    axis.axhline(0, linestyle="--", color="black", alpha=0.5)
    axis.set_xlabel("Mean pathway accessibilty between timepoints")
    axis.set_ylabel("Change in pathway accessibilty (after / before)")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.maplot.svg"), bbox_inches="tight")

    # rank vs cross-patient sensitivity
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['fold_change'].rank(method="dense"), changes['fold_change'], color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (5 ** z_score(changes["mean"])))
    axis.axhline(0, color="black", alpha=0.8, linestyle="--")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['fold_change'].rank(method="dense").ix[l], changes['fold_change'].ix[l], l, fontsize=5, ha="left")
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['fold_change'].rank(method="dense").ix[l], changes['fold_change'].ix[l], l, fontsize=5, ha="right")
    axis.set_xlabel("Rank in change pathway accessibility")
    axis.set_ylabel("Change in pathway accessibilty (after / before)")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.rank.svg"), bbox_inches="tight")

    # Compare with pharmacoscopy
    # annotate atac the same way as pharma
    #

    # Connect pharmacoscopy pathway-level sensitivities with ATAC-seq

    # Sample-level
    atac = path_cov
    pharma = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.score.pathway_space.csv"), index_col=0)
    pharma.loc["patient_id", :] = pd.Series(pharma.columns.str.split("-"), index=pharma.columns).apply(lambda x: x[0]).astype("category")
    pharma.loc["timepoint_name", :] = pd.Series(pharma.columns.str.split("-"), index=pharma.columns).apply(lambda x: x[1]).astype("category")
    pharma = pharma.T

    res = pd.DataFrame()
    # color = iter(cm.rainbow(np.linspace(0, 1, 50)))

    ps = pharma.index.str.extract("(CLL\d+)-.*").drop_duplicates().sort_values().drop('CLL2').drop('CLL8')
    ts = pharma.index.str.extract(".*-(.*)").drop_duplicates().sort_values(ascending=False)
    fig, axis = plt.subplots(len(ts), len(ps), figsize=(len(ps) * 4, len(ts) * 4), sharex=False, sharey=False)
    for i, patient_id in enumerate(ps):
        for j, timepoint in enumerate(ts):
            print(patient_id, timepoint)
            p = pharma[(pharma["patient_id"] == patient_id) & (pharma["timepoint_name"] == timepoint)].sort_index().T
            a = atac.loc[(atac.index.get_level_values("patient_id") == patient_id) & (atac.index.get_level_values("timepoint_name") == timepoint), :].T.sort_index()

            p = p.ix[a.index].dropna()
            a = a.ix[p.index].dropna()

            # stat, p_value = scipy.stats.pearsonr(p, a)
            axis[j][i].scatter(p, a, alpha=0.5)  # , color=color.next())
            axis[j][i].set_title(" ".join([patient_id, timepoint]))

            # res = res.append(pd.Series([patient_id, timepoint, stat, p_value]), ignore_index=True)
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pathway-pharmacoscopy.scatter.png"), bbox_inches='tight', dpi=300)

    # globally
    pharma_global = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"), index_col=0)
    # filter out drugs/pathways which after Ibrutinib are not more CLL-specific
    # pharma_global = pharma_global[pharma_global['after_Ibrutinib'] >= 0]

    pharma_change = pharma_global['fold_change']
    pharma_change.name = "pharmacoscopy"
    p = changes.join(pharma_change)

    # scatter
    n_to_label = 15
    combined_diff = (p['fold_change'] + p['pharmacoscopy']).dropna()
    normalizer = matplotlib.colors.Normalize(vmin=0, vmax=combined_diff.max())

    fig, axis = plt.subplots(1, 1, figsize=(4 * 1, 4 * 1))
    axis.scatter(p['fold_change'], p['pharmacoscopy'], alpha=0.8, color=plt.get_cmap("Blues")(normalizer(abs(combined_diff))))
    axis.axhline(0, color="black", linestyle="--", alpha=0.5)
    axis.axvline(0, color="black", linestyle="--", alpha=0.5)
    # annotate top pathways
    for path in combined_diff.sort_values().head(n_to_label).index:
        axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="right")
    for path in combined_diff.sort_values().tail(n_to_label).index:
        axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="left")
    axis.set_xlabel("ATAC-seq change in pathway accessibility", ha="center")
    axis.set_ylabel("Pharmacoscopy change in pathway sensitivity", ha="center")
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "atac-pharmacoscopy.across_patients.scatter.svg"), bbox_inches='tight', dpi=300)
    # fig.savefig(os.path.join(analysis.results_dir, "atac-pharmacoscopy.across_patients.scatter.only_positive.svg"), bbox_inches='tight', dpi=300)

    # individually
    pharma_patient = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.patient_specific.abs_diff.csv"), index_col=0)
    a = path_cov[path_cov.index.get_level_values("timepoint_name") == "after_Ibrutinib"].reset_index(drop=True)  # .reset_index(level=range(1, len(path_cov.index.levels)))
    b = path_cov[path_cov.index.get_level_values("timepoint_name") == "before_Ibrutinib"].reset_index(drop=True)  # .reset_index(level=range(1, len(path_cov.index.levels)))
    atac_patient = (a - b).T
    atac_patient.columns = pd.Series(path_cov.index.get_level_values("patient_id").drop_duplicates())

    pharma_patient = pharma_patient.T.ix[atac_patient.columns].T
    atac_patient = atac_patient[pharma_patient.columns]

    fig, axis = plt.subplots(3, 4, figsize=(4 * 4, 3 * 4), sharex=True, sharey=True)
    axis = axis.flatten()
    for i, patient_id in enumerate(atac_patient.columns):
        print(patient_id)

        p = pharma_patient[[patient_id]]
        p.columns = ["Pharmacoscopy"]
        a = atac_patient[patient_id]
        a.name = "ATAC-seq"

        o = p.join(a).dropna().apply(z_score, axis=0)

        # stat, p_value = scipy.stats.pearsonr(p, a)
        axis[i].scatter(o["Pharmacoscopy"], o["ATAC-seq"], alpha=0.5)  # , color=color.next())
        axis[i].set_title(" ".join([patient_id]))
        axis[i].axhline(0, linestyle="--", color="black", linewidth=1)
        axis[i].axvline(0, linestyle="--", color="black", linewidth=1)
    sns.despine(fig)
    fig.savefig(os.path.join(analysis.results_dir, "pathway-pharmacoscopy.patient_specific.scatter.png"), bbox_inches='tight', dpi=300)

    # Dissect specific pathways
    n_paths = 5
    paths_of_interest = [
        "Proteasome", "N-Glycan biosynthesis", "Terpenoid backbone biosynthesis", "Legionellosis"]
    paths_of_interest = combined_diff.sort_values().tail(n_to_label).index
    # path_control = ["Regulation of lipolysis in adipocytes",
    #                 "Dilated cardiomyopathy",
    #                 "PPAR signaling pathway",
    #                 "Salivary secretion",
    #                 "Protein processing in endoplasmic reticulum",
    #                 "Rap1 signaling pathway",
    #                 "Adrenergic signaling in cardiomyocytes"]
    path_control = combined_diff[(combined_diff > -.01) & (combined_diff < .01)]

    # pharmacoscopy sensitivity

    # Make drug - sample pivot table
    sensitivity['id'] = sensitivity['patient_id'] + " - " + sensitivity['timepoint_name']
    sens_pivot = pd.pivot_table(data=sensitivity, index="drug", columns=['id', 'patient_id', 'sample_id', 'pharmacoscopy_id', 'timepoint_name'], values="score")
    # transform into time differentials
    sens_diff = pd.DataFrame()
    for patient in sens_pivot.columns.get_level_values("patient_id"):
        try:
            a = sens_pivot[sens_pivot.columns[
                (sens_pivot.columns.get_level_values("patient_id") == patient) &
                (sens_pivot.columns.get_level_values("timepoint_name") == "after_Ibrutinib")]].squeeze()
            b = sens_pivot[sens_pivot.columns[
                (sens_pivot.columns.get_level_values("patient_id") == patient) &
                (sens_pivot.columns.get_level_values("timepoint_name") == "before_Ibrutinib")]].squeeze()
        except:
            continue
        sens_diff[patient] = a - b

    drugs = analysis.drug_annotation[analysis.drug_annotation["kegg_pathway_name"].isin(paths_of_interest)]['drug'].drop_duplicates()
    ctrl_drugs = analysis.drug_annotation[analysis.drug_annotation["kegg_pathway_name"].isin(path_control)]['drug'].drop_duplicates()
    ctrl_drugs = ctrl_drugs[~ctrl_drugs.isin(drugs)]

    g = sns.clustermap(
        sens_pivot.ix[drugs.tolist() + ctrl_drugs.tolist()], metric="correlation", row_cluster=False, col_cluster=True) # , z_score=1
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    g = sns.clustermap(
        p[['before_Ibrutinib', 'after_Ibrutinib']].ix[paths_of_interest + path_control],
        row_cluster=False, col_cluster=False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    g = sns.clustermap(
        sens_diff,
        row_cluster=True, col_cluster=True)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    g = sns.clustermap(
        sens_diff.ix[drugs.tolist() + ctrl_drugs.tolist()],
        metric="correlation", row_cluster=False, col_cluster=True)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    #

    # Similarly, for ATAC
    pathway_genes = pickle.load(open(os.path.join("metadata", "pathway_gene_annotation_kegg.pickle"), "rb"))
    paths_of_interest = ["Proteasome"]
    index = chrom_annot.loc[(chrom_annot['gene_name'].isin([y for p in paths_of_interest for y in pathway_genes[p]])), 'gene_name'].index
    path_cov = (cov.ix[index][[s.name for s in pc_samples]] / cov[[s.name for s in pc_samples]].sum(axis=0)) * 1e6

    cov_diff = pd.DataFrame()
    for patient in path_cov.columns.get_level_values("patient_id"):
        try:
            a = path_cov.T[
                (path_cov.columns.get_level_values("patient_id") == patient) &
                (path_cov.columns.get_level_values("timepoint_name") == "after_Ibrutinib")].T.squeeze()
            b = path_cov.T[
                (path_cov.columns.get_level_values("patient_id") == patient) &
                (path_cov.columns.get_level_values("timepoint_name") == "before_Ibrutinib")].T.squeeze()
        except:
            continue
        cov_diff[patient] = a - b

    path_cov.columns = path_cov.columns.get_level_values("sample_name")

    g = sns.clustermap(
        path_cov,
        metric="correlation", row_cluster=True, col_cluster=True, z_score=0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    g = sns.clustermap(
        cov_diff,
        metric="correlation", row_cluster=True, col_cluster=True)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)


def make_network(analysis):
    """
    Make a network with various types of interactions:
    DRUG -> GENE
    GENE -> PATHWAY
    DRUG -> PATHWAY

    the scores reflect the number and direction of interactions
    """
    import networkx as nx

    # global drug changes
    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    sensitivity['p_id'] = sensitivity['patient_id'].astype(str) + " " + sensitivity['timepoint_name'].astype(str)
    sensitivity_pivot = pd.pivot_table(sensitivity, index="drug", columns="p_id", values="score")
    a = sensitivity_pivot[sensitivity_pivot.columns[sensitivity_pivot.columns.str.contains("after")]].mean(axis=1)
    b = sensitivity_pivot[sensitivity_pivot.columns[sensitivity_pivot.columns.str.contains("before")]].mean(axis=1)
    drug_changes = a - b

    # global pathway changes (with cross-patient changes)
    changes = pd.read_csv(os.path.join("results", "pathway.sample_accessibility.size-log2_fold_change-p_value.q_value.csv"), index_col=0)
    pharma_global = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"), index_col=0)
    pharma_change = pharma_global['fold_change']
    pharma_change.name = "pharmacoscopy"
    p = changes.join(pharma_change).dropna()

    # normalize changes
    drug_changes = (drug_changes - drug_changes.mean()) / drug_changes.std()
    p['fold_change'] = (p['fold_change'] - p['fold_change'].mean()) / p['fold_change'].std()
    p['pharmacoscopy'] = (p['pharmacoscopy'] - p['pharmacoscopy'].mean()) / p['pharmacoscopy'].std()
    # cap at +-3
    drug_changes.loc[drug_changes < -3] = -3
    drug_changes.loc[drug_changes > 3] = 3
    p.loc[p['fold_change'] < -3, 'fold_change'] = -3
    p.loc[p['fold_change'] > 3, 'fold_change'] = 3
    p.loc[p['pharmacoscopy'] < -3, 'pharmacoscopy'] = -3
    p.loc[p['pharmacoscopy'] > 3, 'pharmacoscopy'] = 3

    # DRUG -> PATHWAY
    csv = pd.read_csv(os.path.join("data", "pharmacoscopy.drug-pathway_interactions.tsv"), sep='\t')
    csv = csv[csv["kegg_pathway_name"].isin(p.index.tolist())]
    drugs = csv['drug'].unique()
    G2 = nx.from_pandas_dataframe(csv, source='drug', target='kegg_pathway_name', edge_attr=['interaction_types', "interaction_score"])
    # G2 = G2.to_directed()
    nx.set_node_attributes(G2, "entity_type", {x: "drug" if x in drugs else "pathway" for x in G2.nodes()})
    nx.set_node_attributes(G2, "pathway_size", {x: float(p.loc[p.index == x, "pathway_size"].squeeze()) for x in G2.nodes() if x not in drugs})
    nx.set_node_attributes(G2, "atacseq_change", {x: float(p.loc[p.index == x, "fold_change"].squeeze()) for x in G2.nodes() if x not in drugs})
    nx.set_node_attributes(G2, "pharmacoscopy_change", {x: float(p.loc[p.index == x, "pharmacoscopy"].squeeze()) if x not in drugs else float(drug_changes[x]) for x in G2.nodes()})

    nx.write_gexf(G2, os.path.join("data", "pharmacoscopy.drug-pathway_interactions.attributes_changes.gexf"))
    nx.write_graphml(G2, os.path.join("data", "pharmacoscopy.drug-pathway_interactions.attributes_changes.graphml"))

    # Individual accessibility/sensitivity levels per patient
    # atac
    path_atac = pd.read_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.csv"), index_col=0, header=range(24))
    a = path_atac[path_atac.columns[path_atac.columns.get_level_values("timepoint_name") == "after_Ibrutinib"]]
    b = path_atac[path_atac.columns[path_atac.columns.get_level_values("timepoint_name") == "before_Ibrutinib"]]
    a.columns = a.columns.get_level_values("patient_id")
    b.columns = b.columns.get_level_values("patient_id")
    path_atac = (a - b)

    # pharmacoscopy (pathway-level)
    # path_pharma = pd.read_csv(os.path.join(analysis.results_dir, "pharmacoscopy", "pharmacoscopy.score.pathway_space.csv"), index_col=0)
    path_pharma = pd.read_csv(os.path.join(analysis.results_dir, "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.patient_specific.abs_diff.csv"), index_col=0).drop("CLL2", axis=1)
    # match patients
    path_atac = path_atac[path_atac.columns[path_atac.columns.isin(path_pharma.columns)]]
    path_pharma = path_pharma.ix[path_atac.index].dropna()
    path_atac = path_atac.ix[path_pharma.index].dropna()

    # Change values across patients
    # atac
    changes_atac = pd.read_csv(os.path.join(analysis.results_dir, "pathway.sample_accessibility.size-log2_fold_change-p_value.q_value.csv"), index_col=0)
    # pharmacoscopy
    changes_parma = path_pharma.mean(1)

    # DRUG -> GENE
    csv = pd.read_csv(os.path.join("data", "pharmacoscopy.drug-gene_interactions.tsv"), sep='\t')
    d = csv['drug'].unique()
    G1 = nx.from_pandas_dataframe(csv, source='drug', target='entrez_gene_symbol', edge_attr=['interaction_types', "interaction_score"])
    nx.set_node_attributes(G1, "entity_type", {x: "drug" if x in d else "gene" for x in G1.nodes()})
    G1 = G1.to_directed()
    nx.write_gexf(G1, os.path.join("data", "pharmacoscopy.drug-gene_interactions.gexf"))

    # DRUG -> PATHWAY
    csv = pd.read_csv(os.path.join("data", "pharmacoscopy.drug-pathway_interactions.tsv"), sep='\t')
    csv = csv[csv["kegg_pathway_name"].isin(path_atac.index.tolist())]
    d = csv['drug'].unique()
    p = csv['kegg_pathway_name'].unique()
    G2 = nx.from_pandas_dataframe(csv, source='drug', target='kegg_pathway_name', edge_attr=['interaction_types', "interaction_score"])
    G2 = G2.to_directed()
    nx.set_node_attributes(G2, "entity_type", {x: "drug" if x in d else "pathway" for x in G2.nodes()})
    nx.write_gexf(G2, os.path.join("data", "pharmacoscopy.drug-pathway_interactions.gexf"))
    nx.write_graphml(G2, os.path.join("data", "pharmacoscopy.drug-pathway_interactions.graphml"))

    # GENE -> PATHWAY
    # (pharmacoscopy based)
    gene_net = analysis.drug_annotation[['entrez_gene_symbol', 'kegg_pathway_name']].drop_duplicates().dropna()
    gene_net["interaction_score"] = 1
    gene_net["interaction_types"] = "gene-pathway"
    g = gene_net['entrez_gene_symbol'].unique()
    G3 = nx.from_pandas_dataframe(gene_net, source='entrez_gene_symbol', target='kegg_pathway_name', edge_attr=['interaction_types', 'interaction_score'])
    nx.set_node_attributes(G3, "entity_type", {x: "gene" if x in g else "pathway" for x in G3.nodes()})
    G3 = G3.to_directed()
    nx.write_gexf(G3, os.path.join("data", "pharmacoscopy.gene-pathway_interactions.gexf"))

    F = nx.compose(G1, G2)
    F = nx.compose(F, G3)

    # add patient-specific
    for patient_id in path_pharma.columns:
        nx.set_node_attributes(F, "pharma {}".format(patient_id), {str(y): float(path_pharma[patient_id].ix[y]) if y in p else 0.0 for y in F.nodes()})
    for patient_id in path_atac.columns:
        nx.set_node_attributes(F, "atac {}".format(patient_id), {str(y): float(path_atac[patient_id].ix[y]) if y in p else 0.0 for y in F.nodes()})
    # add global
    nx.set_node_attributes(F, "pharma change", {str(y): float(changes_parma.ix[y]) if y in p else 0.0 for y in F.nodes()})
    for attribute in changes_atac.columns:
        nx.set_node_attributes(F, "atac {}".format(attribute), {str(y): float(changes_atac[attribute].ix[y]) if y in p else 0.0 for y in F.nodes()})
    nx.write_gexf(F, os.path.join("data", "pharmacoscopy.all_interactions.gexf"))


def DESeq_analysis(counts_matrix, experiment_matrix, variable, covariates, output_prefix, alpha=0.05):
    """
    """
    import rpy2.robjects as robj
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    run = robj.r("""
        run = function(countData, colData, variable, covariates, output_prefix, alpha) {
            library(DESeq2)

            # alpha = 0.05
            # output_prefix = "results/cll-ibrutinib_AKH.ibrutinib_treatment/cll-ibrutinib_AKH.ibrutinib_treatment"
            # countData = read.csv("results/cll-ibrutinib_AKH.ibrutinib_treatment/counts_matrix.csv", sep=",", row.names=1)
            # colData = read.csv("results/cll-ibrutinib_AKH.ibrutinib_treatment/experiment_matrix.csv", sep=",")

            # variable = "timepoint_name"
            # covariates = "atac_seq_batch +"

            design = as.formula((paste("~", covariates, variable)))
            print(design)
            dds <- DESeqDataSetFromMatrix(
                countData = countData, colData = colData,
                design)

            dds <- DESeq(dds, parallel=TRUE)
            save(dds, file=paste0(output_prefix, ".deseq_dds_object.Rdata"))
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
            rownames(group_means) = rownames(countData)
            write.table(group_means, paste0(output_prefix, ".", variable, ".group_means.csv"), sep=",")

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
                output_name = paste0(output_prefix, ".", variable, ".", comparison_name, ".csv")
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

    # result_files = ["results/cll-ibrutinib_AKH.ibrutinib_treatment/cll-ibrutinib_AKH.ibrutinib_treatment.timepoint_name.after_Ibrutinib-before_Ibrutinib.csv"]
    # concatenate all files
    results = pd.DataFrame()
    for result_file in result_files:
        df = pd.read_csv(result_file)
        df.index = counts_matrix.index
        df.to_csv(result_file, index=True)  # fix R header in file

        results = results.append(df)

    # save all
    results.to_csv(os.path.join(output_prefix + ".%s.csv" % variable), index=True)

    # return
    return results


def DESeq_interaction(counts_matrix, experiment_matrix, formula, output_prefix, alpha=0.05):
    """
    """
    import rpy2.robjects as robj
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    run = robj.r("""
        run = function(countData, colData, formula, output_prefix, alpha) {
            library(DESeq2)

            design = as.formula(formula)

            dds <- DESeqDataSetFromMatrix(
                countData = countData, colData = colData,
                design)
            dds <- DESeq(dds, parallel=TRUE)
            save(dds, file=paste0(output_prefix, ".interaction.deseq_dds_object.Rdata"))

            # Save DESeq-normalized counts
            normalized_counts = counts(dds, normalized=TRUE)
            colnames(normalized_counts) = colData$sample_name
            output_name = paste0(output_prefix, ".interaction.normalized_counts.csv")
            rownames(normalized_counts) = rownames(countData)
            write.table(normalized_counts, output_name, sep=",")

            # keep track of output files
            result_files = list()

            # Get timepoint-specific regions for each patient specifically
            for (patient in unique(colData$patient_id)[2:length(unique(colData$patient_id))]){
                print(patient)
                cond1 = paste0("patient_id", patient, ".timepoint_nameafter.Ibrutinib")
                cond2 = paste0("patient_id", patient, ".timepoint_namebefore.Ibrutinib")
                contrast = list(cond1, cond2)
                res <- results(dds, contrast=contrast, alpha=alpha, independentFiltering=FALSE, parallel=TRUE)
                res <- as.data.frame(res)
                res["patient"] = patient
                output_name = paste0(output_prefix, ".interaction.", patient, ".csv")
                write.table(res, output_name, sep=",")
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

    result_files = run(counts_matrix, experiment_matrix, formula, output_prefix, alpha)

    # concatenate all files
    results = pd.DataFrame()
    for result_file in result_files:
        df = pd.read_csv("results/interaction/" + result_file[17:])
        df['patient'] = result_file[29:][:-4]
        # df.index = counts_matrix.index  # not actually needed, as it reads from file
        results = results.append(df)

    # save all
    results.to_csv(os.path.join(output_prefix + ".all_patients.csv"), index=True)

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
    #         sbatch -J LOLA_${F} -o ${F/_regions.bed/}_lola.log ~/run_LOLA.sh $F ~/projects/cll-ibrutinib/results/cll-ibrutinib_peaks.bed hg19 `dirname $F`
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
    #         sbatch -J MEME-AME_${F} -o ${F/fa/}ame.log ~/run_AME.sh $F human
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
            attr = "\n".join(list(set(dataframe["gene_name"].dropna().tolist())))
        elif kind == "genes+score":
            d = dataframe[['gene_name', 'score']].astype(str)
            attr = "\n".join((d["gene_name"] + "," + d["score"]).tolist())
        elif kind == "regions":
            # Build payload with bed file
            attr = "\n".join(list(set(dataframe[['chrom', 'start', 'end']].apply(lambda x: "\t".join([str(i) for i in x]), axis=1).tolist())))

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
        if res.shape[0] == 0:
            continue
        if len(res.columns) == 7:
            res.columns = ["rank", "description", "p_value", "z_score", "combined_score", "genes", "adjusted_p_value"]
        elif len(res.columns) == 9:
            res.columns = ["rank", "description", "p_value", "z_score", "combined_score", "genes", "adjusted_p_value", "old_p_value", "old_adjusted_p_value"]

        # Remember gene set library used
        res["gene_set_library"] = gene_set_library

        # Append to master dataframe
        results = results.append(res, ignore_index=True)

    return results

    # for F in `find . -iname *_genes.symbols.txt`
    # do
    #     if  [ ! -f ${F/symbols.txt/enrichr.csv} ]; then
    #         echo $F
    #         sbatch -J ENRICHR_${F} -o ${F/symbols.txt/}enrichr.log ~/run_Enrichr.sh $F
    #     fi
    # done


def characterize_regions_structure(df, prefix, output_dir, universe_df=None):
    # use all sites as universe
    if universe_df is None:
        universe_df = pd.read_csv(os.path.join("results", analysis.name + "_peaks.coverage_qnorm.log2.annotated.csv"), index_col=0)

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


def characterize_regions_function(df, output_dir, prefix, results_dir="results", universe_file=None):
    def scale_score(x):
        return (x - x.min()) / (x.max() - x.min())

    # use all sites as universe
    if universe_file is None:
        universe_file = os.path.join(analysis.results_dir, analysis.name + "_peak_set.bed")

    # make output dirs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save to bed
    bed_file = os.path.join(output_dir, "%s_regions.bed" % prefix)
    df[['chrom', 'start', 'end']].to_csv(bed_file, sep="\t", header=False, index=False)
    # save as tsv
    tsv_file = os.path.join(output_dir, "%s_regions.tsv" % prefix)
    df[['chrom', 'start', 'end']].reset_index().to_csv(tsv_file, sep="\t", header=False, index=False)

    # export gene symbols
    df['gene_name'].str.split(",").apply(pd.Series, 1).stack().drop_duplicates().to_csv(os.path.join(output_dir, "%s_genes.symbols.txt" % prefix), index=False)

    # export gene symbols with scaled absolute fold change
    if "log2FoldChange" in df.columns:
        df["score"] = scale_score(abs(df["log2FoldChange"]))
        df["abs_fc"] = abs(df["log2FoldChange"])

        d = df[['gene_name', 'score']].sort_values('score', ascending=False)

        # split gene names from score if a reg.element was assigned to more than one gene
        a = (
            d['gene_name']
            .str.split(",")
            .apply(pd.Series, 1)
            .stack()
        )
        a.index = a.index.droplevel(1)
        a.name = 'gene_name'
        d = d[['score']].join(a)
        # reduce various ranks to mean per gene
        d = d.groupby('gene_name').mean().reset_index()
        d.to_csv(os.path.join(output_dir, "%s_genes.symbols.score.csv" % prefix), index=False)

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


def add_args(parser):
    """
    Options for project and pipelines.
    """
    # Behaviour
    parser.add_argument("-g", "--generate", dest="generate", action="store_true",
                        help="Should we generate data and plots? Default=False")

    return parser


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    #
    # First an analysis of the Ibrutinib samples in light of the previous CLL and other hematopoietic cells
    analysis = Analysis(name="cll-ibrutinib_all_samples", from_pickle=args.generate)
    subprojects = ["cll-chromatin", "stanford_atacseq", "bcells"]

    # Start project
    prj = Project("metadata/project_config.yaml")
    prj.add_sample_sheet()
    for sample in prj.samples:
        sample.subproject = analysis.name

    # add samples from subprojects
    for subproject in subprojects:
        sp = Project("metadata/project_config.yaml", subproject=subproject)
        sp.add_sample_sheet()
        for sample in sp.samples:
            sample.subproject = subproject
        prj.samples += sp.samples

    # temporary:
    for sample in prj.samples:
        sample.summits = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_summits.bed")

    # work only with ATAC-seq samples
    prj.samples = [sample for sample in prj.samples if sample.library == "ATAC-seq" and sample.cell_type not in ["AML", "pHSC", "LSC"]]
    prj.samples = [s for s in prj.samples if os.path.exists(s.filtered) and s.pass_qc == 1]

    # pair analysis and Project
    analysis.prj = prj
    analysis.samples = prj.samples

    # GET CONSENSUS PEAK SET, ANNOTATE IT, PLOT
    # Get consensus peak set from all samples
    analysis.get_consensus_sites(analysis.samples, "summits")
    analysis.calculate_peak_support(analysis.samples, "summits")

    # GET CHROMATIN OPENNESS MEASUREMENTS, PLOT
    # Get coverage values for each peak in each sample of ATAC-seq and ChIPmentation
    analysis.measure_coverage(analysis.samples)
    # normalize coverage values
    analysis.normalize_coverage_quantiles(analysis.samples)
    analysis.get_peak_gccontent_length()
    analysis.normalize_gc_content(analysis.samples)

    # Annotate peaks with closest gene
    analysis.get_peak_gene_annotation()
    # Annotate peaks with genomic regions
    analysis.get_peak_genomic_location()
    # Annotate peaks with ChromHMM state from CD19+ cells
    analysis.get_peak_chromatin_state()
    # Annotate peaks with closest gene, chromatin state,
    # genomic location, mean and variance measurements across samples
    analysis.annotate(analysis.samples)
    analysis.annotate_with_sample_metadata()

    # Plots
    # plot general peak set features
    analysis.plot_peak_characteristics()
    # Plot coverage features across peaks/samples
    analysis.plot_coverage()
    analysis.plot_variance()

    # Unsupervised analysis
    analysis.unsupervised()

    #

    #
    # Second, an analysis of the Ibrutinib samples in light of the previous CLL only
    # Start analysis object
    analysis = Analysis(name="cll-ibrutinib_CLL", from_pickle=args.generate)
    subprojects = ["cll-chromatin"]

    # Start project
    prj = Project("metadata/project_config.yaml")
    prj.add_sample_sheet()
    for sample in prj.samples:
        sample.subproject = analysis.name

    # add samples from subprojects
    for subproject in subprojects:
        sp = Project("metadata/project_config.yaml", subproject=subproject)
        sp.add_sample_sheet()
        for sample in sp.samples:
            sample.subproject = subproject
        prj.samples += sp.samples

    # work only with ATAC-seq samples
    prj.samples = [sample for sample in prj.samples if sample.library == "ATAC-seq" and sample.cell_type not in ["AML", "pHSC", "LSC"]]
    prj.samples = [s for s in prj.samples if os.path.exists(s.filtered) and s.pass_qc == 1]

    # pair analysis and Project
    analysis.prj = prj
    analysis.samples = prj.samples

    # GET CONSENSUS PEAK SET, ANNOTATE IT, PLOT
    # Get consensus peak set from all samples
    analysis.get_consensus_sites(analysis.samples, "summits")
    analysis.calculate_peak_support(analysis.samples)

    # GET CHROMATIN OPENNESS MEASUREMENTS, PLOT
    # Get coverage values for each peak in each sample of ATAC-seq and ChIPmentation
    analysis.measure_coverage(analysis.samples)
    # normalize coverage values
    analysis.normalize_coverage_quantiles(analysis.samples)
    analysis.get_peak_gccontent_length()
    analysis.normalize_gc_content(analysis.samples)

    # Annotate peaks with closest gene
    analysis.get_peak_gene_annotation()
    # Annotate peaks with genomic regions
    analysis.get_peak_genomic_location()
    # Annotate peaks with ChromHMM state from CD19+ cells
    analysis.get_peak_chromatin_state()
    # Annotate peaks with closest gene, chromatin state,
    # genomic location, mean and variance measurements across samples
    analysis.annotate(analysis.samples)
    analysis.annotate_with_sample_metadata()

    # Plots
    # plot general peak set features
    analysis.plot_peak_characteristics()
    # Plot coverage features across peaks/samples
    analysis.plot_coverage()
    analysis.plot_variance()

    # Unsupervised analysis
    analysis.unsupervised(analysis.samples)

    #

    # Now let's analyse only the CLL samples from AKH (ibrutinib cohort)

    CLL_analysis = Analysis(name="cll-ibrutinib_CLL", from_pickle=True)
    analysis = Analysis(name="cll-ibrutinib_AKH")

    CLL_analysis.prj.samples = [s for s in CLL_analysis.prj.samples if s.clinical_centre == "vienna"]
    analysis.prj = CLL_analysis.prj
    analysis.samples = CLL_analysis.prj.samples

    attrs = ["sites", "coverage", "coverage_annotated", "accessibility"]
    for attr in attrs:
        setattr(analysis, attr, getattr(CLL_analysis, attr))
    analysis.accessibility = analysis.accessibility.iloc[:, analysis.accessibility.columns.get_level_values('clinical_centre') == "vienna"]
    analysis.to_pickle()

    # Unsupervised analysis
    analysis.unsupervised(
        analysis.samples,
        attributes=[
            "patient_id", "timepoint_name", "patient_gender", "patient_age_at_collection",
            "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
            "del11q", "del13q", "del17p", "tri12", "p53",
            "time_since_treatment", "treatment_response"])

    # Supervised analysis
    # Differential analysis and exploration of differential regions
    analysis.differential_region_analysis(
        samples=analysis.samples,
        trait="timepoint_name",
        # variables=["atac_seq_batch", "patient_gender", "ighv_mutation_status", "timepoint_name"],
        variables=["atac_seq_batch", "timepoint_name"],
        output_suffix="{}.ibrutinib_treatment".format(analysis.name)
    )
    analysis.investigate_differential_regions(
        trait="timepoint_name",
        output_suffix="ibrutinib_treatment"
    )

    # Differential analysis for the interaction between patient and treatment
    # (patient-specific changes to treatment) and exploration of differential regions
    analysis.interaction_differential_analysis(
        samples=analysis.samples,
        formula="~patient_id * timepoint_name",
        output_suffix="interaction"
    )
    analysis.investigate_interaction_regions(
        output_suffix="interaction"
    )

    # analysis.sites = pybedtools.BedTool(os.path.join(analysis.results_dir, analysis.name + "_peak_set.bed"))
    # analysis.support = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.support.csv"))
    # analysis.coverage = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.raw_coverage.csv"), index_col=0)
    # analysis.gene_annotation = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.gene_annotation.csv"))
    # analysis.closest_tss_distances = pickle.load(open(os.path.join(analysis.results_dir, analysis.name + "_peaks.closest_tss_distances.pickle"), "rb"))
    # analysis.region_annotation = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.region_annotation.csv"))
    # analysis.region_annotation_b = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.region_annotation_background.csv"))
    # analysis.chrom_state_annotation = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.chromatin_state.csv"))
    # analysis.chrom_state_annotation_b = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.chromatin_state_background.csv"))

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Program canceled by user!")
        sys.exit(1)
