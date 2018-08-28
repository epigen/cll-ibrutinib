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

    def plot_peak_characteristics(self, samples=None):
        def get_sample_reads(bam_file):
            return pysam.AlignmentFile(bam_file).count()

        def get_peak_number(bed_file):
            return len(open(bed_file, "r").read().split("\n"))

        def get_total_open_chromatin(bed_file):
            peaks = pd.read_csv(bed_file, sep="\t", header=None)
            return (peaks.iloc[:, 2] - peaks.iloc[:, 1]).sum()

        def get_peak_lengths(bed_file):
            peaks = pd.read_csv(bed_file, sep="\t", header=None)
            return (peaks.iloc[:, 2] - peaks.iloc[:, 1])

        def get_peak_chroms(bed_file):
            peaks = pd.read_csv(bed_file, sep="\t", header=None)
            return peaks.iloc[:, 0].value_counts()

        # Peaks per sample:
        if samples is None:
            samples = self.samples
        stats = pd.DataFrame([
            map(get_sample_reads, [s.filtered for s in samples]),
            map(get_peak_number, [s.peaks for s in samples]),
            map(get_total_open_chromatin, [s.peaks for s in samples])],
            index=["reads_used", "peak_number", "open_chromatin"],
            columns=[s.name for s in samples]).T

        stats["peaks_norm"] = (stats["peak_number"] / stats["reads_used"]) * 1e3
        stats["open_chromatin_norm"] = (stats["open_chromatin"] / stats["reads_used"])
        stats.to_csv(os.path.join(self.results_dir, "{}.open_chromatin_space.csv".format(self.name)), index=True)

        stats = pd.read_csv(os.path.join(self.results_dir, "{}.open_chromatin_space.csv".format(self.name)), index_col=0)
        stats['patient_id'] = [s.patient_id for s in samples]
        stats['timepoint'] = [s.timepoint_name for s in samples]
        # median lengths per sample (split-apply-combine)
        stats = pd.merge(stats.reset_index(), stats.groupby('timepoint')['open_chromatin'].median().to_frame(name='group_open_chromatin').reset_index())
        stats = pd.merge(stats, stats.groupby('timepoint')['open_chromatin_norm'].median().to_frame(name='group_open_chromatin_norm').reset_index())

        # plot
        fig, axis = plt.subplots(2, 1, figsize=(4 * 2, 6 * 1))
        stats = stats.sort_values("open_chromatin")
        sns.barplot(x="index", y="open_chromatin", data=stats.reset_index(), palette="summer", ax=axis[0])
        stats = stats.sort_values("open_chromatin_norm")
        sns.barplot(x="index", y="open_chromatin_norm", data=stats.reset_index(), palette="summer", ax=axis[1])
        axis[0].set_ylabel("Total open chromatin space (bp)")
        axis[1].set_ylabel("Total open chromatin space (normalized)")
        axis[0].set_xticklabels(axis[0].get_xticklabels(), rotation=45, ha="right")
        axis[1].set_xticklabels(axis[1].get_xticklabels(), rotation=45, ha="right")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.total_open_chromatin_space.per_sample.svg".format(self.name)), bbox_inches="tight")

        # per group
        fig, axis = plt.subplots(2, 1, figsize=(4 * 2, 6 * 1))
        stats = stats.sort_values("group_open_chromatin")
        sns.barplot(x="timepoint", y="open_chromatin", data=stats.reset_index(), palette="summer", ax=axis[0])
        sns.stripplot(x="timepoint", y="open_chromatin", data=stats.reset_index(), palette="summer", ax=axis[0])
        sns.factorplot(x="timepoint", y="open_chromatin", hue="patient_id", data=stats.reset_index(), ax=axis[0])
        stats = stats.sort_values("group_open_chromatin_norm")
        sns.barplot(x="timepoint", y="open_chromatin_norm", data=stats.reset_index(), palette="summer", ax=axis[1])
        sns.stripplot(x="timepoint", y="open_chromatin_norm", data=stats.reset_index(), palette="summer", ax=axis[1])
        sns.factorplot(x="timepoint", y="open_chromatin_norm", hue="patient_id", data=stats.reset_index(), ax=axis[1])
        axis[0].set_ylabel("Total open chromatin space (bp)")
        axis[1].set_ylabel("Total open chromatin space (normalized)")
        axis[0].set_xticklabels(axis[0].get_xticklabels(), rotation=45, ha="right")
        axis[1].set_xticklabels(axis[1].get_xticklabels(), rotation=45, ha="right")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.total_open_chromatin_space.per_group.svg".format(self.name)), bbox_inches="tight")

        # plot distribution of peak lengths
        sample_peak_lengths = map(get_peak_lengths, [s.peaks for s in samples])
        lengths = pd.melt(pd.DataFrame(sample_peak_lengths, index=[s.name for s in samples]).T, value_name='peak_length', var_name="sample_name").dropna()
        # lengths['knockout'] = lengths['sample_name'].str.extract("HAP1_(.*?)_")
        # lengths = lengths[~lengths['sample_name'].str.contains("GFP|HAP1_ATAC-seq_WT_Bulk_")]
        # # median lengths per sample (split-apply-combine)
        lengths = pd.merge(lengths, lengths.groupby('sample_name')['peak_length'].median().to_frame(name='mean_peak_length').reset_index())
        # # median lengths per group (split-apply-combine)
        # lengths = pd.merge(lengths, lengths.groupby('knockout')['peak_length'].median().to_frame(name='group_mean_peak_length').reset_index())

        lengths = lengths.sort_values("mean_peak_length")
        fig, axis = plt.subplots(2, 1, figsize=(8 * 1, 4 * 2))
        sns.boxplot(x="sample_name", y="peak_length", data=lengths, palette="summer", ax=axis[0], showfliers=False)
        axis[0].set_ylabel("Peak length (bp)")
        axis[0].set_xticklabels(axis[0].get_xticklabels(), visible=False)
        sns.boxplot(x="sample_name", y="peak_length", data=lengths, palette="summer", ax=axis[1], showfliers=False)
        axis[1].set_yscale("log")
        axis[1].set_ylabel("Peak length (bp)")
        axis[1].set_xticklabels(axis[1].get_xticklabels(), rotation=45, ha="right")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.peak_lengths.per_sample.svg".format(self.name)), bbox_inches="tight")

        # lengths = lengths.sort_values("group_mean_peak_length")
        # fig, axis = plt.subplots(2, 1, figsize=(8 * 1, 4 * 2))
        # sns.boxplot(x="knockout", y="peak_length", data=lengths, palette="summer", ax=axis[0], showfliers=False)
        # axis[0].set_ylabel("Peak length (bp)")
        # axis[0].set_xticklabels(axis[0].get_xticklabels(), visible=False)
        # sns.boxplot(x="knockout", y="peak_length", data=lengths, palette="summer", ax=axis[1], showfliers=False)
        # axis[1].set_yscale("log")
        # axis[1].set_ylabel("Peak length (bp)")
        # axis[1].set_xticklabels(axis[1].get_xticklabels(), rotation=45, ha="right")
        # sns.despine(fig)
        # fig.savefig(os.path.join(self.results_dir, "{}.peak_lengths.per_knockout.svg".format(self.name)), bbox_inches="tight")

        # peaks per chromosome per sample
        chroms = pd.DataFrame(map(get_peak_chroms, [s.peaks for s in samples]), index=[s.name for s in samples]).fillna(0).T
        chroms_norm = (chroms / chroms.sum(axis=0)) * 100
        chroms_norm = chroms_norm.ix[["chr{}".format(i) for i in range(1, 23) + ['X', 'Y', 'M']]]

        fig, axis = plt.subplots(1, 1, figsize=(8 * 1, 8 * 1))
        sns.heatmap(chroms_norm, square=True, cmap="summer", ax=axis)
        axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
        axis.set_yticklabels(axis.get_yticklabels(), rotation=0, ha="right")
        fig.savefig(os.path.join(self.results_dir, "{}.peak_location.per_sample.svg".format(self.name)), bbox_inches="tight")

        # Loop at summary statistics:
        # interval lengths
        fig, axis = plt.subplots()
        sns.distplot([interval.length for interval in self.sites], bins=300, kde=False, ax=axis)
        axis.set_xlim(0, 2000)  # cut it at 2kb
        axis.set_xlabel("peak width (bp)")
        axis.set_ylabel("frequency")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.lengths.svg".format(self.name)), bbox_inches="tight")

        # plot support
        fig, axis = plt.subplots(1, figsize=(4, 4))
        sns.distplot(self.support["support"], bins=100, ax=axis, kde=False)
        axis.set_ylabel("frequency")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.support.svg".format(self.name)), bbox_inches="tight")

        # Plot distance to nearest TSS
        fig, axis = plt.subplots(1, figsize=(4, 4))
        sns.distplot([x for x in self.closest_tss_distances if x < 100000], bins=50, ax=axis, kde=False)
        axis.set_xlabel("distance to nearest TSS (bp)")
        axis.set_ylabel("frequency")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.tss_distance.svg".format(self.name)), bbox_inches="tight")

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
        fig, axis = plt.subplots(3, sharex=True, sharey=False, figsize=(4, 4 * 3))
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
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.genomic_regions.svg".format(self.name)), bbox_inches="tight")

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

        fig, axis = plt.subplots(3, sharex=True, sharey=False, figsize=(4, 4 * 3))
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
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.chromatin_states.svg".format(self.name)), bbox_inches="tight")

        # distribution of count attributes
        data = self.coverage_annotated.copy()

        fig, axis = plt.subplots(1)
        sns.distplot(data["mean"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.mean.distplot.svg".format(self.name)), bbox_inches="tight")

        fig, axis = plt.subplots(1)
        sns.distplot(data["qv2"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.qv2.distplot.svg".format(self.name)), bbox_inches="tight")

        fig, axis = plt.subplots(1)
        sns.distplot(data["dispersion"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.dispersion.distplot.svg".format(self.name)), bbox_inches="tight")

        # this is loaded now
        df = pd.read_csv(os.path.join(self.data_dir, self.name + "_peaks.support.csv"))
        fig, axis = plt.subplots(1)
        sns.distplot(df["support"], rug=False, ax=axis)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "{}.support.distplot.svg".format(self.name)), bbox_inches="tight")

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
        c = X.corr()
        c.index = c.columns = c.index.get_level_values("sample_name")
        g = sns.clustermap(
            c, xticklabels=False, yticklabels=sample_display_names, annot=True,
            cmap="Spectral_r", figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, col_colors=color_dataframe.T)
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

        xx2 = xx.copy()
        xx2['ighv_mutation_status'] = xx2.index.get_level_values("ighv_mutation_status")
        xx2.index = xx2.index.get_level_values("sample_name")
        xx2.loc[:, ["ighv_mutation_status", "PC1", "PC2"]].to_csv(os.path.join("source_data", "fig2b.csv"))

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

        xx2 = xx.copy()
        xx2.index = xx2.index.get_level_values("sample_name")
        xx2.iloc[:, [2]].to_csv(os.path.join("source_data", "fig2c.csv"))

        order = xx.groupby(level=['patient_id']).apply(lambda x: min(x.loc[x.index.get_level_values("timepoint_name") == "after_Ibrutinib", "PC3"].squeeze(), x.loc[x.index.get_level_values("timepoint_name") == "before_Ibrutinib", "PC3"].squeeze())).sort_values()
        # order = order[[type(i) is np.float64 for i in order]].sort_values()
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
        x = pd.DataFrame(x_new)
        xx = x.apply(lambda j: (j - j.mean()) / j.std(), axis=0)
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
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
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
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.fig.savefig(os.path.join("results", "{}.PCA.PC_pvalues.enrichr_enrichments.{}.svg".format(self.name, gene_set_library)), bbox_inches="tight", dpi=300)

            g = sns.clustermap(
                -np.log10(pivot.ix[top_ids]),
                cbar_kws={"label": "-log10(p-value) z-score"}, col_cluster=True, z_score=0, figsize=(6, 15))
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
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

        sd = pd.DataFrame([np.log2(1 + df2[cond1]), np.log2(1 + df2[cond2])]).T
        sd['differential'] = sd.index.isin(diff2.index.tolist())
        sd.index.name = "region"
        sd.to_csv(os.path.join("source_data", "fig2d.csv"), index=True)

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

        a = self.accessibility.ix[diff2.index][[s.name for s in sel_samples]].T
        a.index = a.index.get_level_values("sample_name")
        import scipy
        az = a.apply(scipy.stats.zscore)
        az.to_csv(os.path.join("source_data", "fig2e.csv"), index=True)

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

        # Per gene changes
        # (for pathway diagram)
        change_df = self.coverage_annotated.join(df[['log2FoldChange', 'padj']])

        g = pd.merge(
            change_df.reset_index()[["index", "log2FoldChange", "padj"]],
            change_df.apply(lambda x: pd.Series(x['gene_name'].split(","), name="gene_name"), axis=1).stack().reset_index(level=1, drop=True).reset_index()
        )
        g = g.groupby([0])['log2FoldChange'].apply(lambda x: x.min() if x.mean() < 0 else x.max()).sort_values()
        g.to_csv(os.path.join(output_dir, "%s.%s.diff_regions.gene_level.fold_change.csv" % (output_suffix, trait)))

        m = max(abs(g.min()), abs(g.max()))
        normalizer = matplotlib.colors.Normalize(vmin=-m, vmax=m)
        # cmap = "coolwarm"
        # plt.scatter(g.rank(), g, alpha=0.8, color=plt.get_cmap(cmap)(normalizer(g)))
        pathway_genes = pickle.load(open(os.path.join("metadata", "pathway_gene_annotation_kegg.pickle"), "rb"))
        pathway_genes2 = pd.concat(
            [pd.Series(genes, index=[path for _ in genes]) for path, genes in pathway_genes.items()]
        ).squeeze()
        pathway_genes3 = pathway_genes2[pathway_genes2.index.str.contains("NF-k|B cell|foxo", case=False)]
        g2 = g[g.index.isin(pathway_genes3.tolist())]
        g2.to_csv(os.path.join(output_dir, "%s.%s.diff_regions.gene_level.in_pathways.fold_change.csv" % (output_suffix, trait)))

        fig, axis = plt.subplots(1, figsize=(4, 4))
        axis.scatter(g2.rank(), g2, color=plt.get_cmap("coolwarm")(normalizer(g2)))
        for i, gene in enumerate(g2.index):
            axis.text(g2.rank()[i], g2[i], gene, color=plt.get_cmap("coolwarm")(normalizer(g2[i])))
        fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.gene_level.in_pathways.svg" % (output_suffix, trait)), bbox_inches="tight")

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
        g = sns.clustermap(regions_pivot)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "region_type_enrichment.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "region_type_enrichment.png"), bbox_inches="tight", dpi=300)

        #

        # LOLA
        # read in
        lola = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_regions.lola.csv" % (output_suffix, trait)))
        # pretty names
        lola["comparison"] = lola["comparison"].str.extract("%s.%s.diff_regions.comparison_(.*)" % (output_suffix, trait), expand=True)

        # unique ids for lola sets
        cols = ['description', u'cellType', u'tissue', u'antibody', u'treatment', u'dataSource', u'filename']
        lola['label'] = lola[cols].astype(str).apply(string.join, axis=1)

        sd = lola[lola['comparison'].str.contains("down")].sort_values('pValueLog', ascending=False).head(13)
        sd[['label', 'pValueLog']].to_csv(os.path.join("source_data", "fig2f.csv"), index=False)

        # pivot table
        lola_pivot = pd.pivot_table(lola, values="pValueLog", columns="label", index="comparison")
        lola_pivot.columns = lola_pivot.columns.str.decode("utf-8")

        # plot correlation
        g = sns.clustermap(lola_pivot.T.corr())
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "lola.correlation.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "lola.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(g.dendrogram_col.linkage, 3, criterion="maxclust")

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
        g = sns.clustermap(lola_pivot[list(set(top_terms))].replace({np.inf: 50}), z_score=0, figsize=(20, 12))
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "lola.cluster_specific.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "lola.cluster_specific.png"), bbox_inches="tight", dpi=300)

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
        g = sns.clustermap(motifs_pivot.T.corr())
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "motifs.correlation.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "motifs.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(g.dendrogram_col.linkage, 5, criterion="maxclust")

        # Get top n terms which are more in each cluster compared with all others
        top_terms = list()
        cluster_means = pd.DataFrame()
        for cluster in set(cluster_assignment):
            cluster_comparisons = motifs_pivot.index[cluster_assignment == cluster].tolist()
            other_comparisons = motifs_pivot.index[cluster_assignment != cluster].tolist()

            terms = (motifs_pivot.ix[cluster_comparisons].mean() - motifs_pivot.ix[other_comparisons].mean()).sort_values()

            top_terms += terms.dropna().head(n).index.tolist()

        # plot clustered heatmap
        g = sns.clustermap(motifs_pivot[list(set(top_terms))], figsize=(20, 12))  # .apply(lambda x: (x - x.mean()) / x.std())
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.png"), bbox_inches="tight", dpi=300)

        df = motifs_pivot[list(set(top_terms))]  # .apply(lambda x: (x - x.mean()) / x.std())

        g = sns.clustermap(df[df.mean(1) > -0.5], figsize=(20, 12))
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.png"), bbox_inches="tight", dpi=300)

        #

        # ENRICHR
        # read in
        enrichr = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_regions.enrichr.scores.csv" % (output_suffix, trait)))
        # pretty names
        enrichr["comparison"] = enrichr["comparison"].str.extract("%s.%s.diff_regions.comparison_(.*)" % (output_suffix, trait), expand=True)

        a = enrichr[enrichr['gene_set_library'].str.contains("NCI-Nature_2016") & enrichr['comparison'].str.contains("down")].sort_values("combined_score", ascending=False).head()
        b = enrichr[enrichr['gene_set_library'].str.contains("KEGG_2016") & enrichr['comparison'].str.contains("down")].sort_values("combined_score", ascending=False).head()
        a['gene_set_library'] = "NCI-Nature_2016"
        b['gene_set_library'] = "KEGG_2016"
        sd = a.append(b)
        sd[['gene_set_library', 'description', 'combined_score']].to_csv(os.path.join("source_data", "fig2h.csv"), index=False)

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
            g = sns.clustermap(enrichr_pivot.T.corr())
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.savefig(os.path.join(output_dir, "enrichr.%s.correlation.svg" % gene_set_library), bbox_inches="tight")
            g.savefig(os.path.join(output_dir, "enrichr.%s.correlation.png" % gene_set_library), bbox_inches="tight", dpi=300)

            cluster_assignment = fcluster(g.dendrogram_col.linkage, 4, criterion="maxclust")

            # Get top n terms which are more in each cluster compared with all others
            top_terms = list()
            cluster_means = pd.DataFrame()
            for cluster in set(cluster_assignment):
                cluster_comparisons = enrichr_pivot.index[cluster_assignment == cluster].tolist()
                other_comparisons = enrichr_pivot.index[cluster_assignment != cluster].tolist()

                terms = (enrichr_pivot.ix[cluster_comparisons].mean() - enrichr_pivot.ix[other_comparisons].mean()).sort_values()

                top_terms += terms.dropna().head(n).index.tolist()

            # plot clustered heatmap
            g = sns.clustermap(enrichr_pivot[list(set(top_terms))], figsize=(20, 12))  # .apply(lambda x: (x - x.mean()) / x.std())
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.svg" % gene_set_library), bbox_inches="tight")
            g.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.png" % gene_set_library), bbox_inches="tight", dpi=300)

    # using genes weighted by score

    def treatment_response_score(
            self, samples=None, trait="timepoint_name",
            output_suffix="ibrutinib_treatment", attributes=[
                "sample_name", "cell_type", "patient_id", "clinical_centre", "timepoint_name", "patient_gender", "patient_age_at_collection",
                "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
                "del11q", "del13q", "del17p", "tri12", "p53",
                "time_since_treatment", "treatment_response",
                "del11q_threshold", "del13q_threshold", "del17p_threshold", "tri12_threshold"]):
        """
        """
        def z_score(x):
            return (x - x.mean()) / x.std()

        if samples is None:
            samples = [s for s in self.samples if s.patient_id != "CLL16"]

        index = self.accessibility.columns[self.accessibility.columns.get_level_values("sample_name").isin([s.name for s in samples])]

        color_dataframe = pd.DataFrame(get_level_colors(self, index=index, levels=attributes), index=attributes, columns=[s.name for s in samples])
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

        numerical_variables = [
            "patient_age_at_collection", "CD38_cells_percentage", "leuko_count (10^3/uL)",
            "% lymphocytes", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
            "del11q", "del13q", "del17p", "tri12",
            "time_since_treatment"
        ]
        for var in numerical_variables:
            for s in samples:
                setattr(s, var, float(getattr(s, var)))

        for measurement in ["score", "combined_score"]:
            for attr in attributes[3:]:
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
                    associations.append([
                        measurement, attr, variable_type, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    continue

                if variable_type == "categorical":
                    # It categorical, test pairwise combinations of attributes
                    for group1, group2 in itertools.combinations(groups, 2):
                        g1_values = scores.loc[scores.index.isin([s.name for s in sel_samples if getattr(s, attr) == group1]), measurement]
                        g2_values = scores.loc[scores.index.isin([s.name for s in sel_samples if getattr(s, attr) == group2]), measurement]

                        # Test ANOVA (or Kruskal-Wallis H-test)
                        st, p = kruskal(g1_values, g2_values)

                        # Append
                        associations.append([measurement, attr, variable_type, group1, group2, g1_values.shape[0], g2_values.shape[0], p, st])

                elif variable_type == "numerical":
                    # It numerical, calculate pearson correlation
                    trait_values = [getattr(s, attr) for s in sel_samples]
                    st, p = pearsonr(scores[measurement].ix[[s.name for s in sel_samples]], trait_values)

                    associations.append([measurement, attr, variable_type, np.nan, np.nan, np.nan, np.nan, p, st])

        associations = pd.DataFrame(
            associations,
            columns=["score", "attribute", "variable_type", "group_1", "group_2", "group_1_n", "group_2_n", "p_value", "stat"])
        associations = associations[(associations['group_1'] != 'nan') & (associations['group_2'] != 'nan')]
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        from statsmodels.sandbox.stats.multicomp import multipletests
        results_dir = "results"
        output_suffix = "cll-ibrutinib_AKH.ibrutinib_treatment"
        trait = "timepoint_name"

        # associations = pd.read_csv(os.path.join(results_dir, "{}.{}.diff_regions.intensity_score_combined.associations.csv".format(output_suffix, trait)))
        associations = associations[(associations['group_1'] != 'nan') & (associations['group_2'] != 'nan')]
        associations["q_value"] = multipletests(associations["p_value"])[1]

        associations = associations.sort_values("p_value")

        p = associations[associations['score'] == 'score']
        fig, axis = plt.subplots(1, 3, sharey=True)
        sns.barplot(-np.log10(p["p_value"]), p["attribute"], orient="horizont", color="blue", ax=axis[0])
        sns.barplot(-np.log10(p["q_value"]), p["attribute"], orient="horizont", color="blue", ax=axis[1])
        sns.barplot(p["stat"], p["attribute"], orient="horizont", color="blue", ax=axis[2])
        axis[2].set_xlim(-6, 6)
        axis[0].axvline(1.3, linestyle="--", zorder=0)
        axis[1].axvline(1.3, linestyle="--", zorder=0)
        axis[2].axvline(0, linestyle="--", zorder=0)
        fig.savefig(os.path.join(results_dir, "{}.{}.diff_regions.intensity_score.associations.q_value_stat.svg".format(output_suffix, trait)), bbox_inches="tight")

        p = associations[associations['score'] == 'combined_score']
        fig, axis = plt.subplots(1, 3, sharey=True)
        sns.barplot(-np.log10(p["p_value"]), p["attribute"], orient="horizont", color="blue", ax=axis[0])
        sns.barplot(-np.log10(p["q_value"]), p["attribute"], orient="horizont", color="blue", ax=axis[1])
        sns.barplot(p["stat"], p["attribute"], orient="horizont", color="blue", ax=axis[2])
        axis[2].set_xlim(-6, 6)
        axis[0].axvline(1.3, linestyle="--", zorder=0)
        axis[1].axvline(1.3, linestyle="--", zorder=0)
        axis[2].axvline(0, linestyle="--", zorder=0)
        fig.savefig(os.path.join(results_dir, "{}.{}.diff_regions.intensity_score_combined.associations.q_value_stat.svg".format(output_suffix, trait)), bbox_inches="tight")


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
        g = sns.clustermap(regions_pivot)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "region_type_enrichment.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "region_type_enrichment.png"), bbox_inches="tight", dpi=300)

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
        g = sns.clustermap(lola_pivot.T.corr())
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "lola.correlation.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "lola.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(g.dendrogram_col.linkage, 3, criterion="maxclust")

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
        g = sns.clustermap(lola_pivot[list(set(top_terms))].replace({np.inf: 50}), z_score=0, figsize=(20, 12))
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "lola.cluster_specific.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "lola.cluster_specific.png"), bbox_inches="tight", dpi=300)

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
        g = sns.clustermap(motifs_pivot.T.corr())
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "motifs.correlation.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "motifs.correlation.png"), bbox_inches="tight", dpi=300)

        cluster_assignment = fcluster(g.dendrogram_col.linkage, 5, criterion="maxclust")

        # Get top n terms which are more in each cluster compared with all others
        top_terms = list()
        cluster_means = pd.DataFrame()
        for cluster in set(cluster_assignment):
            cluster_patients = motifs_pivot.index[cluster_assignment == cluster].tolist()
            other_patients = motifs_pivot.index[cluster_assignment != cluster].tolist()

            terms = (motifs_pivot.ix[cluster_patients].mean() - motifs_pivot.ix[other_patients].mean()).sort_values()

            top_terms += terms.dropna().head(n).index.tolist()

        # plot clustered heatmap
        g = sns.clustermap(motifs_pivot[list(set(top_terms))].apply(lambda x: (x - x.mean()) / x.std()), figsize=(20, 12))  #
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.png"), bbox_inches="tight", dpi=300)

        df = motifs_pivot[list(set(top_terms))].apply(lambda x: (x - x.mean()) / x.std())

        g = sns.clustermap(df[df.mean(1) > -0.5], figsize=(20, 12))
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.svg"), bbox_inches="tight")
        g.savefig(os.path.join(output_dir, "motifs.cluster_specific.only_some.png"), bbox_inches="tight", dpi=300)

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
            g = sns.clustermap(enrichr_pivot.T.corr())
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.savefig(os.path.join(output_dir, "enrichr.%s.correlation.svg" % gene_set_library), bbox_inches="tight")
            g.savefig(os.path.join(output_dir, "enrichr.%s.correlation.png" % gene_set_library), bbox_inches="tight", dpi=300)

            cluster_assignment = fcluster(g.dendrogram_col.linkage, 4, criterion="maxclust")

            # Get top n terms which are more in each cluster compared with all others
            top_terms = list()
            cluster_means = pd.DataFrame()
            for cluster in set(cluster_assignment):
                cluster_comparisons = enrichr_pivot.index[cluster_assignment == cluster].tolist()
                other_comparisons = enrichr_pivot.index[cluster_assignment != cluster].tolist()

                terms = (enrichr_pivot.ix[cluster_comparisons].mean() - enrichr_pivot.ix[other_comparisons].mean()).sort_values()

                top_terms += terms.dropna().head(n).index.tolist()

            # plot clustered heatmap
            g = sns.clustermap(enrichr_pivot[list(set(top_terms))].apply(lambda x: (x - x.mean()) / x.std()), figsize=(20, 12))  #
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.svg" % gene_set_library), bbox_inches="tight")
            g.savefig(os.path.join(output_dir, "enrichr.%s.cluster_specific.png" % gene_set_library), bbox_inches="tight", dpi=300)


def annotate_drugs(analysis, sensitivity):
    """
    """
    from bioservices.chembl import ChEMBL
    from bioservices.kegg import KEGG
    from collections import defaultdict
    import string
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

    output_dir = os.path.join("results", "pharmacoscopy")

    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    rename = {
        "ABT-199 = Venetoclax": "Venetoclax",
        "ABT-263 = Navitoclax": "Navitoclax",
        "ABT-869 = Linifanib": "Linifanib",
        "AC220 = Quizartinib": "Quizartinib",
        "Buparlisib (BKM120)": "Buparlisib",
        "EGCG = Epigallocatechin gallate": "Epigallocatechin gallate",
        "JQ1": "JQ1",
        "MLN-518 = Tandutinib": "Tandutinib",
        "Selinexor (KPT-330)": "Selinexor"}
    sensitivity["proper_name"] = sensitivity["drug"]
    for p, n in rename.items():
        sensitivity["proper_name"] = sensitivity["proper_name"].replace(p, n)

    # CLOUD id/ SMILES
    cloud = pd.read_csv(os.path.join("metadata", "CLOUD_simple_annotation.csv"))
    cloud.loc[:, "name_lower"] = cloud["drug_name"].str.lower()
    sensitivity.loc[:, "name_lower"] = sensitivity["proper_name"].str.lower()
    annot = pd.merge(sensitivity[['drug', 'proper_name', 'name_lower']].drop_duplicates(), cloud, on="name_lower", how="left")

    # DGIdb: drug -> genes
    interact = pd.read_csv("http://dgidb.org/data/interactions.tsv", sep="\t")
    interact.loc[:, "name_lower"] = interact["drug_claim_primary_name"].str.lower()
    cats = pd.read_csv("http://dgidb.org/data/categories.tsv", sep="\t")
    dgidb = pd.merge(interact, cats, how="left", left_on="gene_name", right_on="entrez_gene_symbol")
    drugs = pd.read_table("http://www.dgidb.org/data/drugs.tsv", sep="\t")
    dgidb = pd.merge(dgidb, cats, how="left")
    dgidb.to_csv(os.path.join(output_dir, "dgidb.interactions_categories_drugs.20180809.csv"), index=False)
    # tight match
    annot = pd.merge(annot, dgidb, on="name_lower", how="left")
    annot.to_csv(os.path.join("metadata", "drugs_annotated.20180809.csv"), index=False)
    annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.20180809.csv"))

    # CHEMBL: drug -> mode of action
    import requests
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/{}?format=json"

    chembl = list()
    for drug in annot['drug_chembl_id'].dropna().unique():
        print(drug)
        response = requests.get(base_url.format(drug))
        if not response.ok:
            print(Exception('Error fetching ChEMBL entry:"{}"'.format(drug)))
        # Get enriched sets in gene set
        chembl.append(json.loads(response.text))
    chembl = pd.DataFrame(chembl, index=annot['drug_chembl_id'].dropna().unique())
    chembl.to_csv(os.path.join("metadata", "chembl.20180809.csv"), index=True)

    # Get ATC term
    from bs4 import BeautifulSoup
    base_url = "https://www.whocc.no/atc_ddd_index/?code={}"

    atcs = chembl['atc_classifications'].apply(pd.Series).stack().reset_index().drop("level_1", axis=1)
    atcs.columns = ['CHEMBL_id', "ATC_id"]

    atc_codes = pd.DataFrame()
    for atc in atcs['ATC_id'].dropna().drop_duplicates():
        print(atc)
        response = requests.get(base_url.format(atc))
        soup = BeautifulSoup(response.content, "html.parser")
        tags = list()
        for tag in soup.findAll("div", id="content")[0].findAll("a")[2:-1]:
            tags += [tag.string.replace(r".*code=", "").replace(r"\">.*", "")]
        atc_codes = atc_codes.append(pd.DataFrame([tags, [atc] * len(tags)]).T)
    atc_codes.columns = ['ATC_term', 'ATC_id']
    atc_codes.to_csv(os.path.join("metadata", "atc.20180809.csv"), index=False)

    annot2 = pd.merge(annot, pd.merge(atcs, atc_codes), left_on="drug_chembl_id", right_on="CHEMBL_id")

    annot2.to_csv(os.path.join("metadata", "drugs_annotated.with_ATC.20180809.csv"), index=False)
    annot3 = (annot2[['drug', 'ATC_id', 'ATC_term']]
        .drop_duplicates()
        .groupby('drug')['ATC_term']
        .apply(np.unique)
        .apply(pd.Series).stack()
        .reset_index(level=1, drop=True))
    annot4 = annot3.loc[(annot3.str.lower() != annot3.index.str.lower())]
    annot4.to_csv(os.path.join("metadata", "drugs_annotated.with_ATC.20180809.slim.csv"), index=True)

    # Get PubChem IDs
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/JSON"
    pubchem = list()
    for drug in annot['drug_chembl_id'].dropna().unique():
        print(drug)
        response = requests.get(base_url.format(drug))
        if not response.ok:
            print(Exception('Error fetching ChEMBL entry:"{}"'.format(drug)))
        # Get enriched sets in gene set
        try:
            pubchem.append(json.loads(response.text)['PC_Compounds'][0]['id']['id']['cid'])
        except:
            print(Exception('Not found: ChEMBL entry:"{}"'.format(drug)))
            pubchem.append(np.nan)
    pubchem_ids = pd.Series(pubchem, index=annot['drug_chembl_id'].dropna().unique())

    # Get PubChem entries
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/MolecularFormula,MolecularWeight,InChIKey/JSON"
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/JSON"
    pubchem = list()
    for id_ in pubchem_ids.dropna().unique():
        print(id_)
        response = requests.get(base_url.format(id_))
        if not response.ok:
            print(Exception('Error fetching ChEMBL entry:"{}"'.format(id_)))
        # Get enriched sets in gene set
        try:
            pubchem.append(json.loads(response.text)['PC_Compounds'][0]['id']['id']['cid'])
        except:
            print(Exception('Not found: ChEMBL entry:"{}"'.format(id_)))
            pubchem.append(np.nan)
    pubchem = pd.DataFrame(pubchem, index=annot['drug_chembl_id'].dropna().unique())
    pubchem.to_csv(os.path.join("metadata", "pubchem.20180809.csv"), index=True)


    # CHEMBL -> CHEBI -> KEGG
    k = KEGG()
    chebi_kegg = pd.Series(k.conv("compound", "chebi"))
    chebi_kegg.index.name = "chebi"
    chebi_kegg.name = "kegg"
    uni = UniChem()
    chembl_chebi = pd.Series(uni.get_mapping("chebi", "chembl"))
    chembl_chebi.index = "chebi:" + chembl_chebi.index.astype(str)
    chembl_chebi.index.name = "chebi"
    chembl_chebi.name = "chembl"
    chembl_kegg = pd.Series(uni.get_mapping("chembl", "kegg_ligand"))
    chembl_kegg.index.name = "chembl"
    chembl_kegg.name = "kegg_ligand"

    ids = chembl_chebi.to_frame().join(chebi_kegg.to_frame()).reset_index()
    ids.to_csv(os.path.join("metadata", "ids.chembl_chebi_kegg.20180809.csv"), index=False)

    annot3 = pd.merge(annot2, ids, left_on="drug_chembl_id", right_on="chembl")
    annot3.to_csv(os.path.join("metadata", "drugs_annotated.with_ATC_chembl_chebi_kegg.20180809.csv"), index=False)

    # ChEBI ontology
    ch = ChEBI()
    ontology = pd.DataFrame()
    for id_ in annot3['chebi'].drop_duplicates().dropna().sort_values():
        print(id_)
        ont = ch.getCompleteEntity(id_.upper())

        terms = [x.chebiName for x in ont['OntologyParents']]
        ontology = ontology.append(pd.DataFrame([terms, [id_] * len(terms)]).T)
    ontology.columns = ['ontology_term', 'chebi']
    ontology.to_csv(os.path.join("metadata", "chebi_ontology.20180809.csv"), index=False)

    annot4 = pd.merge(annot3, ontology, on="chebi")
    annot4.to_csv(os.path.join("metadata", "drugs_annotated.with_ATC_chembl_chebi_kegg_ontology.20180809.csv"), index=False)

    annot4[['drug', 'ontology_term']].drop_duplicates().to_csv(os.path.join("metadata", "drugs_annotated.with_ATC_chembl_chebi_kegg_ontology.20180809.slim.csv"), index=False)



    # KEGG target-based annotation of drugs
    kegg = list()
    for name in annot4['proper_name'].dropna().unique():
        print(name)
        o = k.find("drug", name)
        if o == u'\n':
            o = k.find("compound", name)
        if o == u'\n':
            kegg.append(np.nan)
        else:
            kegg.append(o.split('\t')[0].strip())
    kegg = pd.Series(kegg, index=annot4['proper_name'].dropna().unique())
    kegg.name = "kegg_drug"
    kegg.index.name = "proper_name"

    annot5 = pd.merge(annot4, kegg.reset_index()).rename(columns={"kegg": "kegg_compound"})
    annot5.to_csv(os.path.join("metadata", "drugs_annotated.with_ATC_chembl_chebi_kegg_ontology_keggdrug.20180809.csv"), index=False)

    # Add KEGG efficacy and BRITE entries
    efficacy = list()
    brite = list()
    for entry in annot5['kegg_drug'].dropna().unique():
        print(entry)
        o = k.parse(k.get(entry))
        # l = o['BRITE'].replace("             ", "")
        # l[l.index("Target-based"):]
        if type(o) is dict:
            if "BRITE" in o:
                brite.append(o['BRITE'].strip())
            if "EFFICACY" in o:
                efficacy.append(o['EFFICACY'].strip())
        else:
            brite.append(o[o.index("BRITE"):o.index("DBLINKS")].strip())
            efficacy.append(o[o.index("EFFICACY"):o.index("DISEASE")].strip())
    brite = pd.Series(brite, index=annot5['kegg_drug'].dropna().unique())
    efficacy = pd.Series(efficacy, index=annot5['kegg_drug'].dropna().unique())

    efficacy = (
        efficacy.str.split("\n").apply(lambda x: x[0]).str.strip()
        .str.split(",").apply(pd.Series).stack().str.strip()
        .reset_index(level=1, drop=True))
    efficacy.index.name = 'kegg_drug'
    efficacy.name = 'kegg_efficacy'

    annot6 = pd.merge(annot5, efficacy.reset_index())
    annot6.to_csv(os.path.join("metadata", "drugs_annotated.with_ATC_chembl_chebi_kegg_ontology_keggdrug.kegg_efficacy.20180809.csv"), index=False)

    annot6[['drug', 'kegg_efficacy']].drop_duplicates().to_csv(os.path.join("metadata", "drugs_annotated.kegg_efficacy.20180809.slim.csv"), index=False)


    q = brite.str.split("\n").apply(pd.Series).stack().str.strip().value_counts().sort_values()
    q[(~q.index.str.contains(r'^D\d+')) & (~q.index.str.contains(r'^L\d+'))]



    #     brite[entry] = [ss.strip() for ss in s]


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
    annot.to_csv(os.path.join("metadata", "drugs_annotated.20180809.csv"), index=False)
    annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.20180809.csv"))

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
    mats0_a = dgidb.groupby(["drug_claim_primary_name"])['entrez_gene_symbol'].nunique()
    axis[0].hist([mats0, mats0_a], max(mats0.tolist() + mats0_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[0].set_xlabel("Genes")

    ax = zoomed_inset_axes(axis[0], 3, loc=1, axes_kwargs={"xlim": (0, 30), "ylim": (0, .20)})
    ax.hist([mats0, mats0_a], max(mats0.tolist() + mats0_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)

    # support of each drug-> gene assignment
    axis[1].set_title("Support of each interaction")
    mats1 = annot.groupby(["proper_name", "entrez_gene_symbol"])['interaction_claim_source'].nunique()
    mats1_a = dgidb.groupby(["drug_claim_primary_name", "entrez_gene_symbol"])['interaction_claim_source'].nunique()
    axis[1].hist([mats1, mats1_a], max(mats1.tolist() + mats1_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[1].set_xlabel("Sources")

    # types of interactions per drug (across genes)
    axis[2].set_title("Interaction types per drug")
    mats2 = annot.groupby(["proper_name"])['interaction_types'].nunique()
    mats2_a = dgidb.groupby(["drug_claim_primary_name"])['interaction_types'].nunique()
    axis[2].hist([mats2, mats2_a], max(mats2.tolist() + mats2_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[2].set_xlabel("Interaction types")

    # types of interactions per drug-> assignemnt
    axis[3].set_title("Interactions types per drug->gene interaction")
    mats3 = annot.groupby(["proper_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    mats3_a = dgidb.groupby(["drug_claim_primary_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    axis[3].hist([mats3, mats3_a], max(mats3.tolist() + mats3_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[3].set_xlabel("Interaction types")

    # types of categories per drug (across genes)
    axis[4].set_title("Categories per drug")
    mats4 = annot.groupby(["proper_name"])['interaction_types'].nunique()
    mats4_a = dgidb.groupby(["drug_claim_primary_name"])['interaction_types'].nunique()
    axis[4].hist([mats4, mats4_a], max(mats4.tolist() + mats4_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[4].set_xlabel("Categories")

    # types of categories per drug-> assignemnt
    axis[5].set_title("Categories per drug->gene interaction")
    mats5 = annot.groupby(["proper_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    mats5_a = dgidb.groupby(["drug_claim_primary_name", "entrez_gene_symbol"])['interaction_types'].nunique()
    axis[5].hist([mats5, mats5_a], max(mats5.tolist() + mats5_a.tolist()), normed=True, histtype='bar', align='mid', alpha=0.8)
    axis[5].set_xlabel("Categories")

    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene.annotations.20180725.svg"), bbox_inches="tight")

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
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway.annotations.20180725.svg"), bbox_inches="tight")

    # Drug vs Category
    annot['intercept'] = 1
    cat_pivot = pd.pivot_table(annot, index="drug", columns='category', values="intercept")
    # get drugs with not gene annotated and add them to matrix
    for drug in annot['drug'].drop_duplicates()[~annot['drug'].drop_duplicates().isin(cat_pivot.index)]:
        cat_pivot.loc[drug, :] = pd.np.nan
    # heatmap
    g = sns.clustermap(cat_pivot.fillna(0), figsize=(7, 12))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-categories.binary.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-categories.binary.svg"), bbox_inches="tight")

    # Drug vs Gene matrix heatmap
    gene_pivot = pd.pivot_table(gene_net, index="drug", columns='entrez_gene_symbol', values="interaction_score", aggfunc=sum)
    # get drugs with not gene annotated and add them to matrix
    for drug in annot['drug'].drop_duplicates()[~annot['drug'].drop_duplicates().isin(gene_pivot.index)]:
        gene_pivot.loc[drug, :] = pd.np.nan
    # heatmap
    g = sns.clustermap(gene_pivot.fillna(0), figsize=(20, 10))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.svg"), bbox_inches="tight")
    # binary
    path_pivot_binary = (~gene_pivot.isnull()).astype(int)
    g = sns.clustermap(path_pivot_binary, figsize=(20, 10))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.binary.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-gene_interactions.binary.svg"), bbox_inches="tight")

    # Drug vs Gene matrix heatmap
    path_pivot = pd.pivot_table(path_net, index="drug", columns='kegg_pathway_name', values="interaction_score", aggfunc=sum)
    # get drugs with not gene annotated and add them to matrix
    for drug in annot['drug'].drop_duplicates()[~annot['drug'].drop_duplicates().isin(path_pivot.index)]:
        path_pivot.loc[drug, :] = pd.np.nan
    # heatmap
    g = sns.clustermap(path_pivot.fillna(0), figsize=(20, 10))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.svg"), bbox_inches="tight")
    # binary
    path_pivot_binary = (~path_pivot.isnull()).astype(int)
    g = sns.clustermap(path_pivot_binary, figsize=(20, 10))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
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
                master_g = sns.clustermap(np.log2(1 + pathway_matrix.fillna(0)), figsize=(30, 15))
                for tick in master_g.ax_heatmap.get_xticklabels():
                    tick.set_rotation(90)
                for tick in master_g.ax_heatmap.get_yticklabels():
                    tick.set_rotation(0)
                master_fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.score.png"), bbox_inches="tight", dpi=300)
                # fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.score.svg"), bbox_inches="tight")

                g = sns.clustermap(np.log2(1 + pathway_matrix_norm.fillna(0)), figsize=(30, 15), row_linkage=master_fig.dendrogram_row.linkage, col_linkage=master_fig.dendrogram_col.linkage)
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
                g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
                fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.weighted_score.png"), bbox_inches="tight", dpi=300)
                # fig.savefig(os.path.join(output_dir, "pharmacoscopy.drug-pathway_interactions.weighted_score.svg"), bbox_inches="tight")

            g = sns.clustermap(np.log2(1 + pathway_scores.fillna(0)), figsize=(30, 15), row_linkage=master_fig.dendrogram_row.linkage, col_linkage=master_fig.dendrogram_col.linkage)
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
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
        g = sns.clustermap(df_pivot, figsize=(8, 20))
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, ha="left")
        g.savefig(os.path.join(output_dir, "{}.svg".format(name)), bbox_inches="tight")

        # sorted
        p = df_pivot.ix[df_pivot.sum(axis=1).sort_values().index]  # sort drugs
        p = p[p.sum(axis=0).sort_values().index]  # sort samples
        g = sns.clustermap(p, figsize=(8, 20), col_cluster=False, row_cluster=False, metric="correlation")
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, ha="left")
        g.savefig(os.path.join(output_dir, "{}.both_axis_sorted.svg".format(name)), bbox_inches="tight")\

        # only from greg's drug list
        sel_drugs = pd.read_csv(os.path.join("metadata", "signif_drugs.txt")).squeeze()
        df_pivot = pd.pivot_table(df, index="drug", columns="p_id", values=name)
        p = df_pivot.loc[sel_drugs, ~df_pivot.columns.str.contains("CLL2")].T

        g = sns.clustermap(p, figsize=(8, 20), square=True, metric="correlation", cbar_kws={"label": "Pharmacoscopy score"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, ha="left")
        g.savefig(os.path.join(output_dir, "{}.selected_drugs.svg".format(name)), bbox_inches="tight")
        g = sns.clustermap(p, figsize=(8, 20), col_cluster=True, row_cluster=False, square=True, metric="correlation", cbar_kws={"label": "Pharmacoscopy score"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, ha="left")
        g.savefig(os.path.join(output_dir, "{}.selected_drugs.both_axis_sorted.svg".format(name)), bbox_inches="tight")
        # Z-score
        df_pivot = pd.pivot_table(df, index="drug", columns="p_id", values=name)
        p = df_pivot.loc[sel_drugs, ~df_pivot.columns.str.contains("CLL2")].T
        p.index = pd.MultiIndex.from_arrays([[int(x[0][3:]) for x in p.index.str.split(" ")], [x[1] for x in p.index.str.split(" ")]])
        p = p.sortlevel(level=[0, 1], ascending=[True, False])

        # z-transform
        p = z_score(p)

        # select drugs

        g = sns.clustermap(p, figsize=(8, 20), square=True, metric="correlation", cbar_kws={"label": "Pharmacoscopy score\n(Z-score)"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, ha="left")
        # g.savefig(os.path.join(output_dir, "{}.selected_drugs.z_score.svg".format(name)), bbox_inches="tight")
        g = sns.clustermap(p, figsize=(8, 20), col_cluster=True, row_cluster=False, square=True, metric="correlation", cbar_kws={"label": "Pharmacoscopy score\n(Z-score)"})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        # g.savefig(os.path.join(output_dir, "{}.selected_drugs.both_axis_sorted.z_score.svg".format(name)), bbox_inches="tight")

        # across samples in each timepoint - sorted
        a = df_pivot[df_pivot.columns[df_pivot.columns.str.contains("after")]].mean(axis=1)
        b = df_pivot[df_pivot.columns[df_pivot.columns.str.contains("before")]].mean(axis=1)
        a.name = "after_ibrutinib"
        b.name = "before_ibrutinib"

        p = pd.DataFrame([a, b]).T
        p = p.ix[df_pivot.sum(axis=1).sort_values().index]  # sort drugs
        p = p[p.sum(axis=0).sort_values().index]  # sort samples
        g = sns.clustermap(p, figsize=(8, 20), col_cluster=False, row_cluster=False)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "{}.across_patients.both_axis_sorted.svg".format(name)), bbox_inches="tight")

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
        g = sns.clustermap(abs_diff.dropna(), figsize=(20, 8))
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "{}.timepoint_change.abs_diff.svg".format(name)), bbox_inches="tight")

        # clustered
        p = abs_diff.dropna().ix[abs_diff.dropna().sum(axis=1).sort_values().index]  # sort drugs
        p = p[p.sum(axis=0).sort_values().index]  # sort samples
        g = sns.clustermap(p, figsize=(20, 8), col_cluster=False, row_cluster=False)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(os.path.join(output_dir, "{}.timepoint_change.abs_diff.both_axis_sorted.svg".format(name)), bbox_inches="tight")

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
    g = sns.clustermap(pathway_space.T.dropna(), figsize=(20, 8))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.svg"), bbox_inches='tight')
    g = sns.clustermap(pathway_space.T.dropna(), figsize=(20, 8), z_score=1)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.z_score.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.z_score.svg"), bbox_inches='tight')
    g = sns.clustermap(new_drugs.T.dropna(), figsize=(20, 8))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.svg"), bbox_inches='tight')
    g = sns.clustermap(new_drugs.T.dropna(), figsize=(20, 8), z_score=0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
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



    # Pharmacoscopy aggregated by drug classes/ontology 20180809
    pharma = pd.read_csv(os.path.join("metadata", "KEGG_annotate_per_drug_selectivity_score_09082018.csv"), index_col=0)

    # kegg
    ont = pd.read_csv(os.path.join("metadata", 'drugs_annotated.with_ATC_chembl_chebi_kegg_ontology.20180809.slim.csv'), index_col=0, squeeze=True)

    # atc
    all_ = pd.read_csv(os.path.join("metadata", 'drugs_annotated.with_ATC_chembl_chebi_kegg_ontology.20180809.csv'), index_col=0)
    atc = pd.read_csv(os.path.join("metadata", "atc.20180809.csv"))
    atc_ = pd.merge(all_.reset_index(), atc)[['drug', 'ATC_term']].drop_duplicates()
    atc_ = atc_.set_index("drug").squeeze()

    atc_.drop_duplicates().to_csv(os.path.join("metadata", 'drugs_annotated.atc.20180809.slim.csv'), index=True)

    atc_ = atc_.loc[~(atc_.index.str.lower() == atc_.str.lower())]

    # kegg efficacy
    kegg_efficacy = pd.read_csv(os.path.join("metadata", "drugs_annotated.kegg_efficacy.20180809.slim.csv"), index_col=0, squeeze=True)

    for df, label in [(ont, "chebi_ontology"), (atc_, "ATC_term"), (kegg_efficacy, "kegg_efficacy")]:

        df = df.copy()
        df.name = "ontology_term"
        m = pd.merge(pharma, df.reset_index())

        sel = ['selectivity_score_after', 'selectivity_score_before']
        m2 = m[sel + ['ontology_term', 'drug']].drop_duplicates()
        m3 = m2.groupby("ontology_term")[sel].mean()
        cv3 = (m2.groupby("ontology_term")[sel].mean() / m2.groupby("ontology_term")[sel].std()) + 1
        c2 = m2.groupby("ontology_term")[sel[0]].count()
        c2.name = 'count'

        m_diff = (m3['selectivity_score_after'] - m3['selectivity_score_before'])
        m_diff.name = "change"
        cv_diff = np.log2(cv3['selectivity_score_after'] / cv3['selectivity_score_before'])
        cv_diff.name = "change"

        # m_diff.loc[c2[c2 > 1].index]

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axis = plt.subplots(1, 2, figsize=(2 * 4, 4))
        axis[0].set_title("Drug class reduction function: Mean")    
        cs = axis[0].scatter(np.log2(1 + m3.mean(axis=1)), m_diff, c=np.log10(c2), alpha=0.5)
        # divider = make_axes_locatable(axis[0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(mappable=cs, cax=cax, label="drugs per term (log10)")
        axis[0].axhline(0, linestyle="--", color="black", alpha=0.75)
        for t in m_diff.sort_values().dropna().head(10).index:
            axis[0].text(np.log2(1 + m3.mean(axis=1).loc[t]), m_diff.loc[t], s=t, ha="center", fontsize=5, zorder=100)
        for t in m_diff.sort_values().dropna().tail(15).index:
            axis[0].text(np.log2(1 + m3.mean(axis=1).loc[t]), m_diff.loc[t], s=t, ha="center", fontsize=5, zorder=100)

        axis[1].set_title("Drug class reduction function: Mean / Std")    
        cs = axis[1].scatter(np.log2(1 + cv3.mean(axis=1)), cv_diff, c=np.log10(c2), alpha=0.5)
        divider = make_axes_locatable(axis[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mappable=cs, cax=cax, label="drugs per term (log10)")
        axis[1].axhline(0, linestyle="--", color="black", alpha=0.75)
        for t in cv_diff.sort_values().dropna().head(10).index:
            axis[1].text(np.log2(1 + cv3.mean(axis=1).loc[t]), cv_diff.loc[t], s=t, ha="center", fontsize=5, zorder=100)
        for t in cv_diff.sort_values().dropna().tail(15).index:
            axis[1].text(np.log2(1 + cv3.mean(axis=1).loc[t]), cv_diff.loc[t], s=t, ha="center", fontsize=5, zorder=100)
        for ax in axis:
            ax.set_xlabel("Selectivity across timepoints")
        axis[0].set_ylabel("Difference in selectivity (during/before ibrutinib)")
        sns.despine(fig)
        fig.savefig(os.path.join("results", "drugs_annotated.{}.20180809.MA_cv.scatterplot.svg".format(label)), bbox_inches="tight", dpi=300)


        fig, axis = plt.subplots(1, 2, figsize=(2 * 4, 0.12 * m_diff.shape[0]))
        axis[0].set_title("Drug class reduction function: Mean")
        axis[1].set_title("Drug class reduction function: Mean/Std - only terms with >1 drug")
        p = m3.loc[m_diff.sort_values().index]
        sns.heatmap(
            p.join(m_diff),
            cmap="coolwarm", center=0, vmin=p.min().min(), vmax=p.max().max(),
            ax=axis[0], yticklabels=True, rasterized=True)
        p = np.log10(cv3.loc[cv_diff.sort_values().dropna().index].dropna())
        sns.heatmap(
            p.join(cv_diff),
            cmap="coolwarm", center=0, vmin=p.min().min(), vmax=p.max().max(),
            ax=axis[1], yticklabels=True, rasterized=True)
        for ax in axis:
            ax.set_ylabel("Ontology term (sorted by change)")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=3, va="center")
        fig.tight_layout()
        fig.savefig(os.path.join("results", "drugs_annotated.{}.20180809.MA_cv.heatmaps.svg".format(label)), bbox_inches="tight", dpi=300)


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

    # ordered
    p = path_cov_z.sort_index(level=['patient_id', 'timepoint_name'])
    g = sns.clustermap(p.reset_index(drop=True), figsize=(30, 8), yticklabels=p.index.get_level_values("sample_name"), col_cluster=True, row_cluster=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.ordered.svg"), bbox_inches="tight", dpi=300)

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
    fc = (a - b).sort_values()
    fc.name = "fold_change"

    # plot again but only top-bottom ordered by patient
    p = path_cov_z.sort_index(level=['patient_id', 'timepoint_name'])[fc.head(20).index.tolist() + fc.tail(20).index.tolist()]
    g = sns.clustermap(p.reset_index(drop=True), metric="correlation", square=True, figsize=(6, 6), yticklabels=p.index.get_level_values("sample_name"), col_cluster=True, row_cluster=True)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.ordered.top_bottom.all_patients.svg"), bbox_inches="tight", dpi=300)

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

    (changes[['fold_change']]
        .sort_values('fold_change', ascending=False)
        .to_csv(os.path.join("source_data", "fig2i.csv"), index=True))

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
    changes = pd.read_csv(os.path.join("https://biomedical-sequencing.at/bocklab/arendeiro/cll-ibrutinib/pathway.sample_accessibility.size-log2_fold_change-p_value.q_value.csv"), index_col=0)
    # changes = pd.read_csv(os.path.join("results", "pathway.sample_accessibility.size-log2_fold_change-p_value.q_value.csv"), index_col=0)
    # pharma_global = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"), index_col=0)
    # pharma_global = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"), index_col=0)
    # filter out drugs/pathways which after Ibrutinib are not more CLL-specific
    # pharma_global = pharma_global[pharma_global['after_Ibrutinib'] >= 0]

    # changes = pd.read_csv("http://www.medical-epigenomics.org/papers/schmidl2017/data/accessibility.pathway_space.csv", index_col=0)
    # changes = changes.T
    # changes['patient_id'] = list(map(lambda x: x[0], changes.index.str.split("_")))
    # changes['timepoint'] = list(map(lambda x: x[1], changes.index.str.split("_")))
    # changes = changes.groupby("timepoint").mean().T
    # changes['fold_change'] = changes['post'] - changes['pre']

    # pcy = pd.read_csv(os.path.join("metadata", "PCY_with_KEGG_andre_seperate.csv"))
    # pcy = pcy.drop(['kegg_pathway_id', 'kegg_pathway_name'], axis=1)
    pcy = pd.read_csv(os.path.join("source_data", "KEGG_annotate_per_drug_selectivity_score_13082018_full.csv"), index_col=0)
    pcy = pcy.drop(['kegg_pathway_name'], axis=1)

    # add new drug annotation
    rename = {
        "ABT-199 = Venetoclax": "Venetoclax",
        "ABT-263 = Navitoclax": "Navitoclax",
        "ABT-869 = Linifanib": "Linifanib",
        "AC220 = Quizartinib": "Quizartinib",
        "Buparlisib (BKM120)": "Buparlisib",
        "EGCG = Epigallocatechin gallate": "Epigallocatechin gallate",
        "JQ1": "JQ1",
        "MLN-518 = Tandutinib": "Tandutinib",
        "Selinexor (KPT-330)": "Selinexor"}
    pcy["proper_name"] = pcy["drug"]
    for p, n in rename.items():
        pcy["proper_name"] = pcy["proper_name"].replace(p, n)

    # annot = pd.read_csv(os.path.join("metadata", "drugs_annotated.20180725.csv"))
    # annot = pd.read_csv("https://biomedical-sequencing.at/bocklab/arendeiro/cll-ibrutinib/drugs_annotated.20180725.csv")
    # annot = annot[['drug', 'proper_name', 'kegg_pathway_name']].drop_duplicates().dropna()
    # annot['kegg_pathway_name'] = list(map(lambda x: x[0], annot['kegg_pathway_name'].str.split(" - ")))
    # annot = annot.drop_duplicates()

    # pcy2 = pd.merge(pcy, annot, on="proper_name", how="left")

    # mean = pcy2.groupby('kegg_pathway_name')["diff"].mean().sort_values()
    # std = pcy2.groupby('kegg_pathway_name')["diff"].std().sort_values()
    # pharma_change = (mean) # / std
    # # sign =  (pharma_change >= 0).astype(int).replace(0, -1)
    # # pharma_change = ((pharma_change.abs()) ** (1. / 3)) * sign
    # pharma_change = pharma_change.loc[~np.isinf(pharma_change)]
    # pharma_change.name = "pharmacoscopy"
    pharma_change = pcy.groupby('kegg_pathway_name')['diff'].mean()
    pharma_change.name = "pharmacoscopy"
    p = changes.join(pharma_change, how="outer")

    # collapse redundant pathways
    p.index = map(lambda x: x.split(" - ")[0], p.index)

    # and take maximum effect dependent on direction
    def max_effect(x):
        # print(x)
        x = x.dropna()
        sign = (x >= 0).astype(bool).astype(int).replace(0, -1).unique()
        if len(sign) == 1:
            return sign[0] * x.abs().max()
        else:
            return x.mean()

    p2 = p.groupby(level=0)['pharmacoscopy'].apply(max_effect).to_frame()
    p2['fold_change'] = p.groupby(level=0)['fold_change'].apply(max_effect)
    # keep only pathways with more than X drugs
    # above = pcy.groupby('kegg_pathway_name')['proper_name'].count()[(pcy.groupby('kegg_pathway_name')['proper_name'].nunique() >= 5)].index
    # above = map(lambda x: x.split(" - ")[0], above)
    # p = p.loc[above].dropna().drop_duplicates()

    # scatter
    n_to_label = 10
    custom_labels = p.index[p.index.str.contains("apoptosis|pi3k|proteasome|ribosome|nfkb", case=False)].tolist()
    custom_labels = p.index[p.index.str.contains("|".join([
        "Autophagy",
        "Apoptosis",
        "FoxO",
        "JAK-Stat signaling",
        "TGF-beta",
        "PI3K-Akt",
        "Transcriptional misregulation in cancer",
        "RIG-I-like receptor signaling pathway",
        "Inositol phosphate metabolism",
        "Mismatch repair",
        "PPAR signaling pathway",
        "DNA replication",
        "Base excision repair"
    ]), case=False)].tolist()

    import scipy
    p = p2.dropna().drop_duplicates().apply(scipy.stats.zscore)
    # p.to_csv(os.path.join("results", "atac-pharmacoscopy.new_annot.across_patients.combined_rank.20180726.csv"))
    combined_diff = p.mean(axis=1)

    # p.rename(columns={"fold_change": "atac-seq"}).to_csv(os.path.join("source_data", "fig4a.csv"), index=True)
    sd = combined_diff.to_frame(name="combined_change")
    sd.index.name = "pathway"
    # sd.to_csv(os.path.join("source_data", "figs4.csv"), index=True)

    m = max(abs(combined_diff.min()), abs(combined_diff.max()))
    normalizer = matplotlib.colors.Normalize(vmin=-m, vmax=m)

    for cmap in ["Spectral_r", "coolwarm", "BrBG"]:
        already = list()
        fig, axis = plt.subplots(1, 1, figsize=(4 * 1, 4 * 1))
        axis.scatter(p['fold_change'], p['pharmacoscopy'], alpha=0.8, color=plt.get_cmap(cmap)(normalizer(combined_diff.values)))
        axis.axhline(0, color="black", linestyle="--", alpha=0.5)
        axis.axvline(0, color="black", linestyle="--", alpha=0.5)
        # annotate top pathways
        # combined
        for path in [x for x in combined_diff.sort_values().head(n_to_label).index if x not in already]:
            axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="right")
            already.append(path)
        for path in [x for x in combined_diff.sort_values().tail(n_to_label).index if x not in already]:
            axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="left")
            already.append(path)
        # individual
        # for path in [x for x in p['fold_change'].sort_values().head(n_to_label).index if x not in already]:
        #     axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="right", color="orange")
        #     already.append(path)
        # for path in [x for x in p['pharmacoscopy'].sort_values().head(n_to_label).index if x not in already]:
        #     axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="left", color="green")
        #     already.append(path)
        # for path in [x for x in p['fold_change'].sort_values().tail(n_to_label).index if x not in already]:
        #     axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="left", color="orange")
        #     already.append(path)
        # for path in [x for x in p['pharmacoscopy'].sort_values().tail(n_to_label).index if x not in already]:
        #     axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="right", color="green")
        #     already.append(path)
        # custom
        for path in [x for x in custom_labels if x not in already]:
            if path in p.index:
                axis.text(p.loc[path, 'fold_change'], p.loc[path, 'pharmacoscopy'], path, ha="center", color="blue")
            already.append(path)
        # axis.set_ylim((-.2, .2))
        axis.set_xlabel("ATAC-seq change in pathway accessibility", ha="center")
        axis.set_ylabel("Pharmacoscopy change in pathway sensitivity", ha="center")
        sns.despine(fig)
        fig.savefig(os.path.join("results", "atac-pharmacoscopy.new_annot.across_patients.mean.scatter.{}.20180816.svg".format(cmap)), bbox_inches='tight', dpi=300)
        # fig.savefig(os.path.join("results", "atac-pharmacoscopy.new_annot.across_patients.mean.scatter.{}.20180726.svg".format(cmap)), bbox_inches='tight', dpi=300)
        # fig.savefig(os.path.join(analysis.results_dir, "atac-pharmacoscopy.across_patients.scatter.only_positive.svg"), bbox_inches='tight', dpi=300)


    combined_diff = combined_diff.sort_values(ascending=False)
    fig, axis = plt.subplots(1, 3, figsize=(3 * 4, 4))
    axis[0].axhline(0, linestyle="--", alpha=0.5, color="black")
    axis[0].scatter(combined_diff.rank(ascending=False), combined_diff, s=5, alpha=1, c=combined_diff, cmap="coolwarm")
    sns.barplot(combined_diff.head(10).index, combined_diff.head(10), ax=axis[1], palette="Reds_r")
    sns.barplot(combined_diff.tail(10).index, combined_diff.tail(10), ax=axis[2], palette="Blues")
    axis[0].set_xlabel("Combined ATAC-seq/Pharmacoscopy change (rank)", ha="center")
    axis[0].set_ylabel("Combined ATAC-seq/Pharmacoscopy change", ha="center")
    axis[1].set_xlabel("Pathway (top 10)", ha="center")
    axis[2].set_xlabel("Pathway (bottom 10)", ha="center")
    for ax in axis[1:]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel("Combined ATAC-seq/Pharmacoscopy change", ha="center")
    sns.despine(fig)
    fig.savefig(os.path.join("results", "atac-pharmacoscopy.new_annot.across_patients.combined_rank.20180816.svg"), bbox_inches='tight', dpi=300)



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


def predict(analysis):
    from sklearn.linear_model import ElasticNet

    # Load pharma
    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    auc = pd.read_csv(os.path.join("metadata", "pharmacoscopy_AUC_v3.csv")).dropna()
    auc["AUC"] *= -1  # transform AUC to inverse

    # transform to pivot
    sens_pivot = pd.pivot_table(data=sensitivity, index="drug", columns=['sample_id'], values="score")
    auc_pivot = pd.pivot_table(data=auc, index="drug", columns=['sample_id'], values="AUC")

    # Load ATAC
    analysis.accessibility

    # match ATAC and pharma
    matching_samples = [s for s in analysis.samples if s.sample_id in sensitivity['sample_id'].drop_duplicates().tolist()]
    accessibility = analysis.accessibility[[s.name for s in matching_samples]]
    sens_pivot = sens_pivot[[s.sample_id for s in matching_samples]]
    auc_pivot = auc_pivot[[s.sample_id for s in matching_samples]]

    # for score, AUC
    # for absolute and timepoint difference
    # for each drug fit, predict with CV

    X_train = accessibility.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    Y_train = sens_pivot.ix[sens_pivot.index[0]]

    lm = ElasticNet(precompute=True, l1_ratio=0.5, alpha=0.0005, max_iter=10000, )
    lm.fit(X_train.T, Y_train)


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
            # "OMIM_Expanded",
            # "TF-LOF_Expression_from_GEO",
            # "Single_Gene_Perturbations_from_GEO_down",
            # "Single_Gene_Perturbations_from_GEO_up",
            # "Disease_Perturbations_from_GEO_down",
            # "Disease_Perturbations_from_GEO_up",
            # "Drug_Perturbations_from_GEO_down",
            # "Drug_Perturbations_from_GEO_up",
            "WikiPathways_2016",
            "Reactome_2016",
            "BioCarta_2016",
            "NCI-Nature_2016",
            "Panther_2016"
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


def characterize_cohort(analysis):
    df = analysis.prj.sheet.df
    df = df[df["patient_id"] != "CLL16"]

    order = df.groupby(['patient_id'])['time_since_treatment'].min().rank()
    order.name = "patient_change_order"
    order = order.to_frame()
    df = pd.merge(df, order.reset_index()).sort_values(["patient_change_order", "timepoint_name"])

    # stripplot of timecourse
    g = sns.PairGrid(data=df, x_vars="time_since_treatment", y_vars=["patient_id"], hue="timepoint_name", size=6, aspect=.5)
    g.axes[0][0].axvline(0, linestyle="--", color="black", alpha=0.8)
    # g.axes[0][0].set_xscale("log")
    g.map(sns.stripplot, size=10, orient="h")
    g.savefig(os.path.join(analysis.results_dir, "{}.cohort_desription.min_time_sorted.stripplot.svg".format(analysis.name)), bbox_inches="tight")

    # heatmap of patient attributes
    attributes = [
        "patient_id", "timepoint_name", "patient_gender", "patient_age_at_collection",
        "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
        "del11q", "del13q", "del17p", "tri12", "p53",
        "time_since_treatment", "treatment_response"]

    df2 = df[df["timepoint_name"] == "before_Ibrutinib"].drop_duplicates()

    color_dataframe = pd.DataFrame(
        analysis.get_level_colors(index=df2.set_index(attributes).index),
        index=attributes,
        columns=df2['patient_id'])

    fig, axis = plt.subplots(1, 1, figsize=(6, 6))
    for i, patient in enumerate(color_dataframe.index):
        for j, attr in enumerate(color_dataframe.columns):
            axis.scatter(i, j, color=color_dataframe.loc[patient, attr], s=175, marker="s")
    axis.set_yticks(range(len(color_dataframe.columns)))
    axis.set_yticklabels(color_dataframe.columns, rotation=0, ha="right")
    axis.set_xticks(range(len(color_dataframe.index)))
    axis.set_xticklabels(color_dataframe.index, rotation=90, ha="right")
    sns.despine(fig, top=True, bottom=True, left=True, right=True)
    fig.savefig(os.path.join(analysis.results_dir, "{}.cohort_desription.min_time_sorted.heatmap.svg".format(analysis.name)), bbox_inches="tight")


class Analysis(object):
    """
    Class to hold functions and data from analysis.
    """
    from ngs_toolkit.general import pickle_me

    def __init__(
            self,
            name="analysis",
            samples=None,
            prj=None,
            data_dir="data",
            results_dir="results",
            pickle_file=None,
            from_pickle=False,
            **kwargs):
        # parse kwargs with default
        self.name = name
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.samples = samples
        self.prj = prj
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
            self.update()

    @pickle_me
    def to_pickle(self):
        pass

    def from_pickle(self):
        import cPickle as pickle
        return pickle.load(open(self.pickle_file, 'rb'))

    def update(self):
        self.__dict__.update(self.from_pickle().__dict__)


class RNASeqAnalysis(Analysis):
    """
    Class to hold functions and data from analysis.
    """
    def __init__(
            self,
            name="analysis",
            samples=None,
            prj=None,
            data_dir="data",
            results_dir="results",
            pickle_file=None,
            from_pickle=False,
            **kwargs):
        super(RNASeqAnalysis, self).__init__(
            name=name,
            data_dir=data_dir,
            results_dir=results_dir,
            pickle_file=pickle_file,
            samples=samples,
            prj=prj,
            from_pickle=from_pickle,
            **kwargs)

    def annotate_with_sample_metadata(
            self,
            attributes=["sample_name", "cell_line", "knockout", "replicate", "clone"]):

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

    def collect_bitseq_output(self, samples=None):
        """
        Collect gene expression output from Bitseq into expression matrix for `samples`.
        """
        if samples is None:
            samples = self.samples

        first = True
        for i, sample in enumerate(samples):
            if first:
                try:
                    # read the "tr" file of one sample to get indexes
                    tr = pd.read_csv(
                        os.path.join(
                            sample.paths.sample_root, "bowtie1_{}".format(sample.transcriptome),
                            "bitSeq",
                            sample.name + ".tr"),
                        sep=" ", header=None, skiprows=1,
                        names=["ensembl_gene_id", "ensembl_transcript_id", "v1", "v2"])
                except IOError:
                    print("Sample {} is missing.".format(sample.name))
                    continue
                # add id index
                tr.set_index("ensembl_gene_id", append=False, inplace=True)
                tr.set_index("ensembl_transcript_id", append=True, inplace=True)
                # create dataframe
                expr = pd.DataFrame(index=tr.index)
                first = False

            # load counts
            try:
                counts = pd.read_csv(os.path.join(
                    sample.paths.sample_root, "bowtie1_{}".format(sample.transcriptome),
                    "bitSeq",
                    sample.name + ".counts"), sep=" ")
            except IOError:
                print("Sample {} is missing.".format(sample.name))
                continue
            counts.index = expr.index

            # Append
            expr[sample.name] = counts

        return expr

    def get_gene_expression(self, samples=None, attributes=["sample_name", "cell_line", "knockout", "replicate", "clone", "complex_subunit"]):
        """
        """
        def quantile_normalize(df_input):
            df = df_input.copy()
            # compute rank
            dic = {}
            for col in df:
                dic.update({col: sorted(df[col])})
            sorted_df = pd.DataFrame(dic)
            rank = sorted_df.mean(axis=1).tolist()
            # sort
            for col in df:
                t = np.searchsorted(np.sort(df[col]), df[col])
                df[col] = [rank[i] for i in t]
            return df

        if samples is None:
            samples = [s for s in self.samples if s.library == "RNA-seq"]

        self.count_matrix = self.collect_bitseq_output(samples)

        # Map ensembl gene IDs to gene names
        import requests
        url_query = "".join([
            """http://grch37.ensembl.org/biomart/martservice?query=""",
            """<?xml version="1.0" encoding="UTF-8"?>""",
            """<!DOCTYPE Query>""",
            """<Query  virtualSchemaName = "default" formatter = "CSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >""",
            """<Dataset name = "hsapiens_gene_ensembl" interface = "default" >""",
            """<Attribute name = "ensembl_gene_id" />""",
            """<Attribute name = "external_gene_name" />""",
            """</Dataset>""",
            """</Query>"""])
        req = requests.get(url_query, stream=True)
        mapping = pd.DataFrame((x.strip().split(",") for x in list(req.iter_lines())), columns=["ensembl_gene_id", "gene_name"])
        self.count_matrix = self.count_matrix.reset_index()
        self.count_matrix['ensembl_gene_id'] = self.count_matrix['ensembl_gene_id'].str.replace("\..*", "")

        self.count_matrix = pd.merge(self.count_matrix, mapping, how="outer").dropna()
        self.count_matrix.to_csv(os.path.join(self.results_dir, self.name + ".expression_counts.csv"), index=False)

        # Quantile normalize
        self.matrix_qnorm = quantile_normalize(self.count_matrix.drop(["ensembl_transcript_id", "ensembl_gene_id", "gene_name"], axis=1))

        # Log2 TPM
        self.matrix_qnorm_log = np.log2((1 + self.matrix_qnorm) / self.matrix_qnorm.sum(axis=0) * 1e6)
        self.matrix_qnorm_log = self.count_matrix[["gene_name", "ensembl_gene_id", "ensembl_transcript_id"]].join(self.matrix_qnorm_log)
        self.matrix_qnorm_log.to_csv(os.path.join(self.results_dir, self.name + ".expression_counts.transcript_level.quantile_normalized.log2_tpm.csv"), index=False)

        # Reduce to gene-level measurements by max of transcripts
        self.expression_matrix_counts = self.count_matrix.drop(['ensembl_transcript_id'], axis=1).dropna().set_index(['ensembl_gene_id', 'gene_name']).groupby(level=["gene_name"]).max()

        # Quantile normalize
        self.expression_matrix_qnorm = quantile_normalize(self.expression_matrix_counts)

        # Log2 TPM
        self.expression = np.log2((1 + self.expression_matrix_qnorm) / self.expression_matrix_qnorm.sum(axis=0) * 1e6)

        # Annotate with sample metadata
        _samples = [s for s in samples if s.name in self.expression.columns]
        attrs = list()
        for attr in attributes:
            l = list()
            for sample in _samples:  # keep order of samples in matrix
                try:
                    l.append(getattr(sample, attr))
                except AttributeError:
                    l.append(np.nan)
            attrs.append(l)

        # Generate multiindex columns
        index = pd.MultiIndex.from_arrays(attrs, names=attributes)
        self.expression_annotated = self.expression[[s.name for s in _samples]]
        self.expression_annotated.columns = index

        # Save
        self.expression_matrix_counts.to_csv(os.path.join(self.results_dir, self.name + ".expression_counts.gene_level.csv"), index=True)
        self.expression_matrix_qnorm.to_csv(os.path.join(self.results_dir, self.name + ".expression_counts.gene_level.quantile_normalized.csv"), index=True)
        self.expression.to_csv(os.path.join(self.results_dir, self.name + ".expression_counts.gene_level.quantile_normalized.log2_tpm.csv"), index=True)
        self.expression_annotated.to_csv(os.path.join(self.results_dir, self.name + ".expression_counts.gene_level.quantile_normalized.log2_tpm.annotated_metadata.csv"), index=True)

    def plot_expression_characteristics(self):
        fig, axis = plt.subplots(figsize=(4, 4 * np.log10(len(self.expression_matrix_counts.columns))))
        sns.barplot(data=self.expression_matrix_counts.sum().sort_values().reset_index(), y="index", x=0, orient="horiz", color=sns.color_palette("colorblind")[0], ax=axis)
        axis.set_xlabel("Reads")
        axis.set_ylabel("Samples")
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "expression.reads_per_sample.svg"), bbox_inches="tight")

        cov = pd.DataFrame()
        for i in [1, 2, 3, 6, 12, 24, 48, 96, 200, 300, 400, 500, 1000]:
            cov[i] = self.expression_matrix_counts.apply(lambda x: sum(x >= i))

        fig, axis = plt.subplots(1, 2, figsize=(6 * 2, 6))
        sns.heatmap(cov.drop([1, 2], axis=1), ax=axis[0], cmap="GnBu", cbar_kws={"label": "Genes covered"})
        sns.heatmap(cov.drop([1, 2], axis=1).apply(lambda x: (x - x.mean()) / x.std(), axis=0), ax=axis[1], cbar_kws={"label": "Z-score"})
        for ax in axis:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_ylabel("Samples")
            ax.set_xlabel("Genes with #reads")
        axis[1].set_yticklabels(ax.get_yticklabels(), visible=False)
        sns.despine(fig)
        fig.savefig(os.path.join(self.results_dir, "expression.genes_with_reads.svg"), bbox_inches="tight")

        for name, matrix in [("counts", self.expression_matrix_counts), ("qnorm_TPM", self.expression)]:
            # Boxplot with values per sample
            to_plot = pd.melt(matrix, var_name="sample")

            fig, axis = plt.subplots()
            sns.boxplot(data=to_plot, y="sample", x="value", orient="horiz", ax=axis)
            axis.set_xlabel(name)
            axis.set_ylabel("Samples")
            sns.despine(fig)
            fig.savefig(os.path.join(self.results_dir, "expression.boxplot_per_sample.{}.svg".format(name)), bbox_inches="tight")

        # Plot gene expression along chromossomes
        import requests
        url_query = "".join([
            """http://grch37.ensembl.org/biomart/martservice?query=""",
            """<?xml version="1.0" encoding="UTF-8"?>""",
            """<!DOCTYPE Query>""",
            """<Query  virtualSchemaName = "default" formatter = "CSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >""",
            """<Dataset name = "hsapiens_gene_ensembl" interface = "default" >""",
            """<Attribute name = "chromosome_name" />""",
            """<Attribute name = "start_position" />""",
            """<Attribute name = "end_position" />""",
            """<Attribute name = "external_gene_name" />""",
            """</Dataset>""",
            """</Query>"""])
        req = requests.get(url_query, stream=True)
        annot = pd.DataFrame((x.strip().split(",") for x in list(req.iter_lines())), columns=["chr", "start", "end", "gene_name"])
        gene_order = annot[annot['chr'].isin([str(x) for x in range(22)] + ['X', 'Y'])].sort_values(["chr", "start", "end"])["gene_name"]

        exp = self.expression.ix[gene_order].dropna()
        data = exp.rolling(int(3e4), axis=0).mean().dropna()
        plt.pcolor(data)

    def differential_expression_analysis(
            self, samples, trait="timepoint_name",
            variables=["atac_seq_batch", "timepoint_name"],
            output_suffix="ibrutinib_treatment_expression"):
        """
        Discover differential regions across samples that are associated with a certain trait.
        """
        from ngs_toolkit.general import deseq_analysis
        sel_samples = [s for s in samples if not pd.isnull(getattr(s, trait))]

        # Get matrix of counts
        counts_matrix = self.expression_matrix_counts[[s.name for s in sel_samples]]

        # Get experiment matrix
        experiment_matrix = pd.DataFrame([sample.as_series() for sample in sel_samples], index=[sample.name for sample in sel_samples])
        # keep only variables
        experiment_matrix = experiment_matrix[["sample_name"] + variables].fillna("Unknown")

        # Make output dir
        output_dir = os.path.join(self.results_dir, output_suffix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Run DESeq2 analysis
        deseq_table = deseq_analysis(
            counts_matrix, experiment_matrix, trait, covariates=[x for x in variables if x != trait], output_prefix=os.path.join(output_dir, output_suffix), alpha=0.05)

        # to just read in
        # deseq_table = pd.read_csv(os.path.join(output_dir, output_suffix) + ".%s.csv" % trait, index_col=0)

        df = self.expression.join(deseq_table)
        df.to_csv(os.path.join(output_dir, output_suffix) + ".%s.annotated.csv" % trait)
        df = pd.read_csv(os.path.join(output_dir, output_suffix) + ".%s.annotated.csv" % trait, index_col=0)

        # Extract significant based on p-value and fold-change
        # diff = df.loc[df.sort_values(["log2FoldChange"]).head(200).index.tolist() + df.sort_values(["log2FoldChange"]).tail(200).index.tolist(), :]
        diff = df[(df["pvalue"] < 0.01) & (abs(df["log2FoldChange"]) > np.log2(1))]

        if diff.shape[0] < 1:
            print("No significantly different regions found.")
            return

        # groups = list(set([getattr(s, trait) for s in sel_samples]))
        comparisons = pd.Series(df['comparison'].unique())
        groups = sorted(set([y for y in x.split("-") for x in comparisons.tolist()]))

        # Statistics of differential regions
        import string
        total_sites = float(self.expression.shape[0])

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

        # in same
        split_diff = diff.groupby(["comparison", "direction"])['stat'].count().sort_values(ascending=False)

        split_diff = split_diff.reset_index()
        split_diff.loc[split_diff['direction'] == "down", "stat"] = -split_diff.loc[split_diff['direction'] == "down", "stat"]
        fig, axis = plt.subplots(1, figsize=(6, 3))
        sns.barplot(
            x="comparison", y="stat", data=split_diff, hue="direction",  # split=False,
            order=split_diff.groupby(['comparison'])['stat'].apply(lambda x: sum(abs(x))).sort_values(ascending=False).index,
            orient="v", ax=axis)
        axis.axhline(0, color="black", linestyle="--")
        axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.split.pos_neg.svg" % (output_suffix, trait)), bbox_inches="tight")

        # percentage of total
        split_diff['stat'] = (split_diff['stat'].values / total_sites) * 100
        fig, axis = plt.subplots(1, figsize=(6, 3))
        sns.barplot(
            x="comparison", y="stat", data=split_diff, hue="direction",  # split=False,
            order=split_diff.groupby(['comparison'])['stat'].apply(lambda x: sum(abs(x))).sort_values(ascending=False).index,
            orient="v", ax=axis)
        axis.axhline(0, color="black", linestyle="--")
        axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.split_percentage.pos_neg.svg" % (output_suffix, trait)), bbox_inches="tight")

        # Overlap between sets
        # # Pyupset
        # import pyupset as pyu
        # # Build dict
        # diff["comparison_direction"] = diff[["comparison", "direction"]].apply(string.join, axis=1)
        # df_dict = {group: diff[diff["comparison_direction"] == group].reset_index()[['index']] for group in set(diff["comparison_direction"])}
        # # Plot
        # plot = pyu.plot(df_dict, unique_keys=['index'], inters_size_bounds=(10, np.inf))
        # plot['figure'].set_size_inches(20, 8)
        # plot['figure'].savefig(os.path.join(output_dir, "%s.%s.number_differential.upset.svg" % (output_suffix, trait)), bbox_inched="tight")
        diff["intersect"] = 1
        diff["direction"] = diff["log2FoldChange"].apply(lambda x: "up" if x >= 0 else "down")
        piv = pd.pivot_table(diff.reset_index(), index='gene_name', columns=['comparison', 'direction'], values='intersect', fill_value=0)

        import itertools
        intersections = pd.DataFrame(columns=["group1", "group2", "dir1", "dir2", "size1", "size2", "intersection", "union"])
        for ((k1, dir1), i1), ((k2, dir2), i2) in itertools.permutations(piv.T.groupby(level=['comparison', 'direction']).groups.items(), 2):
            print(k1, k2)
            i1 = set(piv[i1][piv[i1] == 1].dropna().index)
            i2 = set(piv[i2][piv[i2] == 1].dropna().index)
            intersections = intersections.append(
                pd.Series(
                    [k1, k2, dir1, dir2, len(i1), len(i2), len(i1.intersection(i2)), len(i1.union(i2))],
                    index=["group1", "group2", "dir1", "dir2", "size1", "size2", "intersection", "union"]
                ),
                ignore_index=True
            )
        # convert to %
        intersections['intersection_perc'] = (intersections['intersection'] / intersections['size2']) * 100.

        # make pivot tables
        piv_up = pd.pivot_table(intersections[(intersections['dir1'] == "up") & (intersections['dir2'] == "up")], index="group1", columns="group2", values="intersection_perc").fillna(0)
        piv_down = pd.pivot_table(intersections[(intersections['dir1'] == "down") & (intersections['dir2'] == "down")], index="group1", columns="group2", values="intersection_perc").fillna(0)
        np.fill_diagonal(piv_up.values, np.nan)
        np.fill_diagonal(piv_down.values, np.nan)

        # heatmaps
        fig, axis = plt.subplots(1, 2, figsize=(8 * 2, 8))
        sns.heatmap(piv_down, square=True, cmap="summer", cbar_kws={"label": "Concordant regions (% of group 2)"}, ax=axis[0])
        sns.heatmap(piv_up, square=True, cmap="summer", cbar_kws={"label": "Concordant regions (% of group 2)"}, ax=axis[1])
        axis[0].set_title("Downregulated regions")
        axis[1].set_title("Upregulated regions")
        axis[0].set_xticklabels(axis[0].get_xticklabels(), rotation=90, ha="right")
        axis[1].set_xticklabels(axis[1].get_xticklabels(), rotation=90, ha="right")
        axis[0].set_yticklabels(axis[0].get_yticklabels(), rotation=0, ha="right")
        axis[1].set_yticklabels(axis[1].get_yticklabels(), rotation=0, ha="right")
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.overlap.up_down_split.svg" % (output_suffix, trait)), bbox_inches="tight")

        # combined heatmap
        # with upregulated regions in upper square matrix and downredulated in down square
        piv_combined = pd.DataFrame(np.triu(piv_up), index=piv_up.index, columns=piv_up.columns).replace(0, np.nan)
        piv_combined.update(pd.DataFrame(np.tril(piv_down), index=piv_down.index, columns=piv_down.columns).replace(0, np.nan))
        piv_combined = piv_combined.fillna(0)
        np.fill_diagonal(piv_combined.values, np.nan)

        fig, axis = plt.subplots(1, figsize=(8, 8))
        sns.heatmap(piv_combined, square=True, cmap="summer", cbar_kws={"label": "Concordant regions (% of group 2)"}, ax=axis)
        axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
        axis.set_yticklabels(axis.get_yticklabels(), rotation=0, ha="right")
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.overlap.up_down_together.svg" % (output_suffix, trait)), bbox_inches="tight")

        # Observe disagreement between knockouts
        # (overlap of down-regulated with up-regulated and vice-versa)
        piv_up = pd.pivot_table(intersections[(intersections['dir1'] == "up") & (intersections['dir2'] == "down")], index="group1", columns="group2", values="intersection")
        piv_down = pd.pivot_table(intersections[(intersections['dir1'] == "down") & (intersections['dir2'] == "up")], index="group1", columns="group2", values="intersection")

        piv_disagree = pd.concat([piv_up, piv_down]).groupby(level=0).max()

        fig, axis = plt.subplots(1, 2, figsize=(16, 8))
        sns.heatmap(piv_disagree, square=True, cmap="Greens", cbar_kws={"label": "Discordant regions"}, ax=axis[0])
        sns.heatmap(np.log2(1 + piv_disagree), square=True, cmap="Greens", cbar_kws={"label": "Discordant regions (log2)"}, ax=axis[1])

        norm = matplotlib.colors.Normalize(vmin=0, vmax=piv_disagree.max().max())
        cmap = plt.get_cmap("Greens")
        log_norm = matplotlib.colors.Normalize(vmin=0, vmax=np.log2(1 + piv_disagree).max().max())
        for i, g1 in enumerate(piv_disagree.columns):
            for j, g2 in enumerate(piv_disagree.index):
                axis[0].scatter(
                    len(piv_disagree.index) - (j + 0.5), len(piv_disagree.index) - (i + 0.5),
                    s=(100 ** (norm(piv_disagree.loc[g1, g2]))) - 1, color=cmap(norm(piv_disagree.loc[g1, g2])), marker="o")
                axis[1].scatter(
                    len(piv_disagree.index) - (j + 0.5), len(piv_disagree.index) - (i + 0.5),
                    s=(100 ** (log_norm(np.log2(1 + piv_disagree).loc[g1, g2]))) - 1, color=cmap(log_norm(np.log2(1 + piv_disagree).loc[g1, g2])), marker="o")

        for ax in axis:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
        fig.savefig(os.path.join(output_dir, "%s.%s.number_differential.overlap.disagreement.svg" % (output_suffix, trait)), bbox_inches="tight")

        # Pairwise scatter plots
        cond2 = "after_Ibrutinib"
        n_rows = n_cols = int(np.ceil(np.sqrt(len(comparisons))))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
        if n_rows > 1 or n_cols > 1:
            axes = iter(axes.flatten())
        else:
            axes = iter([axes])
        for cond1 in sorted(groups):
            if cond1 == cond2:
                continue
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1)].squeeze()
            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            df2 = df[df["comparison"] == comparison]
            if df2.shape[0] == 0:
                continue
            axis = axes.next()

            # Hexbin plot
            axis.hexbin(np.log2(1 + df2[cond1]), np.log2(1 + df2[cond2]), alpha=1, color="black", edgecolors="white", linewidths=0, bins='log', mincnt=1, rasterized=True)
            axis.set_xlabel(cond1)

            diff2 = diff[diff["comparison"] == comparison]
            if diff2.shape[0] > 0:
                # Scatter plot
                axis.scatter(np.log2(1 + diff2[cond1]), np.log2(1 + diff2[cond2]), alpha=0.5, color="red", s=2)
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
        for cond1 in sorted(groups):
            if cond1 == cond2:
                continue
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1)].squeeze()
            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            df2 = df[df["comparison"] == comparison]
            if df2.shape[0] == 0:
                continue
            axis = axes.next()

            # hexbin
            axis.hexbin(df2["log2FoldChange"], -np.log10(df2['pvalue']), alpha=1, color="black", edgecolors="white", linewidths=0, bins='log', mincnt=1, rasterized=True)

            diff2 = diff[diff["comparison"] == comparison]
            if diff2.shape[0] > 0:
                # significant scatter
                axis.scatter(diff2["log2FoldChange"], -np.log10(diff2['pvalue']), alpha=0.5, color="red", s=2)
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
        for cond1 in sorted(groups):
            if cond1 == cond2:
                continue
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1)].squeeze()
            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            df2 = df[df["comparison"] == comparison]
            if df2.shape[0] == 0:
                continue
            axis = axes.next()

            # hexbin
            axis.hexbin(np.log2(df2["baseMean"]), df2["log2FoldChange"], alpha=1, color="black", edgecolors="white", linewidths=0, bins='log', mincnt=1, rasterized=True)

            diff2 = diff[diff["comparison"] == comparison]
            if diff2.shape[0] > 0:
                # significant scatter
                axis.scatter(np.log2(diff2["baseMean"]), diff2["log2FoldChange"], alpha=0.5, color="red", s=2)
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
        # get unique differential regions
        df2 = diff2.join(self.expression)

        # Characterize regions
        prefix = "%s.%s.diff_regions" % (output_suffix, trait)
        # region's structure
        # characterize_regions_structure(df=df2, prefix=prefix, output_dir=output_dir)
        # region's function
        # characterize_regions_function(df=df2, prefix=prefix, output_dir=output_dir)

        # Heatmaps
        # Comparison level
        g = sns.clustermap(np.log2(1 + df2[groups]).corr(), xticklabels=False, cbar_kws={"label": "Pearson correlation\non differential genes"}, cmap="Spectral_r", metric="correlation", rasterized=True)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.groups.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300, metric="correlation")

        g = sns.clustermap(np.log2(1 + df2[groups]), yticklabels=False, cbar_kws={"label": "Expression of\ndifferential genes"}, cmap="BuGn", metric="correlation", rasterized=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.groups.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(np.log2(1 + df2[groups]), yticklabels=False, z_score=0, cbar_kws={"label": "Z-score of expression\non differential genes"}, metric="correlation", rasterized=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.groups.clustermap.z0.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # Fold-changes and P-values
        # pivot table of genes vs comparisons
        sig = diff.index.drop_duplicates()
        fold_changes = pd.pivot_table(df.reset_index(), index="gene_name", columns="comparison", values="log2FoldChange")
        p_values = -np.log10(pd.pivot_table(df.reset_index(), index="gene_name", columns="comparison", values="padj"))

        # fold
        if fold_changes.shape[1] > 1:
            g = sns.clustermap(fold_changes.corr(), xticklabels=False, cbar_kws={"label": "Pearson correlation\non fold-changes"}, cmap="Spectral_r", vmin=0, vmax=1, metric="correlation", rasterized=True)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.groups.fold_changes.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300, metric="correlation")

            g = sns.clustermap(fold_changes.ix[sig], yticklabels=False, cbar_kws={"label": "Fold-changes of\ndifferential genes"}, vmin=-4, vmax=4, metric="correlation", rasterized=True)
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
            g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.groups.fold_changes.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # Sample level
        g = sns.clustermap(df2[[s.name for s in sel_samples]].corr(), xticklabels=False, cbar_kws={"label": "Pearson correlation\non differential genes"}, cmap="Spectral_r", metric="correlation", rasterized=True)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.samples.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(df2[[s.name for s in sel_samples]], yticklabels=False, cbar_kws={"label": "Expression of\ndifferential genes"}, cmap="BuGn", vmin=0, metric="correlation", rasterized=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.samples.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(df2[[s.name for s in sel_samples]], yticklabels=False, z_score=0, cbar_kws={"label": "Z-score of expression\non differential genes"}, metric="correlation", rasterized=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.samples.clustermap.z0.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # ordered by patient
        sel_samples = sorted(sel_samples, key=lambda s: (s.patient_id, s.timepoint_name), reverse=True)

        g = sns.clustermap(df2[[s.name for s in sel_samples]], yticklabels=False, col_cluster=False, z_score=0, cbar_kws={"label": "Z-score of expression\non differential genes"}, metric="correlation", rasterized=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.samples.clustermap.z0.sorted_patients.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # ordered by strnegth of down/upregulation
        g = sns.clustermap(df2[df2.mean().sort_values().index[:-2]], yticklabels=False, col_cluster=False, z_score=0, cbar_kws={"label": "Z-score of expression\non differential genes"}, metric="correlation", rasterized=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_genes.samples.clustermap.z0.sorted_signature.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        # Get enrichments for the top genes in each side
        diff = df.loc[df.sort_values(["log2FoldChange"]).head(200).index.tolist() + df.sort_values(["log2FoldChange"]).tail(200).index.tolist(), :]

        max_diff = 1000
        pathway_enr = pd.DataFrame()
        for cond1 in sorted(groups):
            # get comparison
            comparison = comparisons[comparisons.str.contains(cond1)].squeeze()

            if type(comparison) is pd.Series:
                if len(comparison) > 1:
                    comparison = comparison.iloc[0]

            # Separate in up/down-regulated genes
            for f, direction, top in [(np.less, "down", "head"), (np.greater, "up", "tail")]:
                comparison_df = self.expression.ix[diff[
                    (diff["comparison"] == comparison) &
                    (f(diff["log2FoldChange"], 0))
                ].index]

                # Handle extremes of regions
                if comparison_df.shape[0] < 1:
                    continue

                if comparison_df.shape[0] > max_diff:
                    comparison_df = comparison_df.ix[
                        getattr(
                            diff[
                                (diff["comparison"] == comparison) &
                                (f(diff["log2FoldChange"], 0))]
                            ["log2FoldChange"].sort_values(), top)
                        (max_diff).index]

                # Characterize regions
                prefix = "%s.%s.diff_regions.comparison_%s.%s" % (output_suffix, trait, comparison, direction)

                comparison_dir = os.path.join(output_dir, prefix)
                if not os.path.exists(comparison_dir):
                    os.makedirs(comparison_dir)

                print("Doing regions of comparison %s, with prefix %s" % (comparison, prefix))
                if not os.path.exists(os.path.join(comparison_dir, "enraichr.csv")):
                    comparison_df.reset_index()[['gene_name']].drop_duplicates().to_csv(os.path.join(comparison_dir, "genes.txt"), index=False)
                    enr = enrichr(comparison_df.reset_index())
                    enr.to_csv(os.path.join(comparison_dir, "enrichr.csv"), index=False, encoding="utf8")

                enr = pd.read_csv(os.path.join(comparison_dir, "enrichr.csv"))
                enr["comparison"] = prefix
                pathway_enr = pathway_enr.append(enr, ignore_index=True)

        # write combined enrichments
        pathway_enr.to_csv(
            os.path.join(output_dir, "%s.%s.diff_genes.enrichr.csv" % (output_suffix, trait)), index=False, encoding="utf8")

    def investigate_differential_genes(
            self, samples, trait="timepoint_name",
            variables=["atac_seq_batch", "timepoint_name"],
            output_suffix="ibrutinib_treatment_expression", n=25, method="groups"):
        output_dir = os.path.join(self.results_dir, output_suffix)

        # ENRICHR
        # read in
        pathway_enr = pd.read_csv(os.path.join(output_dir, "%s.%s.diff_genes.enrichr.csv" % (output_suffix, trait)))

        # pretty names
        # pathway_enr["comparison"] = map(lambda x: x[-1], pathway_enr["comparison"].str.split("."))
        pathway_enr["description"] = pathway_enr["description"].str.decode("utf-8")

        for comparison in pathway_enr["comparison"].unique():
            for gene_set_library in pathway_enr["gene_set_library"].unique():
                print(gene_set_library)
                if gene_set_library == "Epigenomics_Roadmap_HM_ChIP-seq":
                    continue

                comparison_dir = os.path.join(output_dir, comparison)

                d = pathway_enr[
                    (pathway_enr["comparison"] == comparison) &
                    (pathway_enr["gene_set_library"] == gene_set_library)]

                # if d.shape[0] == 0:
                #     continue

                # transform p-values
                d["p_value"] = (-np.log10(d['p_value'])).replace({np.inf: 300})

                # show a maximum of N terms
                d = d.sort_values("combined_score", ascending=False).head(n)

                # plot top N terms
                fig, axis = plt.subplots(1)
                sns.barplot(d['combined_score'], d['description'], orient="horizontal", color=sns.color_palette("colorblind")[0])
                axis.set_xlabel("Combined score")
                sns.despine(fig)
                fig.savefig(os.path.join(comparison_dir, "enrichr.%s.cluster_specific.top.comb_score.svg" % gene_set_library), bbox_inches="tight", dpi=300)

                d = d.sort_values("p_value", ascending=False)
                fig, axis = plt.subplots(1)
                sns.barplot(d['p_value'], d['description'], orient="horizontal", color=sns.color_palette("colorblind")[0])
                axis.set_xlabel("-log10(p-value)")
                sns.despine(fig)
                fig.savefig(os.path.join(comparison_dir, "enrichr.%s.cluster_specific.top.p_value.svg" % gene_set_library), bbox_inches="tight", dpi=300)


def rna_to_pathway(rna, samples=None):
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
        samples = rna.samples

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

    # Get expression for each regulatory element assigned to each gene in each pathway
    # Reduce values per gene
    # Reduce values per pathway
    pc_samples = [s for s in samples if hasattr(s, "pharmacoscopy_id")]
    expr = rna.expression[[s.name for s in pc_samples]]
    path_expr = pd.DataFrame()
    path_expr.columns.name = 'kegg_pathway_name'
    for pathway in pathway_genes.keys():
        print(pathway)
        # mean of all reg. elements of all genes
        index = expr.index[expr.index.isin(pathway_genes[pathway])]
        q = expr.ix[index]
        path_expr[pathway] = (q[[s.name for s in pc_samples]].mean(axis=0) / expr[[s.name for s in pc_samples]].sum(axis=0)) * 1e6
    path_expr = path_expr.T.dropna().T

    path_expr_z = path_expr.apply(z_score, axis=0)
    path_expr.T.to_csv(os.path.join(rna.results_dir, "pathway.sample_expression.csv"))
    path_expr_z.T.to_csv(os.path.join(rna.results_dir, "pathway.sample_expression.z_score.csv"))
    path_expr = pd.read_csv(os.path.join(rna.results_dir, "pathway.sample_expression.csv"), index_col=0).T
    path_expr_z = pd.read_csv(os.path.join(rna.results_dir, "pathway.sample_expression.z_score.csv"), index_col=0).T

    # Get null distribution from permutations
    n_perm = 100
    pathway_sizes = pd.Series({k: len(v) for k, v in pathway_genes.items()}).sort_values()
    pathway_sizes.name = "pathway_size"
    all_genes = [j for i in pathway_genes.values() for j in i]

    r_fcs = list()
    for i in range(n_perm):
        r_path_expr = pd.DataFrame()
        r_path_expr.columns.name = 'kegg_pathway_name'
        for pathway in pathway_genes.keys():
            print(i, pathway)
            # choose same number of random genes from all pathways
            r = np.random.choice(all_genes, pathway_sizes.ix[pathway])
            index = expr.index[expr.index.isin(r)]
            q = expr.ix[index]
            r_path_expr[pathway] = (q[[s.name for s in samples]].mean(axis=0) / expr[[s.name for s in samples]].sum(axis=0)) * 1e6

        r_path_expr = r_path_expr.T.dropna().T
        a = r_path_expr[r_path_expr.index.str.contains("post")].mean()
        b = r_path_expr[r_path_expr.index.str.contains("pre")].mean()
        r_fcs.append(a - b)
    pickle.dump(r_fcs, open(os.path.join("metadata", "pathway.sample_expression.random.fold_changes.pickle"), "wb"))
    r_fcs = pickle.load(open(os.path.join("metadata", "pathway.sample_expression.random.fold_changes.pickle"), "rb"))

    [sns.distplot(x) for x in r_fcs]

    # Visualize
    # clustered
    g = sns.clustermap(path_expr_z, figsize=(30, 8))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.svg"), bbox_inches="tight", dpi=300)

    # ordered
    p = path_expr_z.sort_index(level=['patient_id', 'timepoint_name'])
    g = sns.clustermap(p, figsize=(30, 8), col_cluster=True, row_cluster=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.ordered.svg"), bbox_inches="tight", dpi=300)

    # sorted
    p = path_expr[path_expr.sum(axis=0).sort_values().index]
    g = sns.clustermap(p, figsize=(30, 8), col_cluster=False, row_cluster=False)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.sorted.svg"), bbox_inches="tight", dpi=300)

    #

    # Differential
    # change
    a = path_expr[path_expr.index.str.contains("post")].mean()
    a.name = "after_Ibrutinib"
    b = path_expr[path_expr.index.str.contains("pre")].mean()
    b.name = "before_Ibrutinib"
    fc = (a - b).sort_values()
    fc.name = "fold_change"

    # plot again but only top-bottom ordered by patient
    p = path_expr_z.sort_index()[fc.head(20).index.tolist() + fc.tail(20).index.tolist()]
    g = sns.clustermap(p, metric="correlation", square=True, figsize=(6, 6), col_cluster=True, row_cluster=True)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.ordered.top_bottom.all_patients.svg"), bbox_inches="tight", dpi=300)

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
    changes.sort_values("p_value").to_csv(os.path.join(rna.results_dir, "pathway.sample_expression.size-log2_fold_change-p_value.q_value.csv"))
    changes = pd.read_csv(os.path.join(rna.results_dir, "pathway.sample_expression.size-log2_fold_change-p_value.q_value.csv"), index_col=0)

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
    fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.scatter.svg"), bbox_inches="tight")

    # volcano
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())
    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['fold_change'], -np.log10(changes['q_value']), color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (5 ** z_score(changes["mean"])))
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['fold_change'].ix[l], -np.log10(changes['q_value']).ix[l], l, fontsize=7.5, ha="left")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['fold_change'].ix[l], -np.log10(changes['q_value']).ix[l], l, fontsize=7.5, ha="right")
    axis.set_xlabel("Change in pathway expression (after / before)")
    axis.set_ylabel("-log10(p-value)")
    sns.despine(fig)
    fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.volcano.svg"), bbox_inches="tight")

    # maplot
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['mean'], changes['fold_change'], color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (5 ** z_score(changes["mean"])))
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['mean'].ix[l], changes['fold_change'].ix[l], l, fontsize=7.5, ha="left")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['mean'].ix[l], changes['fold_change'].ix[l], l, fontsize=7.5, ha="right")
    axis.axhline(0, linestyle="--", color="black", alpha=0.5)
    axis.set_xlabel("Mean pathway expression between timepoints")
    axis.set_ylabel("Change in pathway expression (after / before)")
    sns.despine(fig)
    fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.maplot.svg"), bbox_inches="tight")

    # rank vs cross-patient sensitivity
    normalizer = matplotlib.colors.Normalize(vmin=changes['fold_change'].min(), vmax=changes['fold_change'].max())

    fig, axis = plt.subplots(1, figsize=(4, 4))
    axis.scatter(changes['fold_change'].rank(method="dense"), changes['fold_change'], color=plt.cm.inferno(normalizer(changes['fold_change'])), alpha=0.5, s=4 + (5 ** z_score(changes["mean"])))
    axis.axhline(0, color="black", alpha=0.8, linestyle="--")
    for l in changes['fold_change'].sort_values().head(15).index:
        axis.text(changes['fold_change'].rank(method="dense").ix[l], changes['fold_change'].ix[l], l, fontsize=5, ha="left")
    for l in changes['fold_change'].sort_values().tail(15).index:
        axis.text(changes['fold_change'].rank(method="dense").ix[l], changes['fold_change'].ix[l], l, fontsize=5, ha="right")
    axis.set_xlabel("Rank in change pathway expression")
    axis.set_ylabel("Change in pathway expression (after / before)")
    sns.despine(fig)
    fig.savefig(os.path.join(rna.results_dir, "pathway.mean_expression.rank.svg"), bbox_inches="tight")

    # Compare with pharmacoscopy
    # annotate atac the same way as pharma
    #

    # Connect pharmacoscopy pathway-level sensitivities with ATAC-seq

    # Sample-level
    pharma = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.score.pathway_space.csv"), index_col=0)
    pharma.loc["patient_id", :] = pd.Series(pharma.columns.str.split("-"), index=pharma.columns).apply(lambda x: x[0]).astype("category")
    pharma.loc["timepoint_name", :] = pd.Series(pharma.columns.str.split("-"), index=pharma.columns).apply(lambda x: x[1]).astype("category")
    pharma = pharma.T

    res = pd.DataFrame()
    # color = iter(cm.rainbow(np.linspace(0, 1, 50)))
    path_expr.index = path_expr.index.str.replace("pre", "before_Ibrutinib")
    path_expr.index = path_expr.index.str.replace("post", "after_Ibrutinib")

    ps = pharma.index.str.extract("(CLL\d+)-.*").drop_duplicates().sort_values().drop('CLL2').drop('CLL8')
    ts = pharma.index.str.extract(".*-(.*)").drop_duplicates().sort_values(ascending=False)
    fig, axis = plt.subplots(len(ts), len(ps), figsize=(len(ps) * 4, len(ts) * 4), sharex=False, sharey=False)
    for i, patient_id in enumerate(ps):
        for j, timepoint in enumerate(ts):
            p = pharma[(pharma["patient_id"] == patient_id) & (pharma["timepoint_name"] == timepoint)].sort_index().T
            a = path_expr.loc[(path_expr.index.str.contains(patient_id + "_")) & (path_expr.index.str.contains(timepoint)), :].T.sort_index()

            if a.shape[1] == 0:
                continue
            print(patient_id, timepoint)

            p = p.ix[a.index].dropna()
            a = a.ix[p.index].dropna()

            # stat, p_value = scipy.stats.pearsonr(p, a)
            axis[j][i].scatter(p, a, alpha=0.5)  # , color=color.next())
            axis[j][i].set_title(" ".join([patient_id, timepoint]))

            # res = res.append(pd.Series([patient_id, timepoint, stat, p_value]), ignore_index=True)
    sns.despine(fig)
    fig.savefig(os.path.join(rna.results_dir, "pathway_expression-pharmacoscopy.scatter.png"), bbox_inches='tight', dpi=300)

    # globally
    changes = pd.read_csv(os.path.join(rna.results_dir, "pathway.sample_expression.size-log2_fold_change-p_value.q_value.csv"), index_col=0)
    pharma_global = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.changes.csv"), index_col=0)
    # filter out drugs/pathways which after Ibrutinib are not more CLL-specific
    # pharma_global = pharma_global[pharma_global['after_Ibrutinib'] >= 0]

    pharma_change = pharma_global['fold_change']
    pharma_change.name = "pharmacoscopy"
    p = changes.join(pharma_change).fillna(0)

    # scatter
    n_to_label = 15
    custom_labels = p.index[p.index.str.contains("apoptosis|pi3k|proteasome|ribosome|nfkb", case=False)].tolist()

    combined_diff = (p['fold_change'] + p['pharmacoscopy']).dropna()
    m = max(abs(combined_diff.min()), abs(combined_diff.max()))
    normalizer = matplotlib.colors.Normalize(vmin=-m, vmax=m)

    for cmap in ["Spectral_r", "coolwarm", "BrBG"]:
        already = list()
        fig, axis = plt.subplots(1, 1, figsize=(4 * 1, 4 * 1))
        axis.scatter(p['fold_change'], p['pharmacoscopy'], alpha=0.8, color=plt.get_cmap(cmap)(normalizer(combined_diff)))
        axis.axhline(0, color="black", linestyle="--", alpha=0.5)
        axis.axvline(0, color="black", linestyle="--", alpha=0.5)
        # annotate top pathways
        # combined
        for path in [x for x in combined_diff.sort_values().head(n_to_label).index if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="right")
            already.append(path)
        for path in [x for x in combined_diff.sort_values().tail(n_to_label).index if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="left")
            already.append(path)
        # individual
        for path in [x for x in p['fold_change'].sort_values().head(n_to_label).index if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="right", color="orange")
            already.append(path)
        for path in [x for x in p['pharmacoscopy'].sort_values().head(n_to_label).index if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="left", color="green")
            already.append(path)
        for path in [x for x in p['fold_change'].sort_values().tail(n_to_label).index if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="left", color="orange")
            already.append(path)
        for path in [x for x in p['pharmacoscopy'].sort_values().tail(n_to_label).index if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="right", color="green")
            already.append(path)
        # custom
        for path in [x for x in custom_labels if x not in already]:
            axis.text(p['fold_change'].ix[path], p['pharmacoscopy'].ix[path], path, ha="center", color="blue")
            already.append(path)
        axis.set_xlabel("RNA-seq change in pathway accessibility", ha="center")
        axis.set_ylabel("Pharmacoscopy change in pathway sensitivity", ha="center")
        sns.despine(fig)
        fig.savefig(os.path.join(rna.results_dir, "rna-pharmacoscopy.across_patients.scatter.{}.svg".format(cmap)), bbox_inches='tight', dpi=300)
        # fig.savefig(os.path.join(rna.results_dir, "rna-pharmacoscopy.across_patients.scatter.only_positive.svg"), bbox_inches='tight', dpi=300)

    # individually
    pharma_patient = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.differential.patient_specific.abs_diff.csv"), index_col=0)
    a = path_expr[path_expr.index.str.contains("after_Ibrutinib")]
    a.index = a.index.str.extract(".*(CLL\d+)_")
    b = path_expr[path_expr.index.str.contains("before_Ibrutinib")]
    b.index = b.index.str.extract(".*(CLL\d+)_")
    rna_patient = (a - b).T

    pharma_patient = pharma_patient.T.ix[rna_patient.columns].T
    rna_patient = rna_patient[pharma_patient.columns]

    fig, axis = plt.subplots(3, 4, figsize=(4 * 4, 3 * 4), sharex=True, sharey=True)
    axis = axis.flatten()
    for i, patient_id in enumerate(rna_patient.columns):
        print(patient_id)

        p = pharma_patient[[patient_id]]
        p.columns = ["Pharmacoscopy"]
        a = rna_patient[patient_id]
        a.name = "ATAC-seq"

        o = p.join(a).dropna().apply(z_score, axis=0)

        # stat, p_value = scipy.stats.pearsonr(p, a)
        axis[i].scatter(o["Pharmacoscopy"], o["ATAC-seq"], alpha=0.5)  # , color=color.next())
        axis[i].set_title(" ".join([patient_id]))
        axis[i].axhline(0, linestyle="--", color="black", linewidth=1)
        axis[i].axvline(0, linestyle="--", color="black", linewidth=1)
    sns.despine(fig)
    fig.savefig(os.path.join(rna.results_dir, "pathway_expression-pharmacoscopy.patient_specific.scatter.png"), bbox_inches='tight', dpi=300)

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
    path_control = combined_diff[(combined_diff > -.01) & (combined_diff < .01)].index

    # pharmacoscopy sensitivity
    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    drug_annotation = pd.read_csv(os.path.join("metadata", "drugs_annotated.csv"))

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
        if len(a.shape) != 1:
            continue
        sens_diff[patient] = a - b

    drugs = drug_annotation[drug_annotation["kegg_pathway_name"].isin(paths_of_interest)]['drug'].drop_duplicates()
    ctrl_drugs = drug_annotation[drug_annotation["kegg_pathway_name"].isin(path_control)]['drug'].drop_duplicates()
    ctrl_drugs = ctrl_drugs[~ctrl_drugs.isin(drugs)]

    g = sns.clustermap(
        sens_pivot.ix[drugs.tolist() + ctrl_drugs.tolist()], metric="correlation", row_cluster=False, col_cluster=True) # , z_score=1
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    p = sens_pivot.T.groupby(level='timepoint_name').mean().T
    g = sns.clustermap(
        p.loc[drugs.tolist() + ctrl_drugs.tolist(), ['before_Ibrutinib', 'after_Ibrutinib']],
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
    index = expr.index[expr.index.isin([y for p in paths_of_interest for y in pathway_genes[p]])]
    path_expr = (expr.ix[index][[s.name for s in pc_samples]] / expr[[s.name for s in pc_samples]].sum(axis=0)) * 1e6

    cov_diff = pd.DataFrame()
    for patient in set(path_expr.columns.str.extract(".*(CLL\d+)_")):
        try:
            a = path_expr.T[
                (path_expr.columns.str.contains(patient)) &
                (path_expr.columns.str.contains("post"))].T.squeeze()
            b = path_expr.T[
                (path_expr.columns.str.contains(patient)) &
                (path_expr.columns.str.contains("pre"))].T.squeeze()
        except:
            continue
        cov_diff[patient] = a - b

    path_expr.columns = pd.MultiIndex.from_arrays(pd.DataFrame(map(pd.Series, path_expr.columns.str.split("_"))).T.values, names=['cell_type', "library", "patient", "sample", "timepoint"])
    path_expr = path_expr.sort_index(axis=1, level=["patient", "timepoint"], ascending=False)

    g = sns.clustermap(
        path_expr,
        metric="correlation", row_cluster=True, col_cluster=True, z_score=0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    g = sns.clustermap(
        path_expr,
        metric="correlation", row_cluster=True, col_cluster=False, z_score=0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.savefig(os.path.join(rna.results_dir, "pathway_expression.Proteasome.heatmap.svg"), bbox_inches='tight', dpi=300)

    g = sns.clustermap(
        cov_diff,
        metric="correlation", row_cluster=True, col_cluster=True)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)


def visualize_interaction_p_values():
    """
    """
    sns.set_style("white")


    variables = [
        "ighv_mutation_status",
        "CD38_positive",
        "del11q_threshold",
        "del13q_threshold",
        "del17p_threshold",
        "tri12_threshold",
        "p53"]

    all_res = pd.DataFrame()

    for i, variable in enumerate(variables):
        res = pd.read_csv(
            os.path.join("results", "cll-ibrutinib_AKH.ibrutinib_treatment",
                "cll-ibrutinib_AKH.ibrutinib_treatment.interaction_timepoint-" + variable + ".csv"),
            index_col=0)
        res["variable"] = variable
        all_res = all_res.append(res.reset_index(), ignore_index=True)

    fig1, axis1 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4), sharex=True, sharey=True)
    fig2, axis2 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4), sharex=True, sharey=True)
    fig3, axis3 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4), sharex=True, sharey=True)
    fig4, axis4 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4), sharex=True, sharey=True)
    for i, variable in enumerate(variables):
        res = all_res[all_res["variable"] == variable]
        ax = axis1.flatten()[i]
        sns.distplot(res['pvalue'].dropna(), ax=ax, kde=False)
        ax.axvline(0.415e-6, linestyle="--", color="grey")
        ax.set_title(variable)

        ax = axis2.flatten()[i]
        sns.distplot(-np.log10(res['pvalue'].dropna()), ax=ax, kde=False)
        ax.axvline(-np.log10(0.415e-6), linestyle="--", color="grey")
        ax.set_title(variable)

        ax = axis3.flatten()[i]
        sns.distplot(res['padj'].dropna(), ax=ax, kde=False)
        ax.axvline(-np.log10(0.05), linestyle="--", color="grey")
        ax.set_title(variable)

        ax = axis4.flatten()[i]
        sns.distplot(-np.log10(res['padj'].dropna()), ax=ax, kde=False)
        ax.axvline(-np.log10(0.05), linestyle="--", color="grey")
        ax.set_title(variable)
    fig1.savefig(os.path.join("results", "interaction_pvalues.svg"), bbox_inches="tight")
    fig2.savefig(os.path.join("results", "interaction_pvalues.log10.svg"), bbox_inches="tight")
    fig3.savefig(os.path.join("results", "interaction_pvalues.padj.svg"), bbox_inches="tight")
    fig4.savefig(os.path.join("results", "interaction_pvalues.padj.log10.svg"), bbox_inches="tight")


    fig, axis = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4), sharex=False, sharey=True)
    for i, variable in enumerate(variables):
        res = all_res[all_res["variable"] == variable]

        ax = axis.flatten()[i]
        ix = res['pvalue'].dropna().index
        ax.scatter(res.loc[ix, "log2FoldChange"], -np.log10(res.loc[ix, "pvalue"]), alpha=0.2, s=2, rasterized=True)
        ax.set_title(variable)
        ax.axvline(0, linestyle="--", color="grey", zorder=0)

    fig.savefig(os.path.join("results", "interaction_pvalues.volcano.svg"), bbox_inches="tight")

    fig, axis = plt.subplots(1, figsize=(4, 4))
    sns.barplot(-np.log10(all_res.groupby('variable')['padj'].min().sort_values()), all_res.groupby('variable')['padj'].min().sort_values().index, ax=axis, orient="horizontal")
    axis.axvline(-np.log10(0.05), linestyle="--", color="grey")
    fig.savefig(os.path.join("results", "interaction_pvalues.min.barplot.svg"), bbox_inches="tight")


def export_tables_for_pathways(
        pathways=['autophagy', 'proteasome']
        # pathways=['mtor', 'pi3k']
    ):
    """
    """
    from scipy.stats import zscore

    # get genes 
    pathway_gene = pickle.load(open(os.path.join("metadata", "pathway_gene_annotation_kegg.pickle"), "rb"))
    pathway_gene = pd.concat(
        [pd.Series(genes, index=[path for _ in genes]) for path, genes in pathway_gene.items()]
    ).squeeze()
    genes = pathway_gene[pathway_gene.index.str.contains("|".join(pathways), case=False)].drop_duplicates()

    # get accessibility changes at gene level
    atac = pd.read_csv(
        os.path.join(
            "results",
            "cll-ibrutinib_AKH.ibrutinib_treatment",
            "cll-ibrutinib_AKH.ibrutinib_treatment.timepoint_name.diff_regions.gene_level.fold_change.csv"), index_col=0, header=None)
    atac.columns = ['log2FoldChange']
    atac.index.name = 'gene'

    atac.loc[genes].sort_values('log2FoldChange', ascending=False).dropna().to_csv(
        os.path.join(
            "ibrutinib_treatment_accessibility.{}_genes.after_ibrutinib-before_ibrutinib.csv".format(",".join(pathways))), index=True)

    # get gene expression changes
    expr = pd.read_csv(
        os.path.join(
            "results",
            "ibrutinib_treatment_expression",
            "ibrutinib_treatment_expression.batch-timepoint_name.after_ibrutinib-before_ibrutinib.csv"), index_col=0)
    expr = expr.rename(columns={"baseMean": "mean_expression", "padj": "adjusted_p_value"}).drop(['lfcSE', 'stat'], axis=1)

    expr.loc[genes].dropna().to_csv(
        os.path.join(
            "ibrutinib_treatment_expression.{}_genes.after_ibrutinib-before_ibrutinib.csv".format(",".join(pathways))), index=True)


    # miracle plot
    p_all = atac.rename(columns={"log2FoldChange": "ATAC-seq"}).join(expr.rename(columns={"log2FoldChange": "RNA-seq"}).loc[:, "RNA-seq"]).dropna()
    p = atac.rename(columns={"log2FoldChange": "ATAC-seq"}).loc[genes].join(expr.rename(columns={"log2FoldChange": "RNA-seq"}).loc[genes, "RNA-seq"]).dropna()

    top_n = 20

    top_genes = p.apply(zscore).dropna().abs().sum(1).sort_values().tail(top_n).index

    fig, axis = plt.subplots(1, 2, figsize=(2 * 4, 4))
    axis[0].scatter(p_all['ATAC-seq'], p_all['RNA-seq'], s=2, alpha=0.1, rasterized=True)
    axis[1].scatter(p['ATAC-seq'], p['RNA-seq'], s=2, alpha=0.5)
    for g in top_genes:
        axis[1].text(x=p.loc[g, 'ATAC-seq'], y=p.loc[g, 'RNA-seq'], s=g, fontsize=5)
    for ax in axis:
        ax.axhline(0, color="black", alpha=0.5, zorder=-100)
        ax.axvline(0, color="black", alpha=0.5, zorder=-100)
    axis[0].set_xlim((-1.5, 1.5))
    axis[0].set_ylim((-1, 1))
    axis[1].set_xlim((-1.5, 1.5))
    axis[1].set_ylim((-1, 1))
    for ax in axis:
        ax.set_xlabel("ATAC-seq\n(log2 fold-change)")
    axis[0].set_ylabel("RNA-seq\n(log2 fold-change)")
    axis[0].set_title("All genes")
    axis[1].set_title("Genes in {} pathways".format(" and ".join(pathways)))
    fig.savefig(
        os.path.join(
            "ibrutinib_treatment_expression.{}_genes.miracle_plot2.svg"
            .format(",".join(pathways))), dpi=300, bbox_inches="tight")



def atac_expression_correlation():

    # atac = pd.read_csv(
    #     os.path.join("results", "cll-ibrutinib_AKH.accessibility.annotated_metadata.csv"),
    #     index_col=0, header=range(24))
    # atac -= atac.min().min()
    # atac_m = atac.T.groupby("timepoint_name").mean().T
    # atac_fc = np.log2(atac_m['after_Ibrutinib'] / atac_m['before_Ibrutinib'])
    # _atac_m = atac.T.groupby("timepoint_name").mean().T / atac.T.groupby("timepoint_name").std().T
    # _atac_fc = np.log2(_atac_m['after_Ibrutinib'] / _atac_m['before_Ibrutinib'])

    # fig, axis = plt.subplots(1, 2, figsize=(2 * 4, 4))
    # axis[0].scatter(atac_m.mean(axis=1), atac_fc, s=1, alpha=0.1, rasterized=True)
    # axis[1].scatter(np.log2(1 + _atac_m.mean(axis=1)), _atac_fc, s=1, alpha=0.1, rasterized=True)
    # sns.despine(fig)
    # fig.savefig("accessibility.ma_plot.svg", dpi=300, bbox_inches="tight")

    # # annotate with gene, get mean
    # atac_gene_map = pd.read_csv(os.path.join("results", "cll-ibrutinib_AKH_peaks.gene_annotation.csv"))
    # atac_gene_map.index = atac_gene_map['chrom'] + ":" + atac_gene_map['start'].astype(str) + "-" + atac_gene_map['end'].astype(str)
    # atac_g_annot = atac_gene_map['gene_name'].str.split(",").apply(pd.Series).stack().reset_index(level=1, drop=True)

    # atac_g_annot.name = "gene_name"
    # atac_fc_g = atac_fc.to_frame(name="atac").join(atac_g_annot).groupby("gene_name").mean().squeeze()
    # atac_fc_g_std = (
    #     atac_fc.to_frame(name="atac").join(atac_g_annot).groupby("gene_name").mean() /
    #     atac_fc.to_frame(name="atac").join(atac_g_annot).groupby("gene_name").std()).squeeze()
    # _atac_fc_g = _atac_fc.to_frame(name="atac").join(atac_g_annot).groupby("gene_name").mean().squeeze()
    # _atac_fc_g_std = (
    #     _atac_fc.to_frame(name="atac").join(atac_g_annot).groupby("gene_name").mean() /
    #     _atac_fc.to_frame(name="atac").join(atac_g_annot).groupby("gene_name").std()).squeeze()

    # # # alternatively, get closest reg. element
    # # q = atac_g_annot.to_frame(name="gene_name").join(atac_gene_map.drop('gene_name', axis=1))
    # # qq = q.groupby('gene_name')['distance'].apply(np.argmin)
    # # qq = qq.reset_index().set_index('distance').squeeze()
    # # atac_fc_g = atac_fc.to_frame(name="atac").join(qq).groupby("gene_name").mean().squeeze()
    # # atac_fc_g_std = (
    # #     atac_fc.to_frame(name="atac").join(qq).groupby("gene_name").mean() /
    # #     atac_fc.to_frame(name="atac").join(qq).groupby("gene_name").std()).squeeze()
    # # _atac_fc_g = _atac_fc.to_frame(name="atac").join(qq).groupby("gene_name").mean().squeeze()
    # # _atac_fc_g_std = (
    # #     _atac_fc.to_frame(name="atac").join(qq).groupby("gene_name").mean() /
    # #     _atac_fc.to_frame(name="atac").join(qq).groupby("gene_name").std()).squeeze()

    # # Landau expression data
    # expr = pd.read_excel(os.path.join("data", "RS_005_annotated_48_PB_median_gt_0_coding_RAW data.xlsx"))
    # expr = expr.set_index("gene_name").drop(['gene_id', 'coding_status'], axis=1)

    # # extract metadata
    # expr2 = expr.copy()
    # patient_id = map(lambda x: x[0], expr.columns.str.split("-"))
    # timepoint = map(lambda x: int(x[-1]), expr.columns.str.split("-"))
    # expr2.columns = pd.MultiIndex.from_arrays([patient_id, timepoint], names=['patient_id', 'timepoint'])

    # # get only first and last timepoint per patient
    # m = expr2.T.reset_index().groupby(['patient_id'])['timepoint'].min().reset_index()
    # m = m.append(expr2.T.reset_index().groupby(['patient_id'])['timepoint'].max().reset_index())
    # expr = expr.loc[:, m['patient_id'] + "-T-0" + m['timepoint'].astype(str)]
    # # extract metadata
    # patient_id = map(lambda x: x[0], expr.columns.str.split("-"))
    # timepoint = map(lambda x: int(x[-1]), expr.columns.str.split("-"))
    # timepoint = [t if t == 1 else 2 for t in timepoint]
    # expr.columns = pd.MultiIndex.from_arrays([patient_id, timepoint], names=['patient_id', 'timepoint'])

    # # Get fold change
    # m = expr.T.groupby("timepoint").mean().T
    # fc = np.log2(m[2] / m[1])
    # _m = expr.T.groupby("timepoint").mean().T / expr.T.groupby("timepoint").std().T
    # _fc = np.log2(_m[2] / _m[1])

    # fig, axis = plt.subplots(1, 2, figsize=(2 * 4, 4))
    # axis[0].scatter(m.mean(axis=1), fc, s=2, alpha=0.2, rasterized=True)
    # axis[1].scatter(np.log2(1 + _m.mean(axis=1)), _fc, s=2, alpha=0.2, rasterized=True)
    # sns.despine(fig)
    # fig.savefig("landau_expression.ma_plot.svg", dpi=300, bbox_inches="tight")


    # # compare with BSF
    # expr_bsf = pd.read_table(os.path.join("results", "gene_exp.diff")).drop(['test_id', 'gene_id'], axis=1)
    # e = expr_bsf['gene'].str.split(",").apply(pd.Series).stack()
    # e = e.reset_index(level=1, drop=True).to_frame(name="gene").join(expr_bsf['log2(fold_change)'])
    # bsf = e.groupby("gene").mean()

    # # compare with deseq
    # expr_deseq = pd.read_csv(
    #     os.path.join("results", "cll-ibrutinib_AKH-RNA.expression_counts.gene_level.quantile_normalized.log2_tpm.annotated_metadata.csv"),
    #     index_col=0, header=range(22))
    # expr_deseq -= expr_deseq.min().min()
    # m_deseq = expr_deseq.T.groupby("timepoint_name").mean().T
    # fc_deseq = np.log2(m_deseq['after_Ibrutinib'] / m_deseq['before_Ibrutinib'])
    # _m_deseq = expr_deseq.T.groupby("timepoint_name").mean().T / expr_deseq.T.groupby("timepoint_name").std().T
    # _fc_deseq = np.log2(_m_deseq['after_Ibrutinib'] / _m_deseq['before_Ibrutinib'])

    # fig, axis = plt.subplots(1, 2, figsize=(2 * 4, 4))
    # axis[0].scatter(m_deseq.mean(axis=1), fc_deseq, s=2, alpha=0.2, rasterized=True)
    # axis[1].scatter(np.log2(1 + _m_deseq.mean(axis=1)), _fc_deseq, s=2, alpha=0.2, rasterized=True)
    # sns.despine(fig)
    # fig.savefig("deseq_expression.ma_plot.svg", dpi=300, bbox_inches="tight")


    # fc.name = 'landau'
    # _fc.name = 'landau_std'
    # fc_deseq.name = 'deseq'
    # _fc_deseq.name = 'deseq_std'
    # atac_fc_g.name = 'atac'
    # atac_fc_g_std.name = 'atac_g'
    # _atac_fc_g.name = 'atac_std'
    # _atac_fc_g_std.name = 'atac_std_g'
    # p = (bsf.squeeze().to_frame(name="bsf")
    #     .join(fc).join(_fc)
    #     .join(fc_deseq).join(_fc_deseq)
    #     .join(atac_fc_g).join(atac_fc_g_std).join(_atac_fc_g).join(_atac_fc_g_std))



    # # correlate means
    # atac = pd.read_csv(
    #     os.path.join("results", "cll-ibrutinib_AKH.accessibility.annotated_metadata.csv"),
    #     index_col=0, header=range(24))
    # atac -= atac.min().min()
    # atac_g = atac.join(atac_g_annot).groupby('gene_name').mean().mean(axis=1)
    # expr_g = expr.mean(axis=1)
    # expr_g.name = "rna"
    # p = atac_g.to_frame(name="atac").join(expr_g)

    # fig, axis = plt.subplots(1, 1, figsize=(1 * 4, 4))
    # axis.scatter(p['atac'], p['rna'], s=2, alpha=0.2, rasterized=True)
    # axis.set_xlabel("ATAC-seq")
    # axis.set_ylabel("RNA-seq (Landau)")
    # axis.text(3, 0, s="Pearson r={}".format(p.corr('pearson').loc['rna', 'atac']))
    # axis.text(3, 1, s="Spearman r={}".format(p.corr('spearman').loc['rna', 'atac']))
    # # axis[1].scatter(np.log2(1 + _m_deseq.mean(axis=1)), _fc_deseq, s=2, alpha=0.2, rasterized=True)
    # sns.despine(fig)
    # fig.savefig("correlation.accessibility_expression.scatter.svg", dpi=300, bbox_inches="tight")


    grid = sns.lmplot(data=p, x='atac', y='rna', scatter_kws={"alpha": 0.5, "s": 3, "rasterized": True})
    fig.axes[0].set_xlabel("ATAC-seq")
    fig.axes[0].set_ylabel("RNA-seq")
    fig.axes[0].text(3, 0, s="Pearson r={}".format(p.corr('pearson').loc['rna', 'atac']))
    fig.axes[0].text(3, 1, s="Spearman r={}".format(p.corr('spearman').loc['rna', 'atac']))
    sns.despine(grid.fig)
    grid.fig.savefig("correlation.accessibility_expression.lmplot.svg", dpi=300, bbox_inches="tight")


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
    sample_attributes = [
        "patient_id", "timepoint_name", "patient_gender", "patient_age_at_collection",
        "ighv_mutation_status", "CD38_cells_percentage", "leuko_count (10^3/uL)", "% lymphocytes", "purity (CD5+/CD19+)", "%CD19/CD38", "% CD3", "% CD14", "% B cells", "% T cells",
        "del11q", "del13q", "del17p", "tri12", "p53",
        "time_since_treatment", "treatment_response"]
    analysis.unsupervised(
        analysis.samples,
        attributes=sample_attributes)

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

    analysis.sites = pybedtools.BedTool(os.path.join(analysis.results_dir, analysis.name + "_peak_set.bed"))
    analysis.support = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.support.csv"))
    # analysis.coverage = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.raw_coverage.csv"), index_col=0)
    analysis.gene_annotation = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.gene_annotation.csv"))
    analysis.closest_tss_distances = pickle.load(open(os.path.join(analysis.results_dir, analysis.name + "_peaks.closest_tss_distances.pickle"), "rb"))
    analysis.region_annotation = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.region_annotation.csv"))
    analysis.region_annotation_b = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.region_annotation_background.csv"))
    analysis.chrom_state_annotation = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.chromatin_state.csv"))
    analysis.chrom_state_annotation_b = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.chromatin_state_background.csv"))

    analysis.accessibility = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + ".accessibility.annotated_metadata.csv"), header=range(len(sample_attributes) + 3), index_col=0)

    #

    # Pharmacoscopy part
    sensitivity = pd.read_csv(os.path.join("metadata", "pharmacoscopy_score_v3.csv"))
    annotate_drugs(analysis, sensitivity)
    pharmacoscopy(analysis)
    atac_to_pathway(analysis)
    make_network(analysis)

    # RNA-seq
    prj = Project("metadata/project_config.yaml")
    prj.samples = [s for s in prj.samples if s.library == "RNA-seq"]
    rna = RNASeqAnalysis(name="cll-ibrutinib_AKH-RNA", prj=prj, samples=prj.samples)

    # get expression and check quality
    rna.get_gene_expression(attributes=sample_attributes)
    rna.expression_matrix_counts = pd.read_csv(os.path.join(rna.results_dir, rna.name + ".expression_counts.gene_level.csv"), index_col=0)
    rna.expression_matrix_qnorm = pd.read_csv(os.path.join(rna.results_dir, rna.name + ".expression_counts.gene_level.quantile_normalized.csv"), index_col=0)
    rna.expression = pd.read_csv(os.path.join(rna.results_dir, rna.name + ".expression_counts.gene_level.quantile_normalized.log2_tpm.csv"), index_col=0)
    rna.expression_annotated = pd.read_csv(os.path.join(rna.results_dir, rna.name + ".expression_counts.gene_level.quantile_normalized.log2_tpm.annotated_metadata.csv"), index_col=0, header=range(len(sample_attributes)))

    rna.plot_expression_characteristics()

    # unsupervised analysis
    attrs = [
        "patient_id", "timepoint_name", "patient_gender", "patient_age_at_collection",
        "ighv_mutation_status",
        "del11q", "del13q", "del17p", "tri12", "p53",
        "time_since_treatment", "treatment_response"]
    rna.unsupervised(quant_matrix="expression_annotated", attributes_to_plot=attrs, plot_prefix="all_genes")

    # supervised
    rna.differential_expression_analysis(
        rna.samples,
        trait="timepoint_name",
        variables=["atac_seq_batch", "timepoint_name"],
        output_suffix="ibrutinib_treatment_expression")

    # pathway-based
    rna_to_pathway(analysis)



def revision_patients():
    from looper.models import Project
    from ngs_toolkit.atacseq import ATACSeqAnalysis
    from ngs_toolkit.general import unsupervised_analysis
    from scipy.stats import zscore

    prj = Project(os.path.join("metadata", "project_config.revision.yaml"))
    for sample in prj.samples:
        sample.filtered = os.path.join(
            sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.filtered.bam")

    analysis = ATACSeqAnalysis(name="cll-ibrutinib.revision", prj=prj, samples=[s for s in prj.samples if s.library == "ATAC-seq"])
    sample_attributes = ["sample_name", "patient_id", "sample_id", "timepoint_name", "timepoint", "patient_gender", "ighv_mutation_status"]
    group_attributes = ["patient_id", "timepoint_name", "timepoint", "patient_gender", "ighv_mutation_status"]
    analysis.sites = pybedtools.BedTool(os.path.join("results", "cll-ibrutinib_AKH_peak_set.bed"))

    # Get samples from CLL umbrella project
    for sample in [s for s in analysis.samples if s.old_sample_name != ""]:
        print(sample.name)
        if not os.path.exists(os.path.join("data", sample.name)):
            os.system("cp -r {} {}".format(
                os.path.join("~/projects", "cll-patients", "pipeline_output/", sample.old_sample_name),
                os.path.join("data", sample.name)))

            os.system("find {} -name '{}*' -exec rename {} {} {{}} \;".format(
                "data", sample.old_sample_name, sample.old_sample_name, sample.name))

    # Get count matrix for new samples
    c = analysis.measure_coverage(
        analysis,
        samples=[s for s in analysis.samples if s.ibrutinib_cohort == "2"],
        output_file=os.path.join("results", "cll-ibrutinib.revision_peaks.raw_coverage.only_revision_samples.csv"))
    # join with previous
    cov = pd.read_csv(os.path.join("results", "cll-ibrutinib_CLL_peaks.raw_coverage.csv"), index_col=0)
    cov = cov[[s for s in cov.columns if s in [ss.name for ss in analysis.samples]]]
    analysis.coverage = cov.join(c)
    analysis.coverage.to_csv(os.path.join(analysis.results_dir, analysis.name + "_peaks.raw_coverage.csv"), index=True)
    analysis.normalize(method="gc_content")
    analysis.annotate_with_sample_metadata(quant_matrix="coverage_gc_corrected", attributes=sample_attributes, numerical_attributes=["timepoint"])

    # Unsupervised
    unsupervised_analysis(
        analysis, data_type="ATAC-seq", quant_matrix="accessibility",
        samples=[s for s in analysis.samples if s.patient_id not in ["CLL16"]],
        attributes_to_plot=group_attributes, test_pc_association=False,
        plot_prefix="unsupervised_analysis.revision",
        output_dir=os.path.join(analysis.results_dir, "unsupervised_analysis.revision"))

    # Observe previous treatment signature
    diff = pd.read_csv(os.path.join(
        "results", "cll-ibrutinib_AKH.ibrutinib_treatment",
        "cll-ibrutinib_AKH.ibrutinib_treatment.timepoint_name.after_Ibrutinib-before_Ibrutinib.csv"), index_col=0)
    sig = diff[diff['padj'] < 0.01]

    acce = analysis.accessibility.sort_index(axis=1, level=['timepoint_name', 'patient_id'])
    grid1 = sns.clustermap(
        acce.loc[sig.index, :].T,
        z_score=1, robust=True, xticklabels=False, yticklabels=acce.columns.get_level_values("sample_name"),
        rasterized=True, cmap="RdBu_r", cbar_kws={"label": "Chromatin accessibility\n(Z-score)"})
    grid1.ax_heatmap.set_yticklabels(grid1.ax_heatmap.get_yticklabels(), rotation=0)
    grid1.ax_heatmap.set_ylabel("Samples")
    grid1.savefig(os.path.join(analysis.results_dir, analysis.name + ".ibrutinib_signature.clustered.svg"), bbox_inches="tight", dpi=300)

    grid1 = sns.clustermap(
        acce.loc[sig.index, :].T,
        row_cluster=False,
        z_score=1, robust=True, xticklabels=False, yticklabels=acce.columns.get_level_values("sample_name"),
        rasterized=True, cmap="RdBu_r", cbar_kws={"label": "Chromatin accessibility\n(Z-score)"})
    grid1.ax_heatmap.set_yticklabels(grid1.ax_heatmap.get_yticklabels(), rotation=0)
    grid1.ax_heatmap.set_ylabel("Samples")
    grid1.savefig(os.path.join(analysis.results_dir, analysis.name + ".ibrutinib_signature.sorted.svg"), bbox_inches="tight", dpi=300)

    acce_z = acce.loc[sig.index, :].apply(zscore, axis=1)

    p = acce_z.loc[sig.index, ~acce_z.columns.get_level_values("sample_name").str.contains('CLL18|CLL19|CLL22')]
    p = p.sort_index(axis=1, level=['timepoint_name', 'patient_id'])
    grid = sns.clustermap(
        p.T,
        row_cluster=False,
        robust=True, xticklabels=False, yticklabels=p.columns.get_level_values("sample_name"),
        rasterized=True, cmap="RdBu_r", cbar_kws={"label": "Chromatin accessibility\n(Z-score)"})
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0)
    grid.ax_heatmap.set_ylabel("Samples")
    grid.savefig(os.path.join(analysis.results_dir, analysis.name + ".ibrutinib_signature.initial_samples.sorted.svg"), bbox_inches="tight", dpi=300)

    p = acce_z.iloc[grid.dendrogram_col.reordered_ind, :]
    p = p.loc[:, (
            p.columns.get_level_values("sample_name").str.contains('_CLL18_|_CLL19_|_CLL22_') &
            ~p.columns.get_level_values("sample_name").str.contains('pre|post'))]
    p = p.sort_index(axis=1, level=['timepoint_name', 'patient_id'])
    grid2 = sns.clustermap(
        p.T,
        figsize=(grid1.fig.get_size_inches()[0], 4),
        row_cluster=False, col_cluster=False,
        robust=True, xticklabels=False, yticklabels=p.columns.get_level_values("sample_name"),
        rasterized=True, cmap="RdBu_r", cbar_kws={"label": "Chromatin accessibility\n(Z-score)"})
    grid2.ax_heatmap.set_yticklabels(grid2.ax_heatmap.get_yticklabels(), rotation=0)
    grid2.ax_heatmap.set_ylabel("Samples")
    grid2.savefig(os.path.join(analysis.results_dir, analysis.name + ".ibrutinib_signature.revision_samples.sorted_time.svg"), bbox_inches="tight", dpi=300)

    p = acce_z.iloc[grid.dendrogram_col.reordered_ind, :]
    p = p.loc[:, (
        p.columns.get_level_values("sample_name").str.contains('_CLL18_|_CLL19_|_CLL22_') &
        ~p.columns.get_level_values("sample_name").str.contains('pre|post'))]
    p = p.sort_index(axis=1, level=["patient_id", "timepoint"])
    grid2 = sns.clustermap(
        p.T,
        figsize=(grid1.fig.get_size_inches()[0], 4),
        row_cluster=False, col_cluster=False,
        robust=True, xticklabels=False, yticklabels=p.columns.get_level_values("sample_name"),
        rasterized=True, cmap="RdBu_r", cbar_kws={"label": "Chromatin accessibility\n(Z-score)"})
    grid2.ax_heatmap.set_yticklabels(grid2.ax_heatmap.get_yticklabels(), rotation=0)
    grid2.ax_heatmap.set_ylabel("Samples")
    grid2.savefig(os.path.join(analysis.results_dir, analysis.name + ".ibrutinib_signature.revision_samples.sorted_patient.svg"), bbox_inches="tight", dpi=300)


    # Run pathway-level analysis
    pathway_genes = pickle.load(open(os.path.join("metadata", "pathway_gene_annotation_kegg.pickle"), "rb"))

    # Get accessibility for each regulatory element assigned to each gene in each pathway
    # Reduce values per gene
    # Reduce values per pathway
    sel_samples = [s for s in analysis.samples]
    cov = analysis.accessibility.loc[:, [s.name for s in sel_samples]]
    chrom_annot = pd.read_csv(os.path.join("results", "cll-ibrutinib_CLL_peaks.coverage_qnorm.log2.annotated.csv"), index_col=0)
    path_cov = pd.DataFrame()
    path_cov.columns.name = 'kegg_pathway_name'
    for pathway in pathway_genes.keys():
        print(pathway)
        # mean of all reg. elements of all genes
        index = chrom_annot.loc[(chrom_annot['gene_name'].isin(pathway_genes[pathway])), 'gene_name'].index
        q = cov.ix[index]
        path_cov[pathway] = (q[[s.name for s in sel_samples]].mean(axis=0) / cov[[s.name for s in sel_samples]].sum(axis=0)) * 1e6
    path_cov = path_cov.T.dropna().T

    path_cov_z = path_cov.apply(zscore, axis=0)
    path_cov.T.to_csv(os.path.join(analysis.results_dir, analysis.name + ".pathway.sample_accessibility.csv"))
    path_cov_z.T.to_csv(os.path.join(analysis.results_dir, analysis.name + ".pathway.sample_accessibility.z_score.csv"))
    path_cov = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + ".pathway.sample_accessibility.csv"), index_col=0, header=range(24)).T
    path_cov_z = pd.read_csv(os.path.join(analysis.results_dir, analysis.name + ".pathway.sample_accessibility.z_score.csv"), index_col=0, header=range(24)).T

    # Visualize
    # clustered
    g = sns.clustermap(
        path_cov_z, figsize=(30, 8), rasterized=True,
        xticklabels=True, yticklabels=path_cov_z.index.get_level_values("sample_name"), cmap="RdBu_r")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.svg"), bbox_inches="tight", dpi=300)

    # ordered
    p = path_cov_z.sort_index(level=['patient_id', 'timepoint'])
    g = sns.clustermap(
        p.reset_index(drop=True), figsize=(30, 8), rasterized=True,
        xticklabels=True, yticklabels=p.index.get_level_values("sample_name"), col_cluster=True, row_cluster=False, cmap="RdBu_r")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.ordered.svg"), bbox_inches="tight", dpi=300)

    # sorted
    p = path_cov[path_cov.sum(axis=0).sort_values().index]
    p = p.ix[p.sum(axis=1).sort_values().index]
    g = sns.clustermap(
        p, figsize=(30, 8), rasterized=True,
        xticklabels=True, yticklabels=p.index.get_level_values("sample_name"), col_cluster=False, row_cluster=False, cmap="RdBu_r")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.sorted.svg"), bbox_inches="tight", dpi=300)

    # For specific pathways only

    sel_pathways = ['FoxO signaling pathway', 'PI3K-Akt signaling pathway', 'NF-kappa B signaling pathway', 'TNF signaling pathway', 'TGF-beta signaling pathway', "Autophagy", "Proteasome"]

    path_cov_z2 = path_cov_z.loc[:, path_cov_z.columns.isin(sel_pathways)]

    # clustered
    g = sns.clustermap(
        path_cov_z2, figsize=(30, 8), rasterized=True, square=True,
        xticklabels=True, yticklabels=path_cov_z.index.get_level_values("sample_name"), cmap="RdBu_r")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.select_pathways.svg"), bbox_inches="tight", dpi=300)

    # ordered
    p = path_cov_z2.sort_index(level=['patient_id', 'timepoint'])
    g = sns.clustermap(
        p.reset_index(drop=True), figsize=(30, 8), rasterized=True, square=True,
        xticklabels=True, yticklabels=p.index.get_level_values("sample_name"), col_cluster=True, row_cluster=False, cmap="RdBu_r")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.select_pathways.ordered.svg"), bbox_inches="tight", dpi=300)

    # sorted
    p = path_cov[path_cov.sum(axis=0).sort_values().index]
    p = p.ix[p.sum(axis=1).sort_values().index]
    g = sns.clustermap(
        p, figsize=(30, 8), rasterized=True,
        xticklabels=True, yticklabels=p.index.get_level_values("sample_name"), col_cluster=False, row_cluster=False, cmap="RdBu_r")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.sorted.svg"), bbox_inches="tight", dpi=300)



    # For new patients only

    for patient in ["CLL18", "CLL19", "CLL22"]:

        ss = [s.name for s in analysis.samples if s.patient_id == patient and s.ibrutinib_cohort == "2"]

        # ordered
        p = path_cov_z.loc[ss, :]
        g = sns.clustermap(
            p.reset_index(drop=True), figsize=(24, 4),
            xticklabels=True, yticklabels=p.index.get_level_values("sample_name"), col_cluster=True, row_cluster=False, cmap="RdBu_r")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
        g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.{}.ordered.svg".format(patient)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(
            p.reset_index(drop=True), figsize=(24, 4), z_score=1,
            xticklabels=True, yticklabels=p.index.get_level_values("sample_name"), col_cluster=True, row_cluster=False, cmap="RdBu_r")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
        g.fig.savefig(os.path.join(analysis.results_dir, analysis.name + ".pathway.mean_accessibility.{}.ordered.z_score.svg".format(patient)), bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Program canceled by user!")
        sys.exit(1)
