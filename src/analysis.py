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
            X.corr(), xticklabels=False, yticklabels=sample_display_names, annot=True, aspect=2,
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
        x = pd.DataFrame(x_new)
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
        fig.savefig(os.path.join(self.results_dir, "{}.all_sites.pca.svg".format(self.name)), bbox_inches="tight")

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

        df = self.coverage_annotated

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

        # # Pyupset
        # import pyupset as pyu
        # # Build dict
        # diff["comparison_direction"] = diff[["comparison", "direction"]].apply(string.join, axis=1)
        # df_dict = {group: diff[diff["comparison_direction"] == group].reset_index()[['index']] for group in set(diff["comparison_direction"])}
        # # Plot
        # plot = pyu.plot(df_dict, unique_keys=['index'], inters_size_bounds=(10, np.inf))
        # plot['figure'].set_size_inches(20, 8)
        # plot['figure'].savefig(os.path.join(output_dir, "%s.%s.number_differential.upset.svg" % (output_suffix, trait)), bbox_inched="tight")

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
        # characterize_regions_structure(df=df2, prefix=prefix, output_dir=output_dir)
        # region's function
        # characterize_regions_function(df=df2, prefix=prefix, output_dir=output_dir)

        # Heatmaps
        # Comparison level
        g = sns.clustermap(np.log2(1 + df2[groups]).corr(), xticklabels=False)
        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.groups.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(np.log2(1 + df2[groups]).T, xticklabels=False)
        for item in g.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.groups.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(np.log2(1 + df2[groups]).T, xticklabels=False, z_score=1)
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
            self.accessibility.ix[df2.index][[s.name for s in sel_samples]].corr(),
            xticklabels=False, yticklabels=sample_display_names, annot=True, vmin=0, vmax=1,
            cmap="Spectral_r", figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.samples.clustermap.corr.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(
            self.accessibility.ix[df2.index][[s.name for s in sel_samples]].T,
            xticklabels=False, yticklabels=sample_display_names, annot=True,
            figsize=(15, 15), cbar_kws={"label": "Pearson correlation"}, row_colors=color_dataframe.values.tolist())
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_xlabel(g.ax_heatmap.get_xlabel(), visible=False)
        g.ax_heatmap.set_ylabel(g.ax_heatmap.get_ylabel(), visible=False)
        g.fig.savefig(os.path.join(output_dir, "%s.%s.diff_regions.samples.clustermap.svg" % (output_suffix, trait)), bbox_inches="tight", dpi=300)

        g = sns.clustermap(
            self.accessibility.ix[df2.index][[s.name for s in sel_samples]].T, z_score=1,
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
                ].index]
                if comparison_df.shape[0] < 1:
                    continue
                # Characterize regions
                prefix = "%s.%s.diff_regions.comparison_%s.%s" % (output_suffix, trait, comparison, direction)

                comparison_dir = os.path.join(output_dir, prefix)

                print("Doing regions of comparison %s, with prefix %s" % (comparison, prefix))

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


def atac_to_pathway(analysis):
    """
    Quantify the activity of each pathway by the accessibility of the regulatory elements of its genes.
    """
    from bioservices.kegg import KEGG
    import scipy

    # Query KEGG for genes member of pathways with drugs annotated
    k = KEGG()
    k.organism = "hsa"

    drug_annot = pd.read_csv(os.path.join("results", "pharmacoscopy", "drugs_annotated.csv"))
    pathway_genes = dict()
    for pathway in drug_annot['kegg_pathway_id'].dropna().drop_duplicates():
        print(pathway)
        res = k.parse(k.get(pathway))
        if type(res) is dict and 'GENE' in res:
            print(len(res['GENE']))
            pathway_genes[res['PATHWAY_MAP'][pathway]] = list(set([x.split(";")[0] for x in res['GENE'].values()]))

    # Get accessibility for each regulatory element assigned to each gene in each pathway
    # Reduce values per gene
    # Reduce values per pathway
    cov = analysis.coverage_annotated
    path_cov = pd.DataFrame()
    path_cov.columns.name = 'kegg_pathway_name'
    for pathway in pathway_genes.keys():
        print(pathway)
        # mean of all reg. elements of all genes
        # path_cov[pathway] = cov.ix[cov['gene_name'][cov['gene_name'].isin(pathway_genes[pathway])].index][[s.name for s in analysis.samples]].mean(axis=0)
        # mean of reg. elements of all genes annoted as overlaping TSSs
        q = cov.ix[cov['gene_name'][cov['gene_name'].isin(pathway_genes[pathway])].index]
        q = q[q['genomic_region'] == "tss2kb"]
        q = q[q['mean'] > 2]
        path_cov[pathway] = q[[s.name for s in analysis.samples]].mean(axis=0)

    path_cov = path_cov.T.dropna().T

    # # Visualize
    # fig = sns.clustermap(path_cov, figsize=(30, 8))
    # for tick in fig.ax_heatmap.get_xticklabels():
    #     tick.set_rotation(90)
    # for tick in fig.ax_heatmap.get_yticklabels():
    #     tick.set_rotation(0)
    # fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.png"), bbox_inches="tight", dpi=300)
    # # fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.svg"), bbox_inches="tight")

    # fig = sns.clustermap(path_cov.T.dropna().T, figsize=(30, 8), z_score=1)
    # for tick in fig.ax_heatmap.get_xticklabels():
    #     tick.set_rotation(90)
    # for tick in fig.ax_heatmap.get_yticklabels():
    #     tick.set_rotation(0)
    # fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.z_score.png"), bbox_inches="tight", dpi=300)
    # # fig.savefig(os.path.join(analysis.results_dir, "pathway.mean_accessibility.svg"), bbox_inches="tight")

    # Compare with pharmacoscopy
    # annotate atac the same way as pharma
    ext = pd.Series(path_cov.index.str.split("_")).apply(pd.Series)
    ext.index = path_cov.index
    path_cov.loc[:, 'patient'] = ext[2].astype('category')
    path_cov.loc[:, 'timepoint'] = ext[4].astype('category')
    path_cov = path_cov.set_index(['patient', 'timepoint'])
    pathway_space = pd.read_csv(os.path.join("results", "pharmacoscopy", "pharmacoscopy.sensitivity.pathway_space.csv"), index_col=0)

    # reduce pharmacoscopy measurements to mean of concentrations
    s = pd.Series(pathway_space.columns).apply(lambda x: pd.Series(x.split("-")))
    s.index = pathway_space.columns
    s.columns = [['patient_id', 'timepoint', 'concentration']]
    pharma = pathway_space.T.join(s).groupby(['patient_id', 'timepoint']).mean()

    # Connect pharmacoscopy pathway-level sensitivities with ATAC-seq
    res = pd.DataFrame()

    fig, axis = plt.subplots(4, 3, sharex=True, sharey=True)
    axis = axis.flatten()
    i = 0
    color = iter(cm.rainbow(np.linspace(0, 1, 50)))
    for patient_id in pharma.index.levels[0].sort_values():
        for timepoint in pharma.index.levels[1].sort_values(ascending=False):
            pharma_path = pharma.ix[patient_id, timepoint].squeeze().drop_duplicates()
            atac_path = path_cov.ix[patient_id, timepoint].squeeze().drop_duplicates()
            print(patient_id, timepoint)

            p = pd.merge(pharma_path.reset_index(), atac_path.reset_index(), on='kegg_pathway_name').set_index('kegg_pathway_name')
            p.columns = ['pharma', 'atac']

            cor = scipy.stats.pearsonr(p['pharma'], p['atac'])[0]

            axis[i].scatter(np.log2(p['pharma']), np.log2(1 + p['atac']), alpha=0.1, color="blue" if timepoint == "pre" else "green")

            res = res.append(pd.Series([patient_id, timepoint, cor]), ignore_index=True)
        axis[i].set_title(" ".join([patient_id]))
        i += 1
    # sns.distplot(res[2], ax=axis[-1])
    fig.savefig(os.path.join("pharmacoscopy.sensitivity.measured_vs_predicted.scatter.svg"), bbox_inches='tight')


def annotate_drugs(analysis, sensitivity):
    """
    """
    from bioservices.kegg import KEGG
    from collections import defaultdict
    import string
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

    output_dir = os.path.join(analysis.results_dir, "pharmacoscopy")

    sensitivity = pd.read_csv(os.path.join(analysis.data_dir, "pharmacoscopy_sensitivity.csv"))
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
    cloud = pd.read_csv(os.path.join("data", "CLOUD_simple_annotation.csv"))
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
    annot.to_csv(os.path.join(output_dir, "drugs_annotated.csv"), index=False)

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
    annot.to_csv(os.path.join(output_dir, "drugs_annotated.csv"), index=False)

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
    annot.to_csv(os.path.join(output_dir, "drugs_annotated.csv"), index=False)
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

        pathway_vector = abs(pathway_scores).mean(axis=0)
        new_drug_space = abs(pathway_scores).mean(axis=1)

        return pathway_vector, new_drug_space

    output_dir = os.path.join(analysis.results_dir, "pharmacoscopy")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # tcn = pd.read_csv(os.path.join(analysis.data_dir, "pharmacoscopy_TCN.csv"))
    # cd19 = pd.read_csv(os.path.join(analysis.data_dir, "pharmacoscopy_cd19.csv"))
    sensitivity = pd.read_csv(os.path.join(analysis.data_dir, "pharmacoscopy_sensitivity.csv"))
    auc = pd.read_csv(os.path.join(analysis.data_dir, "pharmacoscopy_AUC.csv")).dropna()

    # Read up drug annotation
    # annot = analysis.drug_annotation
    # annot = pd.read_csv(os.path.join(analysis.data_dir, "drugs_annotated.csv"))

    # transform AUC to inverse
    auc["AUC"] *= -1
    # scale values
    sensitivity["sensitivity_scaled"] = np.log2(1 + abs(sensitivity["sensitivity"])) * (sensitivity["sensitivity"] > 0).astype(int).replace(0, -1)
    sensitivity["sensitivity_scaled_normal"] = mean_standard_score(sensitivity["sensitivity_scaled"])
    auc["AUC_scaled"] = np.log2(1 + abs(auc["AUC"])) * (auc["AUC"] > 0).astype(int).replace(0, -1)
    auc["AUC_scaled_normal"] = mean_standard_score(auc["AUC_scaled"])

    # demonstrate the scaling
    fig, axis = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
    axis = axis.flatten()
    # axis[0].set_title("original sensitivity")
    sns.distplot(sensitivity['sensitivity'], bins=100, kde=False, ax=axis[0], label="original sensitivity")
    ax = zoomed_inset_axes(axis[0], zoom=6, loc=1, axes_kwargs={"aspect": "auto", "xlim": (-2, 5), "ylim": (0, 100000)})
    sns.distplot(sensitivity['sensitivity'], bins=300, kde=False, ax=ax)
    axis[0].legend()
    # axis[1].set_title("scaled sensitivity")
    sns.distplot(np.log2(1 + abs(sensitivity["sensitivity"])), bins=50, kde=False, ax=axis[1], label="log2 absolute sensitivity")
    sns.distplot(sensitivity["sensitivity_scaled"], bins=50, kde=False, ax=axis[1], label="log2 directional sensitivity")
    axis[1].legend()
    # axis[2].set_title("original AUC")
    sns.distplot(auc['AUC'].dropna(), bins=100, kde=False, ax=axis[2], label="original AUC")
    ax = zoomed_inset_axes(axis[2], zoom=6, loc=1, axes_kwargs={"aspect": "auto", "xlim": (-20, 20), "ylim": (0, 300)})
    sns.distplot(auc['AUC'].dropna(), bins=300, kde=False, ax=ax)
    axis[2].legend()
    # axis[3].set_title("scaled AUC")
    sns.distplot(np.log2(1 + abs(auc["AUC"].dropna())), bins=50, kde=False, ax=axis[3], label="log2 absolute AUC")
    sns.distplot(auc["AUC_scaled"].dropna(), bins=50, kde=False, ax=axis[3], label="log2 directional AUC")
    axis[3].legend()
    sns.despine(fig)
    fig.savefig(os.path.join(output_dir, "scaling_demo.svg"), bbox_inches="tight")

    # merge in one table
    # (
    #     pd.merge(pd.merge(auc, cd19, how="outer"), sensitivity, how="outer")
    #     .sort_values(['original_patient_id', 'timepoint', 'drug', 'concentration'])
    #     .to_csv(os.path.join(analysis.data_dir, "pharmacoscopy_all.csv"), index=False))

    # Plot distributions
    # tcn

    # cd19

    # auc & sensitivity
    for df, name in [(sensitivity, "sensitivity"), (auc, "AUC")]:
        if name == "sensitivity":
            df['p_id'] = df['original_patient_id'] + " " + df['timepoint'] + " " + df['concentration'].astype(str)
        else:
            df['p_id'] = df['original_patient_id'] + " " + df['timepoint']

        # Relation between variability and mean
        if name == "sensitivity":
            keys = ['original_patient_id', 'drug', 'concentration', 'timepoint']
            mean = df.groupby(keys)[name + "_scaled"].mean().reset_index()
            std = df.groupby(keys)[name + "_scaled"].std().reset_index()
            mean_std = pd.merge(mean, std, on=keys, suffixes=["_mean", "_std"])

            fig = sns.jointplot(
                mean_std[name + "_scaled_mean"],
                mean_std[name + "_scaled_std"], alpha=0.5)
            fig.savefig(os.path.join(output_dir, "{}.mean_std.jointplot.svg".format(name)), bbox_inches="tight")

            for concentration in df['concentration'].unique():
                fig = sns.jointplot(
                    mean_std[mean_std['concentration'] == concentration][name + "_scaled_mean"],
                    mean_std[mean_std['concentration'] == concentration][name + "_scaled_std"], alpha=0.5)
                fig.savefig(os.path.join(output_dir, "{}.mean_std.concentration_{}.jointplot.svg".format(name, concentration)), bbox_inches="tight")

        df_pivot = df.pivot_table(index="p_id", columns="drug", values=name + "_scaled")
        # noise = np.random.normal(0, 0.1, df_pivot.shape[0] * df_pivot.shape[1]).reshape(df_pivot.shape)
        # df_pivot = df_pivot.add(noise)

        # Mean across patients
        df_mean = df.groupby(['drug', 'timepoint'])[name + "_scaled"].mean().reset_index()
        fig, axis = plt.subplots(len(df_mean['timepoint'].unique()))
        for i, timepoint in enumerate(df_mean['timepoint'].unique()):
            sns.distplot(df_mean[df_mean['timepoint'] == timepoint][name + "_scaled"].dropna(), kde=False, ax=axis[i])
            axis[i].set_title(timepoint)
        fig.savefig(os.path.join(output_dir, "{}.timepoints.distplot.svg".format(name)), bbox_inches="tight")

        # Timepoints and concentrations together
        fig = sns.clustermap(df_pivot.drop("DMSO", axis=1), figsize=(20, 8))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.svg".format(name)), bbox_inches="tight")

        fig = sns.clustermap(df_pivot.drop("DMSO", axis=1), z_score=1, figsize=(20, 8))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.zscore.svg".format(name)), bbox_inches="tight")

        # Normalized to DMSO
        if name == "sensitivity":
            df_pivot = df[df['concentration'] == 10].pivot_table(index="p_id", columns="drug", values=name + "_scaled")
            df_pivot2 = df_pivot.copy()
            for col in df_pivot.columns:
                df_pivot2.loc[:, col] = df_pivot2.loc[:, col] - df_pivot2['DMSO']

            fig = sns.clustermap(np.log10(10 + df_pivot2.drop("DMSO", axis=1)), figsize=(20, 8))
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "{}.DMSO_norm.svg".format(name)), bbox_inches="tight")

            fig = sns.clustermap(np.log10(10 + df_pivot2.drop("DMSO", axis=1)), z_score=1, figsize=(20, 8))
            for tick in fig.ax_heatmap.get_xticklabels():
                tick.set_rotation(90)
            for tick in fig.ax_heatmap.get_yticklabels():
                tick.set_rotation(0)
            fig.savefig(os.path.join(output_dir, "{}.DMSO_norm.zscore.svg".format(name)), bbox_inches="tight")

        # Concentrations separately
        if name == "sensitivity":
            for concentration in df['concentration'].unique():
                df2 = df[df['concentration'] == concentration]
                df2['p_id'] = df2['original_patient_id'] + df2['timepoint']
                df_pivot2 = df2.pivot_table(index="p_id", columns="drug", values=name + "_scaled")

                fig = sns.clustermap(df_pivot2, figsize=(20, 8))
                for tick in fig.ax_heatmap.get_xticklabels():
                    tick.set_rotation(90)
                for tick in fig.ax_heatmap.get_yticklabels():
                    tick.set_rotation(0)
                fig.savefig(os.path.join(output_dir, "{}.concentration_{}.svg".format(name, concentration)), bbox_inches="tight")

                fig = sns.clustermap(df_pivot2, figsize=(20, 8), z_score=1)
                for tick in fig.ax_heatmap.get_xticklabels():
                    tick.set_rotation(90)
                for tick in fig.ax_heatmap.get_yticklabels():
                    tick.set_rotation(0)
                fig.savefig(os.path.join(output_dir, "{}.concentration_{}.zscore.svg".format(name, concentration)), bbox_inches="tight")

        # Difference between timepoints (mean of concentrations)
        df_pivot2 = (
            df.groupby(['original_patient_id', 'drug', 'timepoint'])
            [name + "_scaled"].mean()
            .reset_index())
        post = df_pivot2[df_pivot2['timepoint'] == "post"].pivot_table(index="original_patient_id", columns="drug", values=name + "_scaled")
        pre = df_pivot2[df_pivot2['timepoint'] == "pre"].pivot_table(index="original_patient_id", columns="drug", values=name + "_scaled")

        if name != "sensitivity":
            post = post.drop("DMSO", axis=1)
            pre = pre.drop("DMSO", axis=1)

        rel_diff = post / pre
        abs_diff = post - pre

        fig = sns.clustermap(rel_diff, figsize=(20, 8))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.rel_diff.svg".format(name)), bbox_inches="tight")

        fig = sns.clustermap(rel_diff, figsize=(20, 8), z_score=1)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.rel_diff.zscore.svg".format(name)), bbox_inches="tight")

        fig = sns.clustermap(abs_diff, figsize=(20, 8))
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.abs_diff.svg".format(name)), bbox_inches="tight")

        fig = sns.clustermap(abs_diff, figsize=(20, 8), z_score=1)
        for tick in fig.ax_heatmap.get_xticklabels():
            tick.set_rotation(90)
        for tick in fig.ax_heatmap.get_yticklabels():
            tick.set_rotation(0)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.abs_diff.zscore.svg".format(name)), bbox_inches="tight")

        # Test
        test_results = pd.DataFrame(columns=[
            "post", "pre", "m", "abs_m", "a", "mannwhitneyu_stat", "mannwhitneyu_p_value",
            "ks_2samp_stat", "ks_2samp_p_value", "ttest_ind_stat", "ttest_ind_p_value"])
        for drug in post.columns:
            r, g = post[drug].mean(), pre[drug].mean()
            m = np.log2(r / g)
            abs_m = r - g
            a = np.log2(r * g) / 2.
            mannwhitneyu_stat, mannwhitneyu_p = mannwhitneyu(post[drug], pre[drug])
            ks_2samp_stat, ks_2samp_p = ks_2samp(post[drug], pre[drug])
            ttest_ind_stat, ttest_ind_p = ttest_ind(post[drug], pre[drug])
            test_results = test_results.append(
                pd.Series([
                    r, g, m, abs_m, a, mannwhitneyu_stat, mannwhitneyu_p,
                    ks_2samp_stat, ks_2samp_p, ttest_ind_stat, ttest_ind_p],
                    index=[
                        "post", "pre", "m", "abs_m", "a", "mannwhitneyu_stat", "mannwhitneyu_p_value",
                        "ks_2samp_stat", "ks_2samp_p_value", "ttest_ind_stat", "ttest_ind_p_value"], name=drug))
        for test in ["mannwhitneyu", "ks_2samp", "ttest_ind"]:
            test_results.loc[:, test + "_q_value"] = multipletests(test_results[test + "_p_value"], method="fdr_bh")[1]
        # save
        test_results.index.name = "drug"
        test_results.sort_values("mannwhitneyu_p_value").to_csv(os.path.join(output_dir, "{}.timepoint_change.tests.csv".format(name)), index=True)

        fig, axis = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False)
        # Plot scatter
        axis[0].scatter(test_results["post"], test_results["pre"], alpha=0.8)
        axis[0].set_title("Scatter")
        axis[0].set_xlabel("Post")
        axis[0].set_ylabel("Pre")

        # Plot MA
        axis[1].scatter(test_results["a"], test_results["m"], alpha=0.8)
        axis[1].set_title("MA")
        axis[1].set_xlabel("A")
        axis[1].set_ylabel("M")

        # Plot Volcano
        axis[2].scatter(test_results["m"], -np.log10(test_results["mannwhitneyu_q_value"]), color="grey", alpha=0.5)
        axis[2].scatter(test_results["m"], -np.log10(test_results["mannwhitneyu_p_value"]), alpha=0.8)
        axis[2].set_title("Volcano")
        axis[2].set_xlabel("M")
        axis[2].set_ylabel("-log10(p-value)")
        sns.despine(fig)
        fig.savefig(os.path.join(output_dir, "{}.timepoint_change.tests.svg".format(name)), bbox_inches="tight")

    # Convert each sample to pathway-space
    pathway_space = pd.DataFrame()
    new_drugs = pd.DataFrame()
    for original_patient_id in sensitivity['original_patient_id'].drop_duplicates().sort_values()[5:]:
        for timepoint in sensitivity['timepoint'].drop_duplicates().sort_values(ascending=False):
            for concentration in sensitivity['concentration'].drop_duplicates().sort_values():
                sample_id = "-".join([original_patient_id, timepoint, str(concentration)])
                print(sample_id)
                # get respective data and reduce replicates (by mean)
                drug_vector = sensitivity.loc[
                    (
                        (sensitivity['original_patient_id'] == original_patient_id) &
                        (sensitivity['timepoint'] == timepoint) &
                        (sensitivity['concentration'] == concentration)),
                    ['drug', 'sensitivity_scaled_normal']].groupby('drug').mean().squeeze()

                # covert between spaces
                pathway_vector, new_drug_space = drug_to_pathway_space(
                    drug_vector=drug_vector,
                    drug_annotation=analysis.drug_annotation,
                    plot=True,
                    plot_label=sample_id)

                # save
                pathway_space[sample_id] = pathway_vector
                new_drugs[sample_id] = new_drug_space
    pathway_space.to_csv(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.csv"))
    new_drugs.to_csv(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drug_space.csv"))

    # Plots

    # evaluate "performance"
    # (similarity between measured and new drug sensitivities)
    s = pd.Series(new_drugs.columns).apply(lambda x: pd.Series(x.split("-")))
    s.index = new_drugs.columns
    s.columns = [['patient_id', 'timepoint', 'concentration']]
    new_drug_space = new_drugs.T.join(s)

    fig, axis = plt.subplots(
        6, 4,
        # len(sensitivity['timepoint'].drop_duplicates()),
        # len(sensitivity['original_patient_id'].drop_duplicates()),
        figsize=(11, 14),
        sharex=True, sharey=True
    )
    axis = axis.flatten()
    i = 0
    for original_patient_id in sensitivity['original_patient_id'].drop_duplicates().sort_values():
        for timepoint in sensitivity['timepoint'].drop_duplicates().sort_values(ascending=False):
            for concentration in sensitivity['concentration'].drop_duplicates().sort_values():
                print(original_patient_id, timepoint, concentration)

                original_drug = sensitivity.loc[
                    (
                        (sensitivity['original_patient_id'] == original_patient_id) &
                        (sensitivity['timepoint'] == timepoint) &
                        (sensitivity['concentration'] == concentration)), ["drug", "sensitivity"]].groupby('drug').mean().squeeze()
                new_drug = new_drug_space.loc[
                    (
                        (new_drug_space['patient_id'] == original_patient_id) &
                        (new_drug_space['timepoint'] == timepoint) &
                        (new_drug_space['concentration'].astype(np.int64) == concentration)), :].squeeze().drop(['patient_id', 'timepoint', 'concentration'])
                p = pd.DataFrame([original_drug, new_drug], index=['original', 'engineered']).T.dropna()
                axis[i].scatter(np.log2(1 + abs(p['original'])), np.log2(1 + abs(p['engineered'])), color="blue" if concentration == 1 else "green", alpha=0.3)
            axis[i].set_title(" ".join([original_patient_id, timepoint]))
            i += 1
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.measured_vs_predicted.scatter.svg"), bbox_inches='tight')

    # vizualize in heatmaps
    fig = sns.clustermap(pathway_space[pathway_space.sum(1) != 0].T, figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.svg"), bbox_inches='tight')
    fig = sns.clustermap(pathway_space[pathway_space.sum(1) != 0].T, figsize=(20, 8), z_score=1)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.z_score.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.z_score.svg"), bbox_inches='tight')
    fig = sns.clustermap(new_drugs.T, figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.svg"), bbox_inches='tight')
    fig = sns.clustermap(new_drugs.T, figsize=(20, 8), z_score=0)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.z_score.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.z_score.svg"), bbox_inches='tight')

    # correlate samples in pathway space
    fig = sns.clustermap(pathway_space[pathway_space.sum(1) != 0].corr(), figsize=(8, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.pearson.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.svg"), bbox_inches='tight')

    # Reduce to mean of concentrations
    s = pd.Series(pathway_space.columns).apply(lambda x: pd.Series(x.split("-")))
    s.index = pathway_space.columns
    s.columns = [['patient_id', 'timepoint', 'concentration']]
    pathway_space2 = pathway_space.T.join(s).groupby(['patient_id', 'timepoint']).mean()
    s = pd.Series(new_drugs.columns).apply(lambda x: pd.Series(x.split("-")))
    s.index = new_drugs.columns
    s.columns = [['patient_id', 'timepoint', 'concentration']]
    new_drugs2 = new_drugs.T.join(s).groupby(['patient_id', 'timepoint']).mean()

    # vizualize in heatmaps
    fig = sns.clustermap(pathway_space2.T[pathway_space2.sum(0) != 0].T, figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.png"), bbox_inches='tight', dpi=300)
    fig = sns.clustermap(pathway_space2.T[pathway_space2.sum(0) != 0].T, figsize=(20, 8), z_score=1)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.z_score.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.z_score.png"), bbox_inches='tight', dpi=300)
    fig = sns.clustermap(new_drugs2, figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.png"), bbox_inches='tight', dpi=300)
    fig = sns.clustermap(new_drugs2.T[new_drugs2.sum(0) != 0].T, figsize=(20, 8), z_score=1)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.z_score.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.z_score.png"), bbox_inches='tight', dpi=300)

    # correlate samples in pathway space
    fig = sns.clustermap(pathway_space2[pathway_space2.sum(1) != 0].T.corr(), figsize=(8, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.pearson.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.svg"), bbox_inches='tight')

    # Calculate fold-changes
    pathway_space3 = pathway_space2.reset_index(1)
    pathway_space3 = pathway_space3[pathway_space3['timepoint'] == "post"].drop('timepoint', axis=1) - pathway_space3[pathway_space3['timepoint'] == "pre"].drop('timepoint', axis=1)
    new_drugs3 = new_drugs2.reset_index(1)
    new_drugs3 = new_drugs3[new_drugs3['timepoint'] == "post"].drop('timepoint', axis=1) - new_drugs3[new_drugs3['timepoint'] == "pre"].drop('timepoint', axis=1)

    # vizualize in heatmaps
    fig = sns.clustermap(pathway_space3.T[pathway_space3.sum(0) != 0].T, figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.timepoint_fold_change.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.timepoint_fold_change.png"), bbox_inches='tight', dpi=300)
    fig = sns.clustermap(pathway_space3.T[pathway_space3.sum(0) != 0].T, figsize=(20, 8), z_score=1)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.timepoint_fold_change.z_score.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.timepoint_fold_change.z_score.png"), bbox_inches='tight', dpi=300)
    fig = sns.clustermap(new_drugs3, figsize=(20, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.timepoint_fold_change.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.timepoint_fold_change.png"), bbox_inches='tight', dpi=300)
    fig = sns.clustermap(new_drugs3.T[new_drugs3.sum(0) != 0].T, figsize=(20, 8), z_score=1)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.timepoint_fold_change.z_score.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.new_drugs.mean_concentrations.timepoint_fold_change.z_score.png"), bbox_inches='tight', dpi=300)

    # correlate samples in pathway space
    fig = sns.clustermap(pathway_space3[pathway_space3.sum(1) != 0].T.corr(), figsize=(8, 8))
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_rotation(90)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.mean_concentrations.timepoint_fold_change.pearson.png"), dpi=300, bbox_inches='tight')
    # fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.pathway_space.svg"), bbox_inches='tight')

    # Connect pharmacoscopy pathway-level sensitivities with ATAC-seq
    pharma = pathway_space2.reset_index()
    enrichr = pd.read_csv(os.path.join("results", "interaction", "interaction.diff_regions.enrichr.csv"))
    atac = enrichr[enrichr.gene_set_library == 'KEGG_2016']
    atac.loc[:, 'kegg_pathway_name'] = pd.Series(atac['description'].str.split("_")).apply(pd.Series)[0]
    ids = pd.Series(atac['patient'].str.split(".")).apply(pd.Series)[[2, 3]]
    atac.loc[:, 'patient_id'] = pd.Series(ids[2].str.split("_")).apply(pd.Series)[1]
    atac.loc[:, 'direction'] = ids[3]
    atac = atac[atac['kegg_pathway_name'].isin(pathway_space3.columns)].set_index('kegg_pathway_name')

    res = pd.DataFrame()
    color = iter(cm.rainbow(np.linspace(0, 1, 50)))
    for patient_id in pharma['patient_id'].drop_duplicates().sort_values():
        for timepoint in pharma['timepoint'].drop_duplicates().sort_values(ascending=False):
            pharma_path = pharma.loc[(
                (pharma['patient_id'] == patient_id) &
                (pharma['timepoint'] == timepoint)), :].drop(['patient_id', 'timepoint'], axis=1).T.squeeze().drop_duplicates()
            for metric in ['p_value', 'z_score', 'combined_score']:
                print(patient_id, timepoint, metric)

                atac_path = atac.loc[(
                    (atac['patient_id'] == patient_id)),  # &
                    # (atac['timepoint'] == timepoint)),
                    metric].squeeze().drop_duplicates()
                p = pd.merge(pharma_path.reset_index(), atac_path.reset_index(), on='kegg_pathway_name').set_index('kegg_pathway_name')
                p.columns = ['pharma', 'atac']

                cor = scipy.stats.pearsonr(p['pharma'], p['atac'])[0]

                if metric == "combined_score":
                    plt.scatter(np.log2(p['pharma']), np.log2(1 + p['atac']), alpha=0.1, color=color.next())

                res = res.append(pd.Series([patient_id, timepoint, metric, cor]), ignore_index=True)
        axis[i].set_title(" ".join([original_patient_id, timepoint]))
        i += 1
    fig.savefig(os.path.join(output_dir, "pharmacoscopy.sensitivity.measured_vs_predicted.scatter.svg"), bbox_inches='tight')


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
    cll_ints = pybedtools.BedTool(os.path.join("data", analysis.name + "_peak_set.bed"))

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

    # export ensembl gene names
    df['gene_name'].str.split(",").apply(pd.Series, 1).stack().drop_duplicates().to_csv(os.path.join(output_dir, "%s_genes.symbols.txt" % prefix), index=False)

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
    a = self.coverage_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples]].mean(axis=1)

    # Get gene-level measurements of accessibility dependent on treatment
    u = self.coverage_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples if not s.under_treatment]].mean(axis=1)
    t = self.coverage_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples if s.under_treatment]].mean(axis=1)

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
    t2 = self.coverage_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in samples if s.timepoint_name == 'EGCG_100uM']].mean(axis=1)

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
        u2 = self.coverage_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in patient if s.timepoint_name == 'control']].mean(axis=1)
        t2 = self.coverage_annotated.groupby("ensembl_gene_id").mean()[[s.name for s in patient if s.timepoint_name == 'EGCG_100uM']].mean(axis=1)
        axis[i].scatter(genes["log2_fold_change"], (u2.ix[g] - t2.ix[g]))
        axis[i].set_title(p)
        axis[i].set_xlabel("Fold change gene expression (log2)")
        axis[i].set_ylabel("Difference in mean accessibility")
    fig.savefig(os.path.join(self.results_dir, "EGCG_targets.expression_vs_accessibility.untreated_EGCG_100uM.patient_specific.svg"), bbox_inches="tight")


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
        sample.peaks = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_peaks.narrowPeak")
        sample.summits = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_summits.bed")
        sample.mapped = os.path.join(sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.bam")
        sample.filtered = os.path.join(sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.filtered.bam")
        sample.coverage = os.path.join(sample.paths.sample_root, "coverage", sample.name + ".cov")

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

    # temporary:
    for sample in prj.samples:
        sample.peaks = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_peaks.narrowPeak")
        sample.summits = os.path.join(sample.paths.sample_root, "peaks", sample.name + "_summits.bed")
        sample.mapped = os.path.join(sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.bam")
        sample.filtered = os.path.join(sample.paths.sample_root, "mapped", sample.name + ".trimmed.bowtie2.filtered.bam")
        sample.coverage = os.path.join(sample.paths.sample_root, "coverage", sample.name + ".cov")

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
