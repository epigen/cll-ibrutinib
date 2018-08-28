<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.231352.svg)](https://doi.org/10.5281/zenodo.231352) -->

### Integrative analysis of cell-specific chemosensitivity and epigenetic cell states prioritizes ibrutinib-based drug combinations for chronic lymphocytic leukemia

Christian Schmidl<sup>*</sup>, Gregory I Vladimer<sup>*</sup>, André F Rendeiro<sup>*</sup>, Susanne Schnabl<sup>*</sup>, Tea Pemovska, Thomas Krausgruber, Mohammad Araghi, Nikolaus Krall, Berend Snijder, Rainer Hubmann, Anna Ringler, Dita Demirtas, Oscar Lopez de la Fuente, Martin Hilgarth, Cathrin Skrabs, Edit Porpaczy, Michaela Gruber, Gregor Hörmann, Stefan Kubicek, Philipp B Staber, Medhat Shehata, Giulio Superti-Furga, Ulrich Jäger, Christoph Bock. Integrative analysis of cell-specific chemosensitivity and epigenetic cell states prioritizes ibrutinib-based drug combinations for chronic lymphocytic leukemia. (2018).

<sup>\*</sup>Shared first authors


**Website**: [cll-synergies.computational-epigenetics.org](http://cll-synergies.computational-epigenetics.org)

This repository contains scripts used in the analysis of the data in the paper.

#### Analysis
In the [supplementary website](http://cll-synergies.computational-epigenetics.org) you can find most of the output of the whole analysis.

If you wish to reproduce the processing of the raw data, download the data [from here](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100672):

1. Clone the repository: `git clone git@github.com:epigen/cll-ibrutinib.git`
2. Install required software for the analysis:`make requirements` or `pip install -r requirements.txt`
1. Download the data localy.
2. Prepare [Looper](https://github.com/epigen/looper) configuration files similar to [these](metadata/project_config.yaml) that fit your local system.
3. Run samples through the pipeline: `make preprocessing` or `looper run metadata/project_config_file.yaml`
<!-- 4. Get external files (genome annotations mostly): `make external_files` or use the files in the [paper website](http://cll-chromatin.computational-epigenetics.org) (`external` folder). -->
5. Run the analysis: `make analysis` or `python src/analysis.py`

Additionaly, processed (bigWig and narrowPeak files together with a chromatin accessibility matrix) are available from [GEO with accession number GSE100672](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100672).

If you wish to reproduce the plots from the analysis you can, in principle:

1. run `python src/analysis.py`
