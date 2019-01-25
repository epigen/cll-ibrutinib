.DEFAULT_GOAL := all

requirements:
	pip install -r requirements.txt


# Process Raw Data:
# tools for data processing from raw reads:
# bowtie2 macs samtools sambamba bedtools picard
# see https://github.com/epigen/open_pipelines/ for
# the pipeline source code
# and the "tools" section of the atacseq pipeline configuration file:
#  https://github.com/epigen/open_pipelines/blob/344986ddfe88f4265c18a9b6fedf3de32e9d8f26/pipelines/atacseq.yaml#L46
preprocess: requirements
	looper run metadata/project_config.yaml

analysis: preprocess
	python src/prepare_external_files.py
	python src/analysis.py
	# from below are plots used in the functional assays during revision:
	python src/regionset_intersect.py
	python src/drug_synergies.py
	python src/drug_combinations.coculture.py
	python src/culture_method.py
	python src/toxicity_vs_specificity.py

all: requirements preprocess analysis

.PHONY: requirements preprocess analysis all
