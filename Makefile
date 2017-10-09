.DEFAULT_GOAL := all

requirements:
	pip install -r requirements.txt

	# other tools:
	# bwa macs samtools sambamba bedtools picard

preprocess: requirements
	looper run metadata/project_config.yaml

analysis: preprocess external_files
	python src/analysis.py
	python src/drug_synergies.py

all: requirements preprocess analysis

.PHONY: requirements preprocess analysis all
