# Geno4SD


[![Build Status](https://travis.ibm.com/ComputationalGenomics/Geno4SD.svg?token=8XHVVZSCStEbEBxrmvno&branch=main)](https://travis.ibm.com/ComputationalGenomics/Geno4SD)
[![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://pages.github.com/ComputationalGenomics/Geno4SD/)


Geno4SD is an omics data toolkit for the analysis of omics data across biological scales, from single-cell analysis to large patient cohorts, and over multiple modalities, including genomics, transcriptomics, clinical medical data, and patient demographics. Within this toolkit are analytic methods that span phylogenetics, epidemilogy, topological data analysis, and ML/AL frameworks for omics scale data.

Geno4SD provides access to individual tools as well as detailed use cases for analyses that demonstrate how multiple methodologies can be leveraged together.


## Analytic tools included in Geno4SD

1. ReVeaL: _Rare Variant Learning_, is a stochastic regularization-based learning algorithm. It partitions the genome into non-overlapping, possibly non-contiguous, windows (_w_) and then aggregates samples into possibly overlapping subsets, using subsampling with replacement (stochastic), giving units called shingles that are utilized by a statistical learning algorithm. Each shingle captures a distribution of the mutational load (the number of mutations in the window _w_ of a given sample), and the first four moments are used as an approximation of the distribution.
2. CuNA:  _Cumulant-based Network Analysis_ finds higher-order genotype-phenotype relationships from multi-omic data with EHR information across different thresholds of statistical significance.
3. RubricOE: is a cross-validated  machine learning framework with feature ranking described and multiple levels of cross validation to obtain interpretable genetic and non-genetic features from genomic or transcriptomic data combined.


