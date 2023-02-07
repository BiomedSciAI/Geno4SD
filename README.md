# Geno4SD


[![Build Status](https://github.com/BiomedSciAI/Geno4SD/actions/workflows/workflow.yml/badge.svg)](https://github.com/BiomedSciAI/Geno4SD/actions/workflows/workflow.yml)(https://travis.ibm.com/ComputationalGenomics/Geno4SD)
[![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://biomedsciai.github.io/Geno4SD/)


Geno4SD is an omics data toolkit for the analysis of omics data across biological scales, from single-cell analysis to large patient cohorts, and over multiple modalities, including genomics, transcriptomics, clinical medical data, and patient demographics. Within this toolkit are analytic methods that span phylogenetics, epidemilogy, topological data analysis, and ML/AL frameworks for omics scale data.

Geno4SD provides access to individual tools as well as detailed use cases for analyses that demonstrate how multiple methodologies can be leveraged together.

![Geno4SD](docs/img/Geno4SD.png)

## Analytic tools included in Geno4SD

1.  **ReVeaL: _Rare Variant Learning_** is a stochastic regularization-based learning algorithm. It partitions the genome into non-overlapping, possibly non-contiguous, windows (_w_) and then aggregates samples into possibly overlapping subsets, using subsampling with replacement (stochastic), giving units called shingles that are utilized by a statistical learning algorithm. Each shingle captures a distribution of the mutational load (the number of mutations in the window _w_ of a given sample), and the first four moments are used as an approximation of the distribution.

    ReVeaL tutorial can be found here: [tutorial](https://github.com/BiomedSciAI/Geno4SD/blob/main/tutorials/ReVeaL.ipynb)

2. **LSM: _Lesion Shedding Model_** can order lesions from the highest to the lowest ctDNA shedding for a given patient from cfDNA liquid and lesion biopsies. Our framework intrinsically models for missing/hidden lesions and operates on blood and lesion cfDNA assays to estimate the potential relative shedding levels of lesions into the blood. By characterizing the lesion-specific cfDNA shedding levels, we can better understand the mechanisms of shedding as well as more accurately contextualize and interpret cfDNA assays to improve their clinical impact.

3. **CuNA:  _Cumulant-based Network Analysis_** is a toolkit for integrating and analyzing multi-omics data which finds higher-order relationships from multi-omic data with EHR information across different thresholds of statistical significance.
CuNA provides two components:
        
    1. A network with nodes representing multi-omics variables and edges reflecting their stre
ngth in higher-order interactions.

    2. A risk score, *CuRES*, which is a holistic view of risk or liability of a target trait or
 disease, per individual.

   CuNA tutorial can be found here: [tutorial](https://github.com/ComputationalGenomics/Geno4SD/blob/main/tutorials/CuNA.ipynb)

   *CuNAviz*, the visualization tool for CuNA can be found here: 
   
   1. [CuNAviz for Parkinson's Disease](https://rawcdn.githack.com/BiomedSciAI/Geno4SD/98784437396363a680e7ecac9d98509793f48cfc/docs/data/cunaviz_demo.html)
   2. [CuNAviz for Breast Cancer, scenario I](https://rawcdn.githack.com/BiomedSciAI/Geno4SD/8d8036b760c2fa486423681a0549ed204eb48380/docs/data/cunaviz_False_25.html)
   3. [CuNAviz for Breast Cancer, scenario II](https://rawcdn.githack.com/BiomedSciAI/Geno4SD/8d8036b760c2fa486423681a0549ed204eb48380/docs/data/cunaviz_False_50.html)
   4. [CuNAviz for Breast Cancer, scenario III](https://rawcdn.githack.com/BiomedSciAI/Geno4SD/8d8036b760c2fa486423681a0549ed204eb48380/docs/data/cunaviz_True_25.html)
   5. [CuNAviz for Breast Cancer, scenario IV](https://rawcdn.githack.com/BiomedSciAI/Geno4SD/8d8036b760c2fa486423681a0549ed204eb48380/docs/data/cunaviz_True_25.html)

3. **RubricOE: a rubric for omics epidemiology** is a cross-validated  machine learning framework with feature ranking described and multiple levels of cross validation to obtain interpretable genetic and non-genetic features from multi-omics data combined.

   RubricOE tutorial can be found here: [tutorial](https://github.com/BiomedSciAI/Geno4SD/blob/main/tutorials/RubricOE.ipynb)
   
4. **StatGen: _Statistical Genetics toolkit_** is a toolkit for performing quality control on imputed genotype data, computing principal component analysis (using [TeraPCA](https://github.com/aritra90/TeraPCA)) and thereafter, genome-wide association studies (using [PLINK]( https://www.cog-genomics.org/plink/2.0/))
   
   StatGen tutorial can be found here: [tutorial](https://github.com/BiomedSciAI/Geno4SD/blob/main/tutorials/StatGen.ipynb)
   
5. **MaSk-LMM: _Matrix Sketching-based Linear Mixed Models_** is a method to compute linear mixed models which are widely used to perform genome-wide association studies on large biobank-scale genotype data using advances in randomized numerical linear algebra. 

   MaSk-LMM tutorial can be found here: [tutorial](https://github.com/BiomedSciAI/Geno4SD/blob/main/tutorials/MaSkLMM_tutorial.ipynb)
   
## Installation and Tutorials
In our detailed [Online Documentation](https://biomedsciai.github.io/Geno4SD/) you'll find:
* Installation [instructions](https://biomedsciai.github.io/Geno4SD/source/installation.html#install-geno4sd).  
* An overview of Geno4SD's [main components](https://biomedsciai.github.io/Geno4SD/source/overview.html) and [API](https://biomedsciai.github.io/Geno4SD/api/geno4sd.html)
* An end-to-end [tutorial](https://biomedsciai.github.io/Geno4SD/source/tutorial.html) using a publicly available dataset.

