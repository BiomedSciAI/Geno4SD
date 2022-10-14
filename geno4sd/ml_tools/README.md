## Geno4S: ML and AI components

ML/AI components of Geno4SD include:

- ReVeaL: _Rare Variant Learning_, is a stochastic regularization-based learning algorithm. It partitions the genome into non-overlapping, possibly non-contiguous, windows (_w_) and then aggregates samples into possibly overlapping subsets, using subsampling with replacement (stochastic), giving units called shingles that are utilized by a statistical learning algorithm. Each shingle captures a distribution of the mutational load (the number of mutations in the window _w_ of a given sample), and the first four moments are used as an approximation of the distribution.
- RubricOE: is a cross-validated  machine learning framework with feature ranking described and multiple levels of cross validation to obtain interpretable genetic and non-genetic features from genomic or transcriptomic data combined.
