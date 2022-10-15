ReVeaL
============

ReVeaL, *Rare Variant Learning*, is a stochastic regularization-based learning algorithm.

Method
-------

ReVeaL partitions the genome into non-overlapping, possibly non-contiguous, windows (*w*) and then aggregates samples into possibly overlapping subsets, using subsampling with replacement (stochastic), giving units called shingles that are utilized by a statistical learning algorithm. Each shingle captures a distribution of the mutational load (the number of mutations in the window *w* of a given sample), and the first four moments are used as an approximation of the distribution.

.. image:: img/reveal_flowchart.jpeg


The entire ReVeaL pipeline can be executed by:

.. autosummary::
    ~geno4sd.ml_tools.ReVeaL.compute


Alternative, one can using the following API:

Pre-processing
^^^^^^^^^^^^^^

Collection of common pre-processing functionalities.

.. autosummary::
    ~geno4sd.ml_tools.ReVeaL.compute_mutational_load
    ~geno4sd.ml_tools.ReVeaL.train_test_split
    ~geno4sd.ml_tools.ReVeaL.permute_labels
   

Shingle Computation
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    ~geno4sd.ml_tools.ReVeaL.compute_shingle
	

An example of usage can be found in this  `tutorial <https://github.com/ComputationalGenomics/Geno4SD/blob/main/tutorials/ReVeaL.ipynb>`_


Citation
--------

Please cite the following article if you use ReVeaL:

Parida L, Haferlach C, Rhrissorrakrai K, Utro F, Levovitz C, Kern W, et al. (2019) Dark-matter matters: Discriminating subtle blood cancers using the darkest DNA. PLoS Comput Biol 15(8): e1007332. https://doi.org/10.1371/journal.pcbi.1007332