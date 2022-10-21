CuNA
============

CuNA or Cumulant-based Network Analysis finds higher-order genotype-phenotype relationships from multi-omic data with EHR information across different thresholds of statistical significance. 


Input 
^^^^^^^^^^^^^^
CuNA takes a csv file as input with the features in columns and samples in rows (see `./sample_data/CuNA_TCGA_sample_data.csv`). It is multithreaded and takes an argument for number of threads

Output 
^^^^^^^^^^^^^^

CuNA outputs three files: 

a) Network file with three columns, `v1`,`v2`,`count`, corresponding to two vertices and the interaction/edge term between them. 
b) A file with the communities corresponding to the features. 
c) A file with the node rank (importance of node)

Usage 
-------
See `./tutorial/CuNA.ipynb`

Contact 
-------
```
Aritra Bose (a dot bose at ibm dot com)
```
