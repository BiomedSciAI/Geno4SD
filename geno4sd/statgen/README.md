# Tutorial on how to do GWAS in Geno4SD

The entire GWAS pipeline can be broadly segmented into four parts: 

1. Extract genotype information for samples of interest from the master genotype file. 
2. Perform quality control (QC) on the extracted data.
3. Obtain top 50 Principal Components (PCs) from the QCed data. 
4. Perform genome-wide association studies (GWAS) on the QCed data with demographic variables and PCs as covariates. The associations are then visualized in a Manhattan plot and QQ plot indicating the false discovery rate. 

The sections of the code are briefly described below: 
* Define the paths of different packages required namely, scripts and packages as prerequisites, input/output data, covariates 
* Then we create a directory to store the QC files (intermediate and final) named after the project name (user-defined)
* After QC, we compute the PCA of the pruned QCed genotype files which are stored in a subdirectory called *PopStrat* withing the output directory. 
* Then we perform GWAS and visualize the results using Manhattan and QQ plots. These are stored within another subdirectory called *GWAS* within the *PopStrat* directory. 

The output directory structure looks like the following: 
```sequence
Project Name ➡ PopStrat ➡ GWAS
```

## QC 

QC is performed on the sampled genotype file (samples are user-defined) by the following filters.

1. Filtering both individuals and variants with at least **95% missing** data. 
2. Checking for problematic **sex assignment** in missing gender fields using the X chromosome. 
3. Filtering for variants with **minor allele frequency (MAF) < 0.05**.
4. Filtering variants which are not in Hardy-Weinberg equilibrium (HWE) with p-values at least 1e-16. This is done separately for cases and controls. P-value threshold for controls is higher. 
5. Removing individuals with high or low **heterozygosity rates**.
6. Removing **closely related individuals** with Identity-by-descent (IBD) method owing to cryptic relatedness. 
7. Remove multiallelic SNPs. (This is done for a further check. We already filtered for biallelic SNPs in the master data set.)

## PCA

Principal Component Analysis (PCA) is computed using the method TeraPCA (https://github.com/aritra90/TeraPCA) which scales to tera-scale genotypes. This requires Intel Math Kernel Library (MKL) compiler which can be found at https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html. 

__Note__:The Intel MKL is installed within OneAPI, which contains a script called *setvars.sh* which sets all environmental variables and links the path variable to them. We set the variables as part of the pipeline and needs the path to MKL before execution. 

## GWAS

GWAS is performed using PLINKv2 and uses the covariates as defined by the user. We use 95% CI in the association test and by-default obtain all those associations which are at least significant (p < 0.05). 


**Contact**
Aritra Bose (a.bose@ibm.com)
