_author__ = "Aritra Bose"
__copyright__ = "Copyright 2023, IBM Research"
__version__ = "0.1"
__maintainer__ = "Aritra Bose"
__email__ = "a.bose@ibm.com"
__status__ = "Development"

#This code computes the Quality Control of a genotype or whole-genome sequencing data. 

import subprocess as sp
import os, sys, re

def create_dir(path=None,
               proj_name=None):
    """
        Create directory to store the final and intermediate files before/after the Quality Control of the genetic data. 

        Parameters:
        ------------

        path: A string containing the path to the user-defined directory for parking the output data.

        proj_name: User-defined name of the project. This will be a sub-directory inside the path. 

        Returns:
        ------------

        Path with the project name as sub-directory
    
    """
    if ((path is None) | (proj_name is None)):
        print("\n Missing location! Please ensure all paths exist.\n")
        exit()
    
    if os.path.exists(path+'/'+proj_name):
        print('\n Directory exists. \n')
    else:
        create_dir = """
                         mkdir {path}/{directory}
                     """
        create_dir = create_dir.format(path=path, directory=proj_name).strip('\n')
        proc = sp.Popen(create_dir, shell=True)
        proc.wait()
    
    if os.path.exists(path+'/'+proj_name):
        print("Successfully created directory")
    else:
        print("Error creating directory")
        
    return path+'/'+proj_name



def keep_indivs(plink_path=None, 
                master_data_path=None, 
                samples_file_path=None, 
                output_data_path=None, 
                proj_name=None):
    
    
    """
        Method to extract list of given individuals for the analyses from the master genotype data. 
        The master data contains chromosomes 1-22, X, Y, XY. 
        The function uses PLINK to extract the samples from the data.  It checks whether the files have been generated and returns an error if there was a problem with execution of the command. 

        Parameters:
        ------------

        plink_path: Path where the PLINK package is stored. This path points to the directory where both PLINKv1.9 (./plink) and PLINKv2 (./plink2) are stored.

        master_data_path: Path where the master data resides. This is a large merged genotype data with all samples and all chromosomes. This data had been produced after preliminary filtering of high-quality imputed SNPs. 

        samples_file_path: Path with the IDs of the samples file. This file contains tab-delimited two columns reflecting the family ID (FID) and individual ID (IID)

        output_data_path: Path to output data, this is returned by create_dir()

        proj_name: User-defined name of the project. This will be a sub-directory inside the path. 

        Returns:
        ------------

        PLINK --keep log file with stdout of the command. The files with subset of individuals are in the output_data_path. 
    
    """

    if ((plink_path is None) | 
        (master_data_path is None) | 
        (samples_file_path is None) | 
        (output_data_path is None) | 
        (proj_name is None)):
        
        print("\n Missing location! Please ensure all paths exist.\n")
        exit()       
    
    keep_cmd = """
                   {plink} --pfile {master} --keep {samples} --make-pgen --out {output}/{project}
               """
    keep_cmd = keep_cmd.format(plink=plink_path, 
                               master = master_data_path,
                               samples = samples_file_path,
                               output = output_data_path,
                               project= proj_name).strip('\n')
    print(keep_cmd)
    proc = sp.Popen(keep_cmd, shell=True, stdout=sp.PIPE)
    output_keep = proc.stdout.read()
    keep_log = re.sub(r'\r', ' ', re.sub(r'\n',' ', re.sub(r'\x08+',' ',output_keep.decode('utf-8'))))
    proc.wait()
    
    if os.path.exists(output_data_path+'/'+proj_name+'.pgen'):
        print("Successfully created genotype file with matching samples")
    else:
        print("Error in execution")
        
    return keep_log


def exec_qc(qc_path=None, 
            plink_path=None, 
            data_path=None, 
            output_data_path=None, 
            proj_name=None, 
            r_scripts=None, 
            py_scripts=None, 
            mkl_path=None, 
            terapca_path=None):
 
    """
        Method to execute the QC script, for which the path is specified by the user. This script saves the stdout of the QC and returns it.  
        The script also computes PCA required for population stratification correction for GWAS and needs pre-installed Intel MKL compiler and TeraPCA (https://github.com/aritra90/TeraPCA).
        
        Parameters:
        ------------
        qc_path: Path to the QC script which contains all the QC steps. 
        
        plink_path: Path where the PLINK package is stored. This path points to the directory where both PLINKv1.9 (./plink) and PLINKv2 (./plink2) are stored.

        data_path: Path where the extracted data for the samples mentioned in keep_indivs() returned. 
        
        output_data_path: Path to output data, this is returned by create_dir()

        proj_name: User-defined name of the project. This will be a sub-directory inside the path. 
        
        r_scripts: Path to R sripts for heterozygosity and incorrect sex filtering. 
        
        py_scripts: Path to Python scripts for sample relatedness filtering. 
        
        mkl_path: Path to INTEL OneAPI. This must be preinstalled to compute PCA using TeraPCA. 
        
        terapca_path: Path to TeraPCA.exe for computing PCA. 
        

        Returns:
        ------------

        QC log file with stdout of the command. The intermediate and final QCed files are stored in the output path. 
    
    """
            
    if ((qc_path is None) |
        (plink_path is None) | 
        (data_path is None) |
        (output_data_path is None) | 
        (proj_name is None) | 
        (r_scripts is None) | 
        (py_scripts is None) | 
        (mkl_path is None) | 
        (terapca_path is None)):
        
        print("\n Missing location! Please ensure all paths exist.\n")
        exit()

        
        
    exec_qc_cmd = """
                    bash {qc} --plink {plink} --data {data}\
                    --output {output} --project_name {proj_name}\
                    --scripts1 {r_scripts} --scripts2 {py_scripts}\
                    --mkl {mkl} --terapca {terapca}
                  """
    exec_qc_cmd = exec_qc_cmd.format(qc=qc_path,
                                     plink=plink_path,
                                     data=data_path,
                                     output=output_data_path,
                                     proj_name=proj_name,
                                     r_scripts=r_scripts,
                                     py_scripts=py_scripts,
                                     mkl=mkl_path,
                                     terapca=terapca_path).strip("\n")
    print(exec_qc_cmd)
    proc = sp.Popen(exec_qc_cmd, shell=True, stdout=sp.PIPE)
    output_qc = proc.stdout.read()
    qc_log = re.sub(r'\r', ' ', re.sub(r'\n',' ', re.sub(r'\x08+',' ',output_qc.decode('utf-8'))))

    proc.wait()

    if os.path.exists(output_data_path+'/'+'qc_final.bed'):
        print("Successfully created genotype file after quality control")
    else:
        print("Error in execution")

    return qc_log