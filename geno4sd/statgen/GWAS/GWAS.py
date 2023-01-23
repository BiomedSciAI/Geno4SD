_author__ = "Aritra Bose"
__copyright__ = "Copyright 2023, IBM Research"
__version__ = "0.1"
__maintainer__ = "Aritra Bose"
__email__ = "a.bose@ibm.com"
__status__ = "Development"

import subprocess as sp
import os, sys, re
import pandas as pd
from glob import glob


from qmplot import manhattanplot, qqplot
import matplotlib.pyplot as plt

def create_dir(path=None):
    """
        Create directory to store the GWAS results (Association test file and plots) 

        Parameters:
        ------------

        path: A string containing the path to the user-defined directory for parking the GWAS output.
        

        Returns:
        ------------

        Path with the GWAS output directory. 
    
    """
    if (path is None):
        print("\n Missing location! Please ensure all paths exist.\n")
        exit()
    
    if os.path.exists(path+'/GWAS'):
        print('\n Directory exists. \n')
    else:
        create_dir = """
                         mkdir {path}/GWAS
                     """
        create_dir = create_dir.format(path=path).strip('\n')
        proc = sp.Popen(create_dir, shell=True)
        proc.wait()
    
    if os.path.exists(path+'/GWAS'):
        print("Successfully created directory")
    else:
        print("Error creating directory")
        
    return path+'/GWAS'


def get_manhattan_plot(df, 
                       title=None,
                       fname=None):
    
    """
        Plotting Manhattan plot from the GWAS output. 
        Prerequisite: Python package qmplot (https://github.com/ShujiaHuang/qmplot)
        
        Parameters:
        ------------
        
        df: Pandas dataframe representation of the GWAS output.
        
        title: Title of the plot. Often, the project name. 
        
        fname: filename (with path) to be used for the Manhattan plot.
        

        Returns:
        ------------

        Nothing. It saves the file in the output path mentioned in fname.
    
    """
    
    f, ax = plt.subplots(figsize=(12,4), facecolor='w', edgecolor='k')
    xtick = set(['chr' + i for i in list(map(str, range(1, 10))) + 
             ['11', '13', '15', '18', '21', 'X']])
    
    manhattanplot(data=df,
                  marker=".",
                  sign_marker_p=1e-6,  # Genome wide significant p-value
                  sign_marker_color="g",
                  snp="ID",
                  genomewideline=1e-6,
                  title=title,
                  xtick_label_set=xtick,
                  xlabel="Chromosome",
                  ylabel=r"$-log_{10}{(P)}$",

                  sign_line_cols=["#D62728", "#2CA02C"],
                  hline_kws={"linestyle": "--", "lw": 1.3},
                  is_show = True,
                  is_annotate_topsnp=True,
                  ld_block_size=50000,  # 50000 bp
                  text_kws={"fontsize": 12,  # The fontsize of annotate text
                            "arrowprops": dict(arrowstyle="-", color="k", alpha=0.6)},

                  dpi=600,
                  figname=fname,
                  ax=ax)
    
def get_qq_plot(df, 
                title=None,
                fname=None):
 
    """
        Plotting QQ plot from the GWAS output. 
        Prerequisite: Python package qmplot (https://github.com/ShujiaHuang/qmplot)
        
        Parameters:
        ------------
        
        df: Pandas dataframe representation of the GWAS output. 
        
        title: Title of the plot. Often, the project name. 
        
        fname: filename (with path) to be used for the Manhattan plot.
        

        Returns:
        ------------

        Nothing. It saves the file in the output path mentioned in fname.
    
    """
    
    f, ax = plt.subplots(figsize=(6, 6), facecolor="w", edgecolor="k")
    qqplot(data=df["P"],
           marker="o",
           title=title,
           is_show=True,
           xlabel=r"Expected $-log_{10}{(P)}$",
           ylabel=r"Observed $-log_{10}{(P)}$",
           dpi=600,
           figname=fname,
           ax=ax)


def exec_gwas(plink_path=None, 
              data_path=None, 
              covar_path=None, 
              output_data_path=None, 
              proj_name=None):
    
    
    """
        Method to perform association test using PLINKv2 --glm. 
        
        Parameters:
        ------------
        
        plink_path: Path where the PLINK package is stored. This path points to the directory where both PLINKv1.9 (./plink) and PLINKv2 (./plink2) are stored.

        data_path: Path where the QCed data is stored.
        
        covar_path: Path where the covariate files are stored. The covariates in a GWAS usually are demographic variables such as age and sex, genomic batch information and genetic principal components (PC). 
        
        output_data_path: Path to output data, where the GWAS results will be stored.

        proj_name: User-defined name of the project. This will be a sub-directory inside the path. 
        

        Returns:
        ------------

        QC log file with stdout of the command. The intermediate and final QCed files are stored in the output path. 
    
    """
    if ((plink_path is None) |(data_path is None) | (output_data_path is None)):
        print("\n Missing location! Please ensure all paths exist.\n")
        exit()
     
    gwas_data_path = create_dir(output_data_path)
        
    gwas_cmd = """
                   {plink} --bfile {data} --covar {covar}\
                   --covar-variance-standardize --ci 0.95\
                   --pfilter 0.05 --glm hide-covar\
                   --out {output}/{project}
               """
    gwas_cmd = gwas_cmd.format(plink=plink_path,
                               data=data_path,
                               covar=covar_path,
                               output=gwas_data_path,
                               project= proj_name).strip('\n')
    print(gwas_cmd)
    
    proc = sp.Popen(gwas_cmd, shell=True, stdout=sp.PIPE)
    output_gwas = proc.stdout.read()
    gwas_log = re.sub(r'\r', ' ', re.sub(r'\n',' ', re.sub(r'\x08+',' ',output_gwas.decode('utf-8'))))
    
    proc.wait()
    
    if glob(output_data_path+'/*.logistic.hybrid'):
            print("Successfully created genotype file after quality control")
    else:
            print("Error in execution")
            
    return gwas_log, gwas_data_path



def plot(gwas_file_path=None,
         output_file_path=None,
         title=None):
    
    """
        Plot Manhattan plot and QQ plot from the GWAS output. 
        Prerequisite: Python package qmplot (https://github.com/ShujiaHuang/qmplot)
        
        Parameters:
        ------------
        
        gwas_file_path: Path to GWAS output.
        
        output_file_path: Path to store the GWAS plots.
        
        title: Title of the plots. This can be the project name (default). 
        

        Returns:
        ------------

        Manhattan and QQ plots in the user-defined output file path. 
    
    """
    
    
    if ((gwas_file_path is None) | (title is None)):
        print("\n Missing location! Please ensure all paths exist.\n")
        exit()
    
    df = pd.read_csv(gwas_file_path, sep='\t')
    manhattan_fname = output_file_path+'/'+title+'_manhattanPlot.png'
    qq_fname = output_file_path+'/'+title+'_qqPlot.png'
    
    get_manhattan_plot(df, title=title, fname=manhattan_fname)
    
    get_qq_plot(df, title=title, fname=qq_fname)
    
    if (glob(output_file_path+'/*.png')):
        
            print("Plotting GWAS data: Manhattan and QQ plots successful")
    else:
            print("Error in Plotting")
    
    