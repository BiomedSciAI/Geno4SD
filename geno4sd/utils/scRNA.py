import scanpy as sc
import pandas as pd
import pathlib as pl

def _load_h5ad(input_file_name):
    
    if not pl.Path(input_file_name).resolve().is_file():
        raise AssertionError("File does not exist: %s" % str(input_file_name))
    else:
        adata = sc.read_h5ad(input_file_name)

    return adata

def _collapse_all_cell_per_sample(df):
     
    return df.groupby('sample').mean()

def preprocess_scRNA(input_file_name):

    #load data
    adata = _load_h5ad(input_file_name)

    #pre-process if needed
    
    #convert to dataframe
    df = adata.to_df()
    df = pd.concat([df,adata.obs[['sample','patient','timepoint']]], axis=1)

    #collapse gene

    df = _collapse_all_cell_per_sample(df)

    return df

    
