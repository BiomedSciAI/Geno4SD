
import pandas
import os, os.path
import pickle
import json
def save_to_csv(x, path, base_filename, kind="list"):
    if kind=="list":
        pandas.DataFrame(x).to_csv(os.path.join(path,base_filename),index=False, header=False)
    if kind=="classifier":
        pandas.DataFrame(x.coef_).to_csv(os.path.join(path,base_filename),index=False, header=False)
    
def save_to_pickle(x,  path, base_filename):
    with open(os.path.join(path,base_filename),"wb") as file:
        pickle.dump(x, file)
        