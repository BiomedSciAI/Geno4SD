import struct, time, sys, multiprocessing as mp
import numpy as np
import pandas


# for each row (1st index) of samples, the columns contain allele info:
#   0 = homozygous in 1st allele listed in the .bim file
#   1 = missing information
#   2 = heterozygous -- one of each of the alleles in the .map file
#   3 = homozygous in the rt hand allele in the .bim
#  in https://www.cog-genomics.org/plink/1.9/formats#bed

def parse_binary_plink(filename, n_jobs=1, verbose=0, dataframe=False):
    """
        Reads bed, bim and fam files and parses their contents into three arrays, respectively, 
        or optionally into a dataframe.

        Parameters
        ----------
        filename: str
            Name (without extension) of the plink files to parse.
        n_jobs: int, default 1
            Number of threads to use when parsing.
        verbose: int, default 0
            Level of verbosity. 0 Omits all output.
        dataframe: bool, default False
            Flag to indicate whether to return three arrays or a pandas dataframe.

    """

    # Read in .bim file (genotype map file) w/ SNP label, chromosome, offset, and alleles
    with open(filename + '.bim', 'r') as fin:
        bim = [x.strip().split() for x in fin]

    V = len(bim)   # Number of variants
    bimA = np.array(bim)
    del bim

    # read in .fam file - lists family and individual ID's + parents, pheno, etc
    with open(filename + '.fam', 'r') as fin:
        fam = [x.strip().split() for x in fin]

    N = len(fam)   # number of samples
    famA = np.array(fam)
    del fam

    S = (N // 4  + (1 if N % 4 != 0 else 0)) * V + 3   # length of .bed file

    stime = time.time()
    
    bedBraw = mp.RawArray('B', S)
    bedB = np.frombuffer(bedBraw, dtype=np.ubyte).reshape(S)
    np.copyto(bedB, np.fromfile(filename + '.bed', dtype = np.ubyte, count = S))

    if verbose>0:
        print('time to depack = ', time.time() - stime)

    stime = time.time()
    
    #Space for results 
    #bedA = np.empty(shape = (N, V), dtype = np.ubyte)
    bedAraw = mp.RawArray('B', N * V)
    bedA = np.frombuffer(bedAraw, dtype=np.ubyte).reshape((N, V))
    
    if verbose>0:
        print('time to allocate space = ', time.time() - stime)
    
    #intermediate storage with rows of variants for samples in columns aligned
    #bedC = np.empty((V, (N // 4 + (1 if N % 4 != 0 else 0))), dtype = np.ubyte)
    bedCraw = mp.RawArray('B', V * (N // 4 + (1 if N % 4 != 0 else 0)))
    bedC = np.frombuffer(bedCraw, dtype=np.ubyte).reshape((V, (N // 4 + (1 if N % 4 != 0 else 0))))
    
    
    
    #align buffer with rows and columns to pick out nibbles
    def slicer1(bedBraw, bedCraw, ll, ul):
        bedB = np.frombuffer(bedBraw, dtype=np.ubyte).reshape(S)
        bedC = np.frombuffer(bedCraw, dtype=np.ubyte).reshape((V, (N // 4 + (1 if N % 4 != 0 else 0))))
        #for row in range(V):
        for row in range(ll, ul):
            lcut = 3 + (N // 4 + (1 if N % 4 != 0 else 0)) * row
            ucut = 3 + (N // 4 + (1 if N % 4 != 0 else 0)) * (row + 1)
            bedC[row, 0: ucut - lcut] = bedB[lcut: ucut] 
    
    splits = [(k * V) // n_jobs for k in range(n_jobs + 1)]
    
    procs = []
    for k in range(n_jobs):
        procs.append(mp.Process(target = slicer1, args = (bedBraw, bedCraw, splits[k], splits[k+1], )))
        procs[k].start()
    for k in range(n_jobs):
        procs[k].join()
        
    del bedBraw
    del bedB
    
    
    def slicer2(bedAraw, bedCraw, ll, ul):
        
        bedA = np.frombuffer(bedAraw, dtype=np.ubyte).reshape((N, V))
        bedC = np.frombuffer(bedCraw, dtype=np.ubyte).reshape((V, (N // 4 + (1 if N % 4 != 0 else 0))))

        for col in range(ll, ul):
            #bedA[col, :] = np.vectorize(lambda x: ((x >> ( 2 * (col % 4))) & 3))(bedC[:, col // 4])
            bedA[col, :] = np.right_shift(bedC[:, col // 4], (2 * (col % 4))) & 3
            
    splits = [(k * N) // n_jobs for k in range(n_jobs + 1)]
    
    procs = []
    for k in range(n_jobs):
        procs.append(mp.Process(target = slicer2, args = (bedAraw, bedCraw, splits[k], splits[k+1], )))
        procs[k].start()
    for k in range(n_jobs):
        procs[k].join()
        
    del bedCraw
    del bedC
    if verbose>0:
        print('time to decode = ', time.time() - stime)
    phenotypes = famA[:,5]

    if dataframe:
        return pandas.DataFrame(data = np.c_[bedA, phenotypes].astype("float32"), index = famA[:,0], columns=list(bimA[:,1])+["phenotype"])
    return bedA, bimA, famA
