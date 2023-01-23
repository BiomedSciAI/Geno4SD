######################################################
## Myson Burch
## Computational Genomics Team, IBM Research, NY
## Purdue University, West Lafayette, IN
## mcburch@purdue.edu
######################################################

#############################################################
##############   Packages & Helper Functions   ##############
#############################################################
import pandas as pd
import numpy as np
import sys, os, time, math, scipy, warnings, random
from scipy import optimize, linalg
import scipy.stats as stats
from pysnptools.snpreader import Bed, Pheno
from pysnptools.kernelreader import SnpKernel
from pysnptools.standardizer import Unit
from pysnptools.util import intersect_apply
from sklearn.decomposition import PCA

def get_data(bed_fn, pheno_fn, cov_fn = None):
    """
    Function reading in data and performing normalization

    :param bed_fn: Filename of of PLINK Bed-formatted files (ex: "/path/to/files/toy.bed")

    :param pheno_fn: Filename of of PLINK phenotype-formatted files (ex: "/path/to/files/toy.phen")

    :param cov_fn: Set to 'None' as top 2 principal components computed after sketching
    """

    ignore_in = None
    #store snp data from Bedfile
    snp_on_disk = Bed(bed_fn, count_A1=False)
    kernel_in = SnpKernel(Bed(bed_fn, count_A1=False), Unit())
    #store covariates
    if cov_fn != None:
        cov_on_disk = Pheno(cov_fn)
    #store phenotype
    pheno_on_disk = Pheno(pheno_fn)

    #intersect and sort IDs from bed, pheno, cov returning new data with same IDs
    if cov_fn != None:
        ignore_out, bed_out, pheno_out, cov_out = intersect_apply([ignore_in,
                                                       snp_on_disk,
                                                       pheno_on_disk,
                                                       cov_on_disk])
    else:
        ignore_out, bed_out, pheno_out = intersect_apply([ignore_in,
                                                       snp_on_disk,
                                                       pheno_on_disk])

    sample_id = bed_out.iid
    num_samples = bed_out.iid_count
    snp_id = bed_out.sid
    snp_chrom = bed_out.pos
    snp_chrom[0][1] = 0; snp_chrom[0][2] = 0; snp_ids_and_chrom = pd.DataFrame(np.hstack([snp_id.reshape(snp_id.shape[0], 1),snp_chrom]))
    snp_ids_and_chrom.columns = ['SNP','Chr','ChrPos', 'ChrPos2']
    num_snps = bed_out.sid_count
    if cov_fn != None:
        num_covars = cov_out.sid_count

    #read pheno, cov values
    pheno = pheno_out.read().standardize().val
    if cov_fn != None:
        cov = cov_out.read().val
    #read and standardize SNP values
    normZ = bed_out.read(dtype=np.float32).standardize().val
    if cov_fn == None:
        cov = None
        num_covars = 2

    return normZ, pheno, cov, snp_ids_and_chrom, num_samples, num_snps, num_covars

def get_K_sketch(Z, sk_sz, type="clkwdf"):
    """
    Function computing sketched relatedness matrix

    :param Z: Sketched genotype matrix

    :param sk_sz: Sketch dimension to be used in GRM calculation

    :param type: Flag for different types of sketching. Currently, support count sketching and gaussian projections
    """

    if type != "clkwdf":
        # marker sketching for GRM
        n, m = Z.shape
        sketch_n_cols = int(sk_sz*m)
        S = np.random.randn(m, sketch_n_cols) / np.sqrt(sketch_n_cols)
        Z_sketched = Z @ S
        return (1/m)*(Z_sketched @ Z_sketched.T)
    else:
        # marker sketching for GRM
        n, m = Z.shape
        sketch_cols = int(sk_sz*m)
        Z_sketched = linalg.clarkson_woodruff_transform(Z.T, sketch_cols, seed = 13).T
        return (1/m)*(Z_sketched @ Z_sketched.T)

def sample_sketch(M, sk_sz, type="clkwdf"):
    """
    Function performing sketching on the samples for the genotype and phenotype matrices

    :param M: Matrix / vector to be sketched (genotype and phenotype matrix in separate calls)

    :param sk_sz: Sketch dimension to be used on the samples

    :param type: Flag for different types of sketching. Currently, support count sketching and gaussian projections
    """

    if type != "clkwdf":
        # sample sketching for Z, y, X
        n, _ = M.shape
        sketch_rows = int(sk_sz*n)
        S = np.random.randn(sketch_rows, n) / np.sqrt(sketch_rows)
        M_sketched = S @ M
        return M_sketched
    else:
        # sample sketching for Z, y, X
        n, _ = M.shape
        sketch_rows = int(sk_sz*n)
        M_sketched = linalg.clarkson_woodruff_transform(M, sketch_rows, seed = 13)
        return M_sketched

def get_H_tau(tau0, n, K):
    """
    Function computing H_tau (see paper for details)

    :param tau0: Current value for tau estimate

    :param n: Number of samples

    :param K: Sketched GRM matrix
    """
    return K + (tau0*np.identity(n))

def get_U_term(X):
    """
    Function computing U_term (see paper for details)

    :param X: Matrix of covariates computed after sketching
    """
    U_x, S, _ = np.linalg.svd(X, full_matrices=False)
    return np.identity(X.shape[0]) - (U_x @ U_x.T)

def get_projected_matrix(U_term, H_tau):
    """
    Function computing projection term (see paper for details)
    """
    UHU = U_term @ H_tau @ U_term
    return UHU

def estimate_variance_comp(y, M_inv, n, c):
    """
    Function computing sigma g squared term (see paper for details)

    :param y: Sketched phenotype vector / matrix

    :param M_inv: Pseudoinverse term (see paper for details)

    :param n: Number of samples

    :param c: Number of covariates
    """
    num = (y.T @ M_inv @ y)[0][0]
    den = n - c
    return num/den

def get_lle(x, n, m, c, pheno, K, U_term):
    """
    Function computing log-likelihood function at estimated values (see paper for details)

    :param x: Current value for tau estimate

    :param n: Number of samples

    :param m: Number of markers

    :param c: Number of covariates

    :param pheno: Sketched phenotype vector / matrix

    :param K: Sketched GRM matrix

    :param U_term: see 'get_U_term()' for details
    """
    H_tau = get_H_tau(x, n, K)
    P = get_projected_matrix(U_term, H_tau)
    P[abs(P) < 0.1] = 0
    P_inv = np.linalg.pinv(P, hermitian = True)
    sigma_g = abs(estimate_variance_comp(pheno, P_inv, n, c))
    fact = (n - c)/2
    comp1 = -1*fact*np.log(2*math.pi)
    comp2 = -1*fact*np.log(sigma_g)
    sign_logdet, logdet = np.linalg.slogdet(P)
    # subtract logdet(P) or add logdet(P^-1) to compute lle correctly
    comp3 = (1/2)*logdet # equivalent to: -(1/2)*sign_logdet*logdet
    comp4 = -1*fact
    result = np.sum([comp1, comp2, comp3, comp4])
    return result

def get_pvals(Z, Y, X, H_tau, sigma_e, sigma_g, LOCO = False):
    """
    Function computing test statistics

    :param Z: Sketched genotype matrix

    :param Y: Sketched phenotype vector / matrix

    :param X: Matrix of covariates computed after sketching

    :param H_tau: see 'get_H_tau()' for details

    :param sigma_e: Estimate for sigma e squared

    :param sigma_g: Estimate for sigma g squared

    :param LOCO: Boolean flag to enable leave-one-chromosome-out validation
    """
    Y = (Y - np.mean(Y)).flatten()
    n, m = Z.shape
    _, c = X.shape
    # inverse of covariance matrix to compute chi-square statistics
    # V = sigma_g*K + sigma_e*np.identity(n)
    Vinv = np.linalg.pinv(sigma_g*H_tau, hermitian = True)
    # Vectorized Computation
    num_vec = np.square(Z.T @ Vinv @ Y)
    den_vec = np.einsum('...i,...i->...', Z.T @ Vinv, Z.T)
    chi2stats = np.absolute( np.array(num_vec / den_vec).reshape((m, 1)) )
    pvals = stats.f.sf(chi2stats, 1, n-(c+1))[:,0]
    return pvals

def get_chisq(Z, Y, X, H_tau, sigma_e, sigma_g):
    """
    Function computing chi-squared statistics (for LOCO implementation)

    :param Z: Sketched genotype matrix

    :param Y: Sketched phenotype vector / matrix

    :param X: Matrix of covariates computed after sketching

    :param H_tau: see 'get_H_tau()' for details

    :param sigma_e: Estimate for sigma e squared

    :param sigma_g: Estimate for sigma g squared
    """
    Y = (Y - np.mean(Y)).flatten()
    n, m = Z.shape
    _, c = X.shape
    # inverse of covariance matrix to compute chi-square statistics
    # V = sigma_g*K + sigma_e*np.identity(n)
    Vinv = np.linalg.pinv(sigma_g*H_tau, hermitian = True)
    # Vectorized Computation
    num_vec = np.square(Z.T @ Vinv @ Y)
    den_vec = np.einsum('...i,...i->...', Z.T @ Vinv, Z.T)
    chi2stats = np.absolute( np.array(num_vec / den_vec).reshape((m, 1)) )
    c_inf = np.mean(np.random.choice(chi2stats, 30, replace = False))
    chi2stats = chi2stats / c_inf
    return chi2stats

def MaSkLMM(normZ, pheno, cov, num_samples, num_snps, num_covars, sample_sketch_size, marker_sketch_size, LOCO, snp_ids_and_chrom, maxiters):
    """
    Function performing matrix sketching-based linear mixed modeling for association studies.

    :param normZ: Normalized genotype matrix

    :param pheno: Normalized phenotype vector / matrix

    :param cov: Set to 'None' as top 2 principal components computed after sketching

    :param num_samples: Number of samples

    :param num_snps: Number of SNPs

    :param num_covars: Number of covariates

    :param sample_sketch_size: Sketch dimension to use for the number of samples in the input (given 'n' samples, the sketch will have 'n * sample_sketch_size' rows)

    :param marker_sketch_size: Sketch dimension to use for the number of markers when computing the GRM (given 'm' markers, the sketch will have 'm * marker_sketch_size' columns)

    :param LOCO: Boolean flag to enable leave-one-chromosome-out validation

    :param snp_ids_and_chrom: Array of labels for rsIDs and corresponding chromosomes

    :param maxiters: Maximum number of iterations to be used in the Newton-Raphson estimation
    """

    normZ = sample_sketch(normZ, sample_sketch_size)
    pheno = sample_sketch(pheno, sample_sketch_size)
    pca = PCA(n_components=num_covars)
    # sketched PCA
    Xt = pca.fit_transform(normZ)
    cov = np.asarray(Xt)
    num_samples, num_snps = normZ.shape
    # flag for leave-one-chromosome-out analysis
    if not LOCO:
        # Compute GRM and projection terms ("constants")
        K = get_K_sketch(normZ, marker_sketch_size)
        U_term = get_U_term(cov)
        # scipy newton raphson (using secant to auto-compute the derivate solved issue of divergence)
        root, newton_output = optimize.newton(get_lle, 1.0, args=( num_samples,
                                                                   num_snps,
                                                                   num_covars,
                                                                   pheno,
                                                                   K,
                                                                   U_term, ), rtol=1e-3, full_output = True, maxiter=maxiters, disp = False)
        # generate test statistics
        H_tau = get_H_tau(root, num_samples, K)
        P = get_projected_matrix(U_term, H_tau)
        P[abs(P) < 0.1] = 0
        P_inv = np.linalg.pinv(P, hermitian = True)
        sigma_g = estimate_variance_comp(pheno, P_inv, num_samples, num_covars)
        pvals = get_pvals(normZ, pheno, cov, H_tau, abs(root)*sigma_g, sigma_g)
    else:
        pvals = []
        chi2stats = []
        # Compute projection term ("constant")
        U_term = get_U_term(cov)
        # determine chromosomes in the data
        chroms = np.sort(np.unique(snp_ids_and_chrom.iloc[:,1]).astype(float))
        for chrom in chroms:
            # print("Chromosome:",chrom)
            all_snp_idx = snp_ids_and_chrom.index.tolist()
            test_snp_idx = snp_ids_and_chrom.index[snp_ids_and_chrom['Chr'] == str(chrom)].tolist()
            # estimating components with the test SNPs excluded
            Z_estim = normZ[:, np.setdiff1d(all_snp_idx, test_snp_idx)]
            num_snps = Z_estim.shape[1]
            K = get_K_sketch(Z_estim, marker_sketch_size)
            # scipy newton raphson (using secant to auto-compute the derivate solved issue of divergence)
            root, newton_output = optimize.newton(get_lle, 0.5, args=( num_samples,
                                                                       num_snps,
                                                                       num_covars,
                                                                       pheno,
                                                                       K,
                                                                       U_term, ), rtol=1e-3, full_output = True, maxiter=maxiters, disp = False)
            # generate test statistics
            H_tau = get_H_tau(root, num_samples, K)
            P = get_projected_matrix(U_term, H_tau)
            P[abs(P) < 0.1] = 0
            P_inv = np.linalg.pinv(P, hermitian = True)
            sigma_g = estimate_variance_comp(pheno, P_inv, num_samples, num_covars)
            # computing test statistics on the excluded SNPs
            chi2stats.extend( list(get_chisq(normZ[:, test_snp_idx], pheno, cov, K, abs(root)*sigma_g, sigma_g)) )
        chi2stats = np.array(chi2stats)
        pvals = stats.f.sf(chi2stats, 1, num_samples-(num_covars+1))[:,0]
    return pvals, newton_output

def run(bed_fn, pheno_fn, cov_fn = None, sample_sketch_size = 0.5, marker_sketch_size = 0.5, LOCO = False, maxiters = 1):
    """
    Function performing matrix sketching-based linear mixed modeling for association studies.

    :param bed_fn: Filename of of PLINK Bed-formatted files (ex: "/path/to/files/toy.bed")

    :param pheno_fn: Filename of of PLINK phenotype-formatted files (ex: "/path/to/files/toy.phen")

    :param cov_fn: Set to 'None' as top 2 principal components computed after sketching

    :param sample_sketch_size: Sketch dimension to use for the number of samples in the input (given 'n' samples, the sketch will have 'n * sample_sketch_size' rows)

    :param marker_sketch_size: Sketch dimension to use for the number of markers when computing the GRM (given 'm' markers, the sketch will have 'm * marker_sketch_size' columns)

    :param LOCO: Boolean flag to enable leave-one-chromosome-out validation

    :param maxiters: Maximum number of iterations to be used in the Newton-Raphson estimation

    :Example:
    >>> import MaSkLMM
    >>> MaSkLMM_results_df, newton = MaSkLMM.run("toy.bed", "toy.phen", cov_fn = None, sample_sketch_size = 0.5, marker_sketch_size = 0.5, LOCO = False, maxiters = 10)
    """

    normZ, pheno, cov, snp_ids_and_chrom, num_samples, num_snps, num_covars = get_data(bed_fn, pheno_fn, cov_fn)
    pvals, newton = MaSkLMM(normZ, pheno, cov, num_samples, num_snps, num_covars, sample_sketch_size, marker_sketch_size, LOCO, snp_ids_and_chrom, maxiters)

    # generate output
    MaSkLMM_results_df = pd.DataFrame(np.hstack([ snp_ids_and_chrom, pvals.reshape(pvals.shape[0], 1) ]))
    MaSkLMM_results_df.columns = ['SNP','Chr','ChrPos', 'ChrPos2','PValue']
    MaSkLMM_results_df.sort_values(by=['PValue'],inplace=True)
    MaSkLMM_results_df["ChrPos2"] = MaSkLMM_results_df["ChrPos2"].astype(float)
    MaSkLMM_results_df["Chr"] = MaSkLMM_results_df["Chr"].astype(float).astype(int)
    MaSkLMM_results_df = MaSkLMM_results_df.drop("ChrPos", axis = 1)
    MaSkLMM_results_df["PValue"] = MaSkLMM_results_df["PValue"].astype(float)

    return MaSkLMM_results_df, newton
