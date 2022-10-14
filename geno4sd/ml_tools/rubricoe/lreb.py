import scipy.stats as st
import numpy



class LinRidgeRegSVD(object):
    """
    Class to compute a linear regressor with L2 regularization and error bars, using SVD decomposition.
    
    Attributes
    ----------
    
    C
        L2 (Ridge) regression parameter.
    B
        Regression coefficients.
    DB 
        Regression coefficient uncertainties.
    PVal 
        Estimated p-value that the coefficient is distinct from 0.
    Power 
        Probability that a true positive won't be rejected.
    coef_
        Estimated relevance of a coefficient in terms of its error bars.
    
    """
    def __init__(self, C=1.0):
        self.C = C
        self.unc = None
        

    def fit(self,X,y,unc=None):
        '''Computes a ridge linear regression returning error bars.

        Parameters
        ----------
        X: array-like
            Data to fit, each row is a data point.
        y: array-like
            Labels of data.
        unc: array-like, optional
            Uncertainties of labels y. Defaults to stddev of labels.

        ''' 

        if len(y.shape)>1:
            raise ValueError("Label data must be a 1d array.")
        if unc is not None:
            unc = unc
        else:
            unc = numpy.full(y.shape,numpy.std(y))
        unc = numpy.square(unc).astype(numpy.float32)
        self.B, self.DB = _fit_class(X,y,self.C,unc)

        self.PVal = 2.0 * st.norm.sf(numpy.abs(self.B) / (1e-8 + self.DB))
        self.Pow = st.norm.cdf((numpy.abs(self.B) - 1.96 * self.DB)/ (1e-8 + self.DB))
        self.coef_ = numpy.abs(self.B/(1e-8 + self.DB))
        return self


def _fit_class(X,y,C,unc):
    '''Computes a ridge linear regression returning error bars.

    Parameters
    ----------
    X: array-like
        Data to fit, each row is a data point.
    y: array-like
        Labels of data.
    unc: array-like, optional
        Uncertainties of labels y. Defaults to stddev of labels.

    ''' 
    X = numpy.hstack((X, numpy.ones((X.shape[0],1)))).astype(numpy.float32)
    U, S, VT = numpy.linalg.svd(X, full_matrices = False)
    Xi = numpy.dot(U, numpy.diag(S))

    Xi_T = numpy.transpose(Xi)

    SigmaM1 = numpy.diag(numpy.reciprocal(unc)).astype(numpy.float32)

    CI = numpy.diag(numpy.full(Xi.shape[1], C)).astype(numpy.float32)

    SM1 = numpy.dot(numpy.dot(Xi_T, SigmaM1), Xi)
    SM1CI = SM1 + CI
    S = numpy.linalg.pinv(SM1CI)
    Gamma = numpy.dot(numpy.dot(S, numpy.dot(Xi_T, SigmaM1)), y)
    VarGamma = numpy.dot(S, numpy.dot(SM1, S))
    B = numpy.dot(Gamma, VT)
    DB = numpy.zeros(B.shape[0])
    for i in range(B.shape[0]):
        DB[i] = numpy.sqrt(numpy.dot(VT[:,i], numpy.dot(VarGamma, VT[:, i])))
    return B, DB


