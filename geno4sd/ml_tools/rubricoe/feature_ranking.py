from .helper_funcs import save_to_pickle, save_to_csv
import multiprocessing as mp
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, make_scorer
import numpy
import time
import os


X = None
y = None


def _tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def _fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def _fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def _tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def _tpr(y_true, y_pred):
    return _tp(y_true, y_pred)/(_tp(y_true, y_pred)+_fn(y_true, y_pred))


def _tnr(y_true, y_pred):
    return _tn(y_true, y_pred)/(_tn(y_true, y_pred)+_fp(y_true, y_pred))


scoring = {'tp': make_scorer(_tp), 'tn': make_scorer(_tn),
           'fp': make_scorer(_fp), 'fn': make_scorer(_fn)}


def youden_index(y_true, y_pred):
    return _tpr(y_true, y_pred) + _tnr(y_true, y_pred) - 1


def _rank_features_job(test_size,
                      stratify_by,
                      clf,
                      fold_number,
                      details_files,
                      details_files_path,
                      seed):
    """
    Worker function that ranks feature indices of X, according to clf, from higher to lower importance. 
    Used by the main rank_features function.
    The final result is obtained through fitting on multiple data subsamplings, as many as number_of_folds.

    Parameters
    ----------
    test_size: float
        Proportion of samples (rows of X0) to be used for testing.
    fold_number: int
        Index of current data split that the worker is assigned to.
    clf
        Estimator used to produce scores. Expected to have an attribute `coef_` and `fit` and `predict` methods.
    details_files: bool
        Flag to enable or disable writing detailed population splits information to files.
    details_files_path: str, optional
        Path where output files will be written to. Not optional if `details_files` is `True`.
    seed: int, optional
        Seed value used for reproducibility. If not specified, no reproducibility is enforced.
    """
    global X
    global y
    idx_train, _ = train_test_split(
        range(X.shape[0]), test_size=test_size, stratify=stratify_by, random_state=seed)
    X_train = X[idx_train]
    y_train = y[idx_train]
    clf.fit(X_train, y_train)
    result = clf.coef_.ravel() / numpy.max(numpy.abs(clf.coef_.ravel()))
    if details_files:
        save_to_csv(idx_train, details_files_path,
                    "ranking_fold{}_train_indices.csv".format(fold_number))
        save_to_pickle(clf, details_files_path,
                       "ranking_fold{}_ranking_object.pickle".format(fold_number))

    return result, fold_number


def rank_features(clf,
                  X0,
                  y0,
                  test_size=.2,
                  number_of_folds=5,
                  verbose=0,
                  return_ranking_coefs=False,
                  n_jobs=3,
                  details_files=False,
                  details_files_path=None,
                  seed=None):
    """
    Returns ranked feature indices of X, according to clf, from higher to lower importance. 
    The final result is obtained through fitting on multiple data subsamplings, as many as number_of_folds.

    Parameters
    ----------
    clf:
        Estimator used to produce scores. Expected to have an attribute `coef_` and `fit` and `predict` methods.
    X0: array-like
        Dataset with rows observations and columns features/variables.
    y0: array-like
        Targets of X0.
    test_size: float, default .2
        Proportion of samples (rows of X0) to be used for testing.
    number_of_folds: int, default 5
        Number of splits for cross-validation purposes.
    n_jobs: int, default 1
        Number of workers used by the multiprocessing pool.
    verbose: int, default 0
        Level of stdout verbosity.
    n_features: int, optional
        Number of features of the dataset to use. If not specified, all features are used.
    details_files: bool, default False
        Flag to enable or disable writing detailed population splits information to files.
    details_files_path: str, optional
        Path where output files will be written to. Not optional if `details_files` is `True`.
    seed: int, optional
        Seed value used for reproducibility. If not specified, no reproducibility is enforced.
    """

    if verbose > 0:
        print("Ranking features...")
    global X
    global y
    X = X0
    y = y0
    influences = [None]*number_of_folds

    if seed is None:
        random_generator = numpy.random.RandomState()
    else:
        random_generator = numpy.random.RandomState(seed)
    fold_seeds = random_generator.randint(2**32, size=number_of_folds)

    def errors(result):
        print(result)

    def gather_influences(result):
        nonlocal influences
        x, fold_number = result
        influences[fold_number] = x
        if verbose > 0:
            print("Gathered fold {} of {}.".format(
                fold_number, number_of_folds), end="\n")

    params = [(test_size, y, clf, i, details_files, details_files_path,
               fold_seeds[i]) for i in range(number_of_folds)]
    pool = mp.Pool(n_jobs)
    for x in params:
        pool.apply_async(_rank_features_job, args=x,
                         callback=gather_influences, error_callback=errors)
    pool.close()
    pool.join()
    influences = numpy.array(influences)
    feature_coefs = numpy.sum(numpy.abs(influences), axis=0)
    sorted_features_indices = numpy.argsort(feature_coefs)[::-1]
    if return_ranking_coefs:
        return sorted_features_indices, numpy.sort(feature_coefs)[::-1]/number_of_folds
    return sorted_features_indices


def _compute_score(index, upper_bound):
    global X
    global y
    results = []
    start_time = time.time()
    for ix_train, ix_test, y_train, y_test in splits:
        X_train, X_test = X[ix_train, :upper_bound], X[ix_test, :upper_bound]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results.append(score_function(y_test, y_pred))
    return index, numpy.mean(results), time.time() - start_time, os.getpid()


def _init_compute(clf_, splits_,  details_files_, details_files_path_, score_function_=youden_index):
    """
    Initializer for multiprocessing workers, we use the fact that labels y are the same for all workers. Similarly for 
    classifiers, test size and scoring function.
    """
    global clf
    global splits
    global score_function
    global details_files
    global details_files_path
    clf = clf_
    splits = splits_
    score_function = score_function_
    details_files = details_files_
    details_files_path = details_files_path_


def score_curve(clf,
                X0,
                y0,
                step_size=10,
                test_size=.2,
                number_of_folds=5,
                n_jobs=1,
                score_function=youden_index,
                verbose=0,
                n_features=None,
                adaptive_features=False,
                tolerance_steps=10,
                window_size=10,
                details_files=False,
                details_files_path=None,
                seed=None
                ):
    """ Computes a score for a sequence of subsets of features.

    Computes a sequence of pairs (x_i, y_i) where y_i is the score of clf on a test set after 
    fitting a training subset of X[:x_i], as many subsets as number_of_folds (these subsets are fixed once selected). 
    The x_i are generated as a range from x_i to the total number of features in X, increasing by step_size.

    Parameters
    ----------
    clf:
        Estimator used to produce scores. Expected to have an attribute `coef_` and `fit` and `predict` methods.
    X0:
        Dataset with rows observations and columns features/variables.
    y0:
        Targets of X0.
    n_features: int, optional
        If specified, upper bound of ranked features used to generate score curves. Otherwise use all features.
    step_size:  int, default 10
        Number of features to be added at each step of the curve. 
    test_size: float, default .2
        Proportion of samples (rows of X0) to be used for testing.
    number_of_folds: int, default 5
        Number of splits for cross-validation purposes.
    n_jobs: int, default 1
        Number of workers used by the multiprocessing pool.
    score_function: Callable, default youden_index
        Callable that computes a score. Assumed to receive parameters `(y_true, y_pred)` and return a float.
    verbose: int, default 0
        Level of stdout verbosity.
    n_features: int, optional
        Number of features of the dataset to use. If not specified, all features are used.
    adaptive_features: bool, default False
        Flag to enable or disable early stopping.
    tolerance_steps: int, default 10
        Steps to wait while observing a downwards trend befor early stopping.
    window_size: int, 10
        Size of rolling average window used in early stopping.
    details_files: bool, default False
        Flag to enable or disable writing detailed population splits information to files.
    details_files_path: str, optional
        Path where output files will be written to. Not optional if `details_files` is `True`.
    seed: int, optional
        Seed value used for reproducibility. If not specified, no reproducibility is enforced.
    """
    if verbose > 0:
        print("Computing score curve...")
    global X
    global y
    X = X0
    y = y0
    if n_features is None:
        n_features = X.shape[1]
    if seed is None:
        random_generator = numpy.random.RandomState()
    else:
        random_generator = numpy.random.RandomState(seed)
    fold_seeds = random_generator.randint(2**32, size=number_of_folds)
    bounds = [x for x in range(step_size, n_features+1, step_size)]
    ix = range(len(X))
    splits = [train_test_split(ix,
                               y,
                               stratify=y,
                               test_size=test_size, random_state=fold_seeds[i]) for i in range(number_of_folds)]
    if details_files:
        for i, (idx_train, idx_test, _, _) in enumerate(splits):
            save_to_csv(idx_train, details_files_path,
                        "scoring_fold{}_train_indices.csv".format(i))
            save_to_csv(idx_test, details_files_path,
                        "scoring_fold{}_test_indices.csv".format(i))
    if n_features not in bounds:
        bounds.append(n_features)
    params = [(i, x) for i, x in enumerate(bounds)]
    sorted_scores = []
    previous_average = -1
    rolling_average_score_values = []
    tolerance = tolerance_steps
    previous_ucb = 0

    def upper_continuous_bound(l):
        def gen():
            i = 0
            while True:
                yield i
                i += 1
        return next(x for x in gen() if x not in l)

    def add_to_list_and_check(result):
        nonlocal sorted_scores
        nonlocal adaptive_features
        nonlocal verbose
        nonlocal tolerance
        nonlocal previous_average
        nonlocal previous_ucb
        end_time = result[2]
        to_append = [result[0], result[1]]
        sorted_scores.append(to_append)
        sorted_scores = sorted(sorted_scores, key=lambda x: x[0])
        if adaptive_features:
            ucb = upper_continuous_bound([x[0] for x in sorted_scores])
            if previous_ucb != ucb:
                current_average = numpy.mean(
                    [x[1] for x in sorted_scores[:ucb][-window_size:]])
                rolling_average_score_values.append(current_average)
                if previous_average > current_average:
                    tolerance -= 1
                else:
                    tolerance = tolerance_steps
                if not tolerance:
                    if verbose > 0:
                        print("Early stopping at number of features:",
                              len(sorted_scores)*step_size, ".")
                    pool.terminate()
                previous_average = current_average
                previous_ucb = ucb
        if verbose > 0:
            print(
                f"Point {len(sorted_scores)} of {len(bounds)}. Took {end_time} seconds at process {result[3]}.", end="\n", flush=True)

    def errors(result):
        print(result)

    pool = mp.Pool(n_jobs,
                   initializer=_init_compute,
                   initargs=(clf, splits,  details_files,
                             details_files_path, score_function),
                   maxtasksperchild=None)
    for x in params:
        pool.apply_async(_compute_score, args=x,
                         callback=add_to_list_and_check, error_callback=errors)
    pool.close()
    pool.join()
    score_values = [x[1] for x in sorted_scores]
    curve = list(zip(bounds, score_values))
    return curve
