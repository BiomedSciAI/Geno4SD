from datetime import datetime
import random, string
import os
from . import feature_ranking
from . import lreb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy,time
from .helper_funcs import save_to_pickle, save_to_csv

def compute_curves(X, y, 
                    iterations, 
                    curve_steps,
                    validation_size,
                    clf_ranking, 
                    clf_scoring,
                    n_features=None,
                    ranking_test_size = .2,
                    ranking_number_of_folds = 5,
                    return_ranking_coefs = False,
                    scoring_test_size = .2,
                    scoring_number_of_folds = 5,
                    score_function = feature_ranking.youden_index,
                    scoring_n_jobs = 1,
                    ranking_n_jobs = 1,
                    scoring_adaptive_features = False,
                    scoring_tolerance_steps = 10,
                    scoring_window_size = 10,
                    verbose = 0,
                    details_files = True,
                    details_files_parent_path = "./",
                    seed = None):
    """
    Splits data into *working* and *validation*, then computes as many score curves as specified in the `iterations` parameter using
    only *validation* data. Each iteration has its own set of random splits for ranking and random splits for scoring. Returns
    a list with ranked features and scores for each iteration, along with indices for samples in *working* and *validation* data. If 
    `return_ranking_coefs` is `True`, it also returns the average rankings for each feature.

    Parameters:
    -----------

    X:
        2D array with rows observations and columns variables.
    
    y:
        1D array with integer labels (0 or 1) for each row of X.
    
    iterations:
        Number of repetitions for the inner RubricOE loop.

    validation_size:
        Proportion of observations that will be held out during the entire procedure for validation.

    clf_ranking:
        Function that will be used for ranking features in the inner loop. An example is lreb.LinRidgeRegSVD().
        Generally a function with a `.fit(data,labels)` method with a `coef_` attribute will work.
    
    clf_scoring:
        Function that will be used for training and testing computations in the inner loop on subsets of features. 
        An example is sklearn's SVC(). Generally a function with `.fit(train_data,labels)` and `.predict(test_data)` 
        methods will work.
    
    n_features:
        If specified, upper bound of ranked features used to generate score curves. Note: All features are still used for feature ranking,
        only score curves are affected.
        
    ranking_test_size:
        Proportion of observations in working data that will be randomly discarded in each fold when training the ranking procedure.
    
    ranking_number_of_folds:
        Number of repetitions of training the ranking procedure.

    return_ranking_coefs:
        Flag to specify if should return actual rankings of features.

    scoring_test_size:
        Proportion of observations in working data that will be used in each fold for testing performance of `clf_scoring`.
    
    scoring_number_of_folds:
        Number of repetitions of training the ranking procedure.
    
    score_function:
        Function that will be used to generate values in the score curve. By default it uses the Youden Index, but more generally
        a function of the form score(true,pred) will work
    
    score_n_jobs:
        Number of jobs that will be spawned to compute curve points in parallel.

    verbose:
        Verbosity level. 0 disables all messages. Greater than 0 prints messages.

    Returns:
    --------
        ranked_features_list, curve_list, idx_working, idx_validation
        A tuple with, in order: 
        * a list with lists of ranked feature indices from higher to lower importance (one per iteration),
        * a list with lists of scores obtained by the classifier when increasing the number of features selected (one per iteration),
        * an array with the indices of the observations corresponding to the working set
        * an array with the indices of the observations corresponding to the validation set

        If the flag `return_ranking_coefs` is set to `True`, the return tuple is

        ranked_features_list, ranking_coefs, curve_list, idx_working, idx_validation
        With, in order: 
        * a list with lists of ranked feature indices from higher to lower importance (one per iteration),
        * a list with lists of rankings of the features from higher to lower importance (one per iteration),
        * a list with lists of scores obtained by the classifier when increasing the number of features selected (one per iteration),
        * an array with the indices of the observations corresponding to the working set
        * an array with the indices of the observations corresponding to the validation set
    """
    unique_id = "RUBRICOE_" + datetime.now().isoformat().replace("-","_").replace(":","_").replace(".","_").replace("T","_")  + \
             ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(8)) 
    if details_files:
        details_files_path = os.path.join(details_files_parent_path, unique_id)
        os.mkdir(details_files_path)
    idx_working, idx_validation = train_test_split(range(X.shape[0]),test_size=validation_size,stratify=y,random_state=seed)
    if details_files:
        save_to_csv(idx_working, details_files_path, "indices_working.csv")
        save_to_csv(idx_validation, details_files_path, "indices_validation.csv")
    X_working = X[idx_working]
    y_working = y[idx_working]
    if return_ranking_coefs:
        ranking_coefs_list = []
    ranked_features_list = []
    curve_list = []
    if seed is None:
        random_generator = numpy.random.RandomState()
    else:
        random_generator = numpy.random.RandomState(seed)
    iteration_scoring_seeds = random_generator.randint(2**32,size=iterations)
    iteration_ranking_seeds = random_generator.randint(2**32,size=iterations)
    for i in range(iterations):
        if details_files:
            details_files_iteration_path = os.path.join(details_files_path,"ITERATION{}".format(i))
            os.mkdir(details_files_iteration_path)
        start_time = time.time()
        if verbose>0:
            print("Iteration {} of {}".format(i+1,iterations))
        if n_features is None:
            n_features = X.shape[1]
        step_size = 1
        while (n_features//step_size) > curve_steps:
            step_size += 1
        ranked_features = feature_ranking.rank_features(clf_ranking, 
                                                        X_working, 
                                                        y_working, 
                                                        test_size=ranking_test_size, 
                                                        number_of_folds=ranking_number_of_folds,
                                                        verbose=verbose,
                                                        return_ranking_coefs=return_ranking_coefs,
                                                        n_jobs = ranking_n_jobs,
                                                        details_files=details_files,
                                                        details_files_path=details_files_iteration_path,
                                                        seed=iteration_scoring_seeds[i])
        if return_ranking_coefs:
            ranked_features, ranking_coefs = ranked_features
        if isinstance(clf_ranking,lreb.LinRidgeRegSVD):  # WORKAROUND FOR CURRENT IMPLEMENTATION OF LREB
            if verbose>0:
                print("Applying workaround for Linear Regression with Error Bars.")
            if return_ranking_coefs:
                temp = [(x,y) for (x,y) in zip(ranked_features,ranking_coefs) if x<X_working.shape[1]]
                ranked_features, ranking_coefs = tuple(zip(*temp))
            else:
                ranked_features = [x for x in ranked_features if x<X_working.shape[1]]
        ranked_features_list.append(ranked_features)
        if return_ranking_coefs:
            ranking_coefs_list.append(ranking_coefs)
        if verbose>0:
            print(f"Feature ranking completed. {time.time()-start_time} seconds.")
        start_time = time.time()
        curve = feature_ranking.score_curve(clf_scoring,
                                            X_working[:,ranked_features[:n_features]],
                                            y_working,
                                            test_size=scoring_test_size,
                                            number_of_folds=scoring_number_of_folds,
                                            verbose=verbose,
                                            n_jobs=scoring_n_jobs,
                                            step_size=step_size,
                                            n_features=n_features,
                                            score_function=score_function,
                                            adaptive_features=scoring_adaptive_features,
                                            tolerance_steps = scoring_tolerance_steps,
                                            window_size = scoring_window_size,
                                            details_files=details_files,
                                            details_files_path=details_files_iteration_path,
                                            seed=iteration_ranking_seeds[i])
        if verbose>0:
            print(f"Curve scoring completed. {time.time()-start_time} seconds.")
        if details_files:
            save_to_csv(ranked_features,details_files_iteration_path,"ranked_features.csv")    
            save_to_csv(curve,details_files_iteration_path,"curve.csv")
        curve_list.append(curve)
    if return_ranking_coefs:
        return ranked_features_list, ranking_coefs_list, curve_list, idx_working, idx_validation
    return ranked_features_list, curve_list, idx_working, idx_validation


def compute_feature_counts(ranked_features_list, curve_list, step_size):
    """
    Computes the proportion of iterations where a feature was selected as a top feature according to its corresponding curve.
    Expects parameters similar to the output of `compute_curves`.
    """
    if len(ranked_features_list)!=len(curve_list):
        raise ValueError("Number of iterations in rankings and curves don't match.")
    scale = len(ranked_features_list)
    top_features_list = []
    for i,c in enumerate(curve_list):
        threshold = (numpy.argmax([x[1] for x in c])+1)*step_size
        top_features_list.append(ranked_features_list[i][:threshold+1])
    counts = numpy.zeros(len(ranked_features_list[0]))
    for y in range(len(counts)):
        for l in top_features_list:
            if y in l:
                counts[y] += 1
    feature_counts = counts/scale
    return feature_counts

def compute_top_features(feature_counts, threshold=1.):
    """ 
    Filters out top features according to `threshold`. A `threshold` value of 1 indicates that the feature was selected as 
    a top feature in all iterations. Expects `feature_counts` to be similar to the output of `compute_feature_counts`.
    """
    return numpy.where(feature_counts>=threshold)[0]

def plot_scores(eval_scores, extra_string=""):
    """
    Basic function to plot scores of a repeated experiment with confidence bars and mean value.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(5,5),dpi=100)
    cil,cir = numpy.percentile(eval_scores,[2.5,97.5])
    sns.histplot(eval_scores, label="Density Estimate",kde=True)
    xc = 0
    xg = numpy.mean(eval_scores)
    plt.suptitle("Score frequency plot\n{}".format(extra_string))
    plt.axvline(x=xc, label='Score = {}'.format(xc),linestyle="dashed", c="red")

    plt.axvline(x=cil, label='CI Left = {:0.3f}'.format(cil),linestyle="dotted", c="blue")
    plt.axvline(x=cir, label='CI Right = {:0.3f}'.format(cir),linestyle="dotted", c="blue")

    plt.axvline(x=xg, label='Mean = {:0.3f}'.format(xg),linestyle="dashed", c="green")
    plt.xlabel("Score")
    plt.legend()
    plt.show()
    plt.close()


def compute(df, 
                        iterations, 
                        curve_steps, 
                        validation_size,
                        n_features, 
                        ranking_test_size,
                        scoring_test_size,
                        ranking_number_of_folds, 
                        scoring_number_of_folds,
                        C,
                        scoring_n_jobs,
                        threshold,
                        output_filename,
                        details=True,
                        details_files_parent_path = "./",
                        verbose=0):
    """
        Function to run the full RubricOE analysis
    
        Parameters:
        -----------
       
        df: 
            Dataframe of input matrix with column 'phenotype' with label
        iterations: int
            Number of iterations of main RubricOE loop
        curve_steps: int
            Scoring curve resolution
        validation_size: float
            Proportion of data to use for validation
        n_features: int
            Number of features to use for score curve. Set to \"all\" for all features.
        ranking_test_size: float
            Proportion of non-validation data to use per split on ranking step.
        scoring_test_size: float
            Proportion of non-validation data to use per split on scoring step.
        ranking_number_of_folds: 
            Number of splits used in ranking step.
        scoring_number_of_folds: int
            Number of splits used in scoring step
        C: float
            Regularization coefficient in Ridge Regression with Error Bars
        scoring_n_jobs: int
            Number of parallel processes to run in scoring step
        threshold: float
            Proportion of iterations where a SNP should be present to be interpreted as a top SNP.
        output_filename: str,
            Base name of output files.
        details: Boolean, default=True,
            Whether to output files with detailed intermediate results.
        details_files_path: str, optional
            Path where output files will be written to. Not optional if `details_files` is `True`.
        verbose: int, default=0, 
            Set to 0 to omit progress notifications.
        
        Returns:
        --------
        A the subset input dataframe, relative to the top features
    """

    X = df[[x for x in df.columns if x!="phenotype"]].values
    y = df["phenotype"].values
    if n_features == "all":
        n_features = X.shape[1]
    else:
        n_features = int(n_features)
    step_size = n_features//curve_steps
    ranked_features, curves, idx_working, idx_validation = ompute_curves(X, y,
                                                                iterations = iterations, 
                                                                curve_steps = curve_steps,
                                                                validation_size = validation_size,
                                                                n_features = n_features,
                                                                ranking_test_size = ranking_test_size,
                                                                scoring_test_size = scoring_test_size,
                                                                ranking_number_of_folds = ranking_number_of_folds,
                                                                scoring_number_of_folds = scoring_number_of_folds,
                                                                clf_ranking = rubricoe.lreb.LinRidgeRegSVD(C = C),
                                                                clf_scoring = SVC(kernel = "linear"),
                                                                scoring_n_jobs = scoring_n_jobs,
                                                                verbose = verbose,
                                                                details_files=details,
                                                                details_files_parent_path = details_files_parent_path)
    feature_counts = compute_feature_counts(ranked_features,curves,step_size)
    top_features = compute_top_features(feature_counts, threshold)
    if verbose>0:
        print("Finished computing. Saving files...")
    if details:
        output_df = pd.DataFrame(df.columns[top_features])
        output_df.to_csv(output_filename,header=False,index=False)
    if verbose>0:
        print("Finished.")
    
    return pd.DataFrame(df.columns[top_features])