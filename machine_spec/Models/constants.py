from typing import List, Union

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING_METRIC = 'roc_auc'
N_JOBS = -1


class LogisticRegressionConstants:
    MAX_ITER = 1000
    RANDOM_STATE = RANDOM_STATE
    C_VALUES: List[float] = [0.001, 0.01, 0.1, 1]
    PENALTY_VALUES: List[str] = ['l1', 'l2']
    SOLVER_VALUES: List[str] = ['liblinear']
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 0


class RandomForestConstants:
    RANDOM_STATE = RANDOM_STATE
    N_ESTIMATORS: List[int] = [100, 301, 10]
    MAX_FEATURES: List[Union[str, None]] = ['sqrt', 'log2', None]
    MAX_DEPTH: List[Union[int, None]] = [None, 10, 20]
    MIN_SAMPLES_SPLIT: List[int] = [2, 5, 10]
    MIN_SAMPLES_LEAF: List[int] = [1, 2, 4]
    BOOTSTRAP: List[bool] = [True, False]
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 0


class NaiveBayesConstants:
    VAR_SMOOTHING: List[float] = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 1


class DecisionTreeConstants:
    RANDOM_STATE = RANDOM_STATE
    MAX_DEPTH: List[int] = [3, 5, 7, 10]
    MIN_SAMPLES_SPLIT: List[int] = [5, 10, 20, 50]
    MIN_SAMPLES_LEAF: List[int] = [2, 5, 10, 20]
    CRITERION: List[str] = ['gini', 'entropy']
    CCP_ALPHA: List[float] = [0.0, 0.0001, 0.0005, 0.001]
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 1


class GradientBoostingConstants:
    N_ESTIMATORS: List[int] = [100, 200]
    LEARNING_RATE: List[float] = [0.01, 0.05, 0.1, 0.15]
    MAX_DEPTH: List[int] = [3, 4, 5, 6]
    MIN_SAMPLES_SPLIT: List[int] = [2, 20]
    MIN_SAMPLES_LEAF: List[int] = [1, 10]
    SUBSAMPLE: List[float] = [0.8, 0.9, 1.0]
    MAX_FEATURES: List[Union[str, None]] = ['sqrt', 'log2', None]
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 0


class KNNConstants:
    N_NEIGHBORS: List[int] = [3, 5, 7, 9]
    WEIGHTS: List[str] = ['uniform', 'distance']
    P_VALUES: List[int] = [1, 2]
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 1


class SVCConstants:
    RANDOM_STATE = RANDOM_STATE
    PROBABILITY = True
    C_VALUES: List[float] = [0.1, 1, 10]
    GAMMA_VALUES: List[str] = ['scale', 'auto']
    CV_FOLDS = CV_FOLDS
    SCORING_METRIC = SCORING_METRIC
    N_JOBS = N_JOBS
    VERBOSE = 1

