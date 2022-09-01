from sklearn.model_selection._search import BaseSearchCV,ParameterGrid
from collections.abc import Mapping
from sklearn.utils import check_random_state
import numpy as np
import time
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._validation import _insert_error_scores
from sklearn.model_selection._validation import _warn_about_fit_failures
from collections import defaultdict
from sklearn.model_selection._split import check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics import check_scoring
from sklearn.utils.validation import indexable, _check_fit_params
from joblib import Parallel
from itertools import product
from sklearn.base import  is_classifier, clone
from sklearn.utils.fixes import delayed
from sklearn.model_selection._search import ParameterSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


def PI(x,gp,y_max=1,xi=0.01,kappa=None):
    mean,std=gp.predict(x,return_std=True)
    z=(mean-y_max-xi)/std
    return norm.cdf(z)

def EI(x,gp,y_max=1,xi=0.01,kappa=None):
    mean,std=gp.predict(x,return_std=True)
    a=(mean-y_max-xi)
    z=a/std
    return a*norm.cdf(z)+std*norm.pdf(z)

def UCB(x,gp,y_max=None,xi=None,kappa=0.1):
    mean,std=gp.predict(x,return_std=True)
    return mean+kappa*std

class BayesSearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter=10,
        random_state=None,
        warm_up=2,
        lam=3,
        y_max=1, xi=0.01, kappa=None,
        acquisition_func='PI',
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
    ):
        # >> - estimator: This is assumed to implement the scikit-learn estimator interface.
        # >> - param_distributions: Dictionary with parameters names ('str') as keys and distributions or lists of parameters to try. Distributions must provide a 'rvs' method for sampling (such as those from scipy.stats.distributions). If a list is given, it is sampled uniformly. If a list of dicts is given, first a dict is sampled uniformly, and then a parameter is sampled using that dict as above.
        # >> - n_iter: Number of iterations.
        # >> - random_state: The state of random seed.
        # >> - warm_up: The number of times to randomly sample parameters in the initial state.
        # >> - lam: The number of parameter groups that need to be sampled for evaluation per iteration.
        # >> - y_max: Valid when acquisition_func is 'PI' and 'EI', it represents the maximum value of the score.
        # >> - xi: Valid when acquisition_func is 'PI' and 'EI', the parameter used to trade off exploration and explotitation.
        # >> - kappa: Valid when acquisition_func is 'UCB', it is used to trade off mean and variance.
        # >> - acquisition_func: The function to estimate the score of the parameter group, optional 'PI', 'EI' and 'UCB' or a function that can be called.
        # >> - scoring: Strategy to evaluate the performance of the cross-validated model on the test set.
        # >> - n_jobs: Number of jobs to run in parallel.
        # >> - refit: Refit an estimator using the best found parameters on the whole dataset.
        # >> - cv: Determines the cross-validation splitting strategy. Int, cross-validation generator or an iterable.
        # >> - verbose: Controls the verbosity: the higher, the more messages.
        # >> - pre_dispatch: Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        # >> - error_score: Value to assign to the score if an error occurs in estimator fitting.
        # >> - return_train_score: If 'False', the 'cv_results_' attribute will not include training scores.
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.lam=lam
        self.warm_up=warm_up
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.acquisition_func=acquisition_func
        self.y_max=y_max
        self.xi=xi
        self.kappa=kappa

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.warm_up, random_state=self.random_state
            )
        )

    def fit(self, X, y=None, *, groups=None, **fit_params):
        self.GP = GaussianProcessRegressor()
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        if self.acquisition_func is 'PI':
            self.acquisition_func=PI
        elif self.acquisition_func is 'EI':
            self.acquisition_func=EI
        elif self.acquisition_func is 'UCB':
            self.acquisition_func=UCB
        elif callable(self.acquisition_func):
            self.acquisition_func=self.acquisition_func
        else:
            self.acquisition_func = PI
        results = {}
        all_candidate_params = []
        all_out = []
        all_more_results = defaultdict(list)

        def evaluate_candidates(candidate_params, cv=None, more_results=None):
            cv = cv or cv_orig
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                print(
                    "Fitting {0} folds for each of {1} candidates,"
                    " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            out = parallel(
                delayed(_fit_and_score)(
                    clone(base_estimator),
                    X,
                    y,
                    train=train,
                    test=test,
                    parameters=parameters,
                    split_progress=(split_idx, n_splits),
                    candidate_progress=(cand_idx, n_candidates),
                    **fit_and_score_kwargs,
                )
                for (cand_idx, parameters), (split_idx, (train, test)) in product(
                    enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                )
            )

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )
            elif len(out) != n_candidates * n_splits:
                raise ValueError(
                    "cv.split and cv.get_n_splits returned "
                    "inconsistent results. Expected {} "
                    "splits, got {}".format(n_splits, len(out) // n_candidates)
                )

            _warn_about_fit_failures(out, self.error_score)

            # For callable self.scoring, the return type is only know after
            # calling. If the return type is a dictionary, the error scores
            # can now be inserted with the correct key. The type checking
            # of out will be done in `_insert_error_scores`.
            if callable(self.scoring):
                _insert_error_scores(out, self.error_score)

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            if more_results is not None:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)

            nonlocal results
            results = self._format_results(
                all_candidate_params, n_splits, all_out, all_more_results
            )
            return results

        with parallel:
            self._run_search(evaluate_candidates)

        first_test_score = all_out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)

        if callable(self.scoring) and self.multimetric_:
            self._check_refit_for_multimetric(first_test_score)
            refit_metric = self.refit
        params = results['params']
        score = results[f"mean_test_{refit_metric}"]
        _X=[]
        for _ in range(self.warm_up):
            _X.append(list(params[_].values()))
        _X=np.array(_X)
        _y=score
        for _ in range(self.n_iter):
            self.GP.fit(_X, _y)
            condidate_params=list(ParameterSampler(
                self.param_distributions, self.lam, random_state=self.random_state
            ))
            _test_X=[]
            for _ in range(self.lam):
                _test_X.append(list(condidate_params[_].values()))
            _test_X=np.array(_test_X)
            _pred_y=self.acquisition_func(_test_X,gp=self.GP,y_max=self.y_max,xi=self.xi,kappa=self.kappa)
            idx=_pred_y.argmax()
            _params=dict()
            key_idx=0
            for key in list(self.param_distributions.keys()):
                _params[key]=_test_X[idx][key_idx]
            evaluate_candidates([_params])
            _X = np.concatenate((_X, np.expand_dims(_test_X[idx],axis=0)), axis=0)
            _y= np.concatenate((_y,np.array([_pred_y[idx]])),axis=0)

        first_test_score = all_out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)

        if callable(self.scoring) and self.multimetric_:
            self._check_refit_for_multimetric(first_test_score)
            refit_metric = self.refit

        self.best_index_ = self._select_best_index(
            self.refit, refit_metric, results
        )
        # With a non-custom callable, we can select the best score
        # based on the best index
        self.best_score_ = results[f"mean_test_{refit_metric}"][
            self.best_index_
        ]
        self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
