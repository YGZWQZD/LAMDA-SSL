import copy
import random

from sklearn.model_selection._search import BaseSearchCV,ParameterGrid
from collections.abc import Mapping, Iterable
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

class Evolve:
    # 1+\lambda
    def __init__(self, param_distributions, *, random_state=None,lam=5,ancestors=None):
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError(
                "Parameter distribution is not a dict or a list ({!r})".format(
                    param_distributions
                )
            )

        if isinstance(param_distributions, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_distributions = [param_distributions]

        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError(
                    "Parameter distribution is not a dict ({!r})".format(dist)
                )
            for key in dist:
                if not isinstance(dist[key], Iterable) and not hasattr(
                    dist[key], "rvs"
                ):
                    raise TypeError(
                        "Parameter value is not iterable "
                        "or distribution (key={!r}, value={!r})".format(key, dist[key])
                    )
        self.random_state = random_state
        self.param_distributions = param_distributions
        self.lam=lam
        self.ancestors=ancestors

    def _is_all_lists(self):
        return all(
            all(not hasattr(v, "rvs") for v in dist.values())
            for dist in self.param_distributions
        )

    def __iter__(self):
        rng = check_random_state(self.random_state)

        # if all distributions are given as lists, we want to sample without
        # replacement
        for _ in range(self.lam):
            dist = rng.choice(self.param_distributions)
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(dist.items())
            params = copy.copy(self.ancestors)
            Mutation_key,Mutation_val=random.choice(items)
            if hasattr(Mutation_val, "rvs"):
                params[Mutation_key] = Mutation_val.rvs(random_state=rng)
            else:
                params[Mutation_key] = Mutation_val[rng.randint(len(Mutation_val))]
            yield params

    def __len__(self):
        """Number of points that will be sampled."""
        if self._is_all_lists():
            grid_size = len(ParameterGrid(self.param_distributions))
            return min(self.lam, grid_size)
        else:
            return self.lam

class EvolutionaryStrategySearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter=10,
        random_state=None,
        lam=3,
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
        # >> - lam: The value of \lambda in the 1+\lambda evolution strategy, that is, the number of children in each iteration.
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
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
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
        results = {}
        rng = check_random_state(self.random_state)
        if isinstance(self.param_distributions, Mapping):
            param_distributions = [self.param_distributions]
        else:
            param_distributions=self.param_distributions
        dist = rng.choice(param_distributions)
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(dist.items())
        self.best_params_ = dict()
        for k, v in items:
            if hasattr(v, "rvs"):
                self.best_params_[k] = v.rvs(random_state=rng)
            else:
                self.best_params_[k] = v[rng.randint(len(v))]
        all_candidate_params = []
        all_out = []
        all_more_results = defaultdict(list)
        for _ in range(self.n_iter):
            with parallel:
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
                    print(all_candidate_params)
                    print(all_out)
                    print(results)
                    return results

                self._run_search(evaluate_candidates)

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

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(
            Evolve(
                self.param_distributions, random_state=self.random_state,lam=self.lam,ancestors=self.best_params_
            )
        )
