# importing metrics
from optuna import create_study
from optuna.samplers import TPESampler
from scipy.stats import randint, uniform
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from xgbse import XGBSEStackedWeibull
from xgbse.converters import convert_y
from xgbse.metrics import approx_brier_score, concordance_index, dist_calibration_score

from model_evaluation import evaluate_model
from pipeline import EarlyStoppingMonitor, create_pipeline
from xgboost_wrapper import XGBSurvival


def bind(instance, method):
    def binding_scope_fn(*args, **kwargs):
        return method(instance, *args, **kwargs)

    return binding_scope_fn


def score(self, X, y):
    survival_prob = self.predict(X)
    risk_score = -survival_prob
    event, time = convert_y(y)
    return concordance_index_censored(event, time, risk_score)[0]


class candidate_model:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):
        print("Model training...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        print("Prediction with the best model...")
        predictions = self.model.predict(X_test)
        return predictions

    def model_pipeline(self, X_train, y_train, X_test):
        self = self.cross_validation(X_train, y_train)
        self = self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        return predictions

    def cross_validation(self, X_train, y_train):
        return self

    def evaluate(self, X_train, X_test, y_train, y_test):
        return evaluate_model(self.model, X_train, X_test, y_train, y_test)


class sksurv_gbt(candidate_model):
    def __init__(self):
        super().__init__()
        self.monitor = EarlyStoppingMonitor(25, 50)
        est_early_stopping = GradientBoostingSurvivalAnalysis()
        # WARNING : it should be self.model, andnot slef.pipe
        self.pipe = create_pipeline(est_early_stopping)
        self.pipe.fit = lambda X_train, y_train: self.pipe.fit(
            X_train, y_train, model__monitor=self.monitor
        )

        self.distributions = dict(
            model__learning_rate=uniform(loc=0, scale=1),
            model__max_depth=randint(1, 4),
            model__loss=["coxph"],
            model__n_estimators=uniform(loc=30, scale=150),
            model__min_samples_split=randint(2, 10),
            model__min_samples_leaf=randint(1, 10),
            model__subsample=uniform(loc=0.5, scale=0.5),
            model__max_leaf_nodes=randint(2, 10),
            model__dropout_rate=uniform(loc=0, scale=1),
        )

    def cross_validation(self, X_train, y_train):
        randsearchcv = RandomizedSearchCV(
            self.pipe,  # WARNING : self.modelm
            self.distributions,
            random_state=0,
            n_iter=2,
            n_jobs=-1,
            verbose=2,
        )
        search = randsearchcv.fit(X_train, y_train)
        self.model = search.best_estimator_
        return self


class sklearn_wei(XGBSEStackedWeibull):
    """ Workaround to use crossvalidation from 
    sklearn
    """

    def get_params2(self):
        return(self.get_params()['xgb_params'])

    def set_params(self, **params):
        old_params = self.get_params2()
        old_params.update(params)
        self.xgb_params = old_params
        return(self)


class xgbse_weibull(candidate_model):
    def __init__(self):
        super().__init__()
        xgb_params = {
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1,
            "booster": "dart",
            "colsample_bynode": 0.5,
            "eval_metric": "aft-nloglik",
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 50,
            "objective": "survival:aft",
            "subsample": 0.5,
            "tree_method": "hist",
        }
        # fitting with early stopping

        self.model = sklearn_wei(xgb_params)

        self.model.score = bind(self, score)

        self.pipe = create_pipeline(self.model)  # WARNING : self.modelm

        self.distributions = dict(
            model__aft_loss_distribution_scale=uniform(loc=0, scale=1),
            model__booster=randint(1, 4),
            model__aft_loss_distribution=["coxph"],
            model__colsample_bynode=uniform(loc=30, scale=150),
            model__eval_metric=randint(2, 10),
            model__learning_rate=randint(1, 10),
            model__max_depth=uniform(loc=0.5, scale=0.5),
            model__min_child_weight=randint(2, 10),
            model__objective=uniform(loc=0, scale=1),
            model__subsample=uniform(loc=0.5, scale=1),
            model__tree_method=["hist"],
        )

        self.model_features = {
            "name": "XGBSE Weibull",
            "pipe": self.pipe,
            "distributions": self.distributions,
            "monitor": None,
        }

    def cross_validation(self, X_train, y_train):
        randsearchcv = RandomizedSearchCV(
            estimator=self.pipe,  # WARNING : self.model
            param_distributions=self.distributions,
            scoring=make_scorer(concordance_index),
            random_state=0,
            n_iter=3,  # 300
            n_jobs=-1,
            verbose=2,
        )
        search = randsearchcv.fit(X_train, y_train)
        self.model = search.best_estimator_
        return self


class xgb_aft(candidate_model):
    def __init__(self):
        super().__init__()
        self.params = {
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1,
            "booster": "dart",
            "colsample_bynode": 0.5,
            "eval_metric": "aft-nloglik",
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 50,
            "objective": "survival:aft",
            "subsample": 0.5,
            "tree_method": "hist",
        }
        self.model = create_pipeline(XGBSurvival(self.params))


class xgb_optuna(candidate_model):
    def __init__(self):
        super().__init__()
        # repeated K-folds
        self.N_SPLITS = 10
        self.N_REPEATS = 1

        # Optuna
        self.N_TRIALS = 100
        self.RS = 124  # random state
        # XGBoost
        self.EARLY_STOPPING_ROUNDS = 100
        self.MULTIVARIATE = True

        self.N_JOBS = -1  # number of parallel threads

        self.sampler = TPESampler(seed=self.RS, multivariate=self.MULTIVARIATE)
        self.optimal_hp = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.20,
            "tree_method": "hist",
            "booster": "dart",
            "learning_rate": 0.032833188230587194,
            "max_depth": 10,
            "subsample": 0.61828926669036,
            "alpha": 0.012673256334558281,
            "lambda": 6.468264510932119,
            "verbosity": 0,
            "n_estimators": 10000,
            "seed": self.RS,
        }

        # WARNING : create a pipeline
        self.model = XGBSurvival(self.optimal_hp, num_boost_round=10000)

    def cross_validation(self, X_train, y_train):

        study = create_study(direction="maximize", sampler=self.sampler)
        study.optimize(
            lambda trial: self.objective(
                trial,
                X_train,
                y_train,
                random_state=self.RS,
                n_splits=self.N_SPLITS,
                n_repeats=self.N_REPEATS,
                n_jobs=1,
                early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            ),
            n_trials=self.N_TRIALS,
            n_jobs=1,
        )
        self.optimal_hp = study.best_params

    def objective(
        self,
        trial,
        X,
        y,
        random_state=22,
        n_splits=3,
        n_repeats=2,
        n_jobs=1,
        early_stopping_rounds=50,
    ):
        # XGBoost parameters

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.20,
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 5e-2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "booster": "dart",
            "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        }

        # WARNING : use a pipeline
        model = XGBSurvival(params, num_boost_round=10000)

        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        X_values = X.values
        y_values = y
        score = 0
        for train_index, test_index in rkf.split(X_values):
            X_A, X_B = X_values[train_index, :], X_values[test_index, :]
            y_A, y_B = y_values[train_index], y_values[test_index]
            model.fit(
                X_A,
                y_A,
                validation_data=(X_B, y_B),
                verbose_eval=0,
                early_stopping_rounds=early_stopping_rounds,
            )
            score += model.score(X_B, y_B)
        score /= n_repeats
        return score

    def fit(self, X_train, y_train):
        model = XGBSurvival(self.optimal_hp, num_boost_round=10000)

        # WARNING : use a pipeline
        model.fit(
            X_train.values,
            y_train,
            verbose_eval=0,
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
        )

    def model_evaluation(self, X_train, y_train, X_test, y_test):
        rkf = RepeatedKFold(
            n_splits=self.N_SPLITS, n_repeats=self.N_REPEATS, random_state=self.RS
        )
        X_values = X_train.values
        y_values = y_train
        for train_index, test_index in rkf.split(X_values):
            X_A, X_B = X_values[train_index, :], X_values[test_index, :]
            y_A, y_B = y_values[train_index], y_values[test_index]
            self.model.fit(
                X_A,
                y_A,
                validation_data=(X_B, y_B),
                verbose_eval=0,
                early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            )
            y_pred += self.model.predict(X_test.values)
        y_pred /= self.N_REPEATS * self.N_SPLITS
