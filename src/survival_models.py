# importing metrics
from optuna import create_study
from optuna.samplers import TPESampler
from scipy.stats import randint, uniform
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from xgbse import XGBSEStackedWeibull
from xgbse.converters import convert_y
from xgbse.metrics import concordance_index

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
        self.estimator = None

    def fit(self, X_train, y_train):
        print("Model training...")
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        print("Prediction with the best model...")
        predictions = self.estimator.predict(X_test)
        return predictions

    def model_pipeline(self, X_train, y_train, X_test):
        self = self.cross_validation(X_train, y_train)
        self = self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        return predictions, self

    def cross_validation(self, X_train, y_train):
        return self

    def evaluate(self, X_train, X_test, y_train, y_test):
        n_splits = 4
        n_repeats = 1
        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=0
        )
        X_values = X_train
        y_values = y_train
        for train_index, test_index in rkf.split(X_values):
            X_A, X_B = X_values.iloc[train_index,
                                     :], X_values.iloc[test_index, :]
            y_A, y_B = y_values[train_index], y_values[test_index]

            if not check_is_fitted(self.estimator):
                self.estimator[:-1].fit(X_A, y_A)

            X_B_transformed = self.estimator.named_steps['preprocessor'].transform(
                X_B)

            if hasattr(self.estimator, 'EARLY_STOPPING_ROUNDS'):
                self.estimator.fit(
                    X_A,
                    y_A,
                    validation_data=(X_B_transformed, y_B),
                    early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
                )
            else:
                self.estimator.fit(X_A, y_A)
            score += self.estimator.score(X_B, y_B)
        score = score/(n_splits*n_repeats)
        return score


class sksurv_gbt(candidate_model):
    def __init__(self):
        super().__init__()
        self.monitor = EarlyStoppingMonitor(25, 50)
        self.model = create_pipeline(GradientBoostingSurvivalAnalysis())
        self.estimator.fit = lambda X_train, y_train: self.model.fit(
            X_train, y_train, model__monitor=self.monitor
        )

        self.distributions = dict(
            model__learning_rate=uniform(loc=0, scale=1),
            model__max_depth=randint(1, 8),
            model__loss=["coxph"],
            model__n_estimators=uniform(loc=30, scale=250),
            model__min_samples_split=randint(2, 10),
            model__min_samples_leaf=randint(1, 10),
            model__subsample=uniform(loc=0.5, scale=0.5),
            model__max_leaf_nodes=randint(2, 30),
            model__dropout_rate=uniform(loc=0, scale=1),
        )

    def cross_validation(self, X_train, y_train):
        randsearchcv = RandomizedSearchCV(
            self.estimator,
            self.distributions,
            random_state=0,
            n_iter=200,
            n_jobs=-1,
            verbose=2,
        )
        search = randsearchcv.fit(X_train, y_train)
        self.estimator = search.best_estimator_
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

        self.estimator = create_pipeline(
            self.model)

        self.distributions = dict(
            model__aft_loss_distribution_scale=uniform(loc=0, scale=1),
            model__colsample_bynode=uniform(loc=0.5, scale=0.5),
            model__learning_rate=uniform(0, 1),
            model__max_depth=randint(2, 10),
            model__min_child_weight=randint(2, 10),
            model__subsample=uniform(loc=0.5, scale=0.5),
            model__tree_method=["hist"],
        )

    def cross_validation(self, X_train, y_train):
        randsearchcv = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.distributions,
            scoring=make_scorer(concordance_index),
            random_state=0,
            n_iter=3,  # 300
            n_jobs=-1,
            verbose=2,
            error_score='raise'
        )
        search = randsearchcv.fit(X_train, y_train)
        self.estimator = search.best_estimator_
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
        self.model = XGBSurvival(self.params)
        self.estimator = create_pipeline(self.model)


class xgb_optuna(candidate_model):
    def __init__(self):
        super().__init__()
        # repeated K-folds
        self.N_SPLITS = 3
        self.N_REPEATS = 1

        # Optuna
        self.N_TRIALS = 100
        self.RS = 124  # random state
        # XGBoost
        self.EARLY_STOPPING_ROUNDS = 50
        self.MULTIVARIATE = True

        self.N_JOBS = -1  # number of parallel threads

        self.sampler = TPESampler(seed=self.RS, multivariate=self.MULTIVARIATE)
        self.base_params = {
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

        self.model = XGBSurvival(self.base_params, num_boost_round=10000)
        self.estimator = create_pipeline(self.model)

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
                n_jobs=-1,
                early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            ),
            n_trials=self.N_TRIALS,
            n_jobs=-1,
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
        n_jobs=-1,
        early_stopping_rounds=50,
    ):
        # XGBoost parameters

        xgb_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": trial.suggest_loguniform('aft_loss_distribution_scale', 0.1, 10.0),
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "booster": "dart",
            "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=False),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        }

        model = XGBSurvival(xgb_params, num_boost_round=10000)
        estimator = create_pipeline(model)

        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        X_values = X
        y_values = y
        score = 0
        for train_index, test_index in rkf.split(X_values):
            X_A, X_B = X_values.iloc[train_index,
                                     :], X_values.iloc[test_index, :]
            y_A, y_B = y_values[train_index], y_values[test_index]

            if not check_is_fitted(estimator):
                estimator[:-1].fit(X_A, y_A)

            X_B_transformed = estimator.named_steps['preprocessor'].transform(
                X_B)

            estimator.fit(
                X_A,
                y_A,
                model__validation_data=(X_B_transformed, y_B),
                model__verbose_eval=0,
                model__early_stopping_rounds=early_stopping_rounds,
            )

            score += estimator.score(X_B, y_B)
        score /= n_repeats
        return score

    def fit(self, X_train, y_train):
        estimator = XGBSurvival(self.optimal_hp, num_boost_round=10000)

        # WARNING : use a pipeline
        estimator.fit(
            X_train,
            y_train,
            verbose_eval=0,
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
        )
