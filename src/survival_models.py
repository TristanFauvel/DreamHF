# importing metrics
from optuna import create_study
from optuna.samplers import TPESampler
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.utils import estimator_html_repr
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from xgbse import XGBSEStackedWeibull
from xgbse.converters import convert_y
from xgbse.metrics import concordance_index

from pipeline import EarlyStoppingMonitor
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
    def __init__(self, with_pca, n_components):
        self.estimator = None
        self.with_pca = with_pca
        self.n_components = n_components
        self.model = None
        
    def fit(self, X_train, y_train):
        print("Model training...")
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        print("Prediction with the best model...")
        predictions = self.estimator.predict(X_test)
        return predictions

    def model_pipeline(self, X_train, y_train, n_iter):
        self = self.cross_validation(X_train, y_train, n_iter)
        self = self.fit(X_train, y_train)
        return self
        
    def risk_score(self, X_test):
        predictions = self.predict(X_test)
        
        # Returns the (normalized) risk score  
        method = getattr(self, "predict_proba", None)
        if callable(method):
            survival_prob = self.predict_proba(X_test)
            risk_score = 1-survival_prob
        else:
            #If loss=’coxph’, predictions can be interpreted as log hazard ratio corresponding to the linear predictor of a Cox proportional hazards model. If loss=’squared’ or loss=’ipcwls’, predictions are the time to event.
            predictions = self.predict(X_test)
            scaler = MinMaxScaler()
            risk_score = scaler.fit_transform(-predictions.reshape(-1, 1))
            #The range of this number has to be between 0 and 1, with larger numbers being associated with higher probability of having HF. The values, -Inf, Inf and NA, are not allowed. 
        return risk_score.to_numpy().flatten()

    def cross_validation(self, X_train, y_train, n_iter):
        return self

    def evaluate(self, X_train, X_test, y_train, y_test):
        self.harrell_C_training = self.estimator.score(X_train, y_train)
        self.harrell_C_test = self.estimator.score(X_test, y_test)
        return self

    def create_pipeline(self):
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        if self.with_pca:
            pca = PCA(self.n_components)
            pca_transformer = ColumnTransformer(
                transformers=[("pca", pca, selector(pattern="k__"))], remainder='passthrough')

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, selector(
                    dtype_exclude=["bool", "category", "Int64"])),
                ("cat", categorical_transformer, selector(
                    dtype_include=["bool", "category", "Int64"])),
            ]
        )

        if self.with_pca:
            regressor = Pipeline(
                steps=[("preprocessor", preprocessor), ("pca", pca_transformer), ("model", self.model)])
        else:
            regressor = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", self.model)])

        with open("regressor.html", "w") as f:
            f.write(estimator_html_repr(regressor))
        return regressor

class sksurv_gbt(candidate_model):
    def __init__(self, with_pca, n_components):
        super().__init__(with_pca, n_components)
        self.monitor = EarlyStoppingMonitor(25, 50)
        
        self.model = GradientBoostingSurvivalAnalysis()
        
        self.estimator = self.create_pipeline()
                
        self.estimator.fit = lambda X_train, y_train: self.estimator.fit(
            X_train, y_train, model__monitor=self.monitor
        )

        self.distributions = dict(
            model__learning_rate=uniform(loc=0, scale=1),
            model__max_depth=randint(1, 8),
            model__loss=["coxph"],
            model__n_estimators=randint(100, 350),
            model__min_samples_split=randint(2, 10),
            model__min_samples_leaf=randint(1, 10),
            model__subsample=uniform(loc=0.5, scale=0.5),
            model__max_leaf_nodes=randint(2, 30),
            model__dropout_rate=uniform(loc=0, scale=1),
        )

    def cross_validation(self, X_train, y_train, n_iter):
        randsearchcv = RandomizedSearchCV(
            self.estimator,
            self.distributions,
            random_state=0,
            n_iter=n_iter,
            n_jobs=-1,
            verbose=2,
            #error_score='raise',
        )
        search = randsearchcv.fit(X_train, y_train)
        self.estimator = search.best_estimator_
        return self


class CoxPH(candidate_model):
    def __init__(self, with_pca, n_components):
        super().__init__(with_pca, n_components)
        self.monitor = EarlyStoppingMonitor(25, 50)

        self.model = CoxPHSurvivalAnalysis(
            ties='breslow', tol=1e-09, verbose=0)

        self.estimator = self.create_pipeline()

        self.estimator.fit = lambda X_train, y_train: self.estimator.fit(
            X_train, y_train, model__monitor=self.monitor
        )

        self.distributions = dict(
            model__alpha=uniform(loc=0, scale=1),
            model__n_iter=randint(80, 200)
        )

    def cross_validation(self, X_train, y_train, n_iter):
        randsearchcv = RandomizedSearchCV(
            self.estimator,
            self.distributions,
            random_state=0,
            n_iter=n_iter,
            n_jobs=-1,
            verbose=2,
            #error_score='raise',
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
    def __init__(self, with_pca = False, n_components = None):
        super().__init__(with_pca, n_components)
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

        self.with_pca = with_pca
        self.n_components = n_components
        self.estimator = self.create_pipeline()

        self.distributions = dict(
            model__aft_loss_distribution_scale=uniform(loc=0, scale=1),
            model__colsample_bynode=uniform(loc=0.5, scale=0.5),
            model__learning_rate=uniform(0, 1),
            model__max_depth=randint(2, 10),
            model__min_child_weight=randint(2, 10),
            model__subsample=uniform(loc=0.5, scale=0.5),
            model__tree_method=["hist"],
        )

    def cross_validation(self, X_train, y_train, n_iter):
        randsearchcv = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.distributions,
            scoring=make_scorer(concordance_index),
            random_state=0,
            n_iter=n_iter,  # 300
            n_jobs=-1,
            verbose=2,
        )
        search = randsearchcv.fit(X_train, y_train)
        self.estimator = search.best_estimator_
        return self


class xgb_aft(candidate_model):
    def __init__(self, with_pca = False, n_components = None):
        super().__init__(with_pca, n_components)
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
        self.estimator = self.create_pipeline()


class xgb_optuna(candidate_model):
    def __init__(self, with_pca = False, n_components = None):
        super().__init__(with_pca, n_components)
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
        self.estimator = self.create_pipeline()

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

        self.model = XGBSurvival(xgb_params, num_boost_round=10000)
        self.estimator = self.create_pipeline()

        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

        score = 0
        for train_index, test_index in rkf.split(X):
            X_A, X_B = X.iloc[train_index,
                              :], X.iloc[test_index, :]
            y_A, y_B = y[train_index], y[test_index]

            if not check_is_fitted(self.estimator):
                self.estimator[:-1].fit(X_A, y_A)

            X_B_transformed = self.estimator.named_steps['preprocessor'].transform(
                X_B)

            self.estimator.fit(
                X_A,
                y_A,
                model__validation_data=(X_B_transformed, y_B),
                model__verbose_eval=0,
                model__early_stopping_rounds=early_stopping_rounds,
            )

            score += self.estimator.score(X_B, y_B)
        score /= n_repeats
        return score

    def fit(self, X_train, y_train):
        self.estimator = XGBSurvival(self.optimal_hp, num_boost_round=10000)

        n_splits = 1
        n_repeats = 1
        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=0
        )
        for train_index, test_index in rkf.split(X_train):
            X_A, X_B = X_train.iloc[train_index,
                                    :], X_train.iloc[test_index, :]
            y_A, y_B = y_train[train_index], y_train[test_index]

            if not check_is_fitted(self.estimator):
                self.estimator[:-1].fit(X_A, y_A)

            X_B_transformed = self.estimator.named_steps['preprocessor'].transform(
                X_B)

            self.estimator.fit(
                X_A,
                y_A,
                model__validation_data=(X_B_transformed, y_B),
                model__verbose_eval=0,
                model__early_stopping_rounds=self.early_stopping_rounds,
            )
        return self
