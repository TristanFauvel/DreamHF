# importing metrics
import types

import numpy as np
import pandas as pd
import sklearn
from optuna import create_study
from optuna.samplers import TPESampler
from scipy.stats import randint, uniform
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.utils import estimator_html_repr
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis, IPCRidge
from sksurv.metrics import concordance_index_censored
from xgbse import XGBSEStackedWeibull
from xgbse.converters import convert_y
from xgbse.metrics import concordance_index

from xgboost_wrapper import XGBSurvival

sklearn.set_config(transform_output="pandas")

def bind(instance, method):
    def binding_scope_fn(*args, **kwargs):
        return method(instance, *args, **kwargs)

    return binding_scope_fn


def cross_validation_score(pipeline, X_train, y_train, n_splits = 3, n_repeats = 1, early_stopping_rounds = 50):
    # Compute the cross-validation score, works with pandas dataframe (the sklearn function does not)
       
        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=0
        )
        score = 0
        for train_index, test_index in rkf.split(X_train):
            X_A, X_B = X_train.iloc[train_index,
                                    :], X_train.iloc[test_index, :]
            y_A, y_B = y_train[train_index], y_train[test_index]

            if not check_is_fitted(pipeline):
                pipeline[:-1].fit(X_A, y_A)

            X_B_transformed = pipeline.estimator.named_steps['preprocessor'].transform(
                X_B)

            pipeline.fit(
                X_A,
                y_A,
                estimator__validation_data=(X_B_transformed, y_B),
                estimator__verbose_eval=0,
                estimator__early_stopping_rounds=early_stopping_rounds
            )
            score += pipeline.score(X_B, y_B)
        score /= (n_repeats*n_splits)
        return score
        
        
class EarlyStoppingMonitor:
    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement
    
"""
def score(self, X, y):
    survival_prob = self.predict(X)
    risk_score = -survival_prob
    event, time = convert_y(y)
    return concordance_index_censored(event, time, risk_score)[0]
"""

def xgb_risk_score(model, X_test):  # OK for models in sksurv which predict the risk score
    # Predict the survival time, take the negative to convert to risk scores
    predictions = - model.pipeline.predict(X_test)
    scaler = MinMaxScaler()
    risk_score = scaler.fit_transform(predictions.reshape(-1, 1))
    #The range of this number has to be between 0 and 1, with larger numbers being associated with higher probability of having HF. The values, -Inf, Inf and NA, are not allowed.
    return risk_score.to_numpy().flatten()


class candidate_model:
    def __init__(self, n_taxa):
        self.n_taxa = n_taxa
        self.cv = 5
        if n_taxa > 0:
            self.base_distribution = dict(pca_transformer__reduce_dim=[
                'passthrough', PCA(0.95), PCA(0.98)])
        else :
            self.base_distribution =dict()

    def cross_validation(self, X_train, y_train, n_iter):
        randsearchcv = RandomizedSearchCV(
            self.pipeline,
            self.distributions,
            random_state=0,
            n_iter=n_iter,
            n_jobs=1,
            verbose=10,
            #error_score='raise',
            cv = self.cv
        )
        self.pipeline = randsearchcv.fit(X_train, y_train)
        return self

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        self.harrell_C_training = self.estimator.score(X_train, y_train)
        self.harrell_C_test = self.estimator.score(X_test, y_test)
        """
        self.harrell_C_training = concordance_index_censored(
            y_train['Event'], y_train['Event_time'], self.risk_score(X_train))[0]
        self.harrell_C_test = concordance_index_censored(
            y_test['Event'], y_test['Event_time'], self.risk_score(X_test))[0]
        
        result = {
            "Harrell C": [self.harrell_C_training, self.harrell_C_test
            ]
        }

        print(pd.DataFrame(result, index=["train", "test"]))

        return self

    def create_pipeline(self):
        numeric_transformer = Pipeline(
            steps=[
                ("mean_imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("frequent_imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, selector(
                    dtype_exclude=["bool", "category", "Int64"])),
                ("cat", categorical_transformer, selector(
                    dtype_include=["bool", "category", "Int64"])),
            ]
        )
        
        if self.n_taxa >0:
            pca_transformer = ColumnTransformer(
                transformers=[("reduce_dim", PCA(), selector(pattern="k__"))], remainder='passthrough')

            regressor = Pipeline(
                steps=[("preprocessor", preprocessor), ("pca_transformer", pca_transformer), ("estimator", self.estimator)])
        else:
            regressor = Pipeline(
                steps=[("preprocessor", preprocessor), ("estimator", self.estimator)])

        # Scale predictions : predictions are risk score between 0 and 1
                
        with open("regressor.html", "w") as f:
            f.write(estimator_html_repr(regressor))

        return regressor

class sksurv_model(candidate_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
        self.monitor = EarlyStoppingMonitor(25, 50)
                
    def risk_score(self, X_test):
        if hasattr(self.pipeline, 'predict_survival_function'):
            risk_score = 1- self.pipeline.predict_survival_function(X_test, return_array= True)[:,-1]
    
        else:
            predictions = self.pipeline.predict(X_test)  # Predict the risk score
            scaler = MinMaxScaler()
            risk_score = scaler.fit_transform(predictions.reshape(-1, 1)).to_numpy().flatten()
            #The range of this number has to be between 0 and 1, with larger numbers being associated with higher probability of having HF. The values, -Inf, Inf and NA, are not allowed.
        return risk_score 
    
class sksurv_gbt(sksurv_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
         
        self.estimator = GradientBoostingSurvivalAnalysis()
        
        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__learning_rate=uniform(loc=1e-2, scale=0.4),
            estimator__max_depth=randint(2, 6),
            estimator__loss=["coxph"], #ipcwls corresponds to aft model
            estimator__n_estimators=randint(100, 350),
            estimator__min_samples_split=randint(2, 6),
            estimator__min_samples_leaf=randint(1, 10),
            estimator__subsample=uniform(loc=0.5, scale=0.5),
            estimator__max_leaf_nodes=randint(2, 30),
            estimator__dropout_rate=uniform(loc=0, scale=0.5)
        )}


class CoxPH(sksurv_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
         
        self.cv = 10
        self.estimator = CoxPHSurvivalAnalysis(
            ties='breslow', tol=1e-09, verbose=0)

        self.pipeline = self.create_pipeline()

        
        self.distributions = {**self.base_distribution, **dict(
            estimator__alpha=uniform(loc=0, scale=1),
            estimator__n_iter=randint(80, 200)
        )}
     
class IPCRidge_sksurv(sksurv_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
        self.estimator = IPCRidge(alpha=1.0)

        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__alpha=uniform(loc=0, scale=1)
            )}


class Coxnet(sksurv_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
        self.estimator =CoxnetSurvivalAnalysis(n_alphas=100, l1_ratio=0.5)
        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__l1_ratio=uniform(loc=0, scale=1),
            estimator__n_alphas=randint(50, 200)
        )}
                 
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

        self.estimator = sklearn_wei(xgb_params)

        self.estimator.score = bind(self, score)

        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__aft_loss_distribution_scale=uniform(loc=0, scale=1),
            estimator__colsample_bynode=uniform(loc=0.5, scale=0.5),
            estimator__learning_rate=uniform(0, 0.5),
            estimator__max_depth=randint(2, 10),
            estimator__min_child_weight=randint(2, 10),
            estimator__subsample=uniform(loc=0.5, scale=0.5),
            estimator__tree_method=["hist"],
        )}

    def cross_validation(self, X_train, y_train, n_iter):
        randsearchcv = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.distributions,
            scoring=make_scorer(concordance_index),
            random_state=0,
            n_iter=n_iter,  # 300
            n_jobs=1,
            verbose=0,
            #error_score='raise'
        )
        search = randsearchcv.fit(X_train, y_train)
        self.estimator = search.best_estimator_
        return self 


class xgb_aft(sksurv_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
         
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

        self.estimator = XGBSurvival(self.base_params, num_boost_round=10000)

        self.pipeline = self.create_pipeline()

        
        self.distributions =  {**self.base_distribution, **dict(
            estimator__alpha=uniform(loc=0, scale=1),
            estimator__n_iter=randint(80, 200)
        )}
        
        
class xgb_optuna(candidate_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)
        # repeated K-folds
        self.N_SPLITS = 3
        self.N_REPEATS = 1

        # Optuna 
        self.RS = 124  # random state
        # XGBoost
        self.EARLY_STOPPING_ROUNDS = 50
        self.MULTIVARIATE = True

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

        self.estimator = XGBSurvival(self.base_params, num_boost_round=10000)
        self.pipeline = self.create_pipeline()

    def cross_validation(self, X_train, y_train, n_iter):
        study = create_study(direction="maximize", sampler=self.sampler)
        study.optimize(
            lambda trial: self.objective(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_iter,
            n_jobs=1,
        )
        self.optimal_hp = study.best_params
        self.pipeline.set_params(**self.optimal_hp)
        self.pipeline = self.pipeline.fit(X_train, y_train)
        return self

    def objective(
        self,
        trial,
        X_train,
        y_train
        ):
        # XGBoost parameters

        params = {
            "estimator__objective": "survival:aft",
            "estimator__eval_metric": "aft-nloglik",
            "estimator__aft_loss_distribution": "normal",
            "estimator__aft_loss_distribution_scale": trial.suggest_float('aft_loss_distribution_scale', 0.1, 10.0, log=True),
            "estimator__tree_method": "hist",
            "estimator__learning_rate": trial.suggest_float("learning_rate", 1e-2, 1, log=True),
            "estimator__max_depth": trial.suggest_int("max_depth", 2, 12),
            "estimator__booster": "dart",
            "estimator__subsample": trial.suggest_float("subsample", 0.4, 0.8, log=False),
            "estimator__alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "estimator__lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "estimator__gamma": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        }

        self.pipeline.set_params(**params)
        score = model_selection.cross_val_score(
            self.pipeline, X_train, y_train, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy    

    def risk_score(self, X_test):
        risk_score =  xgb_risk_score(self, X_test)
        return risk_score
    
    
class sksurv_gbt_optuna(sksurv_model):
    def __init__(self, n_taxa):
        super().__init__(n_taxa)

        # Optuna
        self.RS = 124  # random state
        # XGBoost
        self.EARLY_STOPPING_ROUNDS = 50
        self.MULTIVARIATE = True

        self.sampler = TPESampler(seed=self.RS, multivariate=self.MULTIVARIATE)

        self.estimator = GradientBoostingSurvivalAnalysis()

        self.pipeline = self.create_pipeline()

    def cross_validation(self, X_train, y_train, n_iter):
        self.N_TRIALS = n_iter

        study = create_study(direction="maximize", sampler=self.sampler)
        study.optimize(
            lambda trial: self.objective(
                trial,
                X_train,
                y_train,
            ),
            n_trials=self.N_TRIALS,
            n_jobs=1,
        )
        self.optimal_hp = study.best_params
        self.pipeline.set_params(**self.optimal_hp)
        self.pipeline = self.pipeline.fit(X_train, y_train)
        return self

    def objective(
        self,
        trial,
        X_train,
        y_train
    ):

        params = {
            "pca_transformer__reduce_dim": trial.suggest_categorical("pca_transformer__reduce_dim", ['passthrough', PCA(0.95), PCA(0.98)]),
            "estimator__learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.4, log=False),
            "estimator__max_depth": trial.suggest_int("max_depth", 2, 6),
            "estimator__loss": "coxph",
            "estimator__n_estimators": trial.suggest_int("n_estimators", 100, 350),
            "estimator__min_samples_split":  trial.suggest_int("min_samples_split", 2, 6),
            "estimator__min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "estimator__subsample": trial.suggest_float("subsample", 0.4, 0.8, log=False),
            "estimator__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 30),
            "estimator__dropout_rate": trial.suggest_float("dropout_rate", 0, 1, log=False)
        }
        self.pipeline.set_params(**params)
        score = model_selection.cross_val_score(
            self.pipeline, X_train, y_train, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy
