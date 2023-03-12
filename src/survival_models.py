# importing metrics
from typing import Any, Callable, Dict

import numpy as np
import optuna
import pandas as pd
from optuna import create_study
from optuna.samplers import TPESampler
from scipy.stats import randint, uniform
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import estimator_html_repr
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis, IPCRidge
from sksurv.metrics import concordance_index_censored
from xgbse import XGBSEStackedWeibull
from xgbse.metrics import concordance_index

from xgboost_wrapper import XGBSurvival


def bind(instance: object, method: Callable) -> Callable:
    """Bind an instance to a method, returning a new callable object.

    Args:
        instance (object): The object to bind to the method.
        method (callable): The method to bind to the object.

    Returns:
        callable: A new callable object that calls the method with the given instance as its first argument.
    """
    def binding_scope_fn(*args, **kwargs):
        return method(instance, *args, **kwargs)

    return binding_scope_fn


def cross_validation_score(pipeline, x_train, y_train, n_splits: int = 3, n_repeats: int = 1,
                           early_stopping_rounds: int = 50) -> float:
    """Compute the cross-validation score.

    This function works with pandas dataframes (unlike the similar function in sklearn).

    Args:
        pipeline (object): The pipeline object to use for fitting and scoring.
        x_train (pandas.DataFrame): The input features for training.
        y_train (numpy.ndarray): The target variable for training.
        n_splits (int): The number of splits to use in the cross-validation. Defaults to 3.
        n_repeats (int): The number of times to repeat the cross-validation. Defaults to 1.
        early_stopping_rounds (int): The number of rounds with no improvement to stop early. Defaults to 50.

    Returns:
        float: The mean score across all cross-validation runs.
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    score = 0
    for train_index, test_index in rkf.split(x_train):
        X_A, X_B = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
        y_A, y_B = y_train[train_index], y_train[test_index]

        if not check_is_fitted(pipeline):
            pipeline[:-1].fit(X_A, y_A)

        X_B_transformed = pipeline.estimator.named_steps['preprocessor'].transform(
            X_B)

        pipeline.fit(X_A, y_A,
                     estimator__validation_data=(X_B_transformed, y_B),
                     estimator__verbose_eval=0,
                     estimator__early_stopping_rounds=early_stopping_rounds)

        score += pipeline.score(X_B, y_B)

    score /= (n_repeats*n_splits)
    return score


class EarlyStoppingMonitor:
    """
    A callback function to stop the training process if there is no improvement in the evaluation metric.

    Parameters
    ----------
    window_size : int
        The size of the sliding window to compute the average improvement.
    max_iter_without_improvement : int
        The maximum number of iterations without improvement before stopping the training process.
    """

    def __init__(self, window_size: int, max_iter_without_improvement: int):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration: int, estimator, args) -> bool:
        """
        Callback function to be called after each iteration of the training process.

        Parameters
        ----------
        iteration : int
            The current iteration of the training process.
        estimator : estimator object
            The estimator object to evaluate.
        args : dict
            The arguments passed to the estimator.

        Returns
        -------
        bool
            True if the training process should be stopped, False otherwise.
        """
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


def xgb_risk_score(model, x_test) -> np.ndarray:
    """
    Predict the risk score for the test set using a model.

    Parameters
    ----------
    model : estimator object
        The estimator object to use for prediction.
    x_test : array-like, shape (n_samples, n_features)
        The input data to predict the risk score.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        The predicted risk score.
    """
    # Predict the survival time, take the negative to convert to risk scores
    predictions = -model.pipeline.predict(x_test)
    scaler = MinMaxScaler()
    risk_score = scaler.fit_transform(predictions.reshape(-1, 1))
    # The range of this number has to be between 0 and 1, with larger numbers being associated with higher probability of having HF.
    # The values, -Inf, Inf and NA, are not allowed.
    if not isinstance(risk_score, np.ndarray):
        risk_score = risk_score.to_numpy()
    risk_score = risk_score.flatten()
    return risk_score


class candidate_model:
    def __init__(self, n_taxa: int):
        """
        Initializes a candidate model instance.

        Args:
        - n_taxa (int): number of taxa.

        Returns:
        - None.
        """
        self.n_taxa = n_taxa

        self.cv = RepeatedKFold(n_splits=10, n_repeats=10)

        if n_taxa > 0:
            self.base_distribution = dict(pca_transformer__reduce_dim=[
                'passthrough', PCA(0.95), PCA(0.98)])
        else:
            self.base_distribution = dict()

        self.estimator = None
        self.harrell_c_training = None
        self.harrell_c_test = None

    def cross_validation(self, x_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int) -> "candidate_model":
        """
        Performs cross-validation on the candidate model.

        Args:
        - x_train (pd.DataFrame): training set features.
        - y_train (pd.DataFrame): training set labels.
        - n_iter (int): number of iterations.

        Returns:
        - self.
        """
        randsearchcv = RandomizedSearchCV(
            self.pipeline,
            self.distributions,
            random_state=0,
            n_iter=n_iter,
            n_jobs=-1,
            verbose=10,
            # error_score='raise',
            cv=self.cv
        )
        self.pipeline = randsearchcv.fit(x_train, y_train)
        return self

    def evaluate(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> "candidate_model":
        """
        Evaluates the candidate model.

        Args:
        - x_train (pd.DataFrame): training set features.
        - x_test (pd.DataFrame): test set features.
        - y_train (pd.DataFrame): training set labels.
        - y_test (pd.DataFrame): test set labels.

        Returns:
        - self.
        """
        self.harrell_c_training = concordance_index_censored(
            y_train['Event'], y_train['Event_time'], self.risk_score(x_train))[0]
        self.harrell_c_test = concordance_index_censored(
            y_test['Event'], y_test['Event_time'], self.risk_score(x_test))[0]

        result = {
            "Harrell C": [self.harrell_c_training, self.harrell_c_test
                          ]
        }

        print(pd.DataFrame(result, index=["train", "test"]))

        return self

    def create_pipeline(self) -> Pipeline:
        """
        Creates a pipeline that preprocesses the data, selects the most informative features,
        and fits a model to the data.

        Returns:
        - regressor (Pipeline): A pipeline that preprocesses the data, selects the most informative
                                features, and fits a model to the data.
        """
        # Define transformers for numeric and categorical columns
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

        # Use ColumnTransformer to apply transformers to appropriate columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, selector(
                    dtype_exclude=["bool", "category", "Int64"])),
                ("cat", categorical_transformer, selector(
                    dtype_include=["bool", "category", "Int64"])),
            ]
        )

        # Use RFE to select the most informative features
        rfe = RFE(estimator=self.estimator, n_features_to_select=50, step=0.1)

        # Use Pipeline to apply ColumnTransformer and RFE in sequence
        feature_selector = Pipeline(
            steps=[("preprocessor", preprocessor), ("rfe", rfe)])

        # Use Pipeline to apply feature selection and fit the model
        regressor = Pipeline(
            steps=[("feature_selector", feature_selector), ("estimator", self.estimator)])

        # Scale predictions between 0 and 1
        with open("regressor.html", "w") as f:
            f.write(estimator_html_repr(regressor))

        return regressor


class sksurv_model(candidate_model):
    def __init__(self, n_taxa: int):
        """
        Initializes an instance of the `sksurv_model` class.

        Parameters:
        -----------
        n_taxa : int
            Number of taxa to be included in the model

        Returns:
        --------
        None
        """
        super().__init__(n_taxa)
        self.monitor = EarlyStoppingMonitor(25, 50)

    def risk_score(self, x_test: pd.DataFrame) -> np.ndarray:
        """
        Returns the risk score for a given input data.

        Parameters:
        -----------
        x_test : pd.DataFrame
            Input data containing all the features

        Returns:
        --------
        risk_score : np.ndarray
            The risk score between 0 and 1, with larger numbers being associated with a higher probability of having HF.
        """
        if hasattr(self.pipeline, 'predict_survival_function'):
            risk_score = 1 - \
                self.pipeline.predict_survival_function(
                    x_test, return_array=True)[:, -1]

        else:
            predictions = self.pipeline.predict(
                x_test)  # Predict the risk score
            scaler = MinMaxScaler()

            risk_score = scaler.fit_transform(predictions.reshape(-1, 1))

        if not isinstance(risk_score, np.ndarray):
            risk_score = risk_score.to_numpy()

        risk_score = risk_score.flatten()

        return risk_score


class sksurv_gbt(sksurv_model):
    def __init__(self, n_taxa: int):
        """
        Initializes an instance of the `sksurv_gbt` class.

        Parameters:
        -----------
        n_taxa : int
            Number of taxa to be included in the model

        Returns:
        --------
        None
        """
        super().__init__(n_taxa)

        self.estimator = GradientBoostingSurvivalAnalysis()

        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__learning_rate=uniform(loc=1e-2, scale=0.4),
            estimator__max_depth=randint(2, 6),
            estimator__loss=["coxph"],  # ipcwls corresponds to aft model
            estimator__n_estimators=randint(100, 350),
            estimator__min_samples_split=randint(2, 6),
            estimator__min_samples_leaf=randint(1, 10),
            estimator__subsample=uniform(loc=0.5, scale=0.5),
            estimator__max_leaf_nodes=randint(2, 30),
            estimator__dropout_rate=uniform(loc=0, scale=0.5)
        )}


class sksurv_RF(sksurv_model):
    def __init__(self, n_taxa: int):
        """
        Initializes an instance of the `sksurv_RF` class.

        Parameters:
        -----------
        n_taxa : int
            Number of taxa to be included in the model

        Returns:
        --------
        None
        """
        super().__init__(n_taxa)

        self.estimator = RandomSurvivalForest(n_estimators=1000,
                                              min_samples_split=10,
                                              min_samples_leaf=15,
                                              n_jobs=-1,
                                              random_state=1)

        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__n_estimators=randint(400, 1000),
            estimator__min_samples_split=randint(2, 20),
            estimator__min_samples_leaf=randint(1, 20),
        )}


class CoxPH(sksurv_model):
    """
    Cox proportional hazards regression model.

    Parameters:
        n_taxa (int): Number of taxa in the dataset
    """

    def __init__(self, n_taxa: int):
        super().__init__(n_taxa)

        self.estimator = CoxPHSurvivalAnalysis(
            alpha=0.1, ties='breslow', tol=1e-09, verbose=0)
        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__alpha=uniform(loc=0, scale=1),
            estimator__n_iter=randint(80, 200)
        )}


class IPCRidge_sksurv(sksurv_model):
    """
    IPC Ridge survival model.

    Parameters:
        n_taxa (int): Number of taxa in the dataset
    """

    def __init__(self, n_taxa: int):
        super().__init__(n_taxa)

        self.estimator = IPCRidge(alpha=1.0)
        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__alpha=uniform(loc=0, scale=1)
        )}


class Coxnet(sksurv_model):
    """
    Coxnet survival model.

    Parameters:
        n_taxa (int): Number of taxa in the dataset
    """

    def __init__(self, n_taxa: int):
        super().__init__(n_taxa)

        self.estimator = CoxnetSurvivalAnalysis(n_alphas=100, l1_ratio=0.5)
        self.pipeline = self.create_pipeline()

        self.distributions = {**self.base_distribution, **dict(
            estimator__l1_ratio=uniform(loc=0, scale=1),
            estimator__n_alphas=randint(50, 200)
        )}


class sklearn_wei(XGBSEStackedWeibull):

    def get_params2(self):
        """Get the XGBoost parameters for the estimator."""
        return(self.get_params()['xgb_params'])

    def set_params(self, **params):
        """Set the XGBoost parameters for the estimator."""
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

    def cross_validation(self, x_train, y_train, n_iter):
        randsearchcv = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.distributions,
            scoring=make_scorer(concordance_index),
            random_state=0,
            n_iter=n_iter,  # 300
            n_jobs=1,
            verbose=0,
            # error_score='raise'
        )
        search = randsearchcv.fit(x_train, y_train)
        self.estimator = search.best_estimator_
        return self


class xgb_aft(sksurv_model):
    """
    A class for XGBoost survival analysis using accelerated failure time (AFT) models.
    Inherits from the sksurv_model class.

    Attributes
    ----------
    base_params : dict
        Dictionary containing default parameters for the XGBoost AFT model.
    estimator : XGBSurvival object
        XGBoost survival model object with the base parameters.
    pipeline : Pipeline object
        Pipeline object with the XGBoost AFT model and pre-processing steps.
    distributions : dict
        Dictionary containing the distributions to sample for hyperparameter tuning.

    Methods
    -------
    create_pipeline() -> Pipeline:
        Create a Pipeline object with the pre-processing steps and XGBoost AFT model.
    fit(x_train: Any, y_train: Any) -> xgb_aft:
        Fit the XGBoost AFT model on the training data.
    predict(x_test: Any) -> Tuple[np.ndarray, np.ndarray]:
        Predict the survival probability and the time to event for the test data.
    score(x_test: Any, y_test: Any) -> float:
        Calculate the concordance index (c-index) for the test data.
    tune_hyperparameters(x_train: Any, y_train: Any, n_iter: int) -> xgb_aft:
        Use random search to tune the hyperparameters of the XGBoost AFT model.
    """

    def __init__(self, n_taxa: int):
        """
        Initializes the xgb_aft object.

        Parameters
        ----------
        n_taxa : int
            The number of taxa in the dataset.
        """

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

        self.distributions = {**self.base_distribution, **dict(
            estimator__alpha=uniform(loc=0, scale=1),
            estimator__n_iter=randint(80, 200)
        )}


class xgb_optuna(candidate_model):
    """
    A class for hyperparameter optimization using XGBoost and Optuna.
    """

    def __init__(self, n_taxa: int):
        """
        Initializes an instance of the xgb_optuna class.

        Args:
            n_taxa: The number of taxa in the dataset.

        Returns:
            None.
        """
        super().__init__(n_taxa)

        # repeated K-folds
        self.N_SPLITS: int = 3
        self.N_REPEATS: int = 1

        # Optuna
        self.RS: int = 124  # random state

        # XGBoost
        self.EARLY_STOPPING_ROUNDS: int = 50
        self.MULTIVARIATE: bool = True

        self.sampler = TPESampler(seed=self.RS, multivariate=self.MULTIVARIATE)
        self.base_params: Dict[str, Any] = {
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

    def cross_validation(self, x_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int) -> 'xgb_optuna':
        """
        Performs hyperparameter optimization using Optuna.

        Args:
            x_train: The training data.
            y_train: The training labels.
            n_iter: The number of iterations for the optimization.

        Returns:
            An instance of the xgb_optuna class.
        """
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(
            lambda trial: self.objective(
                trial,
                x_train,
                y_train
            ),
            n_trials=n_iter,
            n_jobs=1,
        )
        self.optimal_hp = study.best_params
        self.pipeline.set_params(**self.optimal_hp)
        self.pipeline = self.pipeline.fit(x_train, y_train)
        return self

    def objective(
        self,
        trial: optuna.Trial,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame
    ) -> float:
        """
        The objective function for hyperparameter optimization.

        Args:
            trial: The trial object.
            x_train: The training data.
            y_train: The training labels.

        Returns:
            The accuracy of the model.
        """
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
            self.pipeline, x_train, y_train, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy

    def risk_score(self, x_test):
        risk_score = xgb_risk_score(self, x_test)
        return risk_score


class sksurv_gbt_optuna:
    def __init__(self, n_taxa: int):
        """Initialization of sksurv_gbt_optuna object

        Args:
        n_taxa: int. Number of taxa.
        """
        self.n_taxa = n_taxa
        self.RS = 124  # random state for optuna
        self.EARLY_STOPPING_ROUNDS = 50
        self.MULTIVARIATE = True
        self.sampler = TPESampler(seed=self.RS, multivariate=self.MULTIVARIATE)
        self.estimator = GradientBoostingSurvivalAnalysis()
        self.pipeline = self.create_pipeline()

    def create_pipeline(self) -> Pipeline:
        """Creates a Pipeline object

        Returns:
        Pipeline. Pipeline object with PCA, StandardScaler, and CoxPHSurvivalAnalysis steps.
        """
        steps = [
            ('pca_transformer', PCA()),
            ('scaler', StandardScaler()),
            ('estimator', CoxPHSurvivalAnalysis())
        ]
        pipeline = Pipeline(steps)
        return pipeline

    def cross_validation(self, x_train: np.ndarray, y_train: np.ndarray, n_iter: int) -> 'sksurv_gbt_optuna':
        """Performs cross validation on training data with given number of iterations

        Args:
        x_train: np.ndarray. Array of shape (n_samples, n_features) representing the training data.
        y_train: np.ndarray. Array of shape (n_samples,) representing the time-to-event and event data.
        n_iter: int. Number of iterations.

        Returns:
        sksurv_gbt_optuna. Returns itself.
        """
        self.N_TRIALS = n_iter
        study = create_study(direction="maximize", sampler=self.sampler)
        study.optimize(
            lambda trial: self.objective(
                trial,
                x_train,
                y_train,
            ),
            n_trials=self.N_TRIALS,
            n_jobs=1,
        )
        self.optimal_hp = study.best_params
        self.pipeline.set_params(**self.optimal_hp)
        self.pipeline = self.pipeline.fit(x_train, y_train)
        return self

    def objective(
        self,
        trial,
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> float:
        """Objective function for the optimizer

        Args:
        trial: The current trial of the optimizer.
        x_train: np.ndarray. Array of shape (n_samples, n_features) representing the training data.
        y_train: np.ndarray. Array of shape (n_samples,) representing the time-to-event and event data.

        Returns:
        float. Returns accuracy of cross validation.
        """

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
            self.pipeline, x_train, y_train, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy
