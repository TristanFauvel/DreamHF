import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.utils import check_consistent_length
from sksurv.nonparametric import kaplan_meier_estimator


class _ExtentedStepFunction:
    """Callable step function.
    Compared to the standard step function, the extended step function allows to compute
    the value of the function at points beyond the range of x used in the function definition.

    .. math::

        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        Values on the x axis in ascending order.

    y : ndarray, shape = (n_points,)
        Corresponding values on the y axis.

    a : float, optional, default: 1.0
        Constant to multiply by.

    b : float, optional, default: 0.0
        Constant offset term.
    """

    def __init__(self, x, y, a=1.0, b=0.0):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def __call__(self, x):
        """Evaluate step function.

        Parameters
        ----------
        x : float|array-like, shape=(n_values,)
            Values to evaluate step function at.

        Returns
        -------
        y : float|array-like, shape=(n_values,)
            Values of step function at `x`.
        """
        x = np.atleast_1d(x)
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")

        i = np.searchsorted(self.x, x, side="left")

        n = np.sum(i >= self.x.shape[0])
        self.x = np.concatenate([self.x, self.x[-1] * np.ones(n)])
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (self.x, self.y, self.a, self.b)


def HosmerLemeshowSurvival(times, model, X_test, y_test, df=2, Q=10):
    """
    df = 2 # Cook-Ridler test
    df = 1 # D'Agostino-Nam test
    """

    if isinstance(times, int):
        times = np.array([times])

    nt = times.shape[0]

    predictions = model.predict_survival_function(X_test)
    pred_surv_prob = np.row_stack([fn(times) for fn in predictions])

    id_surv_prob_sorted = np.argsort(pred_surv_prob[:, -1])
    pred_surv_prob = pred_surv_prob[id_surv_prob_sorted, :]

    categories = pd.cut(
        pred_surv_prob[:, -1],
        np.percentile(pred_surv_prob[:, -1], np.linspace(0, 100, Q + 1)),
        labels=False,
        include_lowest=True,
        duplicates='drop'
    )
    # In case where there are duplicate edges, recompute the effective number of groups
    Q = categories.max()
    expevents = np.zeros((Q, nt))
    obsevents = np.zeros((Q, nt))
    pi = np.zeros((Q, nt))

    for i in range(Q):
        KM_times, KM_survival_prob = kaplan_meier_estimator(
            y_test[categories == i].Event, y_test[categories == i].Event_time
        )
        KM_est = _ExtentedStepFunction(KM_times, KM_survival_prob)
        KM_i_t = KM_est(times)
        ni = np.sum(categories == i)
        obsevents[i, :] = ni * (1 - KM_i_t)
        expevents[i, :] = (1 - pred_surv_prob[categories == i, :]).sum(axis=0)
        pi[i, :] = expevents[i, :] / ni

    chisq_value = np.sum((obsevents - expevents) ** 2 /
                         (expevents * (1 - pi)), axis=0)
    pvalue = 1 - chi2.cdf(chisq_value, Q - df)

    if nt == 1:
        chisq_value = chisq_value[0]
        pvalue = pvalue[0]

    return {'chisq_value': chisq_value, 'pvalue': pvalue}
