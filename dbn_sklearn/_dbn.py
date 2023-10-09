import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression

from sklearn.utils import gen_batches, check_random_state

_STOCHASTIC_SOLVERS = ["sgd", "adam"]
class DBN(MLPClassifier):
    def __init__(self, rbm_lr=0.001, rbm_epoch=100, **kwargs):
        super().__init__(activation='logistic', **kwargs)
        self.rbm_lr = rbm_lr
        self.rbm_epoch = rbm_epoch

    def _pretrain(self, X, y):
        if self.batch_size == "auto":
            batch_size = min(200, X.shape[0])
        visual_layer = X.copy()
        for i in range(self.n_layers_-2):
            rbm = BernoulliRBM(
                n_components=self.hidden_layer_sizes[i], 
                n_iter=self.rbm_epoch, 
                learning_rate=self.rbm_lr,
                batch_size=batch_size).fit(visual_layer)
            self.coefs_[i] = np.transpose(rbm.components_)
            self.intercepts_[i] = rbm.intercept_hidden_
            visual_layer = rbm.transform(visual_layer)
        lr = LogisticRegression(penalty='none').fit(visual_layer, y.argmax(axis=1))
        self.coefs_[-1] = lr.coef_.T
        self.intercepts_[-1] = lr.intercept_

    def _fit(self, X, y, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        # self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X, y = self._validate_input(X, y, incremental, reset=first_pass)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, layer_units, X.dtype)
            self._pretrain(X, y)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]
        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(
                X,
                y,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )
        # Run the LBFGS solver
        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X, y, activations, deltas, coef_grads, intercept_grads, layer_units
            )
        return self
