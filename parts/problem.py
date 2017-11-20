import numpy as np
from pymzn import MiniZincModel, minizinc, MiniZincUnsatisfiableError


class Problem:
    """A problem instance.

    Parameters
    ----------
    n_attributes : int
        Number of attributes x_i.
    n_features : int
        Number of features phi_j.
    parts : list of list of int
        Parts to be used for elicitation.
    x_type : str
        MiniZinc type of the attributes.
    phi_type : str
        MiniZinc type of the features.
    """
    def __init__(self, n_attributes, n_features, parts, x_type, phi_type):
        self.n_attributes = n_attributes
        self.n_features = n_features
        self._parts = parts
        self.parts = list(range(len(parts)))
        self.x_type = x_type
        self.phi_type = phi_type

    def initial_configuration(self):
        attributes = set(range(1, self.n_attributes + 1))
        features = set(range(1, self.n_features + 1))

        model = MiniZincModel()
        model.parameter('ATTRIBUTES', attributes)
        model.array_variable('x', 'ATTRIBUTES', self.x_type)
        model.parameter('FEATURES', features)
        model.array_variable('phi', 'FEATURES', self.phi_type)
        self._declare_constraints(model, 'x', 'phi')
        model.solve('satisfy')
        return np.array(minizinc(model)[0]['x'], dtype=int)

    def _fix_context(self, model, xvar, x, attributes, part):
        for i in attributes:
            if not i in part:
                model.constraint('{}[{}] == {}'.format(xvar, i, x[i - 1]))

    def phi(self, x):
        """Computes the feature representation of a configuration.

        Parameters
        ----------
        x : ndarray of shape (n_attributes,)
            A complete configuration.

        Returns
        -------
        phi : ndarray of shape (n_features,)
            The feature representation of x.
        """
        attributes = set(range(1, self.n_attributes + 1))
        features = set(range(1, self.n_features + 1))

        model = MiniZincModel()
        model.parameter('ATTRIBUTES', attributes)
        model.array_variable('x', 'ATTRIBUTES', self.x_type)
        model.parameter('FEATURES', features)
        model.array_variable('phi', 'FEATURES', self.phi_type)
        self._declare_constraints(model, 'x', 'phi')
        model.solve('satisfy')

        phi = minizinc(model, data={'x': x}, output_vars=['phi'])[0]['phi']
        return np.array(phi)

    def infer(self, w, x=None, part=None):
        """Infers the best (partial) configuration.

        Parameters
        ----------
        w : ndarray of shape (n_features,)
            Weights to optimize the configuration for.
        x : ndarray of shape (n_attributes,), defaults to None
            The starting configuration.
        part : list[int], defaults to None
            Indices of attributes in the part to be inferred. If None, the
            complete configuration will be inferred.

        Returns
        -------
        x : ndarray of shape (n_attributes,)
            The inferred configuration.
        """
        attributes = set(range(1, self.n_attributes + 1))
        features = set(range(1, self.n_features + 1))

        if part is None:
            part = list(sorted(attributes))
        else:
            part = self._parts[part]

        model = MiniZincModel()
        model.parameter('ATTRIBUTES', attributes)
        model.array_variable('x', 'ATTRIBUTES', self.x_type)
        model.parameter('FEATURES', features)
        model.array_variable('phi', 'FEATURES', self.phi_type)
        self._declare_constraints(model, 'x', 'phi')
        model.parameter('w', w)
        model.variable('utility', 'var float',
                       value='sum(j in FEATURES)(w[j] * phi[j])')
        model.solve('maximize utility')
        self._fix_context(model, 'x', x, attributes, part)

        x = minizinc(model, output_vars=['x'])[0]['x']
        return np.array(x, dtype=int)

    def improve(self, w, x, part=None, alpha=0.1):
        """Improves the given configuration by solving:

        .. math::

            min_{xbar_I}    \| xbar_I - x_I \|
            s.t.            u^*_I(xbar_I;x_M) - u^*_I(x_I;x_M) >=
                                alpha_I (u^*(o_I;x_M) - u^*_I(x_I;x_M))

        Parameters
        ----------
        w : ndarray of shape (n_features,)
            Weights to improve the configuration for.
        x : ndarray of shape (n_attributes,)
            The starting configuration.
        part : list of int, defaults to None
            List of attribute indices to improve. The rest will be fixed to
            their values in x.
        alpha : float in (0, 1]
            Degree of improvement to be made.

        Returns
        -------
        xbar : ndarray of shape (n_attributes,)
            The improved configuration or None if no improvement can be found.
        """
        o = self.infer(w, x, part=part)

        attributes = set(range(1, self.n_attributes + 1))
        features = set(range(1, self.n_features + 1))

        if part is None:
            part = list(sorted(attributes))
        else:
            part = self._parts[part]

        model = MiniZincModel()
        model.parameter('ATTRIBUTES', attributes)
        model.array_variable('xbar', 'ATTRIBUTES', self.x_type)
        model.parameter('FEATURES', features)
        model.parameter('phi', self.phi(x))
        model.array_variable('phibar', 'FEATURES', self.phi_type)
        self._declare_constraints(model, 'xbar', 'phibar')
        model.parameter('w', w)
        model.variable('difference', 'var float',
                  value='sum(j in FEATURES)(phi[j] != phibar[j])')
        model.solve('minimize difference')
        model.constraint('sum(j in FEATURES)(w[j] * (phibar[j] - phi[j])) > 0')
        self._fix_context(model, 'xbar', x, attributes, part)

        try:
            xbar = minizinc(model, output_vars=['xbar'])[0]['xbar']
            return np.array(xbar, dtype=int)
        except MiniZincUnsatisfiableError:
            return None
