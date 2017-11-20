import numpy as np


class User:
    def __init__(self, problem, w_star, alpha=0.1, min_regret=0,
                 calculate_true_regret=False):
        self.problem = problem
        self.w_star = w_star
        self.alpha = alpha
        self.min_regret = min_regret
        self.calculate_true_regret = calculate_true_regret
        if self.calculate_true_regret:
            self.x_star = problem.infer(w_star)

    def __repr__(self):
        return 'User({})'.format(self.w_star)

    def utility(self, x):
        return np.dot(self.w_star, self.problem.phi(x))

    def regret(self, x):
        """
        If self.calculate_true_regret = False, returns the utility u^* of the
        object, then calculate the regret after running all the experiment and
        calculate the regret according to the object with highest utility found.
        """
        if not self.calculate_true_regret:
            return None
        return self.utility(self.x_star) - self.utility(x)

    def cregret(self, x, part, solver=None):
        x_star = self.problem.infer(self.w_star, x=x, part=part, solver=solver)
        return self.utility(x_star) - self.utility(x)

    def is_satisfied(self, x):
        return self.regret(self, x) / self.u_star < self.min_regret

    def improve(self, x, part=None, timeout=None, solver=None, improve_margin=None):
        try:
            alpha = float(self.alpha)
        except ValueError:
            if self.alpha == 'pow2':
                alpha = 2 ** -len(part)
            else:
                raise NotImplementedError()
        return self.problem.improve(self.w_star, x, part=part, alpha=alpha, timeout=timeout, solver=solver, improve_margin=improve_margin)
