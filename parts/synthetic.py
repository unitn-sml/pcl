import numpy as np
from pymzn import minizinc, MiniZincModel, MiniZincUnsatisfiableError, gecode
from textwrap import dedent

from . import Problem
from .utils import _phi, _infer, _improve


PROBLEM = r"""
int: SIDE;

array[1..SIDE,1..SIDE] of var {0,1}: grid;
"""


class SyntheticProblem:
    """A square grid problem with pairwise XOR features.

    Each part is a square sub-grid. Parts possibly overlap.

    Attributes
    ----------
    side : int, defaults to 4
        Side of the grid
    part_side : int, defaults to 2
        Side of the parts sub-grid
    overlap : int, defaults to 0
        Amount of overlap between variables, in # of attributes per side.
    """
    def __init__(self, side=8, part_side=4, overlap=0, solver=gecode):
        self.side = side
        self.part_side = part_side
        self.n_features = 2 * side * (side - 1)
        self.solver = solver

        assert 1 < part_side < side

        def rc_to_features(r, c):
            features = []
            if c != side - 1: # horizontal feature
                features.append(r * (side - 1) + c)
            if r != side - 1: # vertical feature
                features.append(side * (side - 1) + r * side + c)
            return features

        def part_to_attributes_features(r, c):
            attributes = set()
            for s in range(r, r + part_side):
                for t in range(c, c + part_side):
                        attributes.add((s, t))

            features = set()
            # horizontal features
            for s in range(r, min(r + part_side, side)):
                for t in range(max(0, c - 1), min(c + part_side, side - 1)):
                    features.add(s * (side - 1) + t)
            # vertical features
            for s in range(max(0, r - 1), min(r + part_side, side - 1)):
                for t in range(c, min(c + part_side, side)):
                    features.add(side * (side - 1) + s * side + t)

            return attributes, features

        def features_to_attributes(features):
            attributes = set()
            half = side * (side - 1)
            for i in features:
                if i < half:
                    r, c = i // (side - 1), i % (side - 1)
                    if c != side - 1:
                        attributes.add((r, c, 'h'))
                else:
                    r, c = (i - half) // side, (i - half) % side
                    if r != side - 1:
                        attributes.add((r, c, 'v'))
            return attributes

        part = 0
        self.part_to_attributes, self.part_to_I = {}, {}
        for r in range(0, side - part_side + 2,  part_side - overlap):
            for c in range(0, side - part_side + 2, part_side - overlap):
                attributes, features = part_to_attributes_features(r, c)
                self.part_to_attributes[part] = attributes
                self.part_to_I[part] = features
                part += 1

        # FIXME generalize to non-4x4-by-2x2 grid
        self.parts = sorted(range(part), reverse=True)

        self.part_to_J = {}
        for i in range(len(self.parts)):
            part = self.parts[i]

            union_Is = set()
            for latter_i in range(i + 1, len(self.parts)):
                union_Is.update(self.part_to_I[self.parts[latter_i]])

            self.part_to_J[part] = self.part_to_I[part] - union_Is

        self.size = []
        for part in self.parts:
            I_minus_J = set(self.part_to_I[part]) - set(self.part_to_J[part])
            self.size.append(len(I_minus_J))

    def _format_grid(self, x):
        grid = x['grid']
        rows = []
        for r in range(self.side):
            rows.append(' '.join(list(map(str, grid[r]))))
        return '\n'.join(rows)

    def _part_phi(self, problem, part):
        features = []
        for p in self.parts:
            J = sorted(self.part_to_J[p])
            if part is None or p == part:
                for i in J:
                    if i < self.side * (self.side - 1):
                        r, c = i // (self.side - 1), i % (self.side - 1)
                        feat = 'grid[{r},{c}] != grid[{r},{c}+1]'.format(r=r+1, c=c+1)
                    else:
                        i2 = i - self.side * (self.side - 1)
                        r, c = i2 // self.side, i2 % self.side
                        feat = 'grid[{r},{c}] != grid[{r}+1,{c}]'.format(r=r+1, c=c+1)
                    features.append('2 * ({feat}) - 1'.format(feat=feat))
            else:
                features.extend([0] * len(J))

        expected_n_features = 2 * self.side * (self.side - 1)

        problem.parameter('FEATURES', set(range(1, expected_n_features + 1)))
        problem.array_variable('phi', 'FEATURES', 'int', features)
        return problem

    def _fix_other_parts(self, problem, part, x):
        """Fix the value of the attributes not in the given part."""
        grid = x['grid']
        for p in self.parts:
            if p != part:
                attributes = self.part_to_attributes[p]
                for r, c in attributes:
                    problem.constraint('grid[{},{}] = {}'
                           .format(r + 1, c + 1, grid[r][c]))
        return problem

    def initial_configuration(self):
        return minizinc(self._part_phi(MiniZincModel(_phi(PROBLEM)), None),
                        data={'SIDE': self.side},
                        solver=self.solver, force_flatten=True)[0]

    def phi(self, x):
        phi = minizinc(self._part_phi(MiniZincModel(_phi(PROBLEM)), None),
                       data={'SIDE': self.side, **x},
                       solver=self.solver, force_flatten=True,
                       output_vars=['phi'])[0]['phi']
        return np.array(phi)

    def multizinc(self, problem, **kwargs):
        timeout = kwargs.pop('timeout', None)
        if timeout is None:
            solns = minizinc(problem, **kwargs)
        else:
            solns = []
            while not len(solns):
                solns = minizinc(problem, **kwargs, timeout=timeout)
                timeout += 1
        return solns[0]

    def infer(self, w, x=None, part=None, timeout=None, local=False,
              solver=gecode):
        solver = solver or self.solver
        w_int = (w * 1000).astype(int)
        problem = MiniZincModel(_infer(PROBLEM))
        if part is None:
            problem = self._part_phi(problem, None)
        elif not local:
            problem = self._part_phi(problem, None)
            problem = self._fix_other_parts(problem, part, x)
        else:
            problem = self._part_phi(problem, part)
            problem = self._fix_other_parts(problem, part, x)
        return self.multizinc(problem, data={'SIDE': self.side, 'w': w_int},
                              solver=solver, timeout=timeout,
                              force_flatten=True)

    def improve(self, w, x, part=None, alpha=0.1, timeout=None, improve_margin=None, solver=gecode):
        solver = solver or self.solver
        w_int = (w * 1000).astype(int)
        if improve_margin is None:
            x_star = self.infer(w, x=x, part=part)
            regret = w_int.dot(self.phi(x_star) - self.phi(x))
            if regret == 0:
                return x
            improve_margin = int(w_int.dot(self.phi(x)) + alpha * regret)
        else:
            improve_margin = int(w_int.dot(self.phi(x)) + 1000 * improve_margin)
        problem = MiniZincModel(_improve(PROBLEM, improve_margin))
        problem = self._part_phi(problem, None)
        if part is not None:
            problem = self._fix_other_parts(problem, part, x)
        try:
            return self.multizinc(problem, data={'SIDE': self.side, 'w': w_int},
                                  solver=solver, timeout=timeout,
                                  force_flatten=True)
        except MiniZincUnsatisfiableError:
            return x

    def local_update(self, w, x, xbar, part, eta=1):
        mask = np.zeros_like(w)
        mask[list(self.part_to_J[part])] = 1
        return w + mask * eta * (self.phi(xbar) - self.phi(x))
