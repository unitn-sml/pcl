
import numpy as np
from pymzn import minizinc, MiniZincModel, MiniZincUnsatisfiableError, gecode
from textwrap import dedent

from . import Problem
from .utils import _phi, _infer, _improve



PROBLEM = r"""

int: N_DAYS;
set of int: DAYS = 1..N_DAYS;

int: N_SLOTS;
set of int: SLOTS = 1..N_SLOTS;

int: N_ACTIVITIES;
set of int: ACTIVITIES = 1..N_ACTIVITIES;

array[DAYS,SLOTS] of var 1..N_ACTIVITIES: schedule;
int: NO_ACTIVITY = N_ACTIVITIES;

% ===========================================================================
% Background Knowledge
% ===========================================================================

int: N_BODY_PARTS;
set of int: BODY_PARTS = 1..N_BODY_PARTS;

% Context: response of athlete to performing an activity
array[ACTIVITIES,BODY_PARTS] of int: ACTIVITY_TO_POWER;
array[ACTIVITIES,BODY_PARTS] of int: ACTIVITY_TO_FATIGUE;

% Context: when is the athlete available?
array[DAYS,SLOTS] of {0,1}: AVAILABLE;

% To prevent injuries, the total fatigue of any body part in three consecutive
% slots should not be above a given threshold
int: MAX_FATIGUE;

constraint
    forall(bodypart in BODY_PARTS)(
        forall(day in DAYS)(
            forall(slot in 1..N_SLOTS-2)(
                (ACTIVITY_TO_FATIGUE[schedule[day,slot],bodypart]
                 + ACTIVITY_TO_FATIGUE[schedule[day,slot+1],bodypart]
                 + ACTIVITY_TO_FATIGUE[schedule[day,slot+2],bodypart])
                 <= MAX_FATIGUE)));

% Athletes can not practice when they are available
constraint
    forall(day in DAYS)(
        forall(slot in SLOTS)(
            (AVAILABLE[day,slot] = 1)
            \/ (schedule[day,slot] = NO_ACTIVITY)));
"""




_BODY_PARTS = ['arms', 'torso', 'back', 'legs', 'heart']
_ACTIVITY_TO_EFFECT = np.array([
        [[4, 1, 2, 1, 2], [3, 2, 2, 1, 1]], # pushups
        [[6, 4, 4, 3, 4], [4, 2, 2, 1, 2]], # weights
        [[0, 1, 1, 2, 1], [0, 1, 1, 1, 1]], # walking
        [[0, 2, 2, 5, 3], [0, 1, 1, 4, 4]], # running
        [[0, 1, 1, 6, 4], [0, 1, 2, 3, 1]], # squats
        [[0, 5, 4, 1, 2], [0, 4, 2, 0, 1]], # abs
        [[4, 3, 3, 4, 2], [4, 2, 3, 2, 3]], # swimming
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], # no activity
    ])


class SportProblem(Problem):

    def __init__(self, num_days=7, num_slots=5, max_fatigue=10, solver=gecode):
        rng = np.random.RandomState(12345)
        available = rng.binomial(1, 0.4, size=(num_days, num_slots))

        self.data = {
                'N_DAYS': num_days,
                'N_SLOTS': num_slots,
                'N_ACTIVITIES': len(_ACTIVITY_TO_EFFECT),
                'N_BODY_PARTS': len(_BODY_PARTS),
                'ACTIVITY_TO_POWER': _ACTIVITY_TO_EFFECT[:,0],
                'ACTIVITY_TO_FATIGUE': _ACTIVITY_TO_EFFECT[:,1],
                'AVAILABLE': available,
                'MAX_FATIGUE': max_fatigue,
            }

        self.n_features = len(_BODY_PARTS) * num_days * 2 + (num_days - 1) * 2
        self.parts = list(range(num_days))
        self.size = [0] + [2] * (num_days - 1)
        self.solver = solver

    def _part_phi(self, problem, part):
        """Define phi in terms of J[part]."""
        n_days, n_body_parts = self.data['N_DAYS'], self.data['N_BODY_PARTS']

        features = []
        for day in range(n_days):
            if part is None or part == day:
                features.extend([
                        'sum(slot in SLOTS)(ACTIVITY_TO_POWER[schedule[{day},slot],{bodypart}])'
                            .format(day=day+1, bodypart=bodypart+1)
                        for bodypart in range(n_body_parts)
                    ])
            else:
                features.extend([0] * n_body_parts)
        for day in range(n_days):
            if part is None or part == day:
                features.extend([
                        'sum(slot in SLOTS)(ACTIVITY_TO_FATIGUE[schedule[{day},slot],{bodypart}])'
                            .format(day=day+1, bodypart=bodypart+1)
                        for bodypart in range(n_body_parts)
                    ])
            else:
                features.extend([0] * n_body_parts)
        for day in range(1, n_days):
            if part is None or part == day:
                temp = dedent('''\
                        schedule[{day}-1,1] == schedule[{day},1] \/
                        schedule[{day}-1,1] == schedule[{day},2] \/
                        schedule[{day}-1,2] == schedule[{day},1] \/
                        schedule[{day}-1,2] == schedule[{day},2]''').format(day=day+1)
                features.append('2 * ({temp}) - 1'.format(**locals()))
                temp = dedent('''\
                        schedule[{day}-1,3] == schedule[{day},3] \/
                        schedule[{day}-1,3] == schedule[{day},4] \/
                        schedule[{day}-1,4] == schedule[{day},3] \/
                        schedule[{day}-1,4] == schedule[{day},4]''').format(day=day+1)
                features.append('2 * ({temp}) - 1'.format(**locals()))
            else:
                features.extend([0] * 2)

        expected_n_features = 2 * n_days * n_body_parts + 2 * (n_days - 1)
        assert len(features) == expected_n_features

        problem.parameter('FEATURES', set(range(1, expected_n_features + 1)))
        problem.array_variable('phi', 'FEATURES', 'int', features)
        return problem

    def _fix_other_parts(self, problem, part, x):
        schedule = x['schedule']
        for day in range(self.data['N_DAYS']):
            if day != part:
                for slot in range(self.data['N_SLOTS']):
                    problem.constraint('schedule[{day},{slot}] = {value}'
                        .format(day=day+1, slot=slot+1, value=schedule[day][slot]))
        return problem

    def initial_configuration(self):
        return minizinc(self._part_phi(MiniZincModel(_phi(PROBLEM)), None),
                        data=self.data,
                        solver=self.solver, force_flatten=True)[0]

    def phi(self, x):
        phi = minizinc(self._part_phi(MiniZincModel(_phi(PROBLEM)), None),
                       data={**self.data, **x},
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

    def infer(self, w, x=None, part=None, local=False, timeout=None,
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
        return self.multizinc(problem, data={**self.data, 'w': w_int},
                              solver=solver, timeout=timeout,
                              force_flatten=True)

    def improve(self, w, x, part=None, alpha=0.1, timeout=None,
                improve_margin=None, solver=gecode):
        solver = solver or self.solver
        w_int = (w * 1000).astype(int)
        timeout = 1 # XXX HACK XXX
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
            return self.multizinc(problem, data={**self.data, 'w': w_int},
                                  solver=solver, timeout=timeout,
                                  force_flatten=True)
        except MiniZincUnsatisfiableError:
            return x

    def local_update(self, w, x, xbar, part, eta=1):
        mask = np.zeros_like(w)
        n_body_parts, n_parts = self.data['N_BODY_PARTS'], len(self.parts)
        mask[n_body_parts * part : n_body_parts * (part + 1)] = 1
        mask[n_body_parts * (n_parts + part) : n_body_parts * (n_parts + part + 1)] = 1
        mask[n_body_parts * n_parts * 2 + part * 2 : n_body_parts * n_parts * 2 + (part + 1) * 2] = 1
        return w + mask * eta * (self.phi(xbar) - self.phi(x))
