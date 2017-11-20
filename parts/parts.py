import numpy as np
from time import time
from sklearn.utils import check_random_state
from textwrap import dedent

import pymzn

rr = 0
selected = {}
def select_part(improvable, part_selection, rng, size=None):
    global rr
    if part_selection == 'smallest_first':
        if not size:
            raise ValueError()
        part_sizes = []
        for part in improvable:
            if part not in selected:
                selected[part] = 0
            part_size = (selected[part] + 1) * (0.1 * size[part] + 1)
            part_sizes.append(part_size)
        part = np.argmin(part_sizes)
        part = improvable[part]
        selected[part] += 1
    elif part_selection == 'full':
        part = None
    elif part_selection == 'ordered':
        part = improvable[0]
    elif part_selection == 'ordered_reverse':
        part = improvable[-1]
    elif part_selection == 'round_robin':
        if rr == improvable[-1]:
            rr = 0
        else:
            rr += 1
        while rr not in improvable:
            print(rr)
            if rr == improvable[-1]:
                rr = 0
            else:
                rr += 1
        part = rr
    elif part_selection == 'round_robin_reverse':
        if rr == -improvable[-1]:
            rr = 0
        else:
            rr -= 1
        while rr not in improvable:
            if rr == -improvable[-1]:
                rr = 0
            else:
                rr -= 1
        part = rr
    elif part_selection == 'random':
        part = rng.choice(improvable)
    else:
        raise NotImplementedError()
    return part


def proj(w, radius=0.0):
    if radius <= 0.0:
        return w
    import cvxpy as cvx
    u = cvx.Variable(w.shape[0])
    dist = cvx.norm(u - w, 2)
    u_norm = cvx.norm(u, 2)
    problem = cvx.Problem(cvx.Minimize(dist), [u_norm <= radius])
    problem.solve()
    return np.array(u.value).reshape(w.shape)


def pcl(problem, user, max_iters=100, eta='const', radius=0.0, local_infer=True,
        part_selection='round_robin', verbose=False, rng=None, local_update=True):
    rng = check_random_state(rng)

    if eta == 'invsqrt':
        eta = lambda t: 1 / np.sqrt(t + 1)
    elif eta == 'invlin':
        eta = lambda t: 1 / (t + 1)
    elif eta == 'const':
        eta = lambda t: 1

    print(dedent('''\
            eliciting user:
            w_star = {}
        ''').format(user.w_star))

    w = np.zeros(problem.n_features)
    x = problem.initial_configuration()
    improvable = list(problem.parts)

    global rr
    global selected
    rr = 0
    selected = {}

    trace = []
    for t in range(max_iters):
        itertime = 0

        if len(improvable) == 0:
            print('no improvable part')
            return trace

        # Part selection
        t0 = time()
        part = select_part(improvable, part_selection, rng, size=problem.size)
        x = problem.infer(w, x=x, part=part, local=local_infer, solver=pymzn.gecode)
        itertime += time() - t0

        cregret = user.cregret(x, part, solver=pymzn.gecode)
        util_x = user.utility(x)

        # Improvement
        xbar = user.improve(x, part=part, solver=pymzn.gecode)
        util_xbar = user.utility(xbar)
        if cregret == 0 and part == improvable[0]:
            improvable.remove(part)
        else:
            improvable = list(problem.parts)

        # Weight update
        t0 = time()
        if cregret > 0:
            delta = problem.phi(xbar) - problem.phi(x)
            if not local_update or w.dot(delta) <= 0:
                w = proj(w + eta(t) * delta, radius)
            else:
                w = proj(problem.local_update(w, x, xbar, part, eta(t)), radius)
        itertime += time() - t0

        trace.append((cregret, util_x, util_xbar, itertime))

        if verbose:
            phi    = problem.phi(x)
            phibar = problem.phi(xbar) if xbar is not None else None
            #avgreg  = None if not regret else sum([tr[0] for t in trace]) / (tr+1)
            avgcreg = sum([tr[0] for tr in trace]) / (t+1)
            print(dedent('''\
                    iteration {t}
                    itertime   = {itertime}
                    improvable = {improvable}
                    part    = {part}
                    creg    = {cregret}
                    avgcreg = {avgcreg}
                ''').format(**locals()))

    print('done after {} iterations'.format(t))
    return trace
