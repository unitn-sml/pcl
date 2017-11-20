
import numpy as np
from time import time
from sklearn.utils import check_random_state
from textwrap import dedent


def pp(problem, user, max_iters=100, verbose=False, rng=None, timeout=None,
       improve_margins=None, use_mean_margin=False):
    rng = check_random_state(rng)

    print(dedent('''\
            eliciting user:
            w_star = {}
        ''').format(user.w_star))

    w = np.zeros(problem.n_features)

    if use_mean_margin and improve_margins:
        mean_margin = sum(improve_margins) / len(improve_margins)

    trace = []
    for t in range(max_iters):
        itertime = 0

        t0 = time()
        x = problem.infer(w, timeout=timeout)
        itertime += time() - t0

        regret = user.regret(x)
        util_x = user.utility(x)

        improve_margin = None
        if improve_margins is not None:
            if use_mean_margin:
                improve_margin = mean_margin
            elif len(improve_margins) > t:
                improve_margin = improve_margins[t]
            else:
                improve_margin = 0.0

        xbar = user.improve(x, timeout=timeout, improve_margin=improve_margin)
        #xbar = user.improve(x, timeout=timeout)
        util_xbar = user.utility(xbar)

        t0 = time()
        w = w + problem.phi(xbar) - problem.phi(x)
        itertime += time() - t0

        trace.append((util_x, util_xbar, itertime))

        if verbose:
            phi      = problem.phi(x)
            phibar   = problem.phi(xbar) if xbar is not None else None
            avg_util = sum(np.array(trace)[:,0]) / len(trace)
            util_diff = util_xbar - util_x
            print(dedent('''\
                    iteration {t}
                    itertime = {itertime}
                    util_x   = {util_x}
                    u_xbar - u_x = {util_diff}
                    avg_util = {avg_util}
                ''').format(**locals()))

    print('done after {} iterations'.format(t))
    return trace
