#!/usr/bin/env python3

import os.path
import numpy as np
from sklearn.utils import check_random_state

import parts
import pymzn
#pymzn.debug()


PROBLEMS = {
    'synthetic': parts.SyntheticProblem,
    'hotels': parts.HotelsProblem,
    'sport': parts.SportProblem,
}

HOTELS = {1: parts.hotels.Hotel.default(), 2: parts.hotels.Hotel.default2()}

SOLVERS = {
    'gecode': pymzn.gecode,
}
try:
    SOLVERS['oscar'] = pymzn.oscar_cbls
except AttributeError:
    pass

def get_results_path(args):
    result_args = map(str, [
        args.problem, 'PP' if args.PP else 'PCL', args.n_users,
        args.max_iters, args.part_selection,
        args.distrib, args.sparsity, args.min_regret,
        args.seed, args.alpha, args.hotel,
        'from=', args.from_user,
        'to=', args.to_user,
    ])
    return os.path.join(args.results, '_'.join(result_args)) + ('_t{}'.format(args.timeout) if args.PP else '') + '.pickle'


def sample_users(problem, n_users, distrib='normal', sparsity=0, rng=None):
    rng = check_random_state(rng)

    DISTRIB = {
        'normal':   lambda shape: rng.normal(0, 1, size=shape),
        'uniform':  lambda shape: rng.uniform(-1, 1, size=shape),
    }
    ws_star = DISTRIB[distrib]((n_users, problem.n_features))

    n_zeros = int(round(problem.n_features * sparsity))
    for i in range(n_users):
        zeros = rng.permutation(problem.n_features)[:n_zeros]
        ws_star[i][zeros] = 0

    return ws_star


def store_users(args):
    rng = np.random.RandomState(args.seed)

    print('creating problem and users')

    if args.problem == 'hotels':
        problem = PROBLEMS[args.problem](hotel=HOTELS[args.hotel])
    else:
        problem = PROBLEMS[args.problem]()
    ws_star = sample_users(problem, args.n_users, distrib=args.distrib,
                           sparsity=args.sparsity, rng=rng)
    users = [parts.User(problem, w_star, args.alpha,
                        min_regret=args.min_regret,
                        calculate_true_regret=False)
                for w_star in ws_star]

    parts.dump(args.user_file, {'problem': problem, 'users': users})

    print('Done')


def run_experiment(args):
    if args.user_file is not None:
        print('loading problem and users')
        user_file = parts.load(args.user_file)
        if args.problem == 'hotels':
            problem = PROBLEMS[args.problem](hotel=HOTELS[args.hotel])
        else:
            problem = PROBLEMS[args.problem]()
        users = user_file['users']
        for user in users:
            user.problem = problem
            user.alpha = args.alpha
    else:
        rng = np.random.RandomState(args.seed)

        print('creating problem and user')

        if args.problem == 'hotels':
            problem = PROBLEMS[args.problem](hotel=HOTELS[args.hotel])
        else:
            problem = PROBLEMS[args.problem]()
        ws_star = sample_users(problem, args.n_users, distrib=args.distrib,
                               sparsity=args.sparsity, rng=rng)
        users = [parts.User(problem, w_star, args.alpha,
                            min_regret=args.min_regret,
                            calculate_true_regret=False)
                 for w_star in ws_star]

    print('running...')

    if args.result_file is not None:
        results = parts.load(args.result_file)
        traces = results['traces']
        improve_margins = [[util_xbar - util_x for _, util_x, util_xbar, _ in trace] for trace in traces]

    rng, traces = np.random.RandomState(args.seed), []
    for i, user in enumerate(users[args.from_user:args.to_user]):
        if args.PP:
            improve_margin = None if args.result_file is None else improve_margins[i]
            trace = parts.pp(problem, user, max_iters=args.max_iters,
                    verbose=args.verbose, rng=rng, timeout=args.timeout, improve_margins=improve_margin, use_mean_margin=args.mean_margin)
        else:
            trace = parts.pcl(problem, user, max_iters=args.max_iters,
                              eta=args.eta, radius=args.radius,
                              part_selection=args.part_selection,
                              verbose=args.verbose, rng=rng,
                              local_infer=(not args.no_local_infer),
                              local_update=(not args.no_local_update))

        traces.append(trace)

    parts.dump(get_results_path(args), {'args': args, 'traces': traces})


def main():
    import argparse
    import logging

    np.seterr(all='raise')
    np.set_printoptions(precision=2)

    cls = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=cls)
    parser.add_argument('problem', choices=PROBLEMS.keys(),
                        help='the problem to run')
    parser.add_argument('-H', '--hotel', type=int, default=1,
                        help='1: big, 2: small')
    parser.add_argument('-o', '--results', type=str, default='results',
                        help='path to directory of learning traces')
    parser.add_argument('-n', '--n-users', type=int, default=20,
                        help='number of users in the experiment')
    parser.add_argument('--from-user', type=int, default=0,
                        help='index of the first user to run')
    parser.add_argument('--to-user', type=int, default=0,
                        help='index of the last user to run (exclusive)')
    parser.add_argument('-r', '--seed', type=int, default=0,
                        help='RNG seed')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable debug spew')

    group = parser.add_argument_group('Learning')
    group.add_argument('--PP', action='store_true',
                       help='use the preference perceptron (without parts)')
    group.add_argument('-p', '--part-selection', default='smallest_first',
                       help='part selection strategy')
    group.add_argument('-t', '--max-iters', type=int, default=100,
                       help='maximum number of iterations')
    group.add_argument('--eta', default='const',
                       help='perceptron step size')
    group.add_argument('--radius', type=float, default=0.0,
                       help=('radius of the projection space. The value 0.0 means'
                             'no projection performed.'))
    group.add_argument('--no-local-infer', action='store_true',
                       help='wheter not to perform local inference')
    group.add_argument('--no-local-update', action='store_true',
                       help='wheter not to perform local update')
    group.add_argument('--solver', default='gecode',
                       help='the solver to use for inference')
    group.add_argument('--timeout', type=int, default=None,
                       help='the solver timeout')
    group.add_argument('-M', '--mean-margin', action='store_true',
                       help='wheter to use the mean improvement margin')

    group = parser.add_argument_group('User Simulation')
    group.add_argument('-d', '--distrib', type=str, default='normal',
                       help='distribution of the true user weights')
    group.add_argument('-s', '--sparsity', type=float, default=0,
                       help='percentage of zero true weights')
    group.add_argument('-a', '--alpha', type=str, default=0.1,
                       help='informativeness constants')
    group.add_argument('-S', '--store', action='store_true',
                       help='whether to generate and store a set of users')
    group.add_argument('-U', '--user_file', type=str, default='users',
                       help='the input file name containing the generated users')
    group.add_argument('--min-regret', type=float, default=0,
                       help='target percentage of true regret for satisfaction')
    group.add_argument('-R', '--result_file', type=str, default=None,
                       help='result file from CPL used in PP to match the improvement margin')

    args = parser.parse_args()

    if args.store:
        store_users(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
