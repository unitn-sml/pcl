#!/usr/bin/env python3

import parts
import numpy as np
import pymzn
from textwrap import dedent
#pymzn.debug()

PROBLEMS = {
    'synthetic': parts.SyntheticProblem,
    'hotels': parts.HotelsProblem,
    'sport': parts.SportProblem,
}

HOTELS = {1: parts.hotels.Hotel.default(), 2: parts.hotels.Hotel.default2()}

def opt(args):
    if args.problem == 'hotels':
        problem = PROBLEMS[args.problem](hotel=HOTELS[args.hotel])
    else:
        problem = PROBLEMS[args.problem]()

    users = parts.load(args.user_file)['users']
    for user in users:
        user.problem = problem

    x_stars = []
    for i, user in enumerate(users):
        print('Calculating x star for user {}'.format(i + 1))
        w_star = user.w_star
        x_star = problem.infer(w_star, timeout=args.timeout, solver=pymzn.gecode)
        u_star = np.dot(w_star, problem.phi(x_star))
        print(dedent('''
                w_star = {w_star}
                x_star = {x_star}
                approx. u(x_star) = {u_star}
            ''').format(**locals()))
        x_stars.append(x_star)
    parts.dump(args.output_file, {'x_stars': x_stars})
    print('Done!')


def main():
    import argparse

    cls = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=cls)
    parser.add_argument('problem', choices=PROBLEMS.keys(),
                        help='the problem to run')
    parser.add_argument('-H', '--hotel', type=int, default=1,
                        help='1: big, 2: small')
    parser.add_argument('-U', '--user_file', type=str, default='users',
                       help='the input file name containing the generated users')
    parser.add_argument('-O', '--output_file', type=str, default='output',
                       help='the output file name containing the optimal configurations')
    parser.add_argument('-t', '--timeout', type=int, default=None,
                       help='timeout to use for inference')

    args = parser.parse_args()
    opt(args)

if __name__ == '__main__':
    main()
