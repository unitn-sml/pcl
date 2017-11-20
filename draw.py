#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import parts


def pad(array, length):
    assert array.ndim == 1
    full = np.zeros(length, dtype=array.dtype)
    full[:len(array)] = array
    return full


def average(array):
    avg = []
    for j in range(1, len(array) + 1):
        avg.append(sum(array[:j]) / j)
    return np.array(avg)

max_u_star = {}

def load(path, draw_args):
    data = parts.load(path)
    exp_args, traces = data['args'], data['traces']

    #assert draw_args.x_stars != None
    assert draw_args.users != None

    pcl = 'PCL' in path
    users = parts.load(draw_args.users)['users']
    if draw_args.x_stars != None:
        x_stars = parts.load(draw_args.x_stars)['x_stars']
    else:
        x_stars = [None] * len(users)

    max_iters = exp_args.max_iters
    if draw_args.max_iters is not None:
        max_iters = min(max_iters, draw_args.max_iters)

    global max_u_star

    creg_matrix, reg_matrix, time_matrix = [], [], []
    avg_creg_matrix, avg_reg_matrix = [], []
    improve_matrix = []
    for i, trace in enumerate(traces[:draw_args.num_users]):
        trace = np.array(trace)
        user = users[i]
        x_star = x_stars[i]
        if x_star is not None:
            u_star = user.utility(x_star)
        else:
            u_star = -np.inf
        if i in max_u_star:
            u_star = max([u_star, max_u_star[i]])
        else:
            max_u_star[i] = u_star
        if pcl:
            u_star = max(u_star, np.max(trace[:,1]), np.max(trace[:,2]), max_u_star[i])
            max_u_star[i] = u_star
            creg = pad(trace[:,0], max_iters)
            reg = pad(u_star - trace[:,1], max_iters)
            avg_creg = average(creg)
            avg_reg = average(reg)
            creg_matrix.append(creg)
            reg_matrix.append(reg)
            avg_creg_matrix.append(avg_creg)
            avg_reg_matrix.append(avg_reg)
            time_matrix.append(pad(trace[:,3], max_iters))
            improve_matrix.append(pad(trace[:,2] - trace[:,1], max_iters))
        else:
            u_star = max(u_star, np.max(trace[:,0]), np.max(trace[:,1]), max_u_star[i])
            max_u_star[i] = u_star
            reg = pad(u_star - trace[:,0], max_iters)
            avg_reg = average(reg)
            reg_matrix.append(reg)
            avg_reg_matrix.append(avg_reg)
            time_matrix.append(pad(trace[:,2], max_iters))
            improve_matrix.append(pad(trace[:,1] - trace[:,0], max_iters))

    return np.array(creg_matrix), np.array(reg_matrix), np.array(avg_creg_matrix), np.array(avg_reg_matrix), np.array(time_matrix), np.array(improve_matrix), {**vars(data['args']), 'path': path}


def get_style(exp_args, draw_args):
    alpha = str(exp_args['alpha'])
    if 'PCL' in exp_args['path']:
        ps = exp_args['part_selection']
        COLORS = {
            'smallest_first': '#DC322F',
            'largest_first': '#f428bc',
            'random': '#73d216',
            'ordered': '#268BD2',
            'ordered_reverse': '#b044a0',
            'round_robin': '#3465a4',
            'round_robin_reverse': '#376e65',
        }
        MARKERS = {
            '0.1': 'o',
            '0.3': '+',
            '0.5': '*',
        }
        return 'PCL ({}, {})'.format(ps, alpha), MARKERS[alpha], COLORS[ps], '-'
    else:
        marks = {1: 's', 2: 'D', 5: '^', 10: 'H', None: 'v'}
        colors = {1: '#DC322F', 2: '#3465a4', 5: '#73d216', 10: '#75507b', None: '#000000'}
        LINESTYLES = {'0.1': '-', '0.3': '--', '0.5': ':'}
        if draw_args.no_timeout:
            name = 'CL ({})'.format(alpha)
        else:
            name = 'CL (t={} alpha={})'.format(exp_args['timeout'], alpha)
        return name, marks[exp_args['timeout']], colors[exp_args['timeout']], LINESTYLES[alpha]


def plot_perfs(ax, xs, matrix, max_x, label, marker, color, linestyle, mean=True):
    if not mean:
        ys = np.median(matrix, axis=0)
    else:
        ys = np.mean(matrix, axis=0)
    yerrs = np.std(matrix, axis=0) / np.sqrt(matrix.shape[0])
    ys, yerrs = ys[:max_x], yerrs[:max_x]
    ax.plot(xs, ys, linewidth=2, label=label, color=color, linestyle=linestyle,
            marker=marker, markersize=6, markevery=4)
    ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color,
                    alpha=0.35, linewidth=0)
    return ys.max()


def draw(args):
    plt.style.use('ggplot')

    data = []
    for path in args.pickles:
        data.append(load(path, args))

    creg_fig, creg_ax = plt.subplots(1, 1)
    reg_fig, reg_ax = plt.subplots(1, 1)
    avg_creg_fig, avg_creg_ax = plt.subplots(1, 1)
    avg_reg_fig, avg_reg_ax = plt.subplots(1, 1)
    time_fig, time_ax = plt.subplots(1, 1)
    improve_fig, improve_ax = plt.subplots(1, 1)

    mean = not args.reg_median

    max_cregret, max_regret, max_time, max_iters = -np.inf, -np.inf, -np.inf, -np.inf
    max_avg_cregret, max_avg_regret = -np.inf, -np.inf
    max_improve = -np.inf
    for creg_matrix, reg_matrix, avg_creg_matrix, avg_reg_matrix, time_matrix, improve_matrix, info in data:
        label, marker, color, linestyle = get_style(info, args)

        pcl = 'PCL' in info['path']

        max_iters = args.max_iters or max(max_iters, info['max_iters'])
        xs = np.arange(1, (args.max_iters or info['max_iters']) + 1)

        if pcl:
            cur_max_cregret = plot_perfs(creg_ax, xs, creg_matrix, max_iters,
                                         label, marker, color, linestyle,
                                         mean=mean)
            max_cregret = args.max_regret or max(max_cregret, cur_max_cregret)

            cur_max_avg_cregret = plot_perfs(avg_creg_ax, xs, avg_creg_matrix,
                                             max_iters, label, marker, color,
                                             linestyle, mean=mean)
            max_avg_cregret = args.max_regret or max(max_avg_cregret, cur_max_cregret)

        cur_max_regret = plot_perfs(reg_ax, xs, reg_matrix, max_iters,
                                    label, marker, color, linestyle, mean=mean)
        max_regret = args.max_regret or max(max_regret, cur_max_regret)

        cur_max_avg_regret = plot_perfs(avg_reg_ax, xs, avg_reg_matrix, max_iters,
                                        label, marker, color, linestyle, mean=mean)
        max_avg_regret = args.max_regret or max(max_avg_regret, cur_max_avg_regret)

        cumtime_matrix = time_matrix.cumsum(axis=1)
        cur_max_time = plot_perfs(time_ax, xs, cumtime_matrix, max_iters,
                                  label, marker, color, linestyle, mean=True)
        max_time = args.max_time or max(max_time, cur_max_time)

        cur_max_improve = plot_perfs(improve_ax, xs, improve_matrix, max_iters,
                                  label, marker, color, linestyle, mean=True)
        max_improve = max(max_improve, cur_max_improve)

    def prettify(ax, max_iters):
        xtick = 5 if max_iters <= 50 else 10
        xticks = np.hstack([[1], np.arange(xtick, max_iters + 1, xtick)])
        reg_ax.set_xticks(xticks)

        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(18)
        ax.grid(True)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_linestyle('-.')

    #reg_ax.set_xlabel('# queries')
    creg_ax.set_ylabel('Conditional Regret')
    creg_ax.set_xlim([1, max_iters])
    creg_ax.set_ylim([0, 1.05 * max_cregret])
    prettify(creg_ax, max_iters)

    #reg_ax.set_xlabel('# queries')
    avg_creg_ax.set_ylabel('Average Conditional Regret')
    avg_creg_ax.set_xlim([1, max_iters])
    avg_creg_ax.set_ylim([0, 1.05 * max_avg_cregret])
    prettify(avg_creg_ax, max_iters)

    #reg_ax.set_xlabel('# queries')
    reg_ax.set_ylabel('Regret')
    reg_ax.set_xlim([1, max_iters])
    reg_ax.set_ylim([0, 1.05 * max_regret])
    prettify(reg_ax, max_iters)

    #reg_ax.set_xlabel('# queries')
    avg_reg_ax.set_ylabel('Average Regret')
    avg_reg_ax.set_xlim([1, max_iters])
    avg_reg_ax.set_ylim([0, 1.05 * max_avg_regret])
    prettify(avg_reg_ax, max_iters)

    #time_ax.set_xlabel('# queries')
    time_ax.set_ylabel('Cumulative time (s)')
    time_ax.set_xlim([1, max_iters])
    time_ax.set_ylim([0, 1.05 * max_time])
    prettify(time_ax, max_iters)

    improve_ax.set_ylabel('Improvement')
    improve_ax.set_xlim([1, max_iters])
    improve_ax.set_ylim([0, 1.05 * max_improve])
    prettify(improve_ax, max_iters)

    creg_ax.set_title(args.title, fontsize=18)
    legend = creg_ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    creg_fig.savefig(args.png_basename + '_creg.png',
                     bbox_inches='tight', pad_inches=0, dpi=120)

    avg_creg_ax.set_title(args.title, fontsize=18)
    legend = avg_creg_ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    avg_creg_fig.savefig(args.png_basename + '_avg_creg.png',
                     bbox_inches='tight', pad_inches=0, dpi=120)

    reg_ax.set_title(args.title, fontsize=18)
    legend = reg_ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    reg_fig.savefig(args.png_basename + '_reg.png',
                     bbox_inches='tight', pad_inches=0, dpi=120)

    avg_reg_ax.set_title(args.title, fontsize=18)
    legend = avg_reg_ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    avg_reg_fig.savefig(args.png_basename + '_avg_reg.png',
                     bbox_inches='tight', pad_inches=0, dpi=120)

    time_ax.set_title(args.title, fontsize=18)
    legend = time_ax.legend(loc='upper left', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    time_fig.savefig(args.png_basename + '_time.png', bbox_inches='tight',
                     pad_inches=0, dpi=120)

    improve_ax.set_title(args.title, fontsize=18)
    legend = time_ax.legend(loc='upper left', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    improve_fig.savefig(args.png_basename + '_improve.png', bbox_inches='tight',
                        pad_inches=0, dpi=120)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png_basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    parser.add_argument('-U', '--users', type=str,
                        help='the file containing the the users')
    parser.add_argument('-n', '--num-users', type=int, default=20,
                        help='number of users to use')
    parser.add_argument('-X', '--x-stars', type=str,
                        help='the file containing the x stars for the users')
    parser.add_argument('-T', '--title', type=str, default='Title',
                        help='plot title')
    parser.add_argument('--no-timeout', action='store_true',
                        help='whether not to print the timeout in the legend')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='max iters')
    parser.add_argument('--max-regret', type=int, default=None,
                        help='max regret')
    parser.add_argument('--max-time', type=int, default=None,
                        help='max time')
    parser.add_argument('-m', '--reg-median', action='store_true',
                        help=('whether to use the meadian instead of mean when'
                              'averaging iterations over users'))
    args = parser.parse_args()

    draw(args)
