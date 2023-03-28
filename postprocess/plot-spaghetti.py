#! /usr/bin/env python3

import numpy as np; print('numpy:', np.__version__)
import matplotlib; matplotlib.use('Agg'); print('matplotlib:', matplotlib.__version__)
from matplotlib import pyplot as plt
import argparse

def plot_spaghetti_all(prefix, n, alpha, levels, nprocs, is_colorbar, linewidth, nskip=10, nout=2000, nx=256, v='vor', cmap='coolwarm', dtype=np.float32):
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = plt.rcParams['axes.linewidth']
    plt.rcParams['ytick.major.width'] = plt.rcParams['axes.linewidth']
    #plt.rcParams['font.size'] = 12

    plt.rcParams['figure.figsize'] = [3, 3]

    # cfd
    def ifilename(t, v, prefix):
        return f'{prefix}/{v}_step{t:010}.dat'
    def ofilename(t, v, prefix, suffix='png'):
        return f'{prefix}/{v}_spaghetti_{t:04}.{suffix}'

    print('start plots:', prefix, v, flush=True)

    def plot_t(t):
        (fig, ax) = plt.subplots()
        prefixc = lambda k: f'{prefix}/ens{k:04}'
        for k in range(n):
            with open(ifilename(t, 'vor', prefix=prefixc(k))) as f:
                vor = np.fromfile(f, dtype=dtype, sep='').reshape(nx, nx)
                c = ax.contour(vor, levels, cmap=cmap, linewidths=linewidth, alpha=alpha)
                if is_colorbar:
                    fig.colorbar(c, label='vorticity')
                print('.', end='', flush=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        fig.savefig(ofilename(t,v,prefix), bbox_inches='tight', dpi=144)
        fig.savefig(ofilename(t,v,prefix, suffix='pdf'), bbox_inches='tight')
        plt.close(fig)
            
    if nprocs != 1:
        import joblib
        joblib.Parallel(n_jobs=nprocs, verbose=1)(joblib.delayed(plot_t)(t) for t in range(nout+1)[::nskip])
    else:
        for t in range(nout+1)[::nskip]:
            plot_t(t)

def main():
    parser = argparse.ArgumentParser(description='plot ensemble spaghetti of contour lines of vorticity')
    parser.add_argument('dirs', nargs='*', help='dir1 dir2 ...')
    parser.add_argument('-n', help='ensemble size. default=64', type=int, default=64)
    parser.add_argument('-a', '--alpha', help='alpha brending rate. default=0.1', type=float, default=0.1)
    parser.add_argument('-l', '--levels', help='contour levels. default=[-5, 5]', type=float, nargs='*', default=[-5, 5])
    parser.add_argument('-j', '--jobs', help='number of threads for joblib.Parallel()', type=int, default=1)
    parser.add_argument('-c', '--colorbar', help='show colorbar', action='store_true')
    parser.add_argument('-w', '--linewidth', help='line width', type=float, default=0.5)
    args = parser.parse_args()
    for d in args.dirs:
        print(d)
        plot_spaghetti_all(prefix=d, n=args.n, alpha=args.alpha, levels=args.levels, nprocs=args.jobs, is_colorbar=args.colorbar, linewidth=args.linewidth)

if __name__ == '__main__':
    main()
