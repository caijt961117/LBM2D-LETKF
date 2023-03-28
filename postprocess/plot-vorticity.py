#! /usr/bin/env python3

import numpy as np; print('numpy:', np.__version__)
import matplotlib; print('matplotlib:', matplotlib.__version__)
from matplotlib import pyplot as plt

def reset_plt():
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = plt.rcParams['axes.linewidth']
    plt.rcParams['ytick.major.width'] = plt.rcParams['axes.linewidth']
    plt.rcParams['figure.figsize'] = (5.8, 4.1)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12

def plot_all(prefix, v='vor', *, nprocs=1, cmap='coolwarm', is_colorbar=False, dtype=np.float32):

    # god
    #from pyconf import Re, nx, nout
    # nx = None ## auto detect from file
    nout = 20000
    nskip = 10
    resol0, resol1 = 4, 256
    clim = 5

    # cfd
    def ifilename(t, v):
        return f'{prefix}/{v}_step{t:010d}.dat'
    def ofilename(t, v, suffix):
        return f'{prefix}/{v}_step{t:010d}.{suffix}'

    print('start plots')

    def plot(v, cmap, *, clip=False):
        print(v)
        def plot_t(t):
            reset_plt()
            try:
                with open(ifilename(t, v)) as f:
                    nda = np.fromfile(f, dtype=dtype, sep='')
                    nx = int(np.sqrt(len(nda))+0.5)
                    nda = nda.reshape(nx, nx)
                if clip:
                    mx = np.max(nda)
                    mn = np.min(nda)
                    nda = np.clip(nda, mn/2, mx/2)
                (fig, ax) = plt.subplots(dpi=resol1)
                pc = ax.imshow(nda, cmap=cmap, clim=(-clim, clim))
                if is_colorbar:
                    plt.colorbar(pc, label='vorticity', ticks=[-clim, 0, clim])
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                #ax.set_title('%s, %s, Re=%d' % (prefix.split('/')[1], v, Re))
                fig.savefig(ofilename(t, v, suffix='png'), bbox_inches='tight')
                fig.savefig(ofilename(t, v, suffix='pdf'), bbox_inches='tight')
                plt.close(fig)
                print(end='.', flush=True)
            except:
                print(end='x', flush=True)
                pass
            
        import joblib
        joblib.Parallel(n_jobs=nprocs, verbose=1)(joblib.delayed(plot_t)(t) for t in range(0, nout, nskip))

    plot(v, cmap, clip=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='helper script that plots 2D vorticity map')
    parser.add_argument('dirs', nargs='*', help='dir1 dir2 ...')
    parser.add_argument('-c', '--colorbar', action='store_true', help='show colorbar on the plot')
    parser.add_argument('-n', '--name', help='set variable name.', type=str, default='vor')
    parser.add_argument('-j', '--jobs', help='number of threads for joblib.Parallel()', type=int, default=1)
    args = parser.parse_args()
    for d in args.dirs:
        print(d)
        plot_all(prefix=d, v=args.name, nprocs=args.jobs, is_colorbar=args.colorbar, )
        
if __name__ == '__main__':
    main()
