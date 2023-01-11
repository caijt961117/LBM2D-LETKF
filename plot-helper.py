#! /usr/bin/env python3

import numpy as np; print('numpy:', np.__version__)
import matplotlib; matplotlib.use('Agg'); print('matplotlib:', matplotlib.__version__)
from matplotlib import pyplot as plt


def plot_all(prefix='io/nature/0', v='vor', cmap='coolwarm', dtype=np.float32):

    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = plt.rcParams['axes.linewidth']
    plt.rcParams['ytick.major.width'] = plt.rcParams['axes.linewidth']
    #plt.rcParams['font.size'] = 12

    # god
    #from pyconf import Re, ny, nx, nout
    ny = 128
    nx = 128
    nout = 2000
    nprocs = -1
    nskip = 10
    plt.rcParams['figure.figsize'] = [3, 3]

    # cfd
    def ifilename(t, v):
        return prefix + '/' + v + '_' + ('%d'%t) + '.dat'
    def ofilename(t, v):
        return prefix + '/' + v + '_' + ('%04d'%t) + '.png'

    print('start plots')

    def plot(v, cmap, *, clip=False):
        print(v)
        def plot_t(t):
            with open(ifilename(t, v)) as f:
                nda = np.fromfile(f, dtype=dtype, sep='').reshape(ny, nx)
            if clip:
                mx = np.max(nda)
                mn = np.min(nda)
                nda = np.clip(nda, mn/2, mx/2)
            (fig, ax) = plt.subplots(dpi=300)
            pc = ax.imshow(nda, cmap=cmap, clim=(-20, 20))
            plt.colorbar(pc)
            ax.invert_yaxis()
            #ax.set_title('%s, %s, Re=%d' % (prefix.split('/')[1], v, Re))
            plt.savefig(ofilename(t, v))
            plt.close(fig)
            print(end='.', flush=True)
            
        import joblib
        joblib.Parallel(n_jobs=nprocs, verbose=1)(joblib.delayed(plot_t)(t) for t in range(0, nout, nskip))

    plot(v, cmap, clip=False)
        
if __name__ == '__main__':
    plot_all(prefix='io/nature/0')
    plot_all(prefix='io/calc/0') ## nudging
    plot_all(prefix='io/observed/0')
    plot_all(prefix='io/calc', v='ens_vor_mean')
    for k in range(16):
        plot_all(prefix=f'io/calc/{k}') ## LETKF
