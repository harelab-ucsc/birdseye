from cf_triad import *
from dbCreate import *
from dbConnector import dbConnector
from generateData import *
from simRotTools import *
from simMain import *
from fieldAI import *
from mm00ggMCMC import *

from scipy.stats import gamma, norm, expon, uniform
from scipy.stats import multivariate_normal as mvn
from scipy.special import gamma as gfun
from itertools import product
import matplotlib.pyplot as plt

import sys
import csv
import pdb

plot_camera = False

save_gif3d = '/home/mwmaster/parsed_flight/gif3d'
save_gifpx = '/home/mwmaster/parsed_flight/gifpx'
dir_name = '/home/mwmaster/parsed_flight'
# clicks_csv = '/home/mwmaster/Documents/dataset_pipeline/data/Flight_clicks.csv'
clicks_csv = '/home/mwmaster/Documents/dataset_pipeline/data/Flight_clicks2.csv'
# db_name = 'flight_data'
db_name = 'flight_data2'

res = (3264, 2464)
K = np.array([[3755.2620514049586,   0.     , 1543.7439125388514, 0],
           [0.     , 3754.5358046182687, 1313.898910616996, 0],
           [0.     ,   0.     ,   1.     , 0]])
RW_SCALE = 2**(-3)
MCMC_ITER = 100000
REPS = 5
BURN = 0.5
THIN = 100

sample_log = lambda x: np.log(x)
sample_unlog = lambda x: np.exp(x)

if __name__ == '__main__':
    with open(dir_name + '/test_flight_results.csv', "r", newline='') as f:
        clickreader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        pred = []
        true = []
        for row in clickreader:
            # print(row)
            pred.append([row[1], row[2]])
            true.append([row[3], row[4]])
    pred = np.array(pred)
    true = np.array(true)
    err = np.linalg.norm(pred-true, axis=1)
    n = len(err)

    # err.astype(np.float128)
    fig, ax = plt.subplots(1,1)
    _, bins, _ = ax.hist(err, bins=100, density=True)
    ax.set_ylim(0,25)


    def getBetaAlpha(samples, n=n):
            val = sample_unlog(samples['alpha'][-1])
            # print(val)
            val *= n
            val += 0.1
            return val


    def getBetaBeta(data):
        return data.sum() + 1


    def betaProp(n=n, hack=None):
        return lambda samples: gamma.rvs(getBetaAlpha(samples),
                                        scale=1/getBetaBeta(err))

    def alphaProp(scale=RW_SCALE, hack=None):
        return lambda samples: norm.rvs(loc=samples['alpha'][-1], scale=scale)


    def getAlphaTargetParams(hack=None):
        return lambda samples: [samples['beta'][-1], err]


    def alphaTarget(x, beta, data, transform=sample_log, inv=sample_unlog):
        if inv is not None:
            x = inv(x)
        val = x * np.sum(np.log(data))
        val += np.log(beta ** x / gfun(x)) * n
        val -= x
        val = np.exp(val)
        if inv is not None:
            val *= x
        return val


    kwargs = {'transform':sample_log, 'inv': sample_unlog}
    fig2, ax2 = plt.subplots(2,1)
    fig3, ax3 = plt.subplots(2,1)
    for rep in range(REPS):
        params = {'alpha': ['rwMetropolis', alphaProp,
                                            alphaTarget,
                                            getAlphaTargetParams,
                                            kwargs],
                  'beta': ['gibbs', betaProp]}

        samples = {'alpha': [expon.rvs().tolist()],
                   'beta': [gamma.rvs(0.1, scale=1).tolist()]}

        keys = samples.keys()
        mcmc = dispatcherMCMC(MCMC_ITER, keys, params)

        print(f'repetition {rep}, MCMC iteration 0')
        for key in keys:
            print(f'{key}, {samples[key][-1]}')

        for i in range(1, MCMC_ITER):
            print(f'repetition {rep}, MCMC iteration {i}')
            mcmc.fullStep(samples)
        a = samples['alpha'][-int(BURN*MCMC_ITER)::THIN]
        b = samples['beta'][-int(BURN*MCMC_ITER)::THIN]
        ax2[0].hist(sample_unlog(a), density=True, bins=50, alpha=0.5, label=f'rep{rep}')
        ax2[1].hist(b, density=True, bins=50, alpha=0.5)
        # ax2[2].acorr(a, maxlags=30, alpha=0.5)
        # ax2[2].acorr(b, maxlags=30, alpha=0.5)
        # ax2[2].set_xlim(-0.5, 30.5)

        ax3[0].plot(a, alpha=0.6)
        ax3[1].plot(b, alpha=0.6)
        params = None
        samples = None

    am = np.array(sample_unlog(a)).mean()
    bm = np.array(b).mean()
    print('\n\n')
    print(f'fit alpha: {am:.4f}, fit beta: {bm:.4f}')
    print(f'fit mean: {am/bm:.4f}, fit var: {am/bm**2:.4f}')
    print(f'data mean: {err.mean():.4f}, data var: {err.var():.4f}')

    ax2[0].legend()
    ax2[0].set_title('$\\alpha$')
    ax2[1].set_title('$\\beta$')
    fig2.subplots_adjust(hspace=0.4)
    ax3[0].set_title('$\\alpha$')
    ax3[1].set_title('$\\beta$')
    fig3.tight_layout()

    print('\n')
    print(f'median: {np.quantile(err, 0.5):.4f}, 95%CI: ({np.quantile(err, 0):.4f}, {np.quantile(err, 0.95):.4f})')
    print('\n')
    print(f'data points: {len(err)}')

    ax.plot(bins, gamma.pdf(bins, am, scale=1/bm))
    fig.savefig('data_hist_mcmc_fit.png')
    fig2.savefig('mcmc_ab_hists.png')
    fig3.savefig('mcmc_ab_trace.png')
    plt.show()
