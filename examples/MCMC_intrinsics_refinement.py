from mcmc import *

from scipy.stats import invwishart as inw
from scipy.stats import multivariate_normal as mvn
from scipy.special import gamma as gfun
from itertools import product
import matplotlib.pyplot as plt

import sys
import csv
import pdb
import pickle
import yaml
import glob2
import os


dir_name = '/home/mwmaster/catch/vi_bags'
files = glob2.glob(os.path.join(dir_name, '*-camchain.yaml'))
data = []
for file in files:
    with open(file, 'r') as f:
        tmp = yaml.safe_load(f)
        data.append(tmp['cam0']['intrinsics'])
data = np.array(data).T

res = (1920, 1200)
N = data.shape[-1]
MCMC_ITER = 100000
REPS = 6
BURN = 0.5
THIN = 100

scale = 3000
Sigma0 = np.eye(data.shape[0])
nu = data.shape[0]
Lambda = scale*np.eye(data.shape[0])

if __name__ == '__main__':

    fig, ax = plt.subplots(data.shape[0],1,sharex=True)
    colors = ['r', 'g', 'b', 'm', 'y', 'c']
    for rep in range(REPS):
        mu0 = mvn.rvs(mean=[4000, 4000, 960, 600], cov=scale*np.eye(data.shape[0]))

        def getMuMu(samples, data, mu0=mu0, Sigma0=Sigma0, n=N):
                val = data.sum(axis=1)
                val = val @ np.linalg.inv(np.array(samples['Sigma'][-1]))
                val += mu0 @ np.linalg.inv(Sigma0)
                tmp = np.linalg.inv(np.array(getMuSigma(samples)))
                val = tmp@val
                return val


        def getMuSigma(samples, Sigma0=Sigma0, n=N):
            val = np.linalg.inv(np.array(samples['Sigma'][-1]))
            # print(val)
            val *= n
            val += np.linalg.inv(Sigma0)
            val = np.linalg.inv(val)
            return val


        def getSigmaNu(nu=nu, n=data.shape[0]):
            return nu+n


        def getSigmaLambda(samples, data, Lambda=Lambda):
            tmp = np.array(samples['mu'][-1])
            tmp = np.expand_dims(tmp, axis=-1)
            val = data - tmp
            val = val@val.T
            val += Lambda
            return val


        def muProp(data=data, hack=None):
            return lambda samples: mvn.rvs(mean=getMuMu(samples, data),
                                            cov=getMuSigma(samples))


        def sigmaProp(data=data, hack=None):
            return lambda samples: inw.rvs(getSigmaNu(), scale=getSigmaLambda(samples, data))


        params = {'mu': ['gibbs', muProp],
                  'Sigma': ['gibbs', sigmaProp]}

        samples = {'mu': [mvn.rvs(np.zeros(data.shape[0]), cov=np.eye(data.shape[0])).tolist()],
                   'Sigma': [inw.rvs(nu, scale=np.eye(data.shape[0])).tolist()]}

        keys = samples.keys()
        mcmc = dispatcherMCMC(MCMC_ITER, keys, params)

        print(f'repetition {rep}, MCMC iteration 0')
        for key in keys:
            print(f'{key}, \n {samples[key][-1]} \n')

        for i in range(1, MCMC_ITER):
            print(f'repetition {rep}, MCMC iteration {i}')
            mcmc.fullStep(samples)

        a = samples['mu'][-int(BURN*MCMC_ITER)::THIN]
        b = samples['Sigma'][-int(BURN*MCMC_ITER)::THIN]

        print('MCMC repetition done. Plotting... ')
        for i,elmt in enumerate(a):
            for j in range(data.shape[0]):
                ax[j].scatter(i,elmt[j], color=colors[rep], s=3, alpha=0.3)

        print('Plotting done. Dumping pkl...')
        with open(f'2_{rep}__2024_12_10.pkl', 'wb') as f:
            tmp = {'mu':a, 'Sigma':b, 'mu0':mu0}
            pickle.dump(tmp, f)

    plt.show()
