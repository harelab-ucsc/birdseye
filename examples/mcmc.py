import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def distLambda(dist, args, kwargs=None):
    # if len(args) > 1:
        # print(f'        distLambda: args: {args[:-1]}')
    # else:
        # print(f'        distLambda: args: {args}')
    if kwargs is not None:
        return lambda z: dist(z, *args, **kwargs)
    else:
        return lambda z: dist(z, *args)


def indepMHAccept(z, prop, tgt=None, eval=None, args1=None, args2=None, kwargs=None):
    num = distLambda(tgt, args1)(prop)*distLambda(eval, args2)(z)
    denom = distLambda(tgt, args1)(z)*distLambda(eval, args2)(prop)
    # print(f'        indepMHAccept: v1: {num}  v2: {denom}')
    return prop, min(num/denom, 1)


def rwMHAccept(z, prop, tgt=None, args=None, kwargs=None):
    num = distLambda(tgt, args, kwargs)(prop)
    print(f'        rwMHAccept: prop, {prop}; num, {num}')
    denom = distLambda(tgt, args, kwargs)(z)
    print(f'        rwMHAccept: z, {z}; denom, {denom}')
    return prop, min(num/denom, 1)


def getHack(key):
    key = key[:-2]
    hack = ''
    while key[-1] != '_':
        key, tmp = key[:-1], key[-1]
        hack = tmp + hack
    return hack


class dispatcherMCMC:
    def __init__(self, iter, keys, params, indivs=[], darray=None):
        self.iter = iter
        self.keys = keys
        self.params = params

        self.indivs = indivs
        self.data = darray


    def gibbsStep(self, key, samples, hack):
        # indivData = None
        # if hack is not None:
            # indivData = deepcopy(self.data[:,:,int(hack)])
        # return self.params[key][1](hack=hack, indivData=indivData)(samples)
        return self.params[key][1](hack=hack)(samples)


    def indepmhStep(self, key, samples, hack):
        indivData = None
        if hack is not None:
            indivData = deepcopy(self.data[:,:,int(hack)])
        prop, rho = indepMHAccept(samples[key][-1],  # z
                                self.params[key][1](hack=hack, indivData=indivData)(samples),  # prop
                                tgt=self.params[key][2],  # tgtDist
                                eval=self.params[key][3],  # evalDist
                                args1=self.params[key][4](hack=hack, indivData=indivData)(samples),  # args1 == tgtDist params
                                args2=self.params[key][5](hack=hack, indivData=indivData)(samples),  # args2 == evalDist params
                                kwargs=self.params[key][6])  # kwargs == transform and inv
        # print(f'    indepmhStep: (prop, rho): ({prop}, {rho})')
        if np.random.uniform() < rho:
            return prop
        else:
            return samples[key][-1]


    def rwmhStep(self, key, samples, hack):
        if callable(self.params[key][3]):
            prop, rho = rwMHAccept(samples[key][-1],  # z
                                    self.params[key][1](hack=hack)(samples),  # propDist
                                    tgt=self.params[key][2],  # tgtDist
                                    args=self.params[key][3](hack=hack)(samples),  # args == tgtDist params
                                    kwargs=self.params[key][4])  # kwargs == transform and inv
        else:
            prop, rho = rwMHAccept(samples[key][-1],  # z
                                    self.params[key][1](hack=hack)(samples),  # propDist
                                    tgt=self.params[key][2],  # tgtDist
                                    args=self.params[key][3],  # args == tgtDist params
                                    kwargs=self.params[key][4])  # kwargs == transform and inv
        if np.random.uniform() < rho:
            return prop
        else:
            return samples[key][-1]


    def fullStep(self, samples):
        hack = None
        for key in self.keys:
            if key in self.indivs:
                hack = getHack(key)
            if self.params[key][0] == 'gibbs':
                samples[key].append(self.gibbsStep(key, samples, hack))
                print(f'    {key}: \n {samples[key][-1]} \n')
            elif self.params[key][0] == 'indepMetropolis':
                samples[key].append(self.indepmhStep(key, samples, hack))
                print(f'    {key}: \n {samples[key][-1]} \n')
            elif self.params[key][0] == 'rwMetropolis':
                samples[key].append(self.rwmhStep(key, samples, hack))
                print(f'    {key}: \n {samples[key][-1]} \n')
            else:
                raise ValueError(f'samples[{key}][0] must be among: \'gibbs\', \'indepMetropolis\', \'rwMetropolis\'')
