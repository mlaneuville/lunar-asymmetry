# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from math import exp
from math import sqrt
from glob import glob
from numpy import random
import argparse
import json
import sys

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 'large'})

def Mg_to_PCS(Mg):
    fits = {}
    fits['80'] = [ -3.19574082e-04, 1.39071011e-01, -2.42050960e+01, 2.10607708e+03, -9.16066292e+04, 1.59355528e+06]
    fits['90'] = [ -2.58074144e-04, 1.11954298e-01, -1.94245707e+01, 1.68486061e+03, -7.30572514e+04, 1.26693581e+06]
    fits['95'] = [ -1.73237110e-04, 7.45526111e-02, -1.28315986e+01, 1.10400284e+03, -4.74796870e+04, 8.16592921e+05]
    fits['00'] = [ 0.5*(a+b) for (a,b) in zip(fits['90'], fits['95'])]
    c = fits['90']

    inv = []
    for value in Mg:
        p = np.poly1d(c) - value
        real_roots = p.r[np.where(p.r.imag == 0)]
        if len(real_roots) > 1:
            print("more than one real root")
        inv.append(real_roots[0].real)
    return np.array(inv)

def PCS_to_Mg(pcs):
    '''Assumes linear variation with PCS.'''
    fits = {}
    fits['80'] = [ -3.19574082e-04, 1.39071011e-01, -2.42050960e+01, 2.10607708e+03, -9.16066292e+04, 1.59355528e+06]
    fits['90'] = [ -2.58074144e-04, 1.11954298e-01, -1.94245707e+01, 1.68486061e+03, -7.30572514e+04, 1.26693581e+06]
    fits['95'] = [ -1.73237110e-04, 7.45526111e-02, -1.28315986e+01, 1.10400284e+03, -4.74796870e+04, 8.16592921e+05]
    fits['00'] = [ 0.5*(a+b) for (a,b) in zip(fits['90'], fits['95'])]
    c = fits['90']

    pcs_ = pcs*100
    mg = np.polyval(c, pcs_)
    mg[np.where(mg < 0)] = 0
    
    return mg

G = 6.67e-11 # gravitational constant, m3/kg/s2
YEAR = 365*24*3600. # a year in seconds
MA = 1e6*YEAR       # a Ma in seconds
RM = 1740e3 # moon radius
RC = 400e3  # moon core radius
RE = 6370e3 # earth radius
MM = 5.972e24 # moon mass
ME = 7.348e22 # earth mass
D0 = 125e3  # initial depth at which the crust starts to crystallize
VOL_MANTLE = 4*np.pi*(RM**3 - RC**3)/3

# this needs to be defined only once
LATENT = 3e5   # latent heat of crystallization, J/kg
RHO = 3300.      # average crustal density, kg
CP = 1000.       # specific heat of the crust, J/K/kg
GRAD_TL = 2.5e-4 #

k2Q = 0.024
A0 = RE

def area(r):
    '''Returns area of the shell at radius RM-r.'''
    a = 4.*np.pi*(RM-r)**2
    return a

def volume(r):
    '''Returns volume of the shell between RM-r and RM-D0.'''
    v = 4.*np.pi*((RM-r)**3 - (RM-D0)**3)/3
    return v

def partitioning(y):
    '''Takes crustal thickness as input and returns PCS.'''    
    mean_crust = 0.5*(y[0]+y[2])
    return 1 - volume(mean_crust)/VOL_MANTLE

def isotherm(y, t, phi, T=800):
    '''Returns the depth of an isotherm for a given timestep.'''
    TS = 250
    TM = 1600
    return np.array([y[0], y[2]])*(T-TS)/(TM-TS)

def radio(t):
    '''Returns radioactive heating as a function of time since t0 (in years)'''
    t0 = 4.5e9
    t *= 1e6 # t is given in Ma
    constants = {
        '238U': {'H': 9.46e-5, 'tau': 4.47e9, 'x': 0.9928, 'c0': 20.3e-9},
        '235U': {'H': 5.69e-4, 'tau': 7.04e8, 'x': 0.0071, 'c0': 20.3e-9},
        '232Th': {'H': 2.64e-5, 'tau': 1.40e10, 'x': 1, 'c0': 79.5e-9},
        '40K': {'H': 2.92e-5, 'tau': 1.25e9, 'x': 1.19e-4, 'c0': 240e-6},
    }
    heat = 0
    for el in constants.values():
        heat += el['c0']*el['H']*el['x']*np.exp((t0-t)*np.log(2)/el['tau'])
    return heat

def get_depths(y, mixing='normal-middle', X=20e3):
    N = 1000
    depths_ns, depths_fs = [], []
    mu, std = {}, {}

    db_mu = {
        'low': {'ns':24.9e3, 'fs':25.8e3},
        'middle': {'ns':24.9e3, 'fs':25.8e3},
        'high': {'ns':24.9e3, 'fs':25.8e3}
    }

    db_std = {
        'low': {'ns':9.4e3, 'fs':10.2e3},
        'middle': {'ns':9.4e3, 'fs':10.2e3},
        'high': {'ns':9.4e3, 'fs':10.2e3}
    }

    mixmodel = mixing.split('-')
    if mixmodel[0] == 'uniform':
        depths_ns = random.rand(N)*X
        depths_fs = random.rand(N)*X
    elif mixmodel[0] == 'normal':
        if mixmodel[1] == 'centered':
            mu['ns'] = db_mu['middle']['ns']
            mu['fs'] = mu['ns']
            std['ns'] = db_std['middle']['ns']
            std['fs'] = std['ns']
        else:
            mu['ns'] = db_mu[mixmodel[1]]['ns']
            mu['fs'] = db_mu[mixmodel[1]]['fs']
            std['ns'] = db_std[mixmodel[1]]['ns']
            std['fs'] = db_std[mixmodel[1]]['fs']
        while len(depths_ns) < N:
            r = random.normal(loc=mu['ns'], scale=std['ns'])
            if r > 1 and r < max(y[:,0]):
                depths_ns.append(r)
        while len(depths_fs) < N:
            r = random.normal(loc=mu['fs'], scale=std['fs'])
            if r > 1 and r < max(y[:,2]):
                depths_fs.append(r)
    else:
        print("This shouldn't happen.")
        sys.exit(1)

    return depths_ns, depths_fs

def plot_results(time, y, iso_n, iso_f, c, a, hfs_n, hfs_f, mixing, run, suffix=''):
    
    # crust size evolution
    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.plot(time, y[:, 0]/1e3, 'r', label='nearside', lw=2)
    if run != "symmetrical":
        ax.plot(time, y[:, 2]/1e3, 'g', label='farside', lw=2)
        ax.legend(loc='best')
    ax.set_xlabel("Time [Ma]")
    ax.set_ylabel("Crustal thickness [km]")
    ax.set_xlim(0, max(time))
    ax.set_ylim(0, 70)
    ax.grid()
    
    plt.savefig("img/evolution"+suffix+".png", format="png", bbox_inches="tight")

    # isotherm depth
    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.plot(time, iso_n/1e3, 'r', label='nearside', lw=2)
    if run != "symmetrical":
        ax.plot(time, iso_f/1e3, 'g', label='farside', lw=2)
        ax.legend(loc='best')
    ax.set_xlabel("Time [Ma]")
    ax.set_ylabel("800 K isotherm depth [km]")
    ax.set_xlim(0, max(time))
    ax.set_ylim(0, 1.05*max(iso_n)/1e3)
    ax.grid()
    
    plt.savefig("img/isotherms"+suffix+".png", format="png", bbox_inches="tight")

    # composition profile [in Mg#]
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.plot(PCS_to_Mg(c), y[:, 0]/1e3, 'r', label='nearside', lw=2)
    if run != "symmetrical":
        ax.plot(PCS_to_Mg(c), y[:, 2]/1e3, 'g', label='farside', lw=2)
        ax.legend(loc='best')
    ax.set_xlabel("Magnesium number [-]")
    ax.set_ylabel("Crustal thickness [km]")
    ax.set_ylim(70, 0)
    ax.grid()
    
    plt.savefig("img/composition"+suffix+".png", format="png", bbox_inches="tight")

    # orbital distance
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(time, a, 'k', lw=2)
    ax.set_xlabel("Time [Ma]")
    ax.set_ylabel("Semimajor axis [R$_{earth}$]")
    ax.set_xscale("log")
    ax.set_ylim(0, 60)
    ax.set_xlim(min(time), max(time))
    ax.grid()

    plt.savefig("img/semimajor"+suffix+".png", format="png", bbox_inches="tight")

    # surface properties
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(time, hfs_n, 'r', lw=2, label='nearside')
    ax.plot(time, hfs_f, 'g', lw=2, label='farside')
    ax.axhline(1361, color='k', ls='--')
    ax.set_xlabel("Time [Ma]")
    ax.set_ylabel("Surface heat flow [W/m$^2$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(time), max(time))
    ax.legend(loc='best')
    ax.grid()

    plt.savefig("img/surface_hf"+suffix+".png", format="png", bbox_inches="tight")

    # composition sample
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    depths_ns, depths_fs = get_depths(y, mixing=mixing)
    compo_NS = interp1d(y[:, 0], PCS_to_Mg(c))
    compo_FS = interp1d(y[:, 2], PCS_to_Mg(c))

    bins = np.linspace(40, 80, 41)
    ax.hist(compo_NS(depths_ns), bins=bins, normed=1, facecolor='red', alpha=0.5, label='nearside')
    if run != "symmetrical":
        ax.hist(compo_FS(depths_fs), bins=bins, normed=1, facecolor='green', alpha=0.5, label='farside')
        ax.legend(loc='best')

    ax.set_xlabel("Magnesium number [-]")
    ax.set_ylabel("Normalized distribution")
    ax.set_xlim(50,75)
    ax.set_yticks(np.arange(0, 0.55, 0.1))
    ax.grid()
    
    plt.savefig("img/distribution"+suffix+".png", format="png", bbox_inches="tight")


class Evolution:
    def __init__(self, run, delay=0, mixing='normal-middle'):
        self.run = run
        self.mixing = mixing
        self.time = np.logspace(-6, 2, 1000) # in Ma
        self.output = 0
        self.compo= 0
        self.semimajor = A0
        self.delay = delay

    def orbital_distance(self, t):
        def orbit(a, t):
            '''Eq. 4.213 from SSD'''
            res = 2/13*a**(13./2)*(1-(A0/a)**(13./2))
            res -= 3/2*k2Q*(G/ME)**0.5*RE**5*MM*t
            return res
        r = fsolve(orbit, self.semimajor*RE, factor=10, args=(t*MA))[0]
        return max(A0, r)

    def get_heat_flow(self, side, t, d, phi, model):
        '''Heat flow from a conductive profile in the crust. Maximum value is set to the radiative heat flow.'''
        self.semimajor = self.orbital_distance(t)/RE
        
        S0 = 1361
        Ts = 331
        sig = 5.67e-8
        if side == 'NS' and model != 'symmetrical':
            Ts = ((S0/2 + 4*S0*(5/self.semimajor)**2)/sig)**0.25

            
        q = 2*(1600-Ts)/min(d, np.sqrt(t*MA*1e-6))
        return q

    def deriv(self, y, t, model, phi, delay=0):
        '''
        Effective cooling rate determines the crystallization rate:
            Qsurf - Qradio = (latent + cooling)*dr/dt
        The only difference is that Qsurf is larger on the farside.
        '''
        denumFS = area(y[2])*LATENT*RHO + RHO*volume(y[2])*CP*GRAD_TL
        numFS = (self.get_heat_flow('FS', t, y[2], phi, model)*area(0) - radio(t)*RHO*VOL_MANTLE)
    
        denumNS = area(y[0])*LATENT*RHO + RHO*volume(y[0])*CP*GRAD_TL
        numNS = (self.get_heat_flow('NS', t, y[0], phi, model)*area(0) - radio(t)*RHO*VOL_MANTLE)
        
        if model == "delay" and t < delay:
            denumNS *= 1e5
        
        return np.array([y[1], MA*numNS/denumNS, y[3], MA*numFS/denumFS])

    def solve(self, phi=0.0):
        yinit = np.array([1.0, 0.0, 1.0, 0.0])
        self.output = odeint(self.deriv, yinit, self.time, args=(self.run, phi, self.delay), rtol=1e-1)

        tfinal = np.where(self.output[:,2] > 70e3)[0][0]
        self.output = self.output[:tfinal]
        self.time = self.time[:tfinal]

        self.compo = np.zeros(len(self.output))
        self.iso_n = np.zeros(len(self.output))
        self.iso_f = np.zeros(len(self.output))
        self.hfs_n = np.zeros(len(self.output))
        self.hfs_f = np.zeros(len(self.output))
        self.orbit = np.zeros(len(self.output))

        for i, t in enumerate(self.time):
            self.compo[i] = partitioning(self.output[i])
            self.iso_n[i], self.iso_f[i] = isotherm(self.output[i], t, phi)
            self.hfs_n[i] = self.get_heat_flow('NS', t, self.output[i][0], phi, self.run)
            self.hfs_f[i] = self.get_heat_flow('FS', t, self.output[i][2], phi, self.run)
            self.orbit[i] = self.orbital_distance(t)/RE

        print("tfinal: %.1f Ma, crust ns/fs: %.1f/%.1f km" % (
              self.time[-1],
              self.output[-1, 0]/1e3,
              self.output[-1, 2]/1e3))

    def get_mg_distribution(self, mixing='normal-middle'):
        y = self.output
        c = self.compo

        depths, depths_fs = get_depths(y, mixing=mixing)

        compo_ns = interp1d(y[:, 0], PCS_to_Mg(c))
        mg_ns = compo_ns(depths)

        compo_fs = interp1d(y[:, 2], PCS_to_Mg(c))
        mg_fs = compo_fs(depths_fs)

        return [mg_ns.mean(), mg_ns.std(), mg_fs.mean(), mg_fs.std()]

    def get_crust_stats(self):
        idx = np.where(self.delay < self.time)[0][0]
        return self.output[-1, 0], self.output[-1, 2], self.output[idx, 2]

    def plot(self, args):
        suffix = '-'
        for k,v in args.items():
            if k == 'delay':
                suffix += "%s:%.2f-" % (k,v)
            else:
                suffix += "%s:%s-" % (k,v)
        suffix = suffix[:-1]

        plot_results(self.time, 
                     self.output, 
                     self.iso_n,
                     self.iso_f,
                     self.compo, 
                     self.orbit,
                     self.hfs_n,
                     self.hfs_f,
                     mixing=self.mixing,
                     run=self.run,
                     suffix=suffix)

def generate_runs(ARGS):
    '''Generate a list of parameter sets to explore phase space.'''
    parameters = []
    for i in range(ARGS.num):
        if ARGS.delay is None:
            delay = random.random()
        else:
            delay = ARGS.delay
        parameters.append({'run': ARGS.run, 'mixing': ARGS.mixing, 'delay': delay})

    return parameters


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-r', '--run', default='delay', type=str,
                        choices=['delay', 'symmetrical'],
                        help="Run type")
    PARSER.add_argument('-m', '--mixing', default='normal-middle', type=str,
                        choices=['normal-low', 'normal-middle', 'normal-high',
                                 'uniform', 'normal-centered'],
                        help="Mixing model")
    PARSER.add_argument('-d', '--delay', type=float,
                        help="Delay in Ma between near and farside cooling")
    PARSER.add_argument('--plot', default=False, action="store_true",
                        help="If true, plot run output in img/ folder")
    PARSER.add_argument('-n', '--num', default=10, type=int,
                        help="Number of runs")
    ARGS = PARSER.parse_args()

    d = generate_runs(ARGS)
    out_stats = []
    
    for i, params in enumerate(d):
        print(params)
        s = Evolution(**params)
        s.solve()

        if ARGS.plot:
            s.plot(params)

        stats = s.get_mg_distribution()
    
        params['nearside_mean'] = stats[0]
        params['nearside_std'] = stats[1]
        params['farside_mean'] = stats[2]
        params['farside_std'] = stats[3]

        crustns, crustfs, front = s.get_crust_stats()

        params['nearside_crust'] = crustns
        params['farside_crust'] = crustfs
        params['front'] = front

    
    with open('dat/data.txt', 'w') as outfile:
        json.dump(d, outfile)
