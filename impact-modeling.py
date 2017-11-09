# coding: utf-8

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.stats import rv_continuous
from scipy.stats import norm
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
import pickle

font = {'size'   : 14}
rc('font', **font)

def ejecta_thickness(r, L=1, a=0.14, b=0.74, c=-3.0):
    # Eq. 1.9 in Sturm 2016, JGR:P
    r = np.array(r)
    R = L*1e3/2
    
    if not r.shape:
        if r < R:
            return 0
        else:
            return a*R**b*(r/R)**c/1e3

    idx = np.where(r >= R)[0]
    dout = a*R**b*(r[idx]/R)**c/1e3

    idx = np.where(r < R)[0]
    din = 0*r[idx]

    return np.concatenate([din, dout])

def basin_shape(x, R=1):
    # assumed paraboloidal; prefactor to match ejected mass
    y = 2*R/5*(x**2-1)
    return y

def max_excavation(L, loc="farside"):
    # From Miljkovic et al 2016 (table 1)
    # and Potter et al 2015 (eq7)
    params = {}
    params["farside"] = (6.59, 0.69, 2.11, 0.85)
    params["nearside"] = (1.56, 0.90, 0.35, 1.12)
    params["pkt"] = (0.57, 1.05, 0.04, 1.45)
    A1, b1, A2, b2 = params[loc]

    C = L*17**0.58
    Dthin = (C/A2)**(1./b2)
    Dtr = A1*Dthin**b1

    return Dtr*0.12

def get_impact_distribution(dmin, dmax):
    '''Define impact population density.'''

    def get_constant(dmin, dmax):
        '''Computes normalization factor for density distribution'''
        def integ(x):
            def f(x, C=1):
                return C*x**(-3)
            return quad(f, dmin, dmax, args=(x,))[0] - 1
        return fsolve(integ, 1)[0]

    constant = get_constant(dmin, dmax)

    class rv(rv_continuous):
        "Impact distribution"
        def _pdf(self, x):
            return constant*x**(-3)
    return rv(a=dmin, b=dmax)

class Surface:
    def __init__(self, n):
        # Box size, in km
        self.Hxy = 100
        self.Hz = 50
        
        self.field = np.ones((n, n, n))
        self.output = []

        self.gridsize = n
        self.d = get_impact_distribution(0, 150)
        
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                for k in range(self.gridsize):
                    # field contains depth of origin
                    self.field[i,j,k] = self.Hz*k/self.gridsize

        self.output.append(deepcopy(self.field[:,:,0]))

    def save_state(self):
        self.output.append(deepcopy(self.field[:,:,0]))
    
    def generate_impact(self):
        '''xy position and impact diameter'''
        x, y = random.random(), random.random()
        L = self.d.rvs()
        return (x,y), L
    
    def impact(self, x, y, L, loc):
        R = L/2
        field2 = deepcopy(self.field)
        depth = int(self.gridsize*max_excavation(L)/self.Hz)
        stratigraphy = self.field[int(x), int(y), :depth]

        x *= self.Hxy
        y *= self.Hxy

        imin = int(x - 2*R)
        imax = int(x + 2*R)
        # loop only over dist(i,j) < 2R
        for i in range(imin, imax):
            dx = i-x
            i = i % self.gridsize # make sure we are within the grid
            if abs(dx) > 2*R:
                continue

            jmin = int(y - np.sqrt(4*R**2 - dx**2))
            jmax = int(y + np.sqrt(4*R**2 - dx**2))
            for j in range(jmin ,jmax):
                dy = j-y
                j = j % self.gridsize # make sure we are within the grid

                dist = np.sqrt(dx**2 + dy**2)
                if dist < R: # within basin, use basin shape
                    excavation = -int(self.gridsize * basin_shape(dist/R, R)/self.Hz)
                    for k in range(self.gridsize-excavation):
                        field2[i,j,k] = self.field[i,j,k+excavation]
                    for k in range(self.gridsize-excavation, self.gridsize):
                        field2[i,j,k] = field2[i,j,k-1] + self.Hz/self.gridsize
                elif dist < 2*R: # outside, use ejecta blanket
                    ejecta = int(self.gridsize * ejecta_thickness(dist, R)/self.Hz)
                    for k in range(self.gridsize-1, ejecta, -1):
                        field2[i,j,k] = self.field[i,j,k-ejecta]
                    field2[i,j,:ejecta] = stratigraphy[ejecta::-1]

        self.field = field2
        return

    def plot(self, bins, counts, suffix, show):
        fig, axarr = plt.subplots(1, 1, figsize=(8, 8))

        im = axarr.imshow(self.output[-1][:,:].T, extent=[0, 50, 50, 0])
        
        divider = make_axes_locatable(axarr)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('Depth of origin [km]') #, rotation=270)
        cb.set_ticks(range(0, int(max(np.reshape(self.output[-1][:,:], (100**2,)))), 5))

        plt.savefig("img/moon-surface"+suffix+".png", format='png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        data = []
        for i in range(len(self.output)):
            data.append(np.reshape(self.output[int(i)], (100**2,)))
            
        mu = np.mean(data[-1])
        std = np.std(data[-1])
        print(mu, std)
        
        fig, axarr = plt.subplots(1, 1, figsize=(8, 8))

        xbins = np.linspace(0, 100, 40)
        for cat, count, val in zip(bins[::-1], counts[::-1], data):
            l = "After %d %d < L < %d km impacts" % (count, cat[0], cat[1]) 
            x, bins, _ = plt.hist(val, bins=xbins, normed=True, alpha=0.8, label=l)

        axarr.set_xlim(0, 50)
        axarr.legend(loc="best")
        axarr.set_xlabel("Depth of origin [km]")
        axarr.set_ylabel("Surface fraction")
        axarr.grid()

        plt.savefig("img/depth-distribution"+suffix+".png", format='png', bbox_inches='tight')
        if show:
            plt.show()

def main(rheo, counts, bins, show=False):
    moon = Surface(100)

    for count, cat in zip(counts[::-1], bins[::-1]):
        print("%d impacts, with %d < L < %d km" % (count, cat[0], cat[1]))
        n_impacts = count

        moon.d = get_impact_distribution(cat[0], cat[1])
        for i in range(n_impacts):
            (x, y), L = moon.generate_impact()
            moon.impact(x, y, L, loc=rheo)
        
        moon.save_state()

    fname = 'dat/moon-surface'
    suffix = '-giant%d' % counts[-1]
    suffix += '-%s' % rheo
    fname += suffix+'.p'

    with open(fname, "wb") as f:
        pickle.dump(moon.output, f)

    moon.plot(bins, counts, suffix, show)

def get_impactors_count(giant, bins):
    '''Return number of impacts per bin for a given number of giant imapcts.'''
    a = giant/bins[-1][0]**(-2.2) # N(x > D) = a D**(-3)
    count = [a]
    print("Total number of impacts: %d" % a)
    for b in bins[1:]:
        c = a*b[0]**(-2.2)
        count.append(c)
        print("\t- %.1f > %d km" % (c, b[0]))
    for i, c in enumerate(count):
        if i == len(count)-1:
            continue
        count[i] -= count[i+1]
    return [int(x) for x in count]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',
                        help="Load results instead of running sim")
    parser.add_argument('--rheo', default='farside', type=str,
                        choices=["farside", "nearside", "pkt"],
                        help="Choose between near and farside rheology")
    parser.add_argument('-b', '--basins', default=5, type=int,
                        help="Number of impacts basins with size 40 < x < 50 km")
    parser.add_argument('-s', '--show', default=False, action="store_true",
                        help="Toggle to show the figures at the end")
    args = parser.parse_args()

    bins = [(1, 20), (20, 100), (100, 150)]
    count = get_impactors_count(args.basins, bins)

    if not args.load:
        main(args.rheo, count, bins, args.show)
    else:
        with open(args.load, "rb") as f:
            surface = pickle.load(f)
        suffix = "-"+"-".join(args.load.split("-")[2:6])[:-2]
        moon = Surface(100)
        moon.output = surface
        moon.plot(bins, count, suffix, args.show)
