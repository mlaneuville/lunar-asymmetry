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

def ejecta_thickness(r, R=1, a=0.14, b=0.74, c=-3.0):
    # Eq. 1.9 in Sturm 2016, JGR:P
    r = np.array(r)
    
    if not r.shape:
        if r < R:
            return 0
        else:
            return a*R**b*(r/R)**c

    idx = np.where(r > R)[0]
    dout = a*R**b*(r[idx]/R)**c

    idx = np.where(r < R)[0]
    din = 0*r[idx]

    return np.concatenate([din, dout])

def basin_shape(x, R=1):
    # assumed paraboloidal; prefactor to match ejected mass
    y = 2*R/5*(x**2-1)
    return y

def max_excavation(R, loc="farside"):
    # From Miljkovic et al 2016 (table 1)
    # and Potter et al 2015 (eq7)
    params = {}
    params["farside"] = (6.59, 0.69, 2.11, 0.85)
    params["nearside"] = (1.56, 0.90, 0.35, 1.12)
    params["pkt"] = (0.57, 1.05, 0.04, 1.45)
    A1, b1, A2, b2 = params[loc]

    C = R*17**0.58 # R is diameter here
    Dthin = (C/A2)**(1./b2)
    Dtr = A1*Dthin**b1

    return Dtr*0.12

def get_impact_distribution(era=''):
    '''Define impact population density.'''
    dmin, dmax = 1, 50
    if era == 'large':
        dmin, dmax = 20, 40
    if era == 'small':
        dmin, dmax = 1, 20
    if era == 'giant':
        dmin, dmax = 40, 50

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
        self.d = {}
        self.d['large'] = get_impact_distribution(era='large')
        self.d['small'] = get_impact_distribution(era='small')
        self.d['giant'] = get_impact_distribution(era='giant')
        self.d['all'] = get_impact_distribution(era='')
        
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                for k in range(self.gridsize):
                    # field contains depth of origin
                    self.field[i,j,k] = self.Hz*k/self.gridsize

        self.output.append(deepcopy(self.field[:,:,0]))

    def save_state(self):
        self.output.append(deepcopy(self.field[:,:,0]))
    
    def generate_impact(self, era='all'):
        '''xy position and impact diameter'''
        x, y = random.random(), random.random()
        D = self.d[era].rvs()
        return (x,y), D
    
    def impact(self, x, y, R, loc):
        depth = int(self.gridsize*max_excavation(R)/self.Hz)
        field2 = deepcopy(self.field)

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
                    field2[i,j,:ejecta] = self.field[int(x),int(y),depth]

        self.field = field2
        return

    def plot(self, suffix, show):
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

        bins = np.linspace(0, 100, 40)
        data, labels = [], []
        for i in np.linspace(1, len(self.output)-1, 3):
            data.append(np.reshape(self.output[int(i)], (100**2,)))
            labels.append(int(i))
            
        mu = np.mean(data[-1])
        std = np.std(data[-1])
        print(mu, std)
        
        fig, axarr = plt.subplots(1, 1, figsize=(8, 8))

        labels = ['After D > 40 km impacts',
                  'After 20 < D < 40 km impacts',
                  'After D < 20 km impacts']

        for epoch, l in zip(data, labels):
            x, bins, _ = plt.hist(epoch, bins=bins, normed=True, alpha=0.8, label=l)

        axarr.set_xlim(0, 50)
        axarr.legend(loc="best")
        axarr.set_xlabel("Depth of origin [km]")
        axarr.set_ylabel("Surface fraction")
        axarr.grid()

        plt.savefig("img/depth-distribution"+suffix+".png", format='png', bbox_inches='tight')
        if show:
            plt.show()

def main(rheo, giant, large, small, show=False):
    moon = Surface(100)

    history = [{'era':'giant', 'number':giant, 'output':min(giant,2)},
               {'era':'large', 'number':large, 'output':min(large,100)},
               {'era':'small', 'number':small, 'output':min(small,5000)}]

    for epoch in history:
        print(epoch)
        n_impacts = epoch['number']
        era = epoch['era']
        output = epoch['output']

        for i in range(n_impacts):
            (x, y), D = moon.generate_impact(era=era)
            moon.impact(x, y, D/2, loc=rheo)
        
            if ((i % output) == output-1) or (i == n_impacts - 1):
                print(i)
                moon.save_state()

    fname = 'dat/moon-surface'
    suffix = ''
    for epoch in history:
        if epoch['number'] > 0:
            suffix += '-%s%d' % (epoch['era'], epoch['number'])
    suffix += '-%s' % rheo
    fname += suffix+'.p'

    with open(fname, "wb") as f:
        pickle.dump(moon.output, f)

    moon.plot(suffix, show)

def get_impactors_count(giant):
    '''Return number of impacts per bin for a given number of giant imapcts.'''
    a = giant/40**(-3) # N(x > D) = a D**(-3)
    large = int(a*20**(-3) - giant)
    small = int(a - large - giant)
    print("Total number of impacts: %d" % a)
    print("\t- %d with D > 40 km" % giant)
    print("\t- %d with 20 < D < 40 km" % large)
    print("\t- %d with D < 20 km" % small)
    return (giant, large, small)

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

    if not args.load:
        giant, large, small = get_impactors_count(args.basins)
        main(args.rheo, giant, large, small, args.show)
    else:
        with open(args.load, "rb") as f:
            surface = pickle.load(f)
        suffix = "-"+"-".join(args.load.split("-")[2:6])[:-2]
        moon = Surface(100)
        moon.output = surface
        moon.plot(suffix, args.show)
