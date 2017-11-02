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
    params["farside"] = (6.59, 0.69)
    params["nearside"] = (1.56, 0.90)
    params["pkt"] = (0.57, 1.05)
    A, b = params[loc]
    
    Dtr = A*R**b
    return Dtr*0.12

def get_impact_distribution(era=''):
    '''Define impact population density.'''
    dmin, dmax = 1, 50
    if era == 'large':
        dmin, dmax = 20, 50
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
        x *= self.Hxy
        y *= self.Hxy
        
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                dist = np.sqrt((x-i*self.Hxy/self.gridsize)**2 + 
                               (y-j*self.Hxy/self.gridsize)**2)
        
                if dist < R: # within basin, use basin shape
                    excavation = -int(self.gridsize * basin_shape(dist/R, R)/self.Hz)
                    for k in range(self.gridsize-excavation):
                        self.field[i,j,k] = self.field[i,j,k+excavation]
                elif dist < 2*R: # outside, use ejecta blanket
                    ejecta = int(self.gridsize * ejecta_thickness(dist, R)/self.Hz)
                    for k in range(self.gridsize-1, ejecta, -1):
                        self.field[i,j,k] = self.field[i,j,k-ejecta]
                    self.field[i,j,:ejecta] = max_excavation(R)
                    
        return

    def plot(self, suffix):
        fig, axarr = plt.subplots(1, 1, figsize=(8, 8))

        im = axarr.imshow(self.output[-1][:,:].T, extent=[0, 50, 50, 0])
        
        divider = make_axes_locatable(axarr)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('Depth of origin [km]') #, rotation=270)
        cb.set_ticks(range(0, int(max(np.reshape(self.output[-1][:,:], (100**2,)))), 5))

        plt.savefig("img/moon-surface"+suffix+".png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

        bins = np.linspace(0, 50, 10)
        data, labels = [], []
        for i in np.linspace(1, len(self.output)-1, 3):
            data.append(np.reshape(self.output[int(i)], (100**2,)))
            labels.append(int(i))
            
        mu = np.mean(data[-1])
        std = np.std(data[-1])
        print(mu, std)
        
        fig, axarr = plt.subplots(1, 1, figsize=(8, 8))

        axarr.hist(data, bins=bins, normed=True, histtype='bar', label=labels, zorder=10)
        axarr.plot(bins, norm.pdf(bins, mu, std))

        axarr.set_xlim(0, 50)
        axarr.legend(loc="best")
        axarr.set_xlabel("Depth of origin [km]")
        axarr.set_ylabel("Surface fraction")
        axarr.grid()

        plt.savefig("img/depth-distribution"+suffix+".eps", format='eps', bbox_inches='tight')
        plt.show()

def main(rheo, giant, large, small):
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

    moon.plot(suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',
                        help="Load results instead of running sim")
    parser.add_argument('--rheo', default='farside',
                        help="Choose between near and farside rheology")
    parser.add_argument('-g', '--giant', default=2, type=int,
                        help="Number of impacts with size 40 < x < 50 km")
    parser.add_argument('-l', '--large', default=100, type=int,
                        help="Number of impacts with size 20 < x < 50 km")
    parser.add_argument('-s', '--small', default=45000, type=int,
                        help="Number of impacts with size 1 < x < 20 km")
    args = parser.parse_args()

    if not args.load:
        main(args.rheo, args.giant, args.large, args.small)
    else:
        with open(args.load, "rb") as f:
            surface = pickle.load(f)
        moon = Surface(100)
        moon.output = surface
        moon.plot()
