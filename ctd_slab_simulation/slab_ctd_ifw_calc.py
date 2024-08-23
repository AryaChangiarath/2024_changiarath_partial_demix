import MDAnalysis as mda
import numpy as np
from numpy.linalg import norm
import pandas as pd
import sklearn.decomposition
import csv
from MDAnalysis.analysis.distances import distance_array
import freud
import os
import glob
from scipy.optimize import least_squares, curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from matplotlib.colors import LogNorm
import sys
import math
import matplotlib.pyplot as plt
import gsd.hoomd
import scipy.optimize as opt
from scipy import stats

# Define parameters

path = './'                      # path to the folder to save the data in
dz = 0.5                        # width of a single bin (~0.5nm)
vaporEndCM = -50.               # left x-position (in distance unit D) for vapor regime in CM frame
useMax = 0                      
#calculation
numCMs = 10                    

# Load file names and frames
fil_name = sys.argv[1]
fil_no = int(sys.argv[2])
s = gsd.hoomd.open(name=fil_name, mode='rb')
nFrames = len(s)

snapshotCount = nFrames
startFrame = 0
nFrames = snapshotCount - startFrame

# Initialize simulation box and particles
simBox = s[0].configuration.box
masses = s[0].particles.mass
screenTimer = int((snapshotCount - startFrame) / 20)
if screenTimer == 0:
    screenTimer = 1

# Determine number and size of bins
histoVolume = simBox[0] * simBox[1] * dz
histoCount = int(simBox[2] / dz) + 1
histoBins = np.linspace(-simBox[2] / 2.0, simBox[2] / 2.0, num=histoCount)

# Prepare histograms
distribMonomerPos = []
distribMonomerCMPos = []

print('number of frames: ', snapshotCount)

def pos_shift(pos, axis):
    nini = 0
    nfin = s[0].particles.position.shape[0]
    
    distribMonomerPos.append(np.histogram(pos[nini:nfin, axis],
                                      bins=histoBins,
                                      weights=masses[nini:nfin],
                                      density=False)[0])

    distribMonomerPos.append(np.histogram(pos[nini:nfin, axis],
                                  bins=histoBins,
                                  weights=masses[nini:nfin],
                                  density=False)[0])

    if useMax:
        particlesPosShifted = pos[:]
        tempDistrib = np.histogram(particlesPosShifted[:, axis], bins=histoBins, weights=masses, density=False)[0]
        maxDensBin = np.argmax(tempDistrib)
        particlesPosShifted[:, axis] = particlesPosShifted[:, axis] - (maxDensBin * dz - simBox[axis] / 2.0)
        particlesPosShifted[particlesPosShifted < -simBox[axis] * 0.5] += simBox[axis]
        particlesPosShifted[particlesPosShifted > simBox[axis] * 0.5] -= simBox[axis]

        CM = np.mean(particlesPosShifted, axis=0)
        particlesPosShifted = particlesPosShifted - CM
        particlesPosShifted[particlesPosShifted < -simBox[axis] * 0.5] += simBox[axis]
        particlesPosShifted[particlesPosShifted > simBox[axis] * 0.5] -= simBox[axis]
    else:
        particlesPosShifted = pos[:]
        for j in range(numCMs):
            CM = np.mean(particlesPosShifted, 0)
            particlesPosShifted[:, axis] = particlesPosShifted[:, axis] - CM[axis]
            particlesPosShifted[particlesPosShifted < -simBox[axis] * 0.5] += simBox[axis]
            particlesPosShifted[particlesPosShifted > simBox[axis] * 0.5] -= simBox[axis]

    return particlesPosShifted

Lx, Ly, Lz = s[0].configuration.box[:3]
lz = Lz
L = 15
edges = np.arange(-lz / 2, lz / 2, 1)
dz = (edges[1] - edges[0]) / 2.
z = edges[:-1] + dz
tmin = 0
tmax = snapshotCount
dt = 1
nt = tmax - tmin

lambda1 = np.zeros(nt)
lambda2 = np.zeros(nt)
lambda3 = np.zeros(nt)
ax1, ax2, ax3 = np.zeros((nt, 3)), np.zeros((nt, 3)), np.zeros((nt, 3))
tims = 0
natoms = s[0].particles.N

position = np.zeros((nFrames, natoms, 3))

j = 0
for ts in range(startFrame, snapshotCount):
    particlesPosShifted_x = pos_shift(s[ts].particles.position, 2)
    pos = particlesPosShifted_x[:]
    
    hm1 = np.histogram(pos[:, 2], bins=edges)[0]
    position[j][:] = pos
    
    j += 1

h1 = np.apply_along_axis(lambda a: np.histogram(a, bins=edges)[0], 1, position[:, :, 2])
import matplotlib
matplotlib.rcParams.update({'font.size': 23})

conv = 10 / 6.022 / 140 / L / L * 1e3

plt.figure(figsize=(7, 4))
plt.rcParams.update({'font.size': 23})
plt.plot(z, conv * h1.mean(axis=0), 'grey', linewidth=3., label='CTD')

plt.legend(fontsize=18)
plt.xlabel('z [nm]')
plt.ylabel('Density [mM]')
plt.tight_layout()
plt.xlim(-75, 75)
plt.savefig('densityPlot_300ctd.png', dpi=350)

positionParticle = position
nFrames = positionParticle.shape[0]

def interWidth(x, a, b, c, d):
    return 0.5 * (a + b) + 0.5 * (b - a) * np.tanh((np.abs(x) - c) / d)

def error(hs):
    return np.std(hs) / np.sqrt(len(hs))

width_error = []
intrWid_avg = []
list_width_avg = []

for gs in np.linspace(2, 8, 10):
    print(gs)
    grid_size_x = gs
    grid_size_y = gs
    num_grids_x = int(Lx / grid_size_x)
    num_grids_y = int(Ly / grid_size_y)

    n_chains1 = 100
    nm1 = 140
    n_chains3 = 200
    nm3 = 71

    histPctd = np.zeros((nFrames, edges.shape[0] - 1))
    histctd = np.zeros((nFrames, edges.shape[0] - 1))
    histhrd = np.zeros((nFrames, edges.shape[0] - 1))
    
    hm1 = np.zeros((num_grids_x * num_grids_y, edges.shape[0] - 1))
    hm2 = np.zeros((num_grids_x * num_grids_y, edges.shape[0] - 1))
    hm3 = np.zeros((num_grids_x * num_grids_y, edges.shape[0] - 1))
    
    list_width = []
    intrWid = []

    for nframe in range(nFrames):
        count = 0
        for i in range(num_grids_x):
            for j in range(num_grids_y):
                x_start = -Lx / 2 + i * grid_size_x
                x_end = x_start + grid_size_x
                y_start = -Ly / 2 + j * grid_size_y
                y_end = y_start + grid_size_y

                particles = positionParticle[nframe][:n_chains1 * nm1]
                particles_in_grid_pctd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
                                              (y_start <= particles[:, 1]) & (particles[:, 1] < y_end)]

                particles = positionParticle[nframe][n_chains1 * nm1:]
                particles_in_grid_hrd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
                                              (y_start <= particles[:, 1]) & (particles[:, 1] < y_end)]

                hm2 = np.histogram(particles_in_grid_pctd[:, 2], bins=edges)[0]
                hm3 = np.histogram(particles_in_grid_hrd[:, 2], bins=edges)[0]

                count += 1
                z = edges[:-1] + dz
                try:
                    y = hm2 + hm3
                    z = edges[:-1] + dz
                    ind = (z < 0)
                    y = y[ind]
                    x = z[ind]
 
                    optimizedParameters1, pcov = opt.curve_fit(interWidth, x, y)

                    y = hm2 + hm3
                    z = edges[:-1] + dz
                    ind = (z > 0)
                    y = y[ind]
                    x = z[ind]
 
                    optimizedParameters2, pcov = opt.curve_fit(interWidth, x, y)

                    list_width.append(((optimizedParameters1[3] + optimizedParameters2[3]) * 0.5) ** 2)
                except:
                    continue

    width_error.append(error(np.array(list_width)))
    list_width_avg.append(np.array(list_width).mean())

gs_list = np.linspace(2, 8, 10)
combined_array = np.column_stack((gs_list, list_width_avg, width_error))

# Save the combined array to a file
np.savetxt('output_file_{}.txt'.format(fil_no), combined_array, delimiter='\t')
