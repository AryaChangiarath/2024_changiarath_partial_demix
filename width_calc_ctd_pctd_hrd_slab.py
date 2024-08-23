
import MDAnalysis as mda
from numpy.linalg import norm
import numpy as np
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
import scipy.optimize as opt
import gsd.hoomd
from scipy import stats

# Parameters
bin_thickness = 0.5
use_max_density = 0
num_cm_iterations = 10
data_path = './'

# Load GSD file
trajectory = gsd.hoomd.open(name='traj_ctd_pctd_hrd.gsd', mode='rb+')
frame_start = 0

# Determine number of frames
num_frames = len(trajectory) - frame_start

# Box dimensions and particle masses
box_dimensions = trajectory[0].configuration.box
particle_masses = trajectory[0].particles.mass
screen_update_interval = max(1, int(num_frames / 20))

# Determine number and size of bins
histogram_volume = box_dimensions[0] * box_dimensions[1] * bin_thickness
num_bins = int(box_dimensions[2] / bin_thickness) + 1
bin_edges = np.linspace(-box_dimensions[2] / 2.0, box_dimensions[2] / 2.0, num=num_bins)

# Prepare histograms
monomer_position_distribution = []
monomer_cm_position_distribution = []

print('Number of frames:', num_frames)

def adjust_positions(positions, axis):
    initial_index = 0
    final_index = trajectory[0].particles.position.shape[0]
    
    # Compute density distribution of monomers
    monomer_position_distribution.append(
        np.histogram(positions[initial_index:final_index, axis],
                     bins=bin_edges,
                     weights=particle_masses[initial_index:final_index],
                     density=False)[0]
    )
    
    # Compute density distribution of monomers in CM frame
    if use_max_density:
        shifted_positions = positions[:]
        temp_distribution = np.histogram(shifted_positions[:, axis], bins=bin_edges, weights=particle_masses, density=False)[0]
        max_density_bin = np.argmax(temp_distribution)
        shifted_positions[:, axis] -= (max_density_bin * bin_thickness - box_dimensions[axis] / 2.0)
        shifted_positions[shifted_positions < -box_dimensions[axis] * 0.5] += box_dimensions[axis]
        shifted_positions[shifted_positions > box_dimensions[axis] * 0.5] -= box_dimensions[axis]
        
        cm = np.mean(shifted_positions, axis=0)
        shifted_positions -= cm
        shifted_positions[shifted_positions < -box_dimensions[axis] * 0.5] += box_dimensions[axis]
        shifted_positions[shifted_positions > box_dimensions[axis] * 0.5] -= box_dimensions[axis]
    else:
        shifted_positions = positions[:]
        for _ in range(num_cm_iterations):
            cm = np.mean(shifted_positions, axis=0)
            shifted_positions -= cm
            shifted_positions[shifted_positions < -box_dimensions[axis] * 0.5] += box_dimensions[axis]
            shifted_positions[shifted_positions > box_dimensions[axis] * 0.5] -= box_dimensions[axis]
    
    return shifted_positions

Lx, Ly, Lz = box_dimensions[:3]
lz = Lz
L = 15
edges = np.arange(-lz/2, lz/2, 1)
bin_thickness = (edges[1] - edges[0]) / 2.
z_values = edges[:-1] + bin_thickness
time_min = 0
time_max = num_frames
time_step = 1
num_time_steps = time_max - time_min

num_atoms = trajectory[0].particles.N
particle_positions = np.zeros((num_frames, num_atoms, 3))

frame_index = 0
for ts in range(frame_start, len(trajectory)):
    shifted_positions = adjust_positions(trajectory[ts].particles.position, 2)
    positions = shifted_positions[:]
    
    histogram = np.histogram(positions[:, 2], bins=edges)[0]
    particle_positions[frame_index][:] = positions
    
    frame_index += 1
position_particle_data = particle_positions




import matplotlib
matplotlib.rcParams.update({'font.size': 23})

def fitting_function(z, d, w, w0):
    return w0 * np.tanh((z - d) / w)

grid_sizes = np.linspace(2, 8, 10)

def calculate_error(data):
    return np.std(data) / np.sqrt(len(data))

position_particle_data = particle_positions
num_frames = position_particle_data.shape[0]
Lx, Ly, Lz = box_dimensions[:3]

grid_size_list = []
average_width_list = []
width_error_list = []


# Define number of chains and monomers
num_chains_ctd = 300
num_monomers_ctd = 140

num_chains_pctd = 100
num_monomers_pctd = 140

num_chains_hrd = 200
num_monomers_hrd = 71

# Recompute histograms with the updated bin_edges
histogram_ctd = np.apply_along_axis(
    lambda a: np.histogram(a, bins=bin_edges)[0], 
    1, 
    particle_positions[:, :num_chains_ctd * num_monomers_ctd, 2]
)

histogram_pctd = np.apply_along_axis(
    lambda a: np.histogram(a, bins=bin_edges)[0], 
    1, 
    particle_positions[:, num_chains_ctd * num_monomers_ctd:(num_chains_ctd * num_monomers_ctd + num_chains_pctd * num_monomers_pctd), 2]
)

histogram_hrd = np.apply_along_axis(
    lambda a: np.histogram(a, bins=bin_edges)[0], 
    1, 
    particle_positions[:, (num_chains_ctd * num_monomers_ctd + num_chains_pctd * num_monomers_pctd):(num_chains_ctd * num_monomers_ctd + num_chains_pctd * num_monomers_pctd + num_chains_hrd * num_monomers_hrd), 2]
)
# Assuming bin_edges corresponds to the histogram edges, make sure z_values aligns with these edges.
bin_edges = np.linspace(-Lz / 2, Lz / 2, num=len(np.mean(histogram_ctd[-1000:], axis=0) ) + 1)  # Define bin_edges based on histograms
z_values = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate z_values from bin edges

# Update conversion factors
conv_ctd = 10 / 6.022 / num_monomers_ctd / L / L * 1e3
conv_pctd = 10 / 6.022 / num_monomers_pctd / L / L * 1e3
conv_hrd = 10 / 6.022 / num_monomers_hrd / L / L * 1e3

# Compute mean histograms and apply conversion factor
mean_histogram_ctd = np.mean(histogram_ctd[-1000:], axis=0) * conv_ctd
mean_histogram_pctd = np.mean(histogram_pctd[-1000:], axis=0) * conv_pctd
mean_histogram_hrd = np.mean(histogram_hrd[-1000:], axis=0) * conv_hrd

# Plotting
plt.figure(figsize=(7, 4))
plt.plot(z_values, mean_histogram_pctd + mean_histogram_hrd, 'r', lw=3, label='pCTD-HRD')
plt.plot(z_values, mean_histogram_ctd, 'grey', lw=3, label='CTD')
plt.xlim(-75, 75)
plt.legend(fontsize=17)  # Set the legend fontsize to 17
plt.xlabel('z [nm]')
plt.ylabel('Density [mM]')
plt.tight_layout()

# Save the plot
plt.savefig('density_ctd_pctd_hrd.pdf')



for grid_size in grid_sizes:
    print(grid_size)
    
    grid_size_x = grid_size
    grid_size_y = grid_size

    num_grids_x = int(Lx / grid_size_x)
    num_grids_y = int(Ly / grid_size_y)

    num_chains_1 = 300
    num_monomers_1 = 140
    num_chains_2 = 100
    num_monomers_2 = 140
    num_chains_3 = 200
    num_monomers_3 = 71

    histogram_pctd = np.zeros((num_frames, edges.shape[0] - 1))
    histogram_ctd = np.zeros((num_frames, edges.shape[0] - 1))
    histogram_hrd = np.zeros((num_frames, edges.shape[0] - 1))
    
    histograms_1 = np.zeros((num_grids_x * num_grids_y, edges.shape[0] - 1))
    histograms_2 = np.zeros((num_grids_x * num_grids_y, edges.shape[0] - 1))
    histograms_3 = np.zeros((num_grids_x * num_grids_y, edges.shape[0] - 1))
    width_list = []

    for frame in range(num_frames):
        count = 0
        for i in range(num_grids_x):
            for j in range(num_grids_y):
                x_start = -Lx / 2 + i * grid_size_x
                x_end = x_start + grid_size_x
                y_start = -Ly / 2 + j * grid_size_y
                y_end = y_start + grid_size_y

                particles = position_particle_data[frame][:num_chains_1 * num_monomers_1]
                particles_in_grid_ctd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
                                                  (y_start <= particles[:, 1]) & (particles[:, 1] < y_end)]

                particles = position_particle_data[frame][num_chains_1 * num_monomers_1:num_chains_1 * num_monomers_1 + num_chains_2 * num_monomers_2]
                particles_in_grid_pctd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
                                                  (y_start <= particles[:, 1]) & (particles[:, 1] < y_end)]

                particles = position_particle_data[frame][num_chains_1 * num_monomers_1 + num_chains_2 * num_monomers_2:]
                particles_in_grid_hrd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
                                                  (y_start <= particles[:, 1]) & (particles[:, 1] < y_end)]

                histogram_ctd = np.histogram(particles_in_grid_ctd[:, 2], bins=edges)[0]
                histogram_pctd = np.histogram(particles_in_grid_pctd[:, 2], bins=edges)[0]
                histogram_hrd = np.histogram(particles_in_grid_hrd[:, 2], bins=edges)[0]
                
                count += 1
                z = edges[:-1] + bin_thickness
                try:
                    y = -(histogram_hrd + histogram_pctd - histogram_ctd) / (histogram_hrd + histogram_ctd + histogram_pctd)
                    valid_indices = ~np.isnan(y)
                    x = z[valid_indices]
                    y = y[valid_indices]
                    
                    x_values = np.linspace(-x.shape[0], 0, x.shape[0])
                    optimized_params_1, _ = opt.curve_fit(fitting_function, x_values, y)

                    y = -(histogram_hrd + histogram_pctd - histogram_ctd) / (histogram_hrd + histogram_ctd + histogram_pctd)
                    valid_indices = ~np.isnan(y)
                    x = z[valid_indices]
                    y = y[valid_indices]

                    x_values = np.linspace(0, x.shape[0], x.shape[0])
                    optimized_params_2, _ = opt.curve_fit(fitting_function, x_values, y)

                    if optimized_params_2[1] < 10 and optimized_params_1[1] < 10:
                        width_list.append(((optimized_params_1[1] + optimized_params_2[1]) * 0.5) ** 2)
                    
                except:
                    continue
        
    width_error_list.append(calculate_error(np.array(width_list)))
    average_width_list.append(np.array(width_list).mean())

np.savetxt('width_ctd_pctd_hrd.txt', average_width_list, fmt='%d', delimiter='\t')





# import MDAnalysis as mda
# from numpy.linalg import norm
# import numpy as np
# import pandas as pd
# import sklearn.decomposition
# import csv
# from MDAnalysis.analysis.distances import distance_array
# import freud
# import os
# import glob
# from scipy.optimize import least_squares, curve_fit
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from cycler import cycler
# from matplotlib.colors import LogNorm
# import sys
# import math
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# import gsd.hoomd
# from scipy import stats


# dz = .5       
# useMax = 0     
# numCMs = 10     
# path='./'

# s = gsd.hoomd.open(name= 'traj_ctd_pctd_hrd.gsd', mode='rb+')
# no=0
# startFrame =0

# snapshotCount = len(s) 

# nFrames= snapshotCount - startFrame #len(s)



# simBox = s[0].configuration.box
# masses = s[0].particles.mass
# screenTimer = int((snapshotCount - startFrame) / 20)                                                                                                      #?
# if screenTimer == 0:
#     screenTimer = 1

# # Determine number and size of bins
# histoVolume = simBox[0] * simBox[1] * dz
# histoCount = int(simBox[2] / dz) + 1
# histoBins = np.linspace(-simBox[2] / 2.0, simBox[2] / 2.0, num=histoCount)

# # Prepare histograms
# distribMonomerPos = []
# distribMonomerCMPos = []

# print('number of frames: ', snapshotCount)

# def pos_shift(pos,axis):


#     nini=0
#     nfin=s[0].particles.position.shape[0]
#     #print(pos)
#     distribMonomerPos.append(np.histogram(pos[nini:nfin, axis],
#                                       bins=histoBins,
#                                       weights=masses[nini:nfin],
#                                       density=False)[0])

#     # Compute density distribution of monomers in lab frame
#     distribMonomerPos.append(np.histogram(pos[nini:nfin, axis],
#                                   bins=histoBins,
#                                   weights=masses[nini:nfin],
#                                   density=False)[0])
#     # Compute density distribution of monomers in CM frame (first remove CM from box boundaries, then pos-=CM)
#     if useMax:
        
#         particlesPosShifted = pos[:]
#         #print(particlesPosShifted)
#         tempDistrib = np.histogram(particlesPosShifted[:, axis], bins=histoBins, weights=masses, density=False)[0]
#         maxDensBin = np.argmax(tempDistrib)
#         particlesPosShifted[:, axis] = particlesPosShifted[:, axis] - (maxDensBin * dz - simBox[axis] / 2.0)
#         particlesPosShifted[particlesPosShifted < -simBox[axis] * 0.5] += simBox[axis]
#         particlesPosShifted[particlesPosShifted >  simBox[axis] * 0.5] -= simBox[axis]

#         CM = np.mean(particlesPosShifted, axis=0)
#         particlesPosShifted = particlesPosShifted - CM
#         particlesPosShifted[particlesPosShifted < -simBox[axis] * 0.5] += simBox[axis]
#         particlesPosShifted[particlesPosShifted >  simBox[axis] * 0.5] -= simBox[axis]
#     else:
#         particlesPosShifted = pos[:]
#         for j in range(numCMs):
#             CM = np.mean(particlesPosShifted, 0)
#             particlesPosShifted = particlesPosShifted - CM
#             particlesPosShifted[particlesPosShifted < -simBox[axis] * 0.5] += simBox[axis]
#             particlesPosShifted[particlesPosShifted >  simBox[axis] * 0.5] -= simBox[axis]

  
#     #print(particlesPosShifted)
#     return particlesPosShifted



# Lx,Ly,Lz= s[0].configuration.box[:3] #u.dimensions[2]
# lz=Lz
# L=15
# edges = np.arange(-lz/2,lz/2,1)
# dz = (edges[1]-edges[0])/2.
# z = edges[:-1]+dz
# tmin = 0

# tmax = snapshotCount  #len(s)
# dt = 1
# nt = tmax - tmin


# # lambda1 = np.zeros(nt) #principal values
# # lambda2 = np.zeros(nt)
# # lambda3 = np.zeros(nt)

# # ax1, ax2, ax3 = np.zeros((nt,3)), np.zeros((nt,3)),np.zeros((nt,3)) #principal axes
# # tims = 0
# natoms=s[0].particles.N

# position = np.zeros((nFrames,natoms,3))

# j=0
# for ts in range (startFrame, snapshotCount):    
#     particlesPosShifted_x = pos_shift(s[ts].particles.position,2)

#     pos = particlesPosShifted_x[:] #u.atoms.positions[ini_no:fin_no] - u.atoms[ini_no:fin_no].center_of_geometry()
    
    
#     hm1 = np.histogram(pos[:,2] ,bins=edges )[0]
#     #plt.figure()
#     #plt.plot(hm1)
#     position[j][:] = pos
    
#     j +=1
# positionParticle = position


# import matplotlib 
# matplotlib.rcParams.update({'font.size': 23})




# def interWidth(z,d,w,w0):    
#     return w0*np.tanh((z-d)/w)

# gs=np.linspace(2,8,10)


# def error(hs):
#     #np.sum(x[:,5:8])
    
#     #[h[h.shape[0]%chunk:][i*chunk:(i+1)*chunk] for i in range(h.shape[0]//chunk)]
#     error_den = np.std(hs)/np.sqrt(len(hs))
#     return error_den

# positionParticle=position
# nFrames = positionParticle.shape[0]
# Lx,Ly,Lz = s[0].configuration.box[:3]
# Lx,Ly,Lz


# grid_size = []
# list_width_avg=[]

# width_error=[]
# for gs in (np.linspace(2,8,10)):
#     print(gs)
#     # Determine the dimensions of the smaller grids
#     grid_size_x = gs
#     grid_size_y = gs

#     # Calculate the number of grids in each dimension
#     num_grids_x = int(Lx / grid_size_x)
#     num_grids_y = int(Ly / grid_size_y)

#     n_chains1 = 300
#     nm1 = 140

#     n_chains2 = 100
#     nm2 = 140

#     n_chains3 = 200
#     nm3 = 71


#     histPctd = np.zeros((nFrames, edges.shape[0]-1 )) 
#     histctd = np.zeros((nFrames, edges.shape[0]-1 )) 
#     histhrd = np.zeros((nFrames, edges.shape[0]-1 )) 
    
#     hm1=np.zeros((num_grids_x*num_grids_y, edges.shape[0]-1 ))
#     hm2=np.zeros((num_grids_x*num_grids_y, edges.shape[0]-1 ))
#     hm3=np.zeros((num_grids_x*num_grids_y, edges.shape[0]-1 ))
#     list_width=[]
#     for nframe in range (nFrames):

#         #print(nframe)
#         # Iterate over the grids and plot a histogram for each grid
#         count = 0
#         for i in range(num_grids_x):
#             for j in range(num_grids_y):

#                 # Calculate the starting and ending coordinates of the grid
#                 x_start = -Lx/2+i * grid_size_x
#                 x_end = x_start + grid_size_x
#                 y_start =-Ly/2+ j * grid_size_y
#                 y_end = y_start + grid_size_y

#                 #print(i,j,x_start,x_end,y_start,y_end)


#                 particles = positionParticle[nframe][:n_chains1*nm1]
#                 # Find the particles within the grid
#                 particles_in_grid_ctd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
#                                               (y_start <= particles[:, 1]) & (particles[:, 1] < y_end) ]


#                 particles = positionParticle[nframe][n_chains1*nm1:n_chains1*nm1+n_chains2*nm2]
#                 # Find the particles within the grid
#                 particles_in_grid_pctd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
#                                               (y_start <= particles[:, 1]) & (particles[:, 1] < y_end) ]



#                 particles = positionParticle[nframe][n_chains1*nm1+n_chains2*nm2:]
#                 # Find the particles within the grid
#                 particles_in_grid_hrd = particles[(x_start <= particles[:, 0]) & (particles[:, 0] < x_end) &
#                                               (y_start <= particles[:, 1]) & (particles[:, 1] < y_end) ]


#                 hm1 = np.histogram(particles_in_grid_ctd[:,2] ,bins=edges )[0]
#                 hm2 = np.histogram(particles_in_grid_pctd[:,2] ,bins=edges )[0]
#                 hm3 = np.histogram(particles_in_grid_hrd[:,2] ,bins=edges )[0]

#                 #         plt.plot( np.histogram(particles_in_grid_pctd[:,2] ,bins=edges )[0])
#                 #         plt.plot( np.histogram(particles_in_grid_ctd[:,2] ,bins=edges )[0] )
#                 #         plt.plot( np.histogram(particles_in_grid_hrd[:,2] ,bins=edges )[0] )
#                 #print(particles_in_grid.shape )
#                 #plt.plot(hm2)
#                 #plt.plot(hm1)
                
                
#                 count +=1
#                 z = edges[:-1]+dz
#                 try:
                    
#                     y = -(hm3+hm2-hm1)/(hm3+hm1+hm2)

#                     ind = ~np.isnan(y)

#                     x = z[ind]
#                     y = y[ind]

#                     #x_min = np.where( y == y.max())[0]
#                     #x_min_middle = x_min[x_min.size // 2]

#                     #ind=x<x[x_min_middle]

#                     #y=y[ind]
#                     #x = x[ind]

#                     x=np.linspace(-x.shape[0],0,x.shape[0])
                   
#                     # The actual curve fitting happens here
#                     optimizedParameters1, pcov = opt.curve_fit(interWidth, x, y) 
                    
#                     #########
#                     #########
#                     #########

#                     y = -(hm3+hm2-hm1)/(hm3+hm1+hm2)

#                     ind = ~np.isnan(y)

#                     x = z[ind]
#                     y = y[ind]
#                     #plt.plot(x,y)

#                     #x_min = np.where( y == y.max())[0]
#                     #x_min_middle = x_min[x_min.size // 2]

#                     #ind=x>x[x_min_middle]

#                     #y=y[ind]
#                     #x = x[ind]

#                     x=np.linspace(0,x.shape[0],x.shape[0])

#                     # The actual curve fitting happens here
#                     optimizedParameters2, pcov = opt.curve_fit(interWidth, x, y)  
                    
#                     if ((optimizedParameters2[1]<10) & (optimizedParameters1[1]<10)):
#                         list_width.append((( (optimizedParameters1[1]+optimizedParameters1[1]))*.5)**2 )
                    
#                     #if ( np.array( list_width).mean() )>10:
#                     #print(nframe , optimizedParameters2[1],optimizedParameters1[1], np.array( list_width).mean() )
#                     # plt.figure()
#                     # plt.plot(x,y)
#                     # plt.plot(x, interWidth(x, *optimizedParameters2),label= np.array( list_width).mean() );
#                     # plt.legend()

#                 except:
#                     continue
                
#                 #print(np.array( list_width).mean())

        
#     width_error.append( error(np.array( list_width)))
#     print( np.array( list_width).mean() )
#     list_width_avg.append( np.array( list_width).mean() )

# np.savetxt('width_ctd_pctd_hrd.txt', list_width_avg, fmt='%d', delimiter='\t')

