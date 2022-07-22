#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:34:21 2020
/*
 *   @author: kissami
 */
"""
import numpy as np
from mpi4py import MPI
from psydac.ddm.partition import compute_dims

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()

nb_neighbours = 4
N = 0
E = 1
S = 2
W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)
ntx = 6
nty = 4

Nx = ntx + 2
Ny = nty + 2

npoints = [ntx, nty]
p1 = [2, 2]
P1 = [False, False]
reorder = True

coef = np.zeros(3)
''' Grid spacing '''
hx = 1 / (ntx + 1.);
hy = 1 / (nty + 1.);

''' Equation Coefficients '''
coef[0] = (0.5 * hx * hx * hy * hy) / (hx * hx + hy * hy);
coef[1] = 1. / (hx * hx);
coef[2] = 1. / (hy * hy);


def create_2d_cart(npoints, p1, P1, reorder):
    # Store input arguments
    npts = tuple(npoints)
    pads = tuple(p1)
    periods = tuple(P1)
    reorder = reorder

    nprocs, block_shape = mpi_compute_dims(nb_procs, npts, pads)

    dims = nprocs

    if (rank == 0):
        print("Execution poisson with", nb_procs, " MPI processes\n"
                                                  "Size of the domain : ntx=", npoints[0], " nty=", npoints[1], "\n"
                                                                                                                "Dimension for the topology :",
              dims[0], " along x", dims[1], " along y\n"
                                            "-----------------------------------------")

    '''
    * Creation of the Cartesian topology
    '''
    cart2d = comm.Create_cart(dims=dims, periods=periods, reorder=reorder)
    return dims, cart2d


def create_2dCoords(cart2d, npoints, dims):
    ''' Create 2d coordinates of each process'''

    centre = [npoints[0] / dims[0], npoints[1] / dims[1]]

    coords = cart2d.Get_coords(rank)

    sx = coords[0] * centre[0]
    ex = centre[0] * (coords[0] + 1)

    sy = coords[1] * centre[1]
    ey = centre[1] * (coords[1] + 1)

    print("Rank in the topology :", rank, " Local Grid Index :", sx, " to ", ex, " along x, ",
          sy, " to", ey, " along y")

    return sx, ex, sy, ey


def create_neighbours(cart2d):
    ''' Get my northern and southern neighbours '''

    neighbour[S], neighbour[N] = cart2d.Shift(direction=1, disp=1)

    ''' Get my western and eastern neighbours '''

    neighbour[W], neighbour[E] = cart2d.Shift(direction=0, disp=1)

    print("Process", rank, " neighbour: N", neighbour[N], " E", neighbour[E], " S ", neighbour[S], " W", neighbour[W])

    return neighbour


'''Creation of the derived datatypes needed to exchange points with neighbours'''


def create_derived_type(sx, ex, sy, ey):
    '''Creation of the type_line derived datatype to exchange points
    with northern to southern neighbours '''

    type_ligne = 0

    '''Creation of the type_column derived datatype to exchange points
    with western to eastern neighbours '''

    type_column = 0

    return type_ligne, type_column

''' Exchange the points at the interface '''

''' Exchange the points at the interface '''


def communications(u, sx, ex, sy, ey, type_column, type_ligne):
    ''' Send to neighbour N and receive from neighbour S '''

    if (neighbour[N] > -1):   # Verifier si le processus a un voisin en nord

        sendbuff = u[IDX(sx - 1, ey):IDX(ex + 1, ey) + 1:ey - sy + 3]
        comm.Send(sendbuff, neighbour[N])

        recvbuff = np.zeros(ex - sx + 3)
        comm.Recv(recvbuff, neighbour[N])
        u[IDX(sx - 1, sy - 1):IDX(ex + 1, sy - 1) + 1:ey - sy + 3] = recvbuff

    ''' Send to neighbour S and receive from neighbour N '''

    if (neighbour[S] > -1):   # Verifier si le processus a un voisin en sud

        sendbuff = u[IDX(sx - 1, sy):IDX(ex + 1, sy) + 1:ey - sy + 3]
        comm.Send(sendbuff neighbour[S])

        recvbuff = np.zeros(ex - sx + 3)
        comm.Recv(recvbuff, neighbour[S])
        u[IDX(sx - 1, ey + 1):IDX(ex + 1, ey + 1) + 1:ey - sy + 3] = recvbuff

    ''' Send to neighbour W and receive from neighbour E '''

    if (neighbour[W] > -1):   # Verifier si le processus a un voisin en ouest

        sendbuff = u[IDX(sx, sy - 1):IDX(sx, ey + 1) + 1]
        comm.Send(sendbuff, neighbour[W])

        recvbuff = np.zeros(ey - sy + 3)
        comm.Recv(recvbuff, neighbour[W])
        u[IDX(sx - 1, sy - 1):IDX(sx - 1, ey + 1) + 1] = recvbuff

    ''' Send to neighbour E  and receive from neighbour W '''

    if (neighbour[E] > -1):   # Verifier si le processus a un voisin en est

        sendbuff = u[IDX(ex, sy - 1):IDX(ex, ey + 1) + 1]
        comm.Send(sendbuff, neighbour[E])

        recvbuff = np.zeros(ey - sy + 3)
        comm.Recv(recvbuff, neighbour[E])
        u[IDX(ex + 1, sy - 1):IDX(ex + 1, ey + 1) + 1] = recvbuff



'''
 * IDX(i, j) : indice de l'element i, j dans le tableau u
 * sx-1 <= i <= ex+1
 * sy-1 <= j <= ey+1
'''


def IDX(i, j):
    return (((i) - (sx - 1)) * (ey - sy + 3) + (j) - (sy - 1))


def initialization(sx, ex, sy, ey):
    ''' Grid spacing in each dimension'''
    ''' Solution u and u_new at the n and n+1 iterations '''

    SIZE = (ex - sx + 3) * (ey - sy + 3)

    u = np.zeros(SIZE)
    u_new = np.zeros(SIZE)
    f = np.zeros(SIZE)
    u_exact = np.zeros(SIZE)

    '''Initialition of rhs f and exact soluction '''

    for i in range(sx-1, ex+1):
        x = i * hx
        for j in range(sy-1, ey+1):
            y = j * hy
            f[IDX(i, j)] = 2 * (x * x - x + y * y - y)
            u_exact[IDX(i,j)] = x * y * (x-1) * (y-1)


    return u, u_new, u_exact, f


####################################  Solving the advection diffusion equation using Finite difference ##############

def burger(u,vn,nu,dt):
    u_burger = np.copy(u)
    for i in range(sx, ex + 1):
        for j in range(sy, ey + 1):
            u_burger[IDX(i, j)] = (u[IDX(i, j)] - dt / hx * u[IDX(i, j)] * (
                        u[IDX(i, j)] - u[IDX(i - 1, j)]) - dt / hy * vn[IDX(i, j)] * (
                                           u[IDX(i, j)] - u[IDX(i, j - 1)]) +
                               nu * dt / hx ** 2 * (u[IDX(i + 1, j)] - 2 * u[IDX(i, j)] + u[
                        IDX(i - 1, j - 1)]) + nu * dt / hy ** 2 * (u[IDX(i, j + 1)] - 2 * u[IDX(i, j)] + u[IDX(i - 1, j - 1)]))
    return u_burger
######################################################################################################################




####################################  Solving Poisson equation using Finite difference and Jacobi’s iterative ##############
def poisson(u,u_new):
    for i in range(sx, ex + 1):
        for j in range(sy, ey + 1):
            u_new[IDX(i, j)] = coef[0] * (coef[1] * (u[IDX(i + 1, j)] + u[IDX(i - 1, j)]) +
                    coef[2] * (u[IDX(i, j + 1)] + u[IDX(i, j - 1)]) - f[IDX(i, j)])
    return u_new
###########################################################################################################################

####################################  Coupling both to solve the NS equation (already done using Numba and Pyccel) ##############
def computation(u,u_new):
    ''' Compute the new value of u using

    '''

    vn = np.ones(len(u))
    sigma = .0009
    nu = 0.01
    dt = sigma * hx * hy / nu

    u_new = poisson(burger(u, vn, nu, dt),u_new)

#################################################################################################

def output_results(u, u_exact):
    print("Exact Solution u_exact - Computed Solution u - difference")
    for itery in range(sy, ey + 1, 1):
        print(u_exact[IDX(1, itery)], '-', u[IDX(1, itery)], u_exact[IDX(1, itery)] - u[IDX(1, itery)]);


''' Calcul for the global error (maximum of the locals errors) '''


def global_error(u, u_new):
    local_error = 0

    for iterx in range(sx, ex + 1, 1):
        for itery in range(sy, ey + 1, 1):
            temp = np.fabs(u[IDX(iterx, itery)] - u_new[IDX(iterx, itery)])
            if local_error < temp:
                local_error = temp;

    return local_error


import meshio


def plot_2d(f):
    f = np.reshape(f, (ex - sx + 3, ey - sy + 3))

    x = np.linspace(0, 1, ey - sy + 3)
    y = np.linspace(0, 1, ex - sx + 3)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, f, cmap=cm.viridis)

    plt.show()


dims, cart2d = create_2d_cart(npoints, p1, P1, reorder)
neighbour = create_neighbours(cart2d)

sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)

type_ligne, type_column = create_derived_type(sx, ex, sy, ey)
u, u_new, u_exact, f = initialization(sx, ex, sy, ey)

''' Time stepping '''
it = 0
convergence = False
it_max = 100000
eps = 2.e-16

''' Elapsed time '''
t1 = MPI.Wtime()

while (not (convergence) and (it < it_max)):
    it = it + 1;

    temp = u.copy()
    u = u_new.copy()
    u_new = temp.copy()

    ''' Exchange of the interfaces at the n iteration '''
    communications(u, sx, ex, sy, ey, type_column, type_ligne)

    ''' Computation of u at the n+1 iteration '''
    computation(u, u_new)

    ''' Computation of the global error '''
    local_error = global_error(u, u_new);
    diffnorm = comm.allreduce(np.array(local_error), op=MPI.MAX)

    ''' Stop if we obtained the machine precision '''
    convergence = (diffnorm < eps)

    ''' Print diffnorm for process 0 '''
    if ((rank == 0) and ((it % 100) == 0)):
        print("Iteration", it, " global_error = ", diffnorm);

''' Elapsed time '''
t2 = MPI.Wtime()

if (rank == 0):
    ''' Print convergence time for process 0 '''
    print("Convergence after", it, 'iterations in', t2 - t1, 'secs')

    ''' Compare to the exact solution on process 0 '''
    output_results(u, u_exact)