# This code is a python adaptation of the code in the paper:
# Title: Remarks around 50 lines of Matlab: short finite element implementation
# Authors: Jochen Alberty, Carsten Carstensen and Stefan A. Funken
# Journal: Numerical Algorithms 20 (1999) 117–137

# Solves the committor problem using the FEM
# grad * (exp(-beta V(x,y) grad q(x,y)) = 0, q(bdry A) = 0, q(bdry B) = 1
# the potential used is the Face potential defined in this cell

import numpy as np
import math
import matplotlib.pyplot as plt
import csv 
import scipy
from scipy.sparse import csr_matrix


def put_pts_on_circle(xc,yc,r,n):
    t = np.linspace(0,math.pi*2,n+1)
    pts = np.zeros((n,2))
    pts[:,0] = xc+r*np.cos(t[0:n])
    pts[:,1] = yc+r*np.sin(t[0:n])
    return pts

def reparametrization(path,h):
    dp = path - np.roll(path,1,axis = 0);
    dp[0,:] = 0;
    dl = np.sqrt(np.sum(dp**2,axis=1));
    lp = np.cumsum(dl);
    len = lp[-1];
    lp = lp/len; # normalize
    npath = int(round(len/h));
    g1 = np.linspace(0,1,npath)
    path_x = np.interp(g1,lp,path[:,0])
    path_y = np.interp(g1,lp,path[:,1])
    path = np.zeros((npath,2))
    path[:,0] = path_x
    path[:,1] = path_y
    return path

def find_ABbdry_pts(pts,xc,yc,r,h0):
    ind = np.argwhere(np.sqrt((pts[:,0]-xc)**2+(pts[:,1]-yc)**2)-r < h0*1e-2)
    Nind = np.size(ind)
    ind = np.reshape(ind,(Nind,))
    return Nind,ind

def stima3(verts):
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    rhs = np.zeros((3,2))
    rhs[1,0] = 1
    rhs[2,1] = 1
    G = np.zeros((3,3))
    G[:,0] = np.linalg.solve(Aux,rhs[:,0])
    G[:,1] = np.linalg.solve(Aux,rhs[:,1])
    M = 0.5*np.linalg.det(Aux)*np.matmul(G,np.transpose(G))
    return M

def FEM_committor_solver(pts,tri,Aind,Bind,fpot,beta):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Dir_bdry = np.hstack((Aind,Bind))
    free_nodes = np.setdiff1d(np.arange(0,Npts,1),Dir_bdry,assume_unique=True)

    A = csr_matrix((Npts,Npts), dtype = np.float).toarray()
    b = np.zeros((Npts,1))
    q = np.zeros((Npts,1))
    q[Bind] = 1

    # stiffness matrix
    for j in range(Ntri):
        v = pts[tri[j,:],:] # vertices of mesh triangle
        vmid = np.reshape(np.sum(v,axis=0)/3,(1,2)) # midpoint of mesh triangle
        fac = np.exp(-beta*fpot(vmid))
        ind = tri[j,:]
        indt = np.array(ind)[:,None]
        B = csr_matrix((3,3),dtype = np.float).toarray()
        A[indt,ind] = A[indt,ind] + stima3(v)*fac

    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
    return q

def reactive_current_and_transition_rate(pts,tri,fpot,beta,q):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
    Z = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        a = np.array([[verts[1,0]-verts[0,0],verts[1,1]-verts[0,1]],[verts[2,0]-verts[0,0],verts[2,1]-verts[0,1]]])
        b = np.array([qtri[1]-qtri[0],qtri[2]-qtri[0]])
        g = np.linalg.solve(a,b)
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        Z = Z + tri_area*mu
        Rcurrent[j,:] = mu*g
        Rrate = Rrate + np.sum(g**2)*mu*tri_area                     
    Rrate = Rrate/(Z*beta)
    Rcurrent = Rcurrent/(Z*beta) 
    # map reactive current on vertices
    Rcurrent_verts = np.zeros((Npts,2))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        Rcurrent_verts[indt,:] = Rcurrent_verts[indt,:] + Rcurrent[j,:]
        tcount[indt] = tcount[indt] + 1
    Rcurrent_verts = Rcurrent_verts/np.concatenate((tcount,tcount),axis = 1)
    return Rcurrent_verts, Rrate
