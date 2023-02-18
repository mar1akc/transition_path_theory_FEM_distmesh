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
from scipy.sparse import diags


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
    G = np.zeros((3,2))
    G[:,0] = np.linalg.solve(Aux,rhs[:,0])
    G[:,1] = np.linalg.solve(Aux,rhs[:,1])
    M = 0.5*np.linalg.det(Aux)*np.matmul(G,np.transpose(G))
    return M

def stimavv(verts,fac):
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    det = np.absolute(np.linalg.det(Aux))              
    aux = det*fac/24
    M = aux*np.array([[2,1,1],[1,2,1],[1,1,2]])
    return M

def stima_Langevin(verts):
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    det = np.absolute(np.linalg.det(Aux))              
    G = np.array([verts[2,0]-verts[1,0],verts[0,0]-verts[2,0],verts[1,0]-verts[0,0]])
    G = np.reshape(G,(3,1))
    M = (0.5/det)*np.matmul(G,np.transpose(G))
    return M
    

def stimavbdv(verts,b1,b2):
    bdv1 = b1*np.array([verts[1,1]-verts[2,1],verts[2,1]-verts[0,1],verts[0,1]-verts[1,1]])
    bdv2 = b2*np.array([verts[2,0]-verts[1,0],verts[0,0]-verts[2,0],verts[1,0]-verts[0,0]]) 
    bdv = np.reshape(bdv1+bdv2,(1,3))
    M = (1/6)*np.concatenate((bdv,bdv,bdv),axis = 0)
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


def FEM_committor_solver_VarCoeff(pts,tri,Aind,Bind,fpot,VarCoeff,beta):
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
        fac = np.exp(-beta*fpot(vmid[:,0],vmid[:,1]))
        VC = VarCoeff(vmid[:,1])
        ind = tri[j,:]
        indt = np.array(ind)[:,None]
        B = csr_matrix((3,3),dtype = np.float).toarray()
        A[indt,ind] = A[indt,ind] + stima3(v)*fac*VC**2

    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
    return q



def FEM_committor_solver_irreversible(pts,tri,Aind,Bind,fpot,divfree,beta):
    # solves for committor for dX = b(X)dt + sqrt(epsilon)dW
    # eps05 = epsilon*0.5
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
        ind = tri[j,:]
        verts = pts[ind,:] # vertices of mesh triangle
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        fac = np.exp(-beta*fpot(vmid))
        f1,f2 = divfree(vmid[:,0],vmid[:,1])
        indt = np.array(ind)[:,None]
        A[indt,ind] = A[indt,ind] + stima3(verts)*fac - beta*fac*stimavbdv(verts,f1,f2)

    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
    return q

def FEM_committor_solver_Langevin(pts,tri,Aind,Bind,fpot,divfree,beta,gamma):
    # solves for committor for 
    # dx = p*dt, 
    # dp = (-V_x - \gamma*p)dt + \sqrt{2\beta^{-1}\gamma}dW
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
        ind = tri[j,:]
        verts = pts[ind,:] # vertices of mesh triangle
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        fac = gamma*np.exp(-beta*fpot(vmid))
        f1,f2 = divfree(vmid[:,0],vmid[:,1])
        indt = np.array(ind)[:,None]
        A[indt,ind] = A[indt,ind] + stima_Langevin(verts)*fac - beta*fac*stimavbdv(verts,f1,f2)

    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
    return q

def FEM_backward_committor_solver_Langevin(pts,tri,Aind,Bind,fpot,divfree,Hamiltonian,beta,gamma):
    # solves for committor for 
    # dx = p*dt, 
    # dp = (-V_x - \gamma*p)dt + \sqrt{2\beta^{-1}\gamma}dW
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Dir_bdry = np.hstack((Aind,Bind))
    free_nodes = np.setdiff1d(np.arange(0,Npts,1),Dir_bdry,assume_unique=True)

    A = csr_matrix((Npts,Npts), dtype = np.float).toarray()
    b = np.zeros((Npts,1))
    q = np.zeros((Npts,1))
    q[Aind] = 1

    # stiffness matrix
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:] # vertices of mesh triangle
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        fac = gamma*np.exp(-beta*fpot(vmid))
        f1,f2 = divfree(vmid[:,0],vmid[:,1])
        indt = np.array(ind)[:,None]
        A[indt,ind] = A[indt,ind] + stima_Langevin(verts)*fac - beta*fac*stimavbdv(verts,f1,f2)

    # invariant measure
    mu = np.exp(-beta*Hamiltonian(pts[:,0],pts[:,1]))
    Mu = diags(mu,0,shape=(Npts, Npts)).toarray()
    Mu_inv = diags(1.0/mu,0,shape=(Npts, Npts)).toarray()
    A = np.matmul(Mu_inv,np.matmul(A,Mu))
    
    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
   
    return q
    
    


def reactive_current_and_transition_rate(pts,tri,fpot,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
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

def reactive_current_and_transition_rate_VarCoeff(pts,tri,fpot,VarCoeff,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
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
        mu = np.exp(-beta*fpot(vmid[:,0],vmid[:,1]))
        VC = VarCoeff(vmid[:,1])
        VC2 = VC**2
        Rcurrent[j,:] = mu*VC2*g
        Rrate = Rrate + np.sum(g**2)*VC2*mu*tri_area                     
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


def probability_reactive(pts,tri,fpot,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    prob = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        qmid = np.sum(qtri)/3
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        prob = prob + tri_area*mu*qmid*(1-qmid)
    prob = prob/Z
    return prob

def probability_last_A(pts,tri,pts_Amesh,tri_Amesh,fpot,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Npts_Amesh = np.size(pts_Amesh,axis=0)
    Ntri_Amesh = np.size(tri_Amesh,axis=0)

    # find the reactive current and the transition rate
    prob = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        qmid = np.sum(qtri)/3
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        prob = prob + tri_area*mu*(1-qmid)
    for j in range(Ntri_Amesh):
        ind = tri_Amesh[j,:]
        verts = pts_Amesh[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        prob = prob + tri_area*mu
    prob = prob/Z
    return prob

def invariant_pdf(pts,tri,pts_Amesh,tri_Amesh,pts_Bmesh,tri_Bmesh,fpot,beta):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Npts_Amesh = np.size(pts_Amesh,axis=0)
    Ntri_Amesh = np.size(tri_Amesh,axis=0)
    Npts_Bmesh = np.size(pts_Bmesh,axis=0)
    Ntri_Bmesh = np.size(tri_Bmesh,axis=0)

    # find the reactive current and the transition rate
    Z = 0
    prob = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        Z = Z + tri_area*mu
    for j in range(Ntri_Amesh):
        ind = tri_Amesh[j,:]
        verts = pts_Amesh[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        Z = Z + tri_area*mu
    for j in range(Ntri_Bmesh):
        ind = tri_Bmesh[j,:]
        verts = pts_Bmesh[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        Z = Z + tri_area*mu
    return Z

def reactive_current_transition_rate_Langevin(pts,tri,fpot,divfree,beta,gamma,q,qminus,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        qmtri = qminus[ind]
        a = np.array([[verts[1,0]-verts[0,0],verts[1,1]-verts[0,1]],[verts[2,0]-verts[0,0],verts[2,1]-verts[0,1]]])
        b = np.array([qtri[1]-qtri[0],qtri[2]-qtri[0]])
        bm = np.array([qmtri[1]-qmtri[0],qmtri[2]-qmtri[0]])
        g = np.linalg.solve(a,b)
        gm = np.linalg.solve(a,bm)
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(2,)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid[0],vmid[1])) 
        qmid = np.sum(qtri)/3
        qmmid = np.sum(qmtri)/3
        Rcurrent[j,:] = mu*qmid*qmmid*np.array(divfree(vmid[0],vmid[1]))
        Rcurrent[j,1] = Rcurrent[j,1] + mu*(gamma/beta)*(qmmid*g[1]-qmid*gm[1])
        Rrate = Rrate + g[1]**2*mu*tri_area                     
    Rrate = Rrate*gamma/(Z*beta)
    Rcurrent = Rcurrent/Z 
    # map reactive current on vertices
    Rcurrent_verts = np.zeros((Npts,2))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        Rcurrent_verts[indt,:] = Rcurrent_verts[indt,:] + Rcurrent[j,:]
        tcount[indt] = tcount[indt] + 1
    Rcurrent_verts = Rcurrent_verts/np.concatenate((tcount,tcount),axis = 1)
    return Rcurrent_verts, Rrate
