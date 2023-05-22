# transition_path_theory_FEM_distmesh
This package allows you to compute the key descriptors of transition processes between two disjont regions, A and B, of the phase space of stochastic systems evolving according to 2D SDEs. These descriptors comprise the framework of the **transition path theory** (W. E and E. Vanden-Eijnden, 2006, 2010):
--> the forward and backward committors,
--> the probability density of reactive trajectories,
--> the reactive currect,
--> the transition rate,
--> the probability that an infinitely long trajectory is reactive at a randomly chosen moment of time,
--> the probability that an infinitely long trajectory at a randomly chosen moment of time last hit A rather than B.

The committors are computed using finite element method. The finite element method rewritten on python from:
**[1]** Title: Remarks around 50 lines of Matlab: short finite element implementation
Authors: Jochen Alberty, Carsten Carstensen and Stefan A. Funken
Journal: Numerical Algorithms 20 (1999) 117–137
https://www.math.hu-berlin.de/~cc/cc_homepage/download/1999-AJ_CC_FS-50_Lines_of_Matlab.pdf

The finite element committor solver and other functions for computing the functions and numbers above are found in 
**FEM_TPT.py**

The mesh generator is Per-Olof Persson's distmesh algorithm rewritten on python:
**[2]** http://persson.berkeley.edu/distmesh/
All functions relevant to mesh generation are found in
**distmesh.py**

**Test problems**
(1) The overdampded Langevin dynamics in the face potential: **face_TPT_drive.ipynb**.
(2) The overdampded Langevin dynamics in Mueller's potential: **Mueller_TPT_driver.ipynb**.
(3) The noisy bistable Duffing oscillator (the Langevin dynamics): **Duffing_TPT_driver.ipynb**.
(4) The Lennard-Jones-7 in the 2D space (the overdamped Langevin dynamics in collective variables, the second and thrid cental moments of the coordination numbers): **LJ7_inCV_TPT_driver.ipynb**. Auxiliary files: **helpers.py**. Input data: folder LJ7data. The mesh is generated using Darren Engwirda's algorithm mesh2d (search GitHub). The free energy and the diffusion matrix are computed by Luke Evans. 

Test problems (2), (3), and (4) are worked out for the paper:
**[3]** Title: Optimal control for sampling the transition path process and estimating rates.
Authors: Jiaxin Yuan, Amar Shah, Channing Bentz, Maria Cameron
ArXiv: TBA (will be available soon)

**Settings fror the overdamped Langevin dynamics**
The background on transition path theory is also found in [3]. Here we lay out the settings for the simplest case, the overdamped Langevin dynamics.

We consider the overamped Langevin dynamics 

$$dX = - \nabla V(X)dt + \sqrt{2\beta^{-1}}dW $$

in the domain with a reflecting boundary given by $$D: = \left(x : V(x) \le V_{\rm bdry}\right).$$

The invariant pdf is the Gibbs density $$\mu(x) = Z^{-1} e^{\beta V(x)},~~ Z = \int_{D} e^{-\beta V}dx.$$

The Bondary Value Problem for the committor function:
$$\nabla \cdot \left( e^{-\beta V(x)} \nabla q(x)\right) = 0, ~~ x \in D \backslash (A\cup B)$$

$$q(x) = 0,~~ x \in A$$

$$q(x) = 1, ~~ x \in B$$

**Important settings in face_TPT_driver.ipynb:**
**(The settings in other driver codes are similar)**
The Face potential has four local minima: the "eyes" are the deepest minima, and the "nose" and the "mouth" are shallower minima that can be considered as a dynamical trap.

--> xa,ya,ra,xb,yb,rb: The sets A and B are circles around the eyes centered at  (xa,ya), (xb,yb), of radii ra and rb, respectively.

--> Vbdry: The outer boundary is the level set of the potential {(x,y) : V(x,y) = Vbdry}. 

--> h0: The important parameter that determines how fine is the mesh is h0, the desired length of mesh edge.

--> generate_mesh: The boolean variable generate_mesh determines whether the mesh needs to be generated or read from the files. Set generate_mesh = True for the first run. The generated mesh will be saved to files. If you want to experiment further with the same mesh, set generate_mesh = False. Then it will be read from the files.

--> q: The variable q is the committor at the mesh points.

--> Rcurrent: The variable Rcurrent is Npts-by-2 array with components of the reactive current at the mesh points. The reactive current is given by:
$$ J_R = Z^{-1}\beta^{-1}e^{-\beta V(x)}\nabla q(x).$$

--> Rrate: The variable Rrate is the transition rate given by:
$$\nu_{AB} = Z^{-1}\beta^{-1}\int_{D\backslash(A\cup B)} \|\nabla q\|^2e^{-\beta V} dx.$$
