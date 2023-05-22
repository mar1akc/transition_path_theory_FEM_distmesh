# Module for LJ7 written by Luke Evans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

################################################################################
# Data Plotting Funtions
################################################################################
def plot_cov_ellipse(cov, x, plot_scale=1, color='k', plot_evecs=False,
                     quiver_scale=1):
    """Plots the ellipse corresponding to a given covariance matrix at x (2D) 

    Args:
        cov (array): 2 by 2 symmetric positive definite array.
        x (vector): 2 by 1 vector.
        scale (float, optional): Scale for ellipse size. Defaults to 1.
    """

    evals, evecs = np.linalg.eig(cov)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    t = np.linspace(0, 2*np.pi)
    val = plot_scale*evecs*evals
    if plot_evecs:
        plt.quiver(*x, *val[:, 0], color=color, angles='xy', scale_units='xy',
                    scale=quiver_scale, width=0.002)
        plt.quiver(*x, *val[:, 1], color=color, angles='xy', scale_units='xy',
                   scale=quiver_scale, width= 0.002)
    else:
        a = np.dot(val, np.vstack([np.cos(t), np.sin(t)]))
    
        plt.plot(a[0, :] + x[0], a[1, :] + x[1], linewidth=0.5, c=color)


def confidence_ellipse(cov, x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics
    
    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    # render plot with "plt.show()".

##############################################################
# Functions helping with committor error, plotting committor, etc.
##############################################################
def committor_contours(zeroset=1e-7, oneset=1.0):

    my_levels = np.arange(0.1, 0.6, 0.1)
    my_levels = np.concatenate((my_levels, np.arange(0.6, 1.0, 0.1)))    

    return my_levels 


def is_in_ABC(data, centerx_A, centery_A, rad_A, centerx_B, centery_B, rad_B):
    A_bool = is_in_circle(data[0, :], data[1, :], centerx_A, centery_A, rad_A)
    B_bool = is_in_circle(data[0, :], data[1, :], centerx_B, centery_B, rad_B)
    C_bool = np.logical_not(np.logical_or(A_bool, B_bool))
    return A_bool, B_bool, C_bool


def is_in_circle(x, y, centerx=0, centery=0, rad=1.0): 
    return ((x - centerx)**2 + (y-centery)**2 <= rad**2)

if __name__ == '__main__':
    main()


