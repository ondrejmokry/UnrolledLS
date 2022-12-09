import numpy as np
from scipy.spatial import Voronoi

def DCF(kx, ky):
    """
    Calculate the density compensation function
    """
    # get rid of the repeated k-space locations, such as (0,0)
    K = np.column_stack((kx, ky))
    K, indices, counts = np.unique(K, axis=0, return_inverse=True, return_counts=True)

    # compute the Voronoi tessellation
    vor = Voronoi(K, qhull_options='Qbb')

    # compute the areas
    areas = np.zeros(len(K))
    for j in range(len(K)):
        # take out the infinity point (denoted by -1)
        if -1 in vor.regions[j]:
            vor.regions[j].remove(-1)
        # if there are at least 3 vertices left, compute the area
        if len(vor.regions[j]) > 2:
            x = vor.vertices[vor.regions[j],0]
            y = vor.vertices[vor.regions[j],1]
            ind = [*range(1,len(vor.regions[j])), 0]
            areas[j] = np.absolute(np.sum(0.5*(x * y[ind] - x[ind] * y)))

    # reorder the areas such that it corresponds to the order of the points in K
    areas = areas[vor.point_region]

    # if the region corresponds to more input points, devide the area
    for j in range(len(K)):
        if counts[j] > 1:
            areas[j] = areas[j]/counts[j]

    # reorder once more (putting the repeated values back)
    return areas[indices]
