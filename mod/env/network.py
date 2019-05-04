
import numpy as np
from random import choices, choice
from collections import defaultdict
import itertools
from scipy.stats import truncnorm

class Point:

    def __init__(self, x, y, point_id):
        self.x = x
        self.y = y
        self.id = point_id
    
    def __str__(self):
        return f'{self.id}'
    
    def __repr__(self):
        return f'Point({self.id:02}, {self.x}, {self.y})'
    
    def __hash__(self):
        return hash((self.x, self.y))
    

def get_point_list(rows, cols):

    # Creating all points
    point_list = [
        Point(*pos, pos_id)
        for pos_id, pos in enumerate(
            [
                (i,j) for i in range(rows) for j in range(cols)
            ]
        )
    ]
    return point_list



def get_neighbor_zones(center, max_range, zone_grid):
    #get_pep.cache_info()
    """[summary]
    
    Arguments:
        center {Point} -- Center point
        max_range {int} -- Range of neighboring zones to select
        zone_grid {array} -- Numpy array representing the 
            zones (content is zone ids)
    
    Returns:
        list -- Ids of neighboring zones in max_range
    """
    
    # Derive rows and columns from numpy array
    rows, cols = zone_grid.shape

    # Garantee feasible bounderies
    min_x, max_x = (
        max(0, center.x - max_range),
        min(rows-1, center.x + max_range)
    )
    min_y, max_y = (
        max(0, center.y - max_range),
        min(cols-1, center.y + max_range)
    )

    #print("X:", min_x, max_x)
    #print("Y:", min_y, max_y)
    
    # Slice zone_grid to extract neighboring zones around center
    neighbors =  zone_grid[min_x:max_x+1,min_y:max_y+1]

    # Return list of neighbor zones
    return np.ravel(neighbors)