import numpy as np
from random import choices, choice
from collections import defaultdict
import itertools
from scipy.stats import truncnorm


class Point:
    def __init__(self, x, y, point_id, level_ids=None):
        self.x = x
        self.y = y
        if level_ids:
            self.level_ids = level_ids
        else:
            level_ids = (point_id,)

    @property
    def id(self):
        return self.level_ids[0]

    def __str__(self):
        return f"{self.level_ids}"

    def __repr__(self):
        return f"Point({self.id:02}, {self.x}, {self.y}, {self.level_ids})"

    def __hash__(self):
        return hash((self.x, self.y))

    def id_level(self, i):
        try:
            return self.level_ids[i]
        except:
            raise IndexError(f'Point {self} has no level "{i}".')


def get_point_list(rows, cols, levels=None):

    if levels:
        point_level_ids = get_cell_level_zone_ids(rows, cols, levels)

        point_list = [
            Point(*pos, pos_id, level_ids=point_level_ids[pos_id])
            for pos_id, pos in enumerate(
                [(i, j) for i in range(rows) for j in range(cols)]
            )
        ]
    else:
        # Creating all points
        point_list = [
            Point(*pos, pos_id)
            for pos_id, pos in enumerate(
                [(i, j) for i in range(rows) for j in range(cols)]
            )
        ]

    return point_list


def get_neighbor_zones(center, max_range, zone_grid):
    # get_pep.cache_info()
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
        min(rows - 1, center.x + max_range),
    )
    min_y, max_y = (
        max(0, center.y - max_range),
        min(cols - 1, center.y + max_range),
    )

    # print("X:", min_x, max_x)
    # print("Y:", min_y, max_y)

    # Slice zone_grid to extract neighboring zones around center
    neighbors = zone_grid[min_x : max_x + 1, min_y : max_y + 1]

    # Return list of neighbor zones
    return np.ravel(neighbors)


def get_aggregated_zones(rows, cols, n_levels):
    """Return 'n_levels' lists of 2D grids with ids aggregated in g levels
    with g in [0, n_levels].

    The aggregation is done on the zones where every 2^(2*g) zones form an
    area. All areas are disjoint at each aggregation level.

    Arguments:
        rows {int} -- Grid total number of rows
        cols {int} -- Grid total number of columns
        levels {int} -- Number of aggregation levels

    Returns:
        list(np.array((max_rows, max_cols))) -- list of 2D areas with
        ids belonging to each aggregated level.

    """

    if n_levels <= 0:
        raise (Exception("Aggregation levels have to be higher than 0."))

    if rows <= 0:
        raise (Exception("Number of rows have to be higher than 0."))

    if cols <= 0:
        raise (Exception("Number of columns have to be higher than 0."))

    level_grid_list = list()

    for g in range(n_levels):

        area_dim = 2 ** g
        grid = np.zeros((rows, cols), dtype=int)
        area_row_count = np.math.ceil(rows / area_dim)
        area_col_count = np.math.ceil(cols / area_dim)

        # print(
        #     f'\n################# Areas of size {area_dim}X{area_dim} ############'
        #     f'\n - Rows [count={area_row_count:>4},'
        #     f' total={area_row_count*area_dim:>4}]'
        #     f'\n - Cols [count={area_col_count:>4},'
        #     f' total={area_col_count*area_dim:>4}]'
        # )

        area_id = 0

        # Loop all areas
        for r in range(area_row_count):
            for c in range(area_col_count):

                # Limits area and assign area id
                from_x, to_x = area_dim * (r), min(rows, area_dim * (r + 1))
                from_y, to_y = area_dim * (c), min(cols, area_dim * (c + 1))
                grid[from_x:to_x, from_y:to_y] = area_id
                area_id += 1

        # List of zones per aggregate level g
        level_grid_list.append(grid)

        # print(grid)

    return level_grid_list


def get_cell_level_zone_ids(rows, cols, n_levels):
    """For each cell in a gridmap, associate a tuple with
    the ids of this cell under "n_levels" aggregation levels.
    
    Arguments:
        rows {int} -- Grid number of rows
        cols {int} -- Grid number of columns
        n_levels {int} -- Number of levels
    
    Returns:
        list(tuple) -- List of zone ids according to aggregation level.
    """

    level_grid_list = get_aggregated_zones(rows, cols, n_levels)
    level_cell_zone_ids = [np.ravel(z) for z in level_grid_list]

    # id -> tuple
    cell_level_zone_ids = list(zip(*level_cell_zone_ids))

    return cell_level_zone_ids
