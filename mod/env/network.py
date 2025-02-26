import random
import sys

import numpy as np

# tenv_mod = "c:/Users/breno/Documents/phd/tenv"
from mod.env.Point import Point

tenv_mod = "C:\\Users\\LocalAdmin\\OneDrive\\leap_forward\\street_network_server\\tenv"

sys.path.insert(0, tenv_mod)

import tenv.util as tenv

# Reproducibility of the experiments
random.seed(1)


def get_distance(o, d):
    return tenv.get_distance(o, d)


########################################################################
# Network grid #########################################################
########################################################################


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
    """Find all grid ids surrounding center in grid zone at a
    "max_range" distance.

    Arguments:
        center {Point} -- Center point
        max_range {int} -- Range of neighboring zones to select
        zone_grid {array} -- Numpy array representing the zones
        (content is zone ids)

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
    neighbors = zone_grid[min_x: max_x + 1, min_y: max_y + 1]

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


def get_demand_origin_centers(points, grid_area, n_centers, zone_size):
    """From a a list of points distributed in a grid area, choose a
    number of centers to be the demand origins. Requests can also depart
    from nodes in the neighboring zones of each center.

    Arguments:
        points {list} -- List of Point objects
        grid_area {narray} -- Grid area with point ids
        n_centers {int} -- How many centers to create
        zone_size {int} -- How many neighboring levels surrounding
            a center.

    Returns:
        list -- Origin points
    """

    # Choose random location
    random_centers = random.choices(points, k=n_centers)

    nodes = []
    for c in random_centers:
        nodes.extend(get_neighbor_zones(c, zone_size, grid_area))

    return [points[n] for n in nodes]


########################################################################
# Network map ##########################################################
########################################################################

# Network information is queried from server
import requests

port = 4999
url = f"http://localhost:{port}"


def query_point_list(
        max_levels=None, projection="GPS", step=60, level_dist_list=None
):
    # Associate nodes to region center ids
    if max_levels or level_dist_list:
        # Dictionary of levels(distances from region centers) and
        # corresponding node ids (in referrence to region centers)
        level_point_ids_list, distance_levels, level_count = query_aggregated_centers(
            step=step, n_levels=max_levels, dist_list=level_dist_list
        )

        # For each level, the set of point ids
        level_point_ids_set = [set(ids) for ids in level_point_ids_list]

        # Sort distances such that they correspond to levels
        # distance_levels = sorted(map(int, distance_levels))

        # Tuples with node ids for each level
        # e.g., [[0,1,2,3], [10,11,12,13], [20,21,22,23]]
        # [(0,10,20), (1,11,21), (2,12,22), (3,13,23)]
        p = list(zip(*level_point_ids_list))
        r = tenv.nodes(projection)
        nodes = {e["id"]: e for e in r["nodes"]}

        # Level zero correspond to node id
        point_list = [
            Point(
                nodes[pos[0]]["x"],
                nodes[pos[0]]["y"],
                pos[0],
                level_ids=list(pos),
                level_ids_dic={
                    int(d): pos[i] for i, d in enumerate(distance_levels)
                },
            )
            for pos in p
        ]
    else:
        # Creating all points, with no region center ids
        point_list = [
            Point(nodes[pos[0]]["x"], nodes[pos[0]]["y"], pos[0]) for pos in p
        ]

    point_list.sort(key=lambda p: p.id)
    # pprint.pprint(point_list)

    # pprint.pprint([p.level_ids_dic for p in point_list])
    return point_list, distance_levels, level_count, level_point_ids_set


# @functools.lru_cache(maxsize=None)
def query_neighbor_zones(center, distance, n_neighbors=4):
    """Return neighbor zone ids of center node.

    Parameters
    ----------
    center : int
        Region center id
    distance : int
        Distance that determines the region center set
    n_neighbors : int, optional
        How many neighbor zones to return, by default 4

    Returns
    -------
    list
        List of center neighbors
    """

    neighbors = tenv.get_center_neighbors(distance, center, n_neighbors)

    return neighbors


# @functools.lru_cache(maxsize=None)
def query_neighbors(node, reach=1):
    """Return neighbor zone ids of center node.

    Parameters
    ----------
    center : int
        Region center id
    distance : int
        Distance that determines the region center set
    n_neighbors : int, optional
        How many neighbor zones to return, by default 4

    Returns
    -------
    list
        List of center neighbors
    """

    neighbors = np.array(tenv.neighbors(node, reach, "forward"))

    return neighbors


# @functools.lru_cache(maxsize=None)
def query_level_neighbors(center, distance):
    """Get the elements belonging to region center of distance.

    Parameters
    ----------
    center : int
        Region center id
    distance : int
        Distance threshold that region center id belongs to

    Returns
    -------
    list
        All nodes in region center
    """

    neighbors = np.array(tenv.get_center_elements(distance, center))

    return neighbors


# # @functools.lru_cache(maxsize=None)
# def get_distance(o, d):
#     """Min. distance between origin o and destination d in kilometers

#     Parameters
#     ----------
#     o : int
#         Origin id
#     d : int
#         Destination id

#     Returns
#     -------
#     float
#         Distance in kilometers
#     """

#     return tenv.get_distance(o, d)
#     # url_distance = f"{url}/distance_meters/{o}/{d}"
#     # r = requests.get(url=url_distance)
#     # return float(r.text) / 1000.0


def query_aggregated_centers(n_levels=None, dist_list=None, step=60):
    """Return 'n_levels' lists of 1D lists with ids aggregated in g
    levels with g in [0, n_levels].

    Arguments:
        n_levels {[type]} -- [description]

    Keyword Arguments:
        step {int} -- [description] (default: {60})

    Returns:
        [type] -- [description]
    
    Example
    -------
    >>> 'http://localhost:4999/node_region_ids/60'
    { '0': [1360, 1030, 741, 742, 1383, 845, 328, 23, 329, ...],
     '60': [1087, 1336, 1917, 679, 181, 965, 860, 57, 339, ...],
    '120': [1083, 968, 684, 684, 1761, 968, 405, 74, 405, ...],
    '300': [1078, 144, 2006, 2006, 144, 1114, 2006, 1114, 2006, ...],
    '360': [1854, 1117, 1042, 1042, 1117, 1117, 1042, 1117, 1117, ...],
    '420': [1077, 1558, 1935, 1935, 1339, 1558, 1339, 1558, 1558, ...],
    '480': [1077, 1561, 1077, 1077, 1561, 1561, 1561, 1129, 1561, ...],
    '540': [1092, 1196, 1092, 1092, 1196, 1196, 1196, 1196, 1196, ...],
    '600': [587, 1195, 587, 587, 1195, 1195, 1195, 1195, 1195, ...]}
    """

    # Get all ids for each level
    node_region_ids = tenv.get_node_region_ids_step(step)

    if n_levels and n_levels <= 0:
        raise (Exception("Aggregation levels have to be higher than 0."))

    # Sort dictionary according to keys (increasing distance)
    sorted_levels = list(node_region_ids.items())
    sorted_levels.sort(key=lambda tp: int(tp[0]))

    level_grid_list = list()
    distances = list()

    i = 0
    # 0 = 0, 30 = 1, 60 = 2, 90 = 3, 120 = 4, 150 = 5, etc.
    for step_nodes in sorted_levels:
        distance, level_node_ids = step_nodes
        # print(f"\n\n##### Level={i:>2} - Step={step}")

        if dist_list and int(distance) not in dist_list:
            continue

        i += 1

        if n_levels and i > n_levels:
            break

        # List of zones per aggregate level g
        level_grid_list.append(np.array(level_node_ids))
        distances.append(int(distance))

    count_points_level = {k: len(set(n)) for k, n in node_region_ids.items()}

    return level_grid_list, distances, count_points_level


def query_centers(points, n_centers, level):
    """From a list of points distributed in a grid area, choose a
    number of centers to be the demand origins. Requests can also depart
    from nodes in the neighboring zones of each center.

    Arguments:
        points {list} -- List of Point objects]
        n_centers {int} -- How many centers to create
        zone_size {int} -- How many neighboring levels surrounding
            a center.

    Returns:
        list -- Origin points
    """

    # Choose random location
    random_centers = random.choices(points, k=n_centers)

    nodes = []
    for c in random_centers:
        nodes.extend(
            query_level_neighbors(c.id_level(level), Point.levels[level])
        )

    return [points[n] for n in nodes]


def query_info():
    info = tenv.get_info()

    center_count = info["centers"]
    edge_count = info["edge_count"]
    node_count = info["node_count"]
    region = info["region"]
    label = info["label"]
    region_type = info["region_type"]

    return region, label, node_count, center_count, edge_count, region_type


def query_sp_sliced(o, d, n_points, steps, projection="GPS", waypoint=None):
    if not waypoint:
        waypoint = o

    # query = (
    #     f"{url}/sp_sliced/{o.id}/{d.id}/{waypoint.id}/"
    #     f"{n_points}/{steps}/{projection}"
    # )

    sp_coords = tenv.sp_sliced(
        o.id, d.id, waypoint.id, n_points, steps, projection=projection
    )
    # print(query)
    # r = requests.get(url=query)
    # sp_coords = r.json()["sp"]

    # pprint.pprint(sp_coords)
    return sp_coords


def query_segmented_sp(
        o, d, n_points, step_duration, projection="GPS", waypoint=None
):
    if not waypoint:
        waypoint = o

    query = (
        f"{url}/sp_segmented/{o.id}/{d.id}/{waypoint.id}/"
        f"{n_points}/{step_duration}/{projection}"
    )

    # print(query)
    r = requests.get(url=query)
    sp_coords = r.json()["sp"]

    # pprint.pprint(sp_coords)
    return sp_coords


# @functools.lru_cache(maxsize=None)
def query_sp(o, d, projection="GPS", waypoint=None):
    if waypoint:
        # Get all ids for each level
        query_o_waypoint = f"{url}/sp/{o.id}/{waypoint.id}/{projection}"
        r_o_waypoint = requests.get(url=query_o_waypoint)
        sp_coords_o_waypoint = r_o_waypoint.json()["sp"]

        # Get all ids for each level
        query_waypoint_d = f"{url}/sp/{waypoint.id}/{d.id}/{projection}"
        r_waypoint_d = requests.get(url=query_waypoint_d)
        sp_coords_waypoint_d = r_waypoint_d.json()["sp"]

        return sp_coords_o_waypoint[:-1] + sp_coords_waypoint_d

    else:
        # Get all ids for each level
        query = f"{url}/sp/{o.id}/{d.id}/{projection}"
        r = requests.get(url=query)
        sp_coords = r.json()["sp"]
        return sp_coords
