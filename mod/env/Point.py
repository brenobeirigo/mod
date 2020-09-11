class Point:
    point_dict = dict()

    levels = []

    level_count = []

    def __init__(self, x, y, point_id, level_ids=None, level_ids_dic=None):
        self.x = x
        self.y = y
        if level_ids:
            self.level_ids = level_ids
        else:
            level_ids = (point_id,)

        if level_ids_dic:
            self.level_ids_dic = level_ids_dic
        else:

            self.level_ids_dic = {0: point_id}

        # Add point to dictionary of all points
        if point_id not in Point.point_dict:
            Point.point_dict[point_id] = self

    @property
    def id(self):
        return self.level_ids[0]

    def __str__(self):
        return str(self.id)
        # return f"{self.level_ids}"

    def __repr__(self):
        return f"Point({self.id:02}, {self.x}, {self.y}, {self.level_ids})"

    def __hash__(self):
        # return hash((self.x, self.y))
        return hash(self.level_ids[0])

    def id_level(self, i):
        try:
            return self.level_ids[i]
        except:
            raise IndexError(
                f'Point {self} has no level "{i}" ({self.level_ids}).'
            )