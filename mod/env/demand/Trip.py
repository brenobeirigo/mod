class Trip:
    trip_count = 0

    def __init__(self, o, d, time):
        self.o = o
        self.d = d
        self.time = time  # step
        self.pk_step = None  # step
        self.dp_step = None
        self.id = Trip.trip_count
        Trip.trip_count += 1
        self.picked_by = None
        self.dropoff_time = None
        self.pk_delay = None
        self.pk_duration = None
        # Accrue backlogging delay
        self.backlog_delay = 0

    def attribute(self, level):
        return (self.o.id_level(level), self.d.id_level(level))

    def __str__(self):
        return f"T{self.id:03}[{self.o:>4},{self.d:>4}]"

    def __repr__(self):
        return (
            f"Trip{{"
            f"id={self.id:03},"
            f"o={self.o.level_ids},"
            f"d={self.d.level_ids},"
            f"time={self.time:03}}}"
        )

    def can_be_picked_by(self, car, level=0):
        return self.o.id_level(level) == car.point.id_level(level)
