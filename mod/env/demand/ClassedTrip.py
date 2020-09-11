from mod.env.demand.Trip import Trip


class ClassedTrip(Trip):
    def __str__(self):
        return (
            f"[{self.time:04}({self.placement})]"
            f"{self.sq_class}{self.id:03}[{self.o:>4},{self.d:>4}]"
            f" - remaining: {self.max_delay_from_placement:>6.2f} min"
        )

    @property
    def sq_class_backlog(self):
        return f"{self.sq_class}_{self.times_backlogged}"

    def __init__(
            self,
            o,
            d,
            time,
            sq_class,
            elapsed_sec=0,
            placement=None,
            max_delay=10,
            tolerance=5,
            distance_km=None,
            time_increment=1,
    ):
        super().__init__(o, d, time)
        self.sq_class = sq_class
        self.times_backlogged = 0

        # How much time has passed from the beginning of the step
        # to the announcement time
        self.elapsed_sec = elapsed_sec

        # Datetime trip was placed in the system
        self.placement = placement

        # Min/Max class delays
        self.max_delay = max_delay
        self.tolerance = tolerance
        self.distance_km = distance_km

        # How much time user waits until the end of the step
        self.delay_close_step = time_increment - self.elapsed_sec / 60

        # Min/Max delays discounting announcement. Attribute elapsed_sec
        # start from the beginning of each step, i.e., <= time_increment
        self.max_delay_from_placement = self.max_delay - self.delay_close_step

    @property
    def attribute(self, level=0):
        return (self.o.id_level(level), self.d.id_level(level), self.sq_class)

    @property
    def attribute_backlog(self, level=0):
        return (
            self.o.id_level(level),
            self.d.id_level(level),
            self.sq_class_backlog,
        )

    def __str__(self):
        return f"{self.sq_class}{self.id:02}({self.o},{self.d})"

    def __repr__(self):
        return (
            f"Trip{{"
            f"id={self.id:03},"
            f"placement={self.placement},"
            f"o={self.o.level_ids},"
            f"d={self.d.level_ids},"
            f"sq={self.sq_class},"
            f"bklog={self.backlog_delay},"
            f"time={self.time:03}}}"
        )

    def info(self):
        return (
            f"Trip{{"
            f"id={self.id:03},"
            f"o={self.o.level_ids},"
            f"d={self.d.level_ids},"
            f"sq={self.sq_class},"
            f"time={self.time:03},"
            f"pk_delay={self.pk_delay},"
            f"max_delay={self.max_delay:6.2f},"
            f"from_placement={self.max_delay_from_placement:6.2f},"
            f"tolerance={self.tolerance:6.2f},"
            f"elapsed={self.elapsed_sec:6.2f},"
            f"backlogged={self.times_backlogged}"
        )