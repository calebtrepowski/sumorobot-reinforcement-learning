from pymunk.vec2d import Vec2d


class LineSensor:
    position: Vec2d
    OFFSET: Vec2d
    status: bool

    def __init__(self, offset: Vec2d) -> None:
        self.OFFSET = offset
        self.position = Vec2d(0, 0)
        self.status = False
