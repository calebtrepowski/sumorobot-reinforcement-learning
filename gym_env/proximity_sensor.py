import pymunk as pm
from pymunk.vec2d import Vec2d

from .constants import COLLISION_TYPES


class ProximitySensor:
    body: pm.Body
    shape: pm.Segment

    start_point: Vec2d
    end_point: Vec2d
    RANGE: float
    OFFSET: Vec2d
    ANGLE_RAD: float
    status: bool

    def __init__(self, offset: Vec2d, angle_rad: float, range: float) -> None:

        self.RANGE = range
        self.OFFSET = offset
        self.ANGLE_RAD = angle_rad
        self.status = False

        self.body = pm.Body(body_type=pm.Body.KINEMATIC)
        self.shape = pm.Segment(self.body, Vec2d(
            0, 0), Vec2d(self.RANGE, 0), radius=2)
        self.shape.collision_type = COLLISION_TYPES["proximity_sensor"]
        self.shape.sensor = True

        self.start_point = self.shape.a
        self.end_point = self.shape.b
