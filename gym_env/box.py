import pymunk as pm
from pymunk.vec2d import Vec2d

from .constants import COLLISION_TYPES


class Box:
    body: pm.Body
    shape: pm.Poly

    MASS_KG: float = 1.5
    SIDE_LENGTH_MM = 200
    FRICTION = 0.1
    ELASTICITY = 0.1

    def __init__(self, scale_factor: float) -> None:
        self.mass_scaled = Box.MASS_KG*scale_factor
        self.side_length_scaled = Box.SIDE_LENGTH_MM*scale_factor
        moment = pm.moment_for_box(self.mass_scaled,
                                   (self.side_length_scaled, self.side_length_scaled))

        self.body = pm.Body(self.mass_scaled, moment)

        self.shape = pm.Poly.create_box(self.body,
                                        (self.side_length_scaled, self.side_length_scaled))

        self.shape.friction = Box.FRICTION
        self.shape.elasticity = Box.ELASTICITY
        self.shape.collision_type = COLLISION_TYPES["box"]

    def stop(self) -> None:
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def set_position(self, position: Vec2d, angle_rad: float = 0) -> None:
        self.body.position = position
        self.body.angle = angle_rad
