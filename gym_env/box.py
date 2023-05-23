import pymunk as pm
from pymunk.vec2d import Vec2d

from .constants import COLLISION_TYPES, BOX_DIMENSIONS, SCALE_FACTOR


class Box:
    body: pm.Body
    shape: pm.Poly

    MASS_KG: float = BOX_DIMENSIONS["mass_kg"]*SCALE_FACTOR
    SIDE_LENGTH_MM = BOX_DIMENSIONS["side_length_mm"]*SCALE_FACTOR

    FRICTION = 0.1
    ELASTICITY = 0.1

    def __init__(self) -> None:
        moment = pm.moment_for_box(Box.MASS_KG,
                                   (Box.SIDE_LENGTH_MM, Box.SIDE_LENGTH_MM))

        self.body = pm.Body(Box.MASS_KG, moment)

        self.shape = pm.Poly.create_box(self.body,
                                        (Box.SIDE_LENGTH_MM, Box.SIDE_LENGTH_MM))

        self.shape.friction = Box.FRICTION
        self.shape.elasticity = Box.ELASTICITY
        self.shape.collision_type = COLLISION_TYPES["box"]

    def stop(self) -> None:
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def set_position(self, position: Vec2d, angle_rad: float = 0) -> None:
        self.body.position = position
        self.body.angle = angle_rad
