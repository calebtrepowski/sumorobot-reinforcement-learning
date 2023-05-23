import pymunk as pm
from pymunk.vec2d import Vec2d
import numpy as np

from .constants import SUMO_DIMENSIONS, SCALE_FACTOR, COLLISION_TYPES
from .proximity_sensor import ProximitySensor
from .line_sensor import LineSensor


class SumoRobot:
    body: pm.Body
    shape: pm.Poly

    MASS_KG: float = SUMO_DIMENSIONS["mass_kg"]*SCALE_FACTOR
    SIDE_LENGTH_MM: float = SUMO_DIMENSIONS["side_length_mm"]*SCALE_FACTOR
    WHEEL_DISTANCE_FROM_CENTER_MM: float = SUMO_DIMENSIONS[
        "wheels_distance_from_center_mm"]*SCALE_FACTOR
    WHEELS_RADIUS_MM: float = SUMO_DIMENSIONS["wheels_radius_mm"]*SCALE_FACTOR
    MAX_TORQUE_N_MM: float = SUMO_DIMENSIONS["max_torque_n_mm"]*SCALE_FACTOR

    LINE_SENSOR_DISTANCE_FROM_CENTER_MM: float = SUMO_DIMENSIONS[
        "line_sensor_distance_from_center_mm"]*SCALE_FACTOR
    PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM: float = SUMO_DIMENSIONS[
        "proximity_sensor_distance_from_center_mm"]*SCALE_FACTOR
    PROXIMITY_SENSOR_RANGE_MM: float = SUMO_DIMENSIONS["proximity_sensor_range_mm"]*SCALE_FACTOR

    FRICTION = 0.9
    ELASTICITY = 0.1

    proximity_sensors: tuple[ProximitySensor]
    line_sensors: tuple[LineSensor]

    def __init__(self, scale_factor: float) -> None:
        self.side_length_scaled = SumoRobot.SIDE_LENGTH_MM*scale_factor
        moment = pm.moment_for_box(SumoRobot.MASS_KG,
                                   (SumoRobot.SIDE_LENGTH_MM, SumoRobot.SIDE_LENGTH_MM))

        self.body = pm.Body(SumoRobot.MASS_KG, moment)

        self.shape = pm.Poly.create_box(self.body,
                                        (SumoRobot.SIDE_LENGTH_MM, SumoRobot.SIDE_LENGTH_MM))

        self.shape.friction = SumoRobot.FRICTION
        self.shape.elasticity = SumoRobot.ELASTICITY
        self.shape.collision_type = COLLISION_TYPES["robot"]

        self.initialize_wheels()
        self.initialize_line_sensors()
        self.initialize_proximity_sensors()

    def initialize_wheels(self) -> None:
        self.WHEELS_MAX_TANGENT_FORCE = SumoRobot.MAX_TORQUE_N_MM*SumoRobot.WHEELS_RADIUS_MM

        self.LEFT_WHEEL_OFFSET = Vec2d(
            0, SumoRobot.WHEEL_DISTANCE_FROM_CENTER_MM)
        self.left_wheel_position = self.body.local_to_world(
            self.LEFT_WHEEL_OFFSET)
        self.left_wheel_force = Vec2d(0, 0)

        self.RIGHT_WHEEL_OFFSET = Vec2d(
            0, -SumoRobot.WHEEL_DISTANCE_FROM_CENTER_MM)
        self.right_wheel_position = self.body.local_to_world(
            self.LEFT_WHEEL_OFFSET)
        self.right_wheel_force = Vec2d(0, 0)

    def update_wheels_positions(self) -> None:
        self.left_wheel_position = self.body.local_to_world(
            self.LEFT_WHEEL_OFFSET)
        self.right_wheel_position = self.body.local_to_world(
            self.RIGHT_WHEEL_OFFSET)

    def initialize_line_sensors(self) -> None:
        front_left_line_sensor = LineSensor(Vec2d(
            SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM, SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM))
        front_right_line_sensor = LineSensor(Vec2d(
            SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM, -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM))
        back_right_line_sensor = LineSensor(Vec2d(
            -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM, -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM))
        back_left_line_sensor = LineSensor(Vec2d(
            -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM, SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM))

        self.line_sensors = (front_left_line_sensor, front_right_line_sensor,
                             back_right_line_sensor, back_left_line_sensor)
        self.update_line_sensors_positions()

    def update_line_sensors_positions(self) -> None:
        for line_sensor in self.line_sensors:
            line_sensor.position = self.body.local_to_world(line_sensor.OFFSET)

    def initialize_proximity_sensors(self) -> None:
        front_center_proximity_sensor = ProximitySensor(
            Vec2d(SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM, 0), 0, SumoRobot.PROXIMITY_SENSOR_RANGE_MM)
        front_left_proximity_sensor = ProximitySensor(
            Vec2d(SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM, SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM), np.deg2rad(45), SumoRobot.PROXIMITY_SENSOR_RANGE_MM)
        left_proximity_sensor = ProximitySensor(
            Vec2d(0, SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM), np.deg2rad(90), SumoRobot.PROXIMITY_SENSOR_RANGE_MM)
        back_proximity_sensor = ProximitySensor(
            Vec2d(-SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM,
                  0), np.deg2rad(180), SumoRobot.PROXIMITY_SENSOR_RANGE_MM
        )
        right_proximity_sensor = ProximitySensor(
            Vec2d(
                0, -SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM), np.deg2rad(270), SumoRobot.PROXIMITY_SENSOR_RANGE_MM
        )
        front_right_proximity_sensor = ProximitySensor(
            Vec2d(SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM, -SumoRobot.PROXIMITY_SENSOR_DISTANCE_FROM_CENTER_MM), np.deg2rad(315), SumoRobot.PROXIMITY_SENSOR_RANGE_MM)

        self.proximity_sensors = (front_center_proximity_sensor,
                                  front_left_proximity_sensor, left_proximity_sensor, back_proximity_sensor, right_proximity_sensor, front_right_proximity_sensor
                                  )

    def update_proximity_sensors_positions(self) -> None:
        robot_angle = self.body.angle
        for proximity_sensor in self.proximity_sensors:
            proximity_sensor_angle = robot_angle + proximity_sensor.ANGLE_RAD
            local_rotation_vector = Vec2d(
                np.cos(proximity_sensor.ANGLE_RAD), np.sin(proximity_sensor.ANGLE_RAD))
            proximity_sensor.start_point = self.body.local_to_world(
                proximity_sensor.OFFSET)
            proximity_sensor.end_point = self.body.local_to_world(
                proximity_sensor.OFFSET + proximity_sensor.RANGE*local_rotation_vector)
            proximity_sensor.body.position = proximity_sensor.start_point
            proximity_sensor.body.angle = proximity_sensor_angle

    def set_position(self, position: Vec2d, angle_rad: float = 0) -> None:
        self.body.position = position
        self.body.angle = angle_rad
        self.update_wheels_positions()
        self.update_line_sensors_positions()
        self.update_proximity_sensors_positions()

    def stop(self) -> None:
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def step(self, action: Vec2d) -> None:
        robot_direction = self.body.rotation_vector
        self.left_wheel_force = self.WHEELS_MAX_TANGENT_FORCE * \
            action[0]*robot_direction
        self.right_wheel_force = self.WHEELS_MAX_TANGENT_FORCE * \
            action[1]*robot_direction

        self.update_wheels_positions()
        self.update_line_sensors_positions()
        self.update_proximity_sensors_positions()

        self.body.apply_force_at_world_point(
            self.left_wheel_force, self.left_wheel_position)
        self.body.apply_force_at_world_point(
            self.right_wheel_force, self.right_wheel_position)
