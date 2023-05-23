import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk as pm
from pymunk.vec2d import Vec2d

from .line_sensor import LineSensor
from .proximity_sensor import ProximitySensor
from .constants import COLLISION_TYPES


class SumoRobot:
    body: pm.Body
    shape: pm.Poly

    MASS_KG: float = 3.0
    SIDE_LENGTH_MM = 200
    WHEEL_DISTANCE_FROM_CENTER_MM = 70
    WHEELS_RADIUS_MM = 20
    MAX_TORQUE_N_MM = 95

    LINE_SENSOR_DISTANCE_FROM_CENTER_MM = 80
    PROXIMITY_SENSOR_RANGE_MM = 750

    FRICTION = 0.9
    ELASTICITY = 0.1

    mass_scaled: float
    side_length_scaled: float

    proximity_sensors: tuple[ProximitySensor]
    line_sensors: tuple[LineSensor]

    def __init__(self, scale_factor: float) -> None:
        self.mass_scaled = SumoRobot.MASS_KG*scale_factor
        self.side_length_scaled = SumoRobot.SIDE_LENGTH_MM*scale_factor
        moment = pm.moment_for_box(self.mass_scaled,
                                   (self.side_length_scaled, self.side_length_scaled))

        self.body = pm.Body(self.mass_scaled, moment)

        self.shape = pm.Poly.create_box(self.body,
                                        (self.side_length_scaled, self.side_length_scaled))

        self.shape.friction = SumoRobot.FRICTION
        self.shape.elasticity = SumoRobot.ELASTICITY
        self.shape.collision_type = COLLISION_TYPES["robot"]

        self.initialize_wheels(scale_factor)
        self.initialize_line_sensors(scale_factor)
        self.initialize_proximity_sensors(scale_factor)

    def initialize_wheels(self, scale_factor: float) -> None:
        self.WHEELS_MAX_TANGENT_FORCE = (
            SumoRobot.MAX_TORQUE_N_MM*scale_factor)*(SumoRobot.WHEELS_RADIUS_MM*scale_factor)

        self.LEFT_WHEEL_OFFSET = Vec2d(
            0, SumoRobot.WHEEL_DISTANCE_FROM_CENTER_MM*scale_factor)
        self.left_wheel_position = self.body.local_to_world(
            self.LEFT_WHEEL_OFFSET)
        self.left_wheel_force = Vec2d(0, 0)

        self.RIGHT_WHEEL_OFFSET = Vec2d(
            0, -SumoRobot.WHEEL_DISTANCE_FROM_CENTER_MM*scale_factor)
        self.right_wheel_position = self.body.local_to_world(
            self.LEFT_WHEEL_OFFSET)
        self.right_wheel_force = Vec2d(0, 0)

    def update_wheels_positions(self) -> None:
        self.left_wheel_position = self.body.local_to_world(
            self.LEFT_WHEEL_OFFSET)
        self.right_wheel_position = self.body.local_to_world(
            self.RIGHT_WHEEL_OFFSET)

    def initialize_line_sensors(self, scale_factor: float) -> None:
        front_left_line_sensor = LineSensor(Vec2d(
            SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor, SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor))
        front_right_line_sensor = LineSensor(Vec2d(
            SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor, -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor))
        back_right_line_sensor = LineSensor(Vec2d(
            -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor, -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor))
        back_left_line_sensor = LineSensor(Vec2d(
            -SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor, SumoRobot.LINE_SENSOR_DISTANCE_FROM_CENTER_MM*scale_factor))

        self.line_sensors = (front_left_line_sensor, front_right_line_sensor,
                             back_right_line_sensor, back_left_line_sensor)
        self.update_line_sensors_positions()

    def update_line_sensors_positions(self) -> None:
        for line_sensor in self.line_sensors:
            line_sensor.position = self.body.local_to_world(line_sensor.OFFSET)

    def initialize_proximity_sensors(self, scale_factor: float) -> None:
        front_center_proximity_sensor = ProximitySensor(
            Vec2d(50*scale_factor, 0), 0, SumoRobot.PROXIMITY_SENSOR_RANGE_MM*scale_factor)
        front_left_proximity_sensor = ProximitySensor(
            Vec2d(50*scale_factor, 50*scale_factor), np.deg2rad(45), SumoRobot.PROXIMITY_SENSOR_RANGE_MM*scale_factor)
        left_proximity_sensor = ProximitySensor(
            Vec2d(0, 50*scale_factor), np.deg2rad(90), SumoRobot.PROXIMITY_SENSOR_RANGE_MM*scale_factor)
        back_proximity_sensor = ProximitySensor(
            Vec2d(-50*scale_factor,
                  0), np.deg2rad(180), SumoRobot.PROXIMITY_SENSOR_RANGE_MM*scale_factor
        )
        right_proximity_sensor = ProximitySensor(
            Vec2d(
                0, -50*scale_factor), np.deg2rad(270), SumoRobot.PROXIMITY_SENSOR_RANGE_MM*scale_factor
        )
        front_right_proximity_sensor = ProximitySensor(
            Vec2d(50*scale_factor, -50*scale_factor), np.deg2rad(315), SumoRobot.PROXIMITY_SENSOR_RANGE_MM*scale_factor)

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


class SumoRobotEnv(gym.Env):
    """
    Observation Space:
        Type: MultiBinary(10)
        Num    Observation
        0      Proximity Sensor: Front
        1      Proximity Sensor: Front Left
        2      Proximity Sensor: Left
        3      Proximity Sensor: Back
        4      Proximity Sensor: Right
        5      Proximity Sensor: Front Right
        6      Line Sensor: Front Left
        7      Line Sensor: Front Right
        8      Line Sensor: Back Right
        9      Line Sensor: Back Left

    Action Space:
        Type: Box(2)
        Num    Action                       Min     Max
        0      Move left motor/wheel        -1.0    1.0
        1      Move right motor/wheel       -1.0    1.0
    """

    metadata = {
        "render_fps": 60,
        "render_modes": ["human", "rgb_array"]
    }

    SCALE_FACTOR = 1/3

    RING_RADIUS_PX: int = int(720*SCALE_FACTOR)
    RING_BORDER_WIDTH_PX: int = int(50*SCALE_FACTOR)
    RING_TOTAL_RADIUS_PX = RING_RADIUS_PX + RING_BORDER_WIDTH_PX
    BOX_SIDE_PX = int(200*SCALE_FACTOR)

    SCREEN_WIDTH_PX: int = 2*(RING_TOTAL_RADIUS_PX) + 50
    SCREEN_HEIGHT_PX: int = 2*(RING_TOTAL_RADIUS_PX) + 50

    RING_CENTER = Vec2d(int(SCREEN_WIDTH_PX/2), int(SCREEN_HEIGHT_PX/2))

    def __init__(self, *, render_mode=None, for_training: bool = False) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.for_training = for_training
        self.screen = None
        self.clock = None

        self.observation_space = spaces.MultiBinary(10)
        self.action_space = spaces.Box(
            low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)

        self.space = pm.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.35

        self.robot = SumoRobot(SumoRobotEnv.SCALE_FACTOR)
        self.reset_robot()

        self.box = Box(SumoRobotEnv.SCALE_FACTOR)
        self.reset_box()

        self.space.add(self.robot.body, self.robot.shape)
        self.space.add(self.box.body, self.box.shape)

        for proximity_sensor in self.robot.proximity_sensors:
            self.space.add(proximity_sensor.body, proximity_sensor.shape)

        self.collision_handler_robot_box = self.space.add_collision_handler(
            COLLISION_TYPES["robot"], COLLISION_TYPES["box"])
        self.collision_handler_robot_box.begin = self._handle_collision_robot_box

        self.collision_handler_robot_box = self.space.add_collision_handler(
            COLLISION_TYPES["box"], COLLISION_TYPES["proximity_sensor"])
        self.collision_handler_robot_box.begin = self._handle_begin_collision_box_proximity_sensor
        self.collision_handler_robot_box.separate = self._handle_separate_collision_box_proximity_sensor

        self.dt = 1.0 / 60.0  # Time step for the physics simulation
        self.num_iterations = 10  # Number of iterations per step

    def step(self, action: np.ndarray):
        self.robot.step(
            Vec2d(float(action[0]), float(action[1])))

        for i in range(self.num_iterations):
            self.space.step(self.dt)

        observation = np.array((*self.get_proximity_sensor_readings(),
                               *self.get_line_sensor_readings()), dtype=np.int0)

        done = False
        reward = 0

        robot_position = self.robot.body.position
        box_position = self.box.body.position

        if np.linalg.norm(robot_position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX or\
                np.linalg.norm(box_position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX:
            done = True

        if self.render_mode is not None:
            self.render(mode=self.render_mode)

        info = {}

        if self.for_training:
            return observation, reward, done, info

        truncated = False

        return observation, reward, done, truncated, info

    def get_proximity_sensor_readings(self) -> list[bool]:
        proximity_sensor_readings = [
            proximity_sensor.status for proximity_sensor in self.robot.proximity_sensors]

        return proximity_sensor_readings

    def get_line_sensor_readings(self) -> list[bool]:
        line_sensor_readings = []
        for line_sensor in self.robot.line_sensors:
            line_sensor_distance = np.linalg.norm(
                line_sensor.position - SumoRobotEnv.RING_CENTER)
            if line_sensor_distance > SumoRobotEnv.RING_RADIUS_PX:
                line_sensor.status = True
                line_sensor_readings.append(True)
            else:
                line_sensor.status = False
                line_sensor_readings.append(False)

        return line_sensor_readings

    def reset(self):
        self.reset_robot()
        self.reset_box()

        if self.render_mode is not None:
            self.render(mode=self.render_mode)

        observation = np.array([False]*10, dtype=np.int0)

        if self.for_training:
            return observation
        info = {}
        return observation, info

    def reset_robot(self) -> None:
        self.robot.stop()
        self.robot.set_position(
            self.generate_random_coordinate(),
            np.deg2rad(np.random.randint(0, 360)))

    def reset_box(self) -> None:
        self.box.stop()
        self.box.set_position(
            SumoRobotEnv.RING_CENTER + Vec2d(600*SumoRobotEnv.SCALE_FACTOR, 50))
        self.box.set_position(
            self.generate_random_coordinate(), np.deg2rad(np.random.randint(0, 360)))

    def generate_random_coordinate(self) -> Vec2d:
        RING_RADIUS = SumoRobotEnv.RING_RADIUS_PX
        return SumoRobotEnv.RING_CENTER + \
            Vec2d(np.random.randint(-RING_RADIUS, RING_RADIUS),
                  np.random.randint(-RING_RADIUS, RING_RADIUS))

    def render(self, *, mode: str = None, action: np.ndarray = None):
        import pygame
        import pygame.locals

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (SumoRobotEnv.SCREEN_WIDTH_PX,
                     SumoRobotEnv.SCREEN_HEIGHT_PX))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        BACKGROUND_COLOR = (111, 29, 29)
        ROBOT_COLOR = (255, 0, 0)
        WHEEL_COLOR = (0, 128, 128)
        BOX_COLOR = (0, 0, 255)
        RING_COLOR = (10, 10, 10)
        RING_BORDER_COLOR = (255, 255, 255)
        SENSOR_RANGE_COLOR = (160, 160, 160)
        SENSOR_DETECTION_COLOR = (255, 255, 0)

        self.surface = pygame.Surface(
            (SumoRobotEnv.SCREEN_WIDTH_PX, SumoRobotEnv.SCREEN_HEIGHT_PX))
        self.surface.fill(BACKGROUND_COLOR)

        pygame.draw.circle(
            self.surface, RING_COLOR,
            SumoRobotEnv.RING_CENTER,
            SumoRobotEnv.RING_RADIUS_PX)
        pygame.draw.circle(
            self.surface, RING_BORDER_COLOR,
            SumoRobotEnv.RING_CENTER,
            SumoRobotEnv.RING_RADIUS_PX + SumoRobotEnv.RING_BORDER_WIDTH_PX,
            SumoRobotEnv.RING_BORDER_WIDTH_PX)

        self._draw_rectangle(self.surface, self.robot.shape, ROBOT_COLOR)
        self._draw_rectangle(self.surface, self.box.shape, BOX_COLOR)

        # # robot orientation
        # pygame.draw.line(self.surface, SENSOR_RANGE_COLOR, self.robot.body.position,
        #                  self.robot.body.position + 50*self.robot.body.rotation_vector, width=2)

        # wheels
        pygame.draw.circle(self.surface, WHEEL_COLOR,
                           self.robot.left_wheel_position, 4)
        pygame.draw.circle(self.surface, WHEEL_COLOR,
                           self.robot.right_wheel_position, 4)

        # forces for each wheel
        # pygame.draw.line(self.surface, SENSOR_RANGE_COLOR,
        #                  self.robot.left_wheel_position,
        #                  self.robot.left_wheel_position + self.robot.left_wheel_force,
        #                  width=2)
        # pygame.draw.line(self.surface, SENSOR_RANGE_COLOR,
        #                  self.robot.right_wheel_position,
        #                  self.robot.right_wheel_position + self.robot.right_wheel_force,
        #                  width=2)

        # line sensors
        for line_sensor in self.robot.line_sensors:
            pygame.draw.circle(self.surface,
                               SENSOR_DETECTION_COLOR if line_sensor.status else SENSOR_RANGE_COLOR,
                               line_sensor.position,
                               int(10)*SumoRobotEnv.SCALE_FACTOR)
        # proximity sensors
        for proximity_sensor in self.robot.proximity_sensors:
            pygame.draw.line(self.surface,
                             SENSOR_DETECTION_COLOR if proximity_sensor.status else SENSOR_RANGE_COLOR,
                             proximity_sensor.start_point,
                             proximity_sensor.end_point,
                             width=2)

        self.surface = pygame.transform.flip(self.surface, False, True)
        self.screen.blit(self.surface, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    def _draw_rectangle(self, surface, shape, color) -> None:
        import pygame

        vertices = shape.get_vertices()

        transformed_vertices = []
        for vertex in vertices:
            world_position = shape.body.local_to_world(vertex)
            transformed_vertices.append(world_position)

        pygame.draw.polygon(surface, color, transformed_vertices)

    def _handle_collision_robot_box(self, arbiter: pm.Arbiter, space: pm.Space, data):
        return True

    def _handle_begin_collision_box_proximity_sensor(self, arbiter: pm.Arbiter, space: pm.Space, data):
        for proximity_sensor in self.robot.proximity_sensors:
            if arbiter.shapes[1] is proximity_sensor.shape:
                proximity_sensor.status = True
        return True

    def _handle_separate_collision_box_proximity_sensor(self, arbiter: pm.Arbiter, space: pm.Space, data):
        for proximity_sensor in self.robot.proximity_sensors:
            if arbiter.shapes[1] is proximity_sensor.shape:
                proximity_sensor.status = False
        return True


if __name__ == "__main__":

    env = SumoRobotEnv(render_mode="human")
    observation, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
