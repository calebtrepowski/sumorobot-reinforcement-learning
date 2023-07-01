import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk as pm
from pymunk.vec2d import Vec2d

from .box import Box
from .sumo_robot import SumoRobot
from .constants import COLLISION_TYPES, SPACE_DAMPING, SCALE_FACTOR, RING_DIMENSIONS


class SumoRobotEnv(gym.Env):
    """
    Observation Space:
        Type: MultiBinary(8)
        Num    Observation
        0      Proximity Sensor: Front
        1      Proximity Sensor: Front Left
        2      Proximity Sensor: Left
        3      Proximity Sensor: Back
        4      Proximity Sensor: Right
        5      Proximity Sensor: Front Right
        6      Line Sensor: Front Left
        7      Line Sensor: Front Right

    Action Space:
        Type: Discrete(4)
        Num    Action
        0      Move forward
        1      Move backward
        2      Turn left
        3      Turn right
    """

    metadata = {
        "render_fps": 60,
        "render_modes": ["human", "rgb_array"]
    }

    RING_RADIUS_PX: int = int(
        RING_DIMENSIONS["inner_radius_mm"]*SCALE_FACTOR)
    RING_BORDER_WIDTH_PX: int = int(
        RING_DIMENSIONS["border_width_mm"]*SCALE_FACTOR)
    RING_TOTAL_RADIUS_PX = RING_RADIUS_PX + RING_BORDER_WIDTH_PX

    SCREEN_WIDTH_PX: int = 2*(RING_TOTAL_RADIUS_PX) + 50
    SCREEN_HEIGHT_PX: int = 2*(RING_TOTAL_RADIUS_PX) + 50

    RING_CENTER = Vec2d(int(SCREEN_WIDTH_PX/2), int(SCREEN_HEIGHT_PX/2))

    converted_actions = {
        0: Vec2d(1, 1),
        1: Vec2d(-1, -1),
        2: Vec2d(-1, 1),
        3: Vec2d(1, -1),
    }

    def __init__(self, *, render_mode=None, with_box: bool = True) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.with_box = with_box
        self.screen = None
        self.clock = None

        if self.with_box:
            self.observation_space = spaces.MultiBinary(8)
        else:
            self.observation_space = spaces.MultiBinary(2)

        self.action_space = spaces.Discrete(4)

        self.space = pm.Space()
        self.space.gravity = (0, 0)
        self.space.damping = SPACE_DAMPING

        self.robot = SumoRobot()
        self.reset_robot()

        self.space.add(self.robot.body, self.robot.shape)
        if self.with_box:
            self.box = Box()
            self.reset_box()
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

    def step(self, action: int):

        self.robot.step(SumoRobotEnv.converted_actions[action])

        for i in range(self.num_iterations):
            self.robot.update_line_sensors_positions()
            self.space.step(self.dt)

        observation = np.array((*self.get_proximity_sensor_readings(),
                               *self.get_line_sensor_readings()), dtype=np.int0)

        done = self.check_done()
        reward = self.calulate_reward(action=action)

        if self.render_mode is not None:
            self.render(mode=self.render_mode)

        info = {}

        return observation, reward, done, info

    def calulate_reward(self, **kwargs) -> float:
        action = kwargs["action"]

        reward = 0

        # per time tick
        reward -= 1

        # penalize going to the borders:
        if not any(self.get_proximity_sensor_readings()):
            for line_sensor in self.robot.line_sensors:
                if line_sensor.status:
                    reward -= 1

        # encourage going front if not seeing with any proximity sensor and not in the line
        # if not any(self.get_proximity_sensor_readings()) and not any(self.get_line_sensor_readings()):
        #     if action == 0:
        #         reward += 1

        # target velocity
        TARGET_VELOCITY = 30

        if action == 0:
            velocity = np.linalg.norm(self.robot.body.velocity)
            if velocity >= TARGET_VELOCITY:
                reward += 3
            else:
                reward += velocity/TARGET_VELOCITY

        # encourage seeing with front sensor
        if self.robot.proximity_sensors[0].status and action == 0:
            reward += 5

        if (self.robot.proximity_sensors[1].status or self.robot.proximity_sensors[2].status) and action == 2:
            reward += 2

        if (self.robot.proximity_sensors[4].status or self.robot.proximity_sensors[5].status) and action == 3:
            reward += 2

        if self.robot.proximity_sensors[3].status and (action != 2 or action != 3):
            reward -= 5

        # penalize falling:
        if np.linalg.norm(self.robot.body.position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX and not np.linalg.norm(self.box.body.position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX:
            reward -= 50
        
        if np.linalg.norm(self.box.body.position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX:
            reward += 50

        return reward

    def check_done(self) -> bool:
        robot_position = self.robot.body.position
        box_position = self.box.body.position

        if np.linalg.norm(robot_position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX or np.linalg.norm(box_position - SumoRobotEnv.RING_CENTER) > SumoRobotEnv.RING_TOTAL_RADIUS_PX:
            return True

        return False

    def get_proximity_sensor_readings(self) -> list[bool]:
        proximity_sensor_readings = [
            proximity_sensor.status for proximity_sensor in self.robot.proximity_sensors]

        if self.with_box:
            return proximity_sensor_readings
        return []

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

        observation = np.array([0]*self.observation_space.n, dtype=np.int0)

        return observation

    def reset_robot(self) -> None:
        self.robot.stop()
        self.robot.set_position(
            self.generate_random_coordinate(),
            np.deg2rad(np.random.randint(0, 360)))

    def reset_box(self) -> None:
        if self.with_box:
            self.box.stop()
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
        if self.with_box:
            self._draw_rectangle(self.surface, self.box.shape, BOX_COLOR)

        # wheels
        pygame.draw.circle(self.surface, WHEEL_COLOR,
                           self.robot.left_wheel_position, 4)
        pygame.draw.circle(self.surface, WHEEL_COLOR,
                           self.robot.right_wheel_position, 4)

        # line sensors
        for line_sensor in self.robot.line_sensors:
            pygame.draw.circle(self.surface,
                               SENSOR_DETECTION_COLOR if line_sensor.status else SENSOR_RANGE_COLOR,
                               line_sensor.position,
                               int(10)*SCALE_FACTOR)
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

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

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
