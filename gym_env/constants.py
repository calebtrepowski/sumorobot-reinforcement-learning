COLLISION_TYPES = {
    "robot": 1,
    "box": 2,
    "proximity_sensor": 3
}

SCALE_FACTOR = 1/3

SUMO_DIMENSIONS = {
    "mass_kg": 3.0,
    "side_length_mm": 200,
    "wheels_distance_from_center_mm": 70,
    "wheels_radius_mm": 20,
    "max_torque_n_mm": 95,
    "line_sensor_distance_from_center_mm": 80,
    "proximity_sensor_distance_from_center_mm": 50,
    "proximity_sensor_range_mm": 750
}

BOX_DIMENSIONS = {
    "mass_kg": 3.0,
    "side_length_mm": 200,
}

RING_DIMENSIONS = {
    "inner_radius_mm": 720,
    "border_width_mm": 50,
}

SPACE_DAMPING = 0.35
