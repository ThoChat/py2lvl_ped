# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import dataclass, replace

import numpy as np

from jupedsim.agent_view import WallView
from jupedsim.models.custom_model import CustomOperationalModel

UNBALANCING_RATE = 0.1
DAMPING_RATE = 0.5
BALANCING_RATE = 0.5
GS_SCALING_FACTOR = 0.26 / (2 * 0.3 * 1.65)
# G = 9.80665


@dataclass(kw_only=True, frozen=True)
class TwoLevelPedestrianModelState:
    velocity: tuple[float, float] = (0.0, 0.0)
    ground_support_velocity: tuple[float, float] = (0.0, 0.0)
    ground_support_position: tuple[float, float] = (0.0, 0.0)
    upper_body_position: tuple[float, float] = (0.0, 0.0)
    desired_speed: float = 1.34
    reaction_time: float = 0.5
    agent_scale: float = 2000.0
    obstacle_scale: float = 2000.0
    force_distance: float = 0.08
    mass: float = 80.0
    body_force: float = 120000.0
    friction: float = 240000.0
    radius: float = 0.3
    height: float = 1.65


class TwoLevelPedestrianModel(CustomOperationalModel):
    """
    Two-Level Pedestrian Model with Social Forces.

    Each agent has an upper body and a ground support. The model combines
    Helbing-style social repulsion with contact forces at two levels and a
    locomotion/recovery coupling via the unit vector e_gs_ub.
    """

    def __init__(self):
        CustomOperationalModel.__init__(self)

    @staticmethod
    def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
        norm = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if norm < 1e-10:
            return (0.0, 0.0)
        return (vector[0] / norm, vector[1] / norm)

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def _social_force_between_points(
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        A: float,
        B: float,
        radiuses_sum: float,
    ) -> tuple[float, float]:
        dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        if dist < 1e-10:
            return (0.0, 0.0)
        pushing_force_norm = A * np.exp((radiuses_sum - dist) / B)
        n = ((pt1[0] - pt2[0]) / dist, (pt1[1] - pt2[1]) / dist)
        return (pushing_force_norm * n[0], pushing_force_norm * n[1])

    @staticmethod
    def _contact_force_between_points(
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        radiuses_sum: float,
        velocity_diff: tuple[float, float],
    ) -> tuple[float, float]:
        dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        if dist < 1e-10:
            return (0.0, 0.0)
        n = ((pt1[0] - pt2[0]) / dist, (pt1[1] - pt2[1]) / dist)
        # t = (-n[1], n[0])
        fx, fy = 0.0, 0.0
        if dist < radiuses_sum:
            normal_pushing_force_norm = 2 * np.exp((radiuses_sum - dist) / 0.5)
            # tangential_friction_norm = /friction * (radiuses_sum - dist) * dot(velocity_diff, t)
            fx = normal_pushing_force_norm * n[0]
            fy = normal_pushing_force_norm * n[1]
        return (fx, fy)

    @staticmethod
    def _desired_force(
        *,
        velocity: tuple[float, float],
        target_direction: tuple[float, float],
        desired_speed: float,
        reaction_time: float,
    ) -> tuple[float, float]:
        v_desired_x = desired_speed * target_direction[0]
        v_desired_y = desired_speed * target_direction[1]
        fx = (v_desired_x - velocity[0]) / reaction_time
        fy = (v_desired_y - velocity[1]) / reaction_time
        return (fx, fy)

    @staticmethod
    def _agent_social_force(state, neighbor) -> tuple[float, float]:
        agent_pos = state.ground_support_position
        neighbor_pos = (
            agent_pos[0] + neighbor.relative_position[0],
            agent_pos[1] + neighbor.relative_position[1],
        )
        min_dist = state.radius + neighbor.state.radius
        return TwoLevelPedestrianModel._social_force_between_points(
            agent_pos, neighbor_pos, state.agent_scale, state.force_distance, min_dist
        )

    @staticmethod
    def _obstacle_social_force(state, wall: WallView) -> tuple[float, float]:
        # wall.closest_point is relative to the agent (agent at origin), so the
        # force must be computed in that same relative frame -- using the origin
        # as the agent point. Mixing it with the absolute ground_support_position
        # makes the distance ~ |agent pos| and the repulsion vanish.
        min_dist = state.radius
        return TwoLevelPedestrianModel._social_force_between_points(
            (0.0, 0.0),
            wall.closest_point,
            state.obstacle_scale,
            state.force_distance,
            min_dist,
        )

    @staticmethod
    def _agent_upper_body_contact_force(state, neighbor) -> tuple[float, float]:
        agent_pos = state.ground_support_position
        neighbor_pos = (
            agent_pos[0] + neighbor.relative_position[0],
            agent_pos[1] + neighbor.relative_position[1],
        )
        min_dist = state.radius + neighbor.state.radius
        velocity_diff = (
            neighbor.state.velocity[0] - state.velocity[0],
            neighbor.state.velocity[1] - state.velocity[1],
        )
        return TwoLevelPedestrianModel._contact_force_between_points(
            agent_pos, neighbor_pos, min_dist, velocity_diff
        )

    @staticmethod
    def _agent_ground_support_contact_force(state, neighbor) -> tuple[float, float]:
        gs_radius1 = state.radius * GS_SCALING_FACTOR * state.height
        gs_radius2 = neighbor.state.radius * GS_SCALING_FACTOR * neighbor.state.height
        min_dist = gs_radius1 + gs_radius2
        velocity_diff = (
            neighbor.state.ground_support_velocity[0]
            - state.ground_support_velocity[0],
            neighbor.state.ground_support_velocity[1]
            - state.ground_support_velocity[1],
        )
        return TwoLevelPedestrianModel._contact_force_between_points(
            state.ground_support_position,
            neighbor.state.ground_support_position,
            min_dist,
            velocity_diff,
        )

    @staticmethod
    def _obstacle_upper_body_contact_force(
        state, wall: WallView
    ) -> tuple[float, float]:
        # wall.closest_point is relative to the agent (agent at origin)
        return TwoLevelPedestrianModel._contact_force_between_points(
            (0.0, 0.0), wall.closest_point, state.radius, state.velocity
        )

    @staticmethod
    def _obstacle_ground_support_contact_force(
        state, wall: WallView
    ) -> tuple[float, float]:
        # Walls are expressed relative to the upper body (agent at origin). The ground
        # support lives elsewhere, so compute its own closest point on the wall segment
        # (gs_rel = ground support position in the upper-body frame).
        ub = state.upper_body_position
        gs = state.ground_support_position
        gs_rel = (gs[0] - ub[0], gs[1] - ub[1])
        gs_radius = state.radius * GS_SCALING_FACTOR * state.height
        closest = wall.segment.closest_point(gs_rel)
        return TwoLevelPedestrianModel._contact_force_between_points(
            gs_rel,
            closest,
            gs_radius,
            state.ground_support_velocity,
        )

    def compute_next_state(self, state, step):
        dt = step.dt
        target_dir = step.orientation_to_next_target
        ub_pos = state.upper_body_position
        gs_pos = state.ground_support_position

        # --- Social forces (divided by mass) ---
        social_forces = self._desired_force(
            velocity=state.velocity,
            target_direction=target_dir,
            desired_speed=state.desired_speed,
            reaction_time=state.reaction_time,
        )

        for neighbor in step.other_agents_in_range(2.5):
            f = self._agent_social_force(state, neighbor)
            social_forces = (
                social_forces[0] + f[0] / state.mass,
                social_forces[1] + f[1] / state.mass,
            )

        for wall in step.walls_in_range(5.0):
            f = self._obstacle_social_force(state, wall)
            social_forces = (
                social_forces[0] + f[0] / state.mass,
                social_forces[1] + f[1] / state.mass,
            )

        # --- Contact forces (NOT divided by mass) ---
        upper_body_contact = (0.0, 0.0)
        for neighbor in step.other_agents_in_range(2.5):
            f = self._agent_upper_body_contact_force(state, neighbor)
            upper_body_contact = (
                upper_body_contact[0] + f[0],
                upper_body_contact[1] + f[1],
            )
        for wall in step.walls_in_range(5.0):
            f = self._obstacle_upper_body_contact_force(state, wall)
            upper_body_contact = (
                upper_body_contact[0] + f[0],
                upper_body_contact[1] + f[1],
            )

        ground_support_contact = (0.0, 0.0)
        for neighbor in step.other_agents_in_range(2.5):
            f = self._agent_ground_support_contact_force(state, neighbor)
            ground_support_contact = (
                ground_support_contact[0] + f[0],
                ground_support_contact[1] + f[1],
            )
        for wall in step.walls_in_range(5.0):
            f = self._obstacle_ground_support_contact_force(state, wall)
            ground_support_contact = (
                ground_support_contact[0] + f[0],
                ground_support_contact[1] + f[1],
            )

        # --- Unit vector: ground_support -> upper body ---
        e_norm = self._normalize((ub_pos[0] - gs_pos[0], ub_pos[1] - gs_pos[1]))
        e_gs_ub = e_norm if e_norm != (0.0, 0.0) else (1.0, 0.0)

        # --- Update upper body velocity and position ---
        new_velocity = (
            state.velocity[0]
            + (
                social_forces[0]
                + upper_body_contact[0]
                + (e_gs_ub[0] * 1.0 - state.velocity[0]) * UNBALANCING_RATE
                - state.velocity[0] * DAMPING_RATE
            )
            * dt,
            state.velocity[1]
            + (
                social_forces[1]
                + upper_body_contact[1]
                + (e_gs_ub[1] * 1.0 - state.velocity[1]) * UNBALANCING_RATE
                - state.velocity[1] * DAMPING_RATE
            )
            * dt,
        )

        # --- Update ground support velocity and position ---
        new_gs_velocity = (
            state.ground_support_velocity[0]
            + (
                ground_support_contact[0]
                + (e_gs_ub[0] * 1.0 - state.ground_support_velocity[0]) * BALANCING_RATE
            )
            * dt,
            state.ground_support_velocity[1]
            + (
                ground_support_contact[1]
                + (e_gs_ub[1] * 1.0 - state.ground_support_velocity[1]) * BALANCING_RATE
            )
            * dt,
        )

        new_gs_position = (
            gs_pos[0] + new_gs_velocity[0] * dt,
            gs_pos[1] + new_gs_velocity[1] * dt,
        )
        new_ub_position = (
            ub_pos[0] + new_velocity[0] * dt,
            ub_pos[1] + new_velocity[1] * dt,
        )

        new_state = replace(
            state,
            velocity=new_velocity,
            ground_support_velocity=new_gs_velocity,
            ground_support_position=new_gs_position,
            upper_body_position=new_ub_position,
        )

        movement = (new_velocity[0] * dt, new_velocity[1] * dt)

        return new_state, movement

    def check_model_constraint(self, state, view):
        if state.mass <= 0:
            raise ValueError(f"mass must be positive, got {state.mass}")
        if state.desired_speed <= 0:
            raise ValueError(
                f"desired_speed must be positive, got {state.desired_speed}"
            )
        if state.reaction_time <= 0:
            raise ValueError(
                f"reaction_time must be positive, got {state.reaction_time}"
            )
        if state.radius <= 0:
            raise ValueError(f"radius must be positive, got {state.radius}")

        for neighbor in view.other_agents_in_range(2.0):
            dist = self._distance(
                state.ground_support_position, neighbor.state.ground_support_position
            )
            if state.radius >= dist:
                raise ValueError(
                    f"Agent too close to neighbor: distance {dist}, radius {state.radius}"
                )

        if view.walls_in_range(state.radius / 2):
            raise ValueError("Agent too close to geometry boundaries")
