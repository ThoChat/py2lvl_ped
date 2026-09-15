import pathlib
import sqlite3
import math
import jupedsim as jps
import numpy as np
from numpy.random import normal  # normal distribution of free movement speed
from shapely import from_wkt
from shapely.ops import unary_union

np.random.seed(42)
from shapely import Point, Polygon
from py2lvl_ped_and_social_force import (
    TwoLevelPedestrianModel,
    TwoLevelPedestrianModelState,
)

## Setup geometries
room = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
inner = Point(0, 0).buffer(0.99, resolution=128)
outer = Point(0, 0).buffer(1.00, resolution=128)
ring = outer.difference(inner)


def radial_slot(center_angle, width, r_in=0.98, r_out=1.02):
    a_in_left = center_angle - (width / 2.0) / r_in
    a_in_right = center_angle + (width / 2.0) / r_in
    a_out_left = center_angle - (width / 2.0) / r_out
    a_out_right = center_angle + (width / 2.0) / r_out
    return Polygon(
        [
            (r_in * math.cos(a_in_left), r_in * math.sin(a_in_left)),
            (r_out * math.cos(a_out_left), r_out * math.sin(a_out_left)),
            (r_out * math.cos(a_out_right), r_out * math.sin(a_out_right)),
            (r_in * math.cos(a_in_right), r_in * math.sin(a_in_right)),
        ]
    )


# hollow pillar: 1 cm wall ring with 16 narrow radial gaps (connected walkable area)
num_gaps = 4
gap_width = 0.1
slots = unary_union(
    [radial_slot(2.0 * math.pi * i / num_gaps, gap_width) for i in range(num_gaps)]
)
pillar = ring.difference(slots)
area = room.difference(pillar)

## Setup spawning area
num_agents = 10
spawning_area = room.difference(outer)
pos_in_spawning_area = jps.distributions.distribute_by_number(
    polygon=spawning_area,
    number_of_agents=num_agents,
    distance_to_agents=0.6,
    distance_to_polygon=0.5,
    seed=1,
)
exit_area = Point(0, 0).buffer(0.25, resolution=32)


## Setup Simulation
trajectory_file = "pillar_room.sqlite"
simulation = jps.Simulation(
    model=TwoLevelPedestrianModel(),
    geometry=area,
    trajectory_writer=jps.SqliteTrajectoryWriter(
        output_file=pathlib.Path(trajectory_file)
    ),
)

## Setup custom SQLite export
custom_db_path = pathlib.Path("pillar_room_custom_export.sqlite")
custom_conn = sqlite3.connect(custom_db_path)
custom_cur = custom_conn.cursor()
custom_cur.execute("DROP TABLE IF EXISTS trajectory_data")
custom_cur.execute(
    "CREATE TABLE trajectory_data ("
    "   frame INTEGER NOT NULL,"
    "   id INTEGER NOT NULL,"
    "   pos_x REAL NOT NULL,"
    "   pos_y REAL NOT NULL,"
    "   vel_x REAL NOT NULL,"
    "   vel_y REAL NOT NULL,"
    "   gs_pos_x REAL NOT NULL,"
    "   gs_pos_y REAL NOT NULL,"
    "   gs_vel_x REAL NOT NULL,"
    "   gs_vel_y REAL NOT NULL,"
    "   height REAL NOT NULL,"
    "   radius REAL NOT NULL)"
)
custom_cur.execute("DROP TABLE IF EXISTS metadata")
custom_cur.execute(
    "CREATE TABLE metadata(key TEXT NOT NULL UNIQUE PRIMARY KEY, value TEXT NOT NULL)"
)
custom_cur.execute("DROP TABLE IF EXISTS geometry")
custom_cur.execute(
    "CREATE TABLE geometry(" "   hash INTEGER NOT NULL, " "   wkt TEXT NOT NULL)"
)
custom_cur.execute("CREATE UNIQUE INDEX geometry_hash on geometry(hash)")

fps = 1 / simulation.delta_time() / 4
geo_wkt = simulation.get_geometry().as_wkt()
geo_hash = hash(geo_wkt)
xmin, ymin, xmax, ymax = from_wkt(geo_wkt).bounds

custom_cur.executemany(
    "INSERT INTO metadata VALUES(?, ?)",
    [
        ("version", "3"),
        ("fps", str(fps)),
        ("xmin", str(xmin)),
        ("ymin", str(ymin)),
        ("xmax", str(xmax)),
        ("ymax", str(ymax)),
    ],
)
custom_cur.execute(
    "INSERT INTO geometry VALUES(?, ?)",
    (geo_hash, geo_wkt),
)
custom_conn.commit()

exit_id = simulation.add_exit_stage(exit_area.exterior.coords[:-1])
journey = jps.JourneyDescription([exit_id])
journey_id = simulation.add_journey(journey)

## Spawn agents
v_distribution = np.clip(normal(1.34, 0.2, num_agents), 0.0, 1.8)

for pos, v0 in zip(pos_in_spawning_area, v_distribution):
    agent_id = simulation.add_agent(
        journey_id=journey_id,
        stage_id=exit_id,
        position=pos,
        state=TwoLevelPedestrianModelState(
            velocity=(0, 0),
            ground_support_position=pos,
            upper_body_position=pos,
            ground_support_velocity=(0, 0),
            height=1.65,
            mass=80,
            desired_speed=v0,
            reaction_time=0.5,
            agent_scale=2000,
            obstacle_scale=2000,
            force_distance=0.08,
            radius=0.3,
        ),
    )

# SocialForceModelIPP per-agent parameters (defaults & meaning)
# ==============================================================
# velocity                (0,0)     current velocity vector of upper body [m/s]
# ground_support_position (0,0)     position of the ground support circle [m]
# ground_support_velocity (0,0)     velocity of the ground support circle [m/s]
# height                  1.65      height of upper body center [m]
# mass                    80        mass [kg]
# desired_speed           1.34      preferred walking speed (v0) [m/s]
# reaction_time           0.5       reaction / relaxation time (tau) [s]
# agent_scale             2000      social repulsion amplitude vs agents (A) [N]
# obstacle_scale          2000      social repulsion amplitude vs walls (A_w) [N]
# force_distance          0.08      social falloff distance (B) [m]
# radius                  0.3       upper body radius (r) [m]
#
# Global model params:
# body_force              120000    contact stiffness (k) [kg s^-2]
# friction                240000    friction coefficient (kappa) [kg m^-1 s^-1]
# Compile-time constants:
# UNBALANCING_RATE        1.0       velocity control during locomotion
# DAMPING_RATE            0.5       dissipation during locomotion
# BALANCING_RATE          0.5       velocity control during recovery
# GS_SCALING_FACTOR       0.26/(2*0.3*1.65) ground support circle radius
# LEG_SCALING_FACTOR      0.5242    leg length [m]

## run simulation
max_iteration = 8000
every_nth_frame = 4


def write_custom_frame():
    """Record the current simulation state, using the same frame convention as
    jps.SqliteTrajectoryWriter: frame = iteration_count() // every_nth_frame,
    with frame 0 holding the initial (pre-iterate) state."""
    frame = simulation.iteration_count() // every_nth_frame
    for agent in simulation.agents():
        st = agent.state
        custom_cur.execute(
            "INSERT INTO trajectory_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                frame,
                agent.id,
                agent.position[0],
                agent.position[1],
                st.velocity[0],
                st.velocity[1],
                st.ground_support_position[0],
                st.ground_support_position[1],
                st.ground_support_velocity[0],
                st.ground_support_velocity[1],
                st.height,
                st.radius,
            ),
        )
    custom_conn.commit()


# frame 0 = initial state (jps.SqliteTrajectoryWriter writes it in begin_writing)
write_custom_frame()

info_every = 10
while simulation.agent_count() > 0 and simulation.iteration_count() < max_iteration:

    simulation.iterate()
    if simulation.iteration_count() % every_nth_frame == 0:
        write_custom_frame()
    if simulation.iteration_count() % info_every == 0:
        iteration = simulation.iteration_count()
        agents = simulation.agent_count()
        progress = 100.0 * iteration / max_iteration
        print(
            f"[{iteration:5d}/{max_iteration}] "
            f"agents: {agents:2d}  "
            f"progress: {progress:5.1f}%  "
            f"sim time: {iteration * simulation.delta_time():6.2f}s"
        )

custom_conn.commit()
custom_conn.close()

if simulation.iteration_count() == max_iteration:
    print("Simulation stopped after " + str(max_iteration) + " iterations.")
