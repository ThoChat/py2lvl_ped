# py2lvl_ped

Python implementation of a **two-level pedestrian model with social forces**, built as a custom operational model for the [Jupedsim](https://github.com/DLR-RM/jupedsim) framework.

Instead of a single position per agent, each pedestrian is modelled as a two-part system:

- **Upper body** — position + velocity, driven by Helbing-style social forces (desired-direction force, exponential repulsion from neighbours and walls) plus contact forces.
- **Ground support** — a smaller circle at foot level with its own position and velocity.

The two levels are coupled through the unit vector `e_gs_ub` (ground support → upper body): an *unbalancing* term acts on the upper body while a *balancing* term acts on the ground support, producing natural locomotion and recovery behaviour. Parameters follow the C++ reference `SocialForceModelIPP`.

## Files

| File | Description |
|---|---|
| `py2lvl_ped_and_social_force.py` | `TwoLevelPedestrianModel` and `TwoLevelPedestrianModelState` (Jupedsim custom model) |
| `sim_pillar_room.py` | Demo scenario: agents walk from a room into a hollow pillar with radial gaps, towards a central exit; writes two SQLite trajectory files |
| `visualiser_2lvl_ped.py` | Interactive 3D viewer for the recorded trajectories |
| `compile_and_run.sh` | Compiles Jupedsim with Ninja, sets up the environment, runs a simulation script |

## Prerequisites

- A local checkout of **Jupedsim** in the sibling folder `../jupedsim`, configured with a `build/` directory (CMake + Ninja) and an `environment` file in `build/`.
- The Python environment in `.venv-python-model` with `numpy`, `shapely` and `matplotlib` (`ffmpeg` additionally needed for video export from the visualiser).

## Quick start

Run the pillar-room simulation (Jupedsim is compiled first):

```bash
./compile_and_run.sh sim_pillar_room.py
```

This produces:

- `pillar_room.sqlite` — trajectory in the standard Jupedsim format
- `pillar_room_custom_export.sqlite` — extended export including the ground-support state, used by the visualiser

Visualise the results:

```bash
python visualiser_2lvl_ped.py pillar_room_custom_export.sqlite
```

An interactive 3D view opens showing the upper-body circles (z = 1.5 m), the ground-support circles (ground level), the blue links between them, and the room geometry. Step through frames with the slider or Play/Pause button. Optionally save an mp4:

```bash
python visualiser_2lvl_ped.py pillar_room_custom_export.sqlite output.mp4
```

## Model

`TwoLevelPedestrianModel.compute_next_state` works as follows:

1. **Social forces** (divided by mass): desired-direction force toward the next target, plus exponential repulsion `A · exp((r₁+r₂−d)/B)` from agents within 2.5 m and from walls within 5 m.
2. **Contact forces** (not divided by mass): elastic normal force when circles overlap, computed separately at upper-body level (radius `r`) and ground-support level (radius `r · s · h`).
3. **Update** (Euler integration):
   - upper-body velocity: `v += (F_social + F_contact + (e − v)·γ_u − v·γ_d) · dt`
   - ground-support velocity: `v_gs += (F_contact,gs + (e − v_gs)·γ_b) · dt`

`e` is the unit vector from ground support to upper body (`e_gs_ub`), falling back to `(1, 0)` when both coincide.

### Parameters

Per-agent state (`TwoLevelPedestrianModelState`):

| Parameter | Default | Meaning |
|---|---|---|
| `velocity` | (0, 0) | Upper-body velocity [m/s] |
| `ground_support_position` | (0, 0) | Ground-support position [m] |
| `ground_support_velocity` | (0, 0) | Ground-support velocity [m/s] |
| `desired_speed` | 1.34 | Preferred walking speed [m/s] |
| `reaction_time` | 0.5 | Relaxation time τ [s] |
| `agent_scale` | 2000 | Social repulsion amplitude vs. agents (A) [N] |
| `obstacle_scale` | 2000 | Social repulsion amplitude vs. walls (A_w) [N] |
| `force_distance` | 0.08 | Social repulsion falloff distance (B) [m] |
| `mass` | 80 | Body mass [kg] |
| `radius` | 0.3 | Upper-body radius [m] |
| `height` | 1.65 | Upper-body centre height [m] |
| `body_force`, `friction` | 120000, 240000 | Defined for reference; tangential friction is not active in the current implementation |

Global model parameters (keyword arguments of `TwoLevelPedestrianModel`, settable at simulation creation): `unbalancing_rate = 1.0`, `damping_rate = 0.5`, `balancing_rate = 0.5`, `gs_scaling_factor = 0.26 / (2 · 0.3 · 1.65)`.

`check_model_constraint` rejects non-positive mass/speed/reaction time/radius, agents closer than their radius to a neighbour, and agents within `radius / 2` of a wall.

## Trajectory file format (custom export)

The visualiser reads a SQLite database with three tables:

- `trajectory_data` — `frame, id, pos_x, pos_y, vel_x, vel_y, gs_pos_x, gs_pos_y, gs_vel_x, gs_vel_y, height, radius`
- `metadata` — `version`, `fps`, `xmin`, `ymin`, `xmax`, `ymax`
- `geometry` — geometry as WKT with a `hash` column

Frames follow the Jupedsim convention: frame 0 holds the initial state, and one frame is written every `every_nth_frame` (4) iterations.

## License

MIT — see [LICENSE](LICENSE).
