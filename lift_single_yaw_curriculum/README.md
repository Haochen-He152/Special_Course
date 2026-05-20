# lift_single_yaw_curriculum

Single-object lift tasks with object yaw added to the policy observation. The yaw observation matches
`lift_xy_yaw`: the object yaw is encoded as `[sin(yaw), cos(yaw)]`.

## Training stages

1. Fixed cube position and fixed yaw:
   `Isaac-Lift-Single-Yaw-Curriculum-CubeFixed-Franka-v0`
   This task uses `skrl_cube_fixed_ppo_cfg.yaml` and enables the official lift-style smoothness curriculum at 10k steps.

2. Random cube XY position, fixed yaw:
   `Isaac-Lift-Single-Yaw-Curriculum-CubeRandomXY-Franka-v0`

3. Continue from the stage-2 cube model and train the six other objects with random XY, fixed yaw:
   - `Isaac-Lift-Single-Yaw-Curriculum-SugarBoxRandomXY-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-TomatoSoupCanRandomXY-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-MustardBottleRandomXY-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-WhiteCubeRandomXY-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-BlackCubeRandomXY-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-SmallTomatoSoupCanRandomXY-Franka-v0`

4. Continue from the stage-3 models and enable random yaw:
   - `Isaac-Lift-Single-Yaw-Curriculum-CubeRandomXYYaw-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-SugarBoxRandomXYYaw-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-TomatoSoupCanRandomXYYaw-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-MustardBottleRandomXYYaw-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-WhiteCubeRandomXYYaw-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-BlackCubeRandomXYYaw-Franka-v0`
   - `Isaac-Lift-Single-Yaw-Curriculum-SmallTomatoSoupCanRandomXYYaw-Franka-v0`

Each task also has a `-Play-v0` variant.

## SKRL logs and training length

- Generic SKRL config: `config/franka/agents/skrl_ppo_cfg.yaml`
- Fixed cube SKRL config: `config/franka/agents/skrl_cube_fixed_ppo_cfg.yaml`
- Logs: `lift_single_yaw_curriculum/logs/skrl`
- Generic trainer timesteps: `36000`
- Fixed cube trainer timesteps: `50000`

For `CubeFixed`, the curriculum changes these reward weights at `num_steps=25000`:

- `action_rate`: `-1e-4` to `-1e-1`
- `joint_vel`: `-1e-4` to `-1e-1`

The gripper-close shaping only penalizes closing while far from the object; it does not reward closing near the object.

## Reset ranges

- `fixed`: `x = 0`, `y = 0`, `yaw = 0` offset from the object's initial pose.
- `RandomXY`: `x in [-0.08, 0.08]`, `y in [-0.15, 0.15]`, `yaw = 0`.
- `RandomXYYaw`: same XY range plus `yaw in [-pi, pi]`.
