# lift_single_yaw_curriculum

Single-object lift tasks with object yaw added to the policy observation. The yaw observation matches
`lift_xy_yaw`: the object yaw is encoded as `[sin(yaw), cos(yaw)]`.

## Training stages

1. Fixed cube position and fixed yaw:
   `Isaac-Lift-Single-Yaw-Curriculum-CubeFixed-Franka-v0`

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

## Reset ranges

- `fixed`: `x = 0`, `y = 0`, `yaw = 0` offset from the object's initial pose.
- `RandomXY`: `x in [-0.08, 0.08]`, `y in [-0.15, 0.15]`, `yaw = 0`.
- `RandomXYYaw`: same XY range plus `yaw in [-pi, pi]`.
