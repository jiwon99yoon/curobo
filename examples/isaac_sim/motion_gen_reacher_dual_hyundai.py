#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# ================================================================================
# DUAL_HDR35_20_ATI REACHER: motion_gen_reacher_dual_hyundai.py
# ================================================================================
# Based on: motion_gen_reacher_hyundai.py + multi_arm_reacher.py
# Modified by: Claude (dyros)
# Date: 2026-02-02
#
# FEATURES:
# 1. Dual HDR35_20_ATI configuration (RH56 + DG5F)
# 2. Chassis loaded at world origin (0, 0, 0)
# 3. Two target cubes: RH56 (RED), DG5F (BLUE)
# 4. GUI-based target control (drag cubes to set goals)
# 5. Multi-arm motion planning with link_poses
# 6. MESH collision checking with Hyundai chassis
#
# ROBOT POSITIONS (defined in dual_hdr35_20_ati URDF):
#   - RH56: (0.75, -1.3, -0.1) - Right side
#   - DG5F: (0.75,  1.3, -0.1) - Left side
#
# USAGE:
#   python examples/isaac_sim/motion_gen_reacher_dual_hyundai.py --visualize_spheres
#   python examples/isaac_sim/motion_gen_reacher_dual_hyundai.py --reactive --visualize_spheres
#
# ================================================================================

try:
    import isaacsim
except ImportError:
    pass

import torch
a = torch.zeros(4, device="cuda:0")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="dual_hdr35_20_ati.yml",
    help="robot configuration to load (default: dual_hdr35_20_ati.yml)"
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

args = parser.parse_args()

############################################################

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

from typing import Dict
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

from omni.isaac.core.utils.types import ArticulationAction

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

############################################################


def main():
    # Create world
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # ========================================
    # INITIAL CAMERA VIEW
    # ========================================
    from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(
        eye=[2.0, 0.0, 4.0],
        target=[0.75, 0.0, 0.0],
    )

    # ========================================
    # DUAL TARGET CUBES
    # ========================================
    # RH56 target (RED) - Right side robot
    # DG5F target (BLUE) - Left side robot
    # ========================================

    # Initial positions near each robot's workspace
    rh56_initial_pos = np.array([0.5, -1.0, 0.5])  # Near RH56 @ (0.75, -1.3, -0.1)
    dg5f_initial_pos = np.array([0.5, 1.0, 0.5])   # Near DG5F @ (0.75, 1.3, -0.1)

    # Euler (180, -90, 0) deg → quat (qw, qx, qy, qz) = (0, 0.7071, 0, -0.7071)
    gripper_orientation = np.array([0.0, 0.7071068, 0.0, -0.7071068])

    target_rh56 = cuboid.VisualCuboid(
        "/World/target_rh56",
        position=rh56_initial_pos,
        orientation=gripper_orientation,  # qw, qx, qy, qz
        color=np.array([1.0, 0.0, 0.0]),  # RED
        size=0.05,
    )

    target_dg5f = cuboid.VisualCuboid(
        "/World/target_dg5f",
        position=dg5f_initial_pos,
        orientation=gripper_orientation,  # qw, qx, qy, qz
        color=np.array([0.0, 0.5, 1.0]),  # BLUE
        size=0.05,
    )

    print("=" * 60)
    print("DUAL_HDR35_20_ATI REACHER")
    print("=" * 60)
    print("Target cubes:")
    print("  - RED cube  : RH56 (Right robot) end-effector target")
    print("  - BLUE cube : DG5F (Left robot) end-effector target")
    print("Drag cubes in Isaac Sim to set motion goals!")
    print("=" * 60)

    # Logger setup (suppress warp cache warnings)
    setup_curobo_logger("error")

    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    usd_help = UsdHelper()
    tensor_args = TensorDeviceType()

    # ========================================
    # LOAD DUAL_HDR35_20_ATI ROBOT CONFIG
    # ========================================
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    print(f"[DUAL_HDR35_20_ATI] Loading robot config: {args.robot}")
    print(f"[DUAL_HDR35_20_ATI] Joint names: {j_names}")
    print(f"[DUAL_HDR35_20_ATI] Default config: {default_config}")

    # Add robot to scene
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    # Empty world config (chassis added dynamically)
    world_cfg = WorldConfig(cuboid=[], mesh=[])

    # Motion gen configuration
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 24
    trim_steps = None
    max_attempts = 10
    interpolation_dt = 0.05
    enable_finetune_trajopt = True

    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=32,
        num_graph_seeds=32,
        num_ik_seeds=256,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
        use_cuda_graph=False,
        position_threshold=0.01,
        rotation_threshold=0.1,
    )
    motion_gen = MotionGen(motion_gen_config)

    if not args.reactive:
        print("[DUAL_HDR35_20_ATI] Warming up cuRobo...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("[DUAL_HDR35_20_ATI] cuRobo is Ready!")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    # ========================================
    # CHASSIS IS ALREADY IN DUAL_HDR35_20_ATI URDF
    # ========================================
    # dual_hdr35_20_ati URDF includes:
    #   - tilt_shassis (chassis mesh)
    #   - wall_1, wall_2, wall_3, wall_4
    #   - RH56 robot @ (0.75, -1.3, -0.1)
    #   - DG5F robot @ (0.75, 1.3, -0.1)
    # No need to load separate chassis USD!
    # ========================================
    print("[DUAL_HDR35_20_ATI] Chassis already included in dual_hdr35_20_ati URDF - no separate loading needed")

    # ========================================
    # WIRE USD LOADING
    # ========================================
    # Load wire/spring environment (chassis is already in URDF)
    # Position: centered with the chassis in URDF
    # ========================================
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    usd_base_path = join_path(get_assets_path(), "scene/hyundai_chassis")

    print("[DUAL_HDR35_20_ATI] Loading wire/spring USD...")
    add_reference_to_stage(
        usd_path=f"{usd_base_path}/env_diated_decomposed_chassis_tilted_srping_wire_flattened_no_rigid.usd",
        prim_path="/World/hyundai_wire"
    )

    stage = my_world.stage
    wire_prim = stage.GetPrimAtPath("/World/hyundai_wire")
    if wire_prim.IsValid():
        wire_pose = [0, 0, 0.1, 1, 0, 0, 0]  # [x, y, z, qw, qx, qy, qz]
        set_prim_transform(wire_prim, wire_pose)
        print(f"[DUAL_HDR35_20_ATI] Wire transform set: {wire_pose[:3]}")

        from pxr import UsdGeom, UsdPhysics

        def hide_chassis_keep_wire(prim, path=""):
            """Hide chassis components, keep wire/spring/ring visible"""
            current_path = path + "/" + prim.GetName() if path else prim.GetName()
            name_lower = prim.GetName().lower()

            chassis_keywords = ["chassis", "frame", "strut", "tilt"]
            is_chassis = any(kw in name_lower for kw in chassis_keywords)

            wire_keywords = ["wire", "spring", "ring", "bar"]
            is_wire = any(kw in name_lower for kw in wire_keywords)

            if is_chassis and not is_wire:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()

            if is_wire and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigidBodyAPI = UsdPhysics.RigidBodyAPI(prim)
                if "ring" in name_lower:
                    rigidBodyAPI.CreateKinematicEnabledAttr().Set(False)
                else:
                    rigidBodyAPI.CreateKinematicEnabledAttr().Set(True)

            for child in prim.GetChildren():
                hide_chassis_keep_wire(child, current_path)

        hide_chassis_keep_wire(wire_prim)
        print("[DUAL_HDR35_20_ATI] Chassis hidden, wire visible")
    else:
        print("[DUAL_HDR35_20_ATI] Warning: Could not find wire prim!")

    print("[DUAL_HDR35_20_ATI] Wire/spring loaded successfully!")

    # ========================================
    # MULTI-ARM SETUP
    # ========================================
    # Get link names from motion gen (includes both EE links)
    link_names = motion_gen.kinematics.link_names
    ee_link_name = motion_gen.kinematics.ee_link  # Primary EE (rh56_gripper_base_link)

    print(f"[DUAL_HDR35_20_ATI] EE link names: {link_names}")
    print(f"[DUAL_HDR35_20_ATI] Primary EE: {ee_link_name}")

    # Get retract pose for all links
    kin_state = motion_gen.kinematics.get_state(motion_gen.get_retract_config().view(1, -1))
    link_retract_pose = kin_state.link_pose

    # Map link names to targets
    # link_names should be: ["rh56_gripper_base_link", "dg5f_ll_dg_palm"]
    link_to_target = {}
    for link_name in link_names:
        if "rh56" in link_name.lower():
            link_to_target[link_name] = target_rh56
            print(f"[DUAL_HDR35_20_ATI] Mapped {link_name} -> RED target (RH56)")
        elif "dg5f" in link_name.lower():
            link_to_target[link_name] = target_dg5f
            print(f"[DUAL_HDR35_20_ATI] Mapped {link_name} -> BLUE target (DG5F)")

    # ========================================
    # MAIN LOOP VARIABLES
    # ========================================
    cmd_plan = None
    cmd_idx = 0
    # Ground plane optional - chassis has its own floor
    # my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None

    # Tracking for both targets
    past_pose_rh56 = None
    past_pose_dg5f = None
    target_pose_rh56 = None
    target_pose_dg5f = None
    past_orientation_rh56 = None
    past_orientation_dg5f = None
    target_orientation_rh56 = None
    target_orientation_dg5f = None

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index

        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()

        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]),
                joint_indices=idx_list
            )
        if step_index < 20:
            continue

        # ========================================
        # OBSTACLE UPDATE (every 2000 frames)
        # ========================================
        # Note: Chassis is part of robot URDF, so we scan /World for external obstacles
        # The robot (including chassis) is excluded via ignore_substring
        if step_index == 50 or step_index % 2000 == 0:
            print(f"[Step {step_index}] Updating world obstacles...")

            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target_rh56",
                    "/World/target_dg5f",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()

            motion_gen.update_world(obstacles)
            print(f"[DUAL_HDR35_20_ATI] Updated world with {len(obstacles.objects)} obstacles")

        # ========================================
        # GET CURRENT JOINT STATE
        # ========================================
        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        sim_js_names = robot.dof_names

        if np.any(np.isnan(sim_js.positions)):
            log_error("NAN joint position values detected!")
            continue

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration

        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # ========================================
        # VISUALIZE SPHERES
        # ========================================
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        # ========================================
        # GET TARGET POSES
        # ========================================
        cube_pos_rh56, cube_ori_rh56 = target_rh56.get_world_pose()
        cube_pos_dg5f, cube_ori_dg5f = target_dg5f.get_world_pose()

        # Initialize tracking
        if past_pose_rh56 is None:
            past_pose_rh56 = cube_pos_rh56
        if past_pose_dg5f is None:
            past_pose_dg5f = cube_pos_dg5f
        if target_pose_rh56 is None:
            target_pose_rh56 = cube_pos_rh56
        if target_pose_dg5f is None:
            target_pose_dg5f = cube_pos_dg5f
        if past_orientation_rh56 is None:
            past_orientation_rh56 = cube_ori_rh56
        if past_orientation_dg5f is None:
            past_orientation_dg5f = cube_ori_dg5f
        if target_orientation_rh56 is None:
            target_orientation_rh56 = cube_ori_rh56
        if target_orientation_dg5f is None:
            target_orientation_dg5f = cube_ori_dg5f

        robot_static = (np.max(np.abs(sim_js.velocities)) < 2.0) or args.reactive

        # Check if any target moved
        rh56_moved = (
            np.linalg.norm(cube_pos_rh56 - target_pose_rh56) > 1e-3 or
            np.linalg.norm(cube_ori_rh56 - target_orientation_rh56) > 1e-3
        )
        dg5f_moved = (
            np.linalg.norm(cube_pos_dg5f - target_pose_dg5f) > 1e-3 or
            np.linalg.norm(cube_ori_dg5f - target_orientation_dg5f) > 1e-3
        )

        rh56_stopped = (
            np.linalg.norm(past_pose_rh56 - cube_pos_rh56) == 0.0 and
            np.linalg.norm(past_orientation_rh56 - cube_ori_rh56) == 0.0
        )
        dg5f_stopped = (
            np.linalg.norm(past_pose_dg5f - cube_pos_dg5f) == 0.0 and
            np.linalg.norm(past_orientation_dg5f - cube_ori_dg5f) == 0.0
        )

        targets_changed = (rh56_moved or dg5f_moved) and rh56_stopped and dg5f_stopped

        # ========================================
        # MULTI-ARM PLANNING
        # ========================================
        if targets_changed and robot_static:
            print("\n" + "=" * 60)
            print("[DUAL_HDR35_20_ATI] Planning dual-arm motion...")
            print(f"  RH56 target: pos={cube_pos_rh56}, ori={cube_ori_rh56}")
            print(f"  DG5F target: pos={cube_pos_dg5f}, ori={cube_ori_dg5f}")
            print("=" * 60)

            # Primary goal (RH56 - primary EE)
            ik_goal = Pose(
                position=tensor_args.to_device(cube_pos_rh56),
                quaternion=tensor_args.to_device(cube_ori_rh56),
            )

            # Secondary link poses (DG5F and any other links)
            link_poses = {}
            for link_name in link_names:
                if link_name != ee_link_name:  # Skip primary EE
                    if "dg5f" in link_name.lower():
                        link_poses[link_name] = Pose(
                            position=tensor_args.to_device(cube_pos_dg5f),
                            quaternion=tensor_args.to_device(cube_ori_dg5f),
                        )
                        print(f"  Added link pose for: {link_name}")

            # Plan with link_poses for multi-arm
            result = motion_gen.plan_single(
                cu_js.unsqueeze(0),
                ik_goal,
                plan_config.clone(),
                link_poses=link_poses
            )

            succ = result.success.item()
            print(f"[DUAL_HDR35_20_ATI] Planning result: Success={succ}, Status={result.status}")

            if succ:
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)

                # Get common joint names
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
                print(f"[DUAL_HDR35_20_ATI] Trajectory generated: {len(cmd_plan.position)} waypoints")
            else:
                carb.log_warn(f"Planning failed: {result.status}")
                print(f"[DUAL_HDR35_20_ATI] Planning FAILED - try a different target position")

            # Update tracking
            target_pose_rh56 = cube_pos_rh56
            target_pose_dg5f = cube_pos_dg5f
            target_orientation_rh56 = cube_ori_rh56
            target_orientation_dg5f = cube_ori_dg5f

        # Update past poses
        past_pose_rh56 = cube_pos_rh56
        past_pose_dg5f = cube_pos_dg5f
        past_orientation_rh56 = cube_ori_rh56
        past_orientation_dg5f = cube_ori_dg5f

        # ========================================
        # EXECUTE TRAJECTORY
        # ========================================
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()

            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_idx += 1

            for _ in range(2):
                my_world.step(render=False)

            if cmd_idx >= len(cmd_plan.position):
                print("[DUAL_HDR35_20_ATI] Trajectory execution complete!")
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None

    simulation_app.close()


if __name__ == "__main__":
    main()
