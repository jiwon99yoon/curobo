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
# HYDEX DUAL-ARM COMMAND VERSION: motion_gen_reacher_dual_command.py
# ================================================================================
# Based on: motion_gen_reacher_command.py + motion_gen_reacher_dual_hyundai.py
# Date: 2026-02-03
#
# DESCRIPTION:
#   Dual-arm motion planning for HYDEX system (두 대의 HDR35_20 로봇).
#   터미널에서 좌표를 입력하면 cuRobo가 양팔 동시 모션 플래닝을 수행합니다.
#   RH56 좌표 입력 후 DG5F 좌표를 입력하면 플래닝이 시작됩니다.
#
# FEATURES:
#   1. HYDEX dual-arm (RH56 + DG5F) with command input
#   2. Two-step input: RH56 먼저 입력 → DG5F 입력 시 planning 시작
#   3. Format: x y z qw qx qy qz (world frame coordinates)
#   4. Chassis included in URDF (no separate loading)
#   5. Wire/spring USD 자동 로드 (chassis 중복 숨김 처리)
#
# ROBOT POSITIONS (in hydex.urdf, relative to world):
#   - RH56 (오른쪽, y-): (0.75, -1.3, -0.1)
#   - DG5F (왼쪽, y+):  (0.75,  1.3, -0.1)
#
# USAGE:
#   python examples/isaac_sim/motion_gen_reacher_dual_command.py --visualize_spheres
#
# ================================================================================
# VERIFIED WORKING COORDINATES (실험적으로 검증된 좌표)
# ================================================================================
# Isaac Sim에서 Play 버튼을 누른 후 아래 좌표를 입력하세요.
#
# [RH56 examples] (x y z qw qx qy qz):
#   0.1 -0.3 0.6 0.0 0.7071 0.0 -0.7071
#   1 -0.5 0.5 0.0 0.7071 0.0 -0.7071
#   0.3 -0.5 0.6 0.0 0.7071 0.0 -0.7071
#   0.3 -0.3 0.6 0.0 0.7071 0.0 -0.7071
#   0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071
#
# [DG5F examples] (x y z qw qx qy qz):
#   0.31 0.2 0.8 0.0 0.7071 0.0 -0.7071
#   1 0.5 0.5 0.0 0.7071 0.0 -0.7071
#   0.3 0.5 0.6 0.0 0.7071 0.0 -0.7071
#   0.3 0.3 0.6 0.0 0.7071 0.0 -0.7071
#   0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071
#
# ================================================================================
# COLLISION SETTINGS (충돌 설정)
# ================================================================================
# Config file: IsaacLab/src/nvidia-curobo/src/curobo/content/configs/robot/dual_hdr35_20_ati.yml
#
# 주요 설정:
#   - collision_sphere_buffer: -0.01
#       → 전체 collision sphere 반지름 조정 (음수=축소, 양수=확대)
#
#   - self_collision_buffer: HAND 링크만 -0.01 설정
#       → ARM 링크: 기본값(0) 사용 → inter-arm collision 감지 O
#       → HAND 링크: -0.01 → 손가락 내부 충돌 무시 (반지름이 작아서 필요)
#
# 문제 발생 시:
#   - INVALID_START_STATE_SELF_COLLISION: self_collision_buffer 값을 더 음수로
#   - Inter-arm collision 미감지: ARM 링크의 buffer를 0 또는 양수로 유지
#   - IK_FAIL: 좌표가 로봇 workspace 밖일 수 있음, 더 가까운 좌표 시도
#
# ================================================================================
# COORDINATE CONVERSION (from single-robot to dual-robot)
# ================================================================================
#   - RH56: new_pos = old_pos + (0.75, -1.3, -0.1)
#   - DG5F: new_pos = old_pos + (0.75,  1.3, -0.1)
#
# ================================================================================

try:
    import isaacsim
except ImportError:
    pass

import torch
a = torch.zeros(4, device="cuda:0")

import argparse
import threading
import queue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket]",
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


def input_listener_thread(rh56_queue, dg5f_queue, stop_event, execution_event):
    """
    Two-step input listener:
    1. Enter RH56 target (stored, not executed)
    2. Enter DG5F target (triggers planning with both targets)
    """
    import time

    print("\n" + "="*70)
    print("HYDEX DUAL-ARM COMMAND MODE")
    print("="*70)
    print("Input format: x y z qw qx qy qz (world frame coordinates)")
    print("")
    print("Workflow:")
    print("  1. Enter RH56 target first")
    print("  2. Enter DG5F target to trigger dual-arm planning")
    print("")
    print("Coordinate Reference (relative to world/chassis origin):")
    print("  - RH56 robot base: (0.75, -1.3, -0.1)")
    print("  - DG5F robot base: (0.75,  1.3, -0.1)")
    print("")
    print("Type 'q' to quit")
    print("="*70 + "\n")

    rh56_target = None

    while not stop_event.is_set():
        try:
            # Wait if execution is in progress
            if execution_event.is_set():
                time.sleep(0.1)
                continue

            if rh56_target is None:
                user_input = input("\n[RH56] Enter target (x y z qw qx qy qz): ")
            else:
                user_input = input("\n[DG5F] Enter target (x y z qw qx qy qz) to START planning: ")

            if user_input.lower() == 'q':
                print("[Command] Quit command received. Exiting...")
                stop_event.set()
                break

            coords = [float(x) for x in user_input.split()]

            if len(coords) != 7:
                print("Invalid format! Use: x y z qw qx qy qz (7 numbers)")
                continue

            pos = coords[:3]
            quat = coords[3:]

            # Normalize quaternion
            quat_norm = sum(q**2 for q in quat) ** 0.5
            if abs(quat_norm - 1.0) > 0.1:
                print(f"Warning: Quaternion norm is {quat_norm:.3f}, normalizing...")
                quat = [q / quat_norm for q in quat]

            target = {
                'position': pos,
                'quaternion': quat,
            }

            if rh56_target is None:
                # Step 1: Store RH56 target
                rh56_target = target
                print(f"RH56 target stored:")
                print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"  Quaternion: [qw={quat[0]:.3f}, qx={quat[1]:.3f}, qy={quat[2]:.3f}, qz={quat[3]:.3f}]")
                print(f"  -> Now enter DG5F target to start planning")
            else:
                # Step 2: Send both targets to trigger planning
                dg5f_target = target
                print(f"DG5F target received:")
                print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"  Quaternion: [qw={quat[0]:.3f}, qx={quat[1]:.3f}, qy={quat[2]:.3f}, qz={quat[3]:.3f}]")

                # Send both targets
                rh56_queue.put(rh56_target)
                dg5f_queue.put(dg5f_target)
                print(f"\nPlanning started for dual-arm motion!")
                execution_event.set()  # Block input during execution

                # Reset for next planning
                rh56_target = None

        except ValueError:
            print("Invalid input! Use numbers only")
        except EOFError:
            print("\n[Command] EOF detected. Exiting...")
            stop_event.set()
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    # Threading components
    rh56_queue = queue.Queue()
    dg5f_queue = queue.Queue()
    stop_event = threading.Event()
    execution_event = threading.Event()  # Block input during execution

    # Start input listener
    input_thread = threading.Thread(
        target=input_listener_thread,
        args=(rh56_queue, dg5f_queue, stop_event, execution_event),
        daemon=True
    )
    input_thread.start()

    # State machine
    planning_state = "IDLE"
    pending_rh56 = None
    pending_dg5f = None
    current_trajectory = None
    trajectory_index = 0
    idx_list = []
    common_js_names = []
    wait_start_step = 0  # For timeout in WAITING_ROBOT

    # Create world
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # Target visualization cubes
    target_rh56 = cuboid.VisualCuboid(
        "/World/target_rh56",
        position=np.array([0.5, -1.0, 0.5]),
        orientation=np.array([1, 0, 0, 0]),
        color=np.array([1.0, 0.0, 0.0]),  # RED
        size=0.05,
    )

    target_dg5f = cuboid.VisualCuboid(
        "/World/target_dg5f",
        position=np.array([0.5, 1.0, 0.5]),
        orientation=np.array([1, 0, 0, 0]),
        color=np.array([0.0, 0.5, 1.0]),  # BLUE
        size=0.05,
    )

    setup_curobo_logger("error")

    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    usd_help = UsdHelper()
    tensor_args = TensorDeviceType()

    # Load robot config
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    print(f"[HYDEX] Loading robot config: {args.robot}")
    print(f"[HYDEX] Joint names: {j_names}")

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    world_cfg = WorldConfig(cuboid=[], mesh=[])

    # Motion gen config
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=32,
        num_graph_seeds=32,
        num_ik_seeds=256,
        interpolation_dt=0.05,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_cuda_graph=False,
        position_threshold=0.01,
        rotation_threshold=0.1,
    )
    motion_gen = MotionGen(motion_gen_config)

    print("[HYDEX] Warming up cuRobo...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[HYDEX] cuRobo is Ready!")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=10,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    print("[HYDEX] Chassis already included in hydex.urdf")

    # ========================================
    # WIRE USD LOADING
    # ========================================
    # Load wire/spring environment (chassis is already in URDF)
    # Position: centered with the chassis in URDF
    # ========================================
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
        print("[HYDEX] Using Isaac Sim 5.0.0 API")
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage
        print("[HYDEX] Using legacy API")

    # 변경: 절대 경로 → cuRobo 표준 scene 경로 (이식성 향상, 2026-02-13)
    usd_base_path = join_path(get_assets_path(), "scene/hyundai_chassis")
    # 기존 절대 경로 (참고용):
    # usd_base_path = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd"

    # Load wire/spring USD
    # HYDEX: chassis is at (0, 0, 0.1) in URDF, robots at (0.75, +-1.3, -0.1)
    # Wire USD should be at origin (aligned with chassis in URDF)
    print("[HYDEX] Loading wire/spring USD...")
    add_reference_to_stage(
        usd_path=f"{usd_base_path}/env_diated_decomposed_chassis_tilted_srping_wire_flattened_no_rigid.usd",
        prim_path="/World/hyundai_wire"
    )

    # Set position to match HYDEX coordinate system
    # The wire USD is designed for single-robot at origin
    # For HYDEX, we need to offset it to match chassis position
    wire_prim = stage.GetPrimAtPath("/World/hyundai_wire")
    if wire_prim.IsValid():
        # Wire offset: x=0 (centered), y=0 (centered), z=0.1 (match chassis z)
        # No xy offset needed since chassis is at origin in HYDEX
        wire_pose = [0, 0, 0.1, 1, 0, 0, 0]  # [x, y, z, qw, qx, qy, qz]
        set_prim_transform(wire_prim, wire_pose)
        print(f"[HYDEX] Wire transform set: {wire_pose[:3]}")

        # Hide chassis from wire USD (already in URDF)
        # Only keep wire/spring/ring visible
        from pxr import UsdGeom, UsdPhysics

        def hide_chassis_keep_wire(prim, path=""):
            """Hide chassis components, keep wire/spring/ring visible"""
            current_path = path + "/" + prim.GetName() if path else prim.GetName()
            name_lower = prim.GetName().lower()

            # Hide chassis-related prims (already in URDF)
            chassis_keywords = ["chassis", "frame", "strut", "tilt"]
            is_chassis = any(kw in name_lower for kw in chassis_keywords)

            # Keep wire-related prims visible
            wire_keywords = ["wire", "spring", "ring", "bar"]
            is_wire = any(kw in name_lower for kw in wire_keywords)

            if is_chassis and not is_wire:
                # Hide this prim
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                    print(f"[HYDEX] Hidden: {current_path}")

            # Set physics for wire parts (keep dynamic for rings)
            if is_wire and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigidBodyAPI = UsdPhysics.RigidBodyAPI(prim)
                if "ring" in name_lower:
                    # Rings can move (for RevoluteJoint)
                    rigidBodyAPI.CreateKinematicEnabledAttr().Set(False)
                else:
                    # Wire/spring/bar are kinematic
                    rigidBodyAPI.CreateKinematicEnabledAttr().Set(True)

            for child in prim.GetChildren():
                hide_chassis_keep_wire(child, current_path)

        hide_chassis_keep_wire(wire_prim)
        print("[HYDEX] Chassis hidden, wire visible")
    else:
        print("[HYDEX] Warning: Could not find wire prim!")

    print("[HYDEX] Wire/spring loaded successfully!")

    # Get link names
    link_names = motion_gen.kinematics.link_names
    ee_link_name = motion_gen.kinematics.ee_link

    print(f"[HYDEX] EE link names: {link_names}")
    print(f"[HYDEX] Primary EE: {ee_link_name}")

    # Main loop variables
    i = 0
    spheres = None
    past_cmd = None

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

        # Obstacle update
        if step_index == 50 or step_index % 2000 == 0:
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

        # Get joint state
        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        sim_js_names = robot.dof_names

        if np.any(np.isnan(sim_js.positions)):
            continue

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # Visualize spheres
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

        robot_static = np.max(np.abs(sim_js.velocities)) < 5.0  # Increased threshold for dual-arm

        # Check for quit
        if stop_event.is_set():
            break

        # State machine
        if planning_state == "IDLE":
            if not rh56_queue.empty() and not dg5f_queue.empty():
                pending_rh56 = rh56_queue.get()
                pending_dg5f = dg5f_queue.get()
                wait_start_step = step_index
                planning_state = "WAITING_ROBOT"

        elif planning_state == "WAITING_ROBOT":
            wait_elapsed = step_index - wait_start_step
            # Proceed if robot is static OR timeout (300 steps ~ 5 sec)
            if robot_static or wait_elapsed > 300:
                if wait_elapsed > 300:
                    print(f"[TIMEOUT] Proceeding to planning after {wait_elapsed} steps")
                planning_state = "PLANNING"
            elif step_index % 100 == 0:
                print(f"[WAITING] Robot velocity: {np.max(np.abs(sim_js.velocities)):.4f} (step {wait_elapsed}/300)")

        elif planning_state == "PLANNING":
            rh56_pos = np.array(pending_rh56['position'])
            rh56_quat = np.array(pending_rh56['quaternion'])
            dg5f_pos = np.array(pending_dg5f['position'])
            dg5f_quat = np.array(pending_dg5f['quaternion'])

            # Update target visualization
            target_rh56.set_world_pose(position=rh56_pos, orientation=rh56_quat)
            target_dg5f.set_world_pose(position=dg5f_pos, orientation=dg5f_quat)

            print("\n" + "="*60)
            print("[PLANNING] Dual-arm motion...")
            print(f"  RH56: pos={rh56_pos}, quat={rh56_quat}")
            print(f"  DG5F: pos={dg5f_pos}, quat={dg5f_quat}")
            print("="*60)

            # Primary goal (RH56)
            ik_goal = Pose(
                position=tensor_args.to_device(rh56_pos),
                quaternion=tensor_args.to_device(rh56_quat),
            )

            # Secondary link poses (DG5F)
            link_poses = {}
            for link_name in link_names:
                if link_name != ee_link_name and "dg5f" in link_name.lower():
                    link_poses[link_name] = Pose(
                        position=tensor_args.to_device(dg5f_pos),
                        quaternion=tensor_args.to_device(dg5f_quat),
                    )

            result = motion_gen.plan_single(
                cu_js.unsqueeze(0),
                ik_goal,
                plan_config.clone(),
                link_poses=link_poses
            )

            if result.success.item():
                print(f"[SUCCESS] Planning succeeded!")
                current_trajectory = result.get_interpolated_plan()
                current_trajectory = motion_gen.get_full_js(current_trajectory)

                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in current_trajectory.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                current_trajectory = current_trajectory.get_ordered_joint_state(common_js_names)
                trajectory_index = 0
                planning_state = "EXECUTING"
            else:
                print(f"[FAILED] Status: {result.status}")
                pending_rh56 = None
                pending_dg5f = None
                planning_state = "IDLE"
                execution_event.clear()  # Allow input again

        elif planning_state == "EXECUTING":
            if current_trajectory is not None and trajectory_index < len(current_trajectory.position):
                cmd_state = current_trajectory[trajectory_index]
                past_cmd = cmd_state.clone()

                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )
                articulation_controller.apply_action(art_action)

                for _ in range(2):
                    my_world.step(render=False)

                trajectory_index += 1

                if trajectory_index % 20 == 0:
                    progress = (trajectory_index / len(current_trajectory.position)) * 100
                    print(f"[EXECUTING] {progress:.1f}%")
            else:
                print("[COMPLETE] Execution finished!\n")
                current_trajectory = None
                trajectory_index = 0
                pending_rh56 = None
                pending_dg5f = None
                planning_state = "IDLE"
                execution_event.clear()  # Allow input again

    simulation_app.close()


if __name__ == "__main__":
    main()
