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
# INTERACTIVE COMMAND VERSION: motion_gen_reacher_command.py
# ================================================================================
# Based on: motion_gen_reacher_hyundai.py
# Modified by: Claude (dyros)
# Date: 2026-01-05
#
# NEW FEATURES (COMMAND MODE):
# 1. Interactive command input via threading (non-blocking)
# 2. State machine for planning workflow (IDLE → WAITING_ROBOT → PLANNING → EXECUTING)
# 3. 2-step Enter: 1) Target recognition, 2) Planning confirmation
# 4. Quaternion input for full 6-DOF pose control (x y z qw qx qy qz)
#    - Euler angle input also implemented (commented out for easy switching)
# 5. Absolute/Relative control modes (--control_mode [absolute|relative])
#    - Absolute: World frame coordinates
#    - Relative: Current EE frame offsets
# 6. Real-time target visualization (RED cube updates with user input)
# 7. EE position visualization (BLUE sphere showing current end-effector position)
# 8. Success/Fail feedback for each planning attempt
# 9. Trajectory execution progress tracking
#
# INHERITED FEATURES (from motion_gen_reacher_hyundai.py):
# - MESH collision checking (CollisionCheckerType.MESH)
# - Hyundai USD environment loading with selective physics
# - Dynamic obstacle support (RevoluteJoint tracking)
# - Isaac Sim 5.0.0 compatibility
#
# USAGE:
#   # Absolute mode (default - world frame coordinates)
#   python examples/isaac_sim/motion_gen_reacher_command.py \
#       --robot hdr35_20_rh56f1_r.yml \
#       --visualize_spheres \
#       --control_mode absolute
#
#   # Relative mode (current EE frame offsets)
#   python examples/isaac_sim/motion_gen_reacher_command.py \
#       --robot hdr35_20_rh56f1_r.yml \
#       --control_mode relative
#
#   Then enter target poses interactively (2-step Enter):
#   >>> Enter target (x y z qw qx qy qz) or 'q': 0.5 0.3 0.8 1.0 0.0 0.0 0.0
#   ✓ Target recognized:
#     Position: [0.500, 0.300, 0.800]
#     Quaternion: [qw=1.000, qx=0.000, qy=0.000, qz=0.000]
#     → Press Enter to confirm and start planning
#   >>> [Press Enter]
#   ✓ Planning started for target: {...}
#   ✓ Planning SUCCESS!
#   [Executing] Progress: 50.0% (120/240)
#   ✓ Execution complete!
#   ...
#
# STATE MACHINE:
#   IDLE → User입력 대기
#   WAITING_ROBOT → Robot 정지 확인
#   PLANNING → Motion planning 실행
#   EXECUTING → Trajectory 실행
#   → IDLE (다시 입력 대기)
#
# TESTED WITH:
#   - Robot: ur5e_robotiq_2f_140.yml
#   - Environment: Hyundai front chassis (샤시만)
#   - Mode: Interactive command-based planning
# ================================================================================
#


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse
import threading
import queue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
# ========================================
# ROBOT LOADING 관련 인자
# ========================================
# --robot: Robot config YAML 파일명
#   기본 위치: {curobo}/src/curobo/content/configs/robot/
#   예: --robot iiwa_allegro_mocked.yml
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

# --external_asset_path: URDF/USD/meshes가 있는 외부 디렉토리
#   이 경로를 지정하면 cuRobo 내장 assets 대신 외부 파일 사용
#   예: --external_asset_path /home/user/my_robot_assets
#
#   사용 시나리오:
#   1. 커스텀 로봇 사용
#   2. 수정된 URDF/meshes 테스트
#   3. 다른 프로젝트의 robot assets 재사용
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)

# --external_robot_configs_path: Robot config YAML이 있는 외부 디렉토리
#   기본 configs/ 대신 다른 위치의 config 사용
#   예: --external_robot_configs_path /home/user/my_configs
#
#   사용 시나리오:
#   1. cuRobo 소스 수정 없이 custom config 사용
#   2. 여러 config 버전 관리
#   3. 프로젝트별 robot config 분리
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--control_mode",
    type=str,
    choices=["absolute", "relative"],
    default="absolute",
    help="Control mode: 'absolute' (world frame) or 'relative' (current EE frame)",
)
# Reactive mode removed - using interactive command mode instead
# parser.add_argument(
#     "--reactive",
#     action="store_true",
#     help="When True, runs in reactive mode",
#     default=False,
# )

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)


args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
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
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

############################################################


########### OV #################;;;;;


# ========================================
# INTERACTIVE COMMAND INPUT (Threading)
# ========================================
def input_listener_thread(target_queue, stop_event, control_mode):
    """
    별도 thread에서 실행되는 사용자 입력 리스너
    - blocking input()이 시뮬레이션을 방해하지 않도록 분리
    - Queue를 통해 main thread에 target 전달
    - 2-step Enter: 1) 목표값 인식, 2) Planning 시작 확정
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE COMMAND MODE")
    print("="*60)
    print(f"Control mode: {control_mode.upper()}")
    if control_mode == "absolute":
        print("  → Position: World frame 절대 좌표")
        print("  → Quaternion: World frame 기준 절대 orientation")
    else:  # relative
        print("  → Position: 현재 EE position 기준 상대 offset")
        print("  → Quaternion: 현재 EE orientation 기준 상대 rotation")
    print("\nFormat: x y z qw qx qy qz (position + quaternion)")
    print("  - Position: meters")
    print("  - Quaternion: [qw, qx, qy, qz] (w-first)")
    print("Type 'q' to quit")
    print("="*60 + "\n")

    # ========================================
    # ALTERNATIVE: Euler Angle Input (주석 처리)
    # ========================================
    # Format: x y z roll pitch yaw (degrees)
    # 사용하려면 아래 코드 주석 해제 + quaternion parsing 부분 주석 처리
    # ========================================
    # print("Format: x y z roll pitch yaw (position + euler angles)")
    # print("  - Position: meters")
    # print("  - Euler angles: degrees (roll, pitch, yaw)")

    pending_target = None  # 1st Enter로 입력된 target (확정 대기 중)

    while not stop_event.is_set():
        try:
            # Step 1: 목표값 입력 (또는 확정)
            if pending_target is None:
                user_input = input("\n[Command] Enter target (x y z qw qx qy qz) or 'q': ")
            else:
                user_input = input("[Command] Press Enter to confirm, or enter new target: ")

            if user_input.lower() == 'q':
                print("[Command] Quit command received. Exiting...")
                stop_event.set()
                break

            # Empty input → confirm pending target
            if user_input.strip() == "" and pending_target is not None:
                # Step 2: Planning 시작 확정
                target_queue.put(pending_target)
                print(f"✓ Planning started for target: {pending_target}")
                pending_target = None  # Reset
                continue

            # Non-empty input → parse new target
            coords = [float(x) for x in user_input.split()]

            # ========================================
            # QUATERNION INPUT (현재 활성화)
            # ========================================
            if len(coords) != 7:
                print("✗ Invalid format! Use: x y z qw qx qy qz (7 numbers)")
                print("  Example: 0.5 0.3 0.8 1.0 0.0 0.0 0.0")
                continue

            pos = coords[:3]
            quat = coords[3:]  # [qw, qx, qy, qz]

            # Quaternion validation (norm should be ~1.0)
            quat_norm = sum(q**2 for q in quat) ** 0.5
            if abs(quat_norm - 1.0) > 0.1:
                print(f"⚠ Warning: Quaternion norm is {quat_norm:.3f} (should be ~1.0)")
                print("  Auto-normalizing quaternion...")
                quat = [q / quat_norm for q in quat]

            pending_target = {
                'position': pos,
                'quaternion': quat,
            }
            print(f"✓ Target recognized:")
            print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"  Quaternion: [qw={quat[0]:.3f}, qx={quat[1]:.3f}, qy={quat[2]:.3f}, qz={quat[3]:.3f}]")
            print(f"  → Press Enter to confirm and start planning")

            # ========================================
            # EULER ANGLE INPUT (주석 처리 - 필요시 활성화)
            # ========================================
            # if len(coords) != 6:
            #     print("✗ Invalid format! Use: x y z roll pitch yaw (6 numbers)")
            #     print("  Example: 0.5 0.3 0.8 0.0 0.0 90.0")
            #     continue
            #
            # pos = coords[:3]
            # euler = coords[3:]  # [roll, pitch, yaw] in degrees
            #
            # # Convert Euler angles to quaternion
            # import math
            # roll, pitch, yaw = [math.radians(angle) for angle in euler]
            #
            # # ZYX convention (yaw-pitch-roll)
            # cy = math.cos(yaw * 0.5)
            # sy = math.sin(yaw * 0.5)
            # cp = math.cos(pitch * 0.5)
            # sp = math.sin(pitch * 0.5)
            # cr = math.cos(roll * 0.5)
            # sr = math.sin(roll * 0.5)
            #
            # qw = cr * cp * cy + sr * sp * sy
            # qx = sr * cp * cy - cr * sp * sy
            # qy = cr * sp * cy + sr * cp * sy
            # qz = cr * cp * sy - sr * sp * cy
            # quat = [qw, qx, qy, qz]
            #
            # pending_target = {
            #     'position': pos,
            #     'quaternion': quat,
            # }
            # print(f"✓ Target recognized:")
            # print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            # print(f"  Euler (deg): [roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}]")
            # print(f"  Quaternion: [qw={quat[0]:.3f}, qx={quat[1]:.3f}, qy={quat[2]:.3f}, qz={quat[3]:.3f}]")
            # print(f"  → Press Enter to confirm and start planning")

        except ValueError:
            print("✗ Invalid input! Use numbers only")
            print("  Example: 0.5 0.3 0.8 1.0 0.0 0.0 0.0")
        except EOFError:
            # Handle Ctrl+D
            print("\n[Command] EOF detected. Exiting...")
            stop_event.set()
            break
        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    # ========================================
    # THREADING COMPONENTS (INTERACTIVE COMMAND MODE)
    # ========================================
    target_queue = queue.Queue()  # Thread-safe queue for target positions
    stop_event = threading.Event()  # Event to signal thread termination

    # Start input listener thread (daemon=True: auto-terminate when main exits)
    input_thread = threading.Thread(
        target=input_listener_thread,
        args=(target_queue, stop_event, args.control_mode),
        daemon=True
    )
    input_thread.start()
    print(f"[Main] Input listener thread started (control_mode={args.control_mode})")

    # ========================================
    # STATE MACHINE VARIABLES
    # ========================================
    # States: IDLE → WAITING_ROBOT → PLANNING → EXECUTING → IDLE
    planning_state = "IDLE"  # Current state
    pending_target = None    # Target position from queue
    current_trajectory = None  # Planned trajectory
    trajectory_index = 0     # Current index in trajectory
    idx_list = []            # Joint indices for articulation control
    common_js_names = []     # Common joint names between sim and plan

    # create a curobo motion gen instance:
    num_targets = 0
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 1.0]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05   #0.015,  # 0.05 * 0.3 = 0.015 (30% of original size) -> 너무 작아서 바꿈 (12222149)
    )

    # ========================================
    # cuRobo Logger Level 설정
    # ========================================
    # "warn":  모든 warning 표시 (warp cache 포함) → 시뮬레이션 끊김 유발
    # "error": error만 표시, warning 숨김 → warp cache warning 억제
    #
    # Warp cache warning 예시:
    #   [Warning] [curobo] Object already in warp cache, using existing instance for: ...
    #   → 정상 동작이지만 매 업데이트마다 27개 출력 → 노이즈
    #
    # 변경: "warn" → "error" (warp cache warning 억제)
    # ========================================
    setup_curobo_logger("error")  # 변경: warn → error
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    # ========================================
    # ROBOT CONFIGURATION LOADING
    # ========================================
    # cuRobo는 YAML config 파일에서 로봇 정보를 로드합니다.
    # Config에는 URDF/USD 경로, joint 정보, collision spheres 등이 포함됩니다.

    tensor_args = TensorDeviceType()

    # Step 1: Robot config 파일 경로 설정
    # 기본: {curobo}/src/curobo/content/configs/robot/
    robot_cfg_path = get_robot_configs_path()

    # 외부 config 디렉토리 사용 시 (--external_robot_configs_path 인자)
    # 예: python motion_gen_reacher_hyundai.py --external_robot_configs_path /home/user/my_configs
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path

    # Step 2: Robot YAML config 로드
    # 예: --robot iiwa_allegro_mocked.yml
    # 로드: {robot_cfg_path}/iiwa_allegro_mocked.yml
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    # Step 3: 외부 asset 경로 설정 (URDF/USD/meshes가 있는 디렉토리)
    # --external_asset_path 인자로 전달 가능
    #
    # 사용 예시:
    # python motion_gen_reacher_hyundai.py \
    #   --robot my_robot.yml \
    #   --external_asset_path /home/user/my_robot_assets
    #
    # 이 경로가 설정되면:
    # - URDF: {external_asset_path}/{urdf_path}
    # - Meshes: {external_asset_path}/{asset_root_path}/meshes/
    #
    # Config 파일 (my_robot.yml) 예시:
    # robot_cfg:
    #   kinematics:
    #     urdf_path: robot/my_robot.urdf  # 상대 경로
    #     asset_root_path: robot/          # 상대 경로
    #     # 실제 URDF: /home/user/my_robot_assets/robot/my_robot.urdf
    #     # 실제 meshes: /home/user/my_robot_assets/robot/meshes/
    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path

    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path

    # Step 4: Joint 정보 추출
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # Step 5: Isaac Sim에 로봇 추가
    # - USD 파일 로드 (Isaac Sim 렌더링용)
    # - Articulation 생성 (물리 시뮬레이션용)
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = None

    # 수정: 기본 obstacle (테이블) 제거 - Hyundai 샤시만 사용
    # 기존: collision_table.yml에서 테이블 로드
    # 변경: 빈 WorldConfig 사용 (Hyundai 샤시는 나중에 동적으로 추가)
    world_cfg = WorldConfig(cuboid=[], mesh=[])

    # 기존 코드 (참고용 - 주석 처리)
    # world_cfg_table = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # )
    # world_cfg_table.cuboid[0].pose[2] -= 0.02
    # world_cfg1 = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # ).get_mesh_world()
    # world_cfg1.mesh[0].name += "_mesh"
    # world_cfg1.mesh[0].pose[2] = -10.5
    # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # trajpot_tsteps, max_attempts, num_trajpot_seeds use_cuda_graph
    # Interactive command mode configuration (non-reactive)
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 24
    trim_steps = None
    max_attempts = 10
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        # MODIFIED: 2025-12-22 - Enable MESH collision checking for Hyundai USD obstacles
        collision_checker_type=CollisionCheckerType.MESH,  # Enable mesh collision checking
        num_trajopt_seeds=32,  # Increased for iiwa_allegro
        num_graph_seeds=32,    # Increased for iiwa_allegro
        num_ik_seeds=256,  # Max increase for iiwa_allegro (GPU has room)
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
        use_cuda_graph=False,
        position_threshold=0.01,   # Further relax for complex robots
        rotation_threshold=0.1,    # Further relax for complex robots
        store_ik_debug=True,  # Enable IK debug output
        store_trajopt_debug=True,  # Enable trajopt debug output
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5,  # Non-reactive mode
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    # ========================================
    # MODIFIED: 2025-12-22 - Load Hyundai USD environment as DYNAMIC mesh obstacles
    # UPDATED: 2025-12-22 (evening) - Added position offset to avoid robot collision
    # UPDATED: 2025-12-22 (night) - Fixed for Isaac Sim 5.0.0 API compatibility
    #   - Using CuRobo's set_prim_transform() for reliable transform setting
    #   - Using isaacsim.core API (Isaac Sim 5.0.0) with fallback to legacy API
    # UPDATED: 2025-12-22 (late night) - Dynamic obstacle support
    #   - Selective physics: chassis=kinematic (fixed), rings=dynamic (rotate)
    #   - RevoluteJoint enabled for rings (articulated obstacles)
    #   - Real-time obstacle pose tracking from physics simulation
    # ========================================
    # Isaac Sim 5.0.0 새 API 사용 (deprecated warning 해결)
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage as add_ref_new
        print("[Hyundai] Using Isaac Sim 5.0.0 new API (isaacsim.core)")
        add_reference_to_stage = add_ref_new
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage
        print("[Hyundai] Using legacy API (omni.isaac.core)")

    # USD 파일 경로 설정
    # 변경: 절대 경로 → cuRobo 표준 scene 경로 (이식성 향상)
    usd_base_path = join_path(get_assets_path(), "scene/hyundai_chassis")
    # 기존 절대 경로 (참고용):
    # usd_base_path = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd"

    # 위치 오프셋 설정 (로봇 기준 상대 위치)
    # rh56f1 요구사항: x=-0.75, y=1.3, z=0.1 (Hyundai 로봇 기준)
    # offset_position = np.array([-0.75, 1.3, 0.1])  # x, y, z offset in meters
    
    # iiwa_allegro 용
    #offset_position = np.array([0.7, -0.5, 0.1])  # x, y, z offset in meters

    # ========================================
    # 샤시 offset 정의 (로봇 위치에 따라 자동 결정)
    # - dg5f (왼쪽 로봇): y=-1.3
    # - rh56f1 (오른쪽 로봇): y=+1.3
    # ========================================
    dg5f_offset_position = np.array([-0.75, -1.3, 0.1])
    rh56f1_offset_position = np.array([-0.75, 1.3, 0.1])

    # Robot 모듈에 따라 샤시 offset 자동 결정
    robot_name = args.robot.lower()
    if "dg5f" in robot_name:
        offset_position = dg5f_offset_position
        print(f"[Chassis] Detected DG5F (Left) → offset y={offset_position[1]}")
    elif "rh56f1" in robot_name:
        offset_position = rh56f1_offset_position
        print(f"[Chassis] Detected RH56F1 (Right) → offset y={offset_position[1]}")
    else:
        offset_position = dg5f_offset_position  # 기본값
        print(f"[Chassis] Unknown robot '{args.robot}', using default offset (dg5f)")

    print(f"[Hyundai] Target offset position: x={offset_position[0]}m, y={offset_position[1]}m, z={offset_position[2]}m")

    # Method 1: 메인 USD 파일 로드 (chassis + spring + wire 포함)
    print("[Hyundai] Loading main environment USD file...")
    add_reference_to_stage(
        usd_path=f"{usd_base_path}/env_diated_decomposed_chassis_tilted_srping_wire_flattened_no_rigid.usd", #env_diated_decomposed_chassis_spring_wire_flattened_no_rigid.usd",
        prim_path="/World/hyundai_env_main"
    )

    # CRITICAL: USD 로드 직후 prim에 직접 translate 설정
    # CuRobo의 set_prim_transform 사용 (Isaac Sim 5.0.0 호환 방식)
    from pxr import UsdPhysics

    hyundai_prim = stage.GetPrimAtPath("/World/hyundai_env_main")
    if hyundai_prim.IsValid():
        # 1. Transform 설정 - CuRobo 공식 방법 사용
        # pose format: [x, y, z, qw, qx, qy, qz] (quaternion w first!)
        pose = [offset_position[0], offset_position[1], offset_position[2], 0, 0, 0, 1]  # quaternion z축 180도 기준 회전 (12222149)
        set_prim_transform(hyundai_prim, pose)

        print(f"[Hyundai] ✓ Transform applied: translation=({offset_position[0]}, {offset_position[1]}, {offset_position[2]})")

        # 2. Selective Physics 설정 (동적 장애물 지원)
        # - Chassis/Frame/Strut: kinematic (고정) → 바닥 역할, 움직이면 안 됨
        # - Rings: dynamic (회전 가능) → RevoluteJoint로 용수철에 매달려 회전

        def set_selective_physics(prim, path_prefix=""):
            """
            Selectively set physics properties:
            - Static parts (chassis, frame, strut): kinematic (fixed)
            - Dynamic parts (rings): keep physics enabled (can rotate)
            """
            current_path = path_prefix + "/" + prim.GetName() if path_prefix else prim.GetName()

            # Kinematic으로 설정할 부품들 (고정)
            #static_keywords = ["chassis", "frame", "strut", "spring", "wire", "bar"]
            static_keywords = ["chassis", "frame", "strut"]

            # Dynamic으로 유지할 부품들 (회전 가능)
            dynamic_keywords = ["ring", "spring", "wire", "bar"]

            is_static = any(keyword in current_path.lower() for keyword in static_keywords)
            is_dynamic = any(keyword in current_path.lower() for keyword in dynamic_keywords)

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigidBodyAPI = UsdPhysics.RigidBodyAPI(prim)

                if is_static and not is_dynamic:
                    # Static: kinematic 활성화 (고정)
                    rigidBodyAPI.CreateKinematicEnabledAttr().Set(True)
                    print(f"[Hyundai Physics] Static (kinematic): {current_path}")
                elif is_dynamic:
                    # Dynamic: kinematic 비활성화 (움직일 수 있음)
                    rigidBodyAPI.CreateKinematicEnabledAttr().Set(False)
                    print(f"[Hyundai Physics] Dynamic (articulated): {current_path}")

            # 재귀적으로 자식 처리
            for child in prim.GetChildren():
                set_selective_physics(child, current_path)

        # Selective physics 적용
        print("[Hyundai] Configuring selective physics (static chassis, dynamic rings)...")
        set_selective_physics(hyundai_prim)
        print("[Hyundai] ✓ Physics configured - chassis fixed, rings can rotate via RevoluteJoint")

    else:
        print("[Hyundai] ✗ Warning: Could not find /World/hyundai_env_main prim!")

    print("[Hyundai] USD environment files loaded successfully!")
    # ========================================
    # END MODIFICATION: 2025-12-22
    # ========================================

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None

    # ========================================
    # EE POSITION VISUALIZATION (파란색 sphere)
    # ========================================
    # 작은 파란색 구로 현재 End-Effector position만 표시
    # ========================================
    ee_position_sphere = None  # Blue sphere for EE position
    ee_sphere_radius = 0.02  # Sphere radius (meters)
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        # ========================================
        # DYNAMIC OBSTACLE UPDATE
        # ========================================
        # 동적 장애물 지원: RevoluteJoint로 회전하는 고리(ring)의 위치를 추적
        # - 기존: 1000 프레임마다 업데이트 (정적 장애물용)
        # - 1차 변경: 50 프레임 (동적 장애물용, ~0.4초 @ 120Hz)
        # - 2차 변경: 500 프레임 (성능 최적화, ~4.2초 @ 120Hz)
        # - 3차 변경: 2000 프레임 (wire 초기 정착 후 정지, ~16.7초 @ 120Hz)
        #
        # 이유:
        # - Isaac Sim physics가 고리/wire를 회전시킴 (RevoluteJoint)
        # - cuRobo는 USD stage에서 현재 pose를 읽어서 collision checking
        # - Wire는 초기 정착 후 거의 움직이지 않으므로 2000 프레임 간격으로 충분
        # - RevoluteJoint 존재 인지가 목적 (정확한 실시간 추적 불필요)
        #
        # 성능:
        # - get_obstacles_from_stage()는 GPU 가속됨 (빠름)
        # - 2000 프레임 간격: 500 대비 4배 성능 향상
        # - Wire/spring/ring/bar 충돌 감지 활성화되어도 부드러운 시뮬레이션
        # ========================================
        if step_index == 50 or step_index % 2000 == 0.0:  # 변경: 1000 → 50 → 500 → 2000 (성능)
            print(f"[Step {step_index}] Updating world, reading w.r.t. {robot_prim_path}")

            # USD stage에서 현재 장애물 위치/자세 읽기
            # - Hyundai 전체 환경을 obstacle로 사용 (샤시, 프레임, wire, spring, ring, bar 모두 포함)
            # - only_paths를 /World/hyundai_env_main으로 제한
            # - UPDATED: wire/spring/ring/bar 충돌 감지 활성화 (충돌 회피 필수)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World/hyundai_env_main"],  # Hyundai 전체 환경 포함
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                    "/World/obstacles",  # 기존 테이블 등 obstacle 무시
                    # wire/spring/ring/bar를 제외하지 않음 → 충돌 감지 활성화!
                ],
            ).get_collision_check_world()

            print(f"[Hyundai Dynamic] Total obstacles detected: {len(obstacles.objects)}")
            if len(obstacles.objects) > 0:
                mesh_count = len([o for o in obstacles.objects if hasattr(o, 'file_path')])
                print(f"[Hyundai Dynamic] - Mesh obstacles: {mesh_count}")
                print(f"[Hyundai Dynamic] - Other obstacles: {len(obstacles.objects) - mesh_count}")

            # cuRobo world 업데이트 (동적 장애물 위치 반영)
            # - motion_gen.update_world()는 GPU에서 빠르게 실행됨
            # - 이후 모든 planning은 업데이트된 장애물 기준으로 실행
            motion_gen.update_world(obstacles)
            print("[Hyundai Dynamic] ✓ Updated World - ring poses synchronized from physics simulation")
            carb.log_info("Synced CuRobo world from stage (dynamic obstacles).")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,  # Zero velocity (non-reactive mode)
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

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
        # EE POSITION VISUALIZATION (파란색 sphere - 매 프레임 업데이트)
        # ========================================
        if step_index % 2 == 0:  # Update every 2 frames for performance
            try:
                # Get current EE pose from FK
                # Note: forward() can return different types depending on cuRobo version
                ee_pose_result = motion_gen.kinematics.forward(cu_js.position)

                # Handle different return types robustly
                if hasattr(ee_pose_result, 'position'):
                    # Case 1: Pose object with .position attribute
                    ee_position = ee_pose_result.position[0].cpu().numpy()
                elif isinstance(ee_pose_result, tuple):
                    # Case 2: Tuple (position, quaternion, ...)
                    # Just take the first element (position)
                    ee_position = ee_pose_result[0][0].cpu().numpy()
                else:
                    # Unknown type - skip visualization this frame
                    continue

                # Create or update EE position sphere (파란색)
                if ee_position_sphere is None:
                    ee_position_sphere = sphere.VisualSphere(
                        prim_path="/curobo/ee_position",
                        position=ee_position,
                        radius=ee_sphere_radius,
                        color=np.array([0.0, 0.5, 1.0]),  # Blue color
                    )
                else:
                    # Update sphere position
                    ee_position_sphere.set_world_pose(position=ee_position)

            except Exception as e:
                # If anything goes wrong, just skip EE visualization this frame
                if step_index < 100:  # Only print warning during startup
                    print(f"[Warning] Could not visualize EE position: {e}")

        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 2.0):  # 1.0 → 2.0 (dg5f_l 호환)
            robot_static = True

        # ========================================
        # STATE MACHINE LOGIC (INTERACTIVE COMMAND MODE)
        # ========================================
        # Check for quit signal
        if stop_event.is_set():
            print("[Main] Stop event received. Exiting simulation...")
            break

        # State: IDLE - Wait for user input
        if planning_state == "IDLE":
            # Check if there's a new target in the queue
            if not target_queue.empty():
                pending_target = target_queue.get()
                print(f"\n[State: IDLE → WAITING_ROBOT] New target received: {pending_target}")
                planning_state = "WAITING_ROBOT"

        # State: WAITING_ROBOT - Wait for robot to stop moving
        elif planning_state == "WAITING_ROBOT":
            if robot_static:
                print(f"[State: WAITING_ROBOT → PLANNING] Robot stopped, starting planning...")
                planning_state = "PLANNING"
            elif step_index % 100 == 0:
                print(f"[State: WAITING_ROBOT] Waiting for robot to stop... (max_vel={np.max(np.abs(sim_js.velocities)):.4f})")

        # State: PLANNING - Execute motion planning
        elif planning_state == "PLANNING":
            # Extract position and quaternion from pending_target dict
            target_position = np.array(pending_target['position'])
            target_quaternion = np.array(pending_target['quaternion'])  # [qw, qx, qy, qz]

            # Apply control mode (absolute or relative)
            if args.control_mode == "relative":
                # Get current EE position from FK
                current_ee_pose = motion_gen.kinematics.forward(cu_js.position)
                current_ee_position = current_ee_pose.position[0].cpu().numpy()
                current_ee_quaternion = current_ee_pose.quaternion[0].cpu().numpy()  # [qw, qx, qy, qz]

                # Add relative offset
                ee_translation_goal = current_ee_position + target_position

                # Quaternion multiplication for relative rotation
                # q_goal = q_current * q_target
                import torch
                q_current = torch.tensor(current_ee_quaternion, dtype=torch.float32)  # [qw, qx, qy, qz]
                q_target = torch.tensor(target_quaternion, dtype=torch.float32)

                # Quaternion multiplication: q1 * q2
                qw = q_current[0]*q_target[0] - q_current[1]*q_target[1] - q_current[2]*q_target[2] - q_current[3]*q_target[3]
                qx = q_current[0]*q_target[1] + q_current[1]*q_target[0] + q_current[2]*q_target[3] - q_current[3]*q_target[2]
                qy = q_current[0]*q_target[2] - q_current[1]*q_target[3] + q_current[2]*q_target[0] + q_current[3]*q_target[1]
                qz = q_current[0]*q_target[3] + q_current[1]*q_target[2] - q_current[2]*q_target[1] + q_current[3]*q_target[0]
                ee_orientation_teleop_goal = np.array([qw, qx, qy, qz])

                print(f"\n[State: PLANNING] Relative mode - Computing trajectory...")
                print(f"  Current EE position: [{current_ee_position[0]:.3f}, {current_ee_position[1]:.3f}, {current_ee_position[2]:.3f}]")
                print(f"  Relative offset: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
                print(f"  Target position: [{ee_translation_goal[0]:.3f}, {ee_translation_goal[1]:.3f}, {ee_translation_goal[2]:.3f}]")
            else:
                # Absolute mode (world frame)
                ee_translation_goal = target_position
                ee_orientation_teleop_goal = target_quaternion

                print(f"\n[State: PLANNING] Absolute mode - Computing trajectory...")
                print(f"  Target position: [{ee_translation_goal[0]:.3f}, {ee_translation_goal[1]:.3f}, {ee_translation_goal[2]:.3f}]")

            print(f"  Target quaternion: [qw={ee_orientation_teleop_goal[0]:.3f}, qx={ee_orientation_teleop_goal[1]:.3f}, qy={ee_orientation_teleop_goal[2]:.3f}, qz={ee_orientation_teleop_goal[3]:.3f}]")

            # Update target cube visualization (RED)
            target.set_world_pose(
                position=ee_translation_goal,
                orientation=ee_orientation_teleop_goal  # [qw, qx, qy, qz]
            )

            # Compute cuRobo solution
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()

            # Handle planning result
            if succ:
                print(f"✓ [State: PLANNING → EXECUTING] Planning SUCCESS!")

                # Apply special pose metrics for first target
                if num_targets == 0:
                    if args.constrain_grasp_approach:
                        pose_metric = PoseCostMetric.create_grasp_approach_metric()
                    if args.reach_partial_pose is not None:
                        reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                        pose_metric = PoseCostMetric(
                            reach_partial_pose=True, reach_vec_weight=reach_vec
                        )
                    if args.hold_partial_pose is not None:
                        hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                        pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)

                num_targets += 1

                # Get interpolated trajectory
                current_trajectory = result.get_interpolated_plan()
                current_trajectory = motion_gen.get_full_js(current_trajectory)

                # Get only joint names that are in both sim and plan
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in current_trajectory.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                current_trajectory = current_trajectory.get_ordered_joint_state(common_js_names)
                trajectory_index = 0

                # Transition to EXECUTING
                planning_state = "EXECUTING"

            else:
                # Planning failed
                print(f"✗ [State: PLANNING → IDLE] Planning FAILED!")
                print(f"  Status: {result.status}")

                # Debug info for common failures
                if result.status is not None:
                    if "JOINT_LIMITS" in result.status.name:
                        print(f"  Reason: Target outside joint limits")
                    elif "IK_FAIL" in result.status.name:
                        print(f"  Reason: Inverse kinematics failed (unreachable pose)")
                    elif "COLLISION" in result.status.name:
                        print(f"  Reason: Collision detected in trajectory")
                else:
                    print(f"  Reason: Unknown (status=None)")

                print(f"\n  Please try a different target position.")

                # Reset and go back to IDLE
                pending_target = None
                planning_state = "IDLE"

            # Update target tracking (for visualization)
            target_pose = cube_position
            target_orientation = cube_orientation

        # State: EXECUTING - Execute planned trajectory
        elif planning_state == "EXECUTING":
            if current_trajectory is not None and trajectory_index < len(current_trajectory.position):
                # Get current waypoint
                cmd_state = current_trajectory[trajectory_index]
                past_cmd = cmd_state.clone()

                # Create articulation action
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )

                # Apply action to robot
                articulation_controller.apply_action(art_action)

                # Step simulation (2 substeps for smoothness)
                for _ in range(2):
                    my_world.step(render=False)

                trajectory_index += 1

                # Progress feedback every 10 waypoints
                if trajectory_index % 10 == 0:
                    progress = (trajectory_index / len(current_trajectory.position)) * 100
                    print(f"[State: EXECUTING] Progress: {progress:.1f}% ({trajectory_index}/{len(current_trajectory.position)})")

            else:
                # Trajectory execution complete
                print(f"✓ [State: EXECUTING → IDLE] Execution complete!")
                print(f"  Ready for next command.\n")

                # Reset trajectory
                current_trajectory = None
                trajectory_index = 0
                past_cmd = None
                pending_target = None

                # Transition back to IDLE
                planning_state = "IDLE"

        # Update past pose tracking
        past_pose = cube_position
        past_orientation = cube_orientation
    simulation_app.close()


if __name__ == "__main__":
    main()
