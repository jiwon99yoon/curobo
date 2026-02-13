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
# MODIFIED VERSION: motion_gen_reacher_hyundai.py
# ================================================================================
# Original file: motion_gen_reacher.py
# Modified by: dyros
# Date: 2025-12-22
#
# MODIFICATIONS:
# 1. Enabled MESH collision checking (CollisionCheckerType.MESH)
# 2. Added Hyundai USD environment loading:
#    - env_diated_decomposed_chassis_spring_wire_flattened.usd
#    - Optional: wire_revolute_collision_flattened.usd
#    - Optional: env_diated_decomposed_chassis_flattened.usd
# 3. Updated obstacle detection to include Hyundai USD meshes
# 4. Added debug output for obstacle counting
# 5. UPDATED (2025-12-22 evening): Added position offset (x=+0.5m, y=+0.5m)
#    to avoid INVALID_START_STATE_WORLD_COLLISION error
# 6. UPDATED (2025-12-22 night): Fixed for Isaac Sim 5.0.0 compatibility
#    - Use CuRobo's set_prim_transform() for reliable USD transform setting
#    - Disable physics (kinematic mode) recursively to prevent objects flying away
#    - Support both isaacsim.core (new) and omni.isaac.core (legacy) APIs
#
# 7. UPDATED (2025-12-22 late night): Dynamic obstacle support
#    - Selective physics: chassis/frame/strut=kinematic, rings=dynamic
#    - Rings can rotate via RevoluteJoint (articulated obstacles)
#    - Obstacle update frequency: 1000 → 50 frames (~0.4s @ 120Hz)
#    - Added status=None edge case handling with debugging
#    - cuRobo tracks rotating rings in real-time for collision avoidance
#
# 8. UPDATED (2025-12-22 final): Performance optimization
#    - Suppressed warp cache warnings (logger: "warn" → "error")
#    - Reduced console I/O overhead for smoother simulation
#    - Kept status=None debugging for edge case detection
#
# USAGE:
#   # 기본 사용 (cuRobo 내장 robot 사용)
#   python examples/isaac_sim/motion_gen_reacher_hyundai.py \
#       --robot iiwa_allegro_mocked.yml \
#       --reactive \
#       --visualize_spheres
#
#   # 외부 로봇 사용 (커스텀 URDF/USD/meshes)
#   python examples/isaac_sim/motion_gen_reacher_hyundai.py \
#       --robot my_custom_robot.yml \
#       --external_asset_path /home/user/my_robot_assets \
#       --external_robot_configs_path /home/user/my_configs \
#       --reactive
#
#   필요한 파일 구조:
#   /home/user/my_configs/
#     └── my_custom_robot.yml  # Robot config (joint, collision spheres 등)
#   /home/user/my_robot_assets/
#     └── robot/
#         ├── my_robot.urdf     # URDF 파일
#         ├── my_robot.usda     # USD 파일 (Isaac Sim 렌더링용)
#         └── meshes/           # Mesh 파일들 (.obj, .stl, .dae 등)
#             ├── visual/       # Visual meshes (렌더링용)
#             └── collision/    # Collision meshes (충돌 감지용)
#
# TESTED WITH:
#   - Robot: iiwa_allegro_mocked.yml (7-DOF arm with locked hand)
#   - Environment: Hyundai factory USD files (chassis + wire + rotating rings)
#   - Mode: Reactive planning with DYNAMIC mesh collision avoidance
#   - Physics: Rings rotate on springs via RevoluteJoint
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
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

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


def main():
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
        orientation=np.array([0.0, 0.7071068, 0.0, -0.7071068]),  # Euler (180, -90, 0) deg
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
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

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
    # 변경: 절대 경로 → cuRobo 표준 scene 경로 (이식성 향상, 2026-02-13)
    usd_base_path = join_path(get_assets_path(), "scene/hyundai_chassis")
    # 기존 절대 경로 (참고용):
    # usd_base_path = "/home/dyros/IsaacLab/hyundai/1_factory_usd_file/flattended_usd"

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
        #
        # 이유:
        # - Isaac Sim physics가 고리를 회전시킴 (RevoluteJoint)
        # - cuRobo는 USD stage에서 현재 pose를 읽어서 collision checking
        # - 고리가 상시 회전하지 않으므로 500 프레임 간격으로 충분
        # - RevoluteJoint 존재 인지가 목적 (정확한 실시간 추적 불필요)
        #
        # 성능:
        # - get_obstacles_from_stage()는 GPU 가속됨 (빠름)
        # - 500 프레임 간격: 50 대비 10배 성능 향상
        # - 실시간 시뮬레이션 부드러움 개선
        # ========================================
        if step_index == 50 or step_index % 2000 == 0.0:  # 변경: 1000 → 50 → 500 → 2000 (성능)
            print(f"[Step {step_index}] Updating world, reading w.r.t. {robot_prim_path}")

            # USD stage에서 현재 장애물 위치/자세 읽기
            # - 수정: Hyundai 샤시만 obstacle로 사용, 나머지 모두 제거
            # - only_paths를 /World/hyundai_env_main으로 제한
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World/hyundai_env_main"],  # Hyundai 샤시만 포함
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
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
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

        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 2.0) or args.reactive:  # 1.0 → 2.0 (dg5f_l 호환)
            robot_static = True

        # Debug: Print planning conditions
        target_moved = np.linalg.norm(cube_position - target_pose) > 1e-3
        target_stopped = np.linalg.norm(past_pose - cube_position) == 0.0
        orientation_stopped = np.linalg.norm(past_orientation - cube_orientation) == 0.0

        if step_index % 100 == 0:
            print(f"Planning conditions: target_moved={target_moved}, target_stopped={target_stopped}, "
                  f"orientation_stopped={orientation_stopped}, robot_static={robot_static}, "
                  f"max_vel={np.max(np.abs(sim_js.velocities)):.4f}")

        if (
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            print(f"[PLANNING START] Target: {ee_translation_goal}, Orientation: {ee_orientation_teleop_goal}")
            print(f"[DEBUG] Current joint positions: {cu_js.position.cpu().numpy()}")

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            print(f"[PLANNING RESULT] Success: {succ}, Status: {result.status}")

            # ========================================
            # DEBUG: Status None handling
            # ========================================
            # result.status가 None인 경우 처리:
            # - 정상: Success=True일 때 status=None
            # - 비정상: Success=False인데 status=None (edge case)
            #
            # 비정상 케이스 발생 원인:
            # 1. Planning timeout (시간 초과)
            # 2. Internal cuRobo error (예외 발생)
            # 3. GPU memory issue (warp cache 문제)
            # 4. Invalid planning parameters
            # ========================================
            if not succ:
                if result.status is None:
                    # Planning 실패했는데 status가 None → 예외 상황
                    print("=" * 80)
                    print("[ERROR] Planning failed but status is None (unexpected edge case)")
                    print("=" * 80)
                    print(f"[DEBUG] Target pose: {ee_translation_goal}")
                    print(f"[DEBUG] Current joint positions: {cu_js.position.cpu().numpy()}")
                    print(f"[DEBUG] IK goal position: {ik_goal.position.cpu().numpy()}")
                    print(f"[DEBUG] IK goal quaternion: {ik_goal.quaternion.cpu().numpy()}")

                    # 가능한 원인 체크
                    print("\n[DEBUG] Possible causes:")
                    print("  1. Planning timeout - reactive mode max_attempts=1 too strict")
                    print("  2. Internal cuRobo exception caught but not set status")
                    print("  3. GPU memory/warp cache issue - check for CUDA errors")
                    print("  4. Invalid IK goal - target unreachable or outside workspace")

                    # 복구 시도
                    print("\n[DEBUG] Recovery: Skipping this planning cycle...")
                    print("=" * 80)
                    continue  # 이번 planning cycle 건너뛰기

                elif result.status.name == "INVALID_START_STATE_JOINT_LIMITS":
                    # Joint limit 위반
                    print(f"[DEBUG] Joint limits violated!")
                    print(f"[DEBUG] Joint lower limits: {motion_gen.kinematics.get_joint_limits().position[0, :, 0].cpu().numpy()}")
                    print(f"[DEBUG] Joint upper limits: {motion_gen.kinematics.get_joint_limits().position[0, :, 1].cpu().numpy()}")

            if num_targets == 1:
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
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


if __name__ == "__main__":
    main()
