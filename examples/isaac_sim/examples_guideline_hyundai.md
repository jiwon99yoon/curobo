# Hyundai HDR35_20 Motion Planning - Examples Guide

**작성일:** 2026-02-10
**대상:** HDR35_20 + RH56F1_R / DG5F_L 로봇 시스템
**목적:** cuRobo 기반 motion planning 예제 스크립트 사용법

---

## 1. 파일 목록

| 파일 | 로봇 | 입력 방식 | 설명 |
|------|------|----------|------|
| `motion_gen_reacher_hyundai.py` | Single-arm | GUI (큐브 드래그) | 단일 로봇, target 큐브를 드래그하면 planning |
| `motion_gen_reacher_command.py` | Single-arm | 터미널 (좌표 입력) | 단일 로봇, 터미널에서 좌표 직접 입력 |
| `motion_gen_reacher_dual_hyundai.py` | Dual-arm | GUI (큐브 드래그) | 듀얼암, RED/BLUE 큐브 드래그로 planning |
| `motion_gen_reacher_dual_command.py` | Dual-arm | 터미널 (좌표 입력) | 듀얼암, RH56 → DG5F 순서로 좌표 입력 |

### Robot Config 매핑

| 파일 | Default Robot Config |
|------|---------------------|
| `motion_gen_reacher_hyundai.py` | `franka.yml` (실행 시 `--robot hdr35_20_ati_rh56f1_r_sensor.yml`, `--robot hdr35_20_ati_dg5f_l.yml` 지정) |
| `motion_gen_reacher_command.py` | `franka.yml` (실행 시 위와 동일한 방법으로 지정해야지 hdr35_20_ati가 붙은 hand model이 불러와짐) |
| `motion_gen_reacher_dual_hyundai.py` | `dual_hdr35_20_ati.yml` |
| `motion_gen_reacher_dual_command.py` | `dual_hdr35_20_ati.yml` |

### Retract 자세 참고
- 모든 config (`hdr35_20_ati_rh56f1_r_sensor.yml`, `hdr35_20_ati_dg5f_l.yml`, `dual_hdr35_20_ati.yml`)에서 gripper joints가 0으로 lock되어 있음
- 즉, **손가락이 다 펴진 상태**로 retract (초기 자세)됨

---

## 2. Single-Arm: GUI 버전 (motion_gen_reacher_hyundai.py)

### 실행
```bash
cd ~/curobo

# RH56F1_R (오른손 그리퍼)
python examples/isaac_sim/motion_gen_reacher_hyundai.py \
    --robot hdr35_20_ati_rh56f1_r_sensor.yml \
    --visualize_spheres
    
# DG5F_L (왼손 그리퍼)
python examples/isaac_sim/motion_gen_reacher_hyundai.py \
    --robot hdr35_20_ati_dg5f_l.yml \
    --visualize_spheres
```

### 사용법
1. Isaac Sim에서 **Play 버튼** 클릭
2. **RED 큐브**를 드래그하여 목표 위치로 이동
3. 큐브를 놓으면 자동으로 planning 시작
4. Planning 성공 시 로봇이 해당 위치로 이동

### Target 초기 Orientation
- Euler: **(180, -90, 0)** degrees
- Quaternion: `(qw=0, qx=0.7071, qy=0, qz=-0.7071)`
- 의미: 로봇 hand가 쭉 편 상태로 샤시 모듈 방향(-X)을 향하는 자세

### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--robot` | Robot config YAML 파일명 | `franka.yml` |
| `--visualize_spheres` | Collision sphere 시각화 | False |

---

## 3. Single-Arm: Command 버전 (motion_gen_reacher_command.py)

### 실행
```bash
cd ~/curobo

# RH56F1_R (오른손 그리퍼)
python examples/isaac_sim/motion_gen_reacher_command.py \
    --robot hdr35_20_ati_rh56f1_r_sensor.yml \
    --visualize_spheres

# DG5F_L (왼손 그리퍼)
python examples/isaac_sim/motion_gen_reacher_command.py \
    --robot hdr35_20_ati_dg5f_l.yml \
    --visualize_spheres
    
```

### 사용법
1. Isaac Sim에서 **Play 버튼** 클릭
2. 터미널에 **목표 좌표 입력** (2-step 확인 방식):

```
[Command] Enter target (x y z qw qx qy qz) or 'q': 0.5 0.3 0.8 0.0 0.7071 0.0 -0.7071
  Target recognized:
    Position: [0.500, 0.300, 0.800]
    Quaternion: [qw=0.000, qx=0.707, qy=0.000, qz=-0.707]
  → Press Enter to confirm and start planning

[Press Enter]
  Planning started...
  Planning SUCCESS!
  Execution complete!
```

3. 종료: `q` 입력 후 Enter

### 입력 형식
```
x y z qw qx qy qz    (총 7개 숫자, 공백으로 구분)
```
- **Position** (x, y, z): meters, world frame 기준
- **Quaternion** (qw, qx, qy, qz): w-first, 정규화 필요

### 검증된 좌표 (복사해서 사용)

**hdr35_20_ati_rh56f1_r.yml**
```
-0.65 1.0 0.70 0.0 0.7071 0.0 -0.7071
-0.5 1.3 1.1 0.0 0.7071 0.0 -0.7071
```

**hdr35_20_ati_dg5f_l.yml**
```
-0.65 -1.0 0.70 0.0 0.7071 0.0 -0.7071
-0.5 -1.3 1.1 0.0 0.7071 0.0 -0.7071
```

### Control Mode

| Mode | 설명 | 사용법 |
|------|------|--------|
| `absolute` (기본) | World frame 기준 절대 좌표 입력 | `--control_mode absolute` |
| `relative` | 현재 EE 위치 기준 상대 좌표 입력 | `--control_mode relative` |

### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--robot` | Robot config YAML 파일명 | `franka.yml` |
| `--visualize_spheres` | Collision sphere 시각화 | False |
| `--control_mode` | absolute / relative | absolute |

---

## 4. Dual-Arm: GUI 버전 (motion_gen_reacher_dual_hyundai.py)

### 실행
```bash
cd ~/curobo

python examples/isaac_sim/motion_gen_reacher_dual_hyundai.py --visualize_spheres
```

Robot config는 `dual_hdr35_20_ati.yml`로 고정되어 있음.

### 사용법
1. Isaac Sim에서 **Play 버튼** 클릭
2. 두 개의 target 큐브가 표시됨:
   - **RED 큐브**: RH56 (오른쪽 로봇) end-effector target
   - **BLUE 큐브**: DG5F (왼쪽 로봇) end-effector target
3. 큐브를 드래그하여 목표 위치로 이동
4. **두 큐브 모두 움직임이 멈추면** 자동으로 planning 시작

### Target 초기 Orientation
- Euler: **(180, -90, 0)** degrees
- Quaternion: `(qw=0, qx=0.7071, qy=0, qz=-0.7071)`
- 의미: 로봇 hand가 쭉 편 상태로 샤시 모듈 방향(-X)을 향하는 자세
- GUI에서 큐브를 드래그하면 position만 변경되고, orientation은 초기값 유지

### Planning 트리거 조건
두 가지 조건이 **모두** 만족되어야 planning 시작:
1. RH56 또는 DG5F 큐브가 이전 위치에서 이동함 (1mm 이상)
2. 두 큐브 **모두** 드래그가 멈춘 상태 (이전 프레임과 위치 동일)

따라서 큐브를 옮긴 직후 바로 반응하지 않을 수 있음 → 놓고 잠시 기다리면 planning 시작됨.

### 환경
- 샤시(chassis)는 URDF에 내장 (별도 로드 불필요)
- Wire/spring/ring USD가 자동 로드됨 (chassis 부분은 숨김 처리)

---

## 5. Dual-Arm: Command 버전 (motion_gen_reacher_dual_command.py)

### 실행
```bash
cd ~/curobo

python examples/isaac_sim/motion_gen_reacher_dual_command.py --visualize_spheres
```

### 사용법
1. Isaac Sim에서 **Play 버튼** 클릭
2. 터미널에서 **Two-step 입력**:

```
=== Step 1/2: RH56 (오른쪽, y-) ===
[RH56] Enter target (x y z qw qx qy qz): 0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071
  RH56 target saved.

=== Step 2/2: DG5F (왼쪽, y+) ===
[DG5F] Enter target (x y z qw qx qy qz): 0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071
  DG5F target saved.
  Planning dual-arm trajectory...
  Planning SUCCESS!
  Execution complete!
```

3. RH56 입력 → 저장만 됨 (실행 안 함)
4. DG5F 입력 → **Planning 시작!** (두 팔 동시)
5. 종료: `q` 입력

### 입력 형식
```
x y z qw qx qy qz    (총 7개 숫자, 공백으로 구분)
```

---

## 6. Dual-Arm 좌표계

### Robot 위치 (World Frame)
```
         +Y (DG5F side)
          ^
          |
    DG5F [*]----[CHASSIS]----[*] RH56
          |      (origin)
          |
         -Y (RH56 side)

          +X (front) →
```

| Robot | Position (x, y, z) | 방향 |
|-------|-------------------|------|
| RH56  | (0.75, -1.3, -0.1) | 오른쪽 (y-) |
| DG5F  | (0.75,  1.3, -0.1) | 왼쪽 (y+) |

### 검증된 좌표 (복사해서 사용)

**RH56 (오른쪽):**
```
0.1 -0.3 0.6 0.0 0.7071 0.0 -0.7071
0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071
0.3 -0.5 0.6 0.0 0.7071 0.0 -0.7071
1 -0.5 0.5 0.0 0.7071 0.0 -0.7071
```

**DG5F (왼쪽):**
```
0.31 0.2 0.8 0.0 0.7071 0.0 -0.7071
0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071
0.3 0.5 0.6 0.0 0.7071 0.0 -0.7071
1 0.5 0.5 0.0 0.7071 0.0 -0.7071
```

**대칭 조합 예시:**

| RH56 | DG5F |
|------|------|
| `0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071` | `0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071` |
| `0.3 -0.3 0.6 0.0 0.7071 0.0 -0.7071` | `0.3 0.3 0.6 0.0 0.7071 0.0 -0.7071` |
| `1 -0.5 0.5 0.0 0.7071 0.0 -0.7071` | `1 0.5 0.5 0.0 0.7071 0.0 -0.7071` |

---

## 7. Orientation 참고

모든 예제에서 사용하는 기본 orientation:

| 표현 | 값 |
|------|-----|
| Euler (deg) | (180, -90, 0) |
| Quaternion (qw, qx, qy, qz) | (0.0, 0.7071, 0.0, -0.7071) |

이 orientation은 **로봇 hand가 쭉 편 상태로 샤시 모듈 방향(-X)을 바라보는 자세**입니다.
- EE의 Z축 (approach 방향) → -X (world)
- EE의 Y축 → -Y (world)
- EE의 X축 → -Z (world, 아래)

GUI 버전에서는 이 orientation이 초기값으로 설정되어 있고, 큐브 드래그 시 position만 변경됩니다.
Command 버전에서는 매번 직접 입력해야 합니다.

---

## 8. Troubleshooting

### INVALID_START_STATE_JOINT_LIMITS
- **원인:** 로봇 초기 관절 상태가 joint limit 밖
- **해결:** 로봇이 안정화될 때까지 잠시 대기 후 다시 시도

### INVALID_START_STATE_SELF_COLLISION
- **원인:** 로봇 초기 자세에서 self-collision 감지
- **해결:** `dual_hdr35_20_ati.yml`에서 `self_collision_buffer` 값을 더 음수로 조정

### IK_FAIL
- **원인:** 도달 불가능한 위치 또는 orientation
- **해결:** 위의 검증된 좌표 사용, 또는 더 가까운 좌표로 변경

### Planning FAILED (COLLISION)
- **원인:** 충돌 없는 경로를 찾지 못함
- **해결:** 두 팔이 교차하지 않는 좌표 사용 (RH56은 y-, DG5F는 y+)

### GUI 버전에서 큐브 옮겨도 반응 없음
- **원인:** 두 큐브가 모두 멈춰야 planning 트리거됨
- **해결:** 큐브를 놓은 후 1-2초 대기. 로봇이 아직 움직이고 있어도 대기 필요

---

## 9. Quick Reference

```bash
# Single-arm GUI
python examples/isaac_sim/motion_gen_reacher_hyundai.py \
    --robot hdr35_20_ati_rh56f1_r_sensor.yml --visualize_spheres

# Single-arm Command
python examples/isaac_sim/motion_gen_reacher_command.py \
    --robot hdr35_20_ati_rh56f1_r_sensor.yml --visualize_spheres

# Dual-arm GUI
python examples/isaac_sim/motion_gen_reacher_dual_hyundai.py --visualize_spheres

# Dual-arm Command
python examples/isaac_sim/motion_gen_reacher_dual_command.py --visualize_spheres
```
