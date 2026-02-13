# HYDEX Dual-Arm Motion Planning Examples

## Overview

HYDEX는 두 대의 Hyundai HDR35_20 로봇으로 구성된 듀얼암 시스템입니다:
- **RH56**: RH56F1_R 그리퍼 장착 (오른쪽, y-)
- **DG5F**: DG5F_L 그리퍼 장착 (왼쪽, y+)

## Robot Positions (World Frame)

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

| Robot | Position (x, y, z) | Notes |
|-------|-------------------|-------|
| RH56  | (0.75, -1.3, -0.1) | 오른쪽 로봇 |
| DG5F  | (0.75,  1.3, -0.1) | 왼쪽 로봇 |

---

## Scripts

### 1. GUI 버전 (Box 드래그)
```bash
python examples/isaac_sim/motion_gen_reacher_dual_hyundai.py --visualize_spheres
```

**사용법:**
1. Isaac Sim에서 Play 버튼 클릭
2. RED 큐브 (RH56) 또는 BLUE 큐브 (DG5F) 드래그
3. 두 큐브가 모두 움직임을 멈추면 자동으로 planning 시작

**주의:** Box 드래그 시 orientation 지정 어려움 → Command 버전 권장

---

### 2. Command 버전 (좌표 입력) ⭐ 권장
```bash
python examples/isaac_sim/motion_gen_reacher_dual_command.py --visualize_spheres
```

**사용법:**
1. Isaac Sim에서 Play 버튼 클릭
2. 터미널에서 RH56 타겟 입력 (저장됨, 실행 안 함)
3. DG5F 타겟 입력 → **Planning 시작!**

**입력 형식:**
```
x y z qw qx qy qz
```

---

## Verified Working Coordinates (검증된 좌표)

### RH56 Examples
```bash
0.1 -0.3 0.6 0.0 0.7071 0.0 -0.7071
1 -0.5 0.5 0.0 0.7071 0.0 -0.7071
0.3 -0.5 0.6 0.0 0.7071 0.0 -0.7071
0.3 -0.3 0.6 0.0 0.7071 0.0 -0.7071
0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071
```

### DG5F Examples
```bash
0.31 0.2 0.8 0.0 0.7071 0.0 -0.7071
1 0.5 0.5 0.0 0.7071 0.0 -0.7071
0.3 0.5 0.6 0.0 0.7071 0.0 -0.7071
0.3 0.3 0.6 0.0 0.7071 0.0 -0.7071
0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071
```

### 조합 예시
| RH56 | DG5F | 비고 |
|------|------|------|
| `0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071` | `0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071` | 대칭 위치 |
| `0.3 -0.3 0.6 0.0 0.7071 0.0 -0.7071` | `0.3 0.3 0.6 0.0 0.7071 0.0 -0.7071` | 대칭 위치 |
| `1 -0.5 0.5 0.0 0.7071 0.0 -0.7071` | `1 0.5 0.5 0.0 0.7071 0.0 -0.7071` | 전방 대칭 |

---

## Collision Settings (충돌 설정)

### Config 파일 위치
```
IsaacLab/src/nvidia-curobo/src/curobo/content/configs/robot/dual_hdr35_20_ati.yml
```

### 주요 설정

| 설정 | 현재 값 | 설명 |
|------|---------|------|
| `collision_sphere_buffer` | `-0.01` | 전체 collision sphere 반지름 조정 (음수=축소) |
| `self_collision_buffer` | HAND만 `-0.01` | ARM은 기본값(0), HAND만 음수 |

### self_collision_buffer 설명

```yaml
self_collision_buffer: {
  # ARM 링크: 지정 안 함 → 기본값(0) → inter-arm collision 감지 O

  # HAND 링크만 -0.01: 손가락 내부 충돌 무시 (반지름이 작아서 필요)
  'rh56_gripper_base_link': -0.01,
  'rh56_plam_1': -0.01,
  'rh56_right_thumb_1': -0.01,
  ...
  'dg5f_ll_dg_palm': -0.01,
  'dg5f_ll_dg_1_1': -0.01,
  ...
}
```

### Buffer 값의 의미

| Buffer 값 | 효과 | 사용 시점 |
|-----------|------|----------|
| **양수** (+0.01) | Sphere 확대 → 충돌 감지 강화 | 안전 마진 필요 시 |
| **0** (기본값) | 원래 크기 | 일반적인 경우 |
| **음수** (-0.01) | Sphere 축소 → 충돌 감지 완화 | 작은 링크의 false positive 방지 |

### Sphere 반지름 참고

| 부위 | 반지름 범위 | -0.01 적용 시 |
|------|-------------|--------------|
| ARM 링크 | 0.07 ~ 0.12 m | 0.06 ~ 0.11 m ✅ |
| ATI 센서 | 0.042 m | 0.032 m ✅ |
| Gripper base | 0.035 ~ 0.04 m | 0.025 ~ 0.03 m ✅ |
| 손가락 | 0.010 ~ 0.018 m | 0.000 ~ 0.008 m ⚠️ |

---

## Coordinate Conversion

기존 single-robot 좌표를 dual-robot (HYDEX) 좌표로 변환:

```
새 좌표 = 기존 좌표 + 로봇 위치
```

| Robot | 변환 공식 |
|-------|----------|
| RH56  | new = old + (0.75, -1.3, -0.1) |
| DG5F  | new = old + (0.75,  1.3, -0.1) |

---

## Troubleshooting

### INVALID_START_STATE_SELF_COLLISION

**원인:** 로봇 초기 상태에서 자기 충돌 감지됨

**해결:**
1. `dual_hdr35_20_ati.yml`에서 `self_collision_buffer` 값을 더 음수로 변경
2. 특정 링크만 문제라면 해당 링크의 buffer만 조정
3. `retract_config`를 안전한 초기 자세로 변경

**예시:**
```yaml
# 기존: -0.01 → 변경: -0.02
'rh56_gripper_base_link': -0.02,
```

### Inter-arm Collision 미감지 (팔끼리 통과)

**원인:** ARM 링크에 음수 buffer가 설정되어 있음

**해결:**
1. ARM 링크는 `self_collision_buffer`에서 **제거** (기본값 0 사용)
2. HAND 링크만 음수 buffer 유지

### IK_FAIL

**원인:**
- 도달 불가능한 위치
- Orientation이 workspace 범위 밖
- 경로에 장애물

**해결:**
1. 더 가까운 위치로 타겟 변경 (위의 검증된 좌표 참고)
2. Orientation 값 확인 (quaternion 정규화 필요)
3. `--visualize_spheres`로 충돌 구 확인

### Planning FAILED

**원인:**
- 시작점에서 목표점까지 충돌 없는 경로 없음
- 두 팔이 교차해야 하는 위치

**해결:**
1. 두 팔이 교차하지 않는 좌표 사용 (RH56은 y-, DG5F는 y+)
2. 중간 waypoint 추가 고려

---

## Quick Reference

```bash
# Command 버전 실행
python examples/isaac_sim/motion_gen_reacher_dual_command.py --visualize_spheres

# 검증된 좌표 예시 (복사해서 사용)
[RH56]: 0.3 -0.25 0.6 0.0 0.7071 0.0 -0.7071
[DG5F]: 0.3 0.25 0.6 0.0 0.7071 0.0 -0.7071
```

### Config 파일 빠른 수정

```bash
# 충돌 설정 수정
vim IsaacLab/src/nvidia-curobo/src/curobo/content/configs/robot/dual_hdr35_20_ati.yml

# Sphere 설정 수정
vim IsaacLab/src/nvidia-curobo/src/curobo/content/configs/robot/spheres/dual_hdr35_20_ati.yml
```
