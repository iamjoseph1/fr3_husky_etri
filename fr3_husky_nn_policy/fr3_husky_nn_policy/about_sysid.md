# SysID 관련 정리 (배경 · 개념 · 로드맵)

이 문서는 `fr3_consistency_probe_node` / `analyze_consistency_probe_logs.py`로 한 작업이
**엄밀한 의미의 System Identification(SysID)이 아니라는** 결론에 도달한 논의와,
앞으로 진짜 sim↔real 정량 검증(SysID)을 어떻게 진행할지를 정리한 노트다.

> 용어 정정: 지금까지 "SysID probe"라고 부른 것은 실제로는
> **response-comparison / consistency probe**(응답 비교·정합성 점검)다.
> 코드 식별자(`fr3_consistency_probe_node`, `fr3_consistency_probe.launch.py`,
> `analyze_consistency_probe_logs.py`, `logs/consistency_raw/`)는 그대로 두되, 개념적으로는
> "SysID를 했다"가 아니라 "설정/응답 정합성을 점검했다"로 이해한다.

---

## 1. 무엇이 문제였나

지금까지 한 것:

```
q_ref(ramp) → [PD 제어기 Kp=80, Kd=4] → [plant] → q_measured
```

- MuJoCo / Isaac / real 에 **같은 PD + 같은(복사한) dynamics 파라미터**를 넣고
  같은 ramp를 준 뒤, **관절 위치 q(t)가 겹치는지**를 비교했다.
- 그리고 "겹친다 → SysID가 잘 됐다"라고 해석했다. **이 해석이 틀렸다.**

이유:

1. **PD 피드백이 plant 차이를 지운다.** 피드백 제어기는 오차를 강제로 0으로
   끌고 가므로, plant(관성·마찰)가 달라도 q는 ref를 추종해 겹친다. 즉
   "q 겹침"은 "plant가 같음"의 증거가 **아니다**. 두 제어기가 각자 잘 추종해서
   겹쳐 보이는 것일 뿐이다.
2. **ramp은 저주파·소진폭(q0 근처) 가진**이라 관성(가속에서 드러남)과
   마찰(다양한 속도·속도반전에서 드러남)이 q에 거의 실리지 않는다.
   dynamics 파라미터가 애초에 **식별 불가능(poorly identifiable)** 한 조건이다.
3. **q(t) 비교는 kinematic level**이다. dynamics(힘↔운동) 정보는 **torque**와
   "torque↔가속" 관계에 실려 있는데, 그것을 fit한 적이 없다.
4. **정책을 real에 돌린 실험도 아니다.** 고정 PD ref 추종일 뿐이라, "정책이
   real에서 되나"(목표 A)도 "sim이 real과 같나"(목표 B)도 증명하지 못한다.

`analyze_consistency_probe_logs.py`의 docstring 자체가 이미 이 한계를 명시하고 있다:
> "It does **not** fit a dynamics model: it makes configuration and response
> mismatches visible before an Isaac Lab/Isaac Sim identification run."

즉 이 도구는 **관절 매핑/부호/초기 q0/스케일 같은 gross error를 잡는 사전
점검용**이며, 그 자체로는 identification이 아니다. (이 용도로는 유효하다.)

---

## 2. 개념 정리

### 2.1 제어기 vs plant (가장 중요)

```
[제어기]  ──τ──▶  [plant = 물리]  ──▶  q, q̇
                                        ↓
         ◀──────── 센서 피드백 ─────────┘
```

- **제어기**: `τ`를 *계산*한다. 우리 것은 PD `τ = Kp·offset − Kd·q̇` (model-free,
  M·C·g·마찰을 안 씀). sim이든 real이든 **동일**.
- **plant**: `τ`를 받아 *실제로 움직인다*.
  - sim: MuJoCo/Isaac **물리엔진**이 `M(q)q̈ + C(q,q̇)q̇ + g(q) + F(q̇) = τ`를
    적분해 운동을 계산. **armature/damping/frictionloss/링크 관성**이 여기 산다.
  - real: 실제 로봇. 그 값들은 **물리적 실체로 존재**할 뿐, 우리가 코드에 적는
    "설정 값"이 아니다 → **미지 = 식별 대상**.
- **MuJoCo는 제어기가 아니라 plant다.** "동역학 제어기"를 쓰는 게 아니라,
  동역학 *방정식(물리)* 을 푸는 것이다.

주의: real FR3 펌웨어는 내부적으로 **중력보상(+일부 마찰보상)** 을 수행한다.
그래서 "우리가 제어하는 plant"는 순수 로봇이 아니라 "로봇 + Franka 내부보상"일
수 있다. 마찰 식별 시 반드시 확인/처리해야 한다.
(참고: 실측 CSV의 `tau_meas`는 정지 시 joint2 ≈ −23.6 Nm 등 **중력을 포함한
전체 관절토크**로 보고된다.)

### 2.2 kinematics vs dynamics

| | Kinematics(기구학) | Dynamics(동역학) |
|---|---|---|
| 다루는 것 | 운동의 기하(위치/속도/가속의 관계) | 힘↔운동의 관계 |
| 대표식 | FK/IK, Jacobian `ẋ = J(q)q̇` | `M(q)q̈ + C q̇ + g(q) + F(q̇) = τ` |
| 등장요소 | 각도·링크길이·속도 (질량·힘 없음) | 질량·관성·마찰·토크 |

**q(t)만 비교 = kinematic level 비교.** dynamics를 보려면 torque가 필요하다.

### 2.3 "torque를 냈지만 dynamics를 안 썼다"

- PD는 τ를 **출력**하지만, τ를 **계산**할 때 dynamics 모델(M·C·g·마찰)을 쓰지
  않는다(model-free). computed-torque `τ = M q̈_des + C q̇ + g + ...` 는 τ 계산에
  dynamics 모델을 **쓴다**. 둘 다 τ를 내지만, 모델을 쓰는 건 후자뿐이다.
- plant는 sim/real 모두 **항상** dynamics를 따른다(물리). "dynamics를 안 쓴다"는
  건 오직 *제어기*가 모델을 안 쓴다는 뜻이다.

### 2.4 RL(PD 배포)에 dynamics 모델이 필요한가

- **제어 법칙만 보면 불필요** — PD+정책은 model-free.
- 하지만 **transfer 성공을 위해서는 sim plant ≈ real plant** 여야 한다. 정책은
  학습 중 Isaac plant의 dynamics를 *암묵적으로* 학습한다. real plant가 다르면
  "이 offset이면 이렇게 움직일 것"이라는 정책의 가정이 깨져 성능이 떨어진다
  (= sim-to-real gap). 이를 줄이는 수단이 (a) **SysID(sim을 real에 맞춤)**,
  (b) **domain randomization(정책을 둔감하게)**, (c) 그냥 배포 후 경험적 확인.
- 참고: transfer가 성공해도 그것은 "이 task/영역에서 gap이 충분히 작다"는
  *간접 증거*일 뿐, "dynamics를 정확히 식별했다"의 증명은 아니다. 특히 DR을 쓰면
  dynamics가 좀 틀려도 성공할 수 있으므로, 정량 검증은 여전히 torque로 해야 한다.

---

## 3. 목표 두 가지를 분리

- **목표 A — "정책이 real에서 되나?"**: 정책을 real에 실제로 돌려 task 성공/궤적
  확인. SysID가 필수는 아님(경험적 검증).
- **목표 B — "sim plant가 real과 같나?" (= SysID / plant 검증)**: torque↔운동을
  비교해야 하며, **PD 아래 q(t) 일치로는 증명 불가**.

우리 원래 목적("MuJoCo↔real 정량 검증")은 **목표 B**다. DR은 이미 사용 중이므로,
**1순위 = 목표 B(SysID로 sim을 real에 맞춤)**로 진행한다.

---

## 4. 현재 데이터 현황

`fr3_consistency_probe_node`가 남기는 raw CSV(`logs/consistency_raw/{mujoco,real}/`)에는
SysID에 필요한 신호가 이미 들어 있다:

- `t_sec`, `active_joint`, `cmd_offset_rad`, `tau_cmd_recon_nm`
- `q_*`(14), `qd_*`(14), **`tau_meas_*`(14)** — real은 중력 포함 전체 관절토크
- (q0-reference 프로파일에서는 `q_ref_delta_from_q0_rad`, `q0_*`도 기록)

**한계**: 현재 가진(느린 ramp, q0 근처 소진폭, 단일 관절)은 관성·마찰이 거의
안 실려서 **식별용으로는 부적합**. 값 자체는 풍부한 가진으로 새로 받아야 한다.
`q̈`는 `qd`의 필터링 미분으로 얻는다.

---

## 5. 앞으로 할 것 (SysID 로드맵)

식별 대상 우선순위:
1. **관절 마찰** `F(q̇) = Fc·sign(q̇) + Fv·q̇ (+offset)` — 개체/온도/마모 의존,
   URDF에 없음. **가장 중요한 미지수.**
2. **reflected rotor 관성(armature)** — 가속항에 실림.
3. 링크 관성(M, C, g)은 Franka URDF / pinocchio / libfranka `model` 값을 신뢰
   (새로 식별하지 않아도 됨).

### 방법 A (권장): inverse-dynamics torque-residual 회귀

측정 `(q, q̇, q̈)`와 강체모델로부터:
```
residual(t) = tau_meas − [ M(q)q̈ + C(q,q̇)q̇ + g(q) ]  ≈  F(q̇)
```
관절별로 `residual`을 `sign(q̇)`, `q̇`에 대해 최소자승 회귀 → `Fc`, `Fv` 추정.
armature는 가속항 `I_rotor·q̈`를 회귀항에 추가해 동시 추정 가능.
**forward 시뮬을 돌리지 않으므로 적분 드리프트가 없고, 제어기 오염도 없다.**

### 방법 B (대안): MuJoCo `mj_inverse` 직접 fit

측정 `(q, q̇, q̈)`를 `mj_inverse`에 넣어 예측 τ를 구하고,
MuJoCo 파라미터(frictionloss/damping/armature)를 조정해
`‖τ_pred − tau_meas‖`를 최소화(비선형 최적화). MuJoCo 고유 효과까지 반영.

### 절차

1. **데이터 품질 선점검(quick win)**: 지금 있는 real CSV로 방법 A를 적용해
   `residual` vs `q̇` 형태와 `tau_meas` 사용 가능성(중력 포함 여부, 부호,
   Franka 내부보상 영향)을 먼저 확인. → 새 가진 설계 전에 신호 유효성 판단.
2. **풍부한 가진 설계**: 느린 ramp 대신 관절별로 **다양한 속도/가속을 sweep**
   (chirp / multi-sine / 여러 속도의 사다리꼴). torque 측정 필수.
3. **마찰(+armature) 식별**: 방법 A로 관절별 `Fc, Fv (, I_rotor)` 추정.
4. **sim 파라미터 반영**: MuJoCo `frictionloss←Fc`, `damping←Fv`,
   `armature←I_rotor`. Isaac도 동일하게. (무작정 Isaac 기본값 복사 X)
5. **정량 검증**: held-out 궤적에서 inverse-dynamics **torque prediction error**
   (관절별 RMSE / 상대오차 %)를 리포트 → "MuJoCo가 real과 X% 이내로 일치"라고
   비로소 말할 수 있다. 필요 시 짧은 구간 forward-sim도 보조 지표로.

## 6. 구현된 데이터 수집 단계

`fr3_sysid_node` / `fr3_sysid.launch.py`가 위 절차의 **1~2단계(데이터 품질
확인과 풍부한 가진 수집)** 를 담당한다. 기존 consistency probe의 q0 latch,
joint-limit margin, 14-joint stationary check, PolicyControl ownership, CSV
logging을 재사용하되, 다음을 별도 구현했다.

- 저속/중속의 양방향 constant-velocity 구간: `Fc`, `Fv` 회귀용
- Hann-windowed sine 가속 구간: armature/관성 잔차를 보이기 위한 excitation
- `q_ref`, `qdot_ref`, `qddot_ref`, 전체 14-joint `q`, `qdot`, `tau_meas`, q0
  를 같은 CSV에 기록
- real hardware는 `dry_run:=true`가 기본이며, 실제 명령은
  `dry_run:=false`를 명시해야만 가능

여기까지는 **식별 데이터 수집**이다. pinocchio/libfranka/MuJoCo inverse
dynamics를 호출하는 회귀·파라미터 반영·held-out torque RMSE는 아직 구현하지
않았다. 특히 `qddot_ref`가 아니라 필터링한 **측정 `qdot`의 미분**을 회귀에
사용해야 한다.

---

## 7. 열린 이슈 / 체크리스트

- [x] real `tau_meas`가 **중력 포함**인지 확정 → **중력 포함, 내부 중력보상
      안 된 원시 link-side 토크**. 정지에서 `tau_meas ≈ g(q)`로 확인(§8).
      (내부 **마찰보상** 여부는 움직이는 데이터의 잔차로 최종 확인 예정)
- [x] 강체모델 소스 결정 → **MuJoCo `mj_inverse`** (튜닝 대상과 동일). pinocchio
      URDF 경로도 검증용으로 병행(`fit_sysid_dynamics.py`).
- [x] `q̈` 미분 필터 → **4차 Butterworth 영위상(filtfilt) 후 미분** 구현
      (`fit_sysid_dynamics*.py`, 기본 25 Hz cutoff).
- [x] 풍부한 가진 프로파일을 별도 `fr3_sysid_node`로 분리하고 real/MuJoCo CSV
      logging 구현.
- [ ] plate 장착 real 데이터로 `dual_fr3_plate` 모델 정지-check → Isaac plate
      관성 ≠ 실제 plate면 plate 질량/COM load-id 보정.
- [ ] MuJoCo/Isaac 파라미터를 어디(XML/config)에서 일괄 관리할지.
- [ ] 검증 metric·리포트 포맷 정의(관절별 torque RMSE, %, 플롯).

---

## 8. 검증 완료 사실 & 실행 방법 (2026-09-16)

### 8.1 확정된 사실 (기존 consistency real CSV로 검증)

- **모델 엔진 = MuJoCo `mj_inverse`.** 팔 장착 pose가 트리에 baked-in이라
  per-arm 중력방향이 자동 처리됨(pinocchio 수동 팔별 hack 불필요).
- **EE(말단) 질량이 잔차를 지배.** 같은 정지 데이터에서:
  - 맨팔(EE 없음): worst 정지 잔차 **~4.0 Nm** (j2/j4)
  - gripper 포함 모델: worst **~0.3~0.6 Nm** (≈ 전체 토크의 **~1.4~2.7%**)
  → 즉 EE payload를 모델에 넣는 것이 결정적.
- **`tau_meas`는 중력 포함 원시 토크** (정지에서 `tau_meas ≈ g(q)`, 부호 일치).
  fit 식 `residual = tau_meas − [Mq̈+Cq̇+g]` 그대로 유효.
- **남은 잔차 = SysID 타깃**: 수직축(j1/j7) ~0.15 Nm = 정지 마찰(stiction) 추정,
  j2/j4 ~0.3~0.6 Nm = EE 질량 미세 오차(양팔 비대칭 포함).
- **pulse/ramp(consistency) 데이터로는 마찰 fit 불가**: 한 방향 운동이라
  `sign(q̇)` 상수 → `Fc`와 bias가 collinear(`Fc==bias`). → **양방향 sweep 필수**.

### 8.2 EE config (중요)

| 용도 | EE | xacro args |
| --- | --- | --- |
| 기존 consistency 데이터 검증 | gripper, 카메라 X | `hand:=true with_realsense:=false` |
| **SysID 본실험 (이걸로 진행)** | **plate, 카메라 X** | `hand:=false with_realsense:=false` |

주의: `hand:=false`면 MJCF가 link7에 **Isaac 학습용 plate 병합 관성**
(`fr3_isaac_reach_inertials.yaml`)을 넣는다. 이는 **실제 plate와 같다는 보장이
없으므로** plate 장착 real 데이터로 검증/보정해야 한다.

### 8.3 실행 방법 (이 저장소 기준, 재현용)

MuJoCo/scipy는 conda `cRobotics` env에 있다(base python3엔 없음).

~~~bash
# (1) MJCF 생성: xacro → MuJoCo xml.  이 머신엔 husky_description mjcf가 없어
#     husky include를 빈 stub 매크로로 대체한 사본을 쓴다(mobile:=false라 husky 미사용).
#     실제 sim 워크스페이스엔 husky mjcf가 있으니 stub 불필요.
source /opt/ros/jazzy/setup.bash
source /home/dyros/fr3_ws/install/setup.bash
#   원본: fr3_husky_description/mjcf/dual_fr3.xml.xacro
#   SysID 본실험용(plate, no cam):
xacro <stub_xacro> hand:=false with_realsense:=false with_azure:=false mobile:=false \
  -o /tmp/dual_fr3_plate_nocam.xml

# (2) 정지-residual check (플러밍 + 모델/EE/중력 검증)
/home/dyros/anaconda3/envs/cRobotics/bin/python \
  scripts/tools/fit_sysid_dynamics_mujoco.py \
  --mjcf /tmp/dual_fr3_plate_nocam.xml \
  --csv 'logs/sysid_raw/real/*.csv' --no-fit

# (3) 마찰/관성 fit (양방향 rich sweep 데이터에서만 유의미)
/home/dyros/anaconda3/envs/cRobotics/bin/python \
  scripts/tools/fit_sysid_dynamics_mujoco.py \
  --mjcf /tmp/dual_fr3_plate_nocam.xml \
  --csv 'logs/sysid_raw/real/*.csv'
~~~

`fit_sysid_dynamics_mujoco.py` 옵션: `--keep-passive`(모델의
damping/frictionloss/armature 유지, 잔차=real−model), `--gravity`,
`--cutoff-hz`, `--cruise-qdd`. 기본은 passive를 0으로 만들어 순수 강체토크를
기준으로 삼는다(잔차 = real의 전체 마찰+관성). pinocchio URDF 버전은
`fit_sysid_dynamics.py`(EE 없는 맨팔 URDF라 검증·비교용).

### 8.4 데이터 수집 (본실험)

plate 장착 + 수동 확인한 collision-free q0에서, repository 루트에서 실행:

~~~bash
# dry_run으로 q0/프로파일/CSV 먼저 확인 (로봇 안 움직임)
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  sweep_mode:=true launch_move_group:=true dry_run:=true
ros2 service call /fr3_sysid_node/start_probe std_srvs/srv/Trigger {}

# 검증되면 실제 수집 (14관절 자동 sweep, 관절당 ~40 s)
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  sweep_mode:=true launch_move_group:=true dry_run:=false
ros2 service call /fr3_sysid_node/start_probe std_srvs/srv/Trigger {}
~~~

수집 후 첫 확인: real `tau_meas_*`가 유한값인지, 그리고 (2) 정지-check 잔차가
plate 모델에서 작은지 → 크면 plate 질량/COM 보정 먼저, 그다음 (3) 마찰 fit.
