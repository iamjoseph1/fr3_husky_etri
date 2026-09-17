# FR3 Husky NN 정책

이 ROS 2 패키지는 학습된 LiftCube 및 Reach PPO actor를 FR3 로봇에서 실행합니다.
두 actor는 pickle을 사용하지 않는 NPZ 파일로 내보내며 NumPy로 동작하므로,
배포 환경에는 PyTorch와 GPU가 필요하지 않습니다.

## 지원 작업

| 작업 | 제어 관절 | 정책 | 외부 작업 입력 |
| --- | --- | --- | --- |
| LiftCube | 오른팔과 그리퍼 | 36차원 관측 → 8차원 행동 | 물체 자세와 들어 올릴 목표 |
| Reach | 왼팔과 오른팔 | 58차원 관측 → 14차원 행동 | 사용자가 입력한 목표 중심 |

두 정책은 같은 정책 제어 인터페이스를 통해 20 Hz로 동작합니다. Launch 파일의
기본값은 로봇에 명령을 보내지 않고 추론만 수행하는 shadow mode입니다. 액추에이터
측 실행 방식은 작업 특성에 따라 의도적으로 다르게 구성되어 있습니다.

| 작업 | 스트리밍하는 팔 명령 | 컨트롤러 실행 방식 |
| --- | --- | --- |
| LiftCube | 절대 `q + 0.1 * action` 목표 | 기존의 1 kHz 속도·가속도·스텝 제한 궤적 |
| Reach | 상대 `0.1 * action` offset | offset을 50 ms 동안 유지하고, 매 1 kHz 컨트롤러 주기마다 `q_target = measured_q + offset` 및 Isaac 방식 PD torque를 다시 계산 |

Reach는 학습에 사용한 공칭 액추에이터 값 `Kp=80`, `Kd=4`를 사용하며, 관절
1--4의 effort 한계는 87 Nm, 관절 5--7의 한계는 12 Nm입니다. 이 경로는 Isaac
Lab의 `RelativeJointPositionAction` substep 동작을 보존하기 위해 기존 궤적
limiter를 우회합니다. timeout, 유한값, 관절 한계, 최대 정책 offset 검증은
계속 적용됩니다.

## 빌드

`PolicyJointCommand`는 7개 단일 팔 목표 또는 14개 양팔 목표를 지원하므로,
message, controller, policy 패키지를 함께 다시 빌드해야 합니다.

~~~bash
cd /home/dyros/etri_ws
colcon build --packages-up-to fr3_husky_nn_policy fr3_husky_controller
source install/setup.bash
~~~

## 공통 인터페이스

- 양팔 joint state(Reach/consistency probe): /dual_fr3/joint_states (sensor_msgs/JointState)
- 정책 명령: /policy_joint_command
- 컨트롤러 action: /fr3_policy_control
- 명령 frame: base
- NPZ 로더: allow_pickle=False를 사용하는 NumpyMLPActor

정책 목표 검증에는 `max_policy_target_delta_rad`를 사용합니다. LiftCube의
액추에이터 명령 smoothing은 `max_actuator_step_rad`, `joint_velocity_scale`,
`joint_acceleration_scale`로 각각 설정합니다. Reach는 이 세 smoothing 값을
사용하지 않습니다. 기존 `relative_target_refresh_hz` 파라미터는 action API
호환성을 위해 남아 있으며 1000 Hz로 설정되지만, 컨트롤러는 더 이상 이를 gate로
사용하지 않습니다.

### Shadow mode

로봇에 명령을 보내지 않고 작업을 실행하려면 해당 launch 명령에
`shadow_mode:=true`를 추가하세요. `shadow_mode:=false`는 로봇 상태, 초기 자세,
좌표 frame, 안전 한계를 확인한 뒤에만 사용해야 합니다.

## LiftCube

### 동작과 모델

LiftCube는 관측된 물체를 설정된 목표 쪽으로 들어 올리도록 오른쪽 FR3 팔과
그리퍼를 제어합니다. 일반적으로 ArUco 인식 pipeline에서 제공하는 물체 자세가
필요합니다.

- 모델: models/dual_fr3_lift_v3_actor.npz
- 네트워크: 36 → 256 → 128 → 64 → 8
- Launch: dual_fr3_lift_policy.launch.py
- 노드: ppo_liftcube_policy_node

### 입력과 서비스

- 물체 자세: /object_pose (geometry_msgs/PoseStamped, base frame)
- 선택적 목표 자세: 설정된 target_pose_topic
- 시작: /ppo_liftcube_policy_node/start_policy
- 중지: /ppo_liftcube_policy_node/stop_policy

### 실행 순서

오른팔을 학습 ready pose로 이동하고 그리퍼를 엽니다. 그런 다음 auto-start 없이
launch한 뒤 정책을 명시적으로 시작합니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_lift_policy.launch.py shadow_mode:=false auto_start:=false

ros2 service call /ppo_liftcube_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

정책은 다음 명령으로 중지합니다.

~~~bash
ros2 service call /ppo_liftcube_policy_node/stop_policy std_srvs/srv/Trigger {}
~~~

## Reach

### 동작과 모델

Reach는 14개 팔 관절 전체를 제어합니다. 물체 자세, 카메라, ArUco marker,
그리퍼가 필요하지 않습니다. 사용자가 로봇 base frame에서 목표 중심을 입력하면,
정책은 그 중심의 양쪽 0.20 m 지점에 해당하는 목표로 두 end effector를 이동시킵니다.

- 모델: models/dual_fr3_reach_actor.npz
- 네트워크: 58 → 256 → 128 → 64 → 14
- 결정론적 출력: tanh(마지막 선형 계층), float32 범위 `[-1, 1]`
- Launch: dual_fr3_reach_policy.launch.py
- 노드: ppo_reach_policy_node

배포 모델은 `dual_fr3_reach_sim2real_v2/2026-09-09_11-05-39/model_4999.pt`에서
내보낸 것입니다. NPZ에는 `output_activation=tanh`가 저장되며, Reach 노드는
squash되지 않은 정책이 로봇에 명령하지 못하도록 기존 identity-output artifact를
거부합니다. 노드 측 `action_clip=1.0`은 수치적 안전장치로 유지됩니다.

각 정책 단계에서 노드는 절대 목표 대신 14개 상대 관절 offset을 publish합니다.
컨트롤러는 이 offset을 50 ms 정책 주기 동안 유지합니다. reach v2에서는 Isaac
Lab의 `RelativeJointPositionAction`과 맞추기 위해 매 1 kHz 컨트롤러 주기마다
최신 측정 관절 위치와 유지된 offset을 더해 목표를 다시 계산한 뒤, PD torque와
1 Nm/update torque-rate limit을 계산합니다. 별도의 100 Hz torque gate 또는
torque-hold 단계는 없습니다.

### Reach v2 MuJoCo 배포 정합성

다음 설정은 Reach v2의 MuJoCo 추론 경로를 Isaac Lab 학습 인터페이스와
일치시킵니다.

- **관측/모델:** Reach v2 `58 -> 14` 정책과 Isaac Lab 3.0의 `xyzw` quaternion
  convention(고정 base는 `[0, 0, 0, 1]`)을 사용합니다. 명시적인 `start_policy`
  요청이 승인될 때까지 previous-action 관측은 0으로 유지됩니다.
- **행동 타이밍:** 정책 추론은 20 Hz로 수행하고, 결과 관절 offset을 50 ms 동안
  유지합니다. 매 1 kHz 컨트롤러 update에서
  `q_desired = q_measured + held_offset`을 다시 계산합니다. 추가 100 Hz effort
  hold는 없습니다.
- **Effort 제어:** `Kp=80`, `Kd=4`의 직접 PD 제어를 사용합니다. 관절 1--4의
  effort 한계는 `87 Nm`, 관절 5--7의 한계는 `12 Nm`이며 torque-rate 한계는
  `1000 Nm/s`(1 ms update당 `1 Nm`)입니다.
- **MuJoCo plant:** `timestep=0.001`, `implicitfast`, 0 gravity, Isaac Reach의
  plate/link inertia 및 팔 dynamics 값 `armature=0.1`, `damping=0.003`,
  `frictionloss=0.2`를 사용합니다.
- **목표와 상태 frame:** `/reach_target_pose` XYZ는 `base` frame에서 해석하며,
  시각화를 위해서만 MuJoCo world 좌표로 변환합니다. 팔별 Y offset(왼쪽 `+0.2 m`,
  오른쪽 `-0.2 m`)은 정확히 한 번만 적용합니다. 정책과 logger는 `link7`에
  local `[0, 0, 0.132] m` offset을 더한 동일한 TCP 정의를 사용합니다.
- **ROS/MoveIt:** 유효한 팔 상태는 `/dual_fr3/joint_states`에서 직접 사용하고
  중복 joint-state publisher를 만들지 않습니다. ready-pose planning이 시작
  collision으로 거부되지 않도록, 동일 end-effector assembly에 고정된
  camera/plate contact pair를 허용합니다.

검증한 MuJoCo 실행 `20260909_191258_839088_KST`는 두 명령 목표 중심에 모두
도달했습니다. 각 목표의 마지막 2초 동안 평균 왼쪽/오른쪽 TCP error는
`(0.5, 0.0, 0.3) m`에서 `0.41/0.49 mm`, `(0.4, 0.0, 0.1) m`에서
`3.62/4.25 mm`였습니다. 최종 action standard deviation의 평균은 `8.3e-4`,
평균 절대 관절 속도는 `3.9e-4 rad/s`로, 이전의 진동 동작이 아닌 안정적인
수렴을 보였습니다.

### 입력과 서비스

- 목표 중심: /reach_target_pose (geometry_msgs/PoseStamped, base frame)
- 대화형 목표 입력: reach_target_cli
- 시작: /ppo_reach_policy_node/start_policy
- 중지: /ppo_reach_policy_node/stop_policy

### 시간 기반 목표 시퀀스

`reach_goal_sequence`에 `dual_fr3_lab/config/reach_goal_sequence.example.json`과
동일한 `goals` 형식의 JSON 파일을 지정합니다. 패키지에 포함된 예시는
`config/reach_goal_sequence.example.json`입니다. 첫 목표는 `start_policy`의
컨트롤러 action이 승인될 때 적용되고, 이후 목표들은 각 `duration_s`가 지난 뒤
monotonic clock에 맞춰 전환되므로 타이머 지터가 누적되지 않습니다. 시퀀스가
목표를 소유하는 동안에는 외부 `/reach_target_pose` 메시지를 무시합니다.

기본적으로 노드는 마지막 duration 후 정책을 취소합니다. 대신 첫 목표부터
반복하려면 `reach_goal_sequence_loop:=true`로 설정하세요.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_policy.launch.py \
  shadow_mode:=false auto_start:=false \
  reach_goal_sequence:="$(ros2 pkg prefix fr3_husky_nn_policy)/share/fr3_husky_nn_policy/config/reach_goal_sequence.example.json"

ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

목표는 반드시 학습 범위 안에 있어야 합니다.

| 축 | 최솟값 | 최댓값 |
| --- | ---: | ---: |
| x | 0.40 m | 0.60 m |
| y | -0.10 m | 0.10 m |
| z | 0.10 m | 0.35 m |

### 실행 순서

다음 순서대로 실행하세요. 두 팔이 학습 ready pose에 도달하고 Reach 목표를
publish하기 전에는 정책을 시작하지 마세요.

1. 터미널 1에서 auto-start 없이 컨트롤러와 Reach 정책 노드를 launch합니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_policy.launch.py shadow_mode:=false auto_start:=false launch_move_group:=true use_mujoco:=true
~~~

2. 터미널 2에서 두 팔을 학습 ready pose로 이동합니다. 명령이 성공적으로
   끝날 때까지 기다리세요.

~~~bash
ros2 run fr3_husky_task_manager move_to_joint
~~~

3. 터미널 3에서 target CLI를 시작하고 원하는 중심 위치를 미터 단위의 `x y z`로
   입력합니다. 아래 예시는 로봇 base frame에서 `(0.50, 0.00, 0.20)`을
   publish합니다.

~~~bash
source /home/dyros/etri_ws/install/setup.bash
ros2 run fr3_husky_nn_policy reach_target_cli
~~~

~~~text
reach target> 0.50 0.00 0.20
~~~

정책 실행 중 목표를 바꿔야 한다면 이 CLI를 계속 열어 두세요.

4. ready-pose 이동과 목표 publish가 끝나면 터미널 4에서 정책을 시작합니다.

~~~bash
ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

두 팔은 다음 명령으로 중지합니다.

~~~bash
ros2 service call /ppo_reach_policy_node/stop_policy std_srvs/srv/Trigger {}
~~~

## 양팔 응답 비교 consistency probe

> **명명:** 이 도구는 System Identification(SysID)이 아닌 **응답 비교 /
> consistency probe**입니다. 공통 PD 경로에서 관절 위치 `q(t)`가 일치하는지는
> 컨트롤러가 크게 지배하는 기구학 수준의 점검입니다. 설정/응답 불일치를 드러낼 수는
> 있지만 plant dynamics를 식별하거나 검증하지는 **않습니다**. 실제 식별 노드와
> 분리하기 위해 코드 식별자는 `sysid_*`에서 `consistency_*`로 바꿨습니다. 이유와
> 실제 SysID 로드맵(real data에 대한 inverse-dynamics torque-residual)은
> [about_sysid.md](fr3_husky_nn_policy/about_sysid.md)를 참고하세요.

`fr3_consistency_probe_node`는 결정론적인 단일 관절 profile을 사용해 Reach와
동일한 measured-relative effort-control 경로를 구동합니다. `sweep_mode:=true`이면
14개 팔 관절을 한 번에 하나씩 실행하고 관절마다 CSV 하나를 기록합니다. profile은
`pulse`, `step`, `ramp`입니다. `pulse`는 짧은 step up/down, `step`은 변위된
reference 유지, `ramp`는 선형 상승과 복귀를 의미합니다.

`dry_run:=true`는 14관절 state mapping, profile timing, CSV logging만
검증합니다. `RunPolicyControl` goal이나 정책 명령을 보내지 않으므로 실제 probe
실행이 아닙니다. 먼저 dry run을 수행하고, 이어서 `dry_run:=false`로 같은
sweep을 수행해 action server, command 경로, 로봇의 측정 응답을 검증하세요.

실제 hardware에서 재현성 있는 비교 sweep을 수행하려면
`return_to_start_between_trials:=true`를 활성화하세요. 먼저 `move_to_joint`로
수동 확인한 collision-free q0에 로봇을 위치시킵니다. probe는 정지 상태가 될
때까지 기다린 뒤 측정된 14관절 q0를 캡처하고, 선택된 각 관절에 다음 순서를
실행합니다.

~~~text
q0 stable -> q0 hold -> step/ramp up -> hold -> step/ramp back to q0
          -> q0 position + velocity settle -> next joint
~~~

초기의 `MoveToJoint`는 collision을 고려해 q0에 도달하는 방법이지만, 모델에
없는 fixture, cable, 사람까지 보호할 수는 없습니다. 이는 trial 사이에서
probe가 사용하는 방식이 **아닙니다**. 이 옵션이 활성화되면 probe는 하나의
`RunPolicyControl` goal을 계속 유지하고 `command_rate_hz`마다
`offset = q_reference - q_measured`를 갱신합니다. 따라서 활성 관절만 profile
변위를 받고, 14개 관절 전체는 동일한 PolicyControl 경로로 q0에 유지됩니다.
새 trial을 시작하려면 모든 측정 관절이 q0에서 0.01 rad 이내이고 0.02 rad/s
미만인 상태가 0.75초 동안 유지되어야 합니다.

모든 명령은 repository root에서 실행하세요(먼저
`cd /home/dyros/fr3_husky_etri`). `launch_move_group:=true`는 별도의
`move_to_joint` 명령이 사용하기 때문에만 필요하며, probe 자체는 trial 도중이나
trial 사이에 MoveIt으로 제어를 넘기지 않습니다. 두 실행을 직접 비교할 수 있도록
MuJoCo와 real sweep은 `use_mujoco`만 다르고 나머지는 동일한 인자를 사용합니다.
둘 다 `ramp` profile과 실제 기본 amplitude(관절 1--4는 `0.05 rad`, 관절 5--7은
`0.03 rad`)를 사용하며, `home_position_tolerance_rad:=0.015`는 trial별 q0
복귀 tolerance를 약간 완화합니다.

~~~bash
# MuJoCo sweep
ros2 launch fr3_husky_nn_policy fr3_consistency_probe.launch.py \
  use_mujoco:=true \
  launch_move_group:=true \
  profile_type:=ramp \
  sweep_mode:=true \
  return_to_start_between_trials:=true \
  home_position_tolerance_rad:=0.015

# 실제 로봇(use_mujoco의 기본값은 false)
ros2 launch fr3_husky_nn_policy fr3_consistency_probe.launch.py \
  launch_move_group:=true \
  profile_type:=ramp \
  sweep_mode:=true \
  return_to_start_between_trials:=true \
  home_position_tolerance_rad:=0.015

# 정지한 뒤에만 q0를 캡처하고 전체 sweep을 시작합니다.
ros2 service call /fr3_consistency_probe_node/start_probe std_srvs/srv/Trigger {}
~~~

### raw CSV 저장 위치

`fr3_consistency_probe.launch.py`의 기본 `log_base`는
`<cwd>/logs/consistency_raw`이며, launch 파일이 `use_mujoco`에 따라 자동으로
`mujoco/` 또는 `real/`을 덧붙입니다. repository root에서 실행하므로, 각 sweep은
활성 관절마다 CSV 하나(양팔 전체 sweep은 14개, 파일당 약 1.2 MB)를 repository에
기록합니다.

~~~text
logs/consistency_raw/
├── mujoco/   # use_mujoco:=true  → left/right_fr3_joint{1..7}_*.csv
└── real/     # use_mujoco:=false → left/right_fr3_joint{1..7}_*.csv
~~~

다른 위치에 기록하려면 `log_base:=/some/other/dir`로 override하세요. raw sweep은
크고 재생성할 수 있으므로, `logs/consistency_raw/`를 version 관리할지 gitignore할지는
프로젝트별로 결정하세요.

마지막 trial도 q0로 돌아가 그곳에서 settle합니다. q0 mode에서는 의도적으로
`return_to_start_after_final_trial:=false`를 무시합니다. 명시적인 q0가 필요할
때만 `home_joint_positions`(`left_fr3_joint1..7`, `right_fr3_joint1..7` 순서의
14개 값)을 입력하세요. 이 값은 이미 정지한 측정 로봇과 q0 tolerance 안에서
일치해야 합니다. 그렇지 않으면 probe는 예기치 않게 로봇을 움직이는 대신 시작을
거부합니다. `return_to_start_between_trials:=false`는 trial이 고정 자세를 공유하지
않아 parameter fitting에 적합하지 않은, 기존의 빠른 measured-relative smoke-test
sweep에만 사용하세요.

실제 기본값은 관절 1--4에서 부호가 있는 `+0.05 rad` offset, 관절 5--7에서
`+0.03 rad` offset이며, 포화 전 공칭 PD term은 각각 4.0 및 2.4 Nm입니다. 시작 전
probe는 컨트롤러의 `0.02 rad` joint-limit margin에 들어가는 offset을 거부합니다.
또한 `max_live_amplitude_rad`를 명시적으로 올리지 않는 한 실제 amplitude를
0.05 rad로 제한합니다. 선형 모델의 양방향 데이터를 얻으려면 동일한 양수·음수
sweep을 별도로 실행하세요. 실행 중인 run은 다음 명령으로 중지합니다.

~~~bash
ros2 service call /fr3_consistency_probe_node/stop_probe std_srvs/srv/Trigger {}
~~~

CSV에는 명령한 활성 관절의 절대 reference `q_ref_active_rad`, q0에 대한 profile
변위 `q_ref_delta_from_q0_rad`, PolicyControl에 전달한 실제 measured-relative
명령 `cmd_offset_rad`가 기록됩니다. `tau_cmd_recon_nm`은 컨트롤러의 torque 크기
및 1000 Nm/s rate limit 이전의 순수 `80 * cmd_offset - 4 * qdot` PD 값입니다.
`tau_meas_*`는 joint-state source가 유한 effort 값을 제공할 때만 torque 기반
식별에 유용하며, 제공하지 않으면 probe가 경고합니다. 모든 새 CSV는 해당 run에서
사용한 단일 14관절 reference `q0_*`도 기록합니다.

### sweep 분석

`scripts/tools/analyze_consistency_probe_logs.py`는 활성 관절 이름으로 MuJoCo와
real CSV를 짝지은 뒤, 각 trial의 pulse 전 baseline을 빼고 response overlay와
machine-readable summary를 생성합니다. dynamics 모델을 fit하는 도구는 **아니며**,
Isaac Lab / Isaac Sim 식별 실행 전에 설정·응답 불일치를 확인할 수 있게 합니다.
`--root`에는 `mujoco/`, `real/` subfolder가 들어 있는 directory를 지정하세요.

~~~bash
# repository root에서 두 sweep이 logs/consistency_raw/에 기록된 뒤 실행
python3 scripts/tools/analyze_consistency_probe_logs.py \
  --root logs/consistency_raw \
  --out-dir logs/consistency_probe_comparison
~~~

`--out-dir`의 기본값은 `logs/consistency_probe_comparison`입니다. 각 실행은
version 관리해도 되는 가벼운 결과물(총 수 MB)을 생성합니다.

~~~text
<out-dir>/
├── position_overlay.png     # 관절별 MuJoCo와 real 상대 위치
├── effort_overlay.png       # 관절별 명령/측정 effort
├── cross_coupling.png       # 비활성 관절의 off-axis 응답
├── initial_pose_delta.png   # 짝지은 실행 간 q0 불일치
├── summary.csv              # 관절별 응답 비율과 RMSE
├── cross_coupling.csv       # 관절별 최대 off-axis coupling
└── report.md                # 핵심 metric(짝지은 trial, RMSE 등)
~~~

`report.md`의 response ratio는 두 trial이 동일한 payload/contact/gravity 조건에서
동일한 14관절 q0로 시작할 때만 의미가 있습니다. 비율을 신뢰하기 전에
`initial_pose_delta.png`를 확인하세요.

## 양팔 torque SysID 데이터 수집

`fr3_sysid_node`는 별도로 구성한 실제 **SysID 실험 데이터 수집기**입니다.
closed-loop 위치 overlay를 식별 결과로 부르지 않습니다. 대신 같은 effort-control
경로로 14개 관절 전체를 하나의 정지 q0에 유지하고, 한 번에 한 관절씩 rich
excitation을 적용하면서 측정 `q`, `qdot`, `tau_meas`를 기록합니다. 이 값들은
[about_sysid.md](fr3_husky_nn_policy/about_sysid.md)에 설명한 후속
inverse-dynamics torque-residual fit의 입력입니다. 이 노드는 **데이터를 수집하고
검증할 뿐, 아직 MuJoCo parameter를 fit하거나 fit된 모델을 주장하지 않습니다.**

관절별 trial은 다음 두 단계를 모두 포함하며, 다음 관절 전에 q0로 복귀해
settle합니다.

~~~text
q0 prehold
  -> 여러 일정 속도에서 +A / 0 / -A / 0 (양방향, 마찰)
  -> 여러 주파수의 Hann-windowed sine motion (가속도, 관성)
  -> q0 최종 hold -> 측정 q0 위치 + 속도 settle
~~~

일반 관절의 기본 profile은 `[0.05, 0.10, 0.15] rad/s`에서 `0.15 rad` 마찰
excursion을 수행한 뒤, `[0.5, 1.0] Hz`에서 `0.05 rad` sine을 세 cycle 수행합니다.
관절 5--7은 기본적으로 excursion의 절반을 사용합니다. 속도 transition은
부드럽고 실제 constant-velocity 구간을 남깁니다. step/ramp만으로는 Coulomb 및
viscous friction을 식별할 수 없으므로 이 특성이 중요합니다. 노드는 설정된
excursion, 속도, 가속도, q0 joint-limit guard 밖의 실제 profile을 거부합니다.

### 안전한 실행 순서

repository root에서 실행하세요. 먼저 수동으로 확인한 collision-free q0로
이동합니다. `launch_move_group:=true`는 이 별도 이동에만 필요합니다. 기본 실행은
의도적으로 `dry_run:=true`입니다. 이 경우 q0를 캡처하고 preview CSV만 기록하며,
PolicyControl goal과 로봇 명령을 보내지 않습니다.

~~~bash
# 1. 하나의 관절만 시작합니다(기본 target_joint는 5).
#    기록된 phase/reference signal을 확인하며, 로봇은 움직이지 않습니다.
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  launch_move_group:=true \
  dry_run:=true

ros2 service call /fr3_sysid_node/start_probe std_srvs/srv/Trigger {}

# 2. q0, collision-free +/- excursion, dry log를 검증한 후
#    실제 로봇 움직임을 명시적으로 활성화해 같은 명령을 반복합니다.
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  launch_move_group:=true \
  dry_run:=false

ros2 service call /fr3_sysid_node/start_probe std_srvs/srv/Trigger {}

# 필요하면 즉시 중지합니다. 활성 PolicyControl goal이 취소됩니다.
ros2 service call /fr3_sysid_node/stop_probe std_srvs/srv/Trigger {}
~~~

`sweep_mode:=true`는 상속된 14관절 sweep 목록을 선택합니다. 다만 선택한 q0에서
모든 관절의 전체 양·음 excursion이 collision-free임을 수동으로 검증한 뒤에만
사용하세요. 시작 전 노드는 모든 관절 속도가 `0.02 rad/s` 미만인 상태를 `0.75 s`
동안 기다리고, 그 측정 q0를 캡처합니다. `q0 +/- full_excursion`이 컨트롤러의
`0.02 rad` joint-limit margin에 들어가면 실행을 거부합니다. 전체 sweep 동안 하나의
`RunPolicyControl` goal을 유지하며 trial 사이에 MoveIt으로 제어를 넘기지 않습니다.

MuJoCo에서 동일한 수집 경로를 실행하려면 plant switch만 바꾸세요.

~~~bash
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  use_mujoco:=true \
  dry_run:=false
~~~

일반 ROS array parameter인 `friction_speeds_rad_s`, `inertia_frequencies_hz`는
위 노드 기본값을 유지합니다. 다른 excitation bandwidth를 승인한 경우에만 ROS
parameter file에서 override하세요. scalar인 `friction_amplitude_rad`,
`inertia_amplitude_rad`, `friction_blend_s`, `inertia_cycles` 및 세 개의
`max_live_*` guard는 launch argument입니다. guard를 올리는 것은 더 큰 reference를
명시적으로 허용하는 행위일 뿐, 실제 경로가 collision-free라는 증거는 아닙니다.

### SysID raw CSV

`fr3_sysid.launch.py`는 `<cwd>/logs/sysid_raw/{mujoco,real}/`에 기록하므로,
이 repository에서 시작한 실행은 plant별로 분리됩니다.

~~~text
logs/sysid_raw/
├── mujoco/  # use_mujoco:=true
└── real/    # hardware 실행
    └── left/right_fr3_joint*_sysid_rich_<timestamp>.csv
~~~

각 행에는 phase label, `q_ref_*`, `qd_ref_active_rad_s`,
`qdd_ref_active_rad_s2`, 전체 측정 14관절 `q_*`, `qd_*`, `tau_meas_*`,
measured-relative PD 명령, 재구성한 unclamped PD torque, 공통 `q0_*`가
포함됩니다. inverse-dynamics fit에는 문서화한 filter로 **측정** velocity에서
가속도를 계산해야 합니다. reference 가속도는 provenance/품질 signal일 뿐,
측정된 운동을 대체하지 않습니다. 유한하지 않은 `tau_meas_*`가 포함된 hardware
실행은 진단용으로 보존하지만 torque SysID 데이터로는 인정하지 않습니다.

### Reach 궤적 로그

성공한 `start_policy` 요청은 20 Hz 궤적 로그를 시작합니다. MuJoCo 실행은
`log/dual_fr3_reach`, 실제 하드웨어 실행은 `log/dual_fr3_reach_real` 아래에
저장됩니다. launch 종료 시 두 모드 모두 동일한 CSV 데이터, metadata, PNG plot
4개를 아래에 기록합니다.

~~~text
fr3_husky_nn_policy/log/<dual_fr3_reach|dual_fr3_reach_real>/<start time>/
├── eef_trajectory.csv
├── policy_trace.csv
├── metadata.json
├── eef_x_vs_target.png
├── eef_y_vs_target.png
├── eef_z_vs_target.png
└── eef_trajectory_3d.png
~~~

EEF 좌표는 학습과 동일하게 각 `fr3_link7` frame의 local
`[0, 0, 0.132] m` offset을 더해 정의합니다. plot의 팔별 목표는 명령한 중심에서
base-frame Y축 방향으로 ±`0.20 m`를 더한 값입니다. 기본 출력 root를 바꾸려면
`log_root` ROS parameter를 설정하세요.

## actor 다시 내보내기

exporter는 PyTorch가 포함된 Isaac/RSL-RL 환경에서만 실행하세요.

~~~bash
export_ppo_actor /path/to/model_4999.pt /path/to/output_actor.npz
~~~

tanh-squash를 사용하는 Reach Sim2Real v2와 같은 정책은 마지막 결정론적 분포
transform을 명시적으로 포함하세요.

~~~bash
export_ppo_actor \
  /path/to/dual_fr3_reach_sim2real_v2/2026-09-09_11-05-39/model_4999.pt \
  /path/to/dual_fr3_reach_actor.npz \
  --output-activation tanh
~~~

출력 파일명 예시는 dual_fr3_lift_v3_actor.npz, dual_fr3_reach_actor.npz입니다.
내보낸 archive에는 마지막 output activation을 포함한 numeric array와 metadata가
들어 있으며, 실행 시 allow_pickle=False로 로드됩니다. `output_activation`이 없는
artifact는 이전 버전 호환성을 위해 format-v1 identity actor로 유지됩니다.
