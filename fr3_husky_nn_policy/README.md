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
| Reach (`physics_step`) | 상대 `0.1 * action` offset | offset을 50 ms 동안 유지하고, 매 컨트롤러 주기마다 `q_target = measured_q + offset`을 다시 계산 |
| Reach (`policy_step`) | 정책 시점의 절대 `measured_q + 0.1 * action` 목표 | 계산한 절대 목표를 다음 20 Hz 정책 시점까지 유지 |

Reach는 학습에 사용한 공칭 액추에이터 값 `Kp=80`, `Kd=4`를 사용하며, 관절
1--4의 effort 한계는 87 Nm, 관절 5--7의 한계는 12 Nm입니다. 두 모드 모두 기존
궤적 limiter를 우회하고 컨트롤러 update마다 같은 PD, effort limit 및
`1000 Nm/s` torque-rate limit을 적용합니다. timeout, 유한값, 관절 한계, 최대
정책 offset 검증도 동일하게 유지됩니다.

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
사용하지 않습니다. Reach의 목표 갱신 기준은 `reach_policy_step_action`으로
선택합니다. 기존 `relative_target_refresh_hz` 파라미터는 action API 호환성을
위해 남아 있지만, 컨트롤러는 더 이상 이를 gate로 사용하지 않습니다.

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

- 기본 모델: models/dual_fr3_reach_actor_friction_w_1000hz.npz
- 네트워크: 58 → 256 → 128 → 64 → 14
- 결정론적 출력: tanh(마지막 선형 계층), float32 범위 `[-1, 1]`
- Launch: dual_fr3_reach_policy.launch.py
- 노드: ppo_reach_policy_node

`reach_policy_step_action`은 action semantic만 선택하며 모델 파일을 자동으로 바꾸지
않습니다. 100 Hz/1000 Hz 및 matched/unmatched 조건에 맞는 actor는 `model_path`로
별도 지정해야 합니다. 패키지의 기존 actor를 사용할 때는 기본값인
`reach_policy_step_action:=false`를 유지하고, policy-step으로 새로 학습·export한
actor에만 `true`를 사용하세요.

배포 모델은 `dual_fr3_reach_sim2real_v2/2026-09-09_11-05-39/model_4999.pt`에서
내보낸 것입니다. NPZ에는 `output_activation=tanh`가 저장되며, Reach 노드는
squash되지 않은 정책이 로봇에 명령하지 못하도록 기존 identity-output artifact를
거부합니다. 노드 측 `action_clip=1.0`은 수치적 안전장치로 유지됩니다.

### Reach action 기준: `physics_step`과 `policy_step`

정책 추론과 raw action 처리(`delta_q = 0.1 * action`)는 두 모드 모두 20 Hz입니다.
차이는 50 ms 동안 유지할 관절 목표를 어느 시점의 측정값으로 만드는지입니다.

| launch 값 | 의미 | 컨트롤러가 유지하는 값 |
| --- | --- | --- |
| `reach_policy_step_action:=false` (기본값) | `physics_step` | `delta_q[k]`; 매 controller update마다 `q_des(t) = q_measured(t) + delta_q[k]` |
| `reach_policy_step_action:=true` | `policy_step` | 정책 시점에 계산한 절대 목표 `q_des[k] = q_measured(t_k) + delta_q[k]` |

`policy_step`에서도 PD와 torque 계산은 20 Hz가 아니라 매 controller update에서
계속 실행됩니다. 달라지는 것은 PD가 추종하는 절대 목표를 다음 정책 호출까지
고정한다는 점입니다. 반대로 `physics_step`은 로봇이 움직일 때 기준 위치도 함께
갱신되므로 offset 오차가 계속 유지됩니다.

이 플래그는 100 Hz/1000 Hz controller 설정이나 matched/unmatched dynamics를
선택하는 플래그가 아닙니다. 네 조건 모두에서 독립적으로 사용할 수 있으며, 반드시
체크포인트를 학습할 때 사용한 action semantic과 맞춰야 합니다.

- `dual_fr3_lab`에서 `--reach_policy_step_action`으로 학습한 모델:
  `reach_policy_step_action:=true`
- 기존 `physics_step` 방식으로 학습한 모델: 플래그 생략 또는
  `reach_policy_step_action:=false`

선택한 모드는 시작 로그의 `action reference=policy_step|physics_step`과 실행 결과의
`metadata.json` 내 `run_context.action_reference_mode`에서 확인할 수 있습니다.

### Reach v2 MuJoCo 배포 정합성

다음 설정은 Reach v2의 MuJoCo 추론 경로를 Isaac Lab 학습 인터페이스와
일치시킵니다.

- **관측/모델:** Reach v2 `58 -> 14` 정책과 Isaac Lab 3.0의 `xyzw` quaternion
  convention(고정 base는 `[0, 0, 0, 1]`)을 사용합니다. 명시적인 `start_policy`
  요청이 승인될 때까지 previous-action 관측은 0으로 유지됩니다.
- **행동 타이밍:** 정책 추론은 20 Hz입니다. `physics_step`은 결과 관절 offset을
  50 ms 동안 유지하며 매 controller update에서 `q_desired = q_measured + offset`을
  다시 계산합니다. `policy_step`은 정책 시점에 만든 절대 `q_desired`를 50 ms 동안
  유지합니다. 두 모드 모두 별도의 effort hold 없이 매 controller update에서
  PD torque를 다시 계산합니다.
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

`dual_fr3_lab`에서 `--reach_policy_step_action`을 넣어 학습한 체크포인트는 다음과
같이 동일한 semantic을 명시하고, 그 체크포인트에서 export한 NPZ를 지정합니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_policy.launch.py \
  use_mujoco:=true shadow_mode:=false auto_start:=false \
  reach_policy_step_action:=true \
  model_path:=/absolute/path/to/policy_step_actor.npz
~~~

기존 physics-step 체크포인트는 플래그를 생략하거나
`reach_policy_step_action:=false`로 실행합니다. `use_mujoco`의 true/false와
관계없이 같은 규칙을 사용합니다. 이 플래그는
`dual_fr3_reach_trajectory.launch.py`에도 동일하게 제공됩니다.

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
검증하는 역할만 담당합니다. 별도의 `fit_sysid_dynamics_mujoco.py`와
`fit_sysid_dynamics.py`가 정지 residual 점검과 마찰 회귀를 수행하지만, 결과를
MuJoCo XML에 자동 반영하거나 검증 완료된 plant를 생성하지는 않습니다.

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

전체 real-hardware sweep은 dry run과 실제 수집을 각각 명시적으로 실행합니다.

~~~bash
# 14관절 reference/phase 미리보기: 로봇 명령 없음
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  sweep_mode:=true launch_move_group:=true dry_run:=true

# 위 결과와 모든 관절의 +/- 경로를 확인한 뒤 실제 수집
ros2 launch fr3_husky_nn_policy fr3_sysid.launch.py \
  sweep_mode:=true launch_move_group:=true dry_run:=false
~~~

단일 관절 모드의 `target_joint`는 0부터 13까지의 인덱스이며 순서는
`left_fr3_joint1..7`, `right_fr3_joint1..7`입니다. 기본값 `5`는
`left_fr3_joint6`입니다.

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

### inverse-dynamics 점검과 마찰 회귀

현재 권장 분석기는 `scripts/tools/fit_sysid_dynamics_mujoco.py`입니다. 수집 당시의
로봇과 동일한 EE/payload를 포함한 MJCF를 사용해야 합니다. 현재 SysID launch의
기본 구성은 plate, no gripper, no camera이므로 예를 들면 다음처럼 생성합니다.

~~~bash
source /opt/ros/jazzy/setup.bash
source /home/dyros/etri_ws/install/setup.bash

ros2 run xacro xacro \
  "$(ros2 pkg prefix fr3_husky_description)/share/fr3_husky_description/mjcf/dual_fr3.xml.xacro" \
  hand:=false with_realsense:=false with_azure:=false mobile:=false \
  -o /tmp/dual_fr3_plate_nocam.xml
~~~

먼저 정지 구간만 이용해 torque 부호, 중력, joint mapping 및 payload 정합성을
확인합니다. 그 뒤 양방향 rich-excitation CSV에 대해서만 마찰 회귀를 수행합니다.

~~~bash
# 정지 residual 점검만 수행
python3 scripts/tools/fit_sysid_dynamics_mujoco.py \
  --mjcf /tmp/dual_fr3_plate_nocam.xml \
  --csv 'logs/sysid_raw/real/*.csv' \
  --no-fit

# 관절별 Coulomb/viscous friction과 bias 회귀
python3 scripts/tools/fit_sysid_dynamics_mujoco.py \
  --mjcf /tmp/dual_fr3_plate_nocam.xml \
  --csv 'logs/sysid_raw/real/*.csv'
~~~

이 스크립트에는 MuJoCo Python package, NumPy 및 SciPy가 필요합니다. 시스템
Python에 없다면 이 repository에서 사용 중인 `cRobotics` 환경의 Python으로
`python3` 부분을 대체하세요.

분석기는 기본적으로 MJCF의 `damping`, `frictionloss`, `armature`를 0으로 만든 뒤
중력 `-9.81 m/s²`를 적용해 `mj_inverse`의 강체 torque를 계산합니다. 따라서
`tau_meas - tau_rigid_body` residual은 실제 plant의 전체 passive effect를
포함합니다. 기존 MJCF passive parameter까지 포함한 모델과의 차이를 보려면
`--keep-passive`를 사용합니다. 측정 `qdot`에는 기본 25 Hz, 4차 Butterworth
zero-phase filter를 적용하며, `--cutoff-hz`, `--gravity`, `--cruise-qdd` 등은
CLI에서 조정할 수 있습니다.

현재 출력은 terminal의 정지 residual과 관절별 `Fc`, `Fv`, bias, 회귀 RMSE입니다.
가속 excitation과 `qdd`는 강체 inverse dynamics 계산에는 사용되지만, 현재 회귀식은
constant-velocity 구간의 `sign(qdot)`, `qdot`, bias만 사용합니다. 따라서
**armature를 별도 계수로 추정하거나 held-out torque RMSE를 생성하고 XML을
갱신하는 단계는 아직 구현되지 않았습니다.**

`scripts/tools/fit_sysid_dynamics.py`는 Pinocchio/URDF 기반 비교 도구입니다. 기본
URDF는 EE가 없는 단일 FR3 모델이므로, 실제 plate 실험의 최종 fitting보다는
mapping과 gravity 검증용입니다. 상세한 가정과 현재 검증 상태는
[about_sysid.md](fr3_husky_nn_policy/about_sysid.md)를 참고하세요.

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

### Open-loop Reach action replay

`dual_fr3_reach_trajectory.launch.py`는 actor 추론을 유한 길이의 20 Hz raw-action
trajectory로 대체합니다. controller action, 선택한 action 기준 모드, 안전 검사,
EEF logger 및 policy-trace logger는 정책 rollout과 동일합니다. 각 CSV row가
logger에서 사용할 목표 중심도 포함하므로 replay 중 외부 `/reach_target_pose`
메시지는 무시합니다.

다음 세 trajectory가 패키지에 포함됩니다.

| 파일 | 내용 |
| --- | --- |
| `reach_1khz_matched_nominal.csv` | 기록된 1 kHz/matched MuJoCo rollout의 정책 action 321개 |
| `reach_1khz_matched_nominal_100hz_matched_noise.csv` | nominal action에 결정론적 zero-mean Gaussian perturbation을 더한 trajectory |
| `reach_1khz_matched_nominal_100hz_unmatched_noise.csv` | 100 Hz/unmatched rollout에서 추정한 perturbation을 더한 trajectory |

두 noisy 파일은 seed 100과 동일한 표준화 random sample을 사용합니다. 관절별 raw
action 표준편차는 time-aligned 100 Hz matched 또는 unmatched trace와 1 kHz/matched
trace 사이의 정상상태 차이에서 bias를 제거해 추정했습니다. 정확한 source hash,
추정·실현 noise scale 및 clipping 횟수는
`trajectories/reach_trajectory_metadata.json`에 기록되어 있습니다.

두 팔을 학습 ready pose로 이동한 뒤 nominal replay를 launch하고 시작합니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_trajectory.launch.py \
  shadow_mode:=false auto_start:=false \
  log_task_name:=dual_fr3_reach_trajectory_nominal

ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

위 명령은 기본 `physics_step` 방식입니다. 같은 trajectory를 `policy_step` 절대
목표 방식으로 처리하려면 launch 명령에 다음 인자를 추가합니다.

~~~bash
reach_policy_step_action:=true
~~~

즉 trajectory CSV, `noise_scale`, 20 Hz replay timing은 그대로이고 controller에
전달되는 목표의 기준만 바뀝니다. 정책 rollout과 마찬가지로 결과
`metadata.json`의 `run_context.action_reference_mode`에서 실제 선택값을 확인할 수
있습니다.

noisy replay 전에는 두 팔을 같은 ready pose로 되돌린 뒤 matched-noise CSV를
선택합니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_trajectory.launch.py \
  shadow_mode:=false auto_start:=false \
  log_task_name:=dual_fr3_reach_trajectory_noisy \
  trajectory_path:="$(ros2 pkg prefix fr3_husky_nn_policy)/share/fr3_husky_nn_policy/trajectories/reach_1khz_matched_nominal_100hz_matched_noise.csv"

ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

100 Hz/unmatched noise profile은 unmatched CSV를 선택합니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_trajectory.launch.py \
  shadow_mode:=false auto_start:=false \
  log_task_name:=dual_fr3_reach_trajectory_unmatched_noise \
  trajectory_path:="$(ros2 pkg prefix fr3_husky_nn_policy)/share/fr3_husky_nn_policy/trajectories/reach_1khz_matched_nominal_100hz_unmatched_noise.csv" \
  noise_scale:=1.0

ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

`noise_scale`은 replay 시 선택한 CSV의 `noise_*` column을 다시 scaling합니다.

- `noise_scale:=0.0`: nominal action sequence
- `noise_scale:=1.0`: CSV에 저장된 matched 또는 unmatched noise 수준
- `noise_scale:=0.5`, `noise_scale:=2.0`: 저장된 noise의 절반 또는 두 배

nominal CSV의 `noise_*` column은 모두 0이므로 `noise_scale` 값과 관계없이 nominal
action을 생성합니다. scaling 후 실제 action에는 기존 `action_clip`과 joint safety
projection이 계속 적용됩니다.

마지막 action 다음의 첫 20 Hz tick에서 controller cancel을 요청해 마지막 응답
sample까지 기록합니다. logger는 정책 rollout과 동일한 `eef_trajectory.csv`,
`policy_trace.csv`, metadata 및 plot을 생성합니다. metadata의 `run_context`에는
trajectory 경로, SHA-256, row 수, 원본 duration, launch-time noise scale 및
scaling 전후 raw-action noise RMS가 기록됩니다.

#### Control rate / MuJoCo dynamics ablation

같은 nominal action을 100/1000 Hz controller와 서로 다른 MuJoCo passive dynamics로
비교할 수 있습니다. `control_rate_hz`는 controller manager와 FR3 controller의
update rate를 함께 바꾸며, MuJoCo physics integration timestep은 비교 중에도
`0.001 s`로 고정됩니다. 세 dynamics 값은 양쪽 팔의 14개 관절에 동일하게
적용됩니다.

~~~bash
ros2 launch fr3_husky_nn_policy dual_fr3_reach_trajectory.launch.py \
  trajectory_path:="$(ros2 pkg prefix fr3_husky_nn_policy)/share/fr3_husky_nn_policy/trajectories/reach_1khz_matched_nominal.csv" \
  noise_scale:=0.0 shadow_mode:=false auto_start:=false \
  control_rate_hz:=1000 \
  mujoco_armature:=0.1 mujoco_damping:=0.003 mujoco_frictionloss:=0.2 \
  log_task_name:=reach_nominal_hz1000_a0p1_d0p003_f0p2

ros2 service call /ppo_reach_policy_node/start_policy std_srvs/srv/Trigger {}
~~~

launch를 종료하고 팔을 같은 ready pose로 되돌린 뒤 인자만 바꿔 다음 조건을
반복합니다. 먼저 아래 8-run one-factor-at-a-time matrix를 권장합니다.

| profile | `mujoco_armature` | `mujoco_damping` | `mujoco_frictionloss` |
| --- | ---: | ---: | ---: |
| matched | 0.1 | 0.003 | 0.2 |
| no-armature | 0.0 | 0.003 | 0.2 |
| no-damping | 0.1 | 0.0 | 0.2 |
| no-friction | 0.1 | 0.003 | 0.0 |

각 profile을 `control_rate_hz:=100`과 `control_rate_hz:=1000`에서 한 번씩
실행합니다. 여러 값을 동시에 제거한 unmatched 조건도 필요하면
`mujoco_armature:=0.0 mujoco_damping:=0.0 mujoco_frictionloss:=0.0`을 추가합니다.
`metadata.json`의 `run_context`에는 실제 control rate와 세 dynamics 값이 함께
기록되므로 directory 이름에만 의존하지 않고 결과 조건을 확인할 수 있습니다.
비교 시에는 각 run의 EEF error plot과 `eef_trajectory.csv`뿐 아니라
`policy_trace.csv`의 관절 속도·effort·action 진동도 함께 확인하세요.

기록된 ablation run으로부터 모든 CSV를 다시 생성하려면 다음 명령을 사용합니다.

~~~bash
cd /home/dyros/etri_ws/src/fr3_husky_etri/fr3_husky_nn_policy
python3 scripts/generate_reach_action_trajectories.py
~~~

generator의 `--noise-scale`은 새 CSV에 저장할 noise 자체를 바꿉니다. launch 인자
`noise_scale`은 CSV를 다시 생성하지 않고 기존 noisy CSV를 scaling합니다.

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
