# RL-ShadowHand-PPO

Isaac Lab 환경에서 Shadow Hand가 물체를 조작하도록 PPO 정책을 학습시키는 RL 프로젝트입니다. HO-Cap 기반 human hand/object trajectory를 reference로 사용하고, 관측값, 보상 함수, curriculum, reset strategy를 직접 설계하여 hand imitation과 object manipulation을 함께 최적화했습니다.

## Project Snapshot

| 항목 | 내용 |
| --- | --- |
| 주제 | Robot hand object manipulation with PPO |
| 환경 | Isaac Lab DirectRLEnv, Shadow Hand, HO-Cap trajectory |
| 알고리즘 | PPO with RL-Games |
| 목표 | 사람 손동작과 물체 trajectory를 따라가는 policy 학습 |
| 주요 구현 | 317-dim observation, dense reward shaping, MANO-to-Shadow DoF reference, contact-aware grasp reward, object-assist curriculum |
| 결과물 | sequence별 play video, 학습 로그 분석, 최종 보고서 |

## Results

| Sequence | Video | 비고 |
| --- | --- | --- |
| seq1 | [seq1_play_video.mp4](media/seq1_play_video.mp4) | main sequence play result |
| seq2 | [seq2_play_video.mp4](media/seq2_play_video.mp4) | optional sequence play result |
| seq3 | [seq3_play_video.mp4](media/seq3_play_video.mp4) | optional sequence play result |
| seq3 training | [seq3_train_video.mp4](media/seq3_train_video.mp4) | training rollout recording |

자세한 실험 설계와 TensorBoard 분석은 [final report](docs/RL_Final_Project_Report.docx)에 정리했습니다.

## Why This Project Matters

이 Project는 단순히 PPO를 실행하는 문제가 아니라, Issac Physical Simulation에서 로봇 손이 실제로 물체에 힘을 전달하도록 학습 신호를 설계하는 문제였습니다. Shadow Hand와 human hand는 관절 구조가 다르기 때문에, 사람 손 keypoint를 그대로 따라가는 것만으로는 자연스러운 robot joint posture나 stable contact가 만들어지지 않았습니다.

그래서 Policy가 학습 중에 다음 네 가지 정보를 동시에 보도록 환경을 재설계했습니다.

- 현재 robot hand와 object의 physical state
- human/object reference와 현재 simulation state 사이의 error
- fingertip과 object 사이의 상대 위치 및 contact force
- 1, 5, 10 frame 뒤의 future reference target

이를 통해 policy가 현재 오차만 뒤쫓는 것이 아니라, 곧 필요한 접촉 위치와 물체 이동 방향을 미리 준비하도록 만들었습니다.

## My Contributions

### 1. Observation Design

최종 observation vector는 317 dimensions로 구성했습니다.

| 구성 요소 | Dimension | 목적 |
| --- | ---: | --- |
| Hand DoF state | 44 | Shadow Hand joint position/velocity 제공 |
| Palm state | 15 | object 기준 palm position, rotation, velocity 제공 |
| Object state | 15 | object position, 6D rotation, velocity 제공 |
| Hand keypoint error | 63 | MANO reference hand와 robot hand의 spatial imitation error 제공 |
| Fingertip-object vectors | 15 | 각 fingertip이 object에 대해 어느 방향/거리인지 제공 |
| Local side direction | 15 | reference와 같은 방향에서 object를 접촉하도록 유도 |
| Hand lookahead | 99 | future hand anchor/fingertip/fingertip-object target 제공 |
| Object lookahead | 27 | future object position/rotation target 제공 |
| Object error | 9 | 현재 object trajectory tracking error 제공 |
| Contact force buffer | 15 | fingertip별 contact force history 제공 |

특히 lookahead reference를 추가한 이유는, 초기 실험에서 agent가 reference motion을 늦게 따라가면서 contact timing이 밀리는 문제가 있었기 때문입니다. 1-frame lookahead는 즉각적인 correction, 5-frame lookahead는 contact timing, 10-frame lookahead는 더 큰 trajectory 흐름을 담당하도록 설계했습니다.

### 2. MANO-to-Shadow Hand Reference

기본 코드에는 Shadow Hand finger joint posture를 지도할 수 있는 joint-space reference가 부족했습니다. 이를 보완하기 위해 `build_mano_to_shadow_dof_seq()`를 구현하여 MANO hand keypoint trajectory로부터 Shadow Hand의 approximate DoF target을 생성했습니다.

구현 내용:

- MANO thumb, index, middle, ring, little finger chain 분리
- 각 finger chain의 flexion, spread, thumb opposition/abduction 추정
- Shadow Hand joint lower/upper limit 범위로 target clamp
- reset 시 reference phase에 맞는 hand DoF posture로 초기화
- reward에서 `hand_dof_reward`로 joint-space imitation 유도

이 설계는 fingertip 위치만 맞추면서 중간 관절이 비정상적으로 꺾이거나 손가락이 어색하게 겹치는 문제를 줄이기 위한 장치입니다.

### 3. Reward Function Shaping

Reward는 하나의 단일 목표가 아니라, manipulation에 필요한 하위 능력을 분리해서 dense signal로 제공하도록 설계했습니다. 대부분의 tracking term은 `exp(-scale * error)` 형태로 구성하여 target 근처에서는 높은 reward를 주고, target에서 벗어난 상태에서도 smooth learning signal을 유지하게 했습니다.

| Reward group | 역할 |
| --- | --- |
| Hand imitation | hand keypoint, palm position, anchor, fingertip, DoF, palm rotation 추종 |
| Finger shape/topology | finger collapse, excessive overlap, unnatural spreading 감소 |
| Fingertip-object relation | object-centered contact geometry와 reference side direction 학습 |
| Contact/grasp | multi-finger contact, contact duration, force balance, stable grasp 유도 |
| Object tracking | object position, displacement, rotation, velocity, future direction 추종 |
| Curriculum bonus | early approach, pre-contact pose, grasp/manipulation phase별 reward 강조 |
| Penalty | excessive action, no-grasp rotation, early lag 억제 |

특히 `reference_side_direction_reward`, `finger_topology_reward`, `stable_grasp_reward`, `transport_support_reward`를 추가해 단순히 손끝이 물체에 가까워지는 정책이 아니라, reference와 유사한 방향에서 여러 손가락으로 물체를 지지하고 이동시키는 정책을 유도했습니다.

### 4. Curriculum, Reset, and Object Assist

긴 trajectory를 frame 0부터만 학습하면, policy가 앞부분을 통과하지 못할 때 후반부의 grasp/transport phase를 거의 경험하지 못합니다. 이를 해결하기 위해 reset과 curriculum을 재설계했습니다.

구현한 학습 안정화 요소:

- random reference phase sampling
- success-biased phase sampling
- phase success score 기반 sampling weight update
- early training에서 imitation/approach reward 강화
- object lag가 있고 hand가 contact/proximity 상태일 때만 작동하는 decaying object assist
- play mode에서는 object assist 비활성화
- hand/object tracking failure, rotation error, no-grasp duration 기반 early termination

Object assist는 policy 대신 task를 해결하는 장치가 아니라, 초기 학습에서 object reward가 너무 sparse해지는 문제를 완화하는 teacher-forcing curriculum으로 사용했습니다.

### 5. Training Monitoring

TensorBoard scalar를 단순 total reward가 아니라, 실제 grading 관점에 가까운 proxy metric으로 나누어 확인했습니다.

- `score/grading_proxy`
- `score/object_pose`
- `score/hand_motion`
- `score/contact_transport`
- `error/object_pos`
- `reward/contact`
- `reward/object_tracking`
- `metric/object_assist_gate`

이를 통해 reward가 하나의 항목에만 과적합되는지, hand imitation과 object motion이 함께 개선되는지를 확인했습니다. 보고서에서는 seq1, seq2, seq3의 reward trend와 object position error, contact transport score를 비교해 학습 추세를 분석했습니다.

## Implementation Details

### Environment

- `source/gr/gr/tasks/direct/gr/gr_env.py`
  - reference preprocessing
  - observation construction
  - reward computation
  - object assist
  - reset and termination logic
  - TensorBoard logging metrics

- `source/gr/gr/tasks/direct/gr/gr_env_cfg.py`
  - sequence/object path
  - observation/action space
  - reward weights and scales
  - contact sensor configuration
  - curriculum parameters
  - simulation and object physical properties

- `source/gr/gr/tasks/direct/gr/agents/rl_games_ppo_cfg.yaml`
  - PPO hyperparameters
  - actor-critic network
  - rollout/minibatch settings
  - adaptive learning rate schedule

### Core Settings

| Setting | Value |
| --- | --- |
| Observation dimension | 317 |
| Action dimension | 27 = 3 translation + 6D rotation + 18 finger joints |
| Action rate | 30 Hz |
| Simulation dt | 1 / 120 s |
| Decimation | 4 |
| Lookahead frames | 1, 5, 10 |
| Object mass | 0.1 kg |
| Target contact fingers | 3 |
| Reward curriculum steps | 18,000 |
| Object assist curriculum steps | 75,000 |
| PPO network | 1024, 1024, 512, 512 MLP with ELU and D2RL |

## Repository Structure

```text
.
├── data/HOCAP/                         # reference trajectories and object USD files
├── docs/
│   └── RL_Final_Project_Report.docx     # final report
├── media/                              # submitted play/training videos
├── scripts/rl_games/                   # train/play entrypoints
├── source/gr/gr/tasks/direct/gr/
│   ├── gr_env.py                       # custom RL environment
│   ├── gr_env_cfg.py                   # training environment config
│   ├── gr_env_cfg_play.py              # play config
│   └── agents/rl_games_ppo_cfg.yaml    # PPO config
├── run_train.sh
└── run_play.sh
```

## How to Run

Isaac Lab 환경이 준비되어 있다는 전제에서 실행합니다.

### Train

```bash
./run_train.sh
```

기본 task는 `Gr_shadow_train`입니다.

```bash
TASK=Gr_shadow_train ./run_train.sh
```

### Play

```bash
./run_play.sh
```

checkpoint를 지정하려면 다음처럼 실행합니다.

```bash
CKPT=/path/to/checkpoint.pth ./run_play.sh
```

## What I Learned

이 프로젝트에서 가장 크게 배운 점은 robot manipulation task에서는 reward를 단순히 "reference와 가까운가"로 정의하면 부족하다는 점입니다. 손동작 imitation, fingertip-object geometry, multi-finger contact, force balance, object trajectory tracking이 서로 얽혀 있기 때문에, policy가 물체를 실제로 움직이도록 하려면 각 단계를 분리해서 관측하고 보상해야 했습니다.

또한 TensorBoard 상의 training score가 final play rollout의 long-horizon robustness를 완전히 보장하지는 않는다는 점도 확인했습니다. seq3에서는 training metric이 개선되었지만 final play에서는 object manipulation이 충분히 재현되지 않았고, 이를 통해 curriculum assist와 실제 deployment mode 사이의 gap을 더 엄격히 검증해야 한다는 교훈을 얻었습니다.


