# Ant 험지 보행 강화학습 (Robotics Simulation 3주차 과제 1)

평지에서 학습한 Ant는 처음 보는 울퉁불퉁한 지형에서 거의 걷지 못합니다. 이번 과제에서는 팀원들이 각자 찾은 방법과 이전에 제출했던 설정을
같은 조건에서 하나씩, 또 섞어서 학습해 보고, 여러 지형에서 점수가 가장 높은 조합을 골랐습니다.

제출 모델은 [`artifacts/combo_v28/submission/model_599.pt`](artifacts/combo_v28/submission/)입니다.
Stick 팀원의 험지와 Lim 팀원의 박스 지형을 섞어서 학습하고, PPO entropy 계수를 0.005로 두고, 1,000번 학습한 뒤 600번을 더 학습한 모델입니다(학습 seed 43).

발표 자료는 [웹 슬라이드](https://williewonker777.github.io/robotics-simulation-week03-ant-robust/report/web/)에서 바로 볼 수 있습니다. 방향키로 넘기고 N 키로 발표 메모를 볼 수 있습니다.
같은 내용의 [PDF](report/week03_ant_v28_slides.pdf)도 있고, 저장소를 받은 뒤 [`report/web/index.html`](report/web/index.html)을 브라우저로 열어도 됩니다.

## 결과 요약

수업 평가 방식대로 환경 100개에서 첫 에피소드 동안 받은 보상의 합을 평균했습니다. 같은 방식으로 28가지 지형과 마찰 조건에서 잰 값을 다시 평균해 비교했습니다.

| | 제공 baseline | 이전 제출 모델 (Robust42) | Lim 팀원 모델 (F3a) | 제출 모델 |
|---|---:|---:|---:|---:|
| 28개 조건 평균 | 30.1 | 33.6 | 61.0 | 62.2 |
| 새로 만든 지형에서 다시 평가 | 30.1 | 33.5 | 60.8 | 62.5 |
| 박스 지형 (블록 높이 위아래 최대 10 cm) | 4.2 | 4.8 | 59.5 | 60.1 |
| 평지 | 138.4 | 146.6 | 84.9 | 93.1 |

학습 seed를 세 번 바꿔 학습한 평균이고, F3a는 팀원이 고른 모델 하나의 값입니다.
험지에서는 크게 좋아졌지만 평지에서는 평지만 학습한 모델보다 느리게 걷습니다.

[![박스 지형에서 제공 baseline과 제출 모델 비교](artifacts/combo_v28/media/boxes_10_baseline_vs_submission.gif)](artifacts/combo_v28/media/boxes_10_baseline_vs_submission.mp4)

박스 지형에서 왼쪽은 제공 baseline, 오른쪽은 제출 모델입니다. baseline은 3.6초 만에 넘어지고 제출 모델은 16초를 끝까지 걷습니다. 그림을 누르면 MP4가 열립니다.

## 기본 코드에서 바꾼 것

수업에서 받은 기본 코드(IsaacLab_RS의 `Isaac-Ant-v0` 환경과 `AntPPORunnerCfg` 학습 설정)와 제출 모델의 설정을 비교하면 아래와 같습니다.
바꾼 것은 다섯 가지이고, 나머지는 기본 코드 그대로입니다.

| 항목 | 기본 코드 | 제출 모델 | 바꾼 코드 |
|---|---|---|---|
| 학습 지형 | 평평한 바닥 (`terrain_type="plane"`) | 지형 생성기로 만든 험지. 작은 요철, 물결, 완만한 경사(Stick 팀원)와 높이가 다른 블록(Lim 팀원)을 반씩 섞음 | [`combo_v28_cfg.py`](src/week03_ant/tasks/combo_v28_cfg.py)의 `stick_lim_terrains`, `apply_terrain` |
| 몸통 높이 관측 | 몸통의 월드 z 좌표 (`base_pos_z`) | 몸통 아래 바닥까지의 거리. 몸통 위에서 아래로 광선 하나를 쏴서 잼 | 같은 파일의 `use_relative_height`, `base_height_above_ground` |
| 넘어짐 판정 | 몸통 z가 0.31 m 미만 (`root_height_below_minimum`) | 바닥에서 몸통까지 거리가 0.31 m 미만 | `height_above_ground_below_minimum` |
| PPO entropy 계수 | 0.0 | 0.005 | 학습 명령의 `agent.algorithm.entropy_coef=0.005` |
| 학습량 | 1,000 iteration | 1,000 iteration 학습 후 그 가중치로 600 iteration 더 학습 | [`train_combo_v28.py`](scripts/train_combo_v28.py)의 `--init_checkpoint` |

그대로 둔 것은 보상 7가지와 가중치, 행동(관절 토크 8개, scale 7.5), 나머지 관측(전체 60차원), 신경망 [400, 200, 100],
다른 PPO 설정(learning rate 5e-4, clip 0.2, epoch 5 등), 환경 4,096개와 32 step, 에피소드 16초, 바닥 마찰 1.0입니다.

높이 관측과 넘어짐 판정은 코드로 보면 이렇게 바뀌었습니다.

```python
# 기본 코드 (isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py)
base_height = ObsTerm(func=mdp.base_pos_z)
torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})

# 제출 모델 (src/week03_ant/tasks/combo_v28_cfg.py, use_relative_height)
env_cfg.scene.height_ray = height_ray_cfg()   # 몸통 위 20 m에서 아래로 쏘는 광선 1개
env_cfg.observations.policy.base_height = ObsTerm(func=base_height_above_ground, params={"sensor_cfg": SceneEntityCfg("height_ray")})
env_cfg.terminations.torso_height = DoneTerm(func=height_above_ground_below_minimum,
                                             params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_ray")})
```

파일과 task 이름에 붙은 `v28`은 이 저장소에서 진행한 실험 번호입니다(앞선 실험 기록은 [전체 실험 기록](docs/EXPERIMENT_HISTORY.md)에 있습니다).
번호와 상관없이 제출 모델이 기본 코드와 다른 점은 위 다섯 가지입니다.

## 개선 과정

위 변경을 기본 코드에 하나씩 더하면서 점수를 확인했습니다. 단계마다 학습 seed 세 개로 따로 학습해서 평가했고, 세 단계 모두 세 번 다 점수가 올랐습니다.

| 단계 | 바꾼 것 | 28개 조건 평균 | 박스 지형 | 평지 | 넘어진 비율 |
|---|---|---:|---:|---:|---:|
| 시작 | 제공 baseline (평지에서 학습) | 30.1 | 4.2 | 138.4 | 48% |
| 1 | 학습 지형을 험지로, 높이 관측과 넘어짐 판정은 바닥 기준으로 | 43.4 | 40.9 | 70.6 | 45% |
| 2 | PPO entropy 계수 0.0에서 0.005로 | 58.7 | 56.5 | 87.0 | 25% |
| 3 | 학습한 가중치로 600 iteration 더 학습 | 62.2 | 60.1 | 93.1 | 22% |

![단계별 점수와 효과가 없었던 방법](artifacts/combo_v28/plots/improvement_ladder.png)

**1단계, 험지에서 학습.** 평평한 바닥 대신 울퉁불퉁한 바닥에서 학습했습니다. Stick 팀원이 쓴 험지(작은 요철, 높이 5~15 cm 물결, 완만한 경사)와
Lim 팀원이 쓴 박스 지형(0.45 m 크기 블록마다 높이가 위아래로 최대 10 cm씩 다른 바닥)을 반씩 섞었습니다.
이때 몸통 높이도 바닥에서부터 재도록 바꿨습니다. 기본 Ant는 높이를 월드 좌표의 z값으로 받기 때문에 바닥이 울퉁불퉁하면 실제 높이와 맞지 않습니다.
그래서 몸통 아래로 광선 하나를 쏴서 바닥까지 거리를 재고, 넘어짐 판정(몸통 높이 0.31 m 미만)도 같은 기준으로 했습니다. 평지에서는 두 값이 같습니다.
이 단계에서 박스 지형 점수는 4.2에서 40.9로 올랐지만 평지에서는 느려졌습니다.

**2단계, entropy 계수 0.005.** PPO의 entropy 보너스는 학습 중에 정책이 너무 빨리 한 가지 동작으로 굳지 않고 여러 동작을 계속 시도하게 하는 항입니다.
Lim 팀원 설정대로 0에서 0.005로 올렸더니 점수가 가장 크게 올랐고 넘어지는 비율도 절반 가까이 줄었습니다. 나머지 PPO 설정은 원래 그대로입니다.

**3단계, 추가 학습.** 1,000번 학습한 모델을 불러와 같은 조건으로 600번 더 학습했습니다(Lim 팀원 방식).
여기서 학습 1번은 환경 4,096개에서 32 step씩 모은 경험으로 정책을 한 번 업데이트하는 것을 말합니다.

### 효과가 없었던 방법

같은 단계에서 다른 선택을 해 보았지만 점수가 떨어졌습니다.

- 이전 제출 모델(Robust42)의 랜덤화를 2단계에 더하면 2.1점 낮아졌습니다. 이 랜덤화는 학습 중에 바닥 마찰, 몸통 질량, 무게중심, 시작 자세,
  몇 초마다 미는 힘, 관측 노이즈를 무작위로 바꾸는 설정입니다. 평지와 미끄러운 바닥에서는 도움이 됐지만 험지에서는 효과가 없었습니다.
- 3단계를 Stick 팀원의 회복 보상으로 학습하면 5.7점 낮았습니다. 회복 보상은 학습할 때 넘어지거나 몸이 기울면 감점하고 속도 보상에 상한을 두는 방식입니다.
  넘어지는 비율은 22%에서 16%로 줄었지만 조심스럽게 걸어서 점수가 낮았습니다.
- 1단계 지형 대신 이전에 쓰던 험지 세트(장애물, 계단, 급경사 등 9가지)로 학습하면 18.3점 낮았습니다.

다른 조합에서도 결과는 같은 방향이었습니다. 다른 조건을 똑같이 두고 entropy만 바꾼 12쌍은 모두 점수가 올랐고(평균 8.1점),
평지 대신 팀원 지형으로 바꾼 12쌍도 모두 올랐습니다(평균 16점). 랜덤화를 켠 경우는 12쌍 중 7쌍만 올랐습니다(평균 1.1점).

## 평가 명령어

수업 환경(Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, RSL-RL 3.0.1)에서 저장소 루트를 기준으로 실행합니다.

```bash
export ROBOTICS_SIM_CLASS_ROOT=/mnt/ssd970/robotics_simulation_class   # 각자 수업 환경 경로
source "$ROBOTICS_SIM_CLASS_ROOT/activate.sh"
python -m pip install --no-deps -e .

python scripts/play_one_episode_official.py --task Week03-Ant-Combo-v28-Play \
  --num_envs 100 --seed 24 --headless \
  --checkpoint artifacts/combo_v28/submission/model_599.pt
# [RESULT] Episode reward total: mean=61.985765, std=17.888110   (박스 지형, 두 번 실행해 같은 값)
```

[`scripts/play_one_episode_official.py`](scripts/play_one_episode_official.py)는 수업 저장소(cailab-hy/IsaacLab_RS e83a5d2)의 `play_one_episode.py`에
우리 task를 등록하는 import 한 줄만 추가한 파일입니다. `Week03-Ant-Combo-v28-Play`는 원래 `Isaac-Ant-v0`과 보상, 종료 조건, 행동, 관측 구성이 같고
높이 관측과 넘어짐 판정만 바닥 기준으로 바꾼 평가용 task입니다. 기본 지형은 박스 지형입니다.
다른 지형으로 평가하려면 [`combo_v28_cfg.py`](src/week03_ant/tasks/combo_v28_cfg.py)의 `ComboV28PlayEnvCfg`에서 `scene.terrain`만 바꾸면 됩니다.
GPU가 두 개인 PC에서는 `--device cuda:1 --kit_args="--/renderer/multiGpu/enabled=false"`를 붙여 실행했습니다.

## 실험 방법

- 모든 모델은 수업 기본 설정을 그대로 씁니다. 관측 60차원, 관절 토크 8개, 신경망 [400, 200, 100], 원래 PPO 설정, 학습량 4,096 환경 × 32 step × 1,000번입니다.
  평가 보상도 원래 Ant 보상 7가지(전진, 살아 있음, 똑바로 서 있음, 목표 쪽 이동, 행동 크기, 에너지, 관절 한계) 그대로이고, 학습에만 쓴 보상은 점수에 넣지 않았습니다.
- 첫 비교에서는 학습 지형 6가지(평지, Stick 험지, Lim 박스, 둘을 섞은 것, 이전 험지 세트, 전부 섞은 것), entropy 2가지, 랜덤화 유무 2가지를 조합한
  24가지를 학습 seed 42, 43, 44로 학습했습니다.
- 두 번째 비교에서는 상위 두 조합을 원래 보상과 회복 보상으로 600번씩 더 학습했고, 팀원 각자의 원래 방식도 같이 학습했습니다.
  이쪽은 학습량이 1.6배라 첫 비교와 따로 봤습니다. 새로 학습한 모델은 모두 84개입니다.
- 평가는 환경 100개, 평가 seed 24, 최대 16초로 각 환경의 첫 에피소드 보상 합을 구했습니다. 지형은 평지와 미끄러운 바닥 4가지, 박스 6가지,
  요철과 물결 4가지, 경사와 계단 5가지, 장애물 6가지, 틈과 구덩이 3가지로 모두 28가지이고, 이 28개 값의 평균으로 순위를 매겼습니다.
- 지형 종류, 점수 계산 방법, 모델 고르는 규칙은 학습 결과를 보기 전에 [정해 두었습니다](docs/experiment_plans/combo_v28.md).
  모델을 고른 뒤에는 모양이 다른 새 지형을 만들어 다시 평가했습니다.

## 결과 자세히

![조합별 28개 조건 평균 점수](artifacts/combo_v28/plots/ranking_selection.png)

| 순위 | 조합 | 28개 조건 평균 | 새 지형에서 다시 평가 | 넘어진 비율 |
|---:|---|---:|---:|---:|
| 1 | Stick+Lim 지형, entropy, 추가 학습 (제출) | 62.2 | 62.5 | 22% |
| 2 | Lim 박스, entropy, 랜덤화, 추가 학습 | 61.2 | 61.4 | 15% |
| | Lim 팀원 모델 (F3a) | 61.0 | 60.8 | 18% |
| 3 | Lim 박스, entropy, 추가 학습 | 60.0 | 60.5 | 22% |
| 4 | Lim 박스, entropy, 랜덤화, 회복 보상으로 추가 학습 | 58.9 | 59.1 | 8% |
| 5 | Stick+Lim 지형, entropy (추가 학습 전) | 58.7 | 58.9 | 25% |
| | 이전 제출 모델 (Robust42) | 33.6 | 33.5 | 39% |
| | 제공 baseline | 30.1 | 30.1 | 48% |

새로 만든 지형에서 다시 평가해도 순위는 같았습니다. 1위 조합은 학습 seed에 따라 61.4점에서 62.8점 사이였습니다.
수업 평가 스크립트처럼 모델마다 따로 실행해서 다시 재 봐도 순위는 같았습니다(1위 62.1, 2위 61.2, F3a 60.9).
다만 계단이 있는 두 조건에서는 여러 모델을 한 번에 이어서 평가할 때 값이 조금 달랐습니다.
조건별 점수와 원자료는 [상세 결과 문서](docs/COMBO_V28.md)에 정리했습니다.

## 영상

[![제출 모델이 여섯 가지 지형을 걷는 모습](artifacts/combo_v28/media/submission_six_terrains.gif)](artifacts/combo_v28/media/submission_six_terrains.mp4)

제출 모델이 계단, 장애물, 물결, 경사, 징검다리, 미끄러운 평지를 걷는 16초 영상입니다(편집 없음).
다섯 지형은 끝까지 걸었고 경사에서는 1.9초 만에 넘어졌습니다. 이 경사 조건에서는 환경 100개 중 32%가 넘어졌습니다.

## 한계

- 최종 조합은 두 팀원의 방법을 합친 것이고, 이전 제출 모델의 랜덤화는 평지와 미끄러운 바닥에서만 도움이 돼서 빠졌습니다.
- Lim 팀원 모델(F3a)과의 차이는 1.2점으로 크지 않습니다. 누구도 학습에 쓰지 않은 지형만 보면 0.5점 차이라 비슷한 수준으로 보는 게 맞습니다.
- 평지에서는 느립니다. 16초 동안 제출 모델은 94 m, 제공 baseline은 145 m를 갔습니다. 아주 미끄러운 바닥에서도 이전 제출 모델보다 점수가 낮습니다.
- 평가에 쓴 28가지 지형은 우리가 정한 것이고, 조교 평가 지형은 공개되지 않았습니다.
- 높이를 바닥 기준으로 관측하기 때문에 `Week03-Ant-Combo-v28-Play` task로 평가해야 합니다.
- 모든 결과는 시뮬레이션 결과이고, 조합마다 학습 seed는 세 개입니다.

## 학습 재현

```bash
# 1,000번 학습
python scripts/train_combo_v28.py --task Week03-Ant-Combo-v28-Sticklim-D0-Stock --seed 43 \
  --num_envs 4096 --max_iterations 1000 --run_name v28s1_sticklim_e5_d0_s43 --headless \
  agent.algorithm.entropy_coef=0.005

# 같은 seed로 600번 추가 학습 (가중치만 불러오고 optimizer는 새로 시작)
python scripts/train_combo_v28.py --task Week03-Ant-Combo-v28-Sticklim-D0-Stock --seed 43 \
  --num_envs 4096 --max_iterations 600 --run_name v28s2_sticklim_e5_d0+stock_s43 --headless \
  --init_checkpoint <1,000번 학습한 model_999.pt> agent.algorithm.entropy_coef=0.005

# 28개 조건 중 하나를 평가 (체크포인트 목록 JSON과 지형 이름)
echo '[{"id": "submission", "path": "artifacts/combo_v28/submission/model_599.pt"}]' > my_checkpoints.json
python scripts/evaluate_demo_v28.py --terrain stairs_10 --terrain_seed 2028 \
  --checkpoints my_checkpoints.json --output_dir outputs/my_eval --headless

python -m pytest -q tests/test_combo_v28.py
```

전체 실험을 다시 돌리는 명령은 [상세 결과 문서](docs/COMBO_V28.md#재현)에 있습니다. 같은 seed라도 GPU 물리 시뮬레이션 특성상 학습 결과가 완전히 똑같이 나오지 않을 수 있습니다.

## 관련 자료

- [상세 결과 문서](docs/COMBO_V28.md)와 [실험 전에 정한 계획](docs/experiment_plans/combo_v28.md)
- [결과 원자료](artifacts/combo_v28/): 순위와 조건별 CSV, 환경별 원시 평가, 그래프, 제출 조합 학습 기록, 영상
- [이전 README](docs/PREVIOUS_README_20261001.md): 이전 제출 모델(Robust42) 결과와 영상, 험지 연구 기록
- [전체 실험 기록](docs/EXPERIMENT_HISTORY.md), [최종 제출 가이드](docs/FINAL_SUBMISSION.md), [제출 체크리스트](docs/SUBMISSION_CHECKLIST.md), [팀원 정보](TEAM.md)
- 이번 발표 자료: [웹 슬라이드](https://williewonker777.github.io/robotics-simulation-week03-ant-robust/report/web/), [PDF](report/week03_ant_v28_slides.pdf), [웹 슬라이드 소스](report/web/)
- [이전 발표 자료](report/week03_ant_robust_report.pptx)와 [대본](report/SPEAKER_NOTES.md), [공개 범위와 검증](docs/PUBLICATION.md)

팀원 저장소: [Stick-0/isaac-ant-rough-terrain](https://github.com/Stick-0/isaac-ant-rough-terrain/tree/3cc718a4214f336fd4db7db5841fa86033b99d35),
[LimDaeKyung/IsaacLab_RS](https://github.com/LimDaeKyung/IsaacLab_RS/tree/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b) (BSD-3-Clause).
팀원 모델은 비교용으로만 썼고 이 저장소에 다시 올리지 않았습니다.
파일 무결성은 `sha256sum -c artifacts/PUBLICATION_SHA256SUMS`로 확인할 수 있습니다. [라이선스](LICENSE), [외부 코드 고지](THIRD_PARTY_NOTICES.md)
