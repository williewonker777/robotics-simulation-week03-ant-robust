# Robust Ant PPO · 처음 보는 지형에서도 걷는 Ant — Robotics Simulation Week 03

**제출 모델: Stick+Lim 지형 + entropy 0.005 + 600 it 이어 학습, 학습 seed 43** —
[`artifacts/combo_v28/submission/model_599.pt`](artifacts/combo_v28/submission/) (SHA-256 `3801761006031b34…`)

팀원 두 명(Stick, Lim)의 결과에서 효과가 확인된 요소와 내 요소를 단독·조합으로 **같은 예산**에서 학습하고
(24조합 + 이어 학습 6가지, 각 학습 seed 3개 = 84개 새 정책), 과제 데모 평가 규칙(100 env, 각 env 첫 episode 누적 보상)으로
**28개 지형·마찰 조건**을 평가해 평균 점수가 가장 높은 조합을 골랐습니다. 지형·점수·선택 규칙은 학습 결과를 보기 전에 고정했습니다.

| 데모 평가 (100 env 첫 episode 평균 return) | 제공 baseline | 이전 제출 Robust42 | 팀원 Lim F3a (원본) | **제출 조합** |
|---|---:|---:|---:|---:|
| 28조건 평균 · 선택용 지형 (seed 2028) | 30.1 ± 3.9 | 33.6 ± 0.9 | 61.0 | **62.2 ± 0.7** |
| 28조건 평균 · 새 지형 (seed 2029, 선택에 안 씀) | 30.1 ± 3.8 | 33.5 ± 0.8 | 60.8 | **62.5 ± 0.5** |
| 박스 ±10 cm (강의 예시와 비슷한 지형) | 4.2 | 4.8 | 59.5 | **60.1** |
| 평지 (학습 때 본 환경) | 138.4 | **146.6** | 84.9 | 93.1 |
| 넘어짐률 (28조건 평균) | 48% | 39% | **18%** | 22% |

`±`는 학습 seed 3개의 표준편차입니다(F3a는 팀원이 고른 체크포인트 1개). 제출 체크포인트(seed 43) 자체는 28조건 평균 62.8이고,
공식 스크립트처럼 조건마다 새 프로세스에서 단독 평가하면 62.6입니다([공식 방식 점검](#결과)).
험지에서는 크게 좋아졌지만 **평지에서는 평지 전용 정책보다 느리게 걷는 trade-off**가 있습니다.

[평가 명령어](#평가-명령어) · [결과](#결과) · [영상](#영상) · [무엇을 바꿨나](#무엇을-바꿨나) · [실험 설계](#실험-설계) ·
[해석과 한계](#해석과-한계) · [재현](#재현) · [이전 기록](#이전-기록) · [상세 결과 문서](docs/COMBO_V28.md)

## 평가 명령어

수업 환경(Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, RSL-RL 3.0.1)에서 저장소 루트 기준입니다.

```bash
export ROBOTICS_SIM_CLASS_ROOT=/mnt/ssd970/robotics_simulation_class   # 각자 수업 환경 경로
source "$ROBOTICS_SIM_CLASS_ROOT/activate.sh"
python -m pip install --no-deps -e .

# 공식 play_one_episode.py와 같은 루프 · 100 env · 첫 episode 누적 보상의 평균/표준편차 출력
python scripts/play_one_episode_official.py --task Week03-Ant-Combo-v28-Play \
  --num_envs 100 --seed 24 --headless \
  --checkpoint artifacts/combo_v28/submission/model_599.pt
```

- [`scripts/play_one_episode_official.py`](scripts/play_one_episode_official.py)는 수업 저장소
  `cailab-hy/IsaacLab_RS@e83a5d2`의 `play_one_episode.py`에 **v28 task 등록 import 한 줄만** 더한 사본입니다.
- `Week03-Ant-Combo-v28-Play`는 원래 `Isaac-Ant-v0`의 보상 7항·종료·행동·60D 관측 구성을 그대로 쓰고,
  **높이 관측과 넘어짐 판정만 발밑 지면 기준**(몸통에서 아래로 쏜 ray 1개)으로 바꾼 평가 task입니다. 평지에서는 원래와 값이 같습니다.
  기본 지형은 강의 예시와 비슷한 **무작위 박스 ±10 cm**(지형 seed 2028)이며, 위 명령은 제출 체크포인트로
  `[RESULT] Episode reward total: mean=61.985765, std=17.888110`을 출력합니다(두 번 실행해 같은 값, v28 평가기와 소수점 6자리까지 같음).
- **다른 지형으로 평가하려면** [`ComboV28PlayEnvCfg`](src/week03_ant/tasks/combo_v28_cfg.py)의 `scene.terrain`
  (prim 경로 `/World/ground`)만 바꾸면 됩니다. 높이 ray가 이 prim을 보므로 다른 설정은 손대지 않아도 됩니다.
- 이 PC는 GPU가 2개라 `--device cuda:1 --kit_args="--/renderer/multiGpu/enabled=false"`를 붙여 실행했습니다.

v28의 28개 평가 조건은 평가기 하나로 다시 낼 수 있습니다. 한 프로세스에서 여러 체크포인트를 평가하며 공식 스크립트의 첫 reset 상태를 재현합니다.

```bash
echo '[{"id": "submission", "path": "artifacts/combo_v28/submission/model_599.pt"}]' > my_checkpoints.json
python scripts/evaluate_demo_v28.py --terrain stairs_10 --terrain_seed 2028 \
  --checkpoints my_checkpoints.json --output_dir outputs/my_eval --headless
```

## 결과

![조합별 데모 점수 — 선택용 지형](artifacts/combo_v28/plots/ranking_selection.png)

| 순위 | 조합 (seed 3개) | 데모 점수 | 학습 지형 밖 25조건 | 넘어짐률 | 박스 ±10 cm | 평지 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Stick+Lim 지형 + entropy · +600 it** | **62.2 ± 0.7** | **59.9** | 22% | 60.1 | 93.1 |
| 2 | Lim 박스 + entropy + 내 랜덤화 · +600 it | 61.2 ± 1.5 | 59.4 | 15% | 63.9 | 83.3 |
| – | Lim F3a (팀원 원본, 1,600 it) | 61.0 | 59.5 | 18% | 59.5 | 84.9 |
| 3 | Lim 박스 + entropy · +600 it | 60.0 ± 1.8 | 58.3 | 22% | 63.9 | 84.5 |
| 4 | Lim 박스 + entropy + 내 랜덤화 · +600 it 회복 보상 | 58.9 ± 2.1 | 57.4 | 8% | 62.4 | 75.0 |
| 5 | Stick+Lim 지형 + entropy (1단계, 1,000 it) | 58.7 ± 0.5 | 56.7 | 25% | 56.5 | 87.0 |
| – | Stick recovery (팀원 원본) | 52.7 | 52.0 | 16% | 29.1 | 76.4 |
| – | 내 Robust42 (평지 + 랜덤화, 이전 제출) | 33.6 ± 0.9 | 30.9 | 39% | 4.8 | 146.6 |
| – | 제공 baseline (평지) | 30.1 ± 3.9 | 27.4 | 48% | 4.2 | 138.4 |

"학습 지형 밖"은 누군가의 학습 지형과 같은 3조건(평지, 박스 ±10 cm, Stick 험지)을 뺀 평균입니다. 이 기준에서는 1위와 F3a·2위의 차이가
0.5로 사실상 같습니다. 전체 38개 조합·기준선과 범주별 표는 [결과 문서](docs/COMBO_V28.md#3-결과--선택용-지형-seed-2028)에 있습니다.

**요소별 효과** (1단계, 같은 예산, 같은 지형·seed에서 한 요소만 바꾼 짝 비교)

| 요소 | 평균 차이 | 좋아진 조합 | 해석 |
|---|---:|---:|---|
| Lim의 PPO entropy 0 → 0.005 | **+8.1** | 12/12 | 모든 지형 범주에서 일관되게 좋아짐 |
| 학습 지형 평지 → Stick·Lim·Stick+Lim 지형 | **+16** | 12/12 | 세 지형끼리는 ±0.3으로 차이 없음 |
| 학습 지형 내 v5 지형군 → Stick·Lim·Stick+Lim 지형 | +12 | 12/12 | 내 v5·전체 혼합 지형은 평지보다 +4~5에 그침 |
| 내 Robust42 랜덤화 끔 → 켬 | +1.2 | 7/12 | 평지·저마찰에서만 +4.5, 험지에는 일관된 효과 없음 |

![요소별 평균 데모 점수](artifacts/combo_v28/plots/factor_effects.png)

**이어 학습(+600 it):** 원래 보상으로 더 학습하면 세 부모 모두 **+3.5 ~ +3.9** 올랐습니다(Lim의 "학습량 증가"와 같은 방향).
Stick 회복 보상은 넘어짐을 크게 줄였지만(예: 15% → 8%) 더 조심스럽게 걸어 return은 원래 보상보다 낮았습니다.

![이어 학습 비교](artifacts/combo_v28/plots/continuation.png)

**조건별:** 제출 조합은 38개 조합·기준선 중 요철·물결, 장애물, 틈·구덩이 범주 1위, 경사·계단 2위(1위와 0.4 차이)입니다.
박스 범주는 Lim 계열이 조금 높고(7위), 평지·저마찰은 평지에서 학습한 정책들이 훨씬 높습니다(13위).
제출 조합에게 가장 어려운 조건은 매우 미끄러운 박스(유효 마찰 0.2), Isaac Lab 표준 rough, 계단 내리막, 박스 ±15 cm입니다.

![조건별 return](artifacts/combo_v28/plots/conditions_selection.png)

**확인 평가 — 선택에 쓰지 않은 새 지형 (seed 2029)**

선택이 끝난 뒤 28조건의 지형을 새 seed로 다시 만들어, 미리 정한 상위 5조합·기준선·팀원 원본 8개(29개 체크포인트, 812회)를 다시 평가했습니다.
**순위가 선택용 지형과 같았습니다:** 1위 62.5 ± 0.5(학습 지형 밖 60.2), 2위 Lim 박스 + entropy + 내 랜덤화 · +600 it 61.4 ± 1.5,
Lim F3a 60.8, Robust42 33.5 ± 0.8, 제공 baseline 30.1 ± 3.8. 제출 체크포인트(seed 43)는 63.0입니다.

![확인 평가 — 새 지형](artifacts/combo_v28/plots/ranking_confirmation.png)

[확인 평가 CSV](artifacts/combo_v28/results/confirmation_recipes.csv) · [원시 결과](artifacts/combo_v28/results/confirmation_evaluations.jsonl.gz)

**공식 방식 점검 — 체크포인트마다 새 프로세스**

v28 평가기는 한 프로세스에서 여러 체크포인트를 차례로 평가합니다. 공식 스크립트처럼 체크포인트마다 새 프로세스로 다시 평가해 보니(195회),
1·2위 조합과 F3a의 28조건 중 26조건은 환경별 return까지 같았고 **피라미드 계단이 있는 두 조건(계단 10 cm, Isaac Lab rough)만** 달랐습니다
(차이 −4.5 ~ +2.8, 방향은 체크포인트마다 다름. 앞서 평가한 체크포인트의 물리 엔진 상태가 남는 것으로 추정하며 원인은 확인하지 않았습니다).
공식 방식 28조건 평균은 1위 **62.1**(본 평가 62.2), 2위 61.2(61.2), F3a 60.9(61.0)로 **순위가 같고**, 제출 seed 43은 62.6으로 세 seed 중 여전히 가장 높습니다.
[점검 결과](artifacts/combo_v28/results/fresh_process_check.json) · [조건별 비교](artifacts/combo_v28/results/fresh_process_conditions.csv)

**학습 곡선** (제출 조합, seed 3개 · [run별 CSV·설정·체크포인트](artifacts/combo_v28/runs/))

![학습 곡선](artifacts/combo_v28/plots/training_curves.png)

1,000 it 지점의 순간 하락은 이어 학습 run이 새로 시작하며 통계가 초기화되고 episode 길이를 무작위로 시작한 기록상 현상입니다(성능 하락 아님).

## 영상

미리 정한 지형에서 16 env 배치의 env 0을 16초 전체 녹화했습니다(지형 seed 2028, 평가 seed 24, 30 fps, 편집 없음).
화면 위 글자는 첫 episode 누적 보상·전진 거리·상태이며, 첫 episode가 끝나면 로봇이 자동 reset되어 계속 움직입니다.
영상은 정성 데모이고 점수는 100 env 평가 기준입니다(배치 크기가 달라 초기 상태도 다름).

**박스 ±10 cm — 제공 baseline(왼쪽) vs 제출 모델(오른쪽)**

[![박스 ±10 cm에서 제공 baseline과 제출 모델 비교 · 6 fps 미리보기](artifacts/combo_v28/media/boxes_10_baseline_vs_submission.gif)](artifacts/combo_v28/media/boxes_10_baseline_vs_submission.mp4)

제공 baseline은 3.6초에 넘어지고(첫 episode 보상 4.3), 제출 모델은 16초를 완주합니다(61.3).

**제출 모델 — 계단 10 cm · 장애물 10 cm · 물결 15 cm · 경사 0.2 · 징검다리 · 저마찰 μ0.1**

[![제출 모델의 6개 지형 · 5 fps 미리보기](artifacts/combo_v28/media/submission_six_terrains.gif)](artifacts/combo_v28/media/submission_six_terrains.mp4)

다섯 지형은 16초를 완주했고, **경사 0.2 패널은 1.9초에 넘어졌습니다**(이 조건의 100 env 넘어짐률 32%). 실패 장면도 자르지 않았습니다.
[원본 MP4 2개·영상별 조건/SHA](artifacts/combo_v28/media/) · [녹화 스크립트](scripts/record_demo_v28.py)

## 무엇을 바꿨나

모든 v28 정책은 과제 인터페이스를 그대로 둡니다: 60D 관측 구성, 8D 관절 토크 행동(scale 7.5),
actor/critic MLP `[400, 200, 100]` ELU, 원래 `AntPPORunnerCfg`(entropy만 요소로 바꿈), **4,096 env × 32 step × 1,000 iteration**.
평가 보상은 원래 7개 항 그대로이며, 학습 전용 보상은 평가 점수에 들어가지 않습니다.

**공통 변경 (모든 v28 학습·평가):** 높이 관측과 넘어짐 판정을 월드 z 대신 **몸통 바로 아래 지면 기준**으로 바꿨습니다
(몸통 위 20 m에서 아래로 쏜 ray 1개, 넘어짐 기준 0.31 m는 그대로). 평지에서는 두 값이 같습니다.
두 팀원 모두 울퉁불퉁한 지면에서 이 변경이 필요했다고 보고했습니다.

| 요소 | 수준 | 출처 | 팀원·내 이전 결과 (각자 환경) |
|---|---|---|---|
| 학습 지형 | 평지 / Stick 험지 / Lim 박스 ±10 cm / Stick+Lim / 내 v5 지형군 / 전체 혼합 | Stick `3cc718a`, Lim `8d9eed1`, 내 v5 | Stick: 16초 생존 16% → 91% · Lim: 박스만 학습이 7종 혼합보다 +8.3 |
| PPO entropy | 0(원래) / 0.005 | Lim | 혼합 지형 +3.7, 박스 지형 +11.5 |
| 내 랜덤화 | 끔 / 켬 | 내 Robust42 | 평지 OOD 3종 평균 return +14.3% (저마찰 +21.6%) |
| 이어 학습 (2단계) | +600 it 원래 보상 / +600 it Stick 회복 보상 | Lim F3a / Stick | Lim: +5.3 · Stick: 생존 90.5 → 95.0% |

- **Stick 험지:** 요철 ±5 cm 50%, 물결 5–15 cm 30%, 완경사 0.05–0.15 오르막·내리막 각 10%.
- **Lim 박스:** 0.45 m 격자의 무작위 박스 ±10 cm만. **Stick+Lim:** 둘을 반씩.
- **내 v5 지형군:** 요철 2–12 cm, 경사 0.05–0.45, 계단 3–15 cm(오르막·내리막), 물결, 장애물 4–22 cm, 징검다리, 평지(각 1/9).
- **내 랜덤화(Robust42):** 마찰 0.45–1.35, 몸통 질량 ×0.8–1.2, COM ±2.5 cm, reset 자세·속도 교란, 4–8초마다 ±0.35 m/s push, 관측 노이즈.
- **Stick 회복 보상(학습 전용):** 넘어짐 −10, 몸체 여유(0.48 m 아래)·기울기 위험 −2, 전진 보상 4.5 m/s 상한, 행동 변화 −0.01, roll/pitch 각속도 −0.025.
- **이어 학습:** 1단계 가중치만 불러오고 optimizer·iteration은 새로 시작(Lim의 `train_finetune.py`와 같은 방식).

**제출 조합 = 공통 변경 + Stick+Lim 학습 지형 + entropy 0.005 + 원래 보상으로 600 it 이어 학습**(총 1,600 it, 내 랜덤화 없음).
91D 관측(발디딤·FK·높이 명령)·교사 prior·이력 gate 같은 내 v6–v24 연구 요소는 조교의 60D 데모 평가에 넣을 수 없어 v28에서 뺐습니다.
코드: [환경·지형·보상](src/week03_ant/tasks/combo_v28_cfg.py) · [task 등록](src/week03_ant/tasks/combo_v28.py) ·
[학습](scripts/train_combo_v28.py) · [큐](scripts/run_combo_v28.py) · [평가기](scripts/evaluate_demo_v28.py) · [집계](src/week03_ant/combo_v28_analysis.py)

## 실험 설계

- **1단계 (같은 예산):** 학습 지형 6 × entropy 2 × 내 랜덤화 2 = **24조합 × 학습 seed 42·43·44**.
  평지·entropy 0의 두 칸은 이미 같은 예산·seed로 학습한 제공 baseline과 Robust42를 그대로 씁니다(새 학습 66 run).
- **2단계 (+600 it):** 1단계 상위 2조합 × {원래 보상, Stick 회복 보상}, Stick 단독(Stick 험지 + 회복 보상),
  Lim 단독(Lim 박스 + entropy + 원래 보상), 각 seed 3개 = 18 run. 학습량이 1단계의 1.6배이므로 1단계와는 같은 예산 비교가 아닙니다.
- **데모 평가:** 과제 규칙 그대로 100 env, 평가 seed 24, 각 env의 첫 episode 누적 보상(원래 7항, 최대 16초)의 평균.
  **지형만 28조건**으로 바꾸고 모든 로봇은 160 m strip의 첫 줄에서 출발합니다. **데모 점수 = 28조건 평균 return**(조건별 동일 가중).
- **선택 규칙** ([사전 계획](docs/experiment_plans/combo_v28.md)): 3-seed 평균 데모 점수가 가장 높은 조합 →
  그 조합에서 선택용 지형 점수가 가장 높은 seed를 제출. 선택에 쓰지 않은 **새 지형 seed 2029**로 상위 5조합·기준선을 다시 평가해 공개합니다.
- **공식 스크립트와 일치:** 박스 ±10 cm에서 공식 스크립트와 v28 평가기가 baseline42 4.267905, Lim F3a 59.476919로 소수점 6자리까지 같습니다.
  평가는 결정적이어서 같은 순서로 다시 실행하면 환경별 return까지 같습니다([대조 기록](artifacts/combo_v28/results/official_parity.json)).
  체크포인트마다 새 프로세스로 평가하면 피라미드 계단 두 조건만 달라지며 순위는 같습니다([공식 방식 점검](artifacts/combo_v28/results/fresh_process_check.json)).
- **무결성:** 모든 최종 체크포인트를 다시 불러와 유한값을 확인하고, 일반 읽기와 O_DIRECT 읽기의 SHA-256이 같은지 확인했습니다
  (84 run 모두 일치, 실패·재시도 0).

| 평가 범주 | 조건 |
|---|---|
| 평지·저마찰 | 평지, 지면 μ0.5, μ0.1, 유효 μ0.2(multiply) |
| 박스 | ±5·±10·±15 cm, 0.3 m 박스 ±10 cm, 박스 ±10 cm + μ0.2(average·multiply) |
| 요철·물결 | 0–5 cm, 0–10 cm, Stick 학습 지형 혼합, 물결 15 cm |
| 경사·계단 | 경사 0.2 오르막·내리막, 계단 10 cm 오르막·내리막, Isaac Lab 표준 rough 혼합 |
| 장애물 | 장애물 10 cm, 레일 8 cm, 원기둥 10 cm, 원뿔 12 cm, 기울어진 블록 8 cm, 단상 10 cm |
| 틈·구덩이 | 틈 20 cm, 구덩이 15 cm, 징검다리 |

## 해석과 한계

- 최종 조합은 **두 팀원의 요소(Stick+Lim 학습 지형, Lim의 entropy·추가 학습)를 합친 것**입니다. 내 Robust42 랜덤화는
  저마찰에서는 효과가 있었지만 전체 점수를 올리지 못해 최종 조합에서 빠졌습니다.
- 1위와 Lim F3a의 차이(1.2)는 seed 간 표준편차(0.7)의 두 배 정도이고, 학습 지형 밖 조건만 보면 0.5입니다.
  "확실히 더 좋다"보다 "같은 수준이거나 조금 높다"가 정확합니다.
- 평지에서는 평지 전용 정책보다 느립니다(제출 체크포인트 16초 94 m vs 제공 baseline seed42 145 m). 매우 미끄러운 지면에서도 Robust42·F3a보다 낮습니다.
- 평가 지형 28조건은 우리가 고른 것이고 조교의 실제 평가 지형은 공개되지 않았습니다. 지형에 따라 순위가 바뀔 수 있어 조건별 결과를 모두 공개합니다.
- 제출 정책은 발밑 지면 높이(ray)를 관측합니다. 원래 `Isaac-Ant-v0`처럼 월드 z를 높이로 넣으면 울퉁불퉁한 지형에서 관측 의미가 달라지므로
  `Week03-Ant-Combo-v28-Play` task로 평가해야 합니다.
- 모든 결과는 시뮬레이션이며 학습 seed는 조합당 3개입니다.

## 재현

```bash
export ROBOTICS_SIM_CLASS_ROOT=/mnt/ssd970/robotics_simulation_class
source "$ROBOTICS_SIM_CLASS_ROOT/activate.sh"
python -m pip install --no-deps -e .

# 제출 조합 1단계: Stick+Lim 지형 + entropy, seed 43
python scripts/train_combo_v28.py --task Week03-Ant-Combo-v28-Sticklim-D0-Stock --seed 43 \
  --num_envs 4096 --max_iterations 1000 --run_name v28s1_sticklim_e5_d0_s43 --headless \
  agent.algorithm.entropy_coef=0.005

# 제출 조합 2단계: 위 체크포인트에서 +600 it (가중치만 불러오고 optimizer는 새로)
python scripts/train_combo_v28.py --task Week03-Ant-Combo-v28-Sticklim-D0-Stock --seed 43 \
  --num_envs 4096 --max_iterations 600 --run_name v28s2_sticklim_e5_d0+stock_s43 --headless \
  --init_checkpoint <1단계 model_999.pt> agent.algorithm.entropy_coef=0.005

# 전체 큐: 1단계 → 평가 → 2단계 → 평가 → 확인 평가 → 집계 (완료된 작업은 건너뜀)
python scripts/run_combo_v28.py --phase stage1
python scripts/run_combo_v28.py --phase eval
python scripts/run_combo_v28.py --phase stage2 --jobs sticklim_e5_d0+stock sticklim_e5_d0+recovery \
  lim_e5_d1+stock lim_e5_d1+recovery stick_e0_d0+recovery lim_e5_d0+stock
python scripts/run_combo_v28.py --phase eval
python scripts/run_combo_v28.py --phase confirm --ids <확인 평가 체크포인트 id들>
python scripts/summarize_combo_v28.py --plots
python -m pytest -q tests/test_combo_v28.py
```

task 이름 규칙은 `Week03-Ant-Combo-v28-<Flat|Stick|Lim|Sticklim|Mine|All>-<D0|D1>-<Stock|Recovery>`입니다
(D1 = 내 랜덤화, Recovery = Stick 회복 보상). entropy는 `agent.algorithm.entropy_coef`로 바꿉니다.
학습 seed가 같아도 GPU 물리 시뮬레이션 특성상 체크포인트가 비트 단위로 같게 재현되지는 않을 수 있습니다.

## 이전 기록

이전 제출 모델은 평지에서 물성·외란을 랜덤화한 **Robust42**였습니다(제공 baseline 대비 평지 OOD 3종 평균 return +14.3%, 저마찰 +21.6%).
v28에서는 그 랜덤화를 하나의 요소로 넣어 비교했고, 처음 보는 지형 기준에서는 팀원 지형·entropy·추가 학습이 훨씬 큰 차이를 냈습니다.

| 자료 | 내용 |
|---|---|
| [v28 상세 결과](docs/COMBO_V28.md) · [사전 계획](docs/experiment_plans/combo_v28.md) | 이 README의 전체 표·조건별 결과·해석 |
| [v28 결과 원자료](artifacts/combo_v28/) | 순위·조건별 CSV, 원시 평가, 그래프, 제출·관련 run |
| [이전 README 전문](docs/PREVIOUS_README_20261001.md) | Robust42 제출 기준 결과·영상·험지 연구(v5/v16/high53)·v25 팀원 요소 포팅 |
| [전체 실험 기록 · v0–v28](docs/EXPERIMENT_HISTORY.md) | 성공·실패·부분 개선·최종 판단 |
| [최종 제출 가이드](docs/FINAL_SUBMISSION.md) | 제출 모델·평가 명령 |
| [5분 발표 PPT](report/week03_ant_robust_report.pptx) · [대본](report/SPEAKER_NOTES.md) | 기존 Robust42 3seed 연구 발표(v28 이전) |
| [제출 체크리스트](docs/SUBMISSION_CHECKLIST.md) · [팀원 정보](TEAM.md) | 과제 요구사항·LMS 항목 |
| [공개 범위·검증](docs/PUBLICATION.md) | 배포 파일·재현 제약 |

팀원 저장소: [Stick-0/isaac-ant-rough-terrain@3cc718a](https://github.com/Stick-0/isaac-ant-rough-terrain/tree/3cc718a4214f336fd4db7db5841fa86033b99d35),
[LimDaeKyung/IsaacLab_RS@8d9eed1](https://github.com/LimDaeKyung/IsaacLab_RS/tree/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b) (BSD-3-Clause).
팀원 체크포인트는 평가 기준선으로만 썼고 이 저장소에 다시 배포하지 않습니다.
무결성: `sha256sum -c artifacts/PUBLICATION_SHA256SUMS`. [BSD-3-Clause](LICENSE) · [원 라이선스 고지](THIRD_PARTY_NOTICES.md)
