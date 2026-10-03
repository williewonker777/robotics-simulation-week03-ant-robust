# Robust Ant PPO · 처음 보는 지형에서도 걷는 Ant — Robotics Simulation Week 03

**제출 모델:** Stick+Lim 지형 + PPO entropy 0.005 + 600 it 이어 학습, 학습 seed 43 —
[`artifacts/combo_v28/submission/model_599.pt`](artifacts/combo_v28/submission/) (SHA-256 `3801761006031b34…`)

팀원(Stick, Lim)과 내가 찾은 요소를 **같은 학습 예산**에서 하나씩·섞어서 학습하고(새 정책 84개, 조합마다 학습 seed 3개),
과제 데모 평가 규칙(100 env, 각 env 첫 episode 누적 보상)으로 **28개 지형·마찰 조건**을 평가해 평균이 가장 높은 조합을 골랐습니다.
평가 지형·점수·선택 규칙은 학습 결과를 보기 전에 [고정](docs/experiment_plans/combo_v28.md)했습니다.

| 100 env 첫 episode 평균 return | 제공 baseline | 이전 제출 Robust42 | 팀원 Lim F3a | **제출 조합** |
|---|---:|---:|---:|---:|
| **28조건 평균** (선택용 지형 seed 2028) | 30.1 | 33.6 | 61.0 | **62.2** |
| 28조건 평균 (새 지형 seed 2029, 선택에 안 씀) | 30.1 | 33.5 | 60.8 | **62.5** |
| 박스 ±10 cm (강의 예시와 비슷한 지형) | 4.2 | 4.8 | 59.5 | **60.1** |
| 평지 (학습 때 본 환경) | 138.4 | **146.6** | 84.9 | 93.1 |

조합 값은 학습 seed 3개 평균(seed 표준편차 0.5–3.9)이고, F3a는 팀원이 고른 체크포인트 1개입니다.
험지에서는 크게 좋아졌지만 **평지에서는 평지 전용 정책보다 느리게 걷습니다**(아래 [한계](#해석과-한계)).

[![박스 ±10 cm: 제공 baseline(왼쪽)은 3.6초에 넘어지고 제출 모델(오른쪽)은 16초를 완주](artifacts/combo_v28/media/boxes_10_baseline_vs_submission.gif)](artifacts/combo_v28/media/boxes_10_baseline_vs_submission.mp4)

박스 ±10 cm — 왼쪽 제공 baseline(3.6초에 넘어짐), 오른쪽 제출 모델(16초 완주). 클릭하면 MP4.

## 평가 명령어

수업 환경(Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, RSL-RL 3.0.1)에서 저장소 루트 기준입니다.

```bash
export ROBOTICS_SIM_CLASS_ROOT=/mnt/ssd970/robotics_simulation_class   # 각자 수업 환경 경로
source "$ROBOTICS_SIM_CLASS_ROOT/activate.sh"
python -m pip install --no-deps -e .

python scripts/play_one_episode_official.py --task Week03-Ant-Combo-v28-Play \
  --num_envs 100 --seed 24 --headless \
  --checkpoint artifacts/combo_v28/submission/model_599.pt
# [RESULT] Episode reward total: mean=61.985765, std=17.888110   (박스 ±10 cm, 두 번 실행해 같은 값)
```

- [`play_one_episode_official.py`](scripts/play_one_episode_official.py)는 수업 저장소 `cailab-hy/IsaacLab_RS@e83a5d2`의
  `play_one_episode.py`에 **v28 task 등록 import 한 줄만** 더한 사본입니다.
- `Week03-Ant-Combo-v28-Play`는 원래 `Isaac-Ant-v0`과 보상 7항·종료·행동·60D 관측 구성이 같고, **높이 관측·넘어짐 판정만
  발밑 지면 기준**(몸통에서 아래로 쏜 ray 1개)입니다. 평지에서는 원래와 값이 같습니다. 기본 지형은 무작위 박스 ±10 cm입니다.
- **다른 지형으로 평가하려면** [`ComboV28PlayEnvCfg`](src/week03_ant/tasks/combo_v28_cfg.py)의 `scene.terrain`(prim 경로 `/World/ground`)만 바꾸면 됩니다.
- 이 PC는 GPU가 2개라 `--device cuda:1 --kit_args="--/renderer/multiGpu/enabled=false"`를 붙였습니다.

## 무엇을 바꿨나

모든 정책은 과제 인터페이스를 그대로 씁니다: 60D 관측, 8D 관절 토크 행동, MLP `[400, 200, 100]`, 원래 PPO 설정,
**4,096 env × 32 step × 1,000 iteration**. 평가 보상은 원래 7개 항이며, 학습에만 쓴 보상은 점수에 들어가지 않습니다.
공통으로 높이 관측과 넘어짐 판정을 발밑 지면 기준으로 바꿨습니다(두 팀원 모두 험지에서 필요했다고 보고).

| 요소 (가설: 처음 보는 지형 점수를 올린다) | 비교한 수준 | 출처 |
|---|---|---|
| 학습 지형 | 평지 / Stick 험지(요철·물결·완경사) / Lim 박스 ±10 cm / Stick+Lim / 내 v5 지형군 / 전체 혼합 | Stick, Lim, 내 v5 |
| PPO entropy | 0(원래) / 0.005 | Lim |
| 물성·외란·관측 랜덤화 | 끔 / 켬 | 내 Robust42 |
| 이어 학습 +600 it | 원래 보상 / Stick 회복 보상(넘어짐·기울기 벌점) | Lim F3a / Stick |

**제출 조합 = 공통 변경 + Stick+Lim 학습 지형 + entropy 0.005 + 원래 보상으로 600 it 이어 학습**(총 1,600 it, 랜덤화 없음).
요소별 세부 설정은 [결과 문서](docs/COMBO_V28.md#1-무엇을-비교했나), 코드는 [`combo_v28_cfg.py`](src/week03_ant/tasks/combo_v28_cfg.py)에 있습니다.

## 실험 설계

- **1단계 (같은 예산):** 학습 지형 6 × entropy 2 × 랜덤화 2 = 24조합 × 학습 seed 42·43·44.
  평지·entropy 0의 두 칸은 같은 예산·seed로 이미 학습한 제공 baseline과 Robust42입니다.
- **2단계 (+600 it):** 1단계 상위 2조합 × {원래 보상, 회복 보상}, Stick 단독, Lim 단독 — 각 seed 3개. 학습량이 1.6배라 1단계와는 따로 비교합니다.
- **평가:** 과제 규칙(100 env, 평가 seed 24, 첫 episode 누적 보상, 최대 16초)에서 **지형만 28조건**으로 바꿉니다
  (평지·저마찰 4, 박스 6, 요철·물결 4, 경사·계단 5, 장애물 6, 틈·구덩이 3). **데모 점수 = 28조건 평균 return.**
- **선택 규칙:** 3-seed 평균이 가장 높은 조합 → 그중 선택용 지형 점수가 가장 높은 seed를 제출. 이후 새 지형 seed 2029로 확인 평가.
- **검증:** 공식 스크립트와 v28 평가기 값이 소수점 6자리까지 같고, 모든 체크포인트의 유한값·SHA-256(일반/O_DIRECT 읽기)을 확인했습니다.

## 결과

![조합별 데모 점수 — 선택용 지형](artifacts/combo_v28/plots/ranking_selection.png)

| 순위 | 조합 (seed 3개) | 데모 점수 | 확인 평가 (새 지형) | 넘어짐률 |
|---:|---|---:|---:|---:|
| 1 | **Stick+Lim 지형 + entropy + 600 it** | **62.2 ± 0.7** | **62.5 ± 0.5** | 22% |
| 2 | Lim 박스 + entropy + 랜덤화 + 600 it | 61.2 ± 1.5 | 61.4 ± 1.5 | 15% |
| – | Lim F3a (팀원 원본, 1,600 it) | 61.0 | 60.8 | 18% |
| 3 | Lim 박스 + entropy + 600 it | 60.0 ± 1.8 | 60.5 ± 2.0 | 22% |
| 4 | Lim 박스 + entropy + 랜덤화 + 600 it 회복 보상 | 58.9 ± 2.1 | 59.1 ± 2.2 | 8% |
| 5 | Stick+Lim 지형 + entropy (1,000 it) | 58.7 ± 0.5 | 58.9 ± 0.6 | 25% |
| – | 내 Robust42 / 제공 baseline (평지) | 33.6 / 30.1 | 33.5 / 30.1 | 39% / 48% |

**요소별 효과** (1단계에서 다른 요소·seed는 같고 한 요소만 바꾼 짝 비교)

| 요소 | 데모 점수 변화 | 좋아진 조합 |
|---|---:|---:|
| entropy 0 → 0.005 | **+8.1** | 12/12 |
| 학습 지형 평지 → Stick·Lim·Stick+Lim (세 지형끼리는 차이 없음) | **+16** | 12/12 |
| 원래 보상으로 600 it 이어 학습 | **+3.5 ~ +3.9** | 3/3 |
| 랜덤화 끔 → 켬 (평지·저마찰에서만 +4.5) | +1.2 | 7/12 |
| 회복 보상으로 이어 학습 (넘어짐은 크게 줄어듦, 예: 15% → 8%) | 원래 보상보다 −2.3 ~ −5.7 | 0/2 |

- **확인 평가:** 선택에 쓰지 않은 새 지형(seed 2029)에서도 순위가 같았습니다.
- **공식 방식 점검:** 공식 스크립트처럼 체크포인트마다 새 프로세스로 다시 평가해도 순위는 같습니다(1위 62.1, 2위 61.2, F3a 60.9).
  한 프로세스에서 여러 체크포인트를 평가하면 피라미드 계단 두 조건에서만 값이 조금 달라집니다([점검 결과](docs/COMBO_V28.md#5-공식-방식-점검--체크포인트마다-새-프로세스)).
- 범주·조건별 표, 이어 학습·학습 곡선·조건별 그래프, 원시 결과는 **[v28 결과 문서](docs/COMBO_V28.md)**에 있습니다.

## 영상

[![제출 모델 — 계단·장애물·물결·경사·징검다리·저마찰](artifacts/combo_v28/media/submission_six_terrains.gif)](artifacts/combo_v28/media/submission_six_terrains.mp4)

제출 모델 — 계단 10 cm · 장애물 10 cm · 물결 15 cm · 경사 0.2 · 징검다리 · 저마찰 μ0.1 (16 env 배치의 env 0, 16초 무편집).
다섯 지형은 완주했고 **경사 0.2는 1.9초에 넘어졌습니다**(이 조건의 100 env 넘어짐률 32%). [MP4·영상별 조건](artifacts/combo_v28/media/)

## 해석과 한계

- 최종 조합은 **두 팀원의 요소를 합친 것**입니다. 내 랜덤화는 저마찰에서만 효과가 있어 최종 조합에서 빠졌습니다.
- 1위와 Lim F3a의 차이(1.2)는 seed 표준편차(0.7)의 두 배 정도이고, 누구의 학습 지형도 아닌 25조건만 보면 0.5입니다.
  "확실히 더 좋다"보다 "같은 수준이거나 조금 높다"가 정확합니다.
- **평지에서는 느립니다:** 제출 체크포인트는 16초에 94 m(return 92.8), 제공 baseline은 145 m(140.1)입니다.
  매우 미끄러운 지면(유효 마찰 0.2)에서도 Robust42·F3a보다 낮습니다.
- 평가 지형 28조건은 우리가 고른 것이고 조교의 실제 평가 지형은 공개되지 않았습니다.
- 제출 정책은 발밑 지면 높이를 관측하므로 `Week03-Ant-Combo-v28-Play` task로 평가해야 합니다(월드 z를 넣으면 험지에서 관측 의미가 달라짐).
- 모든 결과는 시뮬레이션이며 학습 seed는 조합당 3개입니다.

## 재현

```bash
# 제출 조합 학습: 1단계 1,000 it → 같은 seed로 +600 it 이어 학습 (가중치만 불러오고 optimizer는 새로)
python scripts/train_combo_v28.py --task Week03-Ant-Combo-v28-Sticklim-D0-Stock --seed 43 \
  --num_envs 4096 --max_iterations 1000 --run_name v28s1_sticklim_e5_d0_s43 --headless \
  agent.algorithm.entropy_coef=0.005
python scripts/train_combo_v28.py --task Week03-Ant-Combo-v28-Sticklim-D0-Stock --seed 43 \
  --num_envs 4096 --max_iterations 600 --run_name v28s2_sticklim_e5_d0+stock_s43 --headless \
  --init_checkpoint <1단계 model_999.pt> agent.algorithm.entropy_coef=0.005

# 28조건 중 하나를 평가기로: 체크포인트 목록(JSON)과 지형 이름
echo '[{"id": "submission", "path": "artifacts/combo_v28/submission/model_599.pt"}]' > my_checkpoints.json
python scripts/evaluate_demo_v28.py --terrain stairs_10 --terrain_seed 2028 \
  --checkpoints my_checkpoints.json --output_dir outputs/my_eval --headless

python -m pytest -q tests/test_combo_v28.py
```

전체 실험 큐(1·2단계 학습, 평가, 확인 평가, 집계)는 [결과 문서의 재현 절](docs/COMBO_V28.md#재현)에 있습니다.
같은 seed라도 GPU 물리 시뮬레이션 특성상 학습 결과가 비트 단위로 같게 재현되지는 않을 수 있습니다.

## 자료

| 자료 | 내용 |
|---|---|
| [v28 결과 문서](docs/COMBO_V28.md) · [사전 계획](docs/experiment_plans/combo_v28.md) | 전체 순위·범주/조건별 결과·이어 학습·공식 방식 점검·해석 |
| [v28 원자료](artifacts/combo_v28/) | 순위·조건별 CSV, 원시 평가(환경별 return), 그래프, 제출 조합 run 6개, 영상 |
| [이전 README 전문](docs/PREVIOUS_README_20261001.md) | 이전 제출 Robust42 결과·영상, 험지 연구(v5/v16/high53), v25 팀원 요소 포팅 |
| [전체 실험 기록 · v0–v28](docs/EXPERIMENT_HISTORY.md) | 성공·실패·부분 개선·최종 판단 |
| [최종 제출 가이드](docs/FINAL_SUBMISSION.md) · [제출 체크리스트](docs/SUBMISSION_CHECKLIST.md) · [팀원 정보](TEAM.md) | 제출 모델·평가 명령·LMS 항목 |
| [5분 발표 PPT](report/week03_ant_robust_report.pptx) · [대본](report/SPEAKER_NOTES.md) | 이전 Robust42 연구 발표(v28 이전) |
| [공개 범위·검증](docs/PUBLICATION.md) | 배포 파일·재현 제약 |

팀원 저장소: [Stick-0/isaac-ant-rough-terrain@3cc718a](https://github.com/Stick-0/isaac-ant-rough-terrain/tree/3cc718a4214f336fd4db7db5841fa86033b99d35) ·
[LimDaeKyung/IsaacLab_RS@8d9eed1](https://github.com/LimDaeKyung/IsaacLab_RS/tree/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b) (BSD-3-Clause).
팀원 체크포인트는 평가 기준선으로만 썼고 이 저장소에 다시 배포하지 않습니다.
무결성: `sha256sum -c artifacts/PUBLICATION_SHA256SUMS` · [BSD-3-Clause](LICENSE) · [원 라이선스 고지](THIRD_PARTY_NOTICES.md)
