# Week 03 최종 제출·실행 안내 — 2026-09-30

> **2026-09-30 비교 갱신:** 아래 선택 표는 원래 실험 당시 결과입니다.
> 기본 제공 코드 Baseline42와의 **새 짝 평가·비교 영상**은
> [이번 비교 가이드](PROVIDED_BASELINE_COMPARISON.md)와 [README](../README.md)에 있습니다.
> 새 단일-seed 재평가와 과거3seed/모델 선택 수치를 섞지 않습니다.

## 1. 제출 모델과 선택 근거

**원래 Ant 과제에는 `artifacts/runs/robust_seed42/model_999.pt`를 사용합니다.**
관측 60D, 행동 8D, MLP `[400, 200, 100]`, joint-effort action scale 7.5를
유지한 PPO 모델입니다. 체크포인트의 actor 입력/출력도 실제로 60/8임을 확인했습니다.

- 학습: `Week03-Ant-Robust-v0`, seed **42**, 4,096 envs × 32 steps × 1,000 iterations.
- 평가: seed **24**, **100 envs 각각의 첫 episode**, 최대 960 steps(16초).
- SHA-256: `58dc882b6fa38f95ea2bd2d4d5f874836e5e043e0b5ae5c62290180122b57864`.
- [모델·설정·manifest](../artifacts/runs/robust_seed42/),
  [모든 원래 과제 run의 평가표](../artifacts/evaluations/evaluation_per_run.csv).

아래는 **기존 공개 평가의 관측값**입니다. `seed`는 학습 seed이며 평가 seed는 모두 24입니다.
OOD 평균은 low-friction/heavy/push의 평균 return을 같은 가중치로 평균한 선택용 요약입니다.

| 모델 | 학습 seed | ID | 저마찰 | 추가 하중 | 외란 | OOD 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 42 | 144.89 | 132.97 | 133.01 | 143.28 | 136.42 |
| Baseline | 43 | 133.20 | 123.77 | 120.19 | 132.26 | 125.40 |
| Baseline | 44 | 140.05 | 131.37 | 130.49 | 143.05 | 134.97 |
| Friction-only | 42 | 153.20 | 153.19 | 142.46 | **154.68** | 150.11 |
| **Robust — 제출 선택** | **42** | **156.15** | **162.01** | **154.30** | 153.04 | **156.45** |
| Robust | 43 | 143.81 | 157.10 | 143.23 | 139.90 | 146.74 |
| Robust | 44 | 133.11 | 139.43 | 132.05 | 137.79 | 136.42 |

선택 모델의 return **평균 ± population 표준편차**:

| 평가 환경 | Return | 평균 episode length |
|---|---:|---:|
| ID | 156.15 ± 27.93 | 933.51 / 960 |
| Low friction | 162.01 ± 30.79 | 928.14 / 960 |
| Heavy | 154.30 ± 28.06 | 931.81 / 960 |
| Push | 153.04 ± 32.02 | 923.94 / 960 |

Robust seed42는 이 7개 run 중 ID와 세 OOD의 동등가중 종합 return이 가장 높습니다.
**모든 조건에서 최고인 것은 아닙니다**(push는 friction-only가 높음).
이미 본 공개 평가에 근거한 단일 모델 선택이지, 비공개 채점 환경에서의 우승이나
통계적 우월성 보장이 아닙니다. 연구 결론은 최고 seed만 고르지 않고 **Baseline/Robust
각 3seed 전체**를 사용합니다: 공개 OOD 평균 개선 +10.8%, 조건별/seed별 trade-off 존재.
[사전 계획](EXPERIMENT_PLAN.md), [3seed 집계](../artifacts/evaluations/evaluation_summary.csv).

## 2. 실행 환경

기존 수업 환경이 설치되어 있어야 합니다. 이 저장소가 Isaac Sim/Isaac Lab을 배포하지는 않습니다.

| 항목 | 검증된 버전 |
|---|---|
| Python | 3.11.16 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0, commit `f50046758743fb7bc913032d1caafc0d7e536164` |
| PyTorch | 2.7.0+cu128 |
| RSL-RL | 3.0.1 |

이 PC에서 저장소 루트로 이동한 뒤 실행합니다. 다른 PC에서는 `COURSE_ROOT`와 저장소 경로를 바꾸세요.

```bash
COURSE_ROOT=/mnt/ssd970/robotics_simulation_class
cd "$COURSE_ROOT/robotics-simulation-week03-ant-robust"
source "$COURSE_ROOT/activate.sh"
export ROBOTICS_SIM_CLASS_ROOT="$COURSE_ROOT"
python -m pip install --no-deps -e .
CHECKPOINT=artifacts/runs/robust_seed42/model_999.pt
```

```bash
printf '%s  %s\n' \
  58dc882b6fa38f95ea2bd2d4d5f874836e5e043e0b5ae5c62290180122b57864 \
  "$CHECKPOINT" | sha256sum -c -
git rev-parse HEAD  # 제출한 코드의 정확한 commit 기록
```

## 3. 과제용 100환경 평가 — 먼저 실행할 명령

```bash
./scripts/run_evaluate.sh \
  --task Week03-Ant-Baseline-v0 \
  --headless --device cuda:0 \
  --num_envs 100 --seed 24 --max_steps 960 \
  --checkpoint "$CHECKPOINT" \
  --output outputs/submission_eval/id.json \
  --kit_args=--/renderer/multiGpu/enabled=false
```

`Week03-Ant-Baseline-v0`는 **평가 지면/물성**을 선택할 뿐, 로드한 Robust 모델을
Baseline 모델로 바꾸는 옵션이 아닙니다. 누적보상은 `scripts/play_one_episode.py`의
`FirstEpisodeAccumulator`가 환경별 **첫 완료 episode만** 집계합니다.
출력 JSON에서 `completed_episodes=100`, `policy_observation_dimensions=60`,
`episode_return_mean`, `episode_return_std`, `checkpoint_sha256`를 확인하세요.
재실행의 PhysX/GPU 수치가 과거 평가와 비트 단위로 같다고 보장하지 않습니다.

ID와 공개 OOD 3종을 모두 재평가할 때는 아래 명령을 사용합니다.
새 label을 사용하므로 기존 `robust_seed42__*.json`을 덮어쓰지 않습니다.

```bash
./scripts/evaluate_checkpoint.sh submission_recheck "$CHECKPOINT" 0
```

결과: `artifacts/evaluations/submission_recheck__{id,low_friction,heavy,push}.json`.
console은 Git-ignored `artifacts/console/`에 남습니다. 같은 recheck label의 이전 결과는
다시 실행하면 교체되므로 별도로 보관할 결과에는 다른 label을 사용하세요.

발표 당일 비공개 평가 task가 제공되면 위 단일 평가 명령의 `--task`만 그 task ID로
바꾸되, **동일 60D 관측·8D 행동/순서·scale** 계약을 확인하세요. 현재 비공개 task의
이름이나 성능은 알 수 없으며, 공개 holdout 재학습으로 비공개 결과를 추정하지 않습니다.

## 4. GUI 재생 / 영상

GUI는 `--headless`를 빼고 `--real-time`으로 16초 주행을 관찰합니다.
**1환경 데모 return을 100환경 채점 결과로 대신하지 않습니다.**

```bash
./scripts/run_evaluate.sh \
  --task Week03-Ant-Baseline-v0 \
  --device cuda:0 --num_envs 1 --seed 24 --max_steps 960 \
  --checkpoint "$CHECKPOINT" --real-time \
  --kit_args=--/renderer/multiGpu/enabled=false
```

영상 저장은 같은 명령에 `--video --video_dir outputs/submission_video`를 추가합니다.
기존 [Baseline/Robust 저마찰 비교 영상](../artifacts/videos/comparison_low_friction_seed42.mp4),
[5분 발표자료](../report/week03_ant_robust_report.pptx), [발표 대본](../report/SPEAKER_NOTES.md)을
그대로 사용할 수 있습니다. 이 PPT의 결론은 **원래 과제 3seed 실험**에 관한 것이며
v11–v24 전체 연구를 설명하는 새 발표자료는 아닙니다.

## 5. 재학습 — 제출 모델 실행에 필수 아님

같은 수업 PPO/예산으로 seed42를 재현하는 예입니다. 원본 학습/평가 파일을 보존하도록
새 run name을 사용합니다. GPU가 한 장이면 `cuda:0`을 사용하세요.

```bash
./scripts/run_train.sh \
  --task Week03-Ant-Robust-v0 \
  --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 1000 --seed 42 \
  --run_name robust_reproduction_seed42
```

공정 비교에는 Baseline/Robust 모두 seed42/43/44와 동일 budget을 사용합니다.
Friction-only seed42는 단일-seed ablation입니다. 원래 명령과 물성 범위는
[README 학습 재현](../README.md#학습-재현), [experiment matrix](../configs/experiment_matrix.yaml)에 있습니다.
후속 험지 학습의 추가 budget을 원래 1,000-iteration 비교에 합치지 않습니다.

## 6. 험지 실험 마무리 — 과제 제출과 분리

| 실험 | 핵심 결과 / 최종 판단 |
|---|---|
| v0 원래 과제 | 3seed 동일-budget 물성 랜덤화 비교; 공개 OOD 평균 +10.8% |
| v1–v4 지형 확장 | 월드 끝 추락·느려짐·scanner 입력 비호환을 확인 |
| v5 + 복구/rehearsal | 60D/8D 레인 기준선; **험지 기본 데모 모델 유지** |
| v6–v10 센서/발 위치/잔차/착지/prior | 일부 돌다리 개선, 전체 승격 gate 실패 |
| v11–v12 전환/history | 일부 장시간 이득, 낙상/전체 성공 trade-off로 기본 교체 보류 |
| v13–v18 보상·관측·접촉·커리큘럼·스타일 | 부분 이득과 회귀 동시 존재; 기본 교체 없음 |
| v19 공통 지도 재평가 | v16 control의 16초 합산 이득, 지도별/64초 회귀로 기본 유지 |
| v20–v22 접촉 비용·추가학습·seed | 주요 16초 합산 기준 실패; 기본 유지 |
| v23–v24 시간·학습률 진단 | 같은 episode의 16/64초는 의존 관측; LR 저하도 일관 승격 실패 |

기록은 [전체 연대기](EXPERIMENT_HISTORY.md)와 버전별 원시 결과/독립 감사에 보존했습니다.
실패한 실험도 삭제하거나 성공 사례처럼 재해석하지 않습니다. 최고 성능은 **평가 계약별**로
판단해야 하며, 지형 통과율과 원래 과제 return은 서로 대체할 수 없습니다.

험지 기본 데모만 볼 때는 아래 v5 모델을 사용합니다.
이 모델의 과제 ID return은 별도 평가에서 137.07 ± 26.83으로,
원래 과제 제출용 Robust seed42보다 낮아 **제출 모델로 선택하지 않았습니다**.

```bash
./scripts/run_terrain_demo.sh \
  --task Week03-Ant-Rough-Lanes-Demo-v5 \
  --device cuda:0 --seed 7 --cycle-seconds 8 \
  --checkpoint artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

v5 SHA-256: `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e`.
v3/v4 및 v6 이후 확장 관측 모델·hybrid loader는 원래 60D 평가기에 그대로 넣지 않습니다.
깊이는 이상적인 ray/height scan이며 실물 RGB-D나 로봇 강건성을 검증한 결과가 아닙니다.

## 7. 제출 요구사항 대응 / 마지막 확인

| 확인 항목 | 저장소 위치 |
|---|---|
| 소스·실행 명령·환경/라이선스 | `src/`, `scripts/`, 본 문서, `LICENSE`, `THIRD_PARTY_NOTICES.md` |
| 가설·동일 PPO/budget/seed·랜덤화 범위 | `docs/EXPERIMENT_PLAN.md`, `configs/experiment_matrix.yaml` |
| 선택 모델·학습 설정·해시/출처 | `artifacts/runs/robust_seed42/` |
| 100환경 누적보상 평균/표준편차 | `artifacts/evaluations/` JSON·CSV |
| 학습/평가 그래프·주행 영상 | `artifacts/plots/`, `artifacts/videos/` |
| 5분 발표자료·대본·공개 URL | `report/`, README |
| 후속 실험·실패·한계·게시 경계 | `docs/EXPERIMENT_HISTORY.md`, `docs/PUBLICATION.md` |

```bash
"$ROBOTICS_SIM_CLASS_ROOT/run-python" -m pytest -q \
  tests/test_evaluation.py tests/test_command_math.py \
  tests/test_contact_math.py tests/test_direction_math.py \
  tests/test_posture_math.py tests/test_history_gate.py tests/test_hybrid_gate.py
sha256sum -c artifacts/PUBLICATION_SHA256SUMS
```

위 명령은 공개 checkout에서 확인한 **선택된 portable unit tests 147개**입니다.
전체 2,056개 suite에는 과거 PLAN 원래 경로·개발 JSON·학습 설정·초기 checkpoint·
원시 log/event에 의존하는 통합 사례도 있어 **원문 자료가 보존된 로컬 환경**이 필요합니다.
그 환경에서 `pytest -q` 전체 2,056개가 통과했습니다. 공개본에서 전체 suite를 바로 실행하면
누락된 로컬 증거 때문에 실패할 수 있습니다. 공개본에 없는 증거를 만들어 내거나 실패를
무시하는 fallback은 사용하지 않습니다.

원시 TensorBoard/console/cache와 실행 환경은 **로컬 보관**이며 공개 산출물에 새로
추가하지 않습니다. manifest의 원래 log/event 경로는 공개 파일 존재 보장이 아닙니다.
[최종 체크리스트](SUBMISSION_CHECKLIST.md)에서 **팀원 이름/학번/역할 입력 및 LMS에
저장소 URL 제출**은 아직 사용자가 해야 할 항목입니다. 저장소 push가 LMS 제출을 대신하지 않습니다.
