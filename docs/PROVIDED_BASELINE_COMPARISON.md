# 기본 제공 코드 기준 비교 — 2026-09-30

## 기준선은 무엇인가

**제공 코드로 직접 학습한 Baseline**이지, 수업에서 제공받은 pretrained weight가 아닙니다.
`BaselineAntEnvCfg`는 제공 `AntEnvCfg`의 빈 subclass이며,
`Week03AntPPORunnerCfg`는 제공 `AntPPORunnerCfg`에서 로그용 `experiment_name`만 바꿉니다.
관측·보상·종료·물성·reset·PPO 하이퍼파라미터는 제공 설정 그대로입니다.

- 제공 task: `Isaac-Ant-v0`; 학습 alias: `Week03-Ant-Baseline-v0`.
- 제공 소스: Isaac Lab commit `f50046758743fb7bc913032d1caafc0d7e536164`.
  실제 환경/PPO/등록 파일이 해당 Git blob과 동일한지 확인했습니다.
- PPO: 32steps/env, 1,000iterations, MLP400/200/100, ELU, LR5e-4, gamma0.99, lambda0.95.
- 두 과제 모델: 학습 seed42, 4,096환경, **131,072,000 transitions/모델**,
  60D 관측·8D 행동·effort scale7.5. Robust만 학습 시 랜덤화를 추가합니다.

| 역할 | checkpoint | SHA-256 |
|---|---|---|
| 제공 코드 Baseline42 | [`baseline_seed42/model_999.pt`](../artifacts/runs/baseline_seed42/model_999.pt) | `f4a88747c8d56905c4889b0fb0708d4e924a81dc676d92561d2921e8b2bdf83b` |
| 제출 Robust42 | [`robust_seed42/model_999.pt`](../artifacts/runs/robust_seed42/model_999.pt) | `58dc882b6fa38f95ea2bd2d4d5f874836e5e043e0b5ae5c62290180122b57864` |
| 추가 학습 v5 | [`rough_v5_portal_rehearsal4_seed43/model_round_4.pt`](../artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt) | `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e` |

[기준선 학습 provenance](../artifacts/runs/baseline_seed42/manifest.json) ·
[학습 task](../src/week03_ant/tasks/ant_cfg.py) ·
[PPO alias](../src/week03_ant/tasks/agents/rsl_rl_ppo_cfg.py) ·
[실행 wrapper](../scripts/run_experiment.sh) ·
[제공 소스 해시·이번 집계](../artifacts/provided_baseline_comparison_20260930/summary.json)

## 과제 비교

같은 RTX5080에서 두 고정 checkpoint를 순서대로 재평가했습니다. 모델/seed를 결과에 따라
다시 고르거나 학습하지 않았습니다. 각 조건은 평가 seed24,100환경의 **첫 완료 episode**,
최대960steps/16초입니다. ID는 alias 대신 실제 `Isaac-Ant-v0`를 사용했습니다.
`±`는 population SD이며 표준오차·신뢰구간이 아닙니다.

| 조건 | 제공 Baseline42 | Robust42 | 평균 변화 |
|---|---:|---:|---:|
| ID | 144.89 ± 15.12 | 154.32 ± 29.32 | +6.5% |
| Low friction | 132.97 ± 19.79 | 161.69 ± 30.23 | +21.6% |
| Heavy | 133.01 ± 30.86 | 155.44 ± 27.81 | +16.9% |
| Push | 143.28 ± 21.16 | 150.68 ± 34.99 | +5.2% |

OOD는3조건의 raw 평균을 동등가중해136.421728 → 155.938734,
**+14.306376%**입니다. 조건별 증가율의 평균이 아닙니다.
외란에서 평균 episode length는942.35 → 919.23으로 줄어 모든 지표가 개선된 것은 아닙니다.

총 **800 first episodes**입니다. Baseline의 네 조건 return/length 배열은 과거 기록과
정확히 일치했지만 일부 Robust 값은 달랐습니다. 실행 간 차이의 원인은 분리 검증하지 않았습니다.
원래3seed OOD+10.8%와 기존 모델 선택 표는 당시 결과로 그대로 보존합니다.
**이 표는 선택된 단일 seed 모델의 재평가이며 새 독립 holdout·일반화·통계적 우월성 증거가 아닙니다.**

[새 CSV](../artifacts/provided_baseline_comparison_20260930/course_results.csv) ·
[16개 원시 평가 JSON](../artifacts/provided_baseline_comparison_20260930/evaluations/) ·
[보존된 원래 3seed 집계](../artifacts/evaluations/evaluation_summary.csv)

## 험지 전이 비교

기존 v5끼리의 비교 대신 **동일한 제공 코드 Baseline42**를 대조군으로 넣었습니다.
두 모델 모두 변경하지 않은 `Week03-Ant-Rough-Lanes-Eval-v5`로 평가합니다.
단, 이 task는 원래 Ant와 높이/목표/방향 관측의 의미, 보상 및 지형 종료 조건이 다릅니다.
v5는 추가 학습을 거쳤으므로 **험지 전이 성능이지 동일-budget 인과 비교나 원래 과제 점수가 아닙니다**.
제공 코드 정책은 험지에서 새로 학습하지 않았습니다.

고정 geometry/reset 조합은 **51/28,51/29,54/30,55/30**입니다.
조합당175환경 =6험지군×5난이도×5copies +평탄25개, 첫 episode만 집계합니다.
모델당 총700개 중 **험지600개**, **평탄100개**를 분리했습니다.
이 조합은 기존 연구에서 사용했던 설정이며 새로운 독립 holdout으로 부르지 않습니다.

| 험지600회/모델 | 제공 Baseline42 | v5 |
|---|---:|---:|
| Strict1tile | 114/600 (19.0%) | 516/600 (86.0%) |
| Strict6tiles | 0/600 (0.0%) | 279/600 (46.5%) |
| Terminal fall/overturn | 281/600 (46.8%) | 62/600 (10.3%) |
| Footprint lane exit | 51 | 2 |
| World exit | 0 | 0 |
| Mean terrain speed | 1.47m/s | 2.99m/s |

통과는 전진 거리 **13.1m/53.1m** 이상이며 **해당 첫 episode에서 낙상·전복 종료,
발 범위 레인 이탈, 월드 이탈이 없어야** 합니다. 거리만 충족하는 것은 성공이 아닙니다.
8m타일6개와 입구 portal/발 여유를 반영한 기준이며56m 레인 loop와 구분합니다.

| 지형군 · 각100회 | Baseline1tile | v51tile | Baseline→v5 낙상 |
|---|---:|---:|---:|
| Rough | 38 | 95 | 32 → 4 |
| Slope | 25 | 93 | 61 → 7 |
| Stairs | 21 | 94 | 36 → 6 |
| Waves | 8 | 79 | 84 → 20 |
| Obstacles | 21 | 86 | 48 → 14 |
| Stepping stones | 1 | 69 | 20 → 11 |

**남은 한계:** 징검다리6tile은0 → 3/100에 불과합니다. 위 분모 밖의 평탄 레인 낙상은
8 → 12/100으로 늘었습니다. 기존 v5→v5 연구 수치와 이번 재실행 값은 일부 다르며 혼용하지 않습니다.
초기 상태/지형 mesh의 바이트를 별도로 덤프하지 않았으므로, 검증한 것은 동일 seed/설정 및
family/level 할당입니다. 실제 로봇·임의 지형·장시간 안정성을 보장하지 않습니다.

[새 험지 CSV](../artifacts/provided_baseline_comparison_20260930/terrain_results.csv) ·
[군별·난이도별 집계](../artifacts/provided_baseline_comparison_20260930/summary.json) ·
[보존된 v5끼리의 연구](ROUGH_RECOVERY.md)

## 비교 영상

- **저마찰:** 왼쪽 제공 Baseline42, 오른쪽 제출 Robust42. 위 새100환경 평가에서
  env0을 녹화했습니다. 평가 seed24,16초,60fps. 영상 속 추가 reset 후 궤적은 집계에 더하지 않습니다.
- **징검다리:** 왼쪽 같은 Baseline42, 오른쪽 추가 학습 v5. 난이도0.8,
  generator51/reset7,1환경,16초,30fps. 각 모델을 한 번만 녹화했고 전체 시간을 유지했습니다.
  `resets`는 낙상 카운터가 아니며 마지막16초에는 자동 reset이 포함됩니다.
- 나란히640×360으로 축소하고 비교 기준/학습 여부를 표시했습니다. 프레임 컷·배속은 없습니다.
  비교 MP4는1280×452/H.264이며 두 파일 모두10MB 미만, 전체 decode를 확인했습니다.
  GIF는 전체16초를6fps로 표현한 보조 미리보기입니다.

[비교 MP4·개별 원본4개](../artifacts/provided_baseline_comparison_20260930/videos/) ·
[영상 SHA·프레임·조건](../artifacts/provided_baseline_comparison_20260930/media_manifest.json)

## 과제 비교 재현

기존 수업 환경이 설치된 저장소 루트에서 실행합니다. GPU0,seed와 checkpoint를 유지합니다.
새 출력 폴더를 사용하세요. 시뮬레이터 실행 변동 때문에 과거 수치와의 bitwise 동일성은 보장하지 않습니다.

```bash
export ROBOTICS_SIM_CLASS_ROOT=/mnt/ssd970/robotics_simulation_class
BASELINE=artifacts/runs/baseline_seed42/model_999.pt
ROBUST=artifacts/runs/robust_seed42/model_999.pt
OUT="outputs/provided_code_compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

for LABEL in baseline robust; do
  if [[ "$LABEL" == baseline ]]; then CKPT="$BASELINE"; else CKPT="$ROBUST"; fi
  for SPEC in 'id Isaac-Ant-v0' \
              'low_friction Week03-Ant-Test-LowFriction-v0' \
              'heavy Week03-Ant-Test-Heavy-v0' \
              'push Week03-Ant-Test-Push-v0'; do
    read -r SCENARIO TASK <<< "$SPEC"
    VIDEO_ARGS=()
    if [[ "$SCENARIO" == low_friction ]]; then
      VIDEO_ARGS=(--video --video_dir "$OUT/videos/${LABEL}_low_friction")
    fi
    test ! -e "$OUT/${LABEL}__${SCENARIO}.json" || exit 1
    ./scripts/run_evaluate.sh --task "$TASK" --headless --device cuda:0 \
      --num_envs 100 --seed 24 --max_steps 960 --checkpoint "$CKPT" \
      --output "$OUT/${LABEL}__${SCENARIO}.json" "${VIDEO_ARGS[@]}" \
      --kit_args=--/renderer/multiGpu/enabled=false
  done
done
```

## 험지 비교 재현

위 `BASELINE`/`OUT` 설정에 이어 실행합니다.

```bash
ROUGH=artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt
for PAIR in '51 28' '51 29' '54 30' '55 30'; do
  read -r GEOMETRY RESET <<< "$PAIR"
  for LABEL in baseline v5; do
    if [[ "$LABEL" == baseline ]]; then CKPT="$BASELINE"; else CKPT="$ROUGH"; fi
    DEST="$OUT/${LABEL}__terrain_geom${GEOMETRY}_seed${RESET}.json"
    test ! -e "$DEST" || exit 1
    ./scripts/run_evaluate.sh --task Week03-Ant-Rough-Lanes-Eval-v5 \
      --headless --device cuda:0 --num_envs 175 --seed "$RESET" --max_steps 960 \
      --checkpoint "$CKPT" --output "$DEST" \
      "env.scene.terrain.terrain_generator.seed=$GEOMETRY" \
      --kit_args=--/renderer/multiGpu/enabled=false
  done
done

# 동일 조건의16초 징검다리 원본을 각각 녹화
for LABEL in baseline v5; do
  if [[ "$LABEL" == baseline ]]; then CKPT="$BASELINE"; else CKPT="$ROUGH"; fi
  test ! -e "$OUT/videos/${LABEL}_stones08.mp4" || exit 1
  ./scripts/run_terrain_demo.sh --task Week03-Ant-Rough-Lanes-Demo-v5 \
    --headless --device cuda:0 --seed 7 --family stepping_stones --level 3 \
    --cycle-seconds 16 --checkpoint "$CKPT" --record "$OUT/videos/${LABEL}_stones08.mp4" \
    env.scene.terrain.terrain_generator.seed=51 \
    --kit_args=--/renderer/multiGpu/enabled=false
done
```

녹화 대신 GUI를 보려면 `--headless`와 `--record`를 빼고 `--duration 16 --real-time`을 사용합니다.
평가100/175환경을 임의로1환경으로 바꾼 영상 값은 성능표 평균으로 쓰지 않습니다.
실제로 사용한18개 명령은 [명령 기록](../artifacts/provided_baseline_comparison_20260930/commands.json)에 있습니다.

## 검증·공개 경계

평가16회 총 **2,200 first episodes**의 평균·SD·완료 수·성공·낙상·이탈을 원시 배열에서
재계산했습니다. 두 모델의 task/seed/geometry/family/level/판정 등15개 지형 메타데이터가
짝별로 같고 checkpoint의 실제 SHA와 기록이 일치합니다. 독립 원시 수치 감사도 통과했습니다.
기존 소스·모델·과거 원본 MP4·계획을 보존하며, 원시 console/cache/runtime/credentials는 공개하지 않습니다.
공개 JSON은 checkpoint의 로컬 절대경로만 저장소 상대경로로 바꿉니다.

1환경 징검다리 데모 두 실행은 Fabric clone 관련 오류 로그를 남겼지만 source env0이 실행되어
각480프레임을 완성하고 exit0으로 끝났습니다. 원인은 분리 검증하지 않았고 simulator 코드는 바꾸지 않았습니다.
이를 오류 없는 런타임이라고 주장하지 않으며 원시 로그는 로컬에 보존합니다.

[전체 결과](../artifacts/provided_baseline_comparison_20260930/summary.json) ·
[로컬 준비 검증](../artifacts/provided_baseline_comparison_20260930/verification.json) ·
[공개·테스트 제약](PUBLICATION.md) · [원래 모델 선택 가이드](FINAL_SUBMISSION.md)
