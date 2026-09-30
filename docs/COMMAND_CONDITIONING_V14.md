# v14 — 논문 기반 자세 명령·피드백 관측

## 방법과 논문에서 가져온 부분

목표는 장애물이 있으면 몸체를 높이고, 평지에서는 낮은 자세로 빠르게 이동하는
학습을 돕는 것이다. [Walk These Ways (Margolis & Agrawal)](https://proceedings.mlr.press/v205/margolis23a.html)는
몸체·발 높이 등 행동 명령을 보상뿐 아니라 정책 입력으로 전달한다.
여기서는 **명시적 자세 명령 입력**이라는 아이디어만 적용했다.

원 논문의 gait/위상, 과거 관측, 무작위 행동 명령, 제어 출력, 보상 조합,
실제 로봇 전이 전체를 재현하지 않았다. 깊이를 자동으로 목표 높이로 변환하는
규칙은 본 프로젝트의 적용이며, 논문에서 검증한 자동 전환 기능이 아니다.
[조사한 대안과 근거](../artifacts/terrain_demo/command_conditioning_v14/research.md).

## 같은 보상, 다른 입력 접근

기존 88D 관측의 순서와 값을 보존하고 아래 3개를 붙여 **91D**로 만들었다.

1. 지형으로 계산한 목표 몸체 높이: `(목표 − 0.44) / 0.14`.
2. 지면 대비 실제 몸체 높이: `(측정 − 0.44) / 0.14`, 범위 −8~8.
3. 지형·자세·운동 정보의 유효성. 정보가 불확실하면 세 값 모두 0이다.
   유효한 평지는 유효성 1이므로, 모르는 지형을 평지로 표시하지 않는다.

- **masked**: actor와 critic 모두 마지막 3개를 가린다.
- **conditioned**: actor와 critic 모두 마지막 3개를 받는다.
- 양쪽 모두 [v13 자세 보상](ADAPTIVE_POSTURE_V13.md)의 가중치 1을 그대로 쓴다.
  따라서 v13의 보상 유무 비교와 달리 **명령+높이 피드백에 대한 공동 접근**을
  시험한다. actor 단독 또는 목표 값 하나만의 인과 효과는 분리하지 않았다.
- 입력 첫 층을 `W·관측88 + b + U·추가3`으로 계산하고 `U=0`으로 시작한다.
  기존 학습된 깊이 관련 가중치를 버리지 않으며 초기 actor/critic/teacher
  출력은 원래 모델과 CPU·GPU에서 정확히 일치한다.

## 실험 조건

두 모델 모두 원래 v10 anchored seed42의 마지막749 checkpoint에서 출발한다.
학습된 actor/critic와 60D 고정 v5 교사는 보존하고, 탐색 표준편차 0.2와
빈 Adam(학습률 0.0001)으로 시작한다. 교사 prior 0.02는 유지한다.

- 각 모델: **4,096환경 × 32step × 250iteration = 32,768,000 transition**.
- 학습 seed46, 지형85. 마지막249 checkpoint만 평가한다.
- 새 지형86/reset58, 87/reset59에서 같은 초기상태와 관측으로 비교한다.
- 제어기: v13, masked, conditioned, 원래 history hybrid, history+masked,
  history+conditioned. 기존 v12 깊이 히스토리 전환기는 변경하지 않는다.
- 주평가: 혼합 지형 16초, 12파일·2,100 첫 episode.
- 별도평가: 최고난도 돌다리 64초, 12파일·120 첫 episode.
- 성공 거리 13.1m/53.1m, 낙상·레인 이탈·월드 이탈 판정은 동일하다.

같은 학습량의 masked가 처치 비교의 기준이다. 예전 v13와 원래 hybrid는
참고 기준선이며, 다른 지도에서 얻은 예전 표와 직접 순위를 매기지 않는다.
주평가와 64초 결과는 섞지 않으며, 별도평가의 이득이 주평가 실패를 상쇄하지 않는다.

## 검증과 해석의 경계

- 초기 전체91D·88D prefix·root/joints·모델 tensor·CPU/CUDA RNG의 짝을 검증한다.
- 모든 학습 iteration/transition 및 TensorBoard scalar의 완전성·유한성을 검증한다.
- 훈련·평가 코드와 모델을 고정하고, 새 지도 캐시를 먼저 생성한 뒤 모든 평가에
  동일한 캐시 해시를 연결한다. 오류 파일과 개발 기록은 덮어쓰지 않는다.
- 세 hybrid의 평지 결과는 환경별로 원래 v5 분기와 같아야 한다. 평지 동작의
  동일성은 새 학습 방법의 성과로 계산하지 않는다.
- 개발 지도에서 원래88D 모델의 초기상태·관측과14개 결과/보상/전환 필드는 정확히
  재현했다. 단, 관측 시 추가 지형 계산으로 **수동 속도 telemetry의 일부 합계는
  예전88D 평가와 다르다**. 파생 속도의 reset 시점 lazy cache 읽기 순서와 부합한다.
  [차이와 원본 해시](../artifacts/terrain_demo/command_conditioning_v14/evaluator_parity.json)를
  보존했다. 두91D 학습군은 같은 읽기 순서를 쓰고, 성능 속도는 실제 거리/시간이다.
- 학습 후 목표0/1 입력을 바꾸는 정적 실험은 행동의 입력 민감도만 측정한다.
  환경을 움직이지 않으므로 실제 몸체 높이 추종이나 험지 통과의 증거가 아니다.
- 이상적인 ray 높이 정보이며 실제 RGB-D 카메라 영상이 아니다. 단일 학습 seed와
  두 지도 결과로 모든 맵 우월성·통계적 유의성·실제 로봇 안전성을 주장하지 않는다.

## 결과 — 2026-09-23 검증 완료

**두 모델 학습 총65,536,000 transition, 보류 평가24파일·2,220회 완료.**
동일 예산 masked 대비 단독 모델은 일부 이득이 있지만, **주평가의 전체 개선 기준은
단독·하이브리드 모두 FAIL**이다. 기존 기본 모델/실행 설정은 변경하지 않았다.

### 16초 혼합지형: 제어기별 험지300회 + 평지50회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v13 | 267/300 | 143/300 | 30/300 | 2/300 |
| masked | 251/300 | 144/300 | 44/300 | 1/300 |
| conditioned | 261/300 | 135/300 | 30/300 | 8/300 |
| history_original | 272/300 | 146/300 | 25/300 | 2/300 |
| history_masked | 263/300 | 145/300 | 31/300 | 3/300 |
| history_conditioned | 273/300 | 135/300 | 24/300 | 3/300 |

### 별도64초 최고난도 돌다리: 제어기별20회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v13 | 9/20 | 6/20 | 9/20 | 2/20 |
| masked | 9/20 | 5/20 | 9/20 | 2/20 |
| conditioned | 12/20 | 9/20 | 6/20 | 2/20 |
| history_original | 8/20 | 6/20 | 9/20 | 4/20 |
| history_masked | 9/20 | 5/20 | 6/20 | 7/20 |
| history_conditioned | 7/20 | 2/20 | 10/20 | 3/20 |

- 16초 단독: 낙상44→30회, 1타일251→261회는 좋아졌으나, 6타일144→135회와
  레인 이탈1→8회가 악화했다. 평지 속도11.1450→11.2795m/s(+1.21%), 낙상1/50은 동일.
- 16초 hybrid: 6타일145→135회로 감소해 FAIL. 세 hybrid의 평지50회 원시 결과는
  모두 정확히 같으며, 이 평지 동작은 고정 v5 분기의 성질이지 새 학습 효과가 아니다.
- 64초 단독: 6타일5→9/20, 낙상9→6/20, 이탈2→2/20으로 **별도 기준 PASS**.
  그러나 hybrid는6타일5→2/20, 낙상6→10/20으로 FAIL. 단독 이득을 전환 제어기에
  그대로 적용할 수 있다고 결론 내리지 않는다. 모든 평가의 world 이탈은0이다.

### 자세 변화와 남은 한계

| 단독 모델의 방문 상태 통계 | masked | conditioned |
|---|---:|---:|
| 16초 험지 감지 구간 몸체 높이 | 47.97cm | 48.92cm |
| 16초 험지 감지 구간 목표 높이 오차 | 11.25cm | 10.32cm |
| 평지 가족 몸체 높이 | 45.43cm | 45.54cm |
| 64초 험지 감지 구간 몸체 높이 | 43.18cm | 44.60cm |
| 64초 목표 높이 오차 | 15.73cm | 14.49cm |
| 64초 발 끝 지면 여유 | 10.18cm | 12.71cm |
| 64초 험지 유효 구간 저속(<1m/s) 비율 | 64.86% | 56.23% |

험지에서 더 높은 몸체와 줄어든 목표 오차가 관찰되지만, 여전히 목표58cm와의
차이가 크다. **평지에서 더 낮아지는 효과는 관찰되지 않았다.** 위 평균은 각각 방문한
유효 상태의 통계이며 같은 궤적/상태에서의 인과 비교가 아니다. 깊이 유효 샘플 비율은
단독16초99.95% 이상, 운동 정보 유효 비율100%였다. 지연·가림을 가진 실제 센서 검증은 아니다.

정적 목표 입력0→1 시험은35개 유효 초기 상태에서 conditioned의 행동 평균 절대차
0.04737, 최대차0.19169를 보였다. masked와 원래 모델의 차이는 정확히0이었다.
관측/RNG를 변경하거나 환경을 step하지 않았다. **입력 민감도이지 몸체 명령 추종 실험은 아니다.**

### 검증 산출물

- [전체 원시 결과·지형/난도별 통계](../artifacts/terrain_demo/command_conditioning_v14/summary.json)
- [전체 표와 사전 판정](../artifacts/terrain_demo/command_conditioning_v14/summary.md)
- [학습 로그 검증](../artifacts/terrain_demo/command_conditioning_v14/training_validation.json)
- [독립 학습·추론 검토](../artifacts/terrain_demo/command_conditioning_v14/training_independent_review.json)
- [독립 최종 감사](../artifacts/terrain_demo/command_conditioning_v14/final_review.md)

전체631 CPU 회귀 테스트, 컴파일 검사, diff 검사 통과. Ruff/Mypy/Pyright는 설치되어
있지 않아 실행했다고 주장하지 않는다. 새 의존성·원격 Git 작업·기존 frozen 파일 수정은 없다.

## 재현 명령

프로젝트 루트에서 기존 환경 실행기 `../run-python`을 사용한다. 아래는 실제 수행한 단계이며, 고정된 증거 경로가 비어 있는 격리 실행에만 적용한다.
이미 산출물이 있는 현재 checkout에서는 덮어쓰기 방지로 중단된다. 현재 증거를 지우거나 덮어쓰지 않는다.

```bash
../run-python -m pytest -q
../run-python scripts/run_command_study.py development
../run-python scripts/run_command_study.py capacity
# 검토/테스트가 통과한 preflight.json을 확인한 뒤 연구 입력 고정
../run-python scripts/run_command_study.py freeze_train
../run-python scripts/run_command_study.py train
../run-python scripts/run_command_study.py final_smoke
../run-python scripts/run_command_study.py freeze_eval
../run-python scripts/run_command_study.py probe
../run-python scripts/run_command_study.py prepare_eval
../run-python scripts/run_command_study.py evaluate
../run-python scripts/run_command_study.py horizon
../run-python scripts/run_command_study.py report
```

각 단계는 GPU1에서 실행하고 공유 GPU lock으로 무거운 작업의 중복 실행을 막는다.
기존 실행의 숫자만 다시 검산하려면 새 출력 이름으로 다음 명령을 사용한다.

```bash
../run-python scripts/summarize_command_v14.py \
  artifacts/terrain_demo/command_conditioning_v14 \
  --output-prefix outputs/v14_recheck
```

상세 사전 계획: [command_conditioning_v14.md](experiment_plans/command_conditioning_v14.md).
