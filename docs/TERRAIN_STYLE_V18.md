# v18 — 깊이 조건부 v5 교사 스타일 prior (승격 실패)

## 논문에서 가져온 원리와 실제 구현

[Chane-Sane 외, *CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning* (2024), §IV](https://arxiv.org/html/2403.18765v1)은 높이 스캔 분산으로 평지를 구별해 보행 **스타일 제약을 평지에서만** 적용하고 거친 지형에서는 해제한다. 이번 실험은 그 *지형 조건부 스타일* 원리만 적용했다. CaT의 확률적 종료, 안전 제약, 네발 로봇 결과를 재현한 것이 아니다. 여기서 스타일 항은 이미 있던 frozen-v5 행동 평균 보조 손실(`0.02`)이다.

v16 `control` 최종249의 동일 91D actor/critic 가중치에서 `always`와 `gated`를 짝학습했다. 두 군 모두 같은 825-ray 스캔으로 전방 ROI(`0≤x≤1.5m`, `|y|≤0.8m`)의 유효 깊이 비율·상대 높이 표준편차를 계산한다. `gated`는 유효 비율이 20% 이상이고 표준편차가 3.5cm 초과 또는 coverage가 90% 미만이면 v5 교사 손실을 **정확히 0**으로 만든다. 평지·정보 부족 구역은 교사를 유지한다. `always`는 탐지 결과와 관계없이 교사를 항상 적용한다. 탐지 결과는 학습 전용 `[N,1]` 보조 관측이며 **actor/critic 입력 91D와 추론 정책은 바꾸지 않았다**. 계수·임계값은 이 과제의 사전 선언된 공학적 선택이지 논문 값이 아니다.

[결과 전 계획](experiment_plans/terrain_style_v18.md)에 학습·보류 지도·성공 조건을 고정했다. 양군은 같은 v16 시작 모델(SHA `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`), 새 Adam·std 0.2, seed50/geometry104, 동일 물리·보상·지형 배치·PPO를 사용했다. 각 4,096환경×32step×250iteration = **32,768,000전이**, 합계65,536,000전이를 학습하고 마지막 `model_249.pt`만 평가했다. 256환경 사전시험, 4,096환경 용량시험, 초기 정책/관측/RNG 짝, 학습 scalar 250회·모델/Adam 유한성, v5 교사 동일성, 새 평가기의 기존 v16 물리 결과 일치를 검증했다. 학습 중 탐지기가 평지를 교사 유지로 판정한 비율은 양군 100%, 최고난도 징검다리 계열은 약 10.5~10.7%였다. `always`에서 이 비율은 **탐지 진단값**일 뿐, 실제 교사 손실은 모든 표본에 적용됐다.

## 새 지도에서의 결과

geometry105/reset66 및106/reset67을 학습에 사용하지 않고 사전 고정했다. 기존 v12 깊이 히스토리 전환기를 그대로 사용한 하이브리드도 함께 평가했다. `v16_control`과 `history_original`은 설명용 참고선이다. 인과 비교는 **`gated`↔`always`**와 **`history_gated`↔`history_always`**뿐이다. 성공은 13.1m/53.1m 통과 *및* 낙상·레인/월드 이탈 없음이라는 엄격 기준이다.

### 주평가 — 16초 혼합 지형 (제어기당 험지 300회 + 평지 50회)

| 제어기 | 엄격 1타일 | 엄격 6타일 | 험지 낙상 | 험지 레인 이탈 | 평지 낙상 | 평지 평균 episode 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| v16_control (참고) | 276 | 172 | 20 | 3 | 3 | 10.8763 |
| always | 280 | 162 | 19 | 1 | 4 | 10.3253 |
| gated | 273 | 154 | 18 | 9 | 4 | 10.5984 |
| history_original (참고) | 261 | 139 | 34 | 3 | 3 | 9.2060 |
| history_always | 279 | 163 | 20 | 1 | 3 | 9.2060 |
| history_gated | 271 | 156 | 26 | 3 | 3 | 9.2060 |

단독 `gated`는 `always`보다 평지 속도가 10.3253→10.5984m/s(**+2.64%**)이고 험지 낙상은 19→18/300이지만, 엄격 6타일은 **162→154/300**, 레인 이탈은 **1→9/300**으로 악화했다. 히스토리 하이브리드도 6타일 **163→156/300**, 낙상 **20→26/300**, 이탈 **1→3/300**으로 악화했다. 두 주평가 승격 판정 모두 **FAIL**이다. 하이브리드의 평지 속도·낙상은 고정 v5 분기의 환경별 원시 배열이 정확히 같아 새 정책의 개선으로 해석하지 않는다. 모든 제어기의 평지 포함 월드 이탈은 0이었다.

### 별도 진단 — 64초 최고난도 징검다리 (제어기당 20회)

| 제어기 | 엄격 1타일 | 엄격 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v16_control (참고) | 9 | 7 | 9 | 3 |
| always | 9 | 9 | 10 | 2 |
| gated | 6 | 6 | 7 | 8 |
| history_original (참고) | 10 | 7 | 7 | 2 |
| history_always | 7 | 6 | 11 | 2 |
| history_gated | 9 | 7 | 4 | 9 |

히스토리 `gated`는 낙상 11→4/20, 6타일 6→7/20이지만 레인 이탈 **2→9/20**이다. 단독도 6타일 9→6/20, 이탈 2→8/20이다. 별도 진단의 두 승격 판정 역시 **FAIL**이며 주평가 실패를 덮지 않는다.

## 검증·한계·재현

[24개 원시 JSON/지형별 집계](../artifacts/terrain_demo/terrain_style_v18/summary.json) · [판정표](../artifacts/terrain_demo/terrain_style_v18/summary.md) · [독립 원시 감사](../artifacts/terrain_demo/terrain_style_v18/independent_raw_audit.json) · [학습 검증](../artifacts/terrain_demo/terrain_style_v18/training_validation.json) · [평가 전 동결](../artifacts/terrain_demo/terrain_style_v18/frozen.json) · [모델 SHA](../artifacts/terrain_demo/terrain_style_v18/trained_models.json) · [최종 검증](../artifacts/terrain_demo/terrain_style_v18/final_validation.json) · [실행 기록](../artifacts/terrain_demo/terrain_style_v18/commands.jsonl).

독립 원시 감사는 별도 계산으로 24파일·2,220개 *첫 episode*의 분모, 엄격 통과/낙상/이탈, 지형·난이도별 집계, 평지 속도, 초기 상태/관측 prefix/RNG 짝과 하이브리드 평지 분기를 재확인했다. 이는 시뮬레이터 재실행은 아니다. 최종 971개 CPU 회귀 테스트·14개 v18 Python AST/컴파일·14개 계획 복사본/문서 링크·`git diff --check`도 통과했다. Ruff/Mypy/Pyright는 설치되지 않아 실행하지 않았다. 저장된 감사와 독립 계산을 다시 대조하려면 저장소 루트에서 `python3 scripts/audit_style_v18_raw.py`를 실행한다. 이 검증 스크립트는 **점수 확정 후 작성**되어 frozen 학습/평가 소스에 포함되지 않으며 평가 점수에는 관여하지 않는다. 한 학습 seed·두 새 지도, 이상적인 시뮬레이터 높이 ray이므로 보편적 우월성·실제 RGB-D/로봇 안전성을 주장하지 않는다. **기본 모델/하이브리드는 변경하지 않았다.**

기존 결과를 덮어쓰지 않는 새 격리 복제본·새 holdout에서만 아래 절차를 재현한다. 현재 산출물에서 `prepare`/`train`/`report`를 다시 실행하면 기존 출력 보호 장치가 거부한다.

```bash
PYTHONPATH=src ../run-python scripts/run_style_study.py prepare
PYTHONPATH=src ../run-python scripts/run_style_study.py prepare_cache --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_study.py preflight --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_study.py capacity --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_study.py freeze_train
PYTHONPATH=src ../run-python scripts/run_style_study.py train --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_eval.py parity --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_eval.py final_smoke --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_eval.py freeze_eval
PYTHONPATH=src ../run-python scripts/run_style_eval.py prepare_eval --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_eval.py evaluate --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_eval.py horizon --device cuda:1
PYTHONPATH=src ../run-python scripts/run_style_eval.py report
```

새 의존성·원격 Git 작업 없이 로컬에만 보존했다. 다음 비교는 이 보류 지도에 임계값을 맞추지 말고 새 사전 계획·지도에서 검증해야 한다.
