# v17 — 학습 진도 기반 지형 커리큘럼 (승격 실패)

## 참고 논문과 검증 질문

[Li, Li & Hutter, *Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning* (2026)](https://arxiv.org/html/2601.17428v1)의
지형별 **부호 있는 학습 진도**(연속 구간의 평균 episode return 차이)로 학습 지형을
재분배하는 아이디어를 Ant에 적용했다. 논문의 ANYmal-D 학습/지형이나 계수를
재현한 것은 아니며, 논문은 이 과제의 최고 난도 징검다리를 입증하지 않는다.

질문: 같은 episode별 지형 재샘플링과 같은 PPO 예산 아래서, 학습 진도 분포가
고정 가중치 분포보다 **새 험지의 엄격 통과·낙상·이탈**을 개선하는가?
v16의 정적 시작 지형과 이 원인 비교를 혼동하지 않도록, `fixed`와 `lp` **양쪽**에
동일한 reset 재샘플러를 넣었다. 유일한 학습 처리 차이는 샘플 확률의 갱신이다.

[결과 보기 전 작성한 계획·개발 수정 이력](experiment_plans/learning_progress_v17.md) ·
[문헌과 적용 차이](../artifacts/terrain_demo/learning_progress_v17/research.md).

## 방법과 검증 범위

- 두 군 모두 v16 무접촉비용 `control` 최종249의 동일한 91D 정책·학습된 명령
  가중치·고정 v5 교사를 계승한다. 탐색 표준편차만 0.2로 되돌리고 새 Adam
  1e-4/iteration0에서 시작했다. 접촉 보상 가중치0, 기존 물리·깊이 관측·
  history 전환기·자세 보상은 동일하다. **추론 모델이나 깊이 인식기를 바꾼
  실험이 아니라 학습 지형 배분 실험**이다.
- 원래 지형 혼합은 다른 다섯 mesh family 1씩, stepping_stones 4, flat 2다.
  Flat 질량2/11과 mesh 질량9/11을 유지한다. Mesh 30개 family×level 작업의
  완료 episode return을 이전 지형에 귀속하고, 1,024개 mesh episode 이상·
  작업마다 최소4개가 모이면 구간 평균을 갱신한다. 첫 구간은 기준 평균만 만든다.
  이후 `LP_i = mean_i(now) − mean_i(previous)`,
  `q_i ∝ base_i × exp(clip(LP_i/5, −2, 2))`,
  `p_mesh_i = 0.4 × base_i + 0.6 × q_i`다. 이 floor·온도·clip·flat 고정은
  **이번 실험의 사전 선언된 변형**이지 논문의 상수가 아니다.
- RSL의 시작 episode 길이는 임의화되므로 환경별 첫 완료는 새 지형을 뽑되
  LP 통계에서 제외했다. 재설정 전 reward manager의 terminal 포함 episode
  합계를 읽고, 기존 reset을 한 번 실행한 뒤 새 지형 입구로 pose를 옮겼다.
  원래 지형의 완료 거리 기록과 다음 episode의 보상 기준점은 유지한다.
- 두 군 각 **4,096환경 × 32step × 250iteration = 32,768,000전이**,
  합계65,536,000전이. 학습 geometry101/seed49, 기본 window1,024,
  최종 model249만 평가했다. 양쪽 모두 기본 구간27회, 초기 제외4,096개,
  학습 로그250회·모델/Adam 유한성·초기 정책/관측/RNG 짝을 검증했다.
- 개발용 window16의 256환경/120iteration 첫 실행은 드문 지형별4개 조건
  때문에 구간1회에 그쳤다. **본학습·새 지도 평가 전** 수정 사실을 계획에
  기록하고 두 군 모두 256환경/300iteration으로 재실행했다(각 구간4회).
  처음 실패한 로그도 남겼다. 본학습 방식·계수·예산·보류 지도는 바꾸지 않았다.

새 지도 geometry102/reset64와 103/reset65, 16초 혼합 지형의 제어기별
험지300+평지50회, 별도64초 최고난도 징검다리20회, 여섯 제어기 총24파일·
2,220개 첫 episode를 비교했다. `v16_control`과 `history_original`은 설명용
참고선이며 **인과 비교는 `lp`↔`fixed`, `history_lp`↔`history_fixed`**다.
엄격 1/6타일은 13.1/53.1m와 낙상·레인/월드 이탈 없음을 함께 요구한다.

## 실제 결과

### 주평가: 16초 혼합 지형 (험지300회·평지50회/제어기)

| 제어기 | 1타일 | 6타일 | 험지 낙상 | 레인 이탈 | 평지 낙상 | 평지 평균 episode 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| v16_control, 참고 | 278 | 179 | 19 | 3 | 2 | 11.0028 |
| fixed | 270 | 163 | 21 | 7 | 2 | 10.1889 |
| lp | 268 | 159 | 27 | 5 | 2 | 10.6663 |
| history_original, 참고 | 268 | 147 | 30 | 0 | 7 | 8.9741 |
| history_fixed | 279 | 166 | 20 | 1 | 7 | 8.9741 |
| history_lp | 270 | 162 | 24 | 6 | 7 | 8.9741 |

`lp`는 `fixed`보다 평지 속도는 높았지만 험지 1타일270→268,
6타일163→159, 낙상21→27/300으로 **단독 승격 FAIL**이다. History도
1타일279→270, 6타일166→162, 낙상20→24, 레인 이탈1→6/300으로
**승격 FAIL**이다. History의 평지 v5 분기는 환경별 원시 결과가 정확히
같았고 새 정책의 평지 성과로 세지 않았다. 전 제어기의 평지 포함 월드 이탈0이다.

### 별도 평가: 64초 최고난도 징검다리 (20회/제어기)

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v16_control, 참고 | 9 | 9 | 9 | 2 |
| fixed | 5 | 3 | 10 | 7 |
| lp | 7 | 7 | 10 | 4 |
| history_original, 참고 | 13 | 6 | 7 | 0 |
| history_fixed | 13 | 12 | 4 | 3 |
| history_lp | 8 | 8 | 10 | 5 |

여기서는 단독 `lp`가 `fixed`보다 6타일3→7/20, 이탈7→4/20으로
**별도 기준 PASS**였다. 그러나 history는 12→8/20, 낙상4→10/20으로
**FAIL**이며, 64초 단독 개선으로 16초 주평가 실패를 뒤집지 않았다.

실제 학습 전이 점유율에서 징검다리는 `fixed` 35.81%→`lp` 30.95%,
flat은18.17%→18.06%였다. LP가 징검다리 노출을 줄인 것은 관측 사실이나
16초 성능 악화의 **확정 원인이라고 볼 수는 없다**. 단일 학습 seed·새 지도
두 개의 시뮬레이션 결과이며 보편적 우월성, 통계적 유의성, 실제 RGB-D/로봇
안전을 주장하지 않는다. 기존 기본 정책/하이브리드는 교체하지 않았다.

## 원시 증거와 재현 명령

[24개 원시 JSON의 집계·지형별 수치](../artifacts/terrain_demo/learning_progress_v17/summary.json) ·
[판정 요약](../artifacts/terrain_demo/learning_progress_v17/summary.md) ·
[학습 검증](../artifacts/terrain_demo/learning_progress_v17/training_validation.json) ·
[독립 원시 감사](../artifacts/terrain_demo/learning_progress_v17/independent_final_audit.md) ·
[평가 전 고정](../artifacts/terrain_demo/learning_progress_v17/frozen.json) ·
[실행 명령/로그 해시](../artifacts/terrain_demo/learning_progress_v17/commands.jsonl).
각 원시 파일은 `evaluations/`와 `horizon/`에 있다. 새 출력은 덮어쓰기를
거부한다. 아래 명령은 **새 격리 복제본·새 holdout**에서 재현할 때만 사용한다.

```bash
PYTHONPATH=src ../run-python scripts/run_lp_study.py prepare
PYTHONPATH=src ../run-python scripts/run_lp_study.py prepare_cache --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_study.py preflight --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_study.py capacity --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_study.py freeze_train
PYTHONPATH=src ../run-python scripts/run_lp_study.py train --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_eval.py parity --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_eval.py final_smoke --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_eval.py freeze_eval
PYTHONPATH=src ../run-python scripts/run_lp_eval.py prepare_eval --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_eval.py evaluate --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_eval.py horizon --device cuda:1
PYTHONPATH=src ../run-python scripts/run_lp_eval.py report
```

Ruff/Mypy/Pyright는 설치되어 있지 않아 통과를 주장하지 않는다. 새 의존성·
원격 Git 작업·기본 설정 변경은 없다.
