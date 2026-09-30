# v16 — 실제 지면 접촉으로 조건화한 발 미끄러짐 비용

## 질문과 논문 적용 범위

v15의 몸체 측면속도·방향 비용 대신, **지면에 닿은 발끝의 수평 운동만**
억제하면 험지 통과와 장시간 돌다리 주행이 나아지는가? 이 한 가지 변경을
동일 센서가 켜진 대조군과 같은 학습 예산으로 비교했다.

- [Miki et al. (2022), Supplement S7](https://arxiv.org/html/2201.08117v1)의
  접촉 발 속도 비용과 [Aractingi et al. (2023)](https://www.nature.com/articles/s41598-023-38259-7)의
  접촉 중 수평 미끄러짐 억제에서 아이디어를 얻었다. 논문 보상 전체, 계수,
  로봇 구조 또는 학습 방법을 재현한 것은 아니다.
- [Isaac Lab 2.3 ContactSensor 공식 API](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html#contact-sensor)의
  필터된 접촉 법선력을 썼다. 네 발에 독립 센서를 달고 terrain mesh와 별도 flat
  plane의 실제 collision prim을 모두 필터한다. 두 지면 모두 발별 접촉이
  나오는지 개발 실험에서 확인했다.
- 91D actor/critic 관측, 기존 ray/height 기반 깊이 히스토리 전환기,
  v5 교사와 기존 자세 높이 보상은 그대로다. **새 접촉 정보는 보상과 진단에만
  쓰며 추론 관측에는 들어가지 않는다.** 실제 RGB-D 카메라 정보도 아니다.

필터된 지면 접촉력의 크기가 발마다 2N을 넘으면 접촉으로 판단한다. 기존
발끝 기하학에서 계산한 world-XY 속도에 대해 새 보상은 다음과 같다.

```text
c_i = 1(해당 발의 지면 필터 법선력 > 2 N)
r_slip = -(1/4) * Σ_i c_i * min(발끝 수평속도_i² / (1 m/s)², 1)
```

따라서 항의 범위는 −1~0이다. `control`은 이 항의 가중치 0, `slip`은 1이며,
양쪽 모두 똑같은 센서·물리 설정이다. 이 값은 **발끝 속도의 근사 비용**이지
실제 발바닥 접촉점의 접선 미끄러짐, 마찰력 또는 접촉 품질의 직접 계측이 아니다.
지면 필터에서 접촉이 감지되지 않은 순간도 몸체가 공중에 있다는 증거가 아니다.
센서 직후 reset/portal 이동 샘플은 진단에서 무효로 분리했다.

[결과를 보기 전 고정한 계획](experiment_plans/contact_slip_v16.md) ·
[문헌과 구현 차이](../artifacts/terrain_demo/contact_slip_v16/research.md).

## 사전 고정된 학습·평가

v15의 **direction 보상을 쓰지 않은 control 최종249**에서 시작해 학습된
91D 정책/명령 가중치를 보존했다. 탐색 표준편차 0.2, 새 Adam(1e-4)으로만
다시 시작했다. 학습 seed48·지형 geometry98에서 군당
4,096환경 × 32step × 250iteration = **32,768,000 transition**, 합계
**65,536,000 transition**을 실행했다. 최종249만 새 지도에 평가했다.

새 지도 geometry99/reset62와 geometry100/reset63을 동일 초기 상태로
고정했다. `v15_control`과 원래 v5+v10 history는 설명용 참고선이며, 새 비용의
효과는 **`slip` 대 동일 예산 `control`**, 그리고 **`history_slip` 대
`history_control`**로만 판단한다. 16초 혼합 지형은 제어기마다 험지300회+
평지50회, 별도64초 최고난도 돌다리는 제어기마다20회다. 여섯 제어기의
평가24파일, 첫 episode 총2,220회다. 엄격한 1/6타일 판정은 거리
13.1/53.1m 이상과 낙상·레인/월드 이탈 없음을 함께 요구한다.

주평가 승격 조건은 험지 1/6타일 비감소, 낙상/레인 이탈 비증가,
월드 이탈0, 평지 낙상 비증가 및 단독 평지 평균 episode 속도 증가다.
하이브리드는 평지에서 환경별 원시 결과의 고정-v5 분기 동일성과
적어도 한 험지 지표의 엄격한 개선까지 요구한다. 별도64초 결과는
16초의 실패를 상쇄하지 않는다. 중간 checkpoint·계수·seed·지도 재선택은 없다.

## 실제 결과 — 네 비교 모두 FAIL

### 주평가: 16초 혼합 지형 (험지300회, 평지50회/제어기)

| 제어기 | 엄격 1타일 | 엄격 6타일 | 험지 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| v15_control, 참고 | 277 | 148 | 18 | 3 | 3 | 10.0643 |
| control | 279 | 176 | 20 | 1 | 3 | 10.8072 |
| slip | 275 | 163 | 19 | 7 | 3 | 10.7073 |
| history_original, 참고 | 277 | 152 | 21 | 4 | 5 | 9.1413 |
| history_control | 280 | 179 | 17 | 2 | 5 | 9.1413 |
| history_slip | 280 | 172 | 17 | 2 | 5 | 9.1413 |

단독 비용군의 낙상은20→19로 한 건 줄었지만, 6타일176→163/300,
1타일279→275/300, 이탈1→7/300, 평지 속도10.8072→10.7073m/s로
전체 조건 **FAIL**이다. History 전환기도 6타일179→172/300이고 엄격한
지형 개선이 없어 **FAIL**이다. History 두 군의 평지 원시 결과는 환경별
정확히 동일해 새 비용의 평지 개선으로 계산하지 않았다. 모든 제어기의
월드 이탈은 평지를 포함해0건이다.

### 별도 평가: 64초 최고난도 돌다리 (20회/제어기)

| 제어기 | 엄격 1타일 | 엄격 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v15_control, 참고 | 11 | 11 | 4 | 5 |
| control | 10 | 8 | 7 | 3 |
| slip | 9 | 8 | 8 | 4 |
| history_original, 참고 | 17 | 12 | 3 | 2 |
| history_control | 8 | 7 | 11 | 1 |
| history_slip | 6 | 6 | 9 | 4 |

단독은 6타일8→8/20에 머물며 낙상7→8/20, 이탈3→4/20이다.
History는 낙상11→9/20이지만 6타일7→6/20, 이탈1→4/20으로
둘 다 **FAIL**이다. 이20회에서 얻은 방향과 주평가 결과를 합쳐서
성공률을 만들지 않았다.

## 접촉 진단과 해석 한계

16초 혼합 지형에서 단독 control→slip의 유효 방문 상태 평균은
접촉 조건부 bounded cost **0.1046→0.1054**, 감지된 접촉 발끝 속도
**1.0512→1.0370m/s**, 감지된 발 접촉 비율 **0.1339→0.1366**이었다.
접촉 발속도는 조금 줄었어도 실제 비용 평균과 통과율은 개선되지 않았다.
감지 접촉 발 중 속도 상한에 걸린 비율도 약68.9%→67.6%였다.
상한으로 보상이 둔감했을 가능성은 **가설**일 뿐 확인된 실패 원인이 아니다.
각 정책이 방문한 상태가 달라서 이 진단 평균은 상태를 고정한 인과 비교가 아니다.

개발64/4,096환경 짝 검증에서 각 발·양쪽 지면 접촉을 확인하고 무효
센서 샘플 비율0으로 통과했다. 학습 로그250회와 최종 모델·Adam tensor는
유한했고, 원래 v5 교사와 91D 초기 상태·정책·RNG 짝도 검증했다. 기존
v15 평가기와 새 센서 평가기의 개발 지도 물리 결과/초기관측도 정확히
일치했다. 점수화 전 소스20개·기존 소스123개·모델·지도 캐시·명령
ledger를 고정했고 보류 지도는 한 번씩만 평가했다. 전체884개 CPU 테스트,
compileall 및 diff 검사를 통과했다. 한 학습 seed와 새 지도 두 개의
유한 표본이므로 모든 험지 우월성이나 실제 로봇 안전을 주장하지 않는다.

**결론:** 이 방식은 고정된 조건에서 지속 험지 극복을 개선하지 못했다.
기본 권장 v5 정책이나 하이브리드 설정은 바꾸지 않았다. 새로운 의존성,
원격 Git 작업은 없다.

[전체 판정](../artifacts/terrain_demo/contact_slip_v16/summary.md) ·
[원시 집계·지형별 수치](../artifacts/terrain_demo/contact_slip_v16/summary.json) ·
[학습 검증](../artifacts/terrain_demo/contact_slip_v16/training_validation.json) ·
[독립 학습 검토](../artifacts/terrain_demo/contact_slip_v16/training_independent_review.md) ·
[독립 보류 원시 감사](../artifacts/terrain_demo/contact_slip_v16/independent_final_audit.md) ·
[평가 전 고정 기록](../artifacts/terrain_demo/contact_slip_v16/frozen.json).

## 실행 명령과 증거 보존

프로젝트 루트에서 기존 `../run-python` 환경을 사용한다. 실제 실행 명령,
종료 상태와 로그 SHA는 `artifacts/terrain_demo/contact_slip_v16/commands.jsonl`에
있다. 아래 harness는 원래 산출물을 덮어쓰지 않는다. 재학습은 이 저장소
결과를 지우지 말고 별도의 격리 복제본과 새 holdout에서 수행해야 한다.

```bash
../run-python scripts/run_contact_study.py prepare
../run-python scripts/run_contact_study.py preflight --device cuda:1
../run-python scripts/run_contact_study.py capacity --device cuda:1
../run-python scripts/run_contact_study.py freeze_train
../run-python scripts/run_contact_study.py train --device cuda:1
../run-python scripts/run_contact_eval.py final_smoke --device cuda:1
../run-python scripts/run_contact_eval.py freeze_eval
../run-python scripts/run_contact_eval.py prepare_eval --device cuda:1
../run-python scripts/run_contact_eval.py evaluate --device cuda:1
../run-python scripts/run_contact_eval.py horizon --device cuda:1
../run-python scripts/run_contact_eval.py report
```

원시 기록만 다시 집계하려면 새 출력 prefix를 지정한다. 이는 GPU 주행을
다시 하지 않으며 frozen 해시가 달라지면 실패한다.

```bash
../run-python scripts/summarize_contact_v16.py \
  artifacts/terrain_demo/contact_slip_v16 \
  --output-prefix outputs/v16_reaudit_new
```
