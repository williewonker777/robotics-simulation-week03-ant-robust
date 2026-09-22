# 험지 보행 실험 연대기 (v0–v10)

이 문서는 이 저장소에서 수행한 Ant PPO 실험의 **과정, 고정 평가 결과, 채택/보류
결정**을 시간순으로 요약한다. 숫자의 세부 분모, 실행 명령, 소스·체크포인트 해시,
원시 평가 배열은 각 버전의 상세 문서와 `artifacts/` 보고서를 기준으로 한다.

## 현재 결론

- **권장 정책은 v5 후속 복구 정책**이다. 이것은 과제의 최초 평지 정책이나 v0이
  아니라, 60D 과제 인터페이스로 험지 재학습·징검다리 복구를 거친 안정 기준선이다.
  - 체크포인트:
    [`artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt`](../artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt)
  - SHA-256:
    `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e`
- v6–v10은 모두 v5를 교체하지 못했다. 일부 지표나 특정 난이도에서는 개선을
  보였지만, 미리 정한 전체 승격 기준을 통과하지 못했거나 장기 안정성이 부족했다.
- 지형 관측은 실제 RGB-D 카메라가 아니다. v3/v4의 높이 스캔, v6의 143-ray
  스캔, v7–v10의 발·지형 ray/height 지도는 모두 시뮬레이터의 이상적이고
  노이즈 없는 raycast/height 신호다. 카메라 지연·가림·깊이 오차·실물 센서
  보정은 검증하지 않았다.

## 읽는 법과 공통 경계

- 16초 평가의 엄격한 1/6타일 성공은 각각 전진 거리 13.1/53.1 m 이상에 더해,
  첫 에피소드에서 자세 종료·발자국 레인 이탈·world 이탈이 없어야 한다. 생존,
  거리만의 도달, 영상 장면은 성공을 뜻하지 않는다.
- 후속 버전마다 보류 지형 지도와 reset이 다르다. **수치 비교는 같은 버전·같은
  지도/reset·같은 성공 정의 안에서만** 과학적으로 해석한다. 특히 기준 v5는
  후속 실험에서 학습 정책 수만큼 복제하지 않고 조건당 한 번만 평가했으므로
  분모가 다르다.
- 후속 확장 실험의 소수 보류 지도와 유한한 에피소드는 수천 개의 독립 지형
  표본이나 통계적 유의성, 임의의 지형/실물 로봇 안전을 보장하지 않는다.

## 연대기

| 버전 | 질문과 방법 | 고정 결과와 결정 |
|---|---|---|
| **v0** | 과제의 60D/8D PPO에서 마찰·질량/COM·reset·관측 잡음·외란 domain randomization을 3개 seed로 비교했다. | 공개 OOD 세 조건 평균 return은 baseline 대비 +10.8%였지만, push 개선은 작고 robust seed 분산은 더 컸다. 이는 평지 일반화 실험이며 험지 극복의 증거는 아니다. |
| **v1** | flat/rough/slope/stairs procedural curriculum과 자세 기반 종료를 도입했다. | 선택 정책은 100환경에서 23/100 완주으로, 초기 다중지형 보행을 보였지만 복잡한 단절 지형은 범위 밖이었다. |
| **v2** | waves, obstacles, stepping stones를 더한 7개 지형군으로 확대하고 낮은 난이도 warm-up 뒤 전체 curriculum을 학습했다. | 다섯 평가 seed 평균 40.6/100 완주으로 범위는 넓어졌지만, 극한 지형을 보장하지 않았다. |
| **v3** | pit/gap/boxes를 포함한 10개 지형군과 torso 장착 54-ray height scanner(114D)를 추가했다. | 동일 극한 분포에서 scanner 정책은 no-scan보다 33/100 대 23/100 완주이었지만 pit/gap은 여전히 가장 어려웠다. |
| **v4** | v3의 114D scanner를 유지하고 scan 위치, recovery curriculum, gap/pit 보상을 조정했다. | gap/pit 길이는 일부 늘었지만 seed별 완주는 평균 27.7/100이었다. 이 단계도 일반적인 단절 지형 해결을 주장하지 않는다. |
| **v5** | 과제 호환 인터페이스로 돌아가 **60D/8D** 험지 레인 평가를 만들고, 발자국 포함 레인 유지와 실제 8 m 통과를 측정했다. 즉 v3/v4가 이미 scanner를 썼지만 v5는 다시 60D로 재설정했다. | 최초 selected v5는 450 험지 에피소드에서 1타일 82.0%, 6타일 43.8%, 낙상 7.1%였지만 고난도 돌다리는 미해결이었다. |
| **v5 후속** | 돌 사이 정체를 진단한 뒤, 평가 지형·성공 기준·정책 차원을 바꾸지 않고 훈련 시 저 torso/발끝 clearance 신호를 사용한 recovery rehearsal을 수행했다. | 새 paired 평가에서 1타일 472/600→517/600, 돌 33/100→69/100, 전체 험지 낙상 61→61이었다. 단, 돌 6타일은 1/100에 그쳤고 일부 family/평지 낙상 trade-off가 남았다. 이 정책이 현재 권장 기준선이다. |
| **v6** | 사용자의 깊이 관측 요청에 따라 60D에 143 height+143 validity를 붙인 별도 346D 정책을 1,500 iteration 학습했다. 이는 v5 뒤에 depth를 다시 추가한 실험이다. | fresh 600 험지 에피소드에서 돌 62/100→79/100, 난이도 0.8 돌 6/20→15/20이었지만 전체 1타일 524→521, 낙상 46→62, 6타일 274→119로 악화되어 교체하지 않았다. 난이도 1.0에서도 낙상이 늘었다. |
| **v7** | height CNN과 네 발 위치 지도를 사용하는 514D 정책을 blind/height/footmap 세 조건, 각 3 seed의 동일 예산으로 비교했다. | footmap은 height 대비 낙상 108→85/900, 6타일 144→216/900이었지만, frozen v5의 6타일 135/300(45.0%)보다 216/900(24.0%)로 낮아 승격 실패했다. 최고 난이도 돌 6타일은 모든 arm에서 0이었다. |
| **v8** | v5 actor를 고정하고 제한된 `0.5*tanh` 행동 평균 residual만 학습해, 기존 보행을 보존하며 지형 보정을 시험했다. | residual은 v7보다 6타일을 회복했지만, v5보다 낮고 낙상/평지 낙상도 개선하지 못했다. 최고 난이도 돌 6타일은 0이어서 승격 실패했다. |
| **v9** | 825개 ray에서 발끝 착지 후보를 만들고 88D MLP로 feet-only, targets, targets+support-shaping을 3 seed씩 비교했다. | targets/guided는 v5보다 6타일 비율이 약간 높았으나 1타일·낙상·레인 이탈이 회귀했다. 후보+보상은 targets보다 낙상 91→105/900으로 늘었고, 최고 난이도 돌 6타일은 모두 0이었다. |
| **v10** | v9 targets 88D 학생을 유지하되, 실행 때는 사용하지 않는 frozen-v5 행동 평균 prior를 훈련 손실에만 추가했다. prior 없음(λ=0)과 anchored(λ=0.02)를 각 3 seed로 비교했다. | anchored는 free보다 대부분의 16초 지표를 개선했지만 v5보다 레인 이탈이 많아 엄격한 승격 실패했다. 별도 64초 돌 진단에서도 낙상이 높아 완전한 장기 복구로 볼 수 없다. |

## v5가 기준선이 된 이유

v5의 후속 rehearsal은 기존 60D 정책의 인터페이스와 평가를 유지하면서, 기준선과
같은 지형/reset의 fixed 평가 및 선택 뒤의 새 paired 평가에서 모두 strict 통과를
늘리고 전체 험지 낙상을 증가시키지 않았다. 그러나 이는 “모든 험지를 극복했다”는
결론이 아니다. 고난도 돌의 지속 6타일 주행, family별 trade-off, 평지 레인 낙상은
남아 있다. 상세 분모와 원인은 [v5 측정 기준선](ROUGH_V5.md) 및
[v5 복구 연속 실험](ROUGH_RECOVERY.md)을 따른다.

## v10의 최신 비교 결과

v10의 6개 최종 checkpoint는 holdout 전에 고정되었다. 두 새 지도에서 최대 16초인
14개 평가 JSON, 총 2,450 first episode를 집계했다. frozen v5의 분모는 300 terrain
episode이고, 각 학습 arm의 분모는 세 seed를 합친 900 terrain episode이다. 따라서
행의 절대 count를 같은 독립 반복 수로 해석하면 안 된다.

| 정책 | 엄격한 1타일 | 엄격한 6타일 | 험지 낙상 | 레인 이탈 | 평지 낙상 |
|---|---:|---:|---:|---:|---:|
| frozen v5 | 260/300 (86.7%) | 132/300 (44.0%) | 30/300 (10.0%) | 0/300 | 4/50 |
| prior 없음 (free) | 744/900 (82.7%) | 422/900 (46.9%) | 96/900 (10.7%) | 61/900 (6.8%) | 13/150 |
| v5 mean prior (anchored) | 802/900 (89.1%) | 454/900 (50.4%) | 73/900 (8.1%) | 15/900 (1.7%) | 3/150 |

- anchored는 matched free보다 1/6타일 성공과 낙상을 개선했고, 별도의 방향성
  가설 gate는 통과했다.
- 그러나 승격 gate는 **v5 대비 레인 이탈**(15/900 대 0/300) 하나로 실패했다.
  따라서 방향성 결과가 권장 정책 교체를 정당화하지 않는다.
- 최고 난이도 1.0 전체 family에서 anchored의 6타일은 13/180으로 free 24/180보다
  낮다. aggregate 개선을 모든 어려운 경우의 개선으로 일반화하지 않는다.

### 별도 64초 최고난도 돌 진단

이 진단은 promotion에 쓰지 않았고 16초 기본 평가와 합치지 않았다. 두 지도,
정책별 paired reset 배치, stones 1.0만 사용했으며 v5/free/anchored의 분모는 각각
20/60/60이다. 53.1 m를 먼저 넘고 후에 넘어지거나 레인을 벗어난 경우는 strict
성공이 아니다.

| 정책 | 엄격한 1타일 | 엄격한 6타일 | 낙상 | 레인 이탈 | 64초 생존 |
|---|---:|---:|---:|---:|---:|
| frozen v5 | 13/20 | 0/20 | 5/20 | 0/20 | 15/20 |
| free | 19/60 | 14/60 | 29/60 | 12/60 | 31/60 |
| anchored | 23/60 | 15/60 | 31/60 | 5/60 | 29/60 |

anchored의 6타일 count는 free보다 하나 높지만, 낙상은 31/60(51.7%)이고 64초
생존도 free보다 낮다. 두 지도·유한 표본이며, hit time은 도달한 에피소드에 조건부인
통계다. 이 결과는 장기 hard-stone 극복이나 안전성을 입증하지 않는다.

## 남은 한계와 다음 해석의 경계

1. **실제 인지 문제는 아직 남아 있다.** 이상적 높이 ray와 휴리스틱 발 후보는
   RGB-D, 접촉 센서, 캘리브레이션된 foot support 판정, IK/whole-body 계획을 대체하지
   않는다.
2. **지속 극복이 미해결이다.** 최고 난이도 돌에서 짧은 1타일 성과가 있어도 6타일과
   64초 낙상은 여전히 약하다.
3. **보상·prior는 안전 제약이 아니다.** v9 support shaping과 v10 mean prior는
   접촉 안정성 또는 레인 유지의 보증이 아니며, v10 teacher도 추론 시 호출되지 않는다.
4. **버전 간 리더보드화는 부정확하다.** 보류 지도, reset, 모델 수, 분모가 바뀌므로
   각 사전 선언된 비교 안의 gate와 원시 평가를 우선한다.

## 상세 기록과 산출물

| 범위 | 설계·결과 문서 | 결과/감사/비교 화면 |
|---|---|---|
| v5 측정·복구 | [ROUGH_V5](ROUGH_V5.md), [ROUGH_RECOVERY](ROUGH_RECOVERY.md) | [요약](../artifacts/terrain_demo/evaluations/rough_v5/summary_20260921.md), [복구 요약](../artifacts/terrain_demo/evaluations/rough_v5/rehearsal_summary_20260921.md) |
| v6 depth | [DEPTH_V6](DEPTH_V6.md) | [결과](../artifacts/terrain_demo/evaluations/depth_v6/summary.md), [고난도 비교](../artifacts/terrain_demo/depth_v6_hard/index.html) |
| v7 footmap | [FOOTMAP_V7](FOOTMAP_V7.md) | [요약](../artifacts/terrain_demo/footmap_v7/summary.md), [감사](../artifacts/terrain_demo/footmap_v7/independent_review.md), [비교 화면](../artifacts/terrain_demo/footmap_v7/index.html) |
| v8 residual | [RESIDUAL_V8](RESIDUAL_V8.md) | [요약](../artifacts/terrain_demo/residual_v8/summary.md), [감사](../artifacts/terrain_demo/residual_v8/independent_review.md), [비교 화면](../artifacts/terrain_demo/residual_v8/index.html) |
| v9 foothold | [FOOTHOLD_V9](FOOTHOLD_V9.md) | [요약](../artifacts/terrain_demo/foothold_v9/summary.md), [감사](../artifacts/terrain_demo/foothold_v9/independent_review.md), [비교 화면](../artifacts/terrain_demo/foothold_v9/index.html) |
| v10 prior | [PRIOR_V10](PRIOR_V10.md) | [16초 요약](../artifacts/terrain_demo/prior_v10/summary.md), [64초 진단](../artifacts/terrain_demo/prior_v10/horizon_summary.md), [독립 감사](../artifacts/terrain_demo/prior_v10/independent_review.md), [비교 화면](../artifacts/terrain_demo/prior_v10/index.html) |

초기 v0–v4의 과제 설정, 경로 및 수치는 [README](../README.md)와
[`artifacts/terrain_demo/README.md`](../artifacts/terrain_demo/README.md)에 보존한다.
