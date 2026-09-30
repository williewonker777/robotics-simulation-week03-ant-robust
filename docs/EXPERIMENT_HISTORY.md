# 험지 보행 실험 연대기 (v0–v24)

이 문서는 이 저장소에서 수행한 Ant PPO 실험의 **과정, 고정 평가 결과, 채택/보류
결정**을 시간순으로 요약한다. 숫자의 세부 분모, 실행 명령, 소스·체크포인트 해시,
원시 평가 배열은 각 버전의 상세 문서와 `artifacts/` 보고서를 기준으로 한다.

## 현재 결론

**2026-09-30 과제 마무리:** 원래 60D/8D 과제의 제출 모델은 기존 7개 동일-budget
run 중 공개 종합 return이 가장 높은 **Robust seed42 `model_999.pt`**로 정리했다.
ID/OOD 4종을 100환경씩 새로 실행했고 기존 seed24 first-episode return/length 배열을
정확히 재현했다. 아래 v5 권장은 **험지 데모 기준선**이며 과제용 최고 return 모델과 다르다.
실험을 추가하거나 기존 모델·평가 수치를 바꾸지 않았다.
[최종 모델·실행 명령·제출 안내](FINAL_SUBMISSION.md).

**2026-09-28 v24 전역 학습률 민감도:** LR1e-4→1e-5의 동일예산3seed 학습과
신규2지도4,900첫 episode/9,800의존 window를 완료했다. 16초 단독6타일은
high161/150/157→low168/157/166으로 +7/+7/+9지만 낙상·레인·평지 trade-off가 남는다.
low 대 high6합산 gate와 low 대부모6합산 gate는16/64초 모두FAIL이다.
64초 단독6타일 차이도−11/−15/+24여서 일관된 개선이 아니며 **기본정책 유지**.
과거 회귀의 LR 원인 진단이나 actor만의 기전으로 해석하지 않는다.
[방법·전체결과·독립감사](LR_CONTINUATION_V24.md).

**2026-09-28 v23 같은 주행의 짝 시간 평가:** 새 학습 없이 기존8제어기를 신규2지도에서
2,800첫 episode·5,600의존 window로 평가했다. 16초 부모 대비6gate는 모두FAIL,
64초 History seed51·53 대부모는 두 지도 모두PASS다. History seed52의64초 합산PASS는
두 지도별FAIL을 숨기므로 일관된 개선으로 해석하지 않는다. 부모의 늦은 성공72회와
늦은 실패16회처럼 두 방향의 변화가 공존한다. **16초 주요평가를 유지하고 기본정책 유지**.
[방법·전체결과·독립감사](PAIRED_HORIZON_V23.md).

**2026-09-28 v22 추가학습 seed 민감도:** 같은 부모·초기 가중치·훈련 지형에서
무비용 추가학습을 seed51/52/53으로 새로 수행하고 신규 두 지도2,960회를 평가했다.
16초6타일은 단독164~171/300 대부모178, History168~172 대부모177로 세 seed
모두 낮았으며 12합산gate 전부FAIL이다. 별도64초 돌다리의 단독 seed52는17/20
대부모7로 두 지도 모두PASS지만, 같은 seed의 History는4/20으로 FAIL이다.
**조건별 seed 차이를 보존하고 기본정책 유지**. 3seed를 통계적 우월성이나
일반 강건성의 증거로 과장하지 않는다. [방법·전체결과·검증](SEED_CONTINUATION_V22.md).

**2026-09-28 v21 무비용 추가학습 대조군:** 새로 학습한 no_cost와 재사용 v20비용군,
부모 및 history조합을 신규2지도2,960회 비교했다. 무비용 대 부모는16초6타일
169→164/300·낙상27→18이나 이탈0→5로 FAIL,64초도6타일12→8/20으로 FAIL이다.
주요 합산12gate 중2개·별도64초12gate 중1개만 PASS며 지도별 일관성이 없다.
어느 추가학습군도 부모 대비 합산gate를 통과하지 않아 **기본정책 유지**.
회귀를 접촉벌점만의 탓으로 단정하지 않으며 과거학습군/한seed한계가 남는다.
[방법·전체결과·독립검증](CONTACT_CONTINUATION_V21.md).

**2026-09-28 v20 접촉 비용 커리큘럼:** 같은부모에서즉시/점진도입을짝학습한뒤
새두지도2,220회평가했다. 16초점진대즉시6타일158→164/300·낙상26→21이나
이탈1→6으로FAIL, 기존부모178/300에도못미쳤다. 64초단독6타일7→5/20,
history이탈1→9/20. 두평가여섯합산gate모두FAIL로**기본모델유지**.
[방법·결과·논문과의차이](CONTACT_CURRICULUM_V20.md).

**2026-09-28 v19 공통 지도 재평가:** 새 학습 없이 같은 신규 지도3개에서 비교한
v16 control은 v5 대비16초 6타일209→265/450·낙상49→31/450이었다. 하지만
64초 단독 이탈0→4/30과 지도별 회귀가 남는다. History의합산 이득도 모든지도에
일관되지는 않는다. [전체24파일2,220회 결과](REBASELINE_V19.md)는 서술적
고정정책 비교이며, 권장기본모델을 자동교체하지 않는다.

- **험지 데모 권장 정책은 v5 후속 복구 정책**이다. 이것은 과제의 최초 평지 정책이나 v0이
  아니라, 60D 과제 인터페이스로 험지 재학습·징검다리 복구를 거친 안정 기준선이다.
  - 체크포인트:
    [`artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt`](../artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt)
  - SHA-256:
    `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e`
- v6–v11은 모두 v5를 교체하지 못했다. 일부 지표나 특정 난이도에서는 개선을
  보였지만, 미리 정한 전체 승격 기준을 통과하지 못했거나 장기 안정성이 부족했다.
- v12는 최근 깊이 히스토리로 전환 빈도를 낮췄지만, v11 대비 혼합 지형 통과율을
  유지하지 못해 전체 개선 기준을 통과하지 못했다. 별도 장시간 돌다리의 소폭 개선과 구분한다.
- v13은평지낮추기/속도와별도장시간돌다리단독모델에서일부이점이있지만,
  혼합험지낙상과hybrid이탈trade-off로전체승격하지않았다.
- v14는 논문 기반 명령/높이 피드백 관측으로 험지 감지 구간 몸체 높이와 일부
  단독 돌다리 지표를 개선했지만, 혼합지형6타일 통과와 hybrid 장시간 성능을
  유지하지 못했다. 명시적 입력 접근만으로 전체 문제가 해결된 것은 아니다.
- v15의 방향 안정화 비용과 v16의 실제 지면 접촉 기반 발끝 미끄러짐 비용도
  각각 동일 예산 짝 실험에서 전체 기준에 실패했다. v16은 16초 험지6타일
  단독176→163/300, 히스토리 하이브리드179→172/300으로 악화했다.
- 지형 관측은 실제 RGB-D 카메라가 아니다. v3/v4의 높이 스캔, v6의 143-ray
  스캔, v7–v16의 발·지형 ray/height 지도는 모두 시뮬레이터의 이상적이고
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
| **v13** | 깊이조건부몸체목표0.44–0.58m·발여유·평지속도보상을추가하고동일예산추가학습control과비교했다. | 단독평지높이46.1→45.4cm·속도+2.75%지만혼합험지낙상37→39/300. 별도64초돌다리6타일5→10/20·낙상11→6/20; hybrid전체개선FAIL,기본교체없음. |

## v5가 기준선이 된 이유

v5의 후속 rehearsal은 기존 60D 정책의 인터페이스와 평가를 유지하면서, 기준선과
같은 지형/reset의 fixed 평가 및 선택 뒤의 새 paired 평가에서 모두 strict 통과를
늘리고 전체 험지 낙상을 증가시키지 않았다. 그러나 이는 “모든 험지를 극복했다”는
결론이 아니다. 고난도 돌의 지속 6타일 주행, family별 trade-off, 평지 레인 낙상은
남아 있다. 상세 분모와 원인은 [v5 측정 기준선](ROUGH_V5.md) 및
[v5 복구 연속 실험](ROUGH_RECOVERY.md)을 따른다.

## v10 비교 결과

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

## v11: 깊이 기반 고정 정책 전환

사용자의 "거친 지형은 v10, 일반 지형은 v5" 제안을 새 학습 없이 시험했다.
scan-only 선택기, 히스테리시스·최소 유지 시간·0.15초 행동 혼합을 추가하고,
개발 지도에서 세 기준 중 cautious를 선택한 뒤 동결했다. v10 세 학습 seed를
모두 포함했다. 최초 holdout의 초기 관측 SHA 불일치로 세 파일을 보존·제외하고,
사전 복구 선언 후 동일 조건 전체 배치를 한 번 재실행했다. 검증을 완화하지 않았다.

| 정책 | 16초 6타일 | 16초 낙상 | 별도 64초 돌다리 6타일 | 64초 낙상 |
|---|---:|---:|---:|---:|
| v5 | 133/300 (44.3%) | 23/300 (7.7%) | 1/20 (5.0%) | 7/20 (35.0%) |
| v10 anchored | 438/900 (48.7%) | 67/900 (7.4%) | 15/60 (25.0%) | 26/60 (43.3%) |
| hybrid | 451/900 (50.1%) | 81/900 (9.0%) | 14/60 (23.3%) | 21/60 (35.0%) |

16초 hybrid는 평지에서 v5를 유지했고, 험지의 평균 v10 혼합 비중은69.99%였다.
v10 대비 레인 이탈11/900→2/900, 64초 낙상26/60→21/60 등 이점은 있었으나,
16초 낙상이 v5보다 많고 1타일은 v10보다 낮아 전체 교체 기준은 실패했다.
개발875회와 최종2,590개 첫 에피소드의 독립 감사가 통과했다. 269개 CPU 테스트,
기존63개 frozen 소스/계획·4모델 불변도 확인했다. v11은 현재 로컬 추가 작업이다.
[상세 방법·실패·재시도·결과](HYBRID_V11.md).

## v12: 깊이 히스토리로 지속적인 험지/평지 판별

v5/v10 가중치를 고정하고 전환기에 최근1초의 깊이 특징을 저장했다. 짧은 험지 신호나
평지 틈에서 성급하게 전환하지 않도록 서로 다른 진입/복귀 시간창과 연속 확인을
추가했다. RNN/LSTM 재학습이나 전역 지형 지도 구축은 아니다.

최초 초기 관측 불일치350회는 보존·제외했다. 사용자 재검증 요청 후 초기 상태만
캡처해 지형 최초 생성과 OBJ 캐시 재로드 차이를 확인하고, 모든 평가 전에 같은
캐시를 준비했다. 관측 계산·모델·임계값을 바꾸거나 검증을 완화하지 않았다.
두 지도·학습 seed42 하나, 네 제어기1480개 첫 에피소드를 정확 일치 조건에서 비교했다.

| 조건 | 지표 | 기존 v11 | 히스토리 v12 |
|---|---|---:|---:|
|16초 혼합 험지, 각300회|6타일 통과|148/300 (49.3%)|146/300 (48.7%)|
|동일|낙상|28/300 (9.3%)|28/300 (9.3%)|
|동일|전환/활성100초|13.213|10.101 (23.6% 감소)|
|별도64초 돌다리1.0, 각20회|6타일 통과|5/20 (25%)|5/20 (25%)|
|동일|낙상|10/20 (50%)|9/20 (45%)|
|동일|레인 이탈|1/20|0/20|

16초에서는1타일도271→269/300으로 소폭 낮아 전체 개선 판정FAIL이다. 별도64초
기준은PASS지만1회 차이/20회라는 작은 표본이다. 히스토리 추가로 모든 맵에서
더 좋아진 것은 아니며 기본 모델의 과학적 승격을 선언하지 않았다.
353 CPU tests,16개 실제 평가,69개 동결 소스와4모델 불변을 확인했다.
[상세 방법·진단·전체 결과](HISTORY_V12.md).

## v13: 장애물에서는높게, 평지에서는낮고빠르게

같은v10 seed42 최종모델에서control/adaptive각250iteration·32,768,000transition,
학습seed45/지형75로짝학습했다. actor88D·고정v5·기존보상은유지하고새항목하나만
0/1로달리했다. 전체학습tensor/std/Adam·초기관측/RNG·실제설정과캐시를검사했다.

geometry78/79에서7제어기16초2450회+별도64초140회를비교했다. 혼합험지단독
control→adaptive는one262→260/six139→134/낙상37→39/이탈0→0(각300회).
평지몸체46.11→45.38cm·속도10.58→10.87m/s지만평지낙상2→4/50,
rough bin 몸체47.55→47.42cm로장애물상승의일관된개선은확인못했다.

별도64초최고난도돌다리단독모델은one5→12/six5→10/낙상11→6/이탈4→2(각20회),
rough bin 몸체43.60→44.16cm로별도기준PASS였다. 반면hybrid는one12→10/six7→7/
낙상6→5/이탈4→7로FAIL. 장시간단독개선이혼합지형실패를뒤집지는않는다.

기존history도같은지도에서비교했고모든hybrid의평지원시배열은고정v5와정확히같다.
485CPUtests·실제GPU훈련/평가smoke·독립감사로검증했다. 자세평균은방문상태조건부,
학습seed하나/지도두개이므로통계적확정/실물안전성을주장하지않고기본모델을유지했다.
[상세방법·결과](ADAPTIVE_POSTURE_V13.md) ·
[전체표](../artifacts/terrain_demo/adaptive_posture_v13/summary.md).

## v14 — 명시적 자세 명령·피드백 입력 (2026-09-22~23)

- Walk These Ways의 행동 명령을 입력/보상에 함께 사용하는 원리를 제한 적용했다.
  기존88D 뒤 목표 몸체 높이·측정 높이·유효성3개를 붙인다. 같은 v13 보상 아래
  actor/critic가 추가 입력을 가리는 masked와 받는 conditioned를 비교한다.
- 같은 원래v10 모델, seed46/geometry85, 각250iteration·32,768,000transition.
  초기 모든 learned88D tensor·고정60D교사를 보존, 신규 입력 가중치는0으로 시작한다.
  기존89source/6모델은 불변이다. 전체631 CPU tests와64/4096env 짝smoke 통과.
- 새지도86/58·87/59의6제어기16초2,100회와별도64초120회, 총2,220회 검증.
  16초 단독 masked→conditioned: one251→261/six144→135/낙상44→30/이탈1→8,
  각험지300회. 평지 속도11.1450→11.2795m/s, 낙상1/50으로 동일. 주기준FAIL.
- 16초 history: one263→273/six145→135/낙상31→24/이탈3→3, 주기준FAIL.
  원래history는272/146/25/2였다. 세hybrid의평지원시결과는정확동일, 학습이득아님.
- 별도64초최고난도돌다리각20회: 단독six5→9/낙상9→6/이탈2→2, 별도PASS.
  history는six5→2/낙상6→10/이탈7→3, FAIL. 장시간단독개선으로주실패를덮지않는다.
- 단독험지감지구간몸체16초47.97→48.92cm, 목표오차11.25→10.32cm.
  64초몸체43.18→44.60cm, 발여유10.18→12.71cm, 저속64.86→56.23%.
  평지몸체45.43→45.54cm로 더낮아지는효과는없었다. 방문상태통계이며인과추정아님.
- 정적35상태목표0/1입력은conditioned행동평균절대차0.04737, 대조군/원래모델0.
  환경step없는입력민감도만증명하며, 실제높이추종의증거가아니다.
- 개발지도51과거캐시중복/감사키오타는보존하고수정, 동일원시개발결과만재감사했다.
  새91D평가의초기88Dprefix/14결과필드는예전과동일하나, 파생속도lazycache순서로
  일부수동속도telemetry는다른제한을명시했다. 새두학습군/평가는동일순서다.
- [상세 방법·결과·실행](COMMAND_CONDITIONING_V14.md). 기본 모델 변경 및 원격 Git 없음.

## v15 — 방향 안정화 보상 (2026-09-23)

- Miki2022의 직교속도 억제, Aractingi2023의 속도/각속도 추종을 참고한 추가 보상.
  기존91D 관측/높이 보상/전환기를 유지하고 XY측면속도와 heading유도 yaw-rate 비용만 추가.
  v14conditioned에서 모든 학습 가중치를 보존하고 두 군을 각250iter·32,768,000transition 학습.
  seed47/geometry95, 마지막249만 선택, 기존106source/8모델 불변.
- 새지도96/60·97/61, 6제어기16초2,100회+별도64초120회, 총2,220회 완료.
  16초 단독 control→stable: one268→275/six160→158/낙상30→21/이탈2→3(각300회).
  평지속도9.9611→10.0834m/s(+1.23%), 낙상3/50 동일. 전체 기준 FAIL.
- 16초 hybrid: one267→270/six155→149/낙상32→26/이탈1→2, FAIL.
  세 hybrid 평지원시결과는 환경별 정확일치이며 새 학습 효과로 계산하지 않음.
- 64초 최고난도 돌다리: 단독six9→8/낙상4→8/이탈5→3, hybrid six11→6/낙상7→11/
  이탈2→2(각20회), 둘 다 FAIL. 모든world이탈0. 지속 험지 개선 미검증.
- 단독16초 측면속도0.6500→0.6509m/s, yaw-rate오차2.5739→2.6455rad/s로 목표 진단도
  일관되게 개선되지 않음. 평지몸체45.02→45.17cm로 낮아지지 않음.
  64초 hybrid 방향진단은 개선됐으나 완주성능은 악화; 방문상태통계이며 인과 추정 아님.
- 전체822 CPU tests, 64/4096env짝smoke, 기존v14추론18필드동일성, 최종105episode 검증.
  양쪽38tags×250scalar, 모델/Adam 유한성과 교사동일성 통과. 원시로그/캐시/명령prefix 고정.
- [상세 방법·결과·실행](DIRECTIONAL_STABILITY_V15.md). 기본 모델 변경 및 원격 Git 없음.

## v16 — 지면 접촉 조건부 발 미끄러짐 비용 (2026-09-23)

- Miki2022·Aractingi2023의 접촉 발속도 비용을 Ant에 맞춰 적용했다. 네 발 각각
  terrain mesh와 평지 collision plane의 필터된 법선력을 측정하고, 접촉으로
  감지된 발끝의 world-XY 속도 제곱만 최대1로 잘라 벌점으로 더했다.
  이는 실제 접촉점 미끄러짐이나 마찰력을 직접 측정하는 것은 아니다.
- v15의 **direction 비용 없는 control 최종249**에서 센서가 동일한 control/slip을
  seed48·geometry98에 각250iteration/3,276만transition 재학습했다. 기존91D
  관측·깊이 히스토리 전환기·v5 교사·자세 보상은 유지하고, 새 접촉 값은 정책
  관측에 추가하지 않았다.
- 새지도99/62·100/63, 여섯 제어기의16초2,100회와64초120회, 총24파일·2,220
  첫 episode를 고정 평가했다. 16초 단독 control→slip: 험지1타일279→275,
  6타일176→163, 낙상20→19, 레인 이탈1→7(각300회), 평지속도
  10.8072→10.7073m/s로 전체 **FAIL**. History는6타일179→172/300으로 **FAIL**;
  평지 원시 분기는 환경별로 정확히 동일했다.
- 별도64초 최고난도 돌다리: 단독6타일8→8/20·낙상7→8/20,
  History6타일7→6/20·레인 이탈1→4/20으로 두 기준 모두 **FAIL**.
  단독 접촉 발끝 속도는16초에서1.0512→1.0370m/s로 조금 낮았지만,
  접촉 조건부 비용0.1046→0.1054와 통과율은 개선되지 않았다. 다른 방문
  상태에서의 진단 평균이므로 실패 원인이나 동일 상태 인과효과로 단정하지 않는다.
- 독립 원시 감사가 험지300+평지50 분모, 24파일/2,220회, 평지 원시 동일성,
  초기 상태·RNG 짝, 480개 보류 지형 캐시와 source/model SHA를 확인했다.
  884 CPU 테스트·GPU 개발64/4,096환경 짝 검증 통과. **기본 모델 교체 없음**.
  [상세 방법·결과·재현](CONTACT_SLIP_V16.md).

## v17 — 학습 진도 기반 지형 커리큘럼 (2026-09-23)

- Li·Li·Hutter(2026)의 signed per-task learning-progress sampling을 Ant
  지형별 reset 확률에 적용했다. 양쪽 모두 같은 reset 재샘플링을 사용하므로
  `fixed`↔`lp`와 `history_fixed`↔`history_lp`만 인과 비교다.
  v16 무접촉비용 control 최종249에서 시작하여 geometry101/seed49, 군당
  250iteration·3,276만 전이를 동일하게 학습했다. 두 군 모두 기본1024
  episode 구간27회가 기록됐다.
- 새 지도102/64·103/65의16초2,100회+64초120회=24파일·2,220 첫 episode.
  주평가 단독 6타일163→159/300·낙상21→27/300, History 6타일166→162/300·
  낙상20→24/300으로 두 승격 모두 **FAIL**. 별도64초 돌다리에서 단독
  6타일3→7/20은 개선됐지만 History는12→8/20으로 악화했고, 주평가를
  대신하지 않는다. 징검다리 학습 점유율35.81%→30.95%는 관측 사실이나
  실패의 확정 원인은 아니다.
- 첫 개발용 120iteration smoke가 최소 지형 표본 조건에서 1구간만 생성한
  사실을 보존하고, 본학습·보류 평가 전 두 군 공통 300iteration으로 연장했다.
  평가기 v16-control 물리·초기 상태 일치 및 여섯 제어기 smoke 통과.
  **기본 모델 교체 없음**. [상세 방법·결과·재현](LEARNING_PROGRESS_V17.md).

## v18 — 깊이 조건부 교사 스타일 prior (2026-09-23)

- [CaT(2024)](https://arxiv.org/html/2403.18765v1)의 평지에서만 스타일 제약을
  적용한다는 원리를 frozen-v5 행동 평균 prior에 제한 적용했다. CaT 확률적 종료는
  구현하지 않았다. 기존825-ray 높이 스캔의 전방 ROI로 학습 전용 이진 마스크를
  구하며 actor/critic는 여전히91D다.
- v16 무접촉비용 control 최종249에서 seed50/geometry104, `always`와 `gated`
  각250iteration·32,768,000전이를 짝학습했다. 평가 전 고정한 새 지도105/66·
  106/67에서 여섯 제어기16초2,100회+64초120회=24파일·2,220첫 episode를
  평가했다. 독립 원시 감사가 분모·지형/레벨·초기 짝·평지 분기 동일성을 검증했다.
- 16초 단독 `always`→`gated`: 험지1타일280→273, 6타일162→154, 낙상
  19→18, 레인 이탈1→9/300, 평지 속도10.3253→10.5984m/s. 히스토리는
  6타일163→156, 낙상20→26, 이탈1→3/300. **두 주평가 모두 FAIL**.
  64초 최고난도 징검다리도 단독 이탈2→8/20, 히스토리2→9/20으로 FAIL이다.
- 평지 마스크 분율100%와 rough 계열별 약10~78%는 탐지 작동 진단일 뿐,
  실패 원인을 확정하지 않는다. 한 학습 seed·지도 두 개·이상적 ray 조건이므로
  보편성/실물 안전성을 주장하지 않는다. **기본 모델·전환 변경 없음**.
  [상세 방법·결과·실행](TERRAIN_STYLE_V18.md).

## v19 — 공통 신규 지도에서 고정 정책 재평가 (2026-09-28)

- 이전 버전별 지도 차이를 없애기 위해 학습 없이 v5, 원래history, v16 control,
  v5/v16 history를 같은 scene에서 비교했다. geometry/reset107/68·108/69·109/70,
  16초2,100회+별도64초120회, 총24파일2,220첫 episode. 모델/전환조건 고정.
- 16초 단독 v5→v16 control: 1타일375→415,6타일209→265,낙상49→31,
  레인이탈5→4/450, 평지속도9.4624→11.1483m/s. History는6타일225→265,
  낙상41→39,이탈4→2/450. 두 합산 서술적 gate PASS지만 지도별 회귀가 있다.
- 64초 단독6타일0→16/30이나 이탈0→4라FAIL. History6타일13→14/30,
  낙상11·이탈3 동일로합산PASS지만 세지도 모두개별gateFAIL, 짝 개선/악화8/7.
  총합 개선을 모든 지도·장시간의 일관된 우월성으로 해석하지 않는다.
- 개발 첫cold-cache관측불일치를 보존하고, 새지도를 점수화하기 전 초기화전용
  cache준비를 추가했다. 네제어기초기state/RNG·v5/history평지원시결과정확동일,
  개발reference동일성/173기존소스·16모델SHA/교사tensor동일성을검증했다.
- 전체1037CPUtests와boundedMypy/compileall/AST/diff통과. Pyflakesunused-import
  경고9개와미설치Ruff/Pyright는명시. 원문확인은CaT/Miki/Aractingi/Agarwal,
  이번은그논문전체재현이아닌평가전용재기준화. 기본모델/의존성/원격변경없음.
- [상세 결과·재현·한계](REBASELINE_V19.md), [문헌 근거](REBASELINE_V19_REFERENCES.md).

## v20 — 기존 접촉 비용의 즉시/점진 도입 (2026-09-28)

- Aractingi2023의벌점선형상승원리를기존Ant slip항하나의warm-start에제한적용했다.
  전체논문재현이아니며첫4,000step상승은사전공학적선택이다. 계수step합8,000대
  6,000으로시점과노출량이동시에달라순수도입시점효과를주장하지않는다.
- v16 control249에서두arm각250iter/32,768,000전이. 새로운geometry/reset111/71·
  112/72의여섯제어기16초2,100회+별도64초120회로총24파일2,220첫episode평가.
  단독/unchanged-history에서두후보비교와각각부모비교를고정했다.
- 16초즉시→점진:one273→273/six158→164/낙상26→21/이탈1→6(각험지300).
  History six167→172/낙상27→22/이탈1→2. 둘다이탈로FAIL. 부모six178/172에
  비해전체개선이없다. 지도111history두후보비교만PASS, 지도112여섯gate모두FAIL.
- 64초즉시→점진:six7→5/낙상9→10/이탈3→6(각20). History six5동일이나이탈1→9.
  여섯합산및모든지도별gateFAIL. 기본모델·전환을유지하며실패를포함한전체원시증거보존.
- 개발/학습cache를초기화만한뒤고정했고GPU실행재시도없음. 학습reward시점clock/
  원시·가중벌점/dt합과실제초기episodehorizon대입을검증했다. 두군에만새proxy수정을
  적용한것으로,이전v16의학습궤적재현주장은아니다. 무비용동일예산재학습대조군이
  없으므로부모대비회귀를비용만의인과효과로분리하지않는다.
- 독립원시감사와CPU교사/Adam검증,1145tests, 새파일pyflakes/제한적mypy/AST/
  compileall/diff통과. 기존181source·16model및새14실험정의동결보존.
  [상세기록](CONTACT_CURRICULUM_V20.md). 새의존성/commit/원격작업/기본정책교체없음.

## v22 — 무비용 추가학습의 seed 민감도 (2026-09-28)

- Henderson2018·Agarwal2021 원문을 참고해 v21의 한-seed 한계를 보완했다.
  같은 v16 부모·초기weights·훈련geometry110에서 seed51/52/53만 바꾸고 각각
  32,768,000전이를 새로 학습했다. 개발 포함 새 환경 학습 총99,139,584전이다.
- seed51 전체 모델/Adam/metadata와 기록된 보상/접촉·33비시간scalar가 과거v21과
  정확히 재현된 뒤52/53을 진행했다. 과거51과 재현51을 독립 두 시행으로 세지 않는다.
  초기/실제horizon의 비교는 각자 같은seed capacity기록에 대응시켰다.
- 신규115/75·116/76,8제어기32파일2,960첫episode. 16초험지6타일은 부모178,
  seed51/52/53=165/164/171; History부모177,각172/168/169(제어기별300회).
  낙상 감소와 별개로12합산gate 모두FAIL이며 여섯 부모비교는 두지도 모두FAIL이다.
- 별도64초돌다리6타일은 부모7,각8/17/9; History부모7,각8/4/12(각20회).
  6개합산PASS 중 seed53대부모만 지도116에서FAIL,나머지5개는 두지도PASS다.
  단독seed52의이득이 같은seed의History에 이어지지 않아 조건별 차이가 크다.
- 모든seed·지도·지형/난이도·짝결과·range/부모차이를 보존했다. 부모를3회복제하지
  않고 episode/history를독립학습반복으로세지 않는다. 한부모·훈련지형조건부3seed
  서술적 결과로 통계적우월성·일반강건성·회귀인과성은 주장하지 않는다.
- 53GPU명령 전부성공·재시도0;1,501CPUtests와새13Python정적검사 통과,
  별도tensor/scalar검사 및 독립32파일 원시감사로 검증했다. 기존213source·19model과
  새12실험정의를 유지하고 기본정책/의존성/commit/원격작업 변경없이 종료했다.
  [상세기록·논문과의경계](SEED_CONTINUATION_V22.md).

## v23 — 같은 첫 episode의 평가 시간 분리 (2026-09-28)

- v22의16초mixed175env와64초돌다리10env는 시간 외 조건도 달랐다. 새 학습이나
  유리한seed선택 대신 같은64초주행에서16/64초를 기록하는 사전 고정 평가를 했다.
- 16초가상cutoff는 물리/history/RNG를reset하지 않는다. 개발8prefix+2무계측64초대조의
  원시35필드 정확일치와 직렬화전후상태 동등성을 확인하고 신규지도117/118을 평가했다.
- 주요16초 six부모169/seed51=161/52=150/53=160,History부모174/51=167/52=155/53=170.
  모든부모비교FAIL;History53대51/52 두 비교만합산·두지도모두PASS다.
- 동일episode64초 six부모225/51=226/52=223/53=214,History부모220/51=241/52=227/53=239.
  History51/52/53대부모+53대52 합산PASS지만 두지도모두PASS는History51/53대부모뿐이다.
- 본평가16bundle·2,800첫episode·5,600의존window,개발350/630이며 새학습0이다.
  평가window/history조합을 독립학습반복으로 부풀리지 않는다. 부모는 늦은 성공72·실패16,
  History51은79·5로 변화 방향을 따로 기록했다. 실제64초timeout은 낙상이 아니다.
- GPU29명령 전부성공·실패/재시도0. 전체1,673CPUtests·새Python9개 정적검사와
  독립stdlib원시감사 통과, 기존227source·22model/새8실험정의 불변. 기본정책 유지,
  새의존성/commit/원격작업/실물변경 없이 고정분기 종료. [상세기록](PAIRED_HORIZON_V23.md).

## v24 — 고정 전역 학습률의 추가학습 민감도 (2026-09-28)

- 같은 부모/초기std0.2/빈Adam/훈련지형/3seed에서 GLOBAL PPO LR만1e-4→1e-5로
  바꿨다. 논문은 LR민감도의 동기이며 이번1e-5나 개선을 보장하지 않는다.
- high51소형·세seed각capacity·high51전체가 v22 model/Adam/metadata/reward/contact와
  33비시간scalar를 정확재현했다. low각final249·5,000Adamstep의LR1e-5와teacher불변,
  실제YAML전체/초기상태/RNG/horizon을 독립CPU검사와 별도 검토로 확인했다.
- 학습13+평가39 GPU명령 모두성공·실패/재시도0, 새132,677,632학습전이와
  28본평가bundle/4,900첫episode/9,800의존window를 완료했다. 개발280/560은 별도다.
- 16초 low단독6타일168/157/166은 high161/150/157보다 높지만 부모171보다 낮다.
  History low167/164/171도 이탈2회가 high0/부모1보다 높아 개선gate를 통과하지못한다.
  low/high 및 low/부모의 각6합산gate는 양window모두FAIL이다.
- 64초 low−high 단독6타일은−11/−15/+24,History는+1/−3/−7로 조건/seed에 따라 다르다.
  사전18비교의16초합산PASS1개는 기존History high51대부모이며 두지도일관PASS는없다.
  64초합산PASS3개는 기존History high3대부모, 두지도모두PASS는 high53만이다.
- 늦은 실패 원인flag의 중복도 보고했다(low53:23회중새낙상13+레인11의1회중복).
  전체237legacy source·22model과12새동결정의를 보존했다. **기본정책 유지**,
  유리한seed선별/시간window독립표본화/전역LR을actor만의효과로 해석하지않는다.
  [상세결과·문헌·감사](LR_CONTINUATION_V24.md).

## 남은 한계와 다음 해석의 경계

1. **실제 인지 문제는 아직 남아 있다.** 이상적 높이 ray와 휴리스틱 발 후보,
   v16의 시뮬레이터 접촉 센서는 실제 RGB-D, 캘리브레이션된 foot support 판정,
   IK/whole-body 계획을 대체하지 않는다.
2. **지속 극복이 미해결이다.** 최고 난이도 돌에서 짧은 1타일 성과가 있어도 6타일과
   64초 낙상은 여전히 약하다.
3. **보상·prior는 안전 제약이 아니다.** v9 support shaping과 v10 mean prior는
   접촉 안정성 또는 레인 유지의 보증이 아니다. v10 단독 정책은 teacher를 추론에
   사용하지 않고, v11/v12에서 명시적으로 v5/v10 행동을 선택·혼합한다.
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
| v11 switch | [HYBRID_V11](HYBRID_V11.md) | [16초·64초 요약](../artifacts/terrain_demo/hybrid_v11/attempt02/summary.md), [독립 감사](../artifacts/terrain_demo/hybrid_v11/attempt02/independent_audit_numerical.json), [비교 화면](../artifacts/terrain_demo/hybrid_v11/index.html) |
| v12 history switch | [HISTORY_V12](HISTORY_V12.md) | [16초·64초 요약](../artifacts/terrain_demo/history_v12/attempt02/summary.md), [독립 감사](../artifacts/terrain_demo/history_v12/attempt02/independent_review_final.json) |
| v13 adaptive posture | [ADAPTIVE_POSTURE_V13](ADAPTIVE_POSTURE_V13.md) | [16초·64초 요약](../artifacts/terrain_demo/adaptive_posture_v13/summary.md), [훈련검증](../artifacts/terrain_demo/adaptive_posture_v13/training_validation.json), [독립최종감사](../artifacts/terrain_demo/adaptive_posture_v13/final_review.md) |
| v14 command conditioning | [COMMAND_CONDITIONING_V14](COMMAND_CONDITIONING_V14.md) | [16초·64초 요약](../artifacts/terrain_demo/command_conditioning_v14/summary.md), [학습검증](../artifacts/terrain_demo/command_conditioning_v14/training_validation.json), [독립최종감사](../artifacts/terrain_demo/command_conditioning_v14/final_review.md) |
| v15 directional stability | [DIRECTIONAL_STABILITY_V15](DIRECTIONAL_STABILITY_V15.md) | [16초·64초 요약](../artifacts/terrain_demo/directional_stability_v15/summary.md), [학습검증](../artifacts/terrain_demo/directional_stability_v15/training_validation.json), [독립최종감사](../artifacts/terrain_demo/directional_stability_v15/final_review_rough_supplement.md) |
| v16 contact slip | [CONTACT_SLIP_V16](CONTACT_SLIP_V16.md) | [16초·64초 요약](../artifacts/terrain_demo/contact_slip_v16/summary.md), [학습검증](../artifacts/terrain_demo/contact_slip_v16/training_validation.json), [독립원시감사](../artifacts/terrain_demo/contact_slip_v16/independent_final_audit.md) |
| v17 learning progress | [LEARNING_PROGRESS_V17](LEARNING_PROGRESS_V17.md) | [16초·64초 요약](../artifacts/terrain_demo/learning_progress_v17/summary.md), [학습검증](../artifacts/terrain_demo/learning_progress_v17/training_validation.json), [독립원시감사](../artifacts/terrain_demo/learning_progress_v17/independent_final_audit.md) |
| v18 terrain style | [TERRAIN_STYLE_V18](TERRAIN_STYLE_V18.md) | [16초·64초 요약](../artifacts/terrain_demo/terrain_style_v18/summary.md), [학습검증](../artifacts/terrain_demo/terrain_style_v18/training_validation.json), [독립원시감사](../artifacts/terrain_demo/terrain_style_v18/independent_raw_audit.json) |
| v19 common-map rebaseline | [REBASELINE_V19](REBASELINE_V19.md) | [요약](../artifacts/terrain_demo/rebaseline_v19/summary.md), [독립 최종 감사](../artifacts/terrain_demo/rebaseline_v19/independent_final_audit.json) |
| v20 contact curriculum | [CONTACT_CURRICULUM_V20](CONTACT_CURRICULUM_V20.md) | [요약](../artifacts/terrain_demo/contact_curriculum_v20/summary.md), [학습검증](../artifacts/terrain_demo/contact_curriculum_v20/training_validation.json), [독립 원시 감사](../artifacts/terrain_demo/contact_curriculum_v20/independent_raw_audit.json) |
| v21 no-cost continuation | [CONTACT_CONTINUATION_V21](CONTACT_CONTINUATION_V21.md) | [요약](../artifacts/terrain_demo/contact_continuation_v21/summary.md), [학습검증](../artifacts/terrain_demo/contact_continuation_v21/training_validation.json), [독립 원시 감사](../artifacts/terrain_demo/contact_continuation_v21/independent_raw_audit.json) |
| v22 continuation seeds | [SEED_CONTINUATION_V22](SEED_CONTINUATION_V22.md) | [요약](../artifacts/terrain_demo/seed_continuation_v22/summary.md), [학습검증](../artifacts/terrain_demo/seed_continuation_v22/training_validation.json), [독립 원시 감사](../artifacts/terrain_demo/seed_continuation_v22/independent_raw_audit.json) |
| v23 paired horizons | [PAIRED_HORIZON_V23](PAIRED_HORIZON_V23.md) | [요약](../artifacts/terrain_demo/paired_horizon_v23/summary.md), [개발 재현](../artifacts/terrain_demo/paired_horizon_v23/development.json), [독립 원시 감사](../artifacts/terrain_demo/paired_horizon_v23/independent_raw_audit.json) |
| v24 global LR continuation | [LR_CONTINUATION_V24](LR_CONTINUATION_V24.md) | [요약](../artifacts/terrain_demo/lr_continuation_v24/summary.md), [독립 학습 감사](../artifacts/terrain_demo/lr_continuation_v24/independent_training_tensor_audit.json), [독립 원시 감사](../artifacts/terrain_demo/lr_continuation_v24/independent_raw_audit.json) |

초기 v0–v4의 과제 설정, 경로 및 수치는 [README](../README.md)와
[`artifacts/terrain_demo/README.md`](../artifacts/terrain_demo/README.md)에 보존한다.
