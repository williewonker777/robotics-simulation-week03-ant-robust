# v23 — 같은 첫 episode의 16초·64초 짝 평가

## 결론: 평가 시간에 따라 판정은 바뀌지만 기본 정책은 유지

새 학습 없이 **16물리 실행·2,800첫 episode의 16초/64초 짝 관측 5,600건**을
완료했다. 같은 지형·출발점·정책의 같은 주행에서도 평가 시간에 따라 상대 비교
판정이 바뀌었다. 16초에서는 여섯 부모 대비 비교가 모두 FAIL이지만, 64초에서는
History seed51·53 대 부모가 두 지도 모두 PASS다. **64초 이득으로 주요 16초
회귀를 상쇄하지 않으며 기본 정책을 바꾸지 않는다.**

### 16초 주요평가 — 제어기별 험지 300회·평지 50회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모(v16 control) |267|169|31|2|1|11.2070|
| seed51 |276|161|23|1|1|10.7647|
| seed52 |266|150|29|3|1|11.0517|
| seed53 |269|160|26|4|1|10.9844|
| History 부모 |270|174|28|2|3|9.2590|
| History seed51 |269|167|29|2|3|9.2590|
| History seed52 |268|155|27|4|3|9.2590|
| History seed53 |277|170|22|1|3|9.2590|

- 단독 세 seed의 6타일은 **150~161/300**, 부모 169보다 8~19회 낮다.
  History는 **155~170/300**, 부모 174보다 4~19회 낮다.
- 여섯 부모 대비 gate는 합산과 두 지도 각각에서 모두 FAIL이다.
  사전 12개 중 합산 PASS는 **History seed53 대51·52** 두 개뿐이며 두 지도 모두 PASS다.
  seed53도 부모 대비 6타일이 낮으므로 이를 부모 대비 개선으로 바꾸어 주장하지 않는다.

### 64초 보조평가 — 위와 같은 첫 episode·같은 분모

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모(v16 control) |225|225|69|7|2|12.7995|
| seed51 |230|226|58|15|7|12.1861|
| seed52 |224|223|60|15|1|12.8460|
| seed53 |218|214|60|23|3|12.6050|
| History 부모 |222|220|63|15|8|10.4378|
| History seed51 |242|241|50|8|8|10.4378|
| History seed52 |230|227|58|14|8|10.4378|
| History seed53 |241|239|51|10|8|10.4378|

- 합산 PASS 4개는 **History seed51·52·53 대 부모, History seed53 대52**다.
  History seed51·53 대 부모만 두 지도 모두 PASS다.
- **History seed52 대 부모는 합산 PASS지만 두 지도 각각에서는 FAIL**이다.
  History seed53 대52도 지도118에서만 PASS다. 합산 판정만으로 지도별 일관성을
  주장하지 않는다. 지도118의 History seed53 대51은 PASS지만 합산은 FAIL이다.
- 단독 정책의 부모 대비 gate는 세 seed 모두 FAIL이다. seed51의 6타일은 225→226으로
  1회 늘지만 레인 이탈 7→15와 평지 낙상 2→7 등 다른 조건이 악화했다.
- 64초 단독 seed의 6타일 범위는 **214~226/300**, History는 **227~241/300**이다.
  두 window 모두 world 이탈과 평지 레인 이탈은 0이며, 네 History의 평지 원시는 같다.

### 같은 행의 16→64초 변화 — 험지 300회씩

두 시점의 성공 횟수를 빼는 것만으로는 늦은 성공과 늦은 실패가 상쇄된다.
아래 네 칸은 같은 episode의 두 판정을 연결한 것이며 합은 행마다 300이다.

| 제어기 | 둘 다 실패 | 늦은 성공 | 늦은 실패 | 둘 다 성공 | (16,64]초 최초 6타일 도달 |
|---|---:|---:|---:|---:|---:|
| 부모(v16 control) |59|72|16|153|92|
| seed51 |59|80|15|146|105|
| seed52 |69|81|8|142|116|
| seed53 |65|75|21|139|96|
| History 부모 |65|61|15|159|89|
| History seed51 |54|79|5|162|98|
| History seed52 |64|81|9|146|106|
| History seed53 |43|87|18|152|102|

- 부모는 늦은 성공 **72회**와 늦은 실패 **16회**가 함께 발생해 169→225회가 됐다.
  늦은 실패 16회는 새 낙상 14회·새 레인 이탈 2회다.
- History seed51은 늦은 성공 **79회**·늦은 실패 **5회**로 167→241회가 됐다.
  History 부모는 각각 61회·15회로 174→220회다. 이는 이 고정 표본의 서술적
  비교이며 특정 학습 원인의 인과적 설명이나 새로운 seed 선택 근거가 아니다.
- 제어기별 늦은 실패의 새 낙상/레인 이탈은 위 순서대로
  **14/2, 9/6, 7/1, 9/12, 11/4, 5/0, 7/2, 12/6**회다.
  이 표본의 늦은 실패에는 새 world 이탈과 마지막 거리 미달 flag가 없었다.
  일반적으로 flag는 겹칠 수 있으며 64초 timeout을 낙상으로 세지 않는다.
- 마지막 열은 단순 도달 시각이다. 이후 낙상·이탈이 있으면 엄격한 성공은 아니므로
  늦은 성공 열과 같지 않다. 최대 거리 또는 한 번의 도달로 성공 기준을 바꾸지 않았다.
- 최고난도 돌다리 부분집합은 제어기별 10회다. 16초 6타일은 모두 0회이고,
  64초 부모/seed51/52/53은 **5/3/3/3**, History 대응은 **1/5/1/3**회다.
  v22의 다른 지도·출발점·20회 분모와 같은 실험인 것처럼 합치거나 직접 순위를 매기지 않는다.

[전체 합산·지도별 12비교](../artifacts/terrain_demo/paired_horizon_v23/summary.md),
[지형·난이도·짝 변화·seed 범위 전체](../artifacts/terrain_demo/paired_horizon_v23/summary.json),
[독립 원시 감사](../artifacts/terrain_demo/paired_horizon_v23/independent_raw_audit.json).

## 질문과 문헌 근거

[v22](SEED_CONTINUATION_V22.md)의 16초 평가는 혼합 지형 175환경, 64초 평가는
최고난도 돌다리 10환경이었다. 시간뿐 아니라 지형·분모·출발점도 달라서 두 결과의
차이를 시간의 효과로 단정할 수 없다. v23은 **같은 첫 episode의 궤적을 16초와
64초에서 관측**한다. 새 학습이나 유리한 seed 선택에 앞서 평가 시간의 영향을
구분하는 진단이며, 학습률·보상·정책·전환 gate를 조정하는 실험이 아니다.

- [Pardo et al., *Time Limits in Reinforcement Learning*, ICML 2018, §§2–3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf):
  과제 자체의 기한과 외부에서 끊는 관측 시간을 구분한다. 짧은 기한에서 좋았던
  행동이 이후에도 유지되는지 따로 확인해야 한다는 동기다. 주로 학습과 bootstrap을
  다루는 논문이며, 이번 짝 평가 추정량의 증명이나 v22의 버그를 뜻하지 않는다.
- [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf):
  학습 실행과 평가 episode를 구분하고 적은 시행의 불확실성을 고려해야 한다.
  같은 궤적의 두 시점을 독립 표본으로 세거나 유리한 seed만 보고하지 않는다.

[원문 확인 기록](../artifacts/terrain_demo/paired_horizon_v23/reference_check.json)을 남겼다.
정확한 16/64초, 세 seed, 두 지도, 성공 gate는 사전 고정한 공학적 선택이지
논문이 보장하는 표본 수나 성능 조건이 아니다. 학습률 또는 초기 std 변경은 보류했다.

## 결과를 보기 전에 고정한 조건

[사전 계획](experiment_plans/paired_horizon_v23.md)은 GPU 실행 전에 작성·검토하고
바이트 동일 사본과 SHA를 [계획 manifest](experiment_plans/manifest.json)에 기록했다.

- 부모는 v16 control final249, 추가학습 모델은 v22 seed51/52/53 final249다.
  각 단독 정책과 기존 v12 history 조합을 모두 사용한다. 총 4모델·8제어기다.
- 새로운 학습·optimizer step·모델 복사·checkpoint 선택은 **0**이다.
  기존 227개 소스와 22개 checkpoint 경로의 바이트를 보존한다.
- 신규 geometry/reset은 **117/77, 118/78**이다. 각 실행은 7지형 × 5난이도 ×
  5출발점 = 175환경의 혼합 지형이다. 모든 제어기는 지도별 초기 물리 상태·관측·
  CPU/CUDA RNG가 같아야 하며 네 history의 평지 원시 결과도 같아야 한다.
- 물리 episode timeout은 64초, dt는 1/60초, 최대 3,840제어 step이다.
  16초(960step)는 **점수용 가상 cutoff**일 뿐 simulator/history/RNG를 reset하지 않는다.
- 본평가는 **16물리 bundle·2,800첫 episode·5,600의존적인 window 관측**이다.
  5,600개의 독립 episode가 아니다. 제어기별·window별 험지 300회와 평지 50회이며
  부모를 seed마다 복제하지 않는다. 최고난도 돌다리 부분집합은 각각 10회로,
  v22의 별도 돌다리 20회와 같지 않다.

## 비개입 기록과 점수 의미

기존 action → sensor → environment step → 실제 done에 따른 history reset 순서를
유지한다. 64초 tracker만 첫 episode의 활성 마스크를 결정한다. 16초 tracker는
같은 관측을 받아 점수를 닫고, 깊은 JSON 사본으로 결과·접촉·자세·전환 event를
저장한다. 가상 cutoff와 실제 종료가 겹치면 실제 reset 이전 `last_*` 증거가 우선한다.
일찍 끝난 episode를 reset 후 episode로 바꾸거나 분모에서 빼지 않는다.

엄격한 1/6타일 성공은 **각 window의 마지막 전진 거리 ≥13.1/53.1m**이며,
그때까지 첫 episode 낙상·발자국 레인 이탈·world 이탈이 없어야 한다.
최대 거리나 한 번 문턱을 밟은 사실만으로 성공으로 세지 않는다. 실제 tensor 비교의
float32 문턱을 감사에서도 그대로 재현한다.

- 16→64초의 짝 6타일 결과를 둘 다 실패·늦은 성공·늦은 실패·둘 다 성공으로 나눈다.
- 늦은 실패에는 새 낙상, 새 레인 이탈, 새 world 이탈, 마지막 거리 53.1m 미만의
  겹칠 수 있는 flag를 보고한다. 64초 timeout은 물리적 완료이지만 낙상이 아니다.
- (16,64]초의 최초 6타일 도달 횟수는 엄격한 성공과 별도로 기록한다.
- 기존 보조 거리 snapshot은 16초 window 안에서는 8초, 64초 안에서는 16초다.
  `distance_at_snapshot_m`과 명시적인 `snapshot_seconds`로 구별한다.
  v22도 이미 이렇게 기록했으며 기존 필드의 잘못된 이름을 고치는 실험이 아니다.
- 중간 직렬화 전후 root/joint/episode counter/관측/정책/전체 history 상태와
  CPU/CUDA RNG 해시가 같아야 한다. 이 증거는 지정된 상태와 기록 시점의 수동성을
  검사하며, 기록하지 않은 모든 내부 상태나 다른 하드웨어의 결정성을 보장하지 않는다.

## 본평가 전에 통과해야 하는 개발 관문

개발 geometry51/reset24·35환경에서 먼저 cache 준비만 하고,
부모/history 부모의 16초 기록을 끈 64초 대조 실행 2회와 짝 기록 8회를 수행한다.

1. 여덟 짝 실행의 16초 원시 결과가 기존 v22 개발 결과의 **35개 필드 모두 정확히
   동일**해야 한다. 성능 값뿐 아니라 초기 상태/RNG, condition, 전환·접촉·자세를 포함한다.
2. 부모/history 부모의 64초 결과는 기록을 끈 대조 실행과 35개 필드가 정확히 같아야 한다.
3. 같은 실행의 직렬화 전후 상태 해시와 모든 제어기의 초기 짝/평지 분기를 검증한다.
   과거 16초 timeout 뒤의 reset 상태를 새 64초 계속 주행 상태와 비교하지 않는다.

개발은 350첫 episode(짝 280 + 대조 70), 630window 관측이며 본평가 성과가 아니다.
CPU·독립 소스 검토·정확 개발 재현을 통과한 뒤 8개 실험 정의·모델·개발 증거를
동결하고 신규 지형을 scoring 없이 준비해 cache와 명령 ledger를 고정한다.
계획 GPU 명령은 **29회**(개발 준비1 + 대조2 + 짝개발8 + 신규준비2 + 본평가16)다.
실패 시 관문을 느슨하게 하거나 본평가 결과를 보고 재시도하지 않는다.

### 완료한 개발 검증

여덟 제어기의 16초 기록과 두 제어기의 64초 대조 기록이 각각 35개 필드에서
정확히 일치했다. 모든 초기 물리/RNG 짝, history 평지 분기, 직렬화 전후 상태
해시도 통과했다. 개발 GPU 11명령 모두 성공했고 실패·재시도는 없었다.
[개발 원시 연결](../artifacts/terrain_demo/paired_horizon_v23/development.json),
[독립 표준 라이브러리 재검사](../artifacts/terrain_demo/paired_horizon_v23/independent_development_audit.json),
[동결 manifest](../artifacts/terrain_demo/paired_horizon_v23/frozen.json).

GPU 전에 전체 CPU 1,588개와 신규 87개 재검사를 통과했다. 새 Python 7개의
pyflakes·제한된 mypy·compile·공백 검사도 통과했다. 초기 public reexport lint
경고는 GPU 전에 수정하고 로그를 보존했다.
[사전 CPU/정적 검사](../artifacts/terrain_demo/paired_horizon_v23/pre_gpu_validation.json).
독립 CPU checkpoint 검사에서도 네 모델의 바이트·유한 tensor와 각 8개 교사 tensor가
기존 증거와 일치했다. [모델 감사](../artifacts/terrain_demo/paired_horizon_v23/independent_policy_audit.json).

전담 reviewer 새 실행/재개가 runtime thread 제한으로 거절되어, 기존 두 executor가
자신이 작성하지 않은 상대 파일을 독립 read-only 검토했다. 이를 전담 code-reviewer
역할의 승인으로 표시하지 않는다. [구현 검토 기록](../artifacts/terrain_demo/paired_horizon_v23/implementation_review.json).
실험 정의 밖의 독립 원시 감사기는 별도 검토에서 개발 대조군 교환과 모순된 최초
도달 시간의 검증 누락을 고쳤고, 85개 회귀 테스트와 재검토를 통과했다.
GPU 실험 정의를 바꾸거나 결과를 재시도한 수정이 아니다.
[감사기 검토 기록](../artifacts/terrain_demo/paired_horizon_v23/auditor_review.json).

## 비교와 해석 한계

각 window에서 세 seed 대 부모, seed52 대51, seed53 대51, seed53 대52와 각각의
history 대응 비교까지 총 12방향을 모두 보고한다. 양쪽 window 모두 혼합 지형이므로
험지 1/6타일 비감소·낙상/이탈 비증가·world 0과 평지 낙상/이탈 비증가·평균 episode
속도 비감소, 적어도 하나의 엄격한 험지 개선이라는 기존 gate를 적용한다.
부분집합·지도별·지형/난이도별 결과와 seed 범위·부모 차이를 함께 보존한다.

16초가 주요평가이고 64초는 서술적 보조평가다. 64초의 이득으로 16초 회귀를
상쇄하거나 결과가 좋은 모델을 자동 승격하지 않는다. 한 부모·한 훈련 지형·3학습 seed·
2평가지도 조건부 결과이며, 두 window는 의존적이다. 다른 플랫폼, 실물 로봇 안전,
일반 강건성, 무한 시간 안전이나 통계적 우월성을 주장하지 않는다.


## 최종 실행·검증

- 계획한 GPU 29명령 모두 성공, 실패·재시도 0이다. 개발 350첫 episode/630window와
  본평가 2,800첫 episode/5,600의존 window를 분리해 기록했다. 새 학습은 0이다.
- 표준 라이브러리 전용 감사가 16원시 bundle의 32window, 모든 합산·지도/지형/난이도
  비교, seed 범위·부모 차이, 제어기/지도별 49개 시간 변화 부분집합을 재계산해 보고서와
  일치함을 확인했다. 명령 순서·로그·동결·480cache tile·원시 증거의 해시도 검증했다.
- 전체 **1,673개 CPU 테스트 통과**(31.885초), 새 Python 9개 pyflakes·제한된 mypy·
  AST·compile·공백 검사 통과. 실험 정의 8개와 기존 227source/22checkpoint는 불변이다.
- 기본 정책·의존성·실물 로봇 설정을 바꾸지 않았고 commit/원격 Git·GitHub 작업을 하지
  않았다. 이 사전 고정 분기는 결과와 무관하게 종료한다. 기존 별도 데모·게시 업무는
  TASK298의 다른 범위이며, 이 실험의 미완료 실행을 뜻하지 않는다.
