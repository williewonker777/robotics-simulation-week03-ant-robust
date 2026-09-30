# v22 — 무비용 추가학습의 학습 seed 민감도

## 결론: 세 seed 모두 주평가 회귀, 장시간 돌다리는 큰 seed 차이

새 두 지도에서 **32파일·2,960첫episode**를 완료했다. 16초 주요평가의
사전 12개 합산 gate는 **모두 FAIL**이다. 세 seed 모두 자신의 부모보다
6타일 통과가 낮았고, 이는 단독/history와 두 지도 각각에서도 관찰됐다.
반면 별도 64초 돌다리의 12개 합산 gate 중 6개는 PASS다. 일부 조건의
개선을 일반적 개선으로 바꾸어 해석하지 않으며 **기존 기본 정책을 유지한다**.

### 주요 16초 — 제어기별 험지 300회·평지 50회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모(v16 control) |274|178|21|4|3|10.7931|
| seed51 |284|165|12|4|4|10.3866|
| seed52 |276|164|18|5|3|10.5814|
| seed53 |273|171|21|5|3|10.7561|
| History 부모 |277|177|21|2|4|9.0406|
| History seed51 |277|172|23|0|4|9.0406|
| History seed52 |276|168|21|1|4|9.0406|
| History seed53 |277|169|21|2|4|9.0406|

- 단독의 세 seed 6타일은 **164~171/300**, 부모 178 대비 −14~−7회다.
  History는 **168~172/300**, 부모 177 대비 −9~−5회다.
  범위는 각 seed의 별도 결과이며 부모를 세 번 복제한 분모가 아니다.
- seed51의 낙상 21→12회 감소와 1타일 274→284회 증가는 장점이지만,
  6타일 178→165회, 평지 낙상 3→4회·속도 감소 때문에 gate는 FAIL이다.
  seed52·53도 평지 평균 속도가 부모보다 낮다.
- 부모 대비 짝 6타일 개선/악화는 단독 seed51/52/53에서 각각
  15/28, 15/29, 18/25회; History는 14/19, 12/21, 11/19회다.
- 두 지도 모두에서 여섯 부모 대비 gate가 FAIL이다. 지도115의 seed 간 비교 중
  History seed53 대51·52만 PASS이며, 지도116의 12개 gate는 전부 FAIL이다.
- 모든 world 이탈과 평지 레인 이탈은 0이다. 네 History의 평지 원시 결과는 같다.

### 별도 64초 최고난도 돌다리 — 제어기별 20회

16초 분모와 합치지 않는다. 낙상과 레인 이탈 flag는 겹칠 수 있다.

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| 부모(v16 control) |7|7|10|4|
| seed51 |8|8|7|5|
| seed52 |17|17|2|1|
| seed53 |9|9|9|3|
| History 부모 |7|7|8|6|
| History seed51 |8|8|10|3|
| History seed52 |5|4|7|8|
| History seed53 |13|12|6|1|

- 단독 6타일의 seed 범위는 **8~17/20**, History는 **4~12/20**이다.
  각각 범위 폭 9회·8회로, 이 제한된 조건에서도 seed 차이가 크다.
- 단독 seed52 대 부모는 6타일 7→17, 낙상 10→2, 이탈 4→1이며
  두 지도 모두 gate PASS다. 짝 개선/악화는 합산 10/0회다.
  그러나 같은 seed52의 History는 부모 7→4회·이탈 6→8회로 FAIL이다.
  단독의 이득을 전환 제어기의 이득으로 옮겨 주장할 수 없다.
- History seed53 대 부모는 6타일 7→12, 낙상 8→6, 이탈 6→1로
  두 지도 모두 PASS다. 짝 개선/악화는 6/1회다.
- 합산 PASS 6개는 **seed52 대 부모, seed53 대 부모, History seed53 대 부모,
  seed52 대51, History seed53 대51, History seed53 대52**다.
  이 중 seed53 대 부모는 지도116에서 이탈 증가로 FAIL이며, 나머지 5개는
  두 지도 모두 PASS다. 모든 world 이탈은 0이다.

이 표본에서 **무비용 추가학습의 주평가 한계는 seed51만의 결과는 아니었지만,
장시간 돌다리의 결과까지 같은 방향으로 일반화할 수는 없었다**.
세 seed와 두 지도만으로 통계적 우월성·일반 강건성·회귀 원인의 인과성을
확정하지 않는다. 유리한 seed를 고르거나 새 기본 모델로 승격하지 않았다.

[모든 합산·지도별 12비교](../artifacts/terrain_demo/seed_continuation_v22/summary.md),
[지형·난이도·접촉/정체·짝 결과와 seed 범위](../artifacts/terrain_demo/seed_continuation_v22/summary.json),
[독립 원시 감사](../artifacts/terrain_demo/seed_continuation_v22/independent_raw_audit.json).

## 질문과 논문 근거

[v21](CONTACT_CONTINUATION_V21.md)의 무접촉비용 추가학습에서도 부모 대비
회귀와 지표 간 맞교환이 있었다. 이번에는 보상이나 PPO를 다시 조정하지 않고,
같은 부모·초기 가중치·훈련 지형에서 **추가학습 seed만 51·52·53으로 바꾼다**.
이는 한 부모와 한 훈련 지형에 조건부인 RNG 민감도 실험이지, 서로 다른 초기
정책이나 훈련 분포까지 포함하는 일반적인 강건성 검증은 아니다.

- [Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553):
  같은 하이퍼파라미터에서도 학습 seed에 따라 결과가 달라질 수 있으며 유리한
  시행만 고르는 해석을 경계한다. 특정한 보편적 최소 seed 수를 보장하지 않는다.
- [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf):
  독립 학습 실행과 평가 과제·episode를 구분하고 적은 시행의 불확실성을 강조한다.
  같은 seed라도 GPU/프레임워크의 비결정성이 남을 수 있다는 한계도 다룬다.

두 원문의 관련 절을 확인한 [문헌 기록](../artifacts/terrain_demo/seed_continuation_v22/reference_check.json)을
남겼다. **3개 seed와 정확 재현 gate는 이 실험의 공학적 선택**이며 논문이 보장한
표본 수나 결정성 조건이 아니다. 모든 seed 결과와 범위를 서술적으로 보고하며,
평가 episode·지도·history 조합을 추가적인 독립 학습 시행으로 세지 않는다.

## 결과를 보기 전에 고정한 학습 조건

[사전 계획](experiment_plans/seed_continuation_v22.md)은 GPU 작업 전에 작성했고,
12개 학습·평가 정의를 모두 본학습 전에
[동결](../artifacts/terrain_demo/seed_continuation_v22/training_frozen.json)했다.

- 부모: v16 control final249,
  SHA `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
- v21의 공통 초기 checkpoint를 새 실험 경로로 **바이트 그대로** 복사했다.
  SHA `10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd`.
  부모와 non-std tensor 32개 동일, std 0.2, 빈 Adam 1e-4, iteration 0,
  metadata 보존, frozen-v5 교사 tensor 8개 동일을 독립 CPU 검사했다.
- 기존 91D 정책·8D 행동, conditioned/anchored/targets, prior 0.02,
  물리·종료·보상 조건과 geometry 110 cache를 유지했다.
- v21의 `ContinuationContactSlipReward`를 그대로 사용한다. 센서와 원시 비용을
  계산한 뒤 적용 계수를 0으로 한다. 비용 항을 건너뛰거나 새 보상을 추가하지 않는다.
- 새 본학습 순서는 seed51 → seed52 → seed53으로 고정했다. 각각
  4,096환경 × 32step × 250iteration = **32,768,000전이**다.
  완료한 새 본학습 합계는 98,304,000전이이며 최종 model249만 평가한다.
- 학습 CLI seed가 agent 설정과 실제 simulator에 전달됨을 각각 기록·검사한다.
  저장된 seed를 정규화해 차이를 숨기지 않는다. 공통 비-seed 설정 비교에만
  임시 사본의 두 seed 필드를 명시적으로 치환한다.
- 같은 seed의 4,096환경 개발 초기 기록에서 `max_iterations`만 `2 → 250`으로
  투영한 기대값을 해시로 고정했다. 본학습 초기 물리/RNG/실제 episode horizon은
  이 **같은 seed**의 기대 증거와 비교한다. 서로 다른 seed의 물리 상태는 같지 않다.

### 과거 seed51 전체 재현이 먼저다

seed51 전체 학습이 v21의 final249 모델·Adam·metadata, 전체 기록된 보상/접촉
시퀀스와 비시간 scalar 33개를 정확히 재현해야 seed52·53을 시작한다.
기준을 느슨하게 만들거나 과거 모델로 대체하지 않는다. 직렬화 컨테이너 SHA의
교차 실행 동일성은 별도의 관찰이며, gate는 역직렬화한 값의 정확 동일성이다.
각 파일은 자신의 provenance에 해시로 묶인다.

이 검사는 **종단점과 기록된 증거**의 재현이지 기록하지 않은 모든 simulator 상태,
다른 플랫폼의 결정성, 또는 독립적인 네 번째 학습 시행을 입증하지 않는다.
과거 seed51과 재현 seed51은 같은 seed 실현 한 건으로 해석한다. 평가에는 세 seed
모두 새 v22 로그에서 나온 파일을 사용하고, 경로가 과거 모델이나 다른 run을
가리키지 않도록 검증한다.

## 본학습 완료와 정확 재현

세 run은 615.3/614.9/614.8초에 각각 완료했다. 새 본학습 98,304,000전이와
개발 835,584전이를 합한 **새 환경 학습 총량은 99,139,584전이**다.
과거 학습 모델을 새 학습 모델로 대체하지 않았다. 개발과 본학습을 포함한
학습 측 GPU 10명령이 모두 성공했으며 재시도는 없었다.

| Seed | 새 final249 SHA-256 |
|---|---|
| 51 | `298f96135dad68c5d45b081187cad57d932e56d6967c84e24a099f79f2460aee` |
| 52 | `7a7ec701a6477b88cfd368f38b0d20087daacfb0981d3e88d21e176a290c3e97` |
| 53 | `5c86a9b6efebf02bca4b37609acf688d69514d1887fb5b85cf28b79f5dc17c65` |

seed51의 전체 재현 gate와 [독립 CPU 재검증](../artifacts/terrain_demo/seed_continuation_v22/independent_full_replay_audit.json)이
통과했다. checkpoint 전체 90tensor/950,884값과 metadata, 8,000step의 보상/접촉
기록, 비시간 scalar 33개가 과거 실행과 정확히 같았다. 모든 38tag × 250값이
유한했고, 실제 초기 상태/RNG/horizon도 대응 기록과 일치했다. 새 파일의 SHA도
과거와 같았지만, 이는 관찰 결과이지 추가적인 직렬화 동일성 기준이 아니다.
독립 검토에서 seed52가 이 관문 완료 뒤 시작됐다는 시간 순서도 확인했다.

[학습 모델 manifest](../artifacts/terrain_demo/seed_continuation_v22/trained_models.json),
[학습 원시 증거 연결](../artifacts/terrain_demo/seed_continuation_v22/training_validation.json),
[전체 재현 gate](../artifacts/terrain_demo/seed_continuation_v22/full_replay.json).
학습 실행과 재현의 성공은 행동 성능 개선의 증거와 구분한다.

[세 모델 독립 CPU 감사](../artifacts/terrain_demo/seed_continuation_v22/independent_tensor_audit.json)에서
각각 policy 33tensor/400,631값과 Adam 57tensor/550,253값의 유한성을 확인했다.
19개 optimizer state의 step은 모두 5,000, 교사 8tensor는 V5와 같고 actor 9tensor는
부모와 달랐다. 각 38tag × 250값, 총 28,500개 scalar가 유한했다. 각 8,000번
보상 호출의 32,768,000행이 유효했고, 원시 비용은 매 호출에서 0이 아니지만
적용 계수·scaled reward·dt 벌점은 모두 0이었다. 각자의 seed/capacity 초기
상태와 실제 horizon도 확인했다. 이는 실제 업데이트와 조건 준수의 증거다.

## 완료한 개발 검증

- 학습 개발: seed별 256환경 × 2iteration과 4,096환경 × 2iteration,
  총 6회 **835,584전이**. 선택 모델이나 성능 근거로 사용하지 않는다.
- seed51은 두 크기에서 과거 checkpoint 전체·Adam·metadata, 보상 JSON,
  접촉 기록, 비시간 scalar 33개를 정확히 재현했다. 모든 38개 scalar는 유한했다.
  시간 관련 5개 tag만 값의 정확 비교에서 제외했다.
- 세 seed 모두 요청/저장/실제 환경 seed 일치와 공통 비-seed 조건을 통과했다.
  각 크기에서 RNG·초기 root state·실제 horizon 해시는 seed별로 달랐다.
- 학습 cache 준비 1회와 개발 6회의 GPU 명령은 모두 성공했다. 이 단계의 GPU
  재시도는 없었다. 개발 결과를 본평가 성과로 사용하지 않았다.
- 평가 개발 8제어기 280첫episode도 완료했다. 과거 대응 4제어기의 원시 필드
  35개, 전체 초기 상태/RNG 짝, 네 history 평지 분기가 정확히 일치했다.
  [개발 증거](../artifacts/terrain_demo/seed_continuation_v22/development.json)를 확인한 뒤
  [평가 동결](../artifacts/terrain_demo/seed_continuation_v22/frozen.json)과 새 480tile의
  초기화 전용 준비를 마쳤다. Holdout 시작 전 21GPU명령은 모두 성공했다.
- CPU 회귀 **1,501개 통과**, 새 Python 13개 pyflakes·제한된 mypy·AST·compile·
  공백 검사 통과. [CPU 기록](../artifacts/terrain_demo/seed_continuation_v22/cpu_validation.json),
  [정적 검사](../artifacts/terrain_demo/seed_continuation_v22/static_validation.json).
  감사기 작성 중 test 기대 오류와 불필요한 직렬화 SHA 동일성 gate를 수정했으며
  초기 실패 로그를 보존했다. 동결 실험 정의를 수정하거나 GPU 결과를 재시도하지 않았다.

## 고정 평가와 해석 범위

- 부모와 seed51/52/53, 각각의 unchanged v12 history gate 조합: 총 8제어기.
  history는 같은 frozen-v5 교사와 각 전문가를 결합한다.
- 개발 51/24·35환경: 8회 280첫episode. 부모/seed51 및 두 history는 v21의
  대응 개발 원시 필드 35개를 정확히 재현해야 한다. 8개 초기 상태/RNG와
  4개 history 평지 분기의 동일성을 따로 확인한다.
- 새 holdout geometry/reset은 **115/75·116/76**이다. 기존 평가 지도를 신규
  holdout으로 재사용하지 않는다. 지형은 scoring 없이 초기화한 후 cache를 고정한다.
- 주요 16초: 175환경 × 2지도 × 8제어기 = 16파일 2,800첫episode.
  제어기별 험지 300회와 평지 50회다. 부모는 조건당 한 번만 평가·계수한다.
- 별도 64초 최고난도 돌다리: 10환경 × 2지도 × 8제어기 = 16파일 160첫episode.
  제어기별 20회이며, 16초 분모와 합치지 않는다.
- 1/6타일 성공은 13.1/53.1m 전진과 첫 episode 낙상·발자국 레인 이탈·world 이탈
  없음이 동시에 필요하다. 정체·무접촉 episode도 분모에 남긴다.
- 비교 방향은 세 seed 각각 대 부모와 seed52 대51, seed53 대51, seed53 대52,
  그리고 history 대응 6개로 **총 12개**를 사전 고정했다.
- 기존 gate를 유지한다: 험지 1/6타일 비감소, 낙상/레인 이탈 비증가, world 이탈 0,
  평지 낙상/레인 비증가와 평균 episode 속도 비감소, 적어도 하나의 엄격한 험지 개선.
  별도 돌다리 평가에는 평지 조건을 적용하지 않는다.
- seed별·지도별·지형/난이도별 수치, 짝 6타일 개선/악화와 min/max 범위·부모 차이를
  모두 보고한다. 한 부모를 seed별로 복제한 900회 가짜 분모를 만들지 않는다.

한 부모·한 훈련 지형·3개 추가학습 seed·새 평가 지도 2개의 서술적 실험이다.
통계적 우월성, 특정 회귀 원인의 인과성, 실물 로봇 안전성은 보장하지 않는다.
결과에 따른 재학습·보상 변경·checkpoint/seed 선택·지도 교체·scored 재시도 없이
한 번의 고정 비교를 종료하며, 이 실험에서 기본 정책을 자동 승격하지 않는다.

## 검증과 보존

- 표준 라이브러리 전용 독립 감사가 원시 32파일·2,960첫episode의 strict 성공,
  모든 사전 비교·지도/지형/난이도 집계·짝 결과·seed 범위와 부모 차이를 재계산해
  보고서와 일치함을 확인했다. 부모 분모는 한 번만 계수한다.
- 평가가 끝난 뒤에도 전체 CPU 회귀 1,501개를 다시 통과했다.
  [최종 CPU 검증](../artifacts/terrain_demo/seed_continuation_v22/final_cpu_validation.json)은
  본학습 전 검사와 별도로 로그/XML 해시를 보존한다.
- 총 **53 GPU명령이 모두 성공**했다. 학습 cache 1 + 학습 개발 6 + 본학습 3 +
  평가 개발 9 + 신규 지도 준비 2 + scored 평가 32다. GPU 실패·재시도는 0이다.
- 독립 감사기는 checkpoint와 TensorBoard의 파일 해시를 확인하지만 이를
  역직렬화·decode하지 않는다. 실제 tensor/scalar 검사는 별도 독립 CPU 감사가
  수행했다. 접촉/자세 geometry telemetry를 독립적으로 다시 계산했다는 주장은 하지 않는다.
- 실험 정의 12개와 기존 frozen source 213개·기준 모델 19개를 보존했다.
  독립 감사기·회귀 테스트 2개는 동결된 실험 정의 밖의 검증 코드다.
- 원시 데이터·최종 모델·manifest는 `artifacts/terrain_demo/seed_continuation_v22/`,
  console/TensorBoard와 현재 기계의 cache는 Git-ignored 로컬 기록으로 구분했다.
  새 의존성, commit, 원격 Git/GitHub 작업이나 기본 정책 변경은 없다.
- 결과에 따른 재학습·seed/모델 선별·평가 재시도 없이 사전 선언한 v22 분기를
  종료했다. TASK298의 별도 기존 데모·게시 범위까지 완료했다는 뜻은 아니다.

결과 summary SHA-256:
`9a077b6ca36635973a5b062375b2d2a9a8f01c753120af9754899e8e198ec588`.
