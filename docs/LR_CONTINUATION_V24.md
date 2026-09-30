# v24 — 고정 전역 학습률의 추가학습 민감도

## 결론: 짧은 시간의 일부 성공 회복, 전체 개선 기준은 미통과

고정 전역 LR을 1e-4에서 1e-5로 낮추자 16초 단독 정책의 엄격한 6타일 성공은
같은 seed의 high 대비 **+7/+7/+9회(각 300회 중)** 늘었다. 그러나 낙상·레인 이탈·
평지 지표의 trade-off가 남아 **low 대 high의 여섯 합산 gate는 16초와 64초 모두 FAIL**이다.
low 대 부모의 여섯 합산 gate도 양 window에서 모두 FAIL이다. **기본 정책을 유지한다.**

새 학습 **132,677,632전이**, 본평가 **28물리 bundle·4,900첫 episode·9,800의존 관측**을
완료했다. GPU52명령 전부 성공, 실패·재시도0이며 독립 학습/원시 감사를 통과했다.
한 LR 설정과 세 seed의 서술적 결과이지, LR이 이전 회귀의 원인이라는 증명은 아니다.

### 16초 주요평가

각 행은 험지300회·평지50회다. high는 동결된 v22 모델, low는 새 LR1e-5 모델이다.

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모 |270|171|28|2|2|10.9356|
| high51 |269|161|27|4|2|10.5079|
| high52 |268|150|24|8|3|10.8196|
| high53 |263|157|28|9|2|10.7415|
| low51 |266|168|29|5|2|10.7558|
| low52 |268|157|27|5|2|10.6219|
| low53 |278|166|20|3|3|10.7830|
| History 부모 |269|166|30|1|2|9.3569|
| History high51 |274|167|25|0|2|9.3569|
| History high52 |276|157|24|0|2|9.3569|
| History high53 |271|165|29|0|2|9.3569|
| History low51 |273|167|24|2|2|9.3569|
| History low52 |275|164|23|2|2|9.3569|
| History low53 |273|171|24|2|2|9.3569|

- 단독 high의 6타일 범위는150~161, low는157~168회이며 부모171회에는 모두 못 미친다.
  History high는157~167, low는164~171회다. 부모는166회지만 low의 레인 이탈은
  모두2회로 부모1회·high0회보다 많아 합산 gate를 통과하지 못한다.
- low51은 high51보다 6타일+7이지만 1타일−3·낙상+2·레인+1이다. low52는
  6타일+7이나 낙상+3·평지속도 감소가 남는다. low53은 험지 지표가 모두 개선됐지만
  평지 낙상이2→3으로 늘어 전체 gate는 FAIL이다. 유리한 seed나 험지만 골라 채택하지 않는다.
- 사전18비교의 합산 PASS는 **기존 History high51 대 부모 1개**뿐이다.
  이 비교도 지도119에서는 FAIL·120에서만 PASS이고, 18비교 중 두 지도 모두 PASS는 없다.

### 64초 보조평가 — 같은 첫 episode

각 행은 험지300회·평지50회다. high는 동결된 v22 모델, low는 새 LR1e-5 모델이다.

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모 |211|209|76|18|5|12.5701|
| high51 |226|224|62|14|4|12.1065|
| high52 |233|232|55|13|3|12.4900|
| high53 |204|204|68|31|3|12.3031|
| low51 |213|213|69|23|5|12.3452|
| low52 |217|217|57|28|3|12.4462|
| low53 |228|228|57|17|6|12.3607|
| History 부모 |224|220|69|9|8|10.2884|
| History high51 |228|225|69|3|8|10.2884|
| History high52 |231|229|63|8|8|10.2884|
| History high53 |234|234|62|4|8|10.2884|
| History low51 |226|226|66|13|8|10.2884|
| History low52 |230|226|61|11|8|10.2884|
| History low53 |228|227|62|10|8|10.2884|

- low 대 high의 단독 6타일 차이는 **−11/−15/+24**, History는 **+1/−3/−7**회다.
  짧은 시간의 성공 회복이 장시간의 일관된 개선으로 이어지지는 않았다.
- low53은 high53보다 험지 6타일+24·낙상−11·레인−14지만 평지 낙상3→6으로
  합산 gate는 FAIL이다. low51/52는 각각 레인 이탈도+9/+15로 악화했다.
- 18합산비교 중 PASS는 **기존 History high51·52·53 대 부모 3개**다.
  두 지도 모두 PASS는 **History high53 대 부모만**이다. high51/52는 지도120에서만 PASS다.
- 두 window 모두 low 비교의 일부 지도120 PASS가 있지만 지도119에서는 모두 FAIL이다.
  한 지도 또는 합산 지표의 이득을 두 지도 일관성으로 바꾸지 않는다.
- 모든 world 이탈과 평지 레인 이탈은0이며 7개 History의 평지 원시는 같다.
  최고난도 돌다리(각10회)의16초6타일은 전부0이다. 64초 단독 부모/high51·52·53/
  low51·52·53은 **2/3/2/7/4/4/8**, History 대응은 **4/4/5/5/0/2/4**다.

### 같은 seed의 low − high 차이

| 추론 모드 | seed | 16초 6타일 차이 | 64초 6타일 차이 | 16초 합산 gate | 64초 합산 gate |
|---|---:|---:|---:|---|---|
| 단독 |51|+7|-11|FAIL|FAIL|
| 단독 |52|+7|-15|FAIL|FAIL|
| 단독 |53|+9|+24|FAIL|FAIL|
| History |51|+0|+1|FAIL|FAIL|
| History |52|+7|-3|FAIL|FAIL|
| History |53|+6|-7|FAIL|FAIL|

### 16→64초 같은 행의 변화 — 험지300회씩

| 제어기 | 둘 다 실패 | 늦은 성공 | 늦은 실패 | 둘 다 성공 | 늦은 실패의 새 낙상/레인 | (16,64]초 최초 6타일 도달 |
|---|---:|---:|---:|---:|---:|---:|
| 부모 |68|61|23|148|15/8|88|
| high51 |60|79|16|145|14/2|105|
| high52 |60|90|8|142|6/2|116|
| high53 |67|76|29|128|15/14|106|
| low51 |60|72|27|141|16/11|96|
| low52 |57|86|26|131|8/18|108|
| low53 |49|85|23|143|13/11|103|
| History 부모 |65|69|15|151|13/2|93|
| History high51 |62|71|13|154|12/1|97|
| History high52 |57|86|14|143|13/1|111|
| History high53 |55|80|11|154|10/1|102|
| History low51 |58|75|16|151|15/1|99|
| History low52 |56|80|18|146|16/3|102|
| History low53 |52|77|21|150|19/2|102|

- 단독 low51/52의 늦은 실패는27/26회로 high51/52의16/8회보다 많다.
  low53은23회 대 high53의29회다. 이것은 고정 표본의 짝 변화이지 기전 진단이 아니다.
- 원인 flag는 겹칠 수 있다. low53의 늦은 실패23회 중 새 낙상13·레인11은
  **1회가 겹친다**. History low52도 늦은 실패18회 중 낙상16·레인3의1회가 겹친다.
  이 표본의 늦은 실패에는 새 world 이탈·마지막 거리 미달 flag가 없었다.
- 최초 도달은 이후 낙상·이탈을 무시하므로 엄격한 성공 또는 늦은 성공과 같지 않다.
  전체49부분집합과 지도별 변화·모든18비교는 원시 연결 요약에 보존했다.

[전체 합산·지도별 gate](../artifacts/terrain_demo/lr_continuation_v24/summary.md),
[지형·난이도·seed 범위·짝 변화 전체](../artifacts/terrain_demo/lr_continuation_v24/summary.json),
[독립 표준 라이브러리 원시 감사](../artifacts/terrain_demo/lr_continuation_v24/independent_raw_audit.json).

## 질문과 문헌 근거

[v22](SEED_CONTINUATION_V22.md)와 [v23](PAIRED_HORIZON_V23.md)의 짧은 시간 성능
회귀는 학습률이 원인이라는 진단이 아니다. 같은 초기 정책·데이터 예산에서
**고정 GLOBAL PPO 학습률만 1e-4 → 1e-5**로 바꾸어 민감도를 확인한다.
전역 optimizer는 actor뿐 아니라 critic과 학습 가능한 std에도 적용된다.
작은 LR이 최종 가중치 변화나 회귀를 반드시 줄인다고 가정하지 않는다.

- [Schulman et al., PPO v2, §§2–3·5/Algorithm 1](https://arxiv.org/pdf/1707.06347v2):
  반복 최적화와 clipped objective를 검토하는 근거다. clipping은 엄격한 KL
  제한의 보장이 아니며, 다른 benchmark의 설정이 이번 LR 선택을 검증하지 않는다.
- [Andrychowicz et al., What Matters in On-Policy RL? v1, §2·3.2·3.7, Appendix J.1/Fig.69](https://arxiv.org/pdf/2006.05990v1):
  학습률과 초기 std의 조건부 민감도를 다룬다. 확인한 LR sweep은 3e-5부터이며
  인접한 **1e-5는 epsilon 후보**이지 LR 권고가 아니다. 이번 1e-5와 세 seed는
  사전 고정한 공학적 선택이다. 학습된 std 유지 등 다른 개입은 보류했다.

[원문 확인 기록](../artifacts/terrain_demo/lr_continuation_v24/reference_check.json),
[사전 계획](experiment_plans/lr_continuation_v24.md),
[계획 manifest](experiment_plans/manifest.json)를 남겼다. 과거 또는 새 holdout의
유리한 결과를 선택하거나 결과를 보고 예산·지도·gate를 조정하지 않는다.

## 고정한 비교와 예산

- 같은 v16 control final249 부모, 훈련 지형110, seed51/52/53, 초기 std0.2,
  빈 Adam, 91D conditioned/anchored/targets 정책, 8D 행동, 고정 v5 teacher와 prior0.02.
- high는 기존 v22 seed51/52/53 final249를 재사용한다. 새 high51 full replay는
  재현성 검증일 뿐 네 번째 독립 seed나 과거 결과를 대체하는 새 대조군이 아니다.
- low는 각각 4,096환경 × 32step × 250iteration = 32,768,000전이로 학습하고
  **final249만** 평가한다. PPO update당 5epoch × 4minibatch = 20 Adam step이다.
- high initializer는 기존 바이트 그대로 복사한다. low는 역직렬화한 동일 객체에서
  **빈 Adam group의 LR만** 바꾼다. 33개 정책 tensor, std, iteration, param 순서,
  다른 optimizer 옵션과 historical infos는 같다. infos는 현재 LR이 아니라 조상 기록이다.
- 일반 checkpoint load가 복구한 optimizer LR을 검사한다. 잘못된 LR을 자동으로
  고치지 않는다. saved config/algorithm/모든 Adam group LR, fixed schedule,
  desired_kl=None, actor/critic/std 포함 범위를 매 PPO update와 Adam step 전후에 검사한다.

| 새 학습 구분 | 전이 수 |
|---|---:|
| 개발: 소형2회 + capacity6회 |1,605,632|
| high51 전체 재현 1회 |32,768,000|
| low51/52/53 본학습 3회 |98,304,000|
| 총 새 학습 |132,677,632|

과거 high 3개 모델의 98,304,000전이는 재사용 조상 비용이며 새 학습에 더하지 않는다.
계획 GPU 명령은 학습13 + 평가39 = **52회**, 모두 GPU1에서 직렬 실행한다.

## 개발 재현과 동결

high51 소형과 **high51/52/53 각각의 capacity**가 같은 seed의 v22 기록과
model/Adam/metadata, reward/contact, 33개 비시간 TensorBoard scalar에서 정확히
일치했다. 전체 38개 scalar는 유한해야 한다. 시간 관련 5개만 정확 비교에서 제외한다.
low/high는 학습 **전** 상태·관측·정책·RNG·실제 무작위 episode horizon을 짝지으며,
LR 이외 설정은 같다. 학습 후 궤적·보상·최종 가중치가 같아야 한다고 요구하지 않는다.

[독립 개발 tensor/Adam/LR/scalar 감사](../artifacts/terrain_demo/lr_continuation_v24/independent_development_tensor_audit.json)는
8회·1,605,632전이를 다시 검사했다. [학습 동결](../artifacts/terrain_demo/lr_continuation_v24/training_frozen.json)은
전체 high replay와 low 본학습 전에 **평가 코드까지 포함한 12개 정의**를 고정한다.
이후 불일치에는 코드 수정·같은 identity 재시도 대신 불완료 결과를 보존한다.

독립 코드 검토에서 AST 허용 필드의 중복 key 우회와 독립 감사의 불완전한 실제 YAML
검증을 찾아 학습 전에 수정했다. 실행 알고리즘을 바꾸지 않고 음성 회귀 검사를 추가했다.
수정 당시 완료 GPU는 점수를 매기지 않은 cache 준비 1회뿐이었다. 이전 기록은 남기고
수정 후 **2,056 CPU tests**와 Python15파일 정적 검사·두 독립 재검토를 통과했다.
[사전 검증 보완 기록](../artifacts/terrain_demo/lr_continuation_v24/pre_freeze_review_fixes.json).

전체 high51 재현도 model/Adam/metadata·reward/contact·33개 비시간 scalar가 v22와
정확히 일치했다. 각 low 모델은 250 PPO update와 5,000 Adam step 동안 LR1e-5를
유지했다. 학습13 GPU 명령은 모두 성공했고 실패·재시도는 없었다.
[독립 전체 학습 감사](../artifacts/terrain_demo/lr_continuation_v24/independent_training_tensor_audit.json),
[실제 학습 증거 별도 검토](../artifacts/terrain_demo/lr_continuation_v24/independent_training_review.json).

평가 개발의 부모/History 부모는 양 window에서 과거 v23과 35필드가 정확히
같았다. 8개 초기 물리/RNG와 4개 History 평지 분기 검증도 통과했다.
[독립 개발 원시 감사](../artifacts/terrain_demo/lr_continuation_v24/independent_development_raw_audit.json),
[평가·모델 동결](../artifacts/terrain_demo/lr_continuation_v24/frozen.json),
[480개 신규 지형과 평가 입력 고정](../artifacts/terrain_demo/lr_continuation_v24/evaluation_inputs.json).

## 평가 계약 — 같은 첫 episode의 두 시점

새 geometry/reset119/79,120/80에서 부모1 + high3 + low3의 단독/History 조합
**14개 제어기**를 평가한다. 각 실행은 7지형 × 5난이도 × 5출발점 = 175환경이다.
변경하지 않은 v23 tracker로 물리64초 episode를 한 번 실행하고 16/64초를 관측한다.
16초에 simulator나 history를 가상 reset하지 않는다.

- 본평가: 28물리 bundle, **4,900첫 episode / 9,800의존 window 관측**.
  제어기·window별 험지300/평지50, 최고난도 돌다리10이다.
- 개발: geometry51/reset24의 35환경, 부모/History 부모와 low6조합 총8실행,
  280첫 episode/560의존 관측. 부모 두 제어기의 양 window는 과거 v23의
  35개 필드를 정확히 재현해야 한다. low 최종 결과에는 과거 결과 일치 목표가 없다.
- 과거 high 개발6실행을 반복하지 않는 대신 전체 evaluator AST의 제한적 동일성과
  모든14 controller/checkpoint/SHA/추론분기 binding, high/low 교환 음성 검사를 요구한다.
- 엄격한 성공은 window **마지막** 전진 거리13.1/53.1m 이상과 그때까지 첫 episode의
  낙상·레인/world 이탈 없음이다. float32 문턱을 보존하며 timeout은 낙상이 아니다.
  최대거리·최초도달은 엄격한 성공을 대체하지 않는다.
- 양 window 모두 동일한 험지 성공/낙상/이탈 및 평지 낙상/이탈/평균속도 gate를 쓴다.
  저/고 LR의 같은 seed, 각 LR 대 부모의 사전18비교를 모두 보고한다.
  **16초가 주요평가**이며 64초 개선으로 짧은 시간 회귀를 덮지 않는다.
- 같은 행의 16→64초 둘 다 실패/늦은 성공/늦은 실패/둘 다 성공, 겹칠 수 있는
  새 종료·이탈·최종거리미달 flag와 별도 최초도달 시각을 기록한다.

## 해석·공개 범위

한 부모·한 훈련 지형·세 학습 seed·두 평가 지도에 한정된 서술적 실험이다.
episode·window·History 조합을 독립 학습 반복으로 세지 않는다. 전역 LR 실험을
actor만의 인과 기전, 통계적 우월성, 일반 강건성 또는 실물/무기한 안전 보장으로
확대하지 않는다. 과거 버전의 다른 지도 결과를 직접 합치지 않는다.

기존237개 source와22개 checkpoint 경로 및 기본 정책을 보존한다. 원시 console,
TensorBoard, cache, runtime 상태는 ignored 로컬 증거이며 원격 게시 증거가 아니다.
의존성 설치·commit·원격 Git/GitHub 작업·실물 로봇 조작은 하지 않는다.


## 검증 산출물과 재현 경계

- [학습 명령13개](../artifacts/terrain_demo/lr_continuation_v24/training_commands.jsonl)와
  [평가 명령39개](../artifacts/terrain_demo/lr_continuation_v24/commands.jsonl)에 정확한 인자,
  순서, 시간, console SHA가 있다. 출력 덮어쓰기·실패 후 묵시 재시도는 금지한다.
- [초기 high/low 차이](../artifacts/terrain_demo/lr_continuation_v24/initial_shared.json),
  [학습 완료 모델](../artifacts/terrain_demo/lr_continuation_v24/trained_models.json),
  [전체 high 재현](../artifacts/terrain_demo/lr_continuation_v24/full_replay.json)을 각각 핀했다.
- 독립 감사기는 동결된 실험 정의 밖에서 자신의 소스 SHA를 기록한다.
  [stdlib 원시 감사 코드](../scripts/audit_lr_v24_raw.py)와
  [CPU 학습 감사 코드](../scripts/audit_lr_v24_training.py)는 실험 validator를 import하지 않는다.
  원시 상태 해시는 기록된 비개입 증거이며 simulator 내부 모든 상태나 접촉 기하를
  독립 재시뮬레이션한 보장은 아니다. CPU 감사는 실제 역직렬화된 tensor·Adam·로그를 검사한다.
- 전체 학습/평가 GPU 작업은 종료했고 GPU1은18MiB idle로 확인했다.
  이 고정 실험 분기를 종료하며 추가 실험이나 기본 모델 승격을 자동으로 시작하지 않는다.
