# v21 — 같은 예산의 무접촉비용 추가학습 대조군

## 결론: 무비용 추가학습도 부모를 개선하지 못했다

새 두 지도에서 **32파일·2,960첫episode**를 완료했다. 16초 주요평가의
12개 합산 gate 중2개, 별도64초의12개 중1개만 PASS이며, 통과한 비교도
두 지도 각각에서 모두 통과하지는 못했다. **추가학습군의 부모 대비 합산 gate는
모두 FAIL**이다. 기존 권장 기본정책을 유지한다.

### 주요16초 — 제어기별 험지300회·평지50회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모(v16 control) |272|169|27|0|2|11.1235|
| 무비용 |276|164|18|5|3|10.4999|
| 즉시 |279|162|19|2|3|10.7125|
| 점진 |280|165|16|5|3|10.8113|
| History 부모 |274|175|21|5|5|8.9204|
| History 무비용 |282|168|16|1|5|8.9204|
| History 즉시 |269|161|27|4|5|8.9204|
| History 점진 |276|161|23|1|5|8.9204|

- 무비용 대 부모:6타일169→164, 낙상27→18, 레인 이탈0→5,
  평지낙상2→3, 평지속도11.1235→10.4999m/s. 짝6타일 개선/악화15/20.
  비용을 적용하지 않아도 통과/레인/평지 성능의 회귀가 관찰됐다. 따라서
  기존 비용군의 회귀를 **접촉 벌점만의 영향으로 단정할 수 없다**.
- 점진 대 무비용은6타일164→165·낙상18→16·이탈5→5로 합산 PASS지만,
  지도113의6타일은81→79, 지도114는83→86이다. 차이−2/+3의 합+1이며
  짝 개선/악화18/17이다. 부모169보다 낮아 전반적인 개선이나 승격 근거가 아니다.
- History 무비용 대 부모는6타일175→168로 FAIL이다. 낙상21→16·
  이탈5→1의 이득과 구분한다. History 비용군은 무비용보다6타일이 각각7회 적다.
- 나머지 합산 PASS는 History 점진 대 History 즉시다.6타일161로 같고
  낙상27→23·이탈4→1이지만, 무비용168이나 부모175를 넘어선 결과가 아니다.
- 모든world이탈0·평지레인 이탈0. 네 History의 평지 원시 결과는 정확히 같다.
- 지도113에서 PASS는 즉시 대 무비용 하나뿐이다. 지도114에서는 점진 대 무비용,
  History 무비용 대 부모, History 점진 대 즉시만 PASS이며 나머지는 FAIL이다.

### 별도64초 최고난도 돌다리 — 제어기별20회

16초 분모와 합치지 않는다. 낙상과 레인 이탈flag는 겹칠 수 있다.

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| 부모(v16 control) |13|12|5|2|
| 무비용 |8|8|8|4|
| 즉시 |5|4|13|3|
| 점진 |8|8|8|5|
| History 부모 |12|8|6|2|
| History 무비용 |9|7|9|3|
| History 즉시 |7|6|11|4|
| History 점진 |8|6|8|4|

- 무비용 대 부모는6타일12→8·낙상5→8·이탈2→4로 FAIL이다.
  History 무비용도6타일8→7·낙상6→9·이탈2→3으로 FAIL이다.
- 점진 대 무비용은6타일8로 같지만 이탈4→5, 점진 대 즉시는6타일4→8이지만
  이탈3→5로 각각 FAIL이다. 모든world이탈은0이다.
- 유일한 합산 PASS는 History 점진 대 즉시:6타일6으로 같고 낙상11→8,
  이탈4로 같다. 그러나 지도114에서1타일4→3·이탈1→2로 FAIL이다.
- 지도113은 History 무비용 대 부모, History 점진 대 즉시, History 점진 대 부모가
  PASS다. 지도114는 점진 대 즉시만 PASS다. 나머지 지도별 비교는 전부 FAIL이다.

이번 결과는 **추가학습 절차 자체에서도 성능 간 맞교환이 생김**을 보여준다.
특정 보상 하나의 인과효과나 일반적인 학습 실패 원인을 확정하지는 못한다.
평가 후 보상/모델/지도를 바꾸거나 재실행하지 않았으며, 이번 고정 실험을 종료한다.

[모든 합산·지도별 판정](../artifacts/terrain_demo/contact_continuation_v21/summary.md),
[지형군·난이도·접촉/정체·짝 결과](../artifacts/terrain_demo/contact_continuation_v21/summary.json),
[독립 원시 감사](../artifacts/terrain_demo/contact_continuation_v21/independent_raw_audit.json).

## 질문과 논문 참고 범위

[v20](CONTACT_CURRICULUM_V20.md)에는 같은 예산의 무비용 추가학습군이 없었다.
따라서 부모 대비 회귀를 접촉 비용과 추가학습 자체의 영향으로 구분하기 어려웠다.
이번에는 새 보상을 만들지 않고 누락된 대조군을 추가했다.

[Aractingi et al., *Controlling the Solo12 Quadruped Robot with Deep Reinforcement
Learning* (2023)](https://arxiv.org/html/2309.16683v1)의 reward curriculum은
정지로 벌점을 피하는 해를 줄이기 위해 비용 계수를 점진적으로 높인다.
v20은 그 발상을 기존 contact-slip 한 항에만 적용했고, v21은 그 비교 설계의
대조군을 보완한다. 논문 전체 방법이나 Solo12 실물 실험의 재현이 아니다.
원문 reward-curriculum 절을 이번에도 확인했다. 새 학습/평가 조건은
[사전 계획](experiment_plans/contact_continuation_v21.md)에 따랐다.

## 동일 조건과 과거 학습군 재사용의 한계

- 부모는 v16 control final249이며 SHA는
  `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`다.
- v20 init을 새 실험 경로로 바이트 그대로 복사했다. SHA는
  `10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd`.
  부모와32개 non-std tensor 동일, std0.2, 빈 Adam1e-4, iteration0,
  frozen-v5 teacher8개 tensor 동일을 독립 CPU 역직렬화로 확인했다.
- 새로 학습한 것은 `no_cost` 하나다. v20의 `immediate`/`ramped` final249는
  그대로 재사용했으며 이번에 새로 학습한 것으로 계산하지 않는다.
- 모두4096env×32step×250iteration, seed51/geometry110,91D정책/가치·8D동작,
  conditioned/anchored/targets와 prior0.02다. 계수 외의 물리·보상·종료 조건을
  유지했고, geometry110 cache도 v20과 바이트 동일하다.
- no_cost도 매번 같은 센서·접촉 기하·원시 비용을 계산한 뒤 계수0을 곱한다.
  RewardManager weight는1이므로 비용 항을 건너뛴 실험이 아니다.
- 초기 state/91Dobs/88Dprefix/관절/정책/RNG/정규 설정과 실제 환경의 무작위
  episode horizon이 해당 v20 증거와 일치했다. 새 함수명·실험명·load_run·
  계수 mode·기존 출력 식별자만 명시적으로 정규화하며 다른 차이는 허용하지 않는다.

과거 v20의2-iteration 즉시군을256env와4096env에서 각각 재생했다.
전체 checkpoint의90개 tensor/950,884값, Adam/iteration/infos, 실제 reward schedule,
사후 접촉 통계, 비시간 scalar33종이 모두 정확히 같았다. 직렬화 SHA까지 일치했다.
시간/FPS3종 및 시간축 Train중복2종만 정확 비교에서 제외했으며38종 전부의
유한성과 completeness는 검사했다.

- 과거 model1의 SHA는 당시 v20 동결 항목이 아니었다. **이번 재생 전에 새로
  고정한 과거 증거**이며, 원래부터 동결됐다고 주장하지 않는다.
- 초기 환경·horizon 동일성은 계측한 hash를 비교한 것이며 실시간 tensor의
  별도 재수집은 아니다. 짧은 정확 재생은 실행 호환성을 뒷받침하지만 과거
  250iteration 전체 실행이 지금도 동일하다는 증명은 아니다.
- 한 학습 seed·새 지도2개·과거 학습군 재사용이다. 완전한 인과 추론,
  통계적 우월성, 임의 지형 일반화나 실물 로봇 안전을 주장하지 않는다.

[과거 증거 pin](../artifacts/terrain_demo/contact_continuation_v21/historical_references.json),
[독립 재생 검증](../artifacts/terrain_demo/contact_continuation_v21/independent_replay_audit.json).

## 학습량과 실제 학습 검증

| 구분 | 전이 수 | 이번 신규 학습인가 |
|---|---:|---|
| no_cost 본학습 |32,768,000|예|
| 양성대조/무비용 사전 개발4회 |557,056|예, 모델 선택에 미사용|
| 이번 환경 학습 합 |33,325,056|예|
| v20 즉시/점진 본학습 재사용 |65,536,000|아니오|

본학습624.2초, 최종249만 선택했다. 실제 reward8,000회, 유효 행32,768,000개이며
8,000개 step 모두 원시 평균 비용이0이 아닌데도 계수·가중 보상·dt 벌점은 정확히0이다.
개발평가280첫episode와 holdout은 구분하고 평가 중 학습은0이다.

독립 CPU 검증에서33policy tensor/400,631값과57Adam tensor/550,253값이 유한했다.
19optimizer state의step은 모두5,000, actor9tensor는 부모와 달라 실제 학습을 확인했다.
교사8tensor는 원래v5와 그대로 같고,38TensorBoard tag/9,500값도 유한했다.
이는 실제 업데이트 증거이지 행동 개선의 증거는 아니다.

새 no_cost final249 SHA:
`298f96135dad68c5d45b081187cad57d932e56d6967c84e24a099f79f2460aee`.

[학습 동결](../artifacts/terrain_demo/contact_continuation_v21/training_frozen.json),
[학습 증거](../artifacts/terrain_demo/contact_continuation_v21/training_validation.json),
[최종 모델](../artifacts/terrain_demo/contact_continuation_v21/trained_models.json),
[독립 tensor 검증](../artifacts/terrain_demo/contact_continuation_v21/independent_tensor_audit.json).

## 평가 계약

부모/무비용/즉시/점진 및 각각 unchanged v12 history gate 조합, 총8제어기다.
모든 history는 같은 frozen-v5 교사와 해당91D 전문가를 사용한다.

- 개발51/24·35env:8제어기280첫episode. 재사용6제어기는 v20의35개 물리/라우팅/
  계측 필드를 정확히 재현했다.8개 초기state/RNG짝과4history 평지 분기도 같다.
- 새 지도/reset113/73·114/74. 이전111/112 점수를 holdout으로 재활용하지 않는다.
  두 지도의480tile을 step 없이 초기화한 후 cache/source/model/명령ledger를 고정했다.
- 주요16초175env: 제어기별 험지300+평지50,16파일2,800첫episode.
- 별도64초 최고난도 돌다리10env: 제어기별20,16파일160첫episode.
- 두 horizon의 분모를 섞지 않는다. strict1/6타일은13.1/53.1m 전진과
  첫episode 낙상·발자국 레인 이탈·world 이탈 없음이 동시에 필요하다.
  무접촉·저속·정체 episode도 분모에서 빼지 않는다.
- 무비용 대 부모, 즉시/점진 대 무비용, 점진 대 즉시 및 history대응8비교에
  두 비용군 대 부모 및 history4비교를 더해 horizon별12gate를 전부 보고한다.
- Gate는 험지1/6타일 비감소, 낙상/레인 비증가, 평지포함world0,
  평지 낙상/레인 비증가와 속도 비감소, 적어도 하나의 험지 strict개선이다.
  별도64초는 평지 조건을 적용하지 않는다. 모든 지도별 실패를 공개한다.

## 재현 명령과 보존 경계

기존 증거를 덮어쓰거나 동일 holdout을 반복해 유리한 결과를 선택하지 않는다.
아래 명령은 기록된 실행 순서이며 이미 완료된 경로에서는 중복 실행을 거부한다.

```bash
../run-python scripts/run_continuation_training_v21.py prepare
../run-python scripts/run_continuation_training_v21.py prepare_cache
../run-python scripts/run_continuation_training_v21.py preflight
../run-python scripts/run_continuation_training_v21.py capacity
../run-python scripts/run_continuation_training_v21.py freeze_train
../run-python scripts/run_continuation_training_v21.py train
../run-python scripts/run_continuation_eval_v21.py development
../run-python scripts/run_continuation_eval_v21.py freeze
../run-python scripts/run_continuation_eval_v21.py prepare
../run-python scripts/run_continuation_eval_v21.py evaluate
../run-python scripts/run_continuation_eval_v21.py horizon
../run-python scripts/run_continuation_eval_v21.py report
/usr/bin/python3 scripts/audit_continuation_v21_raw.py
```

평가 점수에 따라 보상·지도·최종 모델·통과 기준을 변경하지 않는다. GPU1에서
heavy job을 하나씩 실행하며 기존v0–v20 소스/모델/미커밋 작업과 기본정책을 보존한다.
새 의존성·commit·원격 Git/GitHub 작업이나 자동 정책 승격은 없다.

## 최종 검증과 남은 위험

- 독립 표준라이브러리 감사가32원시파일·2,960episode의 strict배열·분모,
  합산/지도/지형군/난이도별12gate와 paired6타일을 재계산해 summary와 일치했다.
  49개 GPU명령이 모두 성공했고,32개 holdout명령은 고유하며 재시도가 없다.
- 기존197frozen source·18model, 새 실험정의14파일, 학습/평가/cache/사전ledger
  연결을 보존했다. 평가cache480tile/1,440파일과 학습cache240tile/720파일을 검증했다.
- 감사기는 실험정의 동결 후 작성한 별도 코드다. 접촉/자세 telemetry의 물리 기하를
  독립 재계산하지 않으며, checkpoint/TensorBoard는 hash만 검증한다.
  실제tensor역직렬화/학습scalar 검증은 별도 독립CPU검증이 담당했다.
- 독립 검토에서 감사기의 report-controlled parity목록이 비어도 통과할 수 있는
  취약점을 발견했다. **실험 코드는 바꾸지 않고** 독립감사기만35필드 고정·
  누락/변조 검사로 보강했다.15변조 회귀를 추가했고 감사기58tests가 통과했다.
  검토 전 검사로그는 attempt01로 보존했으며 이후 정적/전체 검사를 다시 실행했다.
- 전체**1,303 CPUtests PASS(29.35s)**. 새15Python파일 pyflakes·boundedmypy,
  AST/compileall/공백·diff 검사를 통과했다. Mypy는 follow-imports=skip/
  ignore-missing-imports 범위이며 전체Isaac typeproof가 아니다. Ruff/Pyright는 미설치다.
  기존 frozen파일의 lint 경고를 없애려고 과거 바이트를 수정하지 않았다.
- 논문에서 참고한 개념과 이번 공학적 대조군 추가를 구분했다. 학습 seed복제나
  하드웨어 검증은 없고,두 신규지도와 과거 학습군 재사용의 한계가 남는다.
- 원시로그·TensorBoard·기계cache·runtime상태는 계속 ignored출력이다. 로컬 실험을
  완료했으며 기존 공개 snapshot의 검증을 이번 게시 증거로 사용하지 않는다.
  기본정책/의존성/commit/원격 Git/GitHub 상태를 변경하지 않았다.

[CPU 검증](../artifacts/terrain_demo/contact_continuation_v21/cpu_validation.json),
[정적 검증](../artifacts/terrain_demo/contact_continuation_v21/static_validation.json),
[독립 감사기 검토](../artifacts/terrain_demo/contact_continuation_v21/independent_verifier_review.json).

## 다음 가설 — 이번 실행에 포함하지 않음

추가 보상 튜닝보다 다중 학습 seed의 동시 대조군 복제와 레인 이탈 직전
행동/상태·업데이트 변화를 진단하는 것이 다음 후보이다. 지금의 두 holdout을
튜닝 지도처럼 재사용하거나 단일 합산 이득으로 모델을 승격하지 않는다.
