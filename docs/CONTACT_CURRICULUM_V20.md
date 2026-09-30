# v20 — 접촉 미끄럼 비용의 즉시 도입과 점진 도입

## 16초 주요 결과: 전체 개선 실패

새 두 지도 합산, 제어기별 **험지300회·평지50회**다. 고정된 여섯 비교 gate가
모두 실패했으며 기존 기본 정책을 유지한다. 별도64초 평가는 주요평가의 실패를
뒤집는 승격 근거로 사용하지 않는다.

| 제어기 | 험지1타일 | 험지6타일 | 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| 부모(v16 control) |275|178|23|2|3|10.8225|
| 즉시 |273|158|26|1|2|10.7322|
| 점진 |273|164|21|6|3|10.7974|
| History 부모 |279|172|19|1|2|9.3507|
| History 즉시 |272|167|27|1|2|9.3507|
| History 점진 |276|172|22|2|2|9.3507|

- 점진 대 즉시:6타일+6/300·낙상−5지만 레인 이탈+5·평지 낙상+1.
  짝6타일 개선/악화26/20. History에서도6타일+5·낙상−5이나 이탈+1,
  짝 개선/악화17/12로 안전성을 포함한 gate는 실패했다.
- 부모 대비 즉시/점진의6타일은 각각−20/−14. 점진의낙상−2만 떼어 개선을
  주장할 수 없다. 모든world이탈은0이며 세history의평지 원시결과는 완전히 같다.
- 지도111에서는 history 점진 대 history 즉시만 PASS. 지도112는 여섯 gate 모두
  FAIL이다. 특히 점진 단독의 레인 이탈6건 중5건이 지도112에서 발생했다.
  합산 이득이나 한 지도 결과로 일반화를 주장하지 않는다.
- 부모는 추가학습이 없는 기준선이다. 이번에는 같은 예산의 **무접촉비용 추가학습
  대조군이 없으므로**, 부모 대비 회귀를 접촉 비용만의 인과효과로 분리할 수 없다.
  두 arm 사이에서는 동일 추가학습 예산과 계수 이외의 설정을 고정했다.

## 별도64초 최고난도 돌다리: 장시간 개선도 실패

제어기별20첫episode로16초분모와합치지않는다. 모든world이탈은0이다.
낙상과레인 이탈flag는겹칠수있으므로두수를더해고유실패수를추정하지않는다.

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| 부모 |8|8|11|1|
| 즉시 |8|7|9|3|
| 점진 |5|5|10|6|
| History 부모 |8|7|11|3|
| History 즉시 |8|5|12|1|
| History 점진 |5|5|11|9|

- 점진 대 즉시:6타일7→5, 낙상9→10, 이탈3→6. 짝6타일개선/악화4/6.
- History 점진 대 즉시:6타일5→5, 낙상12→11이나이탈1→9. 짝개선/악화4/4.
- 여섯 합산gate와두지도 각각의여섯gate모두FAIL이다. 지도111의단독두arm은
  표의주요count가동률이지만strict개선이없다. 지도112의점진단독6타일은1/10이다.
- **결론: 이점진 스케줄은 기존부모를 대체하지못한다.** 학습모델·실패원시배열을
  보존하고기본정책을바꾸지않았다. 실패후holdout 재실행이나중간모델재선택은없다.

[전체 합산·지도별 표](../artifacts/terrain_demo/contact_curriculum_v20/summary.md),
[지형군·난이도·접촉/정체 진단](../artifacts/terrain_demo/contact_curriculum_v20/summary.json),
[독립 원시 감사](../artifacts/terrain_demo/contact_curriculum_v20/independent_raw_audit.json).

## 질문과 사전 고정 범위

v19는 v16 control의 합산 통과 개선과 동시에 지도별·장시간 이탈을 확인했다.
이 결과만으로 접촉 미끄럼이나 벌점 도입 시점이 실패 원인이라고 단정할 수 없다.
이번 실험은 **같은 부모 정책을 같은 예산으로 추가 학습할 때 기존 접촉 비용을
점진적으로 도입하는 것이 즉시 도입보다 나은가**를 묻는다.

- [사전 계획](experiment_plans/contact_curriculum_v20.md)
- [실행·검증 산출물](../artifacts/terrain_demo/contact_curriculum_v20/)
- 기존 v0–v19 코드·체크포인트와 권장 기본 정책은 변경하지 않는다.

## 논문과 이번 구현의 차이

[Aractingi et al., *Controlling the Solo12 Quadruped Robot with Deep Reinforcement
Learning* (2023)](https://arxiv.org/html/2309.16683v1)의 reward curriculum은
벌점을 선형으로 키워 속도 추종 대신 정지하는 해를 피하려는 설계다. 본 실험은
그 발상을 참고하되 **기존 contact-slip 한 항에만** 적용한다. 전체 벌점, 처음부터
학습, Solo12 하드웨어, 논문의 전체 학습 과정은 재현하지 않는다.

0→1 상승 구간 4,000 global policy step은 사전에 정한 공학적 선택이며 논문에서
가져온 수치가 아니다. Ant는 기존 토크 동작·이상적 지형 관측을 그대로 쓴다.

## 구현과 비교의 경계

- 부모: v16 control `model_249.pt`, SHA-256
  `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
- 모든 비탐색 네트워크 텐서를 보존하고 두 arm 모두 std0.2, 빈 Adam1e-4,
  iteration0에서 출발한다. 정책/가치 함수91D, 동작8D, frozen-v5 prior0.02는 동일.
- `immediate`: 매 step 계수1. `ramped`: `min((t-1)/3999, 1)`.
  실제 reward call의 global counter를 사용해 개별 episode reset/wrap으로 재시작하지 않는다.
- 두 arm 모두 동일 센서·접촉 기하·원시 보상을 계산하고 마지막 scalar만 곱한다.
  RewardManager weight는 둘 다1. 실제 reward 시점의 counter/계수/원시·가중 평균/
  유효 행 수/dt 적분합을 저장하며, reset/wrap을 제외한 사후 접촉 통계는 별도로 기록한다.
- 각4096env×32step×250iter=32,768,000전이, 두 arm 합65,536,000전이.
  학습geometry110/seed51, 최종249만 평가한다.
- 계수의 step 합은8,000 대6,000이다. 이는 실현된 벌점 총량과 다르며 후자는
  방문 상태와 접촉 비용·dt에도 의존한다. **시점만 바꾼 동량 실험이 아니다.**
- 새 telemetry proxy는 RSL의 `episode_length_buf` 대입을 실제 환경으로 전달한다.
  이전 v16의 읽기 전용 proxy는 수정하지 않았다. 두 새 arm의 실제 초기 horizon
  SHA가 같음을 확인하지만, v16의 과거 학습 궤적 재현을 주장하지 않는다.

## 평가 계약

같은 새 지도/reset111/71·112/72에서 부모/즉시/점진 및 각 unchanged history gate
조합, 총6제어기를 비교한다. 16초175env와 별도64초최고난도돌다리10env로
24파일·2,220첫episode다. controller별 16초험지300+평지50, 별도64초돌다리20이다.

Strict1/6타일은13.1/53.1m 전진과 첫episode 낙상·발자국 레인 이탈·world 이탈 없음이
동시에 필요하다. 정체/무접촉 episode도 분모에서 빼지 않는다. 합산, 지도별, 지형군별,
난이도별, paired6타일 개선/악화를 보고하고 평지 속도·낙상을 별도로 유지한다.

점진 대 즉시뿐 아니라 두 후보 각각의 부모 대비 gate를 보고한다. 회귀한 즉시 arm보다
낫다는 이유만으로 부모를 개선했다고 할 수 없다. 한 학습 seed와 두 지도인 서술적 비교로,
통계적 우월성·실물 안전·임의 지형 일반화나 자동 기본 정책 승격을 주장하지 않는다.

## 재현 명령

새 출력 경로/독립 복사본에서만 재실행한다. 기존 증거 경로는 덮어쓰지 않는다.
원래 평가를 다시 실행해 유리한 결과를 고르거나 실패를 대체하지 않는다.

```bash
../run-python scripts/run_curriculum_training_v20.py prepare
../run-python scripts/run_curriculum_training_v20.py prepare_cache
../run-python scripts/run_curriculum_training_v20.py preflight
../run-python scripts/run_curriculum_training_v20.py capacity
../run-python scripts/run_curriculum_training_v20.py freeze_train
../run-python scripts/run_curriculum_training_v20.py train
../run-python scripts/run_curriculum_eval_v20.py development
../run-python scripts/run_curriculum_eval_v20.py freeze
../run-python scripts/run_curriculum_eval_v20.py prepare
../run-python scripts/run_curriculum_eval_v20.py evaluate
../run-python scripts/run_curriculum_eval_v20.py horizon
../run-python scripts/run_curriculum_eval_v20.py report
```

GPU1에서 heavy job을 하나씩 실행한다. 모든 paired 실행에 앞서 지형cache를
초기화만 하고 점수화 없이 고정한다. source/model/cache/초기state/RNG/평지history
분기 불일치 시 중단하며, 실패 증거를 보존한다. 새 의존성·commit·원격 작업은 없다.

## 학습·검증 기록

- 각250iteration, 최종 두 모델의 본학습 합65,536,000전이. 즉시614.3초·점진615.0초.
  별도 사전검증 학습은256×64×2 +4096×64×2 =557,056전이이며,
  그 모델은 선택·평가에 쓰지 않았다. 환경 학습 실행 전체 합은66,093,056전이다.
  summary의 `new_training_transitions`는 고정한 두 최종 모델의 본학습 예산을 뜻한다.
- 실제reward call각8,000회, reward-time invalid행0.
  계수step합8,000대6,000; 실제 `-sum(reward * dt)`는 각각
  **64,750.662216160395 / 49,263.11250755132**이다. 이는4096환경의 합이며
  정책 성능 점수나 episode당 평균이 아니다.
- [학습 고정 입력](../artifacts/terrain_demo/contact_curriculum_v20/training_frozen.json),
  [학습 증거 검증](../artifacts/terrain_demo/contact_curriculum_v20/training_validation.json),
  [최종 모델](../artifacts/terrain_demo/contact_curriculum_v20/trained_models.json).
- 독립CPU검증: init32개 non-std tensor는부모와완전동일, teacher8개는부모/init/
  두final에서원래v5와완전동일. 각final33개policy·57개Adam tensor유한,
  19개optimizerstate의step5000, actor9개tensor는부모와달라실제학습을확인했다.
  [검증 범위와 해시](../artifacts/terrain_demo/contact_curriculum_v20/independent_tensor_audit.json).

| 모델 | SHA-256 |
|---|---|
| 즉시 final249 |`f3a72e6fc41b8c3f86ec2462e134e7632218bd2ef88de3bdf2eefc2f39d4700e`|
| 점진 final249 |`4e766667f920f70489f0c1d5e26d0812b0a788173490e4269a6277ce16249378`|

## 최종 검증과 남은 위험

- 24개원시파일·2,220첫episode와24개고유성공평가명령을독립표준라이브러리
  계산으로감사했다. 모든strict값/분모/합산·지도·지형·난이도gate/짝6타일,
  source/model/cache/사전ledger 연결과학습counter/계수/dt합이일치했다.
- 독립감사기는실험정의를고정한후작성한별도검증코드이며, 자신의SHA를보고서에
  기록했다. frozen평가정의를바꾸거나원시결과를선별하지않는다. TensorBoard값은
  학습검증기가검사하고, tensor역직렬화는별도독립CPU검증으로확인했다.
  감사기자체는물리접촉/자세telemetry의기하전체를독립재계산하지않는다.
- 전체**1,145 CPUtests PASS(27.77s)**. 새15Python파일pyflakes, boundedmypy11파일,
  AST/compileall/새파일공백/diff검사를통과했다. Mypy는`follow-imports=skip`/
  `ignore-missing-imports`범위이며전체Isaac의typeproof가아니다. Ruff/Pyright미설치.
  과거frozen파일의기존lint경고를없애기위해그바이트를수정하지않았다.
- 최초추가정적검증호출은course가상환경에pyflakes가없어실패했다. 로그보존후
  이미설치된`/usr/bin/python3`로같은검사를실행해통과했으며새의존성설치는없다.
  GPU학습/개발/holdout에는실패나재시도가없었다.
- 기존181개frozen source·16개기준model SHA, 새실험정의14파일의동결을보존했다.
  원시로그/기계cache는ignored출력,재사용가능한계획·보고서는docs/artifacts에분리했다.
  로컬실험완료이며commit/push/원격조회/공개게시를수행하지않았다.

## 다음 가설 — 아직 실행하지 않음

벌점스케줄의범위를임의로재조정해이번holdout에맞추지않는다. 다음독립연구에서는
동일예산의무접촉비용추가학습대조군으로추가학습자체와비용도입을분리하고,
레인 이탈직전상태를진단하는것이우선이다. 현재결과만으로원인이나새기법의성공을
주장하지않는다. 이번요청의사전선언된한쌍실험은여기서종료한다.
