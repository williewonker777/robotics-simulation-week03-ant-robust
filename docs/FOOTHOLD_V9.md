# v9 — 스캔에서 추출한 발끝 착지 후보

## 목적과 경계

v6의 평탄화 높이 관측, v7의 발 위치 CNN, v8의 고정 정책+잔차와 달리,
**관측된 지형에서 명시적인 발끝 후보를 추출**하고 작은 MLP에 제공한다.
기존 권장 v5를 보존하고, 동일 예산의 세 조건을 비교한다. 성능 결론은 고정
학습과 신규 지형 평가가 모두 끝난 뒤 기록한다.

- 실제 RGB-D 영상이 아닌 이상적인 yaw 정렬 수직 레이캐스트다.
- Ant는 4발/8 effort action이며 각 다리는 2자유도다. 후보는 정확한 IK,
  몸체 안정성, 발 전체 캡슐의 지지를 증명하지 않는다.
- 실제 발 캡슐 길이는 약0.566m, 반경0.08m다. 발끝 중심만으로 접촉 전체를
  대표할 수 없으며 발 이름의 순서와 x/y 사분면도 혼동하면 안 된다.
- 최고 난이도 돌다리의 설정상 gap 상한은0.30m지만, 현재0.1m heightfield의
  정수 양자화 후 난이도0.8/1.0의 실제 stone/gap 폭은0.5/0.2m다.

## 구현

3.2×2.4m 영역을0.1m 간격,33×25=825 ray로 관측한다. 이전60D에
4발 FK 좌표12D, 각 발 상대 후보XYZ+valid16D를 붙인 **88D**다.

1. 3×3 이웃이 모두 관측되고 높이 범위가0.06m 이하인 셀만 후보로 삼는다.
   스캔 경계/미관측/클리핑된 높이는 제외한다.
2. 현재 발끝으로부터 XY0.45m 이내, 같은 몸체 사분면, Z차이0.4m 이내를
   사용한다. 관측된 지역 최고면보다0.18m 넘게 낮은 후보는 제외한다.
3. 후보 중심Z는3×3의 **최고 높이+발 반경0.08m**다. 가장 가까운
   `현재 발XY+(0.2,0)` 후보를 관측에 넣는다. 후보가 없으면XYZ=0,valid=0.
4. `guided` 보상은 앞으로 움직이는 후보가 아니라 **현재 발에 가장 가까운
   지지 후보**에 대한 침하 및 낮은 발의 수평 이탈을 벌점화한다.
   정지한 침하도 벌점 대상이고 높은 swing의 수평 벌점은 사라진다.
   4발 평균[0,2], weight−1. 후보가 없는 발의 비용은0으로 기권한다.
5. 벌점은 훈련 시 돌다리 family mask에만 적용한다. 이 마스크는 특권적
   reward shaping이며 actor/후보 추출기는 지형 정답이나 family를 받지 않는다.

후보가 매 스텝 바뀔 수 있다. swing-phase/contact latch는 구현하지 않았고,
후보 없음이나 비용0을 안전으로 해석하지 않는다. reset/wrap 후에는 레이를
강제 갱신한다. 스캔은 noise-free, 기존60D 고유감각 관측 잡음은 그대로다.

## 대조군과 고정 평가

| 모드 | actor/critic 입력 | 추가 보상 |
|---|---|---|
| feet | 60D+발12D, 마지막16D 마스킹 | 없음 |
| targets | 전체88D | 없음 |
| guided | 전체88D | 현재 지지면 벌점−1 |

모두400/200/100 ELU MLP. 기존 v5의60D 가중치를 복사하고 추가28열은0으로
초기화한다. 초기 행동/가치 함수의 일치를 테스트한다. mode는 checkpoint
buffer에 저장하고 학습 설정과 불일치하면 거부한다.

- 3조건×학습seed42/43/44, geometry51/58/59, 각각750iteration×4096env×32step.
- 9개 최종749 checkpoint를 전부 고정한 후 신규geometry64/reset38,
  geometry65/reset39에서 평가한다. 평가 후 모델 선별/추가 학습하지 않는다.
- frozen v5도 같은 v9 scene에서 조건당 한 번씩 평가한다. 복제하여 표본을
  늘리지 않는다. 총20 JSON/3500 first episodes, 각 최대16초.
- 엄격한13.1m/53.1m(1/6타일), 종결/레인·world이탈 배제, margin1.1m 유지.
- guided가 v5/feet/targets 대비1타일·6타일·낙상률을 악화시키지 않고,
  v5 대비 평지 낙상 회귀 없음/world이탈0, 3seed 중2개 이상에서 targets보다
  1타일 개선이 있어야 승격한다. 실패하면 기존 v5를 권장한다.
- 대표 영상은 결과를 보기 전에 seed42/geometry64/reset38/돌1.0/16초로 지정.
  리셋과 실패를 자르지 않으며 영상은 통계 표본에 추가하지 않는다.

## 실행

기존 환경과 `../run-python`을 사용한다. 새 의존성은 없다.

```bash
../run-python -m pytest -q tests
../run-python scripts/probe_foothold.py --headless --device cuda:1 \
  --output artifacts/terrain_demo/foothold_v9/probe.json
../run-python scripts/run_foothold_experiment.py --phase train --device cuda:1
../run-python scripts/run_foothold_experiment.py --phase evaluate --device cuda:1
../run-python scripts/summarize_foothold.py artifacts/terrain_demo/foothold_v9
../run-python scripts/render_foothold_comparison.py --device cuda:1
../run-python scripts/build_foothold_viewer.py
```

실험 runner는 원본 v7/v8 소스 및 모델을 보존하고 기록 덮어쓰기를 거부한다.
재실행 시 새로운 실험 디렉터리/사전 계획이 필요하다. 원시 로그는ignored
`outputs/foothold_v9_20260922/`, 검증된 산출물은`artifacts/terrain_demo/foothold_v9/`.

## 참고한 개념 — 논문 재현은 아님

- [Belter et al., Single-shot Foothold Selection and Constraint Evaluation](https://arxiv.org/abs/2212.00690):
  명목 발 위치 주변 후보 평가. 본 실험은 논문의 모든 충돌/기구학 제약을 구현하지 않는다.
- [Jenelten et al., Perceptive Locomotion in Rough Terrain — Online Foothold Optimization](https://www.research-collection.ethz.ch/bitstreams/74fa2ff1-c94b-4a65-851f-0e56e94eca81/download):
  지형 인지 발 디딤 위치 선택. 본 실험은 해당 whole-body/swing 제어기를 재현하지 않는다.

## 고정 실험 결과 — 2026-09-22

9학습×98,304,000 = **884,736,000 transitions**를 실행했다.
모든 최종749를2026-09-22T04:24:11.244459Z에 고정한 후20개 신규조건 평가를
실행했다. 고정 후 추가 학습/모델 선별은 없다.

| 방식 | 엄격한1타일 | 연속6타일 | 험지 낙상 | 평지 낙상 |
|---|---:|---:|---:|---:|
| 기존 v5 | 264/300 (88.0%) | 136/300 (45.3%) | 24/300 (8.0%) | 5/50 |
| 발 위치만 | 734/900 (81.6%) | 283/900 (31.4%) | 99/900 (11.0%) | 10/150 |
| 착지 후보 | 741/900 (82.3%) | 416/900 (46.2%) | 91/900 (10.1%) | 11/150 |
| 착지 후보+보상 | 763/900 (84.8%) | 424/900 (47.1%) | 105/900 (11.7%) | 12/150 |

**승격 기준 FAIL, 기존 v5 유지.** 후보+보상은 연속6타일 통과가 v5보다
1.8%p 높지만1타일 통과와 낙상이 악화됐다. 같은 예산의 후보 관측만 대비
보상 추가는1타일+22,6타일+8과 함께 낙상+14를 만들었다. 단순히 보상을
더 넣으면 안정적이라는 가설은 지지되지 않는다.

후보 관측만 넣은 조건은 발 위치만 대비 세 시드 모두6타일이 개선됐다
(86→132,94→145,103→139/각300). 다만 레인 이탈도47→66/900으로 늘어
더 긴 주행을 안전 개선과 동일시할 수 없다. 후보+보상은 레인 이탈30/900,
기존v5는0/300. world이탈은 모두0이다.

### 고난도 돌다리

| 방식 | 난이도0.8 1타일 | 난이도1.0 1타일 | 난이도1.0 낙상 | 고난도6타일 |
|---|---:|---:|---:|---:|
| 기존 v5 | 6/10 | 5/10 | 1/10 | 0 |
| 발 위치만 | 15/30 | 18/30 | 3/30 | 0 |
| 착지 후보 | 22/30 | 22/30 | 4/30 | 0 |
| 착지 후보+보상 | 23/30 | 21/30 | 4/30 | 0 |

돌다리 전체 난이도를 합친6타일은 v5 0/50, 발 위치만1/150,
후보25/150, 후보+보상14/150이다. **개선은 낮은 난이도에 있고 고난도
연속 극복은 여전히 해결하지 못했다.** 최고1.0의70% 또는73.3%를 모든 험지
성공으로 일반화하지 않는다. 기준선과 학습 정책의 분모가 다르며 실제 신규
지형 맵은2개뿐이다. 통계적 우월성이나 실물 성공을 주장하지 않는다.

### 후보 없음 진단

학습geometry51/reset24의 별도300step(5초),family/level당1env 진단이다.
고난도1.0 발 평균 후보 유효율은 초기v5 66.4%, guided seed42 75.8%였다.
훈련모델도 약24.2%의 발·스텝에서 후보가 없다. 이때 보상은0으로 기권한다.
후보율이 낮아지는 다른 난이도도 있어 안정적 지지나 보상 해킹의 부재를
보장하지 않는다. `availability_diagnostic.json`에 전 수준 수치를 보존하며
위 통과율/승격 표본에는 포함하지 않는다.

## 검증과 영상

- CPU 회귀 테스트 **151개 통과**, compileall/전체shell구문/diff·whitespace 검사 통과.
  별도Python LSP/typechecker/Ruff는 미설치이며 추가 설치하지 않았다.
- 64env2iter PPO→35env saved-model 평가,256env10iter 프로파일,
  4096env2iter 용량 점검,35env scan/FK/reset/wrap probe 통과.
- 독립 감사가 집계기를 import하지 않고20rawJSON/3500firstepisode를
  다시 계산했다. **불일치0**.35source/10checkpoint/9실제설정 및
  freeze-before-holdout 시간 순서, 기존v7/v8와부모해시를 확인했다.
  감사기에서JSON객체키순서를잘못가정한문제1건은집합비교로수정해기록했다.
- 대표 영상은 사전고정seed42/geometry64/reset38/돌1.0,16초/480frame/30fps/
  1280×720이며두파일전체decode통과.15.9초caption상v5 15.3m,
  guided 15.9m, 양쪽reset0. 마지막시간제한리셋등을편집하지않았다.
  이는정성사례이며위통계나성공기준을대체하지않는다.

[비교 화면](../artifacts/terrain_demo/foothold_v9/index.html) ·
[독립 감사](../artifacts/terrain_demo/foothold_v9/independent_review.md) ·
[모든 수치](../artifacts/terrain_demo/foothold_v9/summary.json) ·
[영상 조건/해시](../artifacts/terrain_demo/foothold_v9/video_manifest.json)

### 변경 범위와 단순화

`foothold_math.py`, `foothold_policy.py`, 별도v9 task/runner설정 및
prepare/probe/fixed-experiment/summary/render/viewer 스크립트와네개테스트파일을
추가했다. 기존학습/평가/demo를opt-inlauncher로재사용하고 기존v7FK·scan인코딩을
재사용했다. CNN/잔차/새의존성/접촉센서/IK계층은추가하지않았다. 과거실험의
해시대상소스와checkpoint는그대로이며원격Git/커밋/실물제어는수행하지않았다.
