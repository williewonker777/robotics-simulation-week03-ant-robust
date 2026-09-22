# Robust Ant PPO — Robotics Simulation Week 03

Isaac Lab의 `Isaac-Ant-v0`에서 **같은 PPO와 sample budget을 유지한 채 domain
randomization이 처음 보는 물리 조건의 일반화를 개선하는가**를 검증한 재현 가능한
프로젝트입니다.

> 공개 저장소 URL: https://github.com/williewonker777/robotics-simulation-week03-ant-robust

## 최신 험지 실험 안내 — 2026-09-22

- **[전체 진행 과정·결과·실패 사례 (v0~v10)](docs/EXPERIMENT_HISTORY.md)**
- 현재 권장 모델은 **v5 portal-rehearsal round4**입니다. 최초 과제 모델이 아니라,
  험지 복구 학습을 거친 60D 기준선입니다. v6~v10은 별도 센서/학습 방법을 시험했지만
  사전 정의한 전체 교체 기준을 통과하지 못했습니다.
- 최신 [v10 실험 방법·16초/64초 결과](docs/PRIOR_V10.md) ·
  [16초 집계](artifacts/terrain_demo/prior_v10/summary.md) ·
  [64초 고난도 진단](artifacts/terrain_demo/prior_v10/horizon_summary.md) ·
  [영상 비교 페이지](artifacts/terrain_demo/prior_v10/index.html)
- **[공개 파일 범위·검증·재현 시 주의사항](docs/PUBLICATION.md)**:
  코드, 평가 데이터, 선택/비교 모델, 영상, 계획과 검토 기록을 포함합니다.
  로컬 실행 로그, TensorBoard 원본, 에이전트 상태와 자격증명은 제외합니다.

아래 원래 과제의 동일-budget PPO 결과와 후속 험지 실험은 서로 다른 실험입니다.
v6 이후의 깊이 정보는 이상적인 ray/height scan이며 실제 RGB-D 카메라 검증은 아닙니다.

## 결론

가설은 **지지되었지만 효과의 크기는 조건별로 달랐습니다.** Baseline과 Robust를 각각
3개 학습 seed로 반복하고, 각 checkpoint를 100개 vectorized environment에서 한 episode씩
평가했습니다. 3-seed pooled 결과에서 Robust는 Baseline 대비 ID `+3.6%`, 저마찰
`+18.1%`, 추가 하중 `+12.0%`, 강한 외란 `+2.9%`의 평균 return 개선을 보였습니다.
세 공개 OOD 조건의 단순 평균은 `+10.8%`입니다.

| Variant | Scenario | Train seeds | Episodes | Return mean ± population std | Mean length |
|---|---:|---:|---:|---:|---:|
| Baseline | ID | 3 | 300 | 139.38 ± 27.81 | 928.3 |
| Baseline | Low friction | 3 | 300 | 129.37 ± 25.95 | 928.0 |
| Baseline | Heavy | 3 | 300 | 127.90 ± 32.20 | 911.9 |
| Baseline | Push | 3 | 300 | 139.53 ± 25.65 | 931.2 |
| Friction | ID | 1 | 100 | 153.20 ± 31.79 | 925.6 |
| Friction | Low friction | 1 | 100 | 153.19 ± 35.50 | 925.3 |
| Friction | Heavy | 1 | 100 | 142.46 ± 32.67 | 925.9 |
| Friction | Push | 1 | 100 | 154.68 ± 22.65 | 942.6 |
| Robust | ID | 3 | 300 | 144.36 ± 33.31 | 919.3 |
| Robust | Low friction | 3 | 300 | 152.84 ± 30.67 | 928.9 |
| Robust | Heavy | 3 | 300 | 143.19 ± 31.86 | 922.6 |
| Robust | Push | 3 | 300 | 143.58 ± 32.97 | 916.7 |

`±`는 환경별 episode return을 합친 population standard deviation입니다. 학습 seed별
100-env 결과와 seed mean 분산은 각각
[`evaluation_per_run.csv`](artifacts/evaluations/evaluation_per_run.csv)와
[`evaluation_summary.csv`](artifacts/evaluations/evaluation_summary.csv)에 있습니다.

### 해석

- 랜덤화 범위 밖인 저마찰에서 가장 큰 개선이 나타나 물성 랜덤화의 일반화 효과가
  가장 명확했습니다.
- 30% torso payload와 COM shift에서도 return과 episode length가 함께 개선됐습니다.
- 강한 push에서는 return이 소폭 개선됐지만 Robust의 mean length는 낮았습니다. 외란
  일반화가 모든 seed에서 안정적이었다고 보기는 어렵습니다.
- Friction-only는 한 seed에서 네 조건 모두 강했으며, 복합 설계의 이득 상당 부분이
  단순 마찰 랜덤화로도 얻어질 수 있음을 시사합니다. 다만 `n=1`이므로 seed 수준의
  결론에는 사용하지 않았습니다.
- Robust의 seed-mean 표준편차가 Baseline보다 컸습니다. 평균 일반화와 학습 안정성
  사이의 trade-off가 후속 과제입니다.

## 실험 설계

관측·행동 차원(60D/8D), PPO 구조, `4096` environments, `32` steps/env,
`1000` iterations와 seed 집합을 고정했습니다. 한 run은 131,072,000 transitions이며
7개 학습 run의 총 budget은 917,504,000 transitions입니다.

| Variant | Seeds | Training distribution |
|---|---|---|
| Baseline | 42, 43, 44 | course `Isaac-Ant-v0` |
| Friction ablation | 42 | static/dynamic friction과 restitution |
| Robust | 42, 43, 44 | friction + torso mass/COM + reset state + observation noise + interval push |

공개 holdout은 Robust 학습 범위 밖의 저마찰, torso mass `×1.30` + off-center COM,
`±0.80 m/s`의 강한 push입니다. 실제 hidden 환경은 과제 지침대로 사용하지 않았습니다.
정확한 범위와 공정성 기준은
[`EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)와
[`experiment_matrix.yaml`](configs/experiment_matrix.yaml)에 있습니다.

## 추가: 다중 지형 보행 데모

과제 본 실험과 별도로 flat, random rough, slope, stairs 네 지형을 포함하는 procedural
terrain curriculum을 추가하고 별도 정책을 학습했습니다. 선택 정책은 100-env 평가에서
평균 return `33.00`, 평균 episode length `595.15/960`을 기록했으며, 네 지형별 상세
수치와 학습 재현 명령은
[`artifacts/terrain_demo/README.md`](artifacts/terrain_demo/README.md)에 있습니다.

후속 v1 정책은 지형 원점 기준의 조기 낙상 판정을 전복 각도 판정으로 바꾸고 전체 지형에서
추가 학습했습니다. 선택 checkpoint는 평균 length `637.17/960`, 완주 `23/100`으로
개선됐습니다.

네 지형을 순서대로 따라가는 라이브 GUI 데모:

```bash
./scripts/run_terrain_demo.sh \
  --task Week03-Ant-Terrain-Posture-v1 \
  --device cuda:0 \
  --seed 7 \
  --cycle-seconds 7 \
  --checkpoint artifacts/terrain_demo/runs/terrain_posture_full_seed42/model_2800.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

### v2: 복잡 지형 확장

v1의 네 지형에 **파도, 불연속 장애물, 징검다리**를 더해 flat/rough/slope/stairs/
waves/obstacles/stepping-stones의 7개 지형군으로 확장했습니다. 60D 관측과 8D
행동 인터페이스는 유지하고, v1 정책을 낮은 난이도(0–0.35)로 warm-up한 뒤 전체
난이도 curriculum에서 2,200 iteration 추가 학습했습니다. 생존을 기준으로 선택한
`model_5500.pt`는 seed 24의 100-env 평가에서 평균 `675.92/960` step, 완주
`41/100`, return `31.82 ± 18.84`를 기록했습니다. 다섯 seed(7, 24, 42, 43, 44)의
평균은 `690.5/960` step, `40.6/100` 완주였습니다. 이는 더 넓은 지형 분포에서의
측정 결과이며 임의의 극한 지형까지 보장한다는 의미는 아닙니다.

7개 지형을 순서대로 따라가는 v2 데모(환경 수는 지형 설정에서 자동 결정):

```bash
./scripts/run_terrain_demo.sh \
  --task Week03-Ant-Terrain-Complex-Posture-v2 \
  --device cuda:0 \
  --seed 7 \
  --cycle-seconds 7 \
  --checkpoint artifacts/terrain_demo/runs/terrain_complex_full_seed42/model_5500.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

### v3: 극한 지형과 선행 스캔

v3는 flat/rough/slope/stairs 수준을 넘어 waves, deep stairs, 불연속 장애물,
stepping-stones, **pit/gap/boxes**를 포함한 10개 지형군으로 확장했습니다. 최대
난이도는 pit 깊이 `0.55 m`, gap 폭 `0.65 m`, box 높이 `0.45 m`이며, 8개 난이도
row를 mild → approach → full curriculum으로 노출합니다. torso 장착 54-ray
height scanner를 추가해 정책 입력을 60D에서 **114D**로 늘렸고 행동은 8D를
유지했습니다. RayCaster reset bookkeeping 때문에 v3 scene은 `clone_in_fabric=False`로
설정했습니다.

선택 checkpoint:

```text
artifacts/terrain_demo/runs/terrain_extreme_full_seed42/model_7150.pt
```

SHA-256: `22b41aed1a7f4c7e2b9e66b5cdb4a61125f4f8f64be52d1766255d5b5baec139`

동일한 극한 분포(seed 24, 100 env)에서 no-scan 정책은 평균 `447.05/960` step,
완주 `23/100`이었고, 선택 scanner 정책은 `598.79/960` step, `33/100`으로
개선됐습니다. 다섯 seed(7, 24, 42, 43, 44)의 평균은 `581.10/960` step,
`32.8/100` 완주입니다. pit과 gap은 여전히 가장 어려운 family이므로 이 수치는
측정한 분포에 대한 검증 결과이지 임의의 미지 지형에 대한 보장은 아닙니다. 상세한
지형별 breakdown, raw JSON, 재현 명령은
[`artifacts/terrain_demo/README.md`](artifacts/terrain_demo/README.md)에 있습니다.

10개 지형 v3 데모:

```bash
./scripts/run_terrain_demo.sh \
  --task Week03-Ant-Terrain-Extreme-Posture-v3 \
  --device cuda:0 \
  --seed 7 \
  --cycle-seconds 7 \
  --checkpoint artifacts/terrain_demo/runs/terrain_extreme_full_seed42/model_7150.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

### v4: 극한 지형 회복 curriculum

v3의 114D scanner 인터페이스를 유지하면서 scanner를 진행 방향으로 `0.60 m`
이동하고, geometry-only warm-up → approach → 전체 10개 지형 순서의 회복 curriculum을
추가했습니다. 전복 판정은 `1.42 rad`로 완화하고, gap/pit을 넘는 데 필요한 토크의
행동·에너지 패널티를 낮췄습니다. 마지막 pit-focus 단계는 pit/gap을 더 자주 보여주고
bounded clearance 보상을 사용해 단절 지형의 회복을 보강합니다.

선택 checkpoint:

```text
artifacts/terrain_demo/runs/terrain_extreme_recovery_seed42/model_9297.pt
```

seed 24/25/26의 100-env full 평가에서 평균 episode length는 `587.1/960`, 완주는
평균 `27.7/100`이었고, v4 approach 대비 gap 평균 길이는 `692.6 → 722.6`, pit은
`224.1 → 272.2` step으로 늘었습니다. seed 24에서는 10개 family 중 pit `3/10`,
gap `5/10`, stepping-stones `8/10`이 960 step을 완료했습니다. 이 결과는 저장된
procedural 분포에 대한 측정치이며 임의의 미지 지형을 보장하지 않습니다. 상세 breakdown,
재현 명령과 SHA-256은 [`artifacts/terrain_demo/README.md`](artifacts/terrain_demo/README.md)에
있습니다.

10개 극한 지형 v4 데모:

```bash
./scripts/run_terrain_demo.sh \
  --task Week03-Ant-Terrain-Extreme-Recovery-v4 \
  --device cuda:0 --seed 7 --cycle-seconds 7 \
  --checkpoint artifacts/terrain_demo/runs/terrain_extreme_recovery_seed42/model_9297.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

![Training curves](artifacts/plots/training_curves.png)

![Evaluation returns](artifacts/plots/evaluation_returns.png)

### v5: 과제 호환 60D 험지 보행 재개 (2026-09-21)

무한 반복 레인에서 재학습하고, 생존뿐 아니라 **실제 8 m 험지 구간 통과 + 16초 무낙상 +
발까지 포함한 레인 유지**를 검사했습니다. 3 seed의 험지 450 episode에서 기존 v5 대비
낙상률 **16.0% → 7.1%**, 통과율 **65.1% → 82.0%**, 전진 속도 **2.94 m/s**입니다.
요철·경사·계단·장애물·파도 통과율은 **85.3–96.0%**지만, 징검다리는 **32.0%**이며
난이도 0.8/1.0은 아직 통과하지 못했습니다. 전체 6개 타일 완주율은 별도로 **43.8%**입니다.

과제 평지 return은 **134.80 ± 29.67**로 원래 robust 정책의 **87.3%**를 유지합니다.
새 지형 seed 52/53에서도 통과율 **77.3% / 78.0%**를 확인했습니다.
[상세 결과·한계·재현 명령](docs/ROUGH_V5.md),
[7개 지형 영상](artifacts/terrain_demo/rough_v5_selected_seed7.mp4).


### v5 후속: 징검다리 복구 검증 완료

발이 돌 사이에 빠져 정체하는 원인을 진단하고, 복구 동작과 기존 안정형 동작을
**단일 60D/8D 정책**으로 함께 학습했습니다. 평가 지형과 성공 기준은 바꾸지 않았습니다.

- 기존 3-seed 벤치마크: 험지 통과 **369/450 → 403/450 (89.6%)**,
  징검다리 **24/75 → 56/75 (74.7%)**. 전체 험지 낙상은 **32회로 동일**.
- 선택 후 새 시드·지형: 통과 **472/600 → 517/600 (86.2%)**,
  징검다리 **33/100 → 69/100**. 전체 험지 낙상은 **61회로 동일**.
- 평지 ID return **137.07 ± 26.83**, 기존 robust의 **88.8%** 유지.
- 권장 체크포인트: [`rough_v5_portal_rehearsal4_seed43/model_round_4.pt`](artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt).
  이전 모델은 그대로 보존했습니다. 원래 과제의 equal-budget 실험과 별개인 후속 학습입니다.

**일부 실패는 남아 있습니다.** 난이도 0.8 징검다리 통과는 40%이고, 전체 6개 타일
통과는 징검다리에서 1/75에 그칩니다. 일부 지형·평탄 레인의 낙상도 소폭 늘어
모든 지형의 안전성을 보장하지 않습니다.

[상세 결과·재현 명령·한계](docs/ROUGH_RECOVERY.md) ·
[개선 모델 보행 영상](artifacts/terrain_demo/rough_v5_rehearsal_stones08_seed7.mp4) ·
[동일 조건의 기존 모델 영상](artifacts/terrain_demo/rough_v5_reference_stones08_seed7.mp4)

```bash
./scripts/run_terrain_demo.sh --task Week03-Ant-Rough-Lanes-Demo-v5 \
  --device cuda:0 --seed 7 --cycle-seconds 8 \
  --checkpoint artifacts/terrain_demo/runs/rough_v5_traverse1499_seed42/model_1499.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

### v7: 논문 기반 발 위치 지도 비교 (2026-09-22)

높이맵 CNN과 **4개 발 위치 지도**를 구현하고, 무지형/높이맵/높이맵+발지도에
각각3학습seed·동일750iteration을 적용했습니다(총8억8,473만6천step). 모델을 먼저
고정한 뒤 신규지형2종, 총3,500firstepisode를 평가했습니다.

- 높이맵→발지도: strict통과755→760/900, 낙상108→85/900,6타일144→216/900.
- 그러나 기존v5의6타일45%보다 새발지도24%가 낮아 **교체gate FAIL, 권장v5유지**.
- 최고난도돌다리: 높이맵18/30→발지도16/30. 고난도극복완료를 주장하지 않습니다.
- 514D/8D의 별도실험이며 실제RGB-D카메라·GRU·residual정책은 아닙니다.

[비교 영상·그래프](artifacts/terrain_demo/footmap_v7/index.html) ·
[설계·한계·재현](docs/FOOTMAP_V7.md) ·
[전체결과](artifacts/terrain_demo/footmap_v7/summary.md) ·
[모든seed·지형수치](artifacts/terrain_demo/footmap_v7/summary.json) ·
[독립재계산검증](artifacts/terrain_demo/footmap_v7/verification.json)

### v8: 기존 보행 고정 + 지형 잔차 정책 (2026-09-22)

기존 v5 actor는 고정하고 `0.5*tanh` 행동 평균 보정(각 성분 ±0.5 이내)만 추가 학습했다.
무지형/지형 잔차 ×3학습seed, 각750iter(총589,824,000transition).
새 지형62/63에서 기존 v5/v7도 같은 조건으로 재평가한3,500first-episode 결과다.

| 방식 | 1타일 통과 | 6타일 통과 | 낙상 |
|---|---:|---:|---:|
| 기존 v5 |259/300 (86.3%)|136/300 (45.3%)|32/300 (10.7%)|
| 발 지도 v7 |753/900 (83.7%)|219/900 (24.3%)|94/900 (10.4%)|
| 무지형 잔차 v8 |712/900 (79.1%)|374/900 (41.6%)|99/900 (11.0%)|
| 지형 잔차 v8 |747/900 (83.0%)|370/900 (41.1%)|96/900 (10.7%)|

연속 통과는 v7보다 회복했지만 기존 v5를 넘지 못했고, 최고난도 돌다리도
12/30 통과에 그쳤다. **전체 교체 기준 FAIL, 권장 v5 유지.**
고정 가중치와 제한된 보정은 폐루프 안전 보장이 아니다. 실제 RGB-D가 아닌
이상적 레이캐스트이며 이전 v7 표와 평가 지형이 다르다.

[비교 영상·그래프](artifacts/terrain_demo/residual_v8/index.html) ·
[구현·검증 설명](docs/RESIDUAL_V8.md) ·
[전체 결과](artifacts/terrain_demo/residual_v8/summary.md) ·
[시드·지형별 수치](artifacts/terrain_demo/residual_v8/summary.json)

### v9: 명시적 착지 후보 + 지지면 보상 (2026-09-22)

825개 고밀도 지형 ray에서 발끝 착지 후보를 추출하는 **88D MLP**를 추가했다.
발 위치만/후보 관측/후보+보상의3조건×3seed를각750iter 학습하고,
모든 최종모델 고정 후 새 지형64/65에서3,500개 첫 에피소드를 평가했다.

| 방식 | 1타일 통과 | 6타일 통과 | 낙상 |
|---|---:|---:|---:|
| 기존 v5 |264/300 (88.0%)|136/300 (45.3%)|24/300 (8.0%)|
| 발 위치만 |734/900 (81.6%)|283/900 (31.4%)|99/900 (11.0%)|
| 착지 후보 |741/900 (82.3%)|416/900 (46.2%)|91/900 (10.1%)|
| 후보+보상 |763/900 (84.8%)|424/900 (47.1%)|105/900 (11.7%)|

연속6타일은 개선됐지만 낙상과1타일 통과가 회귀하여 **교체 기준 FAIL,
기존 v5 유지**다. 최고난도 돌다리1타일은 v5 5/10, 후보22/30,
후보+보상21/30이나 고난도6타일은 모두0이다. 후보 관측만 대비 보상 추가는
낙상이91→105/900으로 늘었다. 실제RGB-D/정확한IK/전체발캡슐 지지계획이
아니며, 후보 없음은 비용0으로 기권한다. 이전표와 평가 지형도 다르다.

151CPU테스트 및 독립20rawJSON/3500episode 재계산에서 불일치0.
[16초 무편집 비교](artifacts/terrain_demo/foothold_v9/index.html) ·
[설계·수치·한계](docs/FOOTHOLD_V9.md) ·
[전체 결과](artifacts/terrain_demo/foothold_v9/summary.md) ·
[독립 감사](artifacts/terrain_demo/foothold_v9/independent_review.md)

### v10: 학습 중에만 기존 v5 행동을 참조 (2026-09-22)

기존 동작을 실행 시 고정하는 v8 대신, **88D 착지 후보 학생을 PPO로 학습하면서
v5 행동 평균과의 차이에만 보조 손실**을 주었습니다. λ0/0.02 각각3seed·750iteration,
총589,824,000step. 모든 최종 모델을 고정한 뒤 새 지도2개에서2,450firstepisode를 평가했습니다.

| 정책 | 6타일 통과 | 낙상 | 레인이탈 |
|---|---:|---:|---:|
| 기존 v5 |132/300 (44.0%)|30/300 (10.0%)|0/300|
| 후보 관측 / prior 없음 |422/900 (46.9%)|96/900 (10.7%)|61/900 (6.8%)|
| 후보 관측 / v5 prior |454/900 (50.4%)|73/900 (8.1%)|15/900 (1.7%)|

전체16초 평가에서는 개선됐지만 **v5 대비 레인이탈 때문에 교체 기준 FAIL, 권장 v5 유지**입니다.
별도64초 최고난도 돌다리는6타일이 v5 **0/20**, prior없음 **14/60**, prior **15/60**입니다.
그러나 prior의 낙상이 **31/60 (51.7%)**로 높아 장시간 안정성·완전 극복은 해결되지 않았습니다.
64초 표본은 기본 평가와 합치지 않았고, 실패 시드도 그대로 보존했습니다.

207CPUtests·53고정소스·7모델·6설정·기본/64초 원시 배열 독립 감사 통과.
실제 RGB-D가 아닌 이상적 레이캐스트이며, 실행 시 teacher는 호출하지 않습니다.
[16초·64초 무편집 비교](artifacts/terrain_demo/prior_v10/index.html) ·
[방법·전체 결과·한계](docs/PRIOR_V10.md) ·
[독립 감사](artifacts/terrain_demo/prior_v10/independent_review.md)

### v6: 깊이 지형 관측 추가 학습

기존 60D에 **143-ray 지형 높이 + 143개 유효성 값**을 추가한 별도 **346D** 정책을
4,096개 환경에서 1,500 iteration 학습했습니다. 렌더링 RGB-D 카메라가 아니라
이상적인 raycast 높이/깊이 스캔이며, 기존 v0–v5와 원래 과제 모델은 보존했습니다.

모델 선택 후 고정한 신규 지형·초기조건 600개 험지 episode에서, 같은 v6 scene의
기존 정책 대비 징검다리 통과는 **62/100 → 79/100**으로 개선됐습니다.
난이도 0.8은 **6/20 → 15/20**입니다. 하지만 전체 통과는 **524/600 → 521/600**,
낙상은 **46 → 62회**, 6개 타일 연속 완주는 **274 → 119회**로 악화됐습니다.
따라서 **전체 지형용 권장 정책은 기존 v5를 유지**하고, v6는 깊이 학습 실험 모델로
별도 제공합니다. 모든 험지를 해결했다는 의미는 아닙니다.

선별 seed24에서 정상 깊이 입력의 통과는 135/150, 입력 제거는 86/150,
다른 환경의 깊이로 교체하면 106/150입니다. 정책의 깊이 입력 의존성은 확인되지만,
동일 예산의 무센서 재학습 대조군이 없으므로 개선을 깊이만의 인과효과로 단정하지 않습니다.

[설계·검증·재현 명령](docs/DEPTH_V6.md) ·
[346D 체크포인트](artifacts/terrain_demo/runs/depth_v6_recovery1499_seed42/model_1499.pt) ·
[실제 센서 관측](artifacts/terrain_demo/depth_v6_sensor_scan.png) ·
[깊이 모델 보행 영상](artifacts/terrain_demo/depth_v6_stones08_seed7.mp4) ·
[동일 v6 scene의 기존 정책 영상](artifacts/terrain_demo/depth_v6_blind_stones08_seed7.mp4)

**고난도 비교:** 기존 신규 검증에서 난이도0.8만 보면 통과99→105/120,
최대난이도1.0은91→85/120이고 낙상19→28회로 악화됐습니다.
[최대난이도 6개 지형 비교 영상](artifacts/terrain_demo/depth_v6_hard/index.html) ·
[난이도별 상세 집계](artifacts/terrain_demo/evaluations/depth_v6/hard_terrain_summary.md).
새 학습이나 독립 평가 표본을 추가한 결과는 아닙니다.

```bash
./scripts/run_terrain_demo.sh --task Week03-Ant-Depth-Lanes-Demo-v6 \
  --device cuda:0 --seed 7 --cycle-seconds 8 \
  --checkpoint artifacts/terrain_demo/runs/depth_v6_recovery1499_seed42/model_1499.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

## 환경

- Python 3.11.16
- Isaac Sim 5.1.0
- Isaac Lab 2.3.0, commit `f50046758743fb7bc913032d1caafc0d7e536164`
- PyTorch 2.7.0+cu128
- RSL-RL 3.0.1
- GPUs used: NVIDIA GeForce RTX 5080 / RTX 5070

```bash
source ../activate.sh
python -m pip install --no-deps -e .
python -m pip install -r requirements-report.txt  # PPT를 다시 만들 때만 필요
```

다른 설치 위치에서는 `ROBOTICS_SIM_CLASS_ROOT`를 course environment root로 지정하면
됩니다.

### 작업 경로

이 공개 저장소가 이 수업의 **canonical task workspace**입니다. 모든 Robotics
Simulation Week 03 태스크의 소스, 학습 로그, 체크포인트, 평가 결과와 데모 산출물은
다음 경로에서 생성·실행합니다.

```text
.
```

실행 스크립트는 자신의 저장소 루트를 자동으로 계산하므로 위 디렉터리에서 바로
호출하면 새 결과도 `logs/`, `artifacts/` 아래에 이 저장소와 함께 남습니다.

## 학습 재현

```bash
./scripts/run_experiment.sh baseline 42 0
./scripts/run_experiment.sh friction 42 0
./scripts/run_experiment.sh robust 42 1
```

개별 명령은 다음과 같습니다.

```bash
./scripts/run_train.sh \
  --task Week03-Ant-Robust-v0 \
  --headless \
  --device cuda:1 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name robust_seed42
```

## 과제 채점용 100-environment 평가 명령

`run_evaluate.sh`는 과제용 [`scripts/play_one_episode.py`](scripts/play_one_episode.py)를
실행합니다. 각 환경의 **첫 episode 누적보상만** 기록한 뒤 100개 return의 평균과
population 표준편차를 terminal과 JSON에 출력합니다.

```bash
./scripts/run_evaluate.sh \
  --task Week03-Ant-Baseline-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 100 \
  --seed 24 \
  --max_steps 960 \
  --checkpoint artifacts/runs/robust_seed42/model_999.pt \
  --output artifacts/evaluations/robust_seed42__id.json \
  --kit_args=--/renderer/multiGpu/enabled=false
```

모든 공개 평가 조건을 한 번에 재현하려면:

```bash
./scripts/evaluate_checkpoint.sh \
  robust_seed42 \
  artifacts/runs/robust_seed42/model_999.pt \
  0
```

평가 task ID:

- ID: `Week03-Ant-Baseline-v0`
- OOD: `Week03-Ant-Test-LowFriction-v0`
- OOD: `Week03-Ant-Test-Heavy-v0`
- OOD: `Week03-Ant-Test-Push-v0`

## 체크포인트 무결성

| Run | SHA-256 (`model_999.pt`) |
|---|---|
| baseline_seed42 | `f4a88747c8d56905c4889b0fb0708d4e924a81dc676d92561d2921e8b2bdf83b` |
| baseline_seed43 | `35a7c93f7b8c901d2db2af0bded36476d643ddccb31925960299f8fb843c0d58` |
| baseline_seed44 | `61389acc39cee4996c3a4e2ab833ea7187151338fe2a17470b316f3df7138678` |
| friction_seed42 | `c0fbd13ecc75abdfd687a4df6a8c84c33eb22bf7905f41ba056751a6d891e9ff` |
| robust_seed42 | `58dc882b6fa38f95ea2bd2d4d5f874836e5e043e0b5ae5c62290180122b57864` |
| robust_seed43 | `e770d3282a447c88794f2f6bb147dd208965cce73a4b3dd9fe369b9ba54de187` |
| robust_seed44 | `f9c84451e14e8d26757c88664aaf524841cc02fefc7d74bc924c03d15cb91e94` |

각 [`artifacts/runs`](artifacts/runs) 하위 폴더에는 checkpoint, TensorBoard event,
Hydra/RSL-RL params와 machine-readable manifest가 함께 있습니다.

## 영상과 발표자료

- 좌우 비교 영상(16초, 1280×420, 60 fps):
  [`comparison_low_friction_seed42.mp4`](artifacts/videos/comparison_low_friction_seed42.mp4)
- Baseline seed 42 / low friction:
  [`artifacts/videos/baseline_seed42_low_friction`](artifacts/videos/baseline_seed42_low_friction)
- Robust seed 42 / low friction:
  [`artifacts/videos/robust_seed42_low_friction`](artifacts/videos/robust_seed42_low_friction)
- 5분 발표자료: [`report/week03_ant_robust_report.pptx`](report/week03_ant_robust_report.pptx)
- 발표 대본: [`report/SPEAKER_NOTES.md`](report/SPEAKER_NOTES.md)

## 저장소 구조

```text
src/week03_ant/          custom task, environment and evaluation utilities
scripts/                 train, play_one_episode, analysis and report commands
configs/                 fair-budget experiment matrix
artifacts/runs/          checkpoints, params, TensorBoard logs and manifests
artifacts/evaluations/   per-checkpoint 100-env JSON and aggregate tables
artifacts/terrain_demo/  multi-terrain checkpoint, evaluations and reproduction notes
artifacts/plots/         learning and evaluation figures
artifacts/videos/        same-seed qualitative rollout evidence
report/                  five-minute PPT and speaker notes
```

## 한계

- hidden evaluation 환경은 발표 당일 공개되므로 결과에 포함하지 않았습니다.
- 학습 seed는 3개이며 통계적 유의성을 폭넓게 주장하기에는 적습니다.
- 공개 holdout은 terrain geometry, actuator latency, correlated shift를 포함하지 않습니다.
- Friction ablation은 seed 42 한 번뿐이라 정성적 attribution에만 사용했습니다.

팀원 정보는 LMS 조 편성 공지 후 [`TEAM.md`](TEAM.md)에 입력해야 합니다. 코드는
BSD-3-Clause로 배포하며 Isaac Lab 파생 파일의 원 저작권 고지를 유지합니다.
