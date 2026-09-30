# v15 — 논문 기반 방향 안정화 보상

## 목적과 적용 범위

v14의 깊이·자세 명령 관측을 그대로 사용하면서 **옆으로 흐르는 이동과 진행
방향에 맞지 않는 회전**을 줄이는 추가 보상을 시험한다. v14의 통과율 저하 원인을
확정한 것은 아니며, 두 항을 합친 보상 유무의 비교다.

- [Miki et al. (2022), Supplement S7](https://arxiv.org/html/2201.08117v1):
  목표 방향에 직교하는 속도를 지수 함수로 억제하는 아이디어.
- [Aractingi et al. (2023)](https://www.nature.com/articles/s41598-023-38259-7):
  명령 선속도·각속도 추종 보상. 논문의 PD 위치 제어와 전체 보상은 재현하지 않았다.
- [IsaacLab 공식 명령 설정](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.commands.commands_cfg.UniformVelocityCommandCfg):
  방향 오차에서 목표 각속도를 만드는 비례 제어 방식.

접촉 기반 발 미끄러짐 보상은 새로운 접촉 센서가 필요해 이번 시험에서 제외했다.
관절 내부 wrench를 지면 접촉력으로 간주하지 않는다.

## 한 가지 변경

기존 목표 방향을 world XY 단위 벡터로 정규화하고, 그에 수직인 **몸체 COM 속도**를
`v_side`로 사용한다. 목표 방향과 몸체 방향의 오차를 `e`라 하면:

```text
w_target = clip(e, -1, 1)                 # rad/s, gain 1
r_direction = -0.75 * (1 - exp(-3 * v_side²))
              -0.25 * (1 - exp(-(w_world_z - w_target)²))
```

보상 범위는 −1~0이다. 세로 이동을 벌주지 않으며, 정상적인 방향 수정 회전을
허용한다. 비정상 입력 또는 거의 수직인 몸체 방향은 보상 0과 별도 무효 표시로
처리한다. 무효 샘플은 추종 성능 평균에 넣지 않고 유효 비율을 함께 보고한다.

world Z 각속도는 기울어진 몸체의 Euler yaw 미분과 정확히 같지 않다. 논문의
양의 보상을 음의 비용으로 옮기면 가변 episode 길이의 유인이 바뀌므로, 논문과
동등한 보상이라고 주장하지 않는다. 정지·저속·낙상도 함께 확인한다.

- **control**: 새 보상 가중치 0.
- **stable**: 새 보상 가중치 1.
- 나머지 91D 관측, conditioned actor/critic, v13 높이 보상, 토크 동작,
  물리·종료·랜덤화, v5 교사 prior 0.02, v12 깊이 히스토리 전환기는 동일하다.
- RewardManager가 가중치와 `dt`를 한 번 곱한다. 수동 진단 보상은 이를 곱하지 않은 값이다.

## 사전 고정 실험

v14 conditioned 최종249에서 학습된 명령 가중치까지 모두 보존해 출발한다.
탐색 표준편차 0.2, 빈 Adam(1e-4), iteration 0만 초기화한다.

- 각 군 4,096환경 × 32step × 250iteration = **32,768,000 transition**.
- 학습 seed47, 지형95. 최종249만 평가하며 중간 모델·계수·seed를 고르지 않는다.
- 새 지형96/reset60 및97/reset61에서 제어기 6개를 동일 초기 상태로 비교한다.
- 주평가: 16초 혼합 지형, 12파일·2,100 첫 episode.
- 별도 평가: 64초 최고난도 돌다리, 12파일·120 첫 episode.
- 비교는 **stable 대 control**, **history_stable 대 history_control**이다.
  v14와 원래 history(v5+v10)는 별도 참고선이다.
- 13.1m/53.1m 통과, 낙상·레인/월드 이탈 판정과 기존 엄격한 개선 기준을 유지한다.
  64초의 이득으로 16초 주평가 실패를 상쇄하지 않는다.

[고정 계획](experiment_plans/directional_stability_v15.md),
[논문 적용 범위](../artifacts/terrain_demo/directional_stability_v15/research.md).

## 결과 — 2026-09-23

**각250회 학습, 총65,536,000 transition과 새 지도24파일·2,220회 평가 완료.**
고정된 동일 예산 대조군 대비 **16초와64초 모두, 단독·하이브리드의 전체 개선 기준은
FAIL**이다. 일부 낙상 감소만으로 기본 모델을 교체하지 않았다.

### 16초 혼합 지형 — 제어기별 험지300회 + 평지50회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v14 | 268/300 | 133/300 | 24/300 | 8/300 |
| control | 268/300 | 160/300 | 30/300 | 2/300 |
| stable | 275/300 | 158/300 | 21/300 | 3/300 |
| history_original | 266/300 | 149/300 | 29/300 | 2/300 |
| history_control | 267/300 | 155/300 | 32/300 | 1/300 |
| history_stable | 270/300 | 149/300 | 26/300 | 2/300 |

- **단독 control→stable**: 낙상30→21회와1타일268→275회는 개선했으나,
  6타일160→158회·레인 이탈2→3회로 전체 기준 FAIL.
- 평지 평균 episode 속도9.9611→10.0834m/s(+1.23%), 낙상3/50→3/50.
  속도는 낙상 episode도 포함한 거리/시간의 평균이다.
- **history_control→history_stable**: 낙상32→26회, 그러나6타일155→149회와
  이탈1→2회로 FAIL. 세 hybrid의 평지 원시 결과는 환경별로 정확히 같다.
- 원래 v14와 비교한 이득을 새 보상의 효과로 돌리지 않는다. control도 동일량
  추가 학습했으며, 새 보상 유무의 비교 대상은 stable 대 control이다.

### 지형별 6타일 통과 — 각 지형50회, 16초

| 지형 | control | stable | history_control | history_stable |
|---|---:|---:|---:|---:|
| obstacles | 31/50 | 29/50 | 27/50 | 25/50 |
| rough | 37/50 | 37/50 | 38/50 | 39/50 |
| slope | 30/50 | 27/50 | 29/50 | 27/50 |
| stairs | 19/50 | 19/50 | 20/50 | 19/50 |
| stepping_stones | 3/50 | 9/50 | 2/50 | 1/50 |
| waves | 40/50 | 37/50 | 39/50 | 38/50 |

돌다리의 단독 단기 통과3→9/50은 부분 이득이지만, 전체 지형 또는 장시간·전환
제어기 우월성으로 확대하지 않는다. 난도0~4별 원시 결과도 JSON에 모두 보존했다.

### 별도64초 최고난도 돌다리 — 제어기별20회

| 제어기 | 1타일 | 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v14 | 4/20 | 3/20 | 12/20 | 4/20 |
| control | 10/20 | 9/20 | 4/20 | 5/20 |
| stable | 10/20 | 8/20 | 8/20 | 3/20 |
| history_original | 10/20 | 5/20 | 10/20 | 0/20 |
| history_control | 11/20 | 11/20 | 7/20 | 2/20 |
| history_stable | 6/20 | 6/20 | 11/20 | 2/20 |

단독6타일9→8/20·낙상4→8/20, hybrid6타일11→6/20·낙상7→11/20으로 모두 FAIL.
모든 평가의 world 이탈은0이다. **새 보상이 지속 험지 극복을 개선했다고 결론 내릴 수 없다.**

### 움직임과 높이 진단

| 단독 control→stable | 16초 혼합 지형 | 64초 돌다리 |
|---|---:|---:|
| 평균 절대 측면속도 (m/s) | 0.6500→0.6509 | 0.5735→0.5694 |
| 평균 절대 방향 오차 (rad) | 0.3121→0.3099 | 0.4007→0.5014 |
| 평균 절대 yaw-rate 오차 (rad/s) | 2.5739→2.6455 | 2.4439→2.6285 |
| 험지 감지 구간 저속(<1m/s) 비율 | 10.67→11.57% | 61.07→58.39% |
| 험지 감지 구간 몸체 높이 | 48.74→48.70cm | 44.58→43.79cm |

방향 진단의16초 평균은 평지를 포함한350회, 64초는20회 전체의 유효 상태를 사용한다.
저속·몸체 높이 행은 험지를 감지한 유효 구간의 값이다. 방향 진단은 이번 방문 상태에서 모두 유효했지만, 이는 실제 센서 신뢰성을 검증한
것이 아니다. 단독 yaw-rate 오차가 줄지 않았으며, 평지 몸체 높이도45.02→45.17cm로
더 낮아지지 않았다. 64초 hybrid의 측면속도/방향 오차는 감소했지만 통과·낙상 성능은
나빠졌다. 진단 개선과 완주 개선은 같지 않다. 각각 다른 방문 상태의 조건부 평균이므로
같은 상태에서의 인과 효과나 실패 원인으로 단정하지 않는다.

## 검증과 한계

- 개발64환경 및4,096환경·2iteration 양쪽 통과. 모델·91D 관측·root/joints·
  CPU/CUDA RNG·정규화 설정이 정확히 짝을 이룬다. v14의 학습된 명령 가중치도 보존했다.
- 개발51/24의 v14 추론18개 비교 필드(결과·보상·전환·자세 진단·초기관측/RNG) 정확일치.
  [원본 비교 증거](../artifacts/terrain_demo/directional_stability_v15/evaluator_parity.json).
- 각38개 TensorBoard 태그×250개 값, 최종 모델과 Adam tensor 모두 유한했다.
  학습 교사는 원래 v5와 정확히 같다. 최종 모델105개 개발episode도 검증했다.
- 훈련/평가 source, 모델, 캐시, 평가 전14행 명령 ledger를 고정했다.
  모든 점수화 과정은 한 번씩만 실행했고 계수·모델·지도 재선별은 없었다.
- 전체822 CPU 회귀 테스트 및 compileall/diff 검사 통과. Ruff/Mypy/Pyright/pyflakes는
  미설치로 실행하지 않았다. 최초 CPU 실행은 수정 중인 가상 증거 fixture와 겹쳐
  14개 실패했으며 로그를 보존했다. fixture 완료 후 전체 통과했고 점수화 재실행은 아니다.
- 기존 frozen source106개와 기준 모델8개는 바뀌지 않았다. 새 의존성·원격 Git 없음.
- 이상적인 ray 높이 정보이며 실제 RGB-D 영상이 아니다. 단일 학습 seed와 두 지도의
  결과로 모든 맵 우월성·통계적 유의성·실물 로봇 안전성을 주장하지 않는다.

[전체 판정과 표](../artifacts/terrain_demo/directional_stability_v15/summary.md) ·
[원시 집계·난도별 진단](../artifacts/terrain_demo/directional_stability_v15/summary.json) ·
[학습 검증](../artifacts/terrain_demo/directional_stability_v15/training_validation.json) ·
[독립 학습 감사](../artifacts/terrain_demo/directional_stability_v15/training_review.md) ·
[독립 최종 감사·분모 보완](../artifacts/terrain_demo/directional_stability_v15/final_review_rough_supplement.md).

최초 독립 감사의350회 전체 지형 합계는 설명용이며, 고정 판정은 험지300회와
평지50회를 분리한다. 이를 보완 감사에서 원시 배열로 다시 계산해 원래 요약과
정확히 일치함을 확인했다. 원래 감사 파일도 보존했고 네 판정은 모두 FAIL로 동일하다.

## 실행과 재현

프로젝트 루트에서 기존 `../run-python` 환경을 사용한다. 실제 실행 인자와 로그는
`artifacts/terrain_demo/directional_stability_v15/commands.jsonl`에 기록한다.
본 실험 경로는 증거 덮어쓰기를 거부하므로 재학습은 별도 격리 복제본에서 수행해야 한다.

```bash
# 개발/용량 확인 → 독립 검토와 preflight 증거 생성 → 고정된 짝학습
../run-python scripts/run_direction_study.py development
../run-python scripts/run_direction_study.py capacity
../run-python scripts/run_direction_study.py freeze_train
../run-python scripts/run_direction_study.py train
../run-python scripts/run_direction_study.py final_smoke

# 최종 모델 검토 뒤 새 지형을 먼저 생성·고정하고 전수 평가
../run-python scripts/run_direction_study.py freeze_eval
../run-python scripts/run_direction_study.py prepare_eval
../run-python scripts/run_direction_study.py evaluate
../run-python scripts/run_direction_study.py horizon
../run-python scripts/run_direction_study.py report
```

학습/평가 완료 후 원시 기록의 판정만 새 출력으로 다시 계산하려면:

```bash
../run-python scripts/summarize_direction_v15.py \
  artifacts/terrain_demo/directional_stability_v15 \
  --output-prefix outputs/v15_reaudit
```

`outputs/v15_reaudit.{json,md}`가 이미 있으면 다른 새 경로를 지정한다. 이 명령은
재학습하거나 지도에서 주행을 다시 하지 않는다. 원본 가중치·캐시·로그 해시가
달라졌으면 실패하도록 되어 있다.
