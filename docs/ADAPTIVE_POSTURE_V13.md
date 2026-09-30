# v13 — 깊이에 따른 몸체 높이·발 여유 높이 보상

## 목적과 비교 방법

사용자 요청: **장애물이 있으면 몸체를 높이고, 없으면 몸체를 낮춰 속도를 높인다.**
지면 거리 기반 발 높이 보상도 함께 시험한다. 낮은 자세 자체가 빠른 보행을
보장하지 않으므로, 자세 변화와 실제 속도·낙상·통과율을 따로 검증한다.

- 같은 v10 anchored seed42의 최종749 모델에서 출발한다. 학습된88D
  actor/critic와 고정v5 teacher의 모든 tensor를 보존하고, 탐색std만0.2,
  Adam 상태와 iteration만 초기화한다.
- `control`: 원래 보상으로 추가 학습. `adaptive`: **새 보상 한 항목만** 추가.
- 각4096환경×32step×250iteration = **32,768,000 transition**.
  학습seed45, 지형seed75, PPO·교사prior·관측·랜덤화 설정은 동일하다.
- 마지막249 checkpoint만 평가한다. 보상이 큰 중간 모델이나 유리한seed를
  사후 선택하지 않는다. 원래v5/v10/v11/v12 코드는 변경하지 않는다.
- 평가는 새지형78/reset52,79/reset53의 동일 첫episode 조건에서
  v5, 원래v10, control, adaptive와 각v10계열의v12 history hybrid를 비교한다.
  **16초 혼합지형2450회**와 **64초 최고난도 돌다리140회**를 별도로 집계한다.

## 보상

관측에 이미 사용하는 이상적인825개 ray height scan을 훈련 보상 계산에도 사용한다.
실제 RGB-D 영상은 아니다. 배우의 입력88D와 압축된 발판 힌트는 그대로이며,
전체scan과 아래 목표높이를 배우에게 새 관측/명령으로 전달하지는 않는다.

1. 몸체 아래지면과 전방0.3–1.5m의 높이 차이를 이용해 난도를0–1로 계산한다.
   균일한 내리막 단차도 평지로 잘못 분류하지 않도록 몸체 아래지면을 포함한다.
   각영역의 유효깊이가90% 미만이면 새보상은0: **모르는 지형을 평지로 취급하지 않는다.**
2. 지면 대비 몸체 목표높이는 평지 **0.44m**, 큰단차에서 **0.58m**.
   높이 오차에 bounded cost를 준다. 기존0.31m 낙상 경계는 바꾸지 않는다.
3. 각발 끝 주변±0.15m의 유효scan에서 가장높은 지면을 사용한다. 발capsule의
   바닥과 지면 간 목표여유는 **0.04–0.12m**. 상대전진 스윙속도에 따라 부족분을
   벌준다. 각속도에 의한 발끝속도를 포함한다. 실제 접촉센서 판정은 아니므로
   stance를 완벽히 배제하는 보상이라고 주장하지 않는다.
4. 평지에서는 낮은 목표자세와 **전진속도**가 함께 맞을 때만 양의보너스가 있다.
   정지·후진·뒤집힌 상태는 양의속도보너스를 받지 않는다.

새항목은 `−body_cost −0.25×foot_cost +0.5×flat_speed_bonus`, 범위[−1.25,0.5].
원래전진·에너지·낙상·돌다리복구 보상은 두학습군 모두 그대로 유지한다.
특히 기존돌다리 저자세 벌점은 추가보상과 상호작용할 수 있다.

## 검증과 해석

- 새학습의 초기root/joints/**전체88D 관측**/policy/RNG 해시와 실제저장YAML을
  정확히 비교한다. 차이를 반올림해 허용하지 않는다.
- Isaac Lab의 첫terrain 생성과OBJ 캐시재로드 미세차이를 피하기 위해 훈련/평가
  지형을 미리생성하고 캐시파일을 고정한다. 평가입력manifest의SHA도 각결과와
  명령에 남긴다. 공유cache의 동시외부변경을 완벽히 막는 OS격리는 아니다.
- 자세측정은 행동 **직전**, 각환경의 **첫episode에만** 누적한다.
  높이·목표오차·발여유·저속비율과 유효scan/motion 비율을 보고한다.
  서로다르게 방문한 상태의 평균이므로 자세변화의 인과적효과라고 해석하지 않는다.
- 단독adaptive는 control보다 통과·낙상·이탈이 악화되지 않으면서 평지속도가
  엄격히 높아야 주기준PASS이다. 하이브리드는 평지의 고정v5 동작을 유지하고
  적어도 한 험지지표가 개선되어야 한다. **고정v5 평지속도는 새학습 성과가 아니다.**
- 학습seed 하나와지도 두개뿐이다. 모든맵 우월성, 통계적확정, 실제카메라/로봇
  안전성을 증명하지 않는다. 별도64초 결과가16초 주기준 실패를 뒤집지는 않는다.

## 코드와 실행

환경은 상위디렉터리의 기존 `../run-python`을 사용한다. 새의존성은 추가하지 않는다.

```bash
# CPU회귀 테스트
../run-python -m pytest tests/test_posture_*.py

# 초기모델 준비(새경로 필요, 기존파일 덮어쓰기 금지)
../run-python scripts/posture_v13.py prepare --output <새로운-init.pt>

# 학습 런처: 필수인자와 실제 시작tensor/Adam을 검증
../run-python scripts/posture_v13.py train --dry-run --arm adaptive \
  --audit-output <새로운-audit.json> --headless --device cuda:1 \
  --num_envs 4096 --max_iterations 250 --seed 45 \
  --run_name v13_paired_adaptive --resume --load_run v13_init \
  --checkpoint model_0.pt env.scene.terrain.terrain_generator.seed=75
```

전체실험 단계는 `scripts/run_posture_study.py`의
`capacity → freeze_train → train → freeze_eval → prepare_eval → evaluate → horizon → report`.
기존증거가 있으면 재실행/덮어쓰기를 거부한다. 재현시 원본아티팩트를 지우지 말고
별도작업공간/명시적인 새실험 버전을 사용한다. `capacity` 전에 지형75캐시를,
`freeze_train` 전에 CPU/GPU점검과preflight 증거를 준비해야 한다.

핵심구현:
- `src/week03_ant/posture_math.py`: 순수tensor 보상과유효성검증
- `src/week03_ant/tasks/posture_v13_cfg.py`: 원래훈련설정 상속, 추가항목 하나
- `scripts/posture_v13.py`: 정확한warm start/설정/초기상태 검사
- `scripts/evaluate_posture_v13.py`: 원래물리환경, 새모델 명시적평가
- `scripts/summarize_posture_v13.py`: 원시episode/전환/자세/출처 재검산

[사전계획](experiment_plans/adaptive_posture_v13.md) ·
[설계검토](../artifacts/terrain_demo/adaptive_posture_v13/design_review.md) ·
[학습구현검토](../artifacts/terrain_demo/adaptive_posture_v13/training_review.md)

## 결과

두학습군 모두250iteration, 합65,536,000 transition을 완료했다. 최종모델·Adam·
TensorBoard scalar 전체가유한하고, 교사는원래v5와동일하다. 단독actor와동일v12
전환기를사용한hybrid는별도평가다. 비교기준은단순v10원본이아닌 **동일예산추가학습control**.

### 16초 혼합지형 — 독립감사 완료

제어기마다험지300회와평지50회,총2450firstepisodes. 아래통과/낙상/이탈은험지만,
평지속도는낙상episode를포함한각episode 거리/활성시간의평균이다.

| 제어기 | 1타일 | 6타일 | 낙상 | 이탈 | 평지낙상/50 | 평지속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| v5 | 257 | 132 | 33 | 1 | 3 | 9.5833 |
| 원래v10 | 254 | 135 | 40 | 4 | 1 | 10.6396 |
| 추가학습control | 262 | 139 | 37 | 0 | 2 | 10.5796 |
| 새보상adaptive | 260 | 134 | 39 | 0 | 4 | 10.8702 |
| 기존history hybrid | 270 | 143 | 30 | 0 | 3 | 9.5833 |
| control history hybrid | 250 | 143 | 45 | 3 | 3 | 9.5833 |
| adaptive history hybrid | 262 | 142 | 34 | 4 | 3 | 9.5833 |

world이탈은모두0. 단독/하이브리드의사전주기준은 **둘다FAIL**.
단독은평지속도가2.75%증가했지만통과와낙상이악화되었다. 하이브리드는control보다
낙상이줄었지만6타일통과1회감소·이탈1회증가이며, 원래history보다우월하지도않다.
모든hybrid의평지원시episode/전환/자세배열은v5와정확히같다.

| 단독actor 자세측정 | control | adaptive |
|---|---:|---:|
| 평지family 평균몸체여유높이 | 46.11cm | 45.38cm |
| rough bin 평균몸체여유높이 | 47.55cm | 47.42cm |
| rough bin 평균목표높이절대오차 | 11.46cm | 11.59cm |

**평지에서낮아지고빨라지는부분은관찰했지만, 혼합지형의장애물에서몸체를더높이는효과는
입증하지못했다.** 자세값은각모델이방문한유효상태의조건부평균이며동일상태의인과비교가
아니다. 유효자세coverage는control99.9736%/adaptive99.9765%, motioncoverage100%.
평지낙상도2/50→4/50으로늘어기본모델교체를뒷받침하지않는다.

[독립주평가감사](../artifacts/terrain_demo/adaptive_posture_v13/primary_review.md) ·
[원시재검산JSON](../artifacts/terrain_demo/adaptive_posture_v13/independent_primary_audit.json)

### 별도64초 최고난도돌다리

제어기마다20회,총140firstepisodes. 주평가와합치지않는다.

| 제어기 | 1타일/20 | 6타일/20 | 낙상/20 | 이탈/20 |
|---|---:|---:|---:|---:|
| v5 | 12 | 0 | 4 | 0 |
| 원래v10 | 11 | 7 | 7 | 1 |
| 추가학습control | 5 | 5 | 11 | 4 |
| 새보상adaptive | 12 | 10 | 6 | 2 |
| 기존history hybrid | 12 | 9 | 7 | 1 |
| control history hybrid | 12 | 7 | 6 | 4 |
| adaptive history hybrid | 10 | 7 | 5 | 7 |

world이탈모두0. **단독actor의별도기준PASS**, hybrid는1타일감소·이탈증가로FAIL.
adaptive의6타일성공은control25%→50%(5→10/20), 낙상55%→30%(11→6/20).
원래v10의6타일은7/20, 기존history는9/20이었다. 반복학습seed 하나·각20회의
탐색적결과이며, 주평가실패를뒤집거나모든맵우월성을증명하지않는다.

이돌다리의rough bin에서단독actor 평균몸체여유높이는43.60→44.16cm,
목표높이절대오차15.45→14.78cm, 유효발여유높이12.17→13.33cm였다.
일부자세변화는관찰됐지만조건부방문상태평균이다. 전진속도1m/s미만의활성시간
비율은55.20%→57.01%여서정체문제까지해결했다고하지않는다.

### 최종결론과보존

**구현·학습·사전비교는완료했지만전체성능승격은보류한다.** 평지낮추기/속도와
장시간돌다리단독actor에는장점이있으나, 혼합험지낙상과hybrid이탈trade-off가남았다.
기존기본모델을덮어쓰지않고두최종249 checkpoint와28평가파일을별도보존한다.

- [독립최종감사](../artifacts/terrain_demo/adaptive_posture_v13/final_review.md) · [재검산JSON](../artifacts/terrain_demo/adaptive_posture_v13/independent_final_audit.json)
- [전체수치·판정](../artifacts/terrain_demo/adaptive_posture_v13/summary.md)
- [가족/난도·자세·저속·coverage 전체JSON](../artifacts/terrain_demo/adaptive_posture_v13/summary.json)
- [학습검증](../artifacts/terrain_demo/adaptive_posture_v13/training_validation.json)
- [최종모델목록](../artifacts/terrain_demo/adaptive_posture_v13/trained_models.json)
- [동결소스·모델](../artifacts/terrain_demo/adaptive_posture_v13/frozen.json)

총485개CPU테스트,64/4096env 학습smoke,35env평가smoke,기존v12평가기와41필드
정확일치검사를통과했다. 훈련관련12·새실험전체17·원래72source와기존4모델불변.
Ruff/mypy/pyflakes는설치되지않아compileall/회귀테스트/독립리뷰로보완했다.
원격Git작업과추가영상생성은이번실험에서하지않았다.

새모델의비정량미리보기평가예시(결과를기존holdout에섞지않는다):

```bash
../run-python scripts/evaluate_posture_v13.py --headless --device cuda:1 \
  --phase smoke --controller adaptive --geometry 51 --seed 24 --num_envs 35 \
  --checkpoint artifacts/terrain_demo/adaptive_posture_v13/runs/adaptive/model_249.pt \
  --output outputs/v13_adaptive_preview_new.json
```


최종독립감사는2590episode와1447전환event,모든집계·판정을재검산해증거/제한된
보고를승인했다(모델승격승인아님). 명령로그34개현재내용/해시/순서는검증했지만,
주평가시점의전체명령ledger prefix SHA는미보존이므로역사적append-only바이트
동일성을입증했다고하지않는다. 공유cache의순간적외부변경도완전히배제하지못한다.
