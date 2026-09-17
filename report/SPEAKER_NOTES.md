# Five-minute speaker notes

## Slide 1 — 20 s

과제 목표는 기본 Ant에서만 높은 점수를 내는 정책이 아니라, 처음 보는 물리 조건에서도
걷는 정책을 만드는 것입니다. 저희는 domain randomization이 일반화에 실제로 도움이
되는지 공정한 budget에서 검증했습니다.

## Slide 2 — 55 s

관측 60차원, 행동 8차원과 PPO 설정을 모두 고정했습니다. 한 run은 4096개 환경에서
1000 iteration, 약 1.31억 transition입니다. Baseline과 robust는 seed 42, 43, 44를
반복했고 평가는 동일 seed에서 100개 환경의 첫 episode를 집계했습니다.

## Slide 3 — 55 s

Friction-only는 단일 요인의 효과를 보기 위한 ablation입니다. Robust는 마찰뿐 아니라
torso 질량과 무게중심, 초기 상태, 작은 관측 노이즈와 interval push를 추가합니다.
다만 관측·행동 shape는 그대로라 같은 checkpoint를 모든 holdout에서 평가할 수 있습니다.

## Slide 4 — 45 s

학습 curve에서 NaN이나 발산 여부, 수렴 속도, seed 민감도를 확인합니다. Robust가 더
어려운 분포를 학습하므로 초반 reward가 느릴 수 있으며, 최종 training reward만으로
성공을 판정하지 않습니다. 실제로 baseline은 먼저 상승했지만 약 400 iteration 이후
robust가 따라잡았고, 둘 다 발산 없이 수렴했습니다. 다만 robust의 seed 편차가 더 컸습니다.

## Slide 5 — 90 s

ID와 세 가지 공개 holdout의 100-env mean/std를 비교합니다. Robust는 baseline보다
ID 3.6%, low friction 18.1%, heavy 12.0%, push 2.9% 높았습니다. 따라서 두 개 이상의
OOD에서 개선되고 ID collapse가 없어 사전 판정 기준상 가설을 지지합니다. 그러나 push의
이득은 작고 episode length는 오히려 낮았습니다. Friction-only 한 seed도 매우 강해,
복합 랜덤화 전체가 항상 필요한지에는 추가 seed 실험이 필요합니다.

## Slide 6 — 35 s

모든 명령, config, TensorBoard event, checkpoint와 SHA-256, 총 28개의 100-env JSON,
같은 seed 저마찰 영상, 분석 코드를 공개합니다. hidden test는 사용하지 않았고 세 seed와
세 holdout만으로 모든 일반화를 주장하지 않는다는 한계도 명시했습니다.
