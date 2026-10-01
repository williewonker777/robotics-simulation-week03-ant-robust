# 팀원 결과에서 고른 v25 전이 가설 — 2026-10-01

두 저장소의 공개 main을 아래 commit으로 고정해 코드·실제 설정·평가 JSON을
확인했다. 외부 코드를 실행하거나 모델을 내려받지 않았다. 외부 수치는 공개 JSON
재계산이며 시뮬레이터 독립 재현이나 checkpoint binary 검증이 아니다.

| 출처 | 근거가 강한 대조 | 관측된 결과 | 이번에 시험할 요소 |
|---|---|---|---|
| [Stick-0](https://github.com/Stick-0/isaac-ant-rough-terrain/tree/3cc718a4214f336fd4db7db5841fa86033b99d35) | 같은 rough 부모·seed42·600iteration, 기존 보상 대 recovery 묶음 | 생존920→969/1024, 낙상104→55, 속도4.399→4.158m/s | clearance/tilt 사전 위험 비용과 행동 변화·몸체 sway 비용의 **부분 적응 포팅** |
| [LimDaeKyung](https://github.com/LimDaeKyung/IsaacLab_RS/tree/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b) | 같은1000iteration, seed42–46의 entropy0 대0.005 | heldout return50.7008→62.1660,5/5 상승; 낙상19.96→21.28% | 기존 우리 entropy0.002를0.005로 증가, recovery가 있는 조건에서 대조 |

Stick의 [동일 예산 결과](https://github.com/Stick-0/isaac-ant-rough-terrain/blob/3cc718a4214f336fd4db7db5841fa86033b99d35/docs/archive/reward_comparison.md)와
[보상 구현](https://github.com/Stick-0/isaac-ant-rough-terrain/blob/3cc718a4214f336fd4db7db5841fa86033b99d35/overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/recovery_mdp.py)은
묶음의 조건부 효과를 보인다. 개별 보상 기여율은 분리되지 않았고 smoothing-only
전 단계는 생존을 개선하지 못했다. 우리에게 이미 있는 speed cap·fall penalty를
중복 추가하거나 global target/종료/대규모 terrain importer를 가져오지 않는다.

Lim의 [전체 결과](https://github.com/LimDaeKyung/IsaacLab_RS/blob/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b/docs/assignment1/RESULTS.md)와
[실제 entropy 설정](https://github.com/LimDaeKyung/IsaacLab_RS/blob/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b/logs/rsl_rl/ant/2026-10-01_02-19-33_f3a_s47/params/agent.yaml)은
탐색 계수 후보를 뒷받침한다. 최고 F3a는 boxes 전용·상대높이·마찰 설정과 추가600회
학습도 포함하므로 전부 entropy 덕분이라고 하지 않는다. 외부 lockbox는 최종 승격에
한 번 쓰였고 전체 seed 평가는 사후 평가다. 우리 평가지형/종료를 바꾸지 않는다.

## 우리 코드에 맞춘 변경·한계

- 기존91D/8D·v5 teacher prior·상대높이·lane 보행을 유지한다. Stick의 고정0.48m
  경고를 그대로 더하면 기존0.44m flat posture와 충돌하므로
  `safe_height=min(0.48, 기존 target_height)`로 사전 고정하고 invalid scan은 abstain한다.
  scan 최대 지면 높이와 실제 낙상 판정의 지면 평균은 다르므로0.31은 경고 상수다.
- action-rate는 action-manager의 비스케일 행동 차이 제곱합이며 관측 noise·실제
  effort·dt로 나눈 미분이 아니다. angular sway는 body-frame roll/pitch 속도다.
  continuous reward rate에 RewardManager dt를 한 번만 적용한다.
- 세 군 control/recovery/combined을 같은 부모·초기값·예산·3개 RNG seed로 학습한다.
  recovery 묶음 효과와 recovery 조건의 entropy 증분만 판단한다. entropy-only 효과,
  상호작용, 네 recovery 항의 개별 기여율은 식별하지 않는다.
- 외부 생존/return을 우리13.1/53.1m strict lane 성공률과 직접 비교하지 않는다.
  속도·낙상·평지·레인 tradeoff와 실패도 그대로 공개하며 좋은 결과를 보장하지 않는다.

[사전 고정 계획](../../../docs/experiment_plans/teammate_port_v25.md) ·
[조회 provenance](research_provenance.json).
로컬 원문 cache·전체 source SHA·재계산 로그는 ignored outputs에 보존한다.
관련 코드의 upstream BSD-3-Clause 저작권·조건·면책문은
[원문 라이선스](../../../licenses/teammate_ant_BSD-3-Clause.txt)에 보존한다.
팀원 저장소와 commit은 아이디어/적응 출처이며 원 저작자의 보증·endorsement가 아니다.
