# v19 — 같은 신규 지도에서 고정 정책 재평가

이번 작업은 **새 학습 0회**인 평가 전용 실험이다. v13–v18의 보류 지도는 서로
달랐으므로 이전 표를 버전별 순위로 읽을 수 없다. 권장 v5, 원래 v5/v10 history,
v16 무미끄러짐비용 control, v5/v16 history를 같은 scene/map/reset에서 비교한다.
새 보상·전환 임계값·정책 구조를 도입하지 않고 기본 모델을 자동 교체하지 않는다.

## 논문 근거와 범위

[원문 확인 기록](REBASELINE_V19_REFERENCES.md)에 CaT(2024), Miki(2022),
Aractingi(2023), Agarwal(2021)의 원리와 Ant 구현의 차이를 구분했다. v18은
CaT의 지형 조건부 스타일 원리만 차용했고 확률적 종료를 구현한 것은 아니다.
이번 공통 지도 비교의 구체적 3지도·16/64초 설정은 이 프로젝트의 선택이며,
논문 전체 재현 또는 학습 알고리즘의 일반적 우월성을 주장하지 않는다.

## 고정 방법

- [점수화 전 계획](experiment_plans/rebaseline_v19.md): geometry/reset107/68,
  108/69,109/70. 동일한 v16 평가 scene·91D 관측 생성·8D 행동·센서·물리.
- v5는 원래 v10 체크포인트 안의 frozen teacher를 사용한다. 교사 tensor가
  권장 v5 actor와 정확히 같은지 확인하며 실제 입력은 기존60D다.
  원래 v10 expert는88D prefix, v16 expert는91D다.
- 16초175환경/지도: 제어기마다 험지450+평지75 첫 episode.
  별도64초최고난도돌다리10환경/지도: 제어기마다30회. 총24파일2,220회.
- Strict1/6tile은13.1/53.1m에 도달하고 첫 episode 전체에서 낙상·발범위
  레인이탈·월드이탈이 없어야 한다. 생존과 완주, 평지와험지,16초와64초를 분리한다.
- 비교는 v16_control↔v5와 history_control↔history_original 두 개다.
  모든 기준과 서술적 gate는 계획에 미리 고정했다. 모델 선택/재학습/튜닝 없음.

## 유효성 검증

개발지도51/24 첫 시도는 재부팅 후 cold cache 때문에 첫 생성 경로의 스캔과
그 다음 로드 경로의 스캔이 달라 초기관측 짝검증에서 중단됐다. 원시 두 파일과
로그를 보존했다. 모델/임계값/관측 허용오차를 바꾸지 않고 attempt02에서
초기화 전용 cache 준비를 추가했다. 이 수정 때 신규지도점수화는0회였다.

Attempt02의4개개발실행140episode에서 초기 full91D/prefix88D/root/joints/
CPU+CUDA RNG가 정확히 같았고, v5와 두 history의 평지 물리/전환 원시배열도
같았다. v16_control과 history_original은 기존 v18 개발결과와 정확히 일치했다.
개발 전후 source SHA를 묶어 동결하고, 신규720타일을 step 없이 준비한 뒤
캐시/모델/source/평가 전 명령 ledger를 고정했다. 기존173소스·16모델을 보존했다.

## 결과

### 주평가 — 16초, 제어기마다 험지 450회 + 평지 75회

| 제어기 | 엄격 1타일 | 엄격 6타일 | 험지 낙상 | 레인 이탈 | 평지 낙상 | 평지 속도 m/s |
|---|---:|---:|---:|---:|---:|---:|
| v5 | 375/450 | 209/450 | 49 | 5 | 3/75 | 9.4624 |
| 원래 v5/v10 history | 402/450 | 225/450 | 41 | 4 | 3/75 | 9.4624 |
| v16 control | 415/450 | 265/450 | 31 | 4 | 2/75 | 11.1483 |
| v5/v16 history | 407/450 | 265/450 | 39 | 2 | 3/75 | 9.4624 |

두 사전 비교 모두 **합산 서술적 gate PASS**. v16↔v5의 짝 6타일 개선/악화는
62/6회, history control↔original은 47/7회다. **모든 지도에서 통과한 것은 아니다.**
v16은 지도107에서 이탈2→3, 지도108에서 평지 낙상1→2로 지도별 gate FAIL.
History는 지도108에서 1타일138→137, 낙상11→13으로 FAIL이다. 지도109는 둘 다 PASS.

### 별도 진단 — 64초 최고난도 돌다리, 제어기마다 30회

| 제어기 | 엄격 1타일 | 엄격 6타일 | 낙상 | 레인 이탈 |
|---|---:|---:|---:|---:|
| v5 | 12/30 | 0/30 | 13 | 0 |
| 원래 v5/v10 history | 16/30 | 13/30 | 11 | 3 |
| v16 control | 16/30 | 16/30 | 12 | 4 |
| v5/v16 history | 16/30 | 14/30 | 11 | 3 |

v16 단독은 6타일이 늘어도 **이탈0→4로 별도 gate FAIL**이며 세 지도 모두
이탈 기준을 통과하지 못했다. History는 합산 6타일13→14, 낙상/이탈 동일로
서술적 PASS지만 짝 개선/악화는8/7회에 불과하다. 지도107은 이탈0→2,
지도108은 낙상5→6, 지도109는 낙상4→5로 **지도별 gate는 모두 FAIL**이다.
따라서 장시간 보편적 개선으로 해석하지 않는다. 모든 평가의 월드 이탈은0회다.

전체 [표](../artifacts/terrain_demo/rebaseline_v19/summary.md)와
[원시 배열·지형/난이도별 집계](../artifacts/terrain_demo/rebaseline_v19/summary.json)를
함께 보아야 한다. 이번 비교는 여러 단계 학습이 누적된 고정 checkpoint 비교이지,
미끄러짐 보상이나 특정 논문 요소의 독립 인과효과가 아니다. 지도3개·기존 단일
학습 seed·이상적 시뮬레이터 센서 조건이며 **기본 모델을 바꾸지 않았다**.
새 훈련에는 이 지도를 개발자료로 간주하고 별도의 미사용 확인 지도를 써야 한다.

## 최종 코드 검증과 독립 감사

- 전체 **1,037 CPU tests**, 새66개 계약/오염 회귀 테스트, compileall/AST 및
  `git diff --check` 통과. 시스템 Mypy의 bounded6파일 검사도 통과했다
  (`--follow-imports=skip --ignore-missing-imports`; 외부 Isaac 의존성 전체 typecheck는 아님).
- Pyflakes는 clean이 아니다: 미사용 import 경고9개 중5개는 재노출 helper,
  2개는 task 등록용 side-effect import, 2개는 불필요한 evaluator import다.
  평가 source 동결을 유지하기 위해 결과 도중 cosmetic 수정하지 않았다.
  Ruff/Pyright는 설치되지 않았다. 새 의존성 없음.
- 독립 stdlib 계산으로 primary12파일2,100회의 strict 판정, 지형별 분모,
  짝 비교, 초기상태/평지 동일성, SHA/720개 cache tile을 재계산했다.
  [주평가 감사](../artifacts/terrain_demo/rebaseline_v19/independent_primary_audit.json).
- 별도 CPU-only 역직렬화로 v10/v16 teacher 각각8개 tensor가 권장 v5 actor와
  key/shape/dtype/value까지 정확히 같고 모든 state가 유한함을 확인했다.
  [Tensor·이전173소스/16모델 감사](../artifacts/terrain_demo/rebaseline_v19/independent_tensor_audit.json).
- [최종24파일2,220회 독립감사](../artifacts/terrain_demo/rebaseline_v19/independent_final_audit.json)
  **PASS**: 원시 strict 배열·분모·지형별 집계·합산/지도별 gate·짝6타일 결과가
  기존 집계와 일치했다. 초기state/RNG와24개실행명령/로그도 검증했다. 접촉/자세
  telemetry 평균을 이 독립 감사에서 다시 계산하지 않았고, GPU 물리 재실행이나
  통계적 유의성 검증도 아니다. [최종 검증 기록](../artifacts/terrain_demo/rebaseline_v19/final_validation.json).

새 파일은 `rebaseline_study_v19.py`, 평가/실행/집계 스크립트3개, 테스트2개와
이 문서/계획/문헌 기록이다. 기존 rollout·tracker·센서·정책·감사 함수를 재사용하고
이전 v0–v18 실험 source와 모델은 변경하지 않았다. commit·원격 Git 작업 없음.

## 재현과 증거

기존 출력 보호 때문에 같은 디렉터리에서 아래 명령을 다시 실행하지 않는다.
재실행은 새 사전계획·격리 디렉터리·새 지도에서만 수행한다. 실제 명령과 로그
해시는 [commands.jsonl](../artifacts/terrain_demo/rebaseline_v19/commands.jsonl)에 있다.

```bash
../run-python scripts/run_rebaseline_v19.py development --device cuda:1
../run-python scripts/run_rebaseline_v19.py freeze
../run-python scripts/run_rebaseline_v19.py prepare --device cuda:1
../run-python scripts/run_rebaseline_v19.py evaluate --device cuda:1
../run-python scripts/run_rebaseline_v19.py horizon --device cuda:1
../run-python scripts/run_rebaseline_v19.py report
```

평가 자체를 다시 돌리지 않는 읽기 전용 재집계는 전체 결과가 생긴 뒤
`../run-python scripts/summarize_rebaseline_v19.py`로 수행한다.
[동결](../artifacts/terrain_demo/rebaseline_v19/frozen.json),
[개발검증](../artifacts/terrain_demo/rebaseline_v19/development.json),
[첫시도유효성실패](../artifacts/terrain_demo/rebaseline_v19/development_attempt01_failure.json).
