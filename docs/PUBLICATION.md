# 공개 범위와 검증 — 2026-09-22

## 포함한 내용

- [전체 실험 과정·결과](EXPERIMENT_HISTORY.md), v5~v10 방법·선택 근거·실패 결과,
  논문에서 참고한 내용과 실제 구현의 차이.
- 환경/정책/학습/평가/시각화 코드와 207개 CPU 회귀 테스트.
- `artifacts/terrain_demo/`의 비교 모델, 설정, first-episode 평가 배열,
  지형/난이도/seed별 집계, frozen manifest, 독립 검토 기록, 무편집 비교 영상.
- [사전 실험 계획](experiment_plans/manifest.json): 무시된 로컬 작업 디렉터리에
  있던 7개 PLAN의 **바이트 동일 복사본**. 계획이 결과를 본 뒤 새로 작성된 것은 아닙니다.
- [RSL-RL 라이선스](../THIRD_PARTY_NOTICES.md). 시뮬레이터/SDK/로봇 asset 자체는 배포하지 않습니다.

현재 권장 모델은 [v5 portal-rehearsal round4](../artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt)입니다.
SHA-256: `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e`.
**최초 과제 모델도, v10의 교체 성공 결과도 아닙니다.**

## 제외 / 보존 정책

`.omx/`, 자격증명, 세션 DB/캐시, `logs/`, `outputs/`, 원시 console 로그,
TensorBoard event, 빌드 출력과 설치 환경은 새 공개 이력에 넣지 않습니다.
중간 학습 로그/모든 iteration의 체크포인트가 아니라, 검증 후 `artifacts/`에
보관한 비교·선택 체크포인트를 포함합니다. 기존 manifest의 `event_files`와
명령 기록의 `log` 경로는 **로컬 보관 이력**이며 배포 파일을 뜻하지 않습니다.

기존 미푸시 커밋 2개에는 PC 정보가 담긴 원시 로그가 있었습니다. 해당 커밋과
작업 원본은 로컬에 보존하고, 기존 원격 기준점에서 별도 공개용 브랜치를 구성했습니다.
기존 커밋의 재작성이나 force push를 하지 않습니다. 이미 과거에 공개된 Git 이력까지
삭제하는 작업은 이 배포에 포함하지 않습니다.

체크포인트는 실험의 SHA 보존을 위해 다시 저장하지 않았습니다. 일부 `.pt` 메타데이터에는
원래 학습 파일의 **비밀정보가 아닌 로컬 mount 경로 문자열**이 남습니다.
따라서 “모든 머신 경로가 제거됐다”거나 “비밀정보 검출이 완벽하다”고 주장하지 않습니다.
기존 `scripts/run_{train,evaluate,terrain_demo}.sh`의 course 경로 기본값도 남아 있습니다.
이 값은 `ROBOTICS_SIM_CLASS_ROOT`로 바꿀 수 있으며 자격증명이 아닙니다.
외부에서 얻은 PyTorch/pickle 파일은 신뢰 여부를 확인한 뒤 로드해야 합니다.

## 원본 증거와 공개용 메타데이터 구분

JSON/YAML/문서의 절대 경로와 호스트명은 공개본에서 상대 경로/일반 이름으로 바꿨습니다.
원본 바이트는 로컬에 따로 보존했습니다. 수치, 배열, seed, 시간, 평가 조건, 정책·소스
SHA는 바꾸지 않았으며 체크포인트/영상/소스 파일은 이 경로 변환의 대상이 아닙니다.

- [메타데이터 변환 manifest](../artifacts/publication_metadata.json)는 파일별
  `original_sha256`와 `published_sha256`를 구분합니다.
- 기존 `verification.json`, `independent_audit*.json`, `completion.json` 안의 파일 digest와
  파일시각 검사는 **실험 당시 원본 증거**에 대한 기록입니다. 공개용 경로 변경 뒤의 파일과
  해시가 같다고 해석하면 안 됩니다. 원래 감사기의 mtime/절대 경로 검사는 Git checkout에서
  그대로 재실행하는 계약이 아닙니다.
- 공개 파일 자체의 검증은 [PUBLICATION_SHA256SUMS](../artifacts/PUBLICATION_SHA256SUMS)를 씁니다.
  소스/모델의 frozen SHA는 그대로이며 공개 텍스트의 새 SHA만 구분됩니다.
- 원본 JSON/JSONL과 공개본을 재귀 비교해 문자열을 제외한 모든 값·배열·구조가 동일함을
  확인했습니다. 변환 결과 전체도 허용한 경로/호스트명 치환과 바이트 단위로 비교했습니다.

이 구분은 과거의 검증 결과를 공개본에 새로 수행한 것처럼 보이게 하지 않기 위한 것입니다.

## 읽기와 검증

저장소 루트에서 공개 파일의 바이트 무결성 검사(추가 패키지 불필요):

```bash
sha256sum -c artifacts/PUBLICATION_SHA256SUMS
```

Isaac Sim 5.1 / Isaac Lab 2.3 / RSL-RL 3.0.1의 기존 course 환경에서 CPU 테스트:

```bash
../run-python -m pytest -q
```

`run-python`과 course 환경은 이 저장소의 상위 디렉터리에 별도 설치되어 있어야 합니다.
기존 shell wrapper를 다른 위치에서 사용할 때는 `ROBOTICS_SIM_CLASS_ROOT`를 실제 course
루트로 설정하세요. 시뮬레이션·재학습은 각 버전 문서의 명령과 pinned 환경이 필요합니다.
학습 harness는 이미 frozen 결과가 있으면 덮어쓰기를 거부합니다. 새 실험을 위해 기존
결과를 지우거나, 같은 holdout으로 반복 튜닝해도 된다는 뜻이 아닙니다.

v10 원본 source freeze에 포함된 로컬 PLAN 경로가 필요한 별도 작업에서는 다음처럼
바이트 동일 계획을 복원할 수 있습니다. 기존 파일이 있으면 덮어쓰지 마세요.

```bash
mkdir -p outputs/prior_v10_20260922
cp -n docs/experiment_plans/prior_v10.md outputs/prior_v10_20260922/PLAN.md
```

HTML 비교 페이지는 GitHub 파일 화면이 아니라 로컬에서 여세요. 저장소 루트에서
`python3 -m http.server 8000 --bind 127.0.0.1` 실행 후
`http://127.0.0.1:8000/artifacts/terrain_demo/prior_v10/`로 접속하면 됩니다.
영상에는 reset 후 궤적이 포함될 수 있으며 **first-episode 성공률을 대신하지 않습니다**.

## 검증 범위와 남은 한계

공개 준비 전 207개 CPU 테스트와 v10의 16초/64초 원본 독립 감사를 다시 통과했습니다.
공개 준비 후 검증 결과는 [publication_verification.json](../artifacts/publication_verification.json)에
별도로 기록합니다. 기존 실험의 simulator smoke, 학습, holdout 평가, 전체 영상 decode는
버전별 검증 기록에 있습니다. 배포를 위해 GPU 학습/평가를 다시 돌리거나 재선별하지 않았습니다.

Ruff/Mypy/Pyright는 설치되어 있지 않아 수행했다고 주장하지 않습니다. 자격증명 검사는
휴리스틱이며 완전 보장은 아닙니다. 고난도 징검다리·장시간 안정성·실제 RGB-D/실물 로봇
일반화는 여전히 미해결입니다. **v6~v10 전체 교체 gate는 FAIL, v5 권장 유지**입니다.

정적 diff 검사에는 기존 CSV의 CRLF, Markdown hard break 공백, frozen prior 테스트
2개의 EOF 빈 줄 경고가 남았습니다. 원본 바이트 보존을 위해 일괄 포맷하지 않았으며,
이를 whitespace-clean 또는 lint 통과로 보고하지 않습니다.
