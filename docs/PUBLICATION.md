# 공개 범위와 검증 — 최종 정리 2026-09-30

## README 표현 정리 — 2026-09-30 후속 작업

README를 최종 모델·결과표·그래프·영상·실행 명령 중심으로 줄였습니다.
기존 두 MP4의 전체 시간축을 유지한 저해상도 GIF 미리보기를 추가했으며,
원본 영상·모델·실험 소스·계획·평가 데이터는 변경하지 않았습니다.
[미리보기 변환 기록](../artifacts/readme_media/manifest.json)과
[README 후속 검증](../artifacts/readme_verification_20260930.json)을 별도로 보존합니다.
기존 `publication_verification_20260930.json`은 앞선 제출 정리 시점의 검증 기록입니다.
현재 파일의 무결성은 갱신된 `PUBLICATION_SHA256SUMS`로 확인합니다.

## 이번 공개 준비 범위: v0–v24 + 과제 제출 안내

- 과제 제출 선택은 **Robust seed42 `model_999.pt`**입니다. 험지 기본 v5와 구분한
  [최종 실행/제출 가이드](FINAL_SUBMISSION.md)와 README 진입점을 추가했습니다.
- 기존 공개 v0–v10에 이어 로컬 후속 v11–v24의 소스/회귀 테스트, 사전 계획,
  선택·비교 checkpoint, first-episode 배열, 집계, 독립 감사, 무편집 비교 영상을 포함합니다.
  후속 모델을 원래 과제의 60D 평가용 모델로 승격하지 않았습니다.
- 원시 console/TensorBoard/cache/agent runtime/credentials와 수업 환경 자체는 제외합니다.
  이번에는 학습을 새로 하지 않았으며, 고정 제출 모델의 ID/OOD **4 × 100환경 추론**만
  실행했습니다. [이번 평가 JSON](../artifacts/submission_validation_20260930/)은 과거 평가를
  덮어쓰지 않는 별도 검증 기록입니다.
- source/model/영상/사전 계획의 bytes는 보존합니다. 새 공개 artifact 메타데이터에서만
  비밀정보가 아닌 machine-local path/hostname 문자열을 일반화합니다. 원문은 ignored
  `outputs/`에 보존하며 [변환 manifest](../artifacts/publication_metadata_20260930.json)가
  원문/공개본 SHA를 구분합니다. 수치/배열/seed/정책 SHA는 바꾸지 않습니다.
  TensorBoard event의 basename은 증거 식별자라 그대로 두며, 비밀정보가 아닌 과거
  hostname이 그 이름에 남을 수 있습니다. 해당 원시 event 파일은 공개하지 않습니다.
- 실험 당시 freeze/감사/문서 해시는 **그때 원문의 해시**입니다. 문자열 변환이나 최종
  안내 추가 뒤 현재 파일과 일치한다고 해석하지 않습니다. cache/원시 log/mtime를 요구하는
  과거 감사기는 공개 checkout만으로 모두 재실행되는 계약이 아닙니다.

현재 공개 snapshot의 byte 검사는 `sha256sum -c artifacts/PUBLICATION_SHA256SUMS`를 사용합니다.
[이번 검증 기록](../artifacts/publication_verification_20260930.json)은 과거 게시 검증과 분리합니다.
9월22일 목록은 [보존한 checksum 목록](../artifacts/PUBLICATION_SHA256SUMS_20260922)에 있으며
과거 commit `49ba30c`의 checkout에 해당합니다. 현재 문서를 과거 checksum으로 검사하지 마세요.

공개 checkout의 CPU 테스트 명령은 [최종 가이드](FINAL_SUBMISSION.md#7-제출-요구사항-대응--마지막-확인)를
따릅니다. 공개본에서 검증한 선택 unit tests는 **147개**이며, 전체 **2,056개** suite는
과거 PLAN 원래 경로·개발 JSON·학습 설정·초기 checkpoint·원시 log/event 등 로컬 원문을
필요로 합니다. “6개 log/event 사례만 제외하면 전체 공개 suite가 통과한다”는 초기 시도는
실패했으며 이번 검증 기록에 보존했습니다. 공개 subset과 로컬 전체 검증을 구분합니다.

로컬 공개 브랜치는 기존 공개 이력에서 이어갑니다. 다른 로컬 브랜치의 미공개 이력은
변경하지 않으며 원격 조회/게시에는 실행 직전 승인을 받습니다. 최종 commit과 remote 반영은
Git 이력/최종 작업 보고로 확인하며, 준비 상태를 push 성공으로 표현하지 않습니다.

학습 결과를 유리한 seed만으로 재해석하지 않았습니다. 3seed 과제 연구 결론과 단일
checkpoint 실행 선택을 구분하고, 험지 v6–v24의 실패/부분 이득/trade-off를 함께 보존합니다.
실물/실제 RGB-D/비공개 채점 강건성, 모든 험지 완주는 미검증입니다.

## 과거 게시/로컬 완료 시점 기록 — 2026-09-22 이후

> 아래는 공개 commit `49ba30c`의 v0~v10 스냅샷 기록입니다. 이후 로컬에서 추가한
> v11–v24의 “미푸시/로컬” 설명은 각 실험 완료 당시의 이력입니다. 과거 공개 체크섬 목록은
> 현재 작업 트리 전체의 검증을 뜻하지 않습니다. 최종 범위는 위 절을, v11 검증·재시도는
> [별도 문서](HYBRID_V11.md)를 봅니다.

## 포함한 내용

- [전체 실험 과정·결과](EXPERIMENT_HISTORY.md), v5~v10 방법·선택 근거·실패 결과,
  논문에서 참고한 내용과 실제 구현의 차이.
- 환경/정책/학습/평가/시각화 코드와 207개 CPU 회귀 테스트.
- `artifacts/terrain_demo/`의 비교 모델, 설정, first-episode 평가 배열,
  지형/난이도/seed별 집계, frozen manifest, 독립 검토 기록, 무편집 비교 영상.
- [사전 실험 계획](experiment_plans/manifest.json): 무시된 로컬 작업 디렉터리에
  있던 7개 PLAN의 **바이트 동일 복사본**. 계획이 결과를 본 뒤 새로 작성된 것은 아닙니다.
- [RSL-RL 라이선스](../THIRD_PARTY_NOTICES.md). 시뮬레이터/SDK/로봇 asset 자체는 배포하지 않습니다.

험지 데모 권장 모델은 [v5 portal-rehearsal round4](../artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt)입니다.
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

아래 과거 전체-suite 명령은 원래 로컬 증거가 있는 환경용입니다. 공개 checkout에서는
위 최종 범위의 선택 unit-test 명령을 사용하세요.

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

## v14 로컬 추가 실험

[명시적 자세 명령 관측](COMMAND_CONDITIONING_V14.md)의 구현·학습·검증은 로컬
후속 실험이다. 기존 공개본 checksum 목록이나 과거 v0–v10 게시 검증을 이 새
실험의 게시 증거로 해석하지 않는다. 기존89개 frozen source와6개 기준 모델은
보존하고 별도 v14 산출물을 사용한다. 이번 요청에는 원격 Git 작업이 포함되지
않아 push/원격 조회/공개 메타데이터 변환은 하지 않았다. 원본 로그·캐시·런타임
상태는 계속 Git-ignored 로컬 자료이며, 새 원격 작업에는 별도 승인이 필요하다.

## v15 로컬 추가 실험

[방향 안정화 보상](DIRECTIONAL_STABILITY_V15.md)은 새 로컬 후속 실험이다.
동일 예산 두 모델과 새 지도24파일·2,220회 결과를 보존했다. 16초·64초의 단독/하이브리드
전체 기준은 모두 FAIL이며 기본 설정을 교체하지 않았다. 기존106개 frozen source와8개
기준 모델은 유지했다. 코드/계획/원시결과/해시/독립감사와 현재 기계의 로그·캐시는 구분한다.
기존 게시 검증을 v15 게시 증거로 사용하지 않는다. 이번에는 commit/push/원격 조회를
하지 않았고, 새로운 GitHub 작업은 별도 승인 대상이다.

## v16 로컬 추가 실험

[지면 접촉 조건부 발 미끄러짐 비용](CONTACT_SLIP_V16.md)도 로컬 후속 실험이다.
두 군 각3,276만 transition과 새 지도24파일·2,220회 결과 및 독립 원시 감사를
별도로 보존했다. 16초/64초 단독·하이브리드 네 판정은 모두 FAIL로 기본
정책을 교체하지 않았다. 기존 공개본의 체크섬·게재 검증은 v16에 적용되지
않는다. 이번 작업은 commit/push/원격 조회 없이 로컬 코드·계획·평가 증거만
추가했으며, 원시 로그·캐시는 Git-ignored 로컬 자료다.

## v17 로컬 추가 실험

[학습 진도 기반 지형 커리큘럼](LEARNING_PROGRESS_V17.md)은 문헌 아이디어를
동일 episode별 재샘플링 대조군과 새 지도24파일·2,220회로 비교한 로컬 실험이다.
16초 주평가의 단독/히스토리 승격은 모두 FAIL이며 기본 정책을 교체하지
않았다. 별도64초 단독 돌다리 개선을 전체 개선으로 일반화하지 않는다.
학습/평가 계획 수정 이력, 최종 원시 JSON과 독립 감사가 새 v17 경로에 있다.
과거 게시 검증/체크섬을 v17의 게시 증거로 사용하지 않으며, 이번 작업은
commit/push/원격 조회 없이 로컬에만 보존했다.

## v18 로컬 추가 실험

[깊이 조건부 교사 스타일 prior](TERRAIN_STYLE_V18.md)는 문헌의 지형 조건부
스타일 원리를 frozen-v5 prior 손실에 제한 적용한 로컬 비교다. 두 군 각3,276만
전이·새 지도24파일/2,220회·독립 원시 감사에서 16초 단독/하이브리드 승격은
모두 **FAIL**이고 기본 모델을 교체하지 않았다. 기존 공개본의 checksum/게시
검증을 v18 증거로 사용하지 않는다. 학습·평가 원시 증거와 모델 SHA는 새
`artifacts/terrain_demo/terrain_style_v18/`에 보존했다. 이번 작업에서는
commit/push/원격 조회를 하지 않았으며, 기존 공개 스냅샷 내용이
v18까지 게시됐다고 주장하지 않는다.


## v19 로컬 평가전용 후속 실험

[공통 지도 재평가](REBASELINE_V19.md)는 새 학습 없이 기존 정책4개를 동일한
신규 지도3개에서2,220회 평가한 로컬 결과다. 16초 두 합산 서술적 gate는PASS지만
지도별 회귀와64초단독 레인 이탈 회귀가 남으며 기본정책은유지한다. 모델·평가
설정의변경이나통계적우월성/실물보장을주장하지않는다. 결과/독립감사/문헌근거는
새v19경로에있고 이전 공개snapshot의검증으로대체하지않는다. commit/push/원격
조회없음; raw로그와현재기계cache는계속Git-ignored이며 게시에는별도승인이필요하다.


## v20 로컬 접촉 비용 커리큘럼 실험

[즉시/점진도입짝비교](CONTACT_CURRICULUM_V20.md)는새로컬후속실험이다.
두최종모델각250iter·새지도24파일2,220첫episode와독립검증을보존했다.
16초/64초여섯합산gate가모두FAIL이라기본모델을교체하지않았다. 문헌의전체방법
재현이나실물보장/통계적우월성을주장하지않는다. 기존181개frozen source와16개
기준모델을보존했다. 새코드/계획/결과/최종모델/감사기는로컬추가파일이며,이전공개
snapshot체크섬을v20의게시증거로사용하지않는다. commit/push/원격조회없음.
원시로그·TensorBoard·기계cache·runtime상태는계속Git-ignored로분리하며,
실제원격검토/게시에는새승인이필요하다.

## v21 로컬 무비용 추가학습 대조군

[동일예산 무접촉비용 대조군](CONTACT_CONTINUATION_V21.md)은 새로운 로컬 실험이다.
새 본학습32,768,000전이와 개발557,056전이를, 재사용 v20학습65,536,000전이와
분리해 기록했다. 신규32파일2,960첫episode·독립감사에서 어떤 추가학습군도
부모 대비 합산gate를 통과하지 못했다. 일부 비용군 간 PASS는 지도별 일관성이
없어 기본정책을 유지했다. 한seed·과거학습군 재사용의 한계를 명시했다.
기존197frozen source·18model과 원시 증거를 보존했다. 새 코드는 `src/`·`scripts/`,
계획/설명은 `docs/`, 최종모델·검증결과는
`artifacts/terrain_demo/contact_continuation_v21/`에 있다.
이번 작업은 commit/push/원격 Git/GitHub 조회를 하지 않았다. 이전 공개 snapshot의
체크섬은 v21의 게시 증거가 아니며 로그/TensorBoard/cache/runtime상태는 ignored다.
실제 원격 검토·게시에는 별도 승인이 필요하다.

## v22 로컬 추가학습 seed 민감도

[무비용 추가학습의 seed 비교](SEED_CONTINUATION_V22.md)는 새 로컬 후속 실험이다.
같은 부모·초기 가중치·훈련 지형에서 세 seed를 새로 학습했다. 본학습98,304,000전이와
개발835,584전이, 평가 개발280회와 holdout32파일2,960회는 분리해 기록했다.
16초12합산gate는 모두FAIL, 별도64초는6개PASS지만 seed/controller별 차이가 커
기본정책을 유지했다. 3seed를 통계적 우월성이나 일반 강건성 보장으로 해석하지 않는다.
기존213source·19model을 보존하고 새 모델·원시결과·감사를
`artifacts/terrain_demo/seed_continuation_v22/`에 보관했다. 이전 공개 checksum은
이번 실험의 게시 증거가 아니다. console/TensorBoard/cache/runtime은 ignored이며,
이번 작업은 commit/push/원격 Git·GitHub 조회 없이 로컬에서만 수행했다.
실제 원격 검토·게시에는 별도 승인이 필요하다.


## v23 로컬 같은 episode의 짝 시간 평가

[16초·64초 짝 평가](PAIRED_HORIZON_V23.md)는 새 학습 없는 로컬 후속 진단이다.
신규16bundle·2,800첫episode·5,600의존window와 개발350/630을 구분했다.
주요16초의 부모 대비6gate가 모두FAIL이며,64초History51/53의 두지도PASS로
주요회귀를 상쇄하거나 기본정책을 승격하지 않았다. 부모의 늦은 성공72·실패16을
따로 보고하며 시간window를 독립표본으로 세지 않는다. GPU29명령과 독립원시감사,
1,673CPUtests/새Python9개 정적검사를 통과했다. 기존227source·22model을 보존했다.
새 코드·계획·원시결과·감사는 `artifacts/terrain_demo/paired_horizon_v23/` 및 해당
`src/`·`scripts/`·`tests/`·`docs/`에 있다. 이전 공개checksum은 이번 결과의 게시
증거가 아니다. console/cache/runtime은 ignored이며 commit/push/원격 Git·GitHub
조회 없이 로컬에서만 수행했다. 실제 원격 검토·게시에는 새 승인이 필요하다.


## v24 로컬 고정 전역 학습률 민감도

[학습률1e-4→1e-5 추가학습](LR_CONTINUATION_V24.md)은 새로 학습한low3개와 동결high3개를
같은 신규2지도에서 비교한 로컬 실험이다. 새학습132,677,632전이, 본평가4,900첫episode와
9,800의존window, 개발280/560을 분리했다. 16초 단독6타일의+7/+7/+9회에도 불구하고
low/high 여섯합산gate는16/64초 모두FAIL, low/부모 여섯gate도 모두FAIL이라 기본정책을
유지했다. 세seed 중 유리한결과나 한시간window만선택하지않는다.
52GPU명령 전부성공·실패/재시도0; 독립CPU tensor/Adam/LR/scalar 및 stdlib 원시감사,
실제학습증거별도검토를 통과했다. 기존237source·22model과 새12동결정의를 보존했다.
새계획/모델/원시/감사는 `artifacts/terrain_demo/lr_continuation_v24/` 및 해당
`src/`·`scripts/`·`tests/`·`docs/`에 있다. 이전공개checksum은 이번결과의게시증거가 아니다.
console/TensorBoard/cache/runtime은ignored다. commit/push/원격Git·GitHub조회와 의존성설치,
실물로봇변경은없다. 실제원격검토·게시에는 새명시적승인이 필요하다.
