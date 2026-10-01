# v25 출판 정제 계약 검토 (read-only)

## 결론

**COMMENT — 조건부 출판 계획 의견이며, 아직 출판물 승인이나 결과 검증은 아님.**

지금 ART 원본을 바꾸면 안 된다. 9개 main, 평가/데모, 원본 전체 감사가 끝난 뒤 **별도 publication view**로 정제하는 것이 최소 안전 경계다. 최종적으로 같은 ART 경로를 공개용 사본으로 교체하더라도, 원본을 먼저 immutable private backup으로 보존하고 그 공개 파일은 더 이상 historical full-audit 입력이 아님을 명시해야 한다. frozen science20 및 모델/수치/과학 SHA는 변경하지 않는다.

## 확인된 실패 경계

1. **HIGH — 실행 중 ART 정제는 다음 main을 깨뜨린다.** `scripts/run_teammate_training_v25.py:300-322`가 매 run 전에 `artifacts_sha256`, `expected_main_sha256`, runtime 파일, config와 development/cache 링크를 재검증한다. 현재 `expected_main/control61_initial.json:319,991`의 `/tmp` 값도 정확 비교 대상이다. 바이트가 달라지는 재포맷만 해도 실패한다.
2. **HIGH — 과학 hash를 공개 hash로 치환하면 과거 freeze를 소급 변경한다.** `scripts/summarize_teammate_v25.py:158-175` 및 `scripts/audit_teammate_v25_raw.py:865-868`은 original raw bytes의 hash/equality를 검사한다. 공개 사본의 경로 문자열 하나를 바꾸면 그 hash는 달라진다. `original_raw_sha256`, `evidence_sha256`, `freeze_sha256`, `training_freeze_sha256` 등의 기존 값은 그대로 두고 새 공개 manifest에서 두 byte domain을 연결해야 한다. 특히 `teammate_study_v25.py:111`은 기존 training freeze 자체의 SHA를 요구한다.
3. **HIGH — 기존 full CLI는 public-checkout CPU replay가 아니다.** `audit_teammate_v25_raw.py:857-868`은 원래 cache와 raw byte hash를, `:641-668`은 local YAML/log/TensorBoard를 요구한다. `run_teammate_eval_v25.py:42-60` 역시 `<terrain-cache>`를 읽는다. 실행해 보고 missing cache를 무시하는 fallback을 추가해서는 안 된다.
4. **MEDIUM — 모든 절대경로가 repository 파일은 아니다.** `initial_shared.json:142-145`는 site-packages 파일 경로를 **키**로 쓴다. `/tmp` 설정, Python 실행 파일, parent `run-python`도 repository 밖에 있다. prefix 제거로 존재하지 않는 repo-relative 파일을 꾸며내지 않는다. 외부 runtime logical identifier임을 분류하거나 그 원본 메타데이터는 private로 두고 별도 public projection을 제공한다.

위 항목은 현재 코드 결함/조작 발견이 아니라, 제안한 정제를 잘못 적용할 때 확정적으로 깨지는 계약이다.

## 최소 출판 절차

1. **원본 완료와 봉인:** 모든 writer 종료 → 기존 source/cache/config/model/전체 training+eval audit PASS → final 원본 summary/audit와 SHA 기록. console 또는 JSONL이 쓰이는 동안 snapshot하지 않는다.
2. **Private byte backup:** `WORK/publication_raw_backup` 아래 원래 repo-relative 계층을 유지한 exclusive byte copy와 private inventory. ART뿐 아니라 그 manifest가 가리키는 original/enriched raw, freezes, cfg와 log 근거도 보존한다. 이전 실패/재실행 evidence도 삭제/재작성하지 않는다.
3. **정제 사본:** 파일별/JSON-pointer별 allowlist만 적용. 경로 문자열과 명시한 runtime-path key만 변경. 키 충돌, 알 수 없는 경로, list 순서 변경, 키 삭제, hash 문자열 변경을 거부한다. float는 `-0.0`까지 bit/type 동등성, int/bool/null은 exact type/value, 배열은 길이·순서·모든 값을 대조한다. 숫자나 JSON 구조를 고쳐서 portable하게 만들지 않는다.
4. **원본과 공개 SHA를 분리:** 새 `publication_provenance.json`은 post-experiment timestamp, original freeze SHA, 변경 불가 source SHA 및 각 파일의 `original_sha256` / `published_sha256` / `published_path` / kind / transform version / 허용 pointer / 의미 동등성 검증을 기록한다. 원래 private 절대 문자열은 공개하지 말고 private mapping에 보존한다. 공개 manifest와 `PUBLICATION_SHA256SUMS`는 실제 공개 bytes만 인증한다. 기존 과학 hash field는 손대지 않는다.
5. **Raw pair 보존:** 원본의 `enriched - v25 == raw`, 공개본의 같은 equality를 모두 확인한다. `windows`, assignments, 초기 상태/RNG hash, 모든 수치/판정은 원본과 동일해야 한다. `v25.original_raw_sha256`는 **원본** raw hash를 유지한다. 공개 raw hash는 새 manifest에 둔다.
6. **추가 공개 검증:** 새 CPU replay 계약대로 clean checkout/임시 복사본에서 확인한다. byte hash 검증 → 의미 replay 순서이며, 불일치를 건너뛰지 않는다. science20 밖의 새 presentation-only 검증 도구를 만들 경우 사후 출판 도구임을 명시하고 별도 검토한다.

**한계 고지:** 원본 bytes를 공개하지 않으면 original→published 동등성은 로컬에서 검증한 publisher attestation이다. 공개 독자는 공개 bytes 및 수치의 일관성은 재검증하지만 비공개 원본과의 동일성을 cryptographically 증명할 수는 없다. 이 한계를 감추는 'full audit reproducible' 표현을 쓰지 않는다.

## Public checkout CPU replay의 정확한 최소 계약

필수 public inputs:
- 수정 없는 `scripts/audit_teammate_v25_raw.py` 및 freeze가 지정한 science20/base-source bytes;
- 정확한 46개 holdout raw/enriched record (23 endpoints × 2 maps), 해당 원본 freeze의 public representation, public summary;
- 모든 freeze 모델의 동일한 checkpoint bytes (hash 검증만, unpickle 없음);
- 위 입력의 실제 공개 hash와 exact file list를 가진 새 publication manifest. glob으로 다른 JSON을 점수 파일에 섞지 않는다.

다음은 **제안 schema**에 대한 절차 예시이며, 아직 이 이름의 manifest가 생성되었다는 뜻은 아니다. repo root에서 stdlib Python만 사용한다:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import audit_teammate_v25_raw as audit

root = Path.cwd()
pub = audit.read('artifacts/terrain_demo/teammate_port_v25/publication_provenance.json')
# manifest itself is pinned by the checked-out publication checksum inventory.
audit.check_hashes(root, pub['published_files_sha256'])
freeze = audit.read(audit.inside(root, pub['evaluation_freeze']))
for key in ('source_sha256', 'base_source_sha256'):
    audit.check_hashes(root, freeze[key])
for model in freeze['models'].values():
    audit.check_hashes(root, {model['checkpoint']: model['sha256']})
records = [audit.read(audit.inside(root, name)) for name in pub['evaluation_records']]
summary = audit.read(audit.inside(root, pub['summary']))
canonical = audit.audit_evaluation_records(records, freeze)
audit.compare_summary(summary, canonical)
print('PASS: published raw-array metrics and summary agree')
```

추가 publication-byte 검증은 raw/enriched 공개 pair equality와 original→published mapping 관계를 확인해야 한다. 이 예시만으로 original hash chain 전체를 replay했다고 주장하지 않는다. `compare_summary`는 **holdout summary 전용**이다. development는 `audit_evaluation_records`의 280/560/parity 결과만 별도로 확인한다.

이 경계는 published arrays의 strict final-distance/F32 success, 두 horizon의 동일 first episode, 모든 cell/seed/map gate 및 summary 일치를 재계산한다. 새 물리 시뮬레이션, 학습 재실행, 비공개 cache/runtime/console 검증 또는 pickle tensor 감사를 수행하지 않는다. 기존 `audit_...py` CLI와 `summarize_...py` CLI는 local original graph용으로 남겨 둔다.

## 공개/비공개 경계

- 공개 가능: 모델 원본 bytes, 검토된 수치/config projection, 모든 평가 arrays, frozen source/hash, 명시적으로 정제한 재현 명령, pinned attribution/license, 판정과 검증 보고서.
- 기본 제외: raw stdout/stderr, 환경 전체 dump, home/mount 식별자, session/plugin DB, `.omx`, fleet telemetry, credential/token/cookie/서명 URL, 검토하지 않은 commands JSONL, WORK/logs 전체, local cache mesh. console은 필요하면 승인된 제한적 structured 숫자/오류 excerpt만 별도 생성한다.
- YAML `params`도 출판 전에 별도 검사한다. 원본 YAML을 바꾸면 saved YAML SHA가 깨진다. 필요하면 원본 hash를 유지한 public JSON config projection을 내고 raw YAML hash와 구분한다.
- 외부 path를 실제 repo file처럼 표현하지 않는다. 공개 root 안으로 매핑한 `external_runtime/...` 같은 이름은 **논리 식별자이고 파일이 아님**을 manifest kind에 명시한다. 존재하는 파일을 요구하는 validator에 이 식별자를 넘기지 않는다.
- 문자열 **값과 키** 모두 secret/path 검사 대상이다. 본 검토는 완성된 출판물 secret scan이 아니다.

## 현재 확인한 근거

- 현재 frozen science20 파일 SHA 20/20 일치; 어떤 source/artifact도 수정하지 않았다.
- 실제 development 8개 record를 `python3 -S` + stdlib auditor 함수로 검증: 280 physical episodes, 560 dependent windows, exact parity PASS.
- 동일 record를 메모리에서 checkpoint/child-command path 문자열만 상대화한 뒤 같은 audit 결과 확인. `torch`/`isaaclab` import 없음.
- holdout 결과는 아직 없으므로 public 46-file replay는 수행하지 않았다.
- GPU, remote, Git 작업 없음. 의견 파일 2개만 WORK/planning에 작성.
