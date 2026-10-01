# 미사용 장애물 배치 데모와 최고 성능 설정 확인 · 2026-10-01

## 결론

**장기 혼합험지의 엄격한 완주율에서는 history high53 교체 방식이 가장 좋았다.**
그러나 **단기 혼합험지는 v16 단독**, **불연속 장애물만 보면 high53 단독**이 더 좋았다.
따라서 “모든 지형에서 평지/험지 모델 분리가 최고”라고 확인한 것은 아니다.

| 평가 범위 | 이번 비교의 최고 후보 | 엄격한 6타일 성공 | 같은 조건의 대조 |
|---|---|---:|---|
| 16초 혼합험지 | v16 control 단독 | **177/300 (59.0%)** | v5 135/300, history high53 166/300 |
| 64초 혼합험지 | v5 ↔ high53 history 교체 | **236/300 (78.7%)** | high53 단독 223/300, v16 단독 224/300 |
| 64초 불연속 장애물만 | high53 단독 | **38/50 (76.0%)** | history high53 32/50, v16 단독 34/50 |

최고 난도 장애물의 작은 부분집합에서도 high53 단독은9/10, history high53은5/10이었다.
이는 두 배치에 속한10개 첫 episode의 서술적 결과이지 통계적 확정이 아니다.

## “평지/험지 분리”의 정확한 의미

기존 전환기는 지형 이름에 맞춰 별도 모델을 하나씩 지정하는 방식이 아니다.
**825개 이상적 높이 ray의 이력으로 v5와 새 expert의 행동을 혼합**하며,
고정 `HistoryDepthPolicyGate`/`HistoryGateConfig`를 그대로 사용했다.
평지에서는 expert 비중이0으로 내려가 v5 행동을 그대로 쓴다.

또한 이 **v5는 순수 평지 전용 모델이 아니라 이미 평지와 험지를 함께 학습한 정책**이다.
high53 역시 기존 혼합지형 continuation 모델이다. 순수 평지 전용 Robust42와
별도 험지 전용 모델을 새로 조합한 실험은 하지 않았다.

64초 history high53은 같은 checkpoint 단독보다 완주가13회 늘고 레인 이탈은24→7로
줄었지만 험지 낙상은52→58로 늘었다. 평지 속도는12.170→10.043m/s,
평지 낙상은3→7/50이었다. 완주·이탈 개선을 모든 안전/평지 성능 개선으로 바꾸지 않는다.
16초 v16에 history를 붙이면 완주는177→167/300으로 줄었다.

## 데모와 영상

새 배치 **geometry901/reset101**, 기존 최고 난도 **1.0**, 한 환경을16초씩 실행했다.
낙상·리셋·정체를 편집하지 않았고, reset 수에는16초 시간 제한 종료도 포함될 수 있다.
데모는 수치 평가와 별도의 정성 사례다. 한 영상의 거리를300회 평균으로 쓰지 않는다.

- [로컬 영상 플레이어](../artifacts/unseen_obstacles_20261001/index.html): clone 후 로컬에서 열며, GitHub 파일 화면에서는 아래 GIF/MP4 링크를 사용한다.
- **장애물 최고 후보 high53 단독:** [16초 원본 MP4](../artifacts/unseen_obstacles_20261001/videos/GUI_high53__obstacles_g901_r101.mp4)
- **장기 혼합험지 최고 후보 history high53:** [장애물](../artifacts/unseen_obstacles_20261001/videos/GUI_history_high53__obstacles_g901_r101.mp4) · [징검다리](../artifacts/unseen_obstacles_20261001/videos/GUI_history_high53__stones_g901_r101.mp4)
- **단기 최고 후보 v16:** [장애물](../artifacts/unseen_obstacles_20261001/videos/GUI_v16_control__obstacles_g901_r101.mp4) · [징검다리](../artifacts/unseen_obstacles_20261001/videos/GUI_v16_control__stones_g901_r101.mp4)

[![high53 단독 · 미사용 최고 난도 장애물 · 16초 전체 미리보기](../artifacts/unseen_obstacles_20261001/GUI_high53__obstacles_g901_r101_preview.gif)](../artifacts/unseen_obstacles_20261001/videos/GUI_high53__obstacles_g901_r101.mp4)

[![history high53 · 미사용 최고 난도 징검다리 · 16초 전체 미리보기](../artifacts/unseen_obstacles_20261001/GUI_history_high53__stones_g901_r101_preview.gif)](../artifacts/unseen_obstacles_20261001/videos/GUI_history_high53__stones_g901_r101.mp4)

GIF는 전체16초를6fps/480×270으로 샘플링한 미리보기이며 편집·배속이 아니다.

원본5개는 각16초/480프레임/30fps/1280×720이며 전체 디코딩으로 확인했다.
카메라 렌더 사이에 물리 step/root state가 바뀌지 않았음을 매 프레임 검사한다.
활성 디스플레이 GPU에서 실제 GUI viewport도 확인했다. 초기 native window presentation
문제는 디스플레이 장치를 바꿔 복구했고 관련 원시 로그·데스크톱/GPU/프로세스 메타데이터는 로컬에 보존한다.
[공개 media manifest](../artifacts/unseen_obstacles_20261001/media_manifest.json)는 영상별 원본 SHA·전체decode와 GIF 정보를 제공한다.
공개 MP4는 로컬 원본과 bytes가 같다. 인증정보·콘솔·세션DB·캐시·원시 장치 telemetry는 공개하지 않는다.

## 모델 선택과 보존

동일 지도 비교의 과거 결과를 기준으로 후보를 **새 점수화 전에** 고정했다.
보수적 권장 기본 v5와 높은 측정 성능 후보를 구분하고, 유리한 새 seed를 찾아 교체하지 않았다.

| 후보 | 고정 checkpoint | SHA256 앞12자리 |
|---|---|---|
| v5 teacher | `artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt` | `889836488fc2` |
| v16 control | `artifacts/terrain_demo/contact_slip_v16/runs/control/model_249.pt` | `1a937bfa9b33` |
| high53 | `artifacts/terrain_demo/seed_continuation_v22/runs/seed53/model_249.pt` | `5c86a9b6efeb` |

high53은 **v22 final249를 v24에서 재사용한 모델**이며 v24 capacity probe `model_1.pt`가 아니다.
v5 실행에는 고정 v10 checkpoint 안의 원본과 tensor가 정확히 같은60D teacher를 사용했다.
평가 실행 중 기존 checkpoint·source·과거 수치·문서1,865개의 bytes를 전부 보존했다.
후속 공개에서는 README/공개 범위 문서와 현재 checksum 목록만 의도적으로 갱신한다.
새 학습·의존성 설치·모델 변경은 없고, 사용자 요청에 따라 이 평가와 원본 영상을 GitHub에 공개한다.

과거의 공통 지도 실험도 시간별로 결과가 달랐다:
- [v19](REBASELINE_V19.md): 16초 v16 단독과 v5/v16 history는 둘 다265/450 완주지만 낙상은31 대39.
- [v24](LR_CONTINUATION_V24.md): 64초 history high53 234/300 대 같은 모델 단독204/300;16초/평지까지 보편적인 승격은 아님.
- [v11](HYBRID_V11.md): v5/v10 단순 깊이 교체는 일부 통과율이 높아도 전체 교체 기준FAIL.

## 이번 신규 평가 조건

- geometry/reset **901/101, 902/102**; 기존7지형×5난이도×5복제=175환경/지도.
- 5제어기: v5, v16 단독, v5/v16 history, high53 단독, v5/high53 history.
- 각 모델당 험지300+평지50 첫 episode, 전체 **1,750개 물리적 첫 episode**.
- 실제64초 주행에서16/64초를 별도 채점: **3,500개 종속 window 관측**, 독립 표본3,500개가 아님.
- 준비 실행은0물리step/0점수로 cache만 준비했다. 모든5제어기의 초기 root/joints/full91D/prefix88D/CPU+CUDA RNG와 지형 설정을 exact pairing했다.
- 성공: 최종 전진거리13.1/53.1m 이상이고 첫 episode 내 낙상·발 범위 레인 이탈·world exit가 없음. 최대거리/리셋 후 주행을 성공으로 세지 않는다.
- 저장된159개 학습 환경 설정/17개 geometry seed와 per-tile RNG seed 범위에서 새 배치와 겹침이 없음을 확인했다.
- **훈련한 장애물 종류의 새 무작위 배치**이며, 훈련 생성기 밖의 새로운 지형 종류 OOD 검증이 아니다.

[전체 수치·지도/지형/난도별 JSON](../artifacts/unseen_obstacles_20261001/summary/summary.json) ·
[한국어 전체 표](../artifacts/unseen_obstacles_20261001/summary/summary.ko.md) ·
[공개 첫episode JSON](../artifacts/unseen_obstacles_20261001/evaluations/) ·
[독립 원시 감사](../artifacts/unseen_obstacles_20261001/independent_data_audit.json) ·
[사전 고정 계획](../artifacts/unseen_obstacles_20261001/plan.json) ·
[실제 실행 명령 · portable 경로](../artifacts/unseen_obstacles_20261001/commands.jsonl).

공개 텍스트의 host 절대경로만 상대경로로 치환하고 모든 수치·배열·seed·조건을 유지했다.
[원문/공개본 SHA 대응](../artifacts/unseen_obstacles_20261001/publication_metadata.json)을 구분한다.
과거 감사 안의 SHA는 평가 당시 로컬 원문을 가리키며, 현재 공개본의 SHA와 같다는 뜻이 아니다.
공개 summary는 공개 JSON에서 다시 집계해 모든 통계가 원문과 정확히 일치한다.

## 재현

저장소 루트에서 기존 수업 환경을 사용한다. GUI의 `--device`는 해당 호스트의
디스플레이가 활성인 NVIDIA GPU를 선택한다. 기존 결과를 덮어쓰지 않도록 새 경로가 필요하다.

```bash
# 장애물 최고 후보 단독: 같은 신규 배치의 GUI + 무편집 원본
../run-python scripts/evaluate_unseen_terrain.py \
  --controller high53 --scenario obstacles --level 4 \
  --geometry 901 --seed 101 --seconds 16 --num-envs 1 \
  --device cuda:0 --real-time \
  --record outputs/unseen_replay/high53_obstacles.mp4 \
  --output outputs/unseen_replay/high53_obstacles.json

# 장기 혼합험지 최고 후보: controller만 history_high53으로 변경 가능
# 첫 점수화 전0step cache 준비와 새로운175환경 첫episode 평가
../run-python scripts/evaluate_unseen_terrain.py \
  --controller history_high53 --geometry 901 --seed 101 \
  --seconds 64 --num-envs 175 --device cuda:1 --headless --prepare \
  --output outputs/unseen_replay/prepare901.json
../run-python scripts/evaluate_unseen_terrain.py \
  --controller history_high53 --geometry 901 --seed 101 \
  --seconds 64 --num-envs 175 --device cuda:1 --headless \
  --output outputs/unseen_replay/history901.json
```

새 opt-in 진입점은 기존 frozen 평가기를 수정하지 않는다. CPU2,094개 tests,
새4파일 Pyflakes/제한적 Mypy/AST/diff 검사와 실제 GPU 평가·독립 수치 감사가 통과했다.
외부 IsaacLab 전체 typecheck, 실제 센서/로봇, 통계적 유의성/모든 지형 우월성은 검증하지 않았다.

Isaac Sim 없이 공개 raw를 다시 집계할 수 있다(새 output 디렉터리 필요):

```bash
python3 scripts/summarize_unseen_terrain.py \
  --input-dir artifacts/unseen_obstacles_20261001/evaluations \
  --output-dir outputs/unseen_public_recheck
```

Python3.11/3.12의 부동소수점 합산 차이로 평균의 마지막 자리에 약1e-15 차이가 날 수 있다.
공개 summary는 최초 집계와 같은 system Python을 사용해 원래 통계를 그대로 보존했다.
