"""Build a local v10 results page without selecting successful videos/seeds."""

import json
from pathlib import Path


def table(summary, scope):
    rows = []
    names = {"frozen_v5": "기존 v5", "free": "착지 후보 / prior 없음", "anchored": "착지 후보 / v5 prior"}
    for group, name in names.items():
        row = summary["totals"][group] if scope == "all" else summary["per_family"][group]["stepping_stones"]["4"]
        values = [f"{row[key]}/{row['n']} ({100 * row[key] / row['n']:.1f}%)" for key in ("one", "six", "falls", "lane")]
        rows.append(f"<tr><th>{name}</th>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    return '<table><tr><th>정책</th><th>1타일</th><th>6타일</th><th>낙상</th><th>레인이탈</th></tr>' + "".join(rows) + "</table>"


def main():
    directory = Path(__file__).resolve().parents[1] / "artifacts/terrain_demo/prior_v10"
    summary = json.loads((directory / "summary.json").read_text())
    verdict = "교체 기준 PASS" if summary["promotion_gate"]["passed"] else "교체 기준 FAIL — 기존 v5 유지"
    direction = "PASS" if summary["directional_gate"]["passed"] else "FAIL"
    videos = []
    for seconds in (16, 64):
        panels = []
        for name, caption in (("frozen_v5", "기존 v5"), ("anchored_seed42", "v5 prior (seed42)")):
            panels.append(f'<div><h3>{caption}</h3><video class="v{seconds}" controls muted preload="metadata" '
                          f'src="videos/{name}_stones10_{seconds}s.mp4"></video></div>')
        videos.append(f'<h2>{seconds}초 무편집 비교</h2><div class="videos">' + "".join(panels) + '</div>'
                      f'<button onclick="playPair({seconds})">처음부터 함께 재생</button>'
                      f'<button onclick="pausePair({seconds})">정지</button>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ant v10 학습 전용 행동 prior 비교</title>
<style>body{background:#101827;color:#e9eff8;font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1300px}
h1{font-size:28px}h2{margin-top:30px}a{color:#82ccff}table{width:100%;border-collapse:collapse;margin:16px 0}
th,td{padding:12px;text-align:left;border-bottom:1px solid #344052}th{color:#c4dded}
.verdict{padding:16px;background:#343146;border-radius:10px;font-weight:bold}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}
video{width:100%;background:#000;border-radius:10px}button{padding:12px 20px;margin:8px 8px 0 0;border:0;border-radius:8px;cursor:pointer}
.note{color:#b9c7d8;line-height:1.7}@media(max-width:750px){.videos{grid-template-columns:1fr}table{font-size:13px}th,td{padding:6px}}
</style><h1>험지 보행 v10: 학습 중에만 기존 동작을 참조</h1>
<p>착지 후보 88D 학생 정책 + 고정 v5 행동 평균 보조 손실. 두 조건은 보상·물리·학습 예산이 같고, 실행 시에는 학생만 사용합니다.</p>
<div class="verdict">VERDICT<br>별도 안전성 개선 가설: DIRECTION</div>
<h2>신규 지형 전체 / 16초 기존 평가</h2>ALLTABLE
<p class="note">6회 학습: prior λ0 / λ0.02 × 시드42·43·44, 각750×4096×32단계.
신규 geometry66/67, reset40/41. 14개 평가 / 평지 포함2,450개 첫 에피소드.
험지 분모 v5 300회 / 학습된 각 조건900회. v5 기준선 표본을 복제하지 않았습니다.
두 개 지도이며2,450개 독립 지도가 아닙니다.</p>
<h2>최고 난이도1.0 돌다리 / 16초</h2>HARDTABLE
<h2>별도64초 지속 보행 진단</h2>
<p class="note">최고 난이도 돌다리만 지도당10개 초기화: v5 20회 / 각 학습 조건60회, 총140회.
16초 평가의 표본에 합치거나 교체 기준에 사용하지 않습니다. 10환경과175환경은 초기화 배치가 다릅니다.
첫13.1m/53.1m 도달 뒤 낙상·이탈하면 최종 통과 성공이 아닙니다.</p>
<p><a href="horizon_summary.md">64초 수치와 도달 시간</a> · <a href="horizon_summary.json">64초 진단 JSON</a></p>
<p class="note">아래 영상은 사전 지정 geometry66 / reset40 / 돌다리1.0 / 학습seed42입니다.
성공 사례를 골라내지 않았으며 실패와 리셋을 그대로 포함합니다. 별도1환경 정성 사례로 통계에 추가하지 않습니다.</p>
VIDEOS
<p class="note">이상적 높이 레이캐스트이지 실제 RGB-D가 아닙니다. 후보 발끝/보조 손실은 전체 접촉 지지·정확한 IK·낙상 방지·실물 전이를 보장하지 않습니다.</p>
<p><a href="summary.md">전체 수치</a> · <a href="summary.json">시드/지형별 집계</a> ·
<a href="verification.json">검증</a> · <a href="frozen.json">고정 모델/조건</a> ·
<a href="video_manifest.json">영상 조건</a> · <a href="../../../docs/PRIOR_V10.md">방법/한계</a></p>
<script>function playPair(s){document.querySelectorAll('.v'+s).forEach(v=>{v.currentTime=0;v.play().catch(console.error);});}
function pausePair(s){document.querySelectorAll('.v'+s).forEach(v=>v.pause());}</script></html>'''
    page = (page.replace("VERDICT", verdict).replace("DIRECTION", direction)
            .replace("ALLTABLE", table(summary, "all")).replace("HARDTABLE", table(summary, "hard"))
            .replace("VIDEOS", "".join(videos)))
    (directory / "index.html").write_text(page)
    print(directory / "index.html")


if __name__ == "__main__":
    main()
