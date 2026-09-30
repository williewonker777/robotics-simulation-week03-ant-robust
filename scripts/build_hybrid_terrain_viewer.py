"""Build a local gallery for every predeclared qualitative terrain video."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/terrain_demo/hybrid_v11/other_terrains"
MODES = ("v5", "v10", "hybrid")
FAMILIES = (
    ("stairs", "계단"), ("obstacles", "불연속 장애물"),
    ("rough", "울퉁불퉁한 지형"), ("slope", "경사"), ("waves", "파도형 지형"),
)


def main():
    groups = []
    for family, label in FAMILIES:
        paths = [f"{mode}/{family}.mp4" for mode in MODES]
        groups.append({"family": family, "label": label, "paths": paths,
                       "note": "새 영상: 5환경 중 지형별 1개 로봇을 추적합니다. 같은 지형의 세 정책은 같은 초기 조건입니다."})
    groups.append({"family": "stepping_stones", "label": "돌다리 (기존 영상)",
                   "paths": [f"../attempt02/videos/{mode}_seed42_stones10.mp4" for mode in MODES],
                   "note": "기존 돌다리 영상: 10환경 중 env0를 추적합니다. 새 5환경 영상과 배치가 다르며 정량 결과와 별도입니다."})
    for group in groups:
        for path in group["paths"]:
            if not (DIRECTORY / path).is_file():
                raise FileNotFoundError(path)
    page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ant v11 — 다른 험지 비교</title>
<style>body{margin:0;padding:22px;background:#101827;color:#edf4ff;font:17px system-ui;line-height:1.6}
h1{font-size:27px;margin:0 0 12px}h2{font-size:20px;margin:12px 0 8px}nav{display:flex;gap:10px;flex-wrap:wrap}
button{padding:10px 16px;border:1px solid #4a6482;border-radius:8px;background:#24354c;color:#edf4ff;font:inherit;cursor:pointer}
button[aria-pressed=true]{background:#166799;border-color:#81cdff}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
video{width:100%;background:black;border-radius:8px}.note{color:#becde0}#status{color:#91d8ff;min-height:1.6em;margin:12px 0}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}a{color:#91d8ff}
@media(max-width:900px){main{grid-template-columns:1fr}}input{width:18px;height:18px;vertical-align:middle}
</style></head><body><h1>다른 험지에서도 v5 · v10 · 하이브리드 비교</h1>
<nav id="terrain-tabs" aria-label="지형 선택"></nav><div id="status" role="status"></div>
<main><section><h2>v5 — 기존 정책</h2><video controls muted playsinline preload="auto"></video></section>
<section><h2>v10 — anchored</h2><video controls muted playsinline preload="auto"></video></section>
<section><h2>하이브리드 — 깊이로 전환</h2><video controls muted playsinline preload="auto"></video></section></main>
<div class="controls"><button id="restart">처음부터 함께 재생</button><button id="pause">모두 정지</button>
<label><input id="cycle" type="checkbox" checked> 16초마다 다음 지형 자동 재생</label></div>
<p class="note"><strong>난이도 1.0 · 각 16초 무편집 · geometry68 / reset42 / 정책 seed42</strong><br>
<span id="condition-note"></span><br>alpha=0은 v5, alpha=1은 v10입니다. 낙상·리셋을 그대로 포함합니다.
리셋 횟수에는 시간 제한 종료도 포함됩니다.
리셋 뒤 이동은 첫 에피소드 성공이 아니며, 이 영상들은 기존 수치 집계나 모델 선택에 추가하지 않습니다.
이상적인 높이 스캔이며 실제 RGB-D 카메라 검증이 아닙니다.</p>
<p><a href="../index.html">기존 정량 결과와 돌다리 비교</a> · <a href="manifest.json">새 영상 조건·검증</a></p>
<script>
const groups = GROUP_DATA;
const videos = [...document.querySelectorAll('video')];
const status = document.getElementById('status');
const cycle = document.getElementById('cycle');
let current = 0, generation = 0, loading = false;
const tabs = groups.map((g, i) => {
  const b = document.createElement('button'); b.textContent = g.label;
  b.setAttribute('aria-pressed', 'false'); b.onclick = () => selectGroup(i);
  document.getElementById('terrain-tabs').appendChild(b); return b;
});
async function selectGroup(index) {
  const stamp = ++generation; loading = true; current = index;
  const group = groups[index]; videos.forEach(v => v.pause());
  tabs.forEach((b, i) => b.setAttribute('aria-pressed', String(i === index)));
  document.getElementById('condition-note').textContent = group.note;
  status.textContent = group.label + ' — 불러오는 중';
  try {
    await Promise.all(videos.map((v, i) => new Promise((resolve, reject) => {
      const ready = () => { cleanup(); resolve(); };
      const error = () => { cleanup(); reject(new Error(v.error?.message || '영상 로드 실패')); };
      const cleanup = () => { v.removeEventListener('canplay', ready); v.removeEventListener('error', error); };
      v.addEventListener('canplay', ready); v.addEventListener('error', error);
      v.muted = true; v.src = group.paths[i]; v.load();
    })));
    if (stamp !== generation) return;
    videos.forEach(v => { v.currentTime = 0; });
    await Promise.all(videos.map(v => v.play()));
    if (stamp === generation) { loading = false; status.textContent = group.label + ' — 세 정책 동시 재생'; }
  } catch (error) {
    if (stamp === generation) { loading = false; status.textContent = group.label + ' — ' + error.message; }
  }
}
document.getElementById('restart').onclick = () => selectGroup(current);
document.getElementById('pause').onclick = () => {
  ++generation; loading = false; videos.forEach(v => v.pause());
  status.textContent = groups[current].label + ' — 정지';
};
videos.forEach(v => v.addEventListener('ended', () => {
  if (!loading && cycle.checked && videos.every(item => item.ended)) selectGroup((current + 1) % groups.length);
}));
selectGroup(0);
</script></body></html>'''
    page = page.replace("GROUP_DATA", json.dumps(groups, ensure_ascii=False))
    with (DIRECTORY / "index.html").open("x") as stream:
        stream.write(page)
    print(DIRECTORY / "index.html")


if __name__ == "__main__":
    main()
