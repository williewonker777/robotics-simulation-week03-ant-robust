"""Offline publication checks; CPU-only, no browser or Isaac Sim imports."""

import csv
import gzip
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import shutil
import sys
from statistics import mean
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'report/web'
RESULTS = ROOT / 'artifacts/combo_v28/results'
SPEC = importlib.util.spec_from_file_location('web_slide_builder', ROOT / 'scripts/build_web_slides_data.py')
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def rows(name):
    with (RESULTS / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_generated_data_matches_and_is_deterministic(tmp_path):
    first, second = tmp_path / 'first.js', tmp_path / 'second.js'
    for output in (first, second):
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_web_slides_data.py'),
                        '--output', str(output)], cwd=tmp_path, check=True)
    assert first.read_bytes() == second.read_bytes() == (WEB / 'assets/data.js').read_bytes()
    assert first.read_text(encoding='utf-8') == BUILDER.serialize_data(BUILDER.build_data())


def test_recipe_values_match_independent_csv_sources():
    data = BUILDER.build_data()
    for phase in ('selection', 'confirmation'):
        expected = rows(phase + '_recipes.csv')
        assert len(data[phase]) == len(expected)
        for actual, source in zip(data[phase], expected):
            assert actual['recipe'] == source['recipe']
            for field in ('demo_mean', 'demo_seed_std', 'flat', 'boxes_10', 'fall_rate'):
                assert actual[field] == float(source[field])
            for key, value in actual['category_means'].items():
                assert value == float(source['cat_' + key])
    selection = {row['recipe']: row for row in data['selection']}
    confirmation = {row['recipe']: row for row in data['confirmation']}
    names = ('sticklim_e5_d0+stock', 'lim_e5_d1+stock', 'ref_lim_f3a', 'flat_e0_d1', 'flat_e0_d0')
    assert [selection[n]['demo_mean'] for n in names] == [62.18, 61.17, 60.96, 33.57, 30.13]
    assert [confirmation[n]['demo_mean'] for n in names] == [62.47, 61.38, 60.77, 33.46, 30.08]
    assert [selection[n]['demo_seed_std'] for n in names[:2]] == [.70, 1.47]
    assert len(data['ranking']) == 11
    assert not any(row['recipe'].startswith('ref_') for row in data['ranking'][:8])
    expected_new = [r for r in rows('selection_recipes.csv') if not r['recipe'].startswith('ref_')
                    and r['recipe'] not in ('flat_e0_d0', 'flat_e0_d1')]
    assert [r['recipe'] for r in data['ranking'][:8]] == [r['recipe'] for r in expected_new[:8]]


def test_ladder_and_alternatives_match_csv_means_and_seed_pairs():
    data = BUILDER.build_data()
    recipes = {row['recipe']: row for row in rows('selection_recipes.csv')}
    checkpoints = {row['checkpoint']: row for row in rows('selection_checkpoints.csv')}
    expected_ladder = ['flat_e0_d0', 'sticklim_e0_d0', 'sticklim_e5_d0', 'sticklim_e5_d0+stock']
    assert [row['recipe'] for row in data['ladder']] == expected_ladder
    for index, row in enumerate(data['ladder']):
        assert row['mean'] == float(recipes[row['recipe']]['demo_mean'])
        assert row['pairs'] == 3
        assert row['seed_values'] == [
            {'seed': seed, 'mean': float(checkpoints[f"{row['recipe']}_s{seed}"]['demo'])}
            for seed in (42, 43, 44)]
        if index == 0:
            assert row['delta'] is None and row['wins'] is None
        else:
            reference = expected_ladder[index - 1]
            assert row['delta'] == row['mean'] - float(recipes[reference]['demo_mean'])
            wins = sum(float(checkpoints[f"{row['recipe']}_s{seed}"]['demo']) >
                       float(checkpoints[f'{reference}_s{seed}']['demo']) for seed in (42, 43, 44))
            assert row['wins'] == wins == 3
    expected_alternatives = [('sticklim_e5_d1', 'sticklim_e5_d0'),
                             ('sticklim_e5_d0+recovery', 'sticklim_e5_d0+stock'),
                             ('mine_e5_d0', 'sticklim_e5_d0')]
    assert [(row['recipe'], row['reference']) for row in data['alternatives']] == expected_alternatives
    for row in data['alternatives']:
        actual, reference = recipes[row['recipe']], recipes[row['reference']]
        assert row['mean'] == float(actual['demo_mean'])
        assert row['delta'] == float(actual['demo_mean']) - float(reference['demo_mean'])
        assert row['fall_rate'] == float(actual['fall_rate'])
        assert row['reference_fall_rate'] == float(reference['fall_rate'])
        assert row['pairs'] == 3
        wins = sum(float(checkpoints[f"{row['recipe']}_s{seed}"]['demo']) >
                   float(checkpoints[f"{row['reference']}_s{seed}"]['demo']) for seed in (42, 43, 44))
        assert row['wins'] == wins == 0


def test_fresh_and_official_values_preserve_source_precision():
    data = BUILDER.build_data()
    fresh = json.loads((RESULTS / 'fresh_process_check.json').read_text())
    assert data['fresh'] == [r for r in fresh['recipes'] if r['kind'] == 'fresh_demo']
    # Published prose says 60.92, but the source value is 60.914705... (60.91).
    assert [round(r['fresh_demo'], 2) for r in data['fresh']] == [62.08, 61.16, 60.91]
    parity = json.loads((RESULTS / 'official_parity.json').read_text())
    assert list(data['official'].values()) == parity['post_fix_2026_10_03']['official']['submission_seed43_run1']
    assert data['official'] == {'mean': 61.985765, 'std': 17.888110}


def test_effects_are_paired_and_protocol_counts_match_sources():
    data = BUILDER.build_data()
    source = json.loads((RESULTS / 'summary.json').read_text())
    ranking = {r['recipe']: r for r in source['selection']['ranking']}
    terrains = ('flat', 'stick', 'lim', 'sticklim', 'mine', 'all')
    expected = {
        'entropy': [(f'{t}_e0_d{d}', f'{t}_e5_d{d}') for t in terrains for d in (0, 1)],
        'randomization': [(f'{t}_e{e}_d0', f'{t}_e{e}_d1') for t in terrains for e in (0, 5)],
        'terrain': [(f'flat_e{e}_d{d}', f'{t}_e{e}_d{d}') for t in ('stick', 'lim', 'sticklim')
                    for e in (0, 5) for d in (0, 1)],
    }
    for name, pairs in expected.items():
        deltas = [ranking[b]['demo_mean'] - ranking[a]['demo_mean'] for a, b in pairs]
        assert data['effects'][name]['mean'] == pytest.approx(mean(deltas))
        assert data['effects'][name]['wins'] == sum(d > 0 for d in deltas)
        assert data['effects'][name]['pairs'] == 12
    for row in data['effects']['continuation']:
        assert row['delta'] == ranking[row['recipe']]['demo_mean'] - ranking[row['parent']]['demo_mean']
    for row in data['effects']['recovery']:
        assert row['delta'] == ranking[row['recipe']]['demo_mean'] - ranking[row['stock']]['demo_mean']
        assert row['recovery_fall_rate'] == ranking[row['recipe']]['fall_rate']
    assert data['protocol']['new_policies'] == len(rows('training_runs.csv')) == 84
    assert data['protocol']['new_stage1_policies'] == 66
    assert data['protocol']['continuation_policies'] == 18
    assert sum(c['count'] for c in data['categories']) == data['protocol']['conditions'] == 28
    assert data['protocol']['selection_evaluations'] == source['selection']['results'] == 2744


def test_learning_curve_values_match_six_published_run_csvs():
    data = BUILDER.build_data()
    points = data['learning_curves']['points']
    assert [p['iteration'] for p in points] == sorted({p['iteration'] for p in points})
    assert points[0]['iteration'] == 0 and points[-1]['iteration'] == 1599
    sources = [path for path in data['sources'] if path.endswith('/curves.csv')]
    assert len(sources) == 6
    curves = {}
    for path in sources:
        with (ROOT / path).open() as stream:
            curves[path] = {int(r['iteration']): float(r['mean_reward']) for r in csv.DictReader(stream)}
    for point in points:
        stage2 = point['iteration'] >= 1000
        values = [curve[point['iteration'] - (1000 if stage2 else 0)] for path, curve in curves.items()
                  if ('v28s2_' in path) == stage2]
        assert point['mean'] == mean(values)
        assert (point['min'], point['max']) == (min(values), max(values))


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.slides = []
        self.notes = 0

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'section' and 'slide' in attrs.get('class', '').split():
            self.slides.append(attrs)
        if 'speaker-notes' in attrs.get('class', '').split():
            self.notes += 1
        for key in ('src', 'href', 'poster', 'data'):
            if key in attrs:
                self.references.append((attrs[key], tag == 'a' and key == 'href'))
        assert 'srcset' not in attrs, 'Add explicit srcset parsing before using responsive resources'


def check_reference(value, parent, navigation=False):
    parsed = urlsplit(value)
    if parsed.scheme in ('http', 'https') or parsed.netloc:
        assert navigation, f'External resource load: {value}'
        return
    assert not parsed.scheme, f'Non-local resource scheme: {value}'
    if parsed.path:
        assert not parsed.path.startswith('/'), f'Not file/Pages portable: {value}'
        assert (parent / unquote(parsed.path)).is_file(), value


def test_html_and_css_resources_are_local_and_present():
    parser = Page()
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    parser.feed(html)
    for value, navigation in parser.references:
        check_reference(value, WEB, navigation)
    for css in (WEB / 'assets').glob('*.css'):
        text = css.read_text(encoding='utf-8')
        assert '@import' not in text
        for value in re.findall(r'url\(\s*[\"\']?([^\)\"\']+)', text):
            check_reference(value.strip(), css.parent)
    for script in (WEB / 'assets').glob('*.js'):
        assert not re.search(r'https?://|\bfetch\s*\(|XMLHttpRequest|WebSocket|import\s*\(', script.read_text())
    assert len(parser.slides) == 18
    main_slides = [slide for slide in parser.slides if 'appendix' not in slide.get('class', '').split()]
    assert len(main_slides) == 14
    assert [slide['id'] for slide in parser.slides] == [f'slide-{i}' for i in range(1, len(parser.slides) + 1)]
    assert len(parser.slides) == parser.notes
    seconds = [int(slide['data-note-seconds']) for slide in parser.slides]
    assert all(20 <= value <= 35 for value in seconds)
    assert sum(int(slide['data-note-seconds']) for slide in main_slides) == 280
    assert sum(seconds) == 360
    assert len(re.findall(r'<video\b', html)) == 2
    for video in re.findall(r'<video\b[^>]*>', html):
        assert all(re.search(rf'\b{name}(?:\s|=|>)', video) for name in ('muted', 'loop', 'playsinline', 'controls', 'poster'))


def test_problem_baseline_uses_unrounded_raw_episode_aggregates():
    selected = []
    with gzip.open(RESULTS / 'selection_evaluations.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if (row['checkpoint_id'] in ['flat_e0_d0_s42', 'flat_e0_d0_s43', 'flat_e0_d0_s44']
                    and row['terrain'] in ('flat', 'boxes_10')
                    and row['friction'] == 1.0 and row['combine_mode'] == 'average'):
                selected.append(row)
    data = BUILDER.build_data()
    assert len(selected) == 6
    for terrain in ('flat', 'boxes_10'):
        source = [row['return_mean'] for row in selected if row['terrain'] == terrain]
        assert len(source) == 3
        assert data['baseline_problem'][terrain] == mean(source)
    assert data['baseline_problem']['boxes_10'] == pytest.approx(4.247312788935475)
    assert round(data['baseline_problem']['boxes_10'], 1) == 4.2
    assert round(data['baseline_problem']['flat'], 1) == 138.4
    # Recipe CSV remains unchanged for data fidelity.
    assert next(r for r in data['selection'] if r['recipe'] == 'flat_e0_d0')['boxes_10'] == 4.25


def plain_text(markup):
    class Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, text):
            self.parts.append(text)

    parser = Text()
    parser.feed(markup)
    return ' '.join(' '.join(parser.parts).split())


def test_main_presenter_notes_fit_character_budget():
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    sections = re.findall(r'<section\b([^>]*)>(.*?)</section>', html, re.S)
    main = [body for attrs, body in sections if 'appendix' not in attrs]
    assert len(main) == 14
    notes = []
    for section in main:
        matches = re.findall(r'<aside\b[^>]*class="speaker-notes"[^>]*>(.*?)</aside>', section, re.S)
        assert len(matches) == 1
        notes.append(plain_text(matches[0]))
    assert len(' '.join(notes)) <= 1500
    assert all(notes)


def test_actual_javascript_rendered_scores_and_labels():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable; browser verification must cover rendered text')

    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = []

        def handle_starttag(self, tag, attributes):
            self.elements.append({'tag': tag, 'attrs': dict(attributes)})

    elements = Elements()
    elements.feed((WEB / 'index.html').read_text(encoding='utf-8'))
    # Execute the actual script, not a duplicate number formatter. The tiny DOM
    # supports its current UI API only; layout remains a real-browser check.
    runner = r'''
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor(item = {attrs:{}}) {
    this.attrs = item.attrs; this.tag = item.tag; this.innerHTML = ''; this.textContent = '';
    this.dataset = Object.fromEntries(Object.entries(this.attrs).filter(([k]) => k.startsWith('data-')).map(([k,v]) => [k.slice(5).replace(/-([a-z])/g, (_,c) => c.toUpperCase()),v]));
    this.style = {setProperty(){}};
    this.classList = {contains: name => (this.attrs.class || '').split(' ').includes(name), toggle(){}};
    this.firstElementChild = {style:{}};
  }
  setAttribute(k,v) { this.attrs[k] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener() {}
  querySelectorAll() { return []; }
  querySelector() { return {textContent: 'presentation'}; }
}
const elements = input.elements.map(item => new Element(item));
const match = (element, selector) => {
  if (selector === '.slide') return element.classList.contains('slide');
  const m = /^\[([^=\]]+)(?:="([^"]*)")?\]$/.exec(selector);
  return m && Object.hasOwn(element.attrs,m[1]) && (m[2] === undefined || element.attrs[m[1]] === m[2]);
};
const document = {
  getElementById: id => {const e=elements.find(e=>e.attrs.id===id); if(!e) throw Error('Missing id '+id);return e;},
  querySelectorAll: selector => elements.filter(e=>match(e,selector)),
  querySelector: selector => {const e=elements.find(e=>match(e,selector));if(!e) throw Error('Missing selector '+selector);return e;},
  body: new Element(), documentElement: new Element()
};
const events = {};
const window = {innerWidth:1920,innerHeight:1080,location:{hash:'#1'},addEventListener(name,callback){events[name]=callback;}};
const context = {window,document,history:{replaceState(){}},console};
vm.createContext(context);
vm.runInContext(fs.readFileSync(input.data,'utf8'),context);
vm.runInContext(fs.readFileSync(input.script,'utf8'),context);
if(document.getElementById('position').textContent !== '1 / 18') throw Error('Initial slide position');
window.location.hash = '#18'; events.hashchange();
if(document.documentElement.dataset.ready !== 'true') throw Error('Deck not ready');
process.stdout.write(JSON.stringify(elements.map(e=>({attrs:e.attrs,text:e.textContent,html:e.innerHTML}))));
'''
    result = subprocess.run([node, '-e', runner], input=json.dumps({
        'elements': elements.elements, 'data': str(WEB / 'assets/data.js'),
        'script': str(WEB / 'assets/slides.js')}), text=True, capture_output=True, check=True)
    rendered = json.loads(result.stdout)
    by_id = {row['attrs']['id']: row for row in rendered if 'id' in row['attrs']}
    assert by_id['position']['text'] == '18 / 18'
    assert by_id['progress']['attrs']['aria-valuenow'] == 18
    assert by_id['slide-18']['attrs']['aria-hidden'] == 'false'
    assert by_id['slide-1']['attrs']['aria-hidden'] == 'true'
    by_value = {row['attrs']['data-value']: row['text'] for row in rendered if 'data-value' in row['attrs']}
    assert by_value['baseline_problem.flat'] == '138.4'
    assert by_value['baseline_problem.boxes_10'] == '4.2'
    assert by_value['official.mean'] == '61.985765'
    assert by_value['official.std'] == '17.888110'
    for row in rendered:
        value = row['attrs'].get('data-value', '')
        if value and not value.startswith(('protocol.', 'official.')):
            assert re.fullmatch(r'-?\d+\.\d', row['text']), (value, row['text'])
    for name in ('ranking-chart', 'ladder-chart', 'effects-chart', 'confirmation-chart',
                 'fresh-chart', 'heatmap-chart', 'learning-chart', 'alternatives-rows'):
        visible = plain_text(by_id[name]['html'])
        assert visible
        # Check numeric value text, not configuration labels such as entropy 0.005.
        for text_markup in re.findall(r'<(?:text|td)\b[^>]*>(.*?)</(?:text|td)>', by_id[name]['html'], re.S):
            value_text = plain_text(text_markup)
            if re.fullmatch(r'[+−-]?\d+\.\d+', value_text):
                assert re.fullmatch(r'[+−-]?\d+\.\d', value_text), (name, value_text)
        assert not re.search(r'\bv(?:5|28)\b', visible, re.I), (name, visible)
    ladder = plain_text(by_id['ladder-chart']['html'])
    for phrase in ('험지 학습', '높이 관측도 바닥 기준', 'entropy 0.005', '600 iteration 추가',
                   '30.1', '43.4', '58.7', '62.2', '+13.2', '+15.3', '+3.5'):
        assert phrase in ladder
    for row in rendered:
        assert not re.search(r'\bv(?:5|28)\b', plain_text(row['html']) + row['text'], re.I)
    ranking = plain_text(by_id['ranking-chart']['html'])
    assert '62.2' in ranking and '61.2' in ranking and '61.0' in ranking
    assert '+600 it' in ranking or '추가 600 it' in ranking
    assert '±' not in ranking and 'n=1' not in ranking
    assert 'F3a 원자료' not in plain_text(by_id['fresh-chart']['html'])
    learning = plain_text(by_id['learning-chart']['html'])
    assert '재시작' in learning or '새 run' in learning


def test_methods_and_glossary_keep_required_explanations():
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    sections = {int(re.search(r'id="slide-(\d+)"', attrs).group(1)): body
                for attrs, body in re.findall(r'<section\b([^>]*)>(.*?)</section>', html, re.S)}
    methods = plain_text(sections[8])
    assert '적용한 방법' in methods
    assert len(re.findall(r'<article\b', sections[8])) == 3
    for phrase in ('험지 학습', 'Stick', 'Lim', '반반', '0.45 m', 'ray 1개', '0.31 m',
                   '−5~5 cm, 50%', '5~15 cm, 30%', '각 10%', '−10~10 cm',
                   'PPO entropy 0.005', '0에서 0.005', '추가 학습 600 it',
                   '1,000 iteration', '600 iteration', 'optimizer', '4,096개', '32 step', '9종'):
        assert phrase in methods
    assert '개선 과정' in plain_text(sections[9])
    glossary = plain_text(sections[18])
    assert '용어 정리' in glossary
    body = sections[18].split('<div class="glossary-columns">', 1)[1].split('<footer', 1)[0]
    assert len(re.findall(r'<p>', body)) == 8
    for phrase in ('첫 episode', '16초(960 step)', '7항', '100개', '28개', '0.31 m',
                   '42, 43, 44', 'seed 24', '2028', '2029', '짝 비교', '랜덤화(Robust42)',
                   '0.45~1.35', '×0.8~1.2', '4~8초', '회복 보상(Stick)', '−10',
                   '원래 보상', 'play_one_episode.py', '새 프로세스'):
        assert phrase in glossary
    assert '2029는 선택에 쓰지 않은 새 지형' in glossary
    assert '발밑 지면 기준 몸통 높이가 0.31 m보다 낮아져' in glossary
    # No publication source footers or first-person analysis notes on stage.
    assert all('<a ' not in footer for number, section in sections.items() if number != 14
               for footer in re.findall(r'<footer[^>]*>(.*?)</footer>', section, re.S))
    assert not re.search(r'(?:^|\s)(?:내|제)\s', plain_text(html))
    for internal in ('F3a 원자료', '표기 정밀도', '60.914705', '1.1475'):
        assert internal not in plain_text(html)


def test_cover_date_and_glossary_present_tense():
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    cover_meta = re.search(r'<div class="cover-meta">(.*?)</div>', html, re.S)
    assert cover_meta is not None
    assert plain_text(cover_meta.group(1)) == '2026. 10. 08.'
    glossary = html.split('<div class="glossary-columns">', 1)[1].split('<footer', 1)[0]
    text = plain_text(glossary)
    for noun in ('합', '값', '비율', '학습', '새 지형', '것', '뜻', '방법'):
        assert noun + '입니다.' in text
        assert noun + '이었습니다' not in text
    # Definitions are present tense; descriptions of the experiment stay past tense.
    for phrase in ('7항을 썼습니다.', '관측 노이즈를 바꿨습니다.',
                   '밀기도 적용했습니다.', '원래 보상으로 계산했습니다.'):
        assert phrase in text


def test_basic_code_comparison_and_version_free_visible_text():
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    sections = {int(re.search(r'id="slide-(\d+)"', attrs).group(1)): body
                for attrs, body in re.findall(r'<section\b([^>]*)>(.*?)</section>', html, re.S)}
    comparison = sections[4]
    assert '기본 코드와 다른 점' in plain_text(comparison)
    body = re.search(r'<tbody>(.*?)</tbody>', comparison, re.S)
    assert body is not None
    rows = re.findall(r'<tr\b[^>]*>(.*?)</tr>', body.group(1), re.S)
    assert len(rows) == 5
    expectations = (
        ('학습 지형', 'plane', 'Stick', 'Lim', '반씩'),
        ('몸통 높이', '월드 z', 'base_pos_z', '광선 1개', '거리'),
        ('넘어짐', '몸통 z 0.31 m 미만', '바닥에서 몸통까지 0.31 m 미만'),
        ('PPO entropy', '0.0', '0.005'),
        ('학습량', '1,000', '600'),
    )
    for row, phrases in zip(rows, expectations):
        cells = re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', row, re.S)
        assert len(cells) == 3
        text = plain_text(row)
        for phrase in phrases:
            assert phrase in text
    comparison_text = plain_text(comparison)
    for phrase in ('보상 7가지', '행동', '신경망', '다른 PPO', '환경 수', '에피소드 길이', '기본 코드'):
        assert phrase in comparison_text
    visible = plain_text(html).replace('Week03-Ant-Combo-v28-Play', '')
    assert not re.search(r'\bv(?:5|28)\b', visible, re.I)
    conclusion = plain_text(sections[14])
    assert '$CHECKPOINT' in conclusion
    assert '상세 실험 기록' in conclusion
