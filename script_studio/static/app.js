'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (t) => { t = Math.max(0, t || 0); const m = Math.floor(t / 60), s = (t - m * 60).toFixed(1).padStart(4, '0'); return `${m}:${s}`; };

const state = { mode: 'story', inc: '10', docs: [], audio: null, jobId: null, job: null, rendered: new Map(), timer: null, config: {} };

// ------------------------------------------------------------------ config & status
async function loadConfig() {
  try {
    state.config = await (await fetch('/api/config')).json();
    const c = state.config;
    $('#appVersion').textContent = c.version ? 'version ' + c.version : '';
    $('#status').innerHTML =
      `<span class="pill ${c.has_api_key ? 'ok' : 'bad'}">${c.has_api_key ? 'Claude ready · ' + esc(c.model) : 'Add your API key in Settings'}</span>` +
      `<span class="pill ${c.transcription ? 'ok' : ''}">${c.transcription ? 'Transcription: ' + (c.transcription === 'openai' ? 'OpenAI' : 'on this computer') : 'No transcription (text & music only)'}</span>`;
    if (!c.has_api_key) openSettings(true);
  } catch { /* offline */ }
}

// ------------------------------------------------------------------ settings
async function openSettings(firstRun) {
  $('#settingsModal').classList.remove('hidden');
  $('#settingsMsg').textContent = firstRun ? 'Welcome! Add your Anthropic key to start writing scripts.' : '';
  $('#settingsMsg').className = 'msg';
  try {
    const s = await (await fetch('/api/settings')).json();
    $('#anthHint').textContent = s.ANTHROPIC_API_KEY ? `saved ${s.ANTHROPIC_API_KEY_hint}` : '';
    $('#oaiHint').textContent = s.OPENAI_API_KEY ? `saved ${s.OPENAI_API_KEY_hint}` : '';
    $('#modelSel').value = s.SCRIPT_MODEL || '';
  } catch { /* ignore */ }
}
$('#gearBtn').addEventListener('click', () => openSettings(false));
$('#closeSettings').addEventListener('click', () => $('#settingsModal').classList.add('hidden'));
$('#saveSettings').addEventListener('click', async () => {
  const body = { SCRIPT_MODEL: $('#modelSel').value };
  const a = $('#anthKey').value.trim(), o = $('#oaiKey').value.trim();
  if (a) body.ANTHROPIC_API_KEY = a;
  if (o) body.OPENAI_API_KEY = o;
  const msg = $('#settingsMsg');
  msg.className = 'msg'; msg.textContent = 'Saving and checking your key…';
  try {
    await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    $('#anthKey').value = ''; $('#oaiKey').value = '';
    const t = await (await fetch('/api/settings/test', { method: 'POST' })).json();
    msg.className = 'msg ' + (t.ok ? 'ok' : 'bad');
    msg.textContent = t.ok ? '✓ ' + t.message + ' You can close this window and start writing.' : t.message;
    await loadConfig2();
    openSettingsHints();
  } catch (e) { msg.className = 'msg bad'; msg.textContent = 'Could not save: ' + e.message; }
});
$('#clearKeys').addEventListener('click', async () => {
  if (!confirm('Remove the saved API keys from this computer?')) return;
  await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ANTHROPIC_API_KEY: '', OPENAI_API_KEY: '' }) });
  openSettingsHints(); loadConfig2();
});
async function openSettingsHints() {
  const s = await (await fetch('/api/settings')).json();
  $('#anthHint').textContent = s.ANTHROPIC_API_KEY ? `saved ${s.ANTHROPIC_API_KEY_hint}` : '';
  $('#oaiHint').textContent = s.OPENAI_API_KEY ? `saved ${s.OPENAI_API_KEY_hint}` : '';
}
async function loadConfig2() {   // refresh the status pills without re-opening the dialog
  state.config = await (await fetch('/api/config')).json();
  const c = state.config;
  $('#status').innerHTML =
    `<span class="pill ${c.has_api_key ? 'ok' : 'bad'}">${c.has_api_key ? 'Claude ready · ' + esc(c.model) : 'Add your API key in Settings'}</span>` +
    `<span class="pill ${c.transcription ? 'ok' : ''}">${c.transcription ? 'Transcription: ' + (c.transcription === 'openai' ? 'OpenAI' : 'on this computer') : 'No transcription (text & music only)'}</span>`;
}

// ------------------------------------------------------------------ mode, chips, files
const HINTS = {
  story: ['Upload a story (Word, PDF, text, HTML), a narration or podcast recording, or a link. The script follows its events and pacing.',
          'Narration, audiobook or podcast (transcribed and paced to the recording)'],
  music: ['Upload the song. Add lyrics, a script or a concept (document, link or pasted text) to steer the story. Cuts land on beats and the visuals follow the song\'s sections and energy.',
          'The song or track for the video (required)'],
};
function setMode(mode) {
  state.mode = mode;
  $$('.mode-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#modeHint').textContent = HINTS[mode][0];
  $('#audioSub').textContent = HINTS[mode][1];
  $('#audioDrop').classList.toggle('required', mode === 'music');
  $('#docLabel').textContent = mode === 'music' ? 'Drop lyrics, script or concept (optional)' : 'Drop documents here';
  $('#targetRow').classList.toggle('hidden', mode === 'music');
}
$$('.mode-btn').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

function setInc(inc) {
  state.inc = inc;
  $$('#incChips button').forEach((b) => b.classList.toggle('active', b.dataset.inc === inc));
  $('#incHint').textContent = inc === 'full'
    ? 'A complete script: scenes of natural length with every shot timed, like a shooting script.'
    : `The script is cut into ${inc}-second segments, each ready to generate as ${+inc <= 10 ? 'one clip' : 'a short sequence of clips'}.`;
}
$$('#incChips button').forEach((b) => b.addEventListener('click', () => setInc(b.dataset.inc)));

const AUDIO_RE = /\.(mp3|wav|m4a|aac|flac|ogg|oga|opus|webm|mp4|aiff?)$/i;
function addFiles(list, toAudio) {
  for (const f of list) {
    if (toAudio || AUDIO_RE.test(f.name) || (f.type || '').startsWith('audio/')) state.audio = f;
    else state.docs.push(f);
  }
  renderFiles();
}
function renderFiles() {
  $('#docList').innerHTML = state.docs.map((f, i) => `<li>${esc(f.name)} <button data-i="${i}" title="Remove">✕</button></li>`).join('');
  $('#audioList').innerHTML = state.audio ? `<li>♪ ${esc(state.audio.name)} <button title="Remove">✕</button></li>` : '';
  $$('#docList button').forEach((b) => b.onclick = () => { state.docs.splice(+b.dataset.i, 1); renderFiles(); });
  $$('#audioList button').forEach((b) => b.onclick = () => { state.audio = null; renderFiles(); });
}
function wireDrop(zone, input, toAudio) {
  input.addEventListener('change', () => { addFiles(input.files, toAudio); input.value = ''; });
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('over'); addFiles(e.dataTransfer.files, toAudio); });
}
wireDrop($('#docDrop'), $('#docInput'), false);
wireDrop($('#audioDrop'), $('#audioInput'), true);

function parseDuration(s) {
  s = (s || '').trim();
  if (!s) return null;
  if (/^\d+(\.\d+)?$/.test(s)) return parseFloat(s) * 60;           // plain number = minutes
  const p = s.split(':').map(Number);
  if (p.some(isNaN)) return null;
  return p.reduce((acc, v) => acc * 60 + v, 0);
}

// ------------------------------------------------------------------ submit
$('#goBtn').addEventListener('click', async () => {
  $('#formError').textContent = '';
  const url = $('#urlInput').value.trim(), text = $('#textInput').value.trim();
  if (state.mode === 'music' && !state.audio && !(url && AUDIO_RE.test(url))) {
    $('#formError').textContent = 'Music video mode needs the song: drop an audio file (or link straight to one).'; return;
  }
  if (!state.docs.length && !state.audio && !url && !text) {
    $('#formError').textContent = 'Add a document, audio file, link or pasted text first.'; return;
  }
  const settings = {
    mode: state.mode, increment: state.inc, target_seconds: state.mode === 'story' ? parseDuration($('#targetInput').value) : null,
    visual_style: $('#styleSel').value, tone: $('#toneInput').value.trim(), aspect_ratio: $('#aspectSel').value,
    target_generator: $('#genSel').value, include_dialogue: $('#dialogueChk').checked, direction: $('#directionInput').value.trim(),
  };
  const fd = new FormData();
  fd.append('settings', JSON.stringify(settings));
  fd.append('url', url); fd.append('text', text);
  state.docs.forEach((f) => fd.append('files', f));
  if (state.audio) fd.append('audio', state.audio);
  $('#goBtn').disabled = true; $('#goBtn').textContent = 'Uploading…';
  try {
    const r = await fetch('/api/jobs', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload failed');
    openJob(d.id);
  } catch (e) {
    $('#formError').textContent = e.message;
  } finally {
    $('#goBtn').disabled = false; $('#goBtn').textContent = 'Write the script';
  }
});

// ------------------------------------------------------------------ job polling & rendering
function openJob(id) {
  state.jobId = id; state.job = null; state.rendered = new Map();
  location.hash = 'job=' + id;
  $('#emptyState').classList.add('hidden');
  $('#tab-script').innerHTML = ''; $('#tab-bible').innerHTML = '';
  $('#scriptBox').classList.add('hidden');
  $('#progressBox').classList.remove('hidden', 'error');
  clearTimeout(state.timer);
  poll();
}

async function poll() {
  const id = state.jobId;
  let job;
  try {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) throw new Error((await r.json()).detail);
    job = await r.json();
  } catch (e) {
    $('#stage').textContent = e.message; $('#progressBox').classList.add('error'); return;
  }
  if (id !== state.jobId) return;
  state.job = job;
  render(job);
  if (job.status === 'queued' || job.status === 'running') state.timer = setTimeout(poll, 1500);
  else loadRecent();
}

function render(job) {
  const pb = $('#progressBox');
  $('#progressFill').style.width = `${Math.round((job.progress || 0) * 100)}%`;
  $('#stage').textContent = job.status === 'error' ? 'Stopped: ' + job.error : job.stage;
  pb.classList.toggle('error', job.status === 'error');
  pb.classList.toggle('hidden', job.status === 'done');
  $('#resumeBtn').classList.toggle('hidden', !(job.status === 'error' && job.source));
  const res = job.result;
  if (!res) return;
  $('#scriptBox').classList.remove('hidden');
  const b = res.bible, plan = res.plan;
  $('#sTitle').textContent = b.title;
  $('#sLogline').textContent = b.logline;
  const inc = res.settings.increment === 'full' ? 'Complete script' : `${res.settings.increment}-second segments`;
  const shots = res.segments.reduce((n, s) => n + s.shots.length, 0);
  $('#sMeta').textContent = `${inc} · ${fmt(plan.total_seconds)} running time · ${plan.segments.length} segments · ${shots} shots · ${b.genre} · ${res.settings.aspect_ratio}`;
  $('#sNotes').innerHTML = (res.notes || []).map((n) => `<div>${esc(n)}</div>`).join('');
  renderTimeline(job);
  if (!$('#tab-bible').dataset.done) renderBible(res);
  res.segments.forEach((s) => {
    const key = JSON.stringify(s);
    if (state.rendered.get(s.index) !== key) { renderSegment(s, plan.segments[s.index]); state.rendered.set(s.index, key); }
  });
}

const intensityColor = (v) => ['#3a5f8a', '#3a7f8a', '#7f8a3a', '#b88a2a', '#c0603a'][Math.min(4, Math.max(0, Math.round(v) - 1))];
const sectionColor = (label) => ['#d6a440', '#4f8cc9', '#5fae7c', '#c0708a', '#9a7fd0', '#d08a50'][(label.charCodeAt(0) - 65) % 6];

function renderTimeline(job) {
  const res = job.result, total = res.plan.total_seconds || 1, audio = job.source && job.source.audio;
  const cv = $('#energyCanvas');
  cv.classList.toggle('hidden', !audio);
  $('#sectionsBar').classList.toggle('hidden', !audio);
  if (audio) {
    const w = cv.clientWidth || 800; cv.width = w;
    const ctx = cv.getContext('2d'), e = audio.energy_curve, h = cv.height;
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#3a3f4b';
    e.forEach((v, i) => { const x = (i / total) * w; ctx.fillRect(x, h - v * h, Math.max(1, w / total - 0.5), v * h); });
    ctx.fillStyle = '#d6a44088';
    audio.downbeats.forEach((t) => ctx.fillRect((t / total) * w, 0, 1, 4));
    $('#sectionsBar').innerHTML = audio.sections.map((s) =>
      `<div style="left:${(s.start / total) * 100}%;width:${((s.end - s.start) / total) * 100}%;background:${sectionColor(s.label)}" title="${esc(s.role)} ${esc(s.label)}">${esc(s.role)}</div>`).join('');
  }
  const done = new Set(res.segments.map((s) => s.index));
  $('#segBar').innerHTML = res.plan.segments.map((p) =>
    `<div data-i="${p.index}" class="${done.has(p.index) ? '' : 'pending'}" style="left:${(p.start / total) * 100}%;width:${((p.end - p.start) / total) * 100}%;background:${intensityColor(p.intensity)}" title="Segment ${p.index + 1} · ${fmt(p.start)}–${fmt(p.end)} · intensity ${p.intensity}"></div>`).join('');
  $$('#segBar div').forEach((d) => d.onclick = () => { const el = $(`#seg-${d.dataset.i}`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); });
}

function shotHTML(sh, segIdx, k) {
  const dlg = (sh.dialogue || []).map((d) => `<div class="dlg"><b data-f="character">${esc(d.character.toUpperCase())}</b>${d.delivery ? `<i data-f="delivery">(${esc(d.delivery)})</i>` : ''}<span data-f="line">${esc(d.line)}</span></div>`).join('');
  const extra = [['V.O.', 'voiceover'], ['Sound', 'sound'], ['VFX', 'vfx'], ['Negative', 'negative']]
    .filter(([, f]) => sh[f]).map(([l, f]) => `<div class="extra"><b>${l}:</b> <span data-f="${f}">${esc(sh[f])}</span></div>`).join('');
  return `<div class="shot" data-k="${k}">
    <div class="shead"><span class="cam" data-f="camera">${esc(sh.camera)}</span><span>${segIdx + 1}.${k + 1} · ${fmt(sh.start)}–${fmt(sh.end)} · ${(sh.end - sh.start).toFixed(1)} s</span>
      <button class="ghost copy" title="Copy the video prompt">Copy prompt</button></div>
    <div class="prompt" data-f="prompt">${esc(sh.prompt)}</div>
    <div class="action" data-f="action">${esc(sh.action)}</div>
    ${dlg}${sh.lyrics ? `<div class="lyr">♪ <span data-f="lyrics">${esc(sh.lyrics)}</span> ♪</div>` : ''}${extra}
  </div>`;
}

function renderSegment(s, p) {
  let el = $(`#seg-${s.index}`);
  if (!el) {
    el = $('#segTpl').content.firstElementChild.cloneNode(true);
    el.id = `seg-${s.index}`;
    const after = [...$('#tab-script').children].find((c) => +c.id.slice(4) > s.index);
    $('#tab-script').insertBefore(el, after || null);
    wireSegment(el, s.index);
  }
  el.classList.remove('working');
  $('.num', el).textContent = s.index + 1;
  $('.time', el).textContent = `${fmt(p.start)} – ${fmt(p.end)}`;
  $('.heading', el).textContent = s.scene_heading;
  $('.intensity', el).textContent = (p.section ? p.section.split(' (')[0] + ' · ' : '') + `intensity ${p.intensity}`;
  $('.summary', el).textContent = s.summary;
  $('.shots', el).innerHTML = s.shots.map((sh, k) => shotHTML(sh, s.index, k)).join('');
  $('.transition', el).textContent = s.transition_out ? `${s.transition_out} →` : '';
  $$('.copy', el).forEach((b) => b.onclick = async () => {
    const txt = $('.prompt', b.closest('.shot')).textContent;
    try { await navigator.clipboard.writeText(txt); b.textContent = 'Copied'; setTimeout(() => (b.textContent = 'Copy prompt'), 1200); } catch { /* ignore */ }
  });
}

function wireSegment(el, index) {
  $('.rewrite', el).onclick = () => $('.rewrite-box', el).classList.toggle('hidden');
  $('.rewrite-box .go', el).onclick = async () => {
    const note = $('.rewrite-box textarea', el).value;
    el.classList.add('working');
    try {
      const r = await fetch(`/api/jobs/${state.jobId}/segments/${index}/rewrite`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ note }) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      state.job.result.segments[index] = d;
      renderSegment(d, state.job.result.plan.segments[index]);
      state.rendered.set(index, JSON.stringify(d));
      $('.rewrite-box', el).classList.add('hidden');
    } catch (e) { alert('Rewrite failed: ' + e.message); el.classList.remove('working'); }
  };
  $('.edit', el).onclick = async () => {
    const btn = $('.edit', el), editing = btn.textContent === 'Save';
    if (!editing) {
      $$('[data-f]', el).forEach((n) => n.contentEditable = 'true');
      $$('.heading, .summary, .transition', el).forEach((n) => n.contentEditable = 'true');
      btn.textContent = 'Save'; return;
    }
    const seg = structuredClone(state.job.result.segments[index]);
    seg.scene_heading = $('.heading', el).textContent.trim();
    seg.summary = $('.summary', el).textContent.trim();
    seg.transition_out = $('.transition', el).textContent.replace(/→\s*$/, '').trim();
    $$('.shot', el).forEach((sn) => {
      const sh = seg.shots[+sn.dataset.k];
      ['camera', 'prompt', 'action', 'lyrics', 'voiceover', 'sound', 'vfx', 'negative'].forEach((f) => {
        const n = $(`:scope > [data-f="${f}"], :scope .shead [data-f="${f}"], :scope .lyr [data-f="${f}"], :scope .extra [data-f="${f}"]`, sn);
        if (n) sh[f] = n.textContent.trim();
      });
      $$('.dlg', sn).forEach((dn, i) => {
        const d = sh.dialogue[i]; if (!d) return;
        d.character = $('[data-f="character"]', dn).textContent.trim();
        d.line = $('[data-f="line"]', dn).textContent.trim();
        const del = $('[data-f="delivery"]', dn); if (del) d.delivery = del.textContent.replace(/^\(|\)$/g, '').trim();
      });
    });
    try {
      const r = await fetch(`/api/jobs/${state.jobId}/segments/${index}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(seg) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      state.job.result.segments[index] = d;
      $$('[contenteditable]', el).forEach((n) => n.removeAttribute('contenteditable'));
      btn.textContent = 'Edit';
      renderSegment(d, state.job.result.plan.segments[index]);
    } catch (e) { alert('Save failed: ' + e.message); }
  };
}

function renderBible(res) {
  const b = res.bible;
  const beats = b.beats.map((bt, i) => {
    const w = res.plan.beat_times[i] || [0, 0];
    return `<tr><td>${i + 1}</td><td>${fmt(w[0])}–${fmt(w[1])}</td><td>${esc(bt.summary)}</td><td>${esc(bt.emotion)}</td><td>${bt.intensity}</td><td>${esc(bt.location)}</td></tr>`;
  }).join('');
  $('#tab-bible').innerHTML = `<div class="bible">
    <h4>Style</h4><div class="item"><p><b>Visual style:</b> ${esc(b.visual_style)}</p><p><b>Tone:</b> ${esc(b.tone)}</p>
      <p><b>Pacing:</b> ${esc(b.pacing_notes)}</p><p><b>World rules & motifs:</b> ${esc(b.world_rules)}</p></div>
    <h4>Characters</h4>${b.characters.map((c) => `<div class="item"><b>${esc(c.name)}</b> · ${esc(c.role)}<p>${esc(c.visual_signature)}</p><p>Voice: ${esc(c.voice)}</p></div>`).join('')}
    <h4>Locations</h4>${b.locations.map((l) => `<div class="item"><b>${esc(l.name)}</b><p>${esc(l.visual_signature)}</p></div>`).join('')}
    <h4>Beats & pacing</h4><table><tr><th>#</th><th>Time</th><th>Beat</th><th>Emotion</th><th>Int.</th><th>Location</th></tr>${beats}</table></div>`;
  $('#tab-bible').dataset.done = '1';
}

$$('.tab').forEach((t) => t.addEventListener('click', () => {
  $$('.tab').forEach((x) => x.classList.toggle('active', x === t));
  $('#tab-script').classList.toggle('hidden', t.dataset.tab !== 'script');
  $('#tab-bible').classList.toggle('hidden', t.dataset.tab !== 'bible');
}));

$$('.exports a').forEach((a) => a.addEventListener('click', () => {
  if (state.jobId) window.location = `/api/jobs/${state.jobId}/export/${a.dataset.fmt}`;
}));

$('#resumeBtn').addEventListener('click', async () => {
  const r = await fetch(`/api/jobs/${state.jobId}/resume`, { method: 'POST' });
  if (r.ok) { $('#progressBox').classList.remove('error'); poll(); } else alert((await r.json()).detail);
});

async function loadRecent() {
  try {
    const jobs = await (await fetch('/api/jobs')).json();
    $('#recentList').innerHTML = jobs.map((j) => `<li data-id="${j.id}">${esc(j.title)}<span>${j.mode === 'music' ? '♪ ' : ''}${j.increment === 'full' ? 'full' : j.increment + 's'} · ${esc(j.status)}</span></li>`).join('') || '<li><span>None yet</span></li>';
    $$('#recentList li[data-id]').forEach((li) => li.onclick = () => openJob(li.dataset.id));
  } catch { /* ignore */ }
}

window.addEventListener('resize', () => { if (state.job && state.job.result) renderTimeline(state.job); });
setMode('story'); setInc('10'); loadConfig(); loadRecent();
const m = location.hash.match(/job=([a-f0-9]+)/);
if (m) openJob(m[1]);
