/* The studio front-end.
 *
 * One rule that shapes everything here: the form is generated from the server's
 * field list (which is the terminal UI's field list), so a new CLI flag appears
 * here automatically instead of being re-typed into a second UI that then drifts. */

(function () {
  var $ = function (id) { return document.getElementById(id); };

  var state = {
    mode: 'dock',
    effort: 'standard',
    fields: [],
    fieldByKey: {},
    stages: [],
    examples: [],
    values: {},
    job: null,
    poll: null,
    since: 0,
    lastPoses: 0,
    validateTimer: null
  };

  // Fields with hand-built controls in the main form; everything else is
  // rendered into the Advanced drawer so nothing is unreachable.
  var MAIN_KEYS = ['peptide', 'receptor', 'site', 'box', 'blind', 'mode', 'peptide_pdb',
                   'offtarget_receptor', 'offtarget_site', 'offtarget_box'];

  var EFFORTS = {
    quick:    { n_samples: '20',  refine_topk: '0', ultra: '0' },
    standard: { n_samples: '100', refine_topk: '0', ultra: '0' },
    ultra:    { n_samples: '100', refine_topk: '5', ultra: '32' }
  };

  var RUN_LABEL = { dock: 'Run docking', selectivity: 'Compare the two', crystal: 'Score this pose' };

  // ---------------------------------------------------------------- utils

  function api(path, body) {
    var opts = body ? {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    } : {};
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok) throw new Error((data.error && data.error.message) || ('HTTP ' + r.status));
        return data;
      });
    });
  }

  var toastTimer = null;
  function toast(msg) {
    var el = $('toast');
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.hidden = true; }, 4200);
  }

  function clock(sec) {
    var m = Math.floor(sec / 60), s = sec % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function humanTime(sec) {
    if (sec < 90) return 'about ' + Math.max(5, Math.round(sec / 5) * 5) + ' seconds';
    if (sec < 3600) return 'about ' + Math.round(sec / 60) + ' minutes';
    return 'about ' + (sec / 3600).toFixed(1) + ' hours';
  }

  // ---------------------------------------------------------------- boot

  function boot() {
    state.cloud = new window.Cloud($('cloud'));

    api('/api/fields').then(function (d) {
      state.fields = d.fields;
      state.stages = d.stages;
      d.fields.forEach(function (f) {
        state.fieldByKey[f.key] = f;
        if (state.values[f.key] === undefined) state.values[f.key] = f.default;
      });
      buildAdvanced();
      buildPips();
      applyEffort('standard');
      syncForm();
      validate();
    }).catch(function (e) { toast('Could not load settings: ' + e.message); });

    api('/api/examples').then(function (d) {
      state.examples = d.examples;
      renderExamples();
      if (d.examples.length) pickExample(d.examples[0], true);
    });

    refreshEnv();
    wire();
  }

  function refreshEnv() {
    api('/api/env').then(function (env) {
      var lamp = $('lamp');
      var broken = Object.keys(env.checks).filter(function (k) { return !env.checks[k].ok; });
      lamp.classList.toggle('is-ready', env.ready);
      lamp.classList.toggle('is-trouble', !env.ready);
      lamp.querySelector('.lamp-text').textContent =
        env.ready ? (broken.length ? 'ready · ' + broken.length + ' optional missing' : 'ready')
                  : 'setup needed';
      state.env = env;
    }).catch(function () {
      $('lamp').querySelector('.lamp-text').textContent = 'server unreachable';
    });
  }

  function envReport() {
    if (!state.env) return;
    var lines = Object.keys(state.env.checks).map(function (k) {
      var c = state.env.checks[k];
      return (c.ok ? '✓ ' : '✗ ') + c.detail + (c.ok || !c.fix ? '' : '  →  ' + c.fix);
    });
    showLayer('error');
    $('errtitle').textContent = 'What this machine can do';
    $('errmsg').textContent = state.env.ready
      ? 'Everything essential is here. Anything marked ✗ below only limits optional features.'
      : 'Something essential is missing — the runs below will fail until it is fixed.';
    $('errlog').textContent = lines.join('\n');
  }

  // ---------------------------------------------------------------- form

  function renderExamples() {
    ['examples', 'examples-off'].forEach(function (host) {
      var box = $(host);
      box.innerHTML = '';
      state.examples.forEach(function (ex) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'ex';
        b.dataset.id = ex.id;
        b.innerHTML = '<span class="ex-name"></span><span class="ex-blurb"></span>';
        b.querySelector('.ex-name').textContent = ex.name;
        b.querySelector('.ex-blurb').textContent = ex.blurb;
        b.addEventListener('click', function () {
          if (host === 'examples') pickExample(ex);
          else pickOffTarget(ex);
        });
        box.appendChild(b);
      });
    });
  }

  function pickExample(ex, keepPeptide) {
    state.values.receptor = ex.receptor_path;
    state.values.site = ex.site.join(' ');
    state.values.box = String(ex.box);
    if (!keepPeptide || !state.values.peptide) state.values.peptide = ex.peptide;
    markActive('examples', ex.id);
    syncForm();
    validate();
  }

  function pickOffTarget(ex) {
    state.values.offtarget_receptor = ex.receptor_path;
    state.values.offtarget_site = ex.site.join(' ');
    state.values.offtarget_box = String(ex.box);
    markActive('examples-off', ex.id);
    syncForm();
    validate();
  }

  function markActive(host, id) {
    Array.prototype.forEach.call($(host).children, function (el) {
      el.classList.toggle('is-active', el.dataset.id === id);
    });
  }

  function buildAdvanced() {
    var grid = $('advgrid');
    grid.innerHTML = '';
    var shown = 0;
    state.fields.forEach(function (f) {
      if (MAIN_KEYS.indexOf(f.key) !== -1) return;
      shown++;
      var wrap = document.createElement('div');
      wrap.className = 'adv-field';
      wrap.dataset.key = f.key;

      var id = 'adv-' + f.key;
      var label = document.createElement('label');
      label.setAttribute('for', id);
      label.textContent = f.label;

      var input;
      if (f.kind === 'toggle') {
        wrap.classList.add('adv-toggle-wrap');
        input = document.createElement('input');
        input.type = 'checkbox';
        input.id = id;
        var line = document.createElement('div');
        line.className = 'adv-toggle';
        line.appendChild(input);
        line.appendChild(label);
        wrap.appendChild(line);
      } else {
        input = document.createElement('input');
        input.type = 'text';
        input.id = id;
        if (f.kind === 'number') input.inputMode = 'numeric';
        if (f.kind === 'decimal') input.inputMode = 'decimal';
        wrap.appendChild(label);
        wrap.appendChild(input);
      }

      input.dataset.key = f.key;
      input.addEventListener('input', onAdvancedInput);
      input.addEventListener('change', onAdvancedInput);

      var help = document.createElement('p');
      help.className = 'adv-help';
      help.textContent = f.help;
      wrap.appendChild(help);
      grid.appendChild(wrap);
    });
    $('advcount').textContent = '(' + shown + ' settings)';
  }

  function onAdvancedInput(e) {
    var key = e.target.dataset.key;
    state.values[key] = e.target.type === 'checkbox' ? (e.target.checked ? 'y' : 'n') : e.target.value;
    if (['n_samples', 'refine_topk', 'ultra'].indexOf(key) !== -1) markEffortCustom();
    markChanged();
    validate();
  }

  function markChanged() {
    Array.prototype.forEach.call($('advgrid').children, function (wrap) {
      var f = state.fieldByKey[wrap.dataset.key];
      if (!f) return;
      wrap.classList.toggle('adv-changed', String(state.values[f.key] || '') !== String(f.default || ''));
    });
  }

  function markEffortCustom() {
    var match = null;
    Object.keys(EFFORTS).forEach(function (name) {
      var preset = EFFORTS[name];
      var same = Object.keys(preset).every(function (k) {
        return String(state.values[k] || '') === preset[k];
      });
      if (same) match = name;
    });
    state.effort = match || 'custom';
    Array.prototype.forEach.call($('efforts').children, function (b) {
      b.classList.toggle('is-active', b.dataset.effort === state.effort);
    });
  }

  function applyEffort(name) {
    var preset = EFFORTS[name];
    if (!preset) return;
    Object.keys(preset).forEach(function (k) { state.values[k] = preset[k]; });
    state.effort = name;
    Array.prototype.forEach.call($('efforts').children, function (b) {
      b.classList.toggle('is-active', b.dataset.effort === name);
    });
    syncForm();
    validate();
  }

  // Push state.values into every control (main + advanced).
  function syncForm() {
    MAIN_KEYS.forEach(function (k) {
      var el = $(k);
      if (el && document.activeElement !== el) el.value = state.values[k] || '';
    });
    Array.prototype.forEach.call($('advgrid').querySelectorAll('input'), function (el) {
      var v = state.values[el.dataset.key] || '';
      if (document.activeElement === el) return;
      if (el.type === 'checkbox') el.checked = /^(y|yes|true|1)$/i.test(v);
      else el.value = v;
    });
    var blind = /^(y|yes)$/i.test(state.values.blind || '');
    $('siterow').hidden = blind;
    var radio = document.querySelector('input[name="where"][value="' + (blind ? 'blind' : 'site') + '"]');
    if (radio) radio.checked = true;
    markChanged();
    renderPeptideChips();
  }

  function renderPeptideChips(stats) {
    var seq = (state.values.peptide || '').toUpperCase().replace(/[^A-Z]/g, '');
    var host = $('pepchips');
    host.innerHTML = '';
    if (!seq) return;
    var s = stats || { length: seq.length };
    add(s.length + ' residues');
    if (s.band) add(s.band + ' peptide');
    if (typeof s.net_charge === 'number') {
      add('net charge ' + (s.net_charge > 0 ? '+' : '') + s.net_charge, s.charged ? 'is-warn' : '');
    }
    if (s.ends_in_cys) add('ends in cysteine', 'is-warn');

    function add(text, cls) {
      var el = document.createElement('span');
      el.className = 'chip ' + (cls || '');
      el.textContent = text;
      host.appendChild(el);
    }
  }

  function setMode(mode) {
    state.mode = mode;
    state.values.mode = (mode === 'crystal') ? 'crystal' : 'ai';
    Array.prototype.forEach.call(document.querySelectorAll('.mode'), function (b) {
      b.classList.toggle('is-active', b.dataset.mode === mode);
    });
    $('step-offtarget').hidden = mode !== 'selectivity';
    $('step-where').hidden = mode === 'crystal';
    $('step-pose').hidden = mode !== 'crystal';
    $('step-effort').hidden = mode === 'crystal';
    $('advanced').hidden = mode === 'crystal';
    $('run').querySelector('.run-label').textContent = RUN_LABEL[mode];
    validate();
  }

  // ---------------------------------------------------------------- validate

  function validate() {
    clearTimeout(state.validateTimer);
    state.validateTimer = setTimeout(function () {
      api('/api/validate', { mode: state.mode, values: state.values }).then(function (res) {
        renderPeptideChips(res.peptide);
        showErrors(res.errors || {});
        applySkips(res.skipped || []);
        $('run').disabled = !res.ok || (state.job && isLive(state.job.state));
        $('runsub').textContent = res.ok
          ? humanTime(res.estimate_seconds) + ' on this machine'
          : 'fix the highlighted settings first';
        $('cmd').textContent = res.command || '';
      }).catch(function (e) { toast(e.message); });
    }, 220);
  }

  function showErrors(errors) {
    ['peptide', 'receptor', 'site', 'peptide_pdb', 'offtarget_receptor'].forEach(function (k) {
      var box = $('err-' + k);
      if (!box) return;
      var msg = errors[k] || (k === 'site' ? errors.box : null) ||
                (k === 'offtarget_receptor' ? (errors.offtarget_site || errors.offtarget_box) : null);
      box.hidden = !msg;
      box.textContent = msg || '';
    });
    Array.prototype.forEach.call($('advgrid').children, function (wrap) {
      var msg = errors[wrap.dataset.key];
      var existing = wrap.querySelector('.field-error');
      if (msg && !existing) {
        var p = document.createElement('p');
        p.className = 'field-error';
        p.textContent = msg;
        wrap.appendChild(p);
      } else if (msg) {
        existing.textContent = msg;
      } else if (existing) {
        existing.remove();
      }
    });
  }

  // Grey out advanced settings that do nothing given the current choices —
  // the server decides, using the same rules the terminal UI uses.
  function applySkips(skipped) {
    Array.prototype.forEach.call($('advgrid').children, function (wrap) {
      var off = skipped.indexOf(wrap.dataset.key) !== -1;
      wrap.style.opacity = off ? '0.4' : '';
      Array.prototype.forEach.call(wrap.querySelectorAll('input'), function (i) { i.disabled = off; });
    });
  }

  // ---------------------------------------------------------------- run

  function buildPips() {
    var host = $('pips');
    host.innerHTML = '';
    state.stages.forEach(function (s) {
      var li = document.createElement('li');
      li.className = 'pip';
      li.dataset.stage = s.key;
      li.innerHTML = '<span class="pip-dot"></span>';
      li.appendChild(document.createTextNode(s.label));
      host.appendChild(li);
    });
  }

  function isLive(s) { return s === 'queued' || s === 'running'; }

  function startRun() {
    api('/api/run', { mode: state.mode, values: state.values }).then(function (d) {
      state.job = d.job;
      state.since = 0;
      state.lastPoses = 0;
      $('console').textContent = '';
      $('stop').hidden = false;
      $('run').disabled = true;
      state.cloud.setRunning();
      showLayer('run');
      poll();
    }).catch(function (e) {
      toast(e.message);
    });
  }

  function poll() {
    clearTimeout(state.poll);
    if (!state.job) return;
    fetch('/api/jobs/' + state.job.id + '?since=' + state.since)
      .then(function (r) { return r.json(); })
      .then(function (job) {
        state.job = job;
        state.since = job.line_count;
        if (job.lines && job.lines.length) {
          var con = $('console');
          con.textContent += (con.textContent ? '\n' : '') + job.lines.join('\n');
          con.scrollTop = con.scrollHeight;
          // one dot per pose the pipeline has finished, read off its own counter
          var m = /(\d+)\s*\/\s*(\d+)/.exec(job.counter || '');
          if (m) {
            var done = parseInt(m[1], 10);
            for (var i = state.lastPoses; i < done && i < state.lastPoses + 40; i++) {
              state.cloud.addPose();
            }
            state.lastPoses = done;
          }
        }
        updateProgress(job);
        if (isLive(job.state)) {
          state.poll = setTimeout(poll, 900);
        } else {
          finishRun(job);
        }
      })
      .catch(function () { state.poll = setTimeout(poll, 2000); });
  }

  function updateProgress(job) {
    $('barfill').style.width = Math.round((job.fraction || 0) * 100) + '%';
    $('runstage').textContent = job.stage_label + (job.counter ? '  ' + job.counter : '');
    $('runclock').textContent = clock(job.elapsed || 0);
    var order = state.stages.map(function (s) { return s.key; });
    var at = order.indexOf(job.stage);
    Array.prototype.forEach.call($('pips').children, function (li, i) {
      li.classList.toggle('is-now', li.dataset.stage === job.stage);
      li.classList.toggle('is-done', at > -1 && i < at);
    });
  }

  function finishRun(job) {
    $('stop').hidden = true;
    $('run').disabled = false;
    if (job.state === 'done') {
      loadResults(job);
    } else {
      showLayer('error');
      $('errtitle').textContent = job.state === 'cancelled' ? 'Run stopped' : 'That run did not finish';
      $('errmsg').textContent = job.error || 'No error message was reported.';
      $('errlog').textContent = ($('console').textContent || '').split('\n').slice(-40).join('\n');
      state.cloud.reset();
    }
  }

  function loadResults(job) {
    api('/api/jobs/' + job.id + '/results').then(function (res) {
      showLayer('result');
      if (res.cloud && res.cloud.length) state.cloud.settle(res.cloud);
      renderHeadline(res, job);
      renderTable(res);
      renderDownloads(res, job);
    }).catch(function (e) { toast('Run finished but results would not load: ' + e.message); });
  }

  function renderHeadline(res, job) {
    var h = res.headline;
    var sel = res.selectivity;
    var chips = $('reschips');
    chips.innerHTML = '';
    $('caveat').textContent = '';

    if (sel) {
      $('reslabel').textContent = 'Selectivity (on-target minus off-target)';
      countTo($('dgnum'), sel.ddg, true);
      $('strength').textContent = sel.verdict === 'on-target selective'
        ? 'Prefers your target — the direction you want.'
        : 'Prefers the off-target. Not selective.';
      $('caveat').textContent = sel.caveat;
    } else if (h) {
      $('reslabel').textContent = 'Predicted binding energy';
      countTo($('dgnum'), h.delta_g, false);
      $('strength').textContent = strengthWords(h.delta_g, h.kd);
      if (h.charged_confidence && h.charged_confidence.toLowerCase() === 'low') {
        $('caveat').textContent = 'This peptide is charge-driven, and charged binding is the one ' +
          'thing a single-pose scorer cannot resolve. Treat the number as a ranking, not a measurement.';
      }
      addChip(chips, 'from <b>' + (h.n_poses || 0) + '</b> poses');
      if (h.n_clusters) addChip(chips, '<b>' + h.n_clusters + '</b> binding modes found');
      if (h.pose_file) addChip(chips, 'best pose <b>' + h.pose_file + '</b>');
      if (h.charged_confidence) {
        addChip(chips, 'charge confidence <b>' + h.charged_confidence + '</b>',
                h.charged_confidence.toLowerCase() === 'low' ? 'is-warn' : 'is-good');
      }
    } else {
      $('reslabel').textContent = 'Run finished';
      $('dgnum').textContent = '—';
      $('strength').textContent = 'No score was written. The log above has the detail.';
    }
    addChip(chips, 'took <b>' + clock(job.elapsed || 0) + '</b>');
  }

  function addChip(host, html, cls) {
    var el = document.createElement('span');
    el.className = 'reschip ' + (cls || '');
    el.innerHTML = html;
    host.appendChild(el);
  }

  function strengthWords(dg, kd) {
    if (typeof dg !== 'number') return '';
    var word = dg <= -11 ? 'Very strong' : dg <= -9 ? 'Strong'
             : dg <= -7 ? 'Moderate' : dg <= -5 ? 'Weak' : 'Barely binding';
    return word + (kd ? ' — roughly ' + kd + '. More negative is tighter.' : '');
  }

  function countTo(el, value, showSign) {
    if (typeof value !== 'number') { el.textContent = '—'; return; }
    var text = function (v) {
      return (showSign && v > 0 ? '+' : '') + v.toFixed(2);
    };
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.textContent = text(value);
      return;
    }
    var start = null, dur = 750;
    function step(ts) {
      if (start === null) start = ts;
      var t = Math.min(1, (ts - start) / dur);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = text(value * eased);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function renderTable(res) {
    var host = $('restable');
    host.innerHTML = '';
    if (!res.poses || !res.poses.length) return;
    var cols = [
      ['rank', 'rank'],
      ['pooled_affinity_dg', 'ΔG'],
      ['cluster_id', 'mode'],
      ['n_contact_residues', 'contacts'],
      ['vina_score', 'vina'],
      ['pose_filename', 'file']
    ].filter(function (c) { return res.poses[0][c[0]] !== undefined; });

    var table = document.createElement('table');
    var thead = document.createElement('thead');
    var tr = document.createElement('tr');
    cols.forEach(function (c) {
      var th = document.createElement('th');
      th.textContent = c[1];
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    res.poses.slice(0, 12).forEach(function (row, i) {
      var r = document.createElement('tr');
      if (i === 0) r.className = 'is-best';
      cols.forEach(function (c) {
        var td = document.createElement('td');
        td.textContent = row[c[0]] || '';
        r.appendChild(td);
      });
      tbody.appendChild(r);
    });
    table.appendChild(tbody);
    host.appendChild(table);
  }

  function renderDownloads(res, job) {
    var host = $('downloads');
    host.innerHTML = '';
    // plots are inserted as siblings above the download row, so clear the ones
    // a previous run left behind or they stack up
    Array.prototype.forEach.call(document.querySelectorAll('.shot'), function (img) {
      img.remove();
    });
    (res.files || []).forEach(function (name) {
      var href = '/api/jobs/' + job.id + '/file?name=' + encodeURIComponent(name);
      if (/\.png$/i.test(name)) {
        var img = document.createElement('img');
        img.className = 'shot';
        img.src = href;
        img.alt = name.replace(/_/g, ' ').replace(/\.png$/, '');
        host.parentNode.insertBefore(img, host);
        return;
      }
      var a = document.createElement('a');
      a.className = 'dl';
      a.href = href + '&download=1';
      a.textContent = 'Download ' + name;
      host.appendChild(a);
    });
    var where = document.createElement('span');
    where.className = 'reschip';
    where.textContent = 'Saved in ' + res.output_dir;
    host.appendChild(where);
  }

  function showLayer(which) {
    ['idle', 'run', 'result', 'error'].forEach(function (n) {
      $('stage-' + n).hidden = n !== which;
    });
  }

  // ---------------------------------------------------------------- wiring

  function wire() {
    Array.prototype.forEach.call(document.querySelectorAll('.mode'), function (b) {
      b.addEventListener('click', function () { setMode(b.dataset.mode); });
    });

    Array.prototype.forEach.call($('efforts').children, function (b) {
      b.addEventListener('click', function () { applyEffort(b.dataset.effort); });
    });

    MAIN_KEYS.forEach(function (k) {
      var el = $(k);
      if (!el) return;
      el.addEventListener('input', function () {
        state.values[k] = k === 'peptide' ? el.value.toUpperCase() : el.value;
        if (k === 'peptide') renderPeptideChips();
        validate();
      });
    });

    Array.prototype.forEach.call(document.querySelectorAll('input[name="where"]'), function (r) {
      r.addEventListener('change', function () {
        state.values.blind = r.value === 'blind' ? 'y' : 'n';
        syncForm();
        validate();
      });
    });

    upload($('pdbfile'), $('uploadstate'), 'receptor');
    upload($('posefile'), $('posestate'), 'peptide_pdb');

    $('runform').addEventListener('submit', function (e) {
      e.preventDefault();
      startRun();
    });

    $('stop').addEventListener('click', function () {
      if (!state.job) return;
      api('/api/jobs/' + state.job.id + '/cancel', {}).then(function () { toast('Stopping the run…'); });
    });

    $('again').addEventListener('click', function () {
      showLayer('idle');
      state.cloud.reset();
    });
    $('errback').addEventListener('click', function () {
      showLayer('idle');
      state.cloud.reset();
    });
    $('lamp').addEventListener('click', envReport);
  }

  function upload(input, stateEl, targetKey) {
    if (!input) return;
    input.addEventListener('change', function () {
      var file = input.files && input.files[0];
      if (!file) return;
      stateEl.textContent = 'reading ' + file.name + '…';
      var reader = new FileReader();
      reader.onload = function () {
        api('/api/upload', { name: file.name, content: String(reader.result) }).then(function (d) {
          state.values[targetKey] = d.path;
          input.parentNode.classList.add('is-loaded');
          stateEl.textContent = d.name + ' · ' + d.atoms + ' atoms · chain ' + (d.chains.join('') || '?');
          markActive('examples', '');
          syncForm();
          validate();
        }).catch(function (e) {
          stateEl.textContent = e.message;
          input.parentNode.classList.remove('is-loaded');
        });
      };
      reader.readAsText(file);
    });
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
