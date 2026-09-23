/* The stage canvas: the pose cloud.
 *
 * Three states, and they all mean something real:
 *   idle     — a slow drift, so the stage isn't a dead rectangle
 *   running  — one dot lands per pose the pipeline reports finishing
 *   settled  — the ACTUAL Ca centroids of the scored poses, projected to 2D by
 *              the server and coloured by the cluster they ended up in
 *
 * Nothing here is faked once real data arrives; before it arrives the dots are
 * clearly just a placeholder scatter, not pretend results. */

(function (global) {
  var reduce = global.matchMedia &&
               global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Hues picked around the wiki's Carolina blue so clusters read as one family,
  // with two warm outliers for when a run genuinely finds 5+ binding modes.
  var HUES = [205, 188, 224, 168, 250, 32, 140];

  function Cloud(canvas) {
    this.canvas = canvas;
    // No 2d context (canvas switched off, or a very locked-down browser): the
    // stage just stays empty. The run itself must not care.
    this.ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
    this.dots = [];
    this.mode = 'idle';
    this.target = 0;
    this.raf = null;
    this.t = 0;
    var self = this;
    this.onResize = function () { self.resize(); };
    global.addEventListener('resize', this.onResize, { passive: true });
    this.resize();
    this.seedIdle();
    this.start();
  }

  Cloud.prototype.resize = function () {
    if (!this.ctx) return;
    var dpr = global.devicePixelRatio || 1;
    var w = this.canvas.clientWidth || 600;
    var h = this.canvas.clientHeight || 400;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
    this.draw();
  };

  Cloud.prototype.seedIdle = function () {
    this.dots = [];
    for (var i = 0; i < 46; i++) {
      this.dots.push({
        x: Math.random(), y: Math.random(),
        tx: Math.random(), ty: Math.random(),
        r: 1.6 + Math.random() * 2.2,
        hue: 205, alpha: 0.13 + Math.random() * 0.16,
        drift: 0.00016 + Math.random() * 0.00042,
        best: false
      });
    }
  };

  // A pose finished. Drop a dot somewhere plausible; position is meaningless
  // until settle() replaces it with the real projection, hence the low alpha.
  Cloud.prototype.addPose = function () {
    if (this.mode !== 'running') return;
    var a = Math.random() * Math.PI * 2;
    var rad = Math.sqrt(Math.random()) * 0.42;
    this.dots.push({
      x: 0.5 + Math.cos(a) * rad, y: 0.5 + Math.sin(a) * rad,
      tx: 0.5 + Math.cos(a) * rad, ty: 0.5 + Math.sin(a) * rad,
      r: 2.1 + Math.random() * 1.6,
      hue: 205, alpha: 0.42, drift: 0.0009, best: false, born: this.t
    });
    if (this.dots.length > 360) this.dots.shift();
  };

  Cloud.prototype.setRunning = function () {
    this.mode = 'running';
    this.dots = this.dots.slice(0, 20).map(function (d) { d.alpha = 0.1; return d; });
  };

  // Real data in: [{x, y, cluster, rank, dg}] with x/y already normalised to -1..1
  Cloud.prototype.settle = function (points) {
    this.mode = 'settled';
    var self = this;
    this.dots = points.map(function (p) {
      var hue = HUES[Math.abs(p.cluster || 0) % HUES.length];
      var best = p.rank === 1;
      return {
        x: Math.random(), y: Math.random(),
        tx: 0.5 + p.x * 0.34, ty: 0.5 + p.y * 0.34,
        r: best ? 7 : 3.1,
        hue: hue,
        alpha: best ? 1 : 0.62,
        drift: reduce ? 1 : 0.055,
        best: best,
        label: best ? p.dg : null
      };
    });
    if (reduce) {
      this.dots.forEach(function (d) { d.x = d.tx; d.y = d.ty; });
      this.draw();
    }
    if (!this.dots.length) self.seedIdle();
  };

  Cloud.prototype.reset = function () {
    this.mode = 'idle';
    this.seedIdle();
  };

  Cloud.prototype.start = function () {
    if (!this.ctx) return;
    if (reduce) { this.draw(); return; }
    var self = this;
    function frame() {
      self.t += 1;
      self.step();
      self.draw();
      self.raf = global.requestAnimationFrame(frame);
    }
    this.raf = global.requestAnimationFrame(frame);
  };

  Cloud.prototype.step = function () {
    var mode = this.mode;
    for (var i = 0; i < this.dots.length; i++) {
      var d = this.dots[i];
      if (mode === 'settled') {
        d.x += (d.tx - d.x) * d.drift;
        d.y += (d.ty - d.y) * d.drift;
      } else {
        // wander: pick a new target when close enough
        if (Math.abs(d.tx - d.x) < 0.01 && Math.abs(d.ty - d.y) < 0.01) {
          d.tx = Math.random(); d.ty = Math.random();
        }
        d.x += (d.tx - d.x) * d.drift * 60;
        d.y += (d.ty - d.y) * d.drift * 60;
      }
    }
  };

  Cloud.prototype.draw = function () {
    var ctx = this.ctx, w = this.w, h = this.h;
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    // faint guide ring so the cloud has a frame of reference at rest
    if (this.mode !== 'settled') {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.3, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(163, 203, 232, 0.07)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    for (var i = 0; i < this.dots.length; i++) {
      var d = this.dots[i];
      var px = d.x * w, py = d.y * h;
      if (d.best) {
        var glow = ctx.createRadialGradient(px, py, 0, px, py, d.r * 5);
        glow.addColorStop(0, 'hsla(' + d.hue + ', 72%, 68%, 0.42)');
        glow.addColorStop(1, 'hsla(' + d.hue + ', 72%, 68%, 0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(px, py, d.r * 5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(px, py, d.r, 0, Math.PI * 2);
      ctx.fillStyle = 'hsla(' + d.hue + ', 68%, ' + (d.best ? 78 : 62) + '%, ' + d.alpha + ')';
      ctx.fill();
    }
  };

  global.Cloud = Cloud;
})(window);
