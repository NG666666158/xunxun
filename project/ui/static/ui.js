"use strict";

/* Interaction kit shared by the main window and the quick-search window (design note:
   design/UI_REDESIGN.md). Six interactions, each a small piece:
     1. runDone        完成反馈按钮  label → spinner → drawn check / red shake
     2. Selection      勾选联动      tri-state folder ↔ file selection (pure)
     3. Deck           叠卡展开      source cards stacked as a deck, fanned into the grid
     4. SnapRail       吸附卡+画线   scroll-snap rail with a line drawn to the active card;
        connect        a line drawn from a citation to its source card
     5. Wheel          惯性滚轮      drum picker with momentum and snapping (math is pure)
     6. circleTransition 圆形扩散转场 View Transitions with a growing circle; revealFrom for one element
   Everything respects prefers-reduced-motion. The pure parts run under Node for tests. */
(function (root) {
  const doc = root.document;
  const reduced = () => !!root.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let uidSeq = 0;
  const uid = (prefix) => `${prefix}-${++uidSeq}`;

  /** Where an interaction started: the pointer, or the element's centre for keyboard clicks. */
  function pointOf(source) {
    if (source && "clientX" in source && (source.detail || source.clientX || source.clientY)) return { x: source.clientX, y: source.clientY };
    const el = source?.currentTarget || source?.target || source;
    if (el?.getBoundingClientRect) {
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    }
    return { x: (root.innerWidth || 0) / 2, y: (root.innerHeight || 0) / 2 };
  }

  // ================================================================ 1. done-feedback button

  const CHECK_SVG = '<svg class="db-svg" viewBox="0 0 24 24"><path pathLength="1" d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

  function prepareDone(btn) {
    if (btn.querySelector(":scope > .db-label")) return;
    const label = doc.createElement("span");
    label.className = "db-label";
    while (btn.firstChild) label.appendChild(btn.firstChild);
    btn.append(label);
    btn.insertAdjacentHTML("beforeend", `<span class="db-spin" aria-hidden="true"><i></i></span><span class="db-check" aria-hidden="true">${CHECK_SVG}</span>`);
    btn.dataset.done = "";
  }

  /** Run `task` behind the button: spinner while it runs, a drawn check when it resolves,
      a red shake when it throws (the error is rethrown). A second click while busy is ignored. */
  async function runDone(btn, task, { hold = 1300, failHold = 1700 } = {}) {
    if (!btn) return task();
    if (btn.dataset.state === "busy") return undefined;
    prepareDone(btn);
    clearTimeout(btn._doneTimer);
    btn.style.minWidth = `${Math.ceil(btn.getBoundingClientRect().width)}px`;
    btn.dataset.state = "busy";
    btn.setAttribute("aria-busy", "true");
    const settle = (ms) => {
      btn.removeAttribute("aria-busy");
      btn._doneTimer = setTimeout(() => { delete btn.dataset.state; btn.style.minWidth = ""; }, reduced() ? Math.min(ms, 900) : ms);
    };
    try {
      const result = await task();
      btn.dataset.state = "done";
      settle(hold);
      return result;
    } catch (err) {
      btn.dataset.state = "fail";
      settle(failHold);
      throw err;
    }
  }

  // ================================================================ 2. tri-state selection

  /** Files grouped by folder; a folder is "all", "some" or "none" selected. Pure. */
  class Selection {
    constructor(groups = []) {
      this.selected = new Set();
      this.setGroups(groups);
    }

    /** Replace the groups; selected ids that no longer exist are dropped. */
    setGroups(groups) {
      this.groups = new Map(groups.map((g) => [g.id, [...g.items]]));
      const known = new Set(groups.flatMap((g) => g.items));
      for (const id of [...this.selected]) if (!known.has(id)) this.selected.delete(id);
      return this;
    }

    has(id) { return this.selected.has(id); }

    toggle(id, on = !this.selected.has(id)) {
      if (on) this.selected.add(id);
      else this.selected.delete(id);
      return on;
    }

    groupState(gid) {
      const items = this.groups.get(gid) || [];
      let n = 0;
      for (const id of items) if (this.selected.has(id)) n++;
      return n === 0 ? "none" : n === items.length ? "all" : "some";
    }

    /** A folder that is not fully selected selects all its files; a full one clears them. */
    toggleGroup(gid, on) {
      const items = this.groups.get(gid) || [];
      if (on === undefined) on = this.groupState(gid) !== "all";
      for (const id of items) this.toggle(id, on);
      return on;
    }

    /** Every file of every group. */
    get total() {
      let n = 0;
      for (const items of this.groups.values()) n += items.length;
      return n;
    }

    allState() {
      return this.size === 0 ? "none" : this.size === this.total ? "all" : "some";
    }

    get size() { return this.selected.size; }
    ids() { return [...this.selected]; }
    clear() { this.selected.clear(); }
  }

  /** Paint a checkbox from a group state. */
  function setCheck(input, state) {
    input.checked = state === "all";
    input.indeterminate = state === "some";
    input.setAttribute("aria-checked", state === "some" ? "mixed" : String(state === "all"));
  }

  // ================================================================ 3. stacked deck

  /** Order of cards in a collapsed deck: the peeked card on top, then the rest in order. */
  function deckOrder(count, peek = -1) {
    const order = Array.from({ length: count }, (_, i) => i);
    if (peek > 0 && peek < count) {
      order.splice(peek, 1);
      order.unshift(peek);
    }
    return order;
  }

  const DECK_STEP = 7; // px each card behind sits lower
  const DECK_SHOWN = 3; // cards visible in the stack

  class Deck {
    constructor(el, { collapsed = true, onToggle } = {}) {
      this.el = el;
      this.onToggle = onToggle;
      this.cards = [...el.querySelectorAll(":scope > .source")];
      this.collapsed = collapsed && this.cards.length > 1;
      this.peek = -1;
      el.classList.add("deck");
      if (this.cards.length > 1) {
        el.insertAdjacentHTML("beforeend", `<span class="deck-count" aria-hidden="true">+${this.cards.length - 1}</span>`);
      }
      this.layout(false);
      this.resize = new ResizeObserver(() => this.layout(false));
      this.resize.observe(el);
      el.addEventListener("click", (e) => {
        if (this.collapsed) { e.stopPropagation(); e.preventDefault(); this.set(false); }
      }, true);
    }

    set(collapsed) {
      collapsed = collapsed && this.cards.length > 1;
      if (collapsed === this.collapsed) return;
      this.collapsed = collapsed;
      this.peek = -1;
      this.layout(true);
      this.onToggle?.(!collapsed);
    }

    /** While collapsed, lift the card with this index to the top of the stack (-1 restores). */
    peekAt(index) {
      if (!this.collapsed || index === this.peek) return;
      this.peek = index;
      this.layout(true);
    }

    layout(animate = true) {
      const { el, cards } = this;
      if (!cards.length || !el.isConnected) return;
      const from = el.getBoundingClientRect().height;
      el.style.height = "";
      el.classList.toggle("collapsed", this.collapsed);
      const slots = cards.map((c) => ({ x: c.offsetLeft, y: c.offsetTop, h: c.offsetHeight }));
      const natural = el.getBoundingClientRect().height;
      let to = natural;
      if (this.collapsed) {
        const order = deckOrder(cards.length, this.peek);
        const home = slots[0];
        order.forEach((i, k) => {
          const c = cards[i];
          const dx = home.x - slots[i].x;
          const dy = home.y - slots[i].y + k * DECK_STEP;
          c.style.transform = `translate(${dx}px, ${dy}px) scale(${1 - k * 0.045})`;
          c.style.zIndex = String(cards.length - k);
          c.style.opacity = k < DECK_SHOWN ? "1" : "0";
          c.classList.toggle("top", k === 0);
          c.tabIndex = k === 0 ? 0 : -1;
        });
        to = slots[order[0]].h + DECK_STEP * (Math.min(cards.length, DECK_SHOWN) - 1) + 2;
        const count = el.querySelector(":scope > .deck-count");
        if (count) count.style.left = `${home.x + cards[0].offsetWidth + 8}px`;
      } else {
        cards.forEach((c) => {
          c.style.transform = "";
          c.style.zIndex = "";
          c.style.opacity = "";
          c.classList.remove("top");
          c.tabIndex = 0;
        });
      }
      el.setAttribute("aria-expanded", String(!this.collapsed));
      if (!animate || reduced()) {
        el.style.height = this.collapsed ? `${to}px` : "";
        return;
      }
      el.style.height = `${from}px`;
      void el.offsetHeight;
      el.style.height = `${to}px`;
      clearTimeout(this.heightTimer);
      if (!this.collapsed) this.heightTimer = setTimeout(() => { if (!this.collapsed) el.style.height = ""; }, 600);
    }

    destroy() { this.resize?.disconnect(); }
  }

  // ================================================================ 4. snap rail + drawn line

  class SnapRail {
    constructor(rail, { onActive } = {}) {
      this.rail = rail;
      this.track = rail.querySelector(".snap-track");
      this.onActive = onActive;
      this.active = 0;
      this.hover = -1;
      let svg = rail.querySelector(".snap-line");
      if (!svg) {
        rail.insertAdjacentHTML("beforeend", '<svg class="snap-line" aria-hidden="true"><line class="track"/><line class="ink"/><g class="knot"><use href="#i-paw" x="-8" y="-8" width="16" height="16" transform="rotate(90)"/></g></svg>');
        svg = rail.querySelector(".snap-line");
      }
      this.svg = svg;
      let frame = 0;
      this.track.addEventListener("scroll", () => {
        if (frame) return;
        frame = requestAnimationFrame(() => { frame = 0; this.fromScroll(); });
      }, { passive: true });
      this.track.addEventListener("pointerover", (e) => {
        const card = e.target.closest(".sg-card");
        if (card) this.setHover(this.cards().indexOf(card));
      });
      this.track.addEventListener("pointerleave", () => this.setHover(-1));
      this.track.addEventListener("focusin", (e) => {
        const card = e.target.closest(".sg-card");
        if (card) this.setHover(this.cards().indexOf(card));
      });
      this.track.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        const cards = this.cards();
        const at = cards.indexOf(e.target.closest(".sg-card"));
        const next = cards[clamp(at + (e.key === "ArrowRight" ? 1 : -1), 0, cards.length - 1)];
        if (!next) return;
        e.preventDefault();
        next.querySelector("button")?.focus({ preventScroll: true });
        next.scrollIntoView({ inline: "nearest", block: "nearest", behavior: reduced() ? "auto" : "smooth" });
      });
      this.resize = new ResizeObserver(() => this.draw(false));
      this.resize.observe(rail);
    }

    cards() { return [...this.track.querySelectorAll(".sg-card")]; }

    /** The card snapped at the start of the rail. */
    fromScroll() {
      const cards = this.cards();
      const start = this.track.scrollLeft + this.track.offsetLeft + (parseFloat(getComputedStyle(this.track).scrollPaddingLeft) || 0);
      let best = 0;
      cards.forEach((c, i) => {
        if (Math.abs(c.offsetLeft - start) < Math.abs(cards[best].offsetLeft - start)) best = i;
      });
      if (best !== this.active) { this.active = best; this.onActive?.(best); }
      this.draw();
    }

    /** An edge fades while more cards wait beyond it; its arrow only shows then too. */
    edge() {
      const t = this.track;
      const start = t.scrollLeft <= 4;
      const end = t.scrollLeft + t.clientWidth >= t.scrollWidth - 4;
      t.classList.toggle("at-start", start);
      t.classList.toggle("at-end", end);
      this.rail.classList.toggle("at-start", start);
      this.rail.classList.toggle("at-end", end);
    }

    setHover(i) {
      this.hover = i;
      this.draw();
    }

    scrollBy(step) {
      const cards = this.cards();
      const next = cards[clamp(this.active + step, 0, cards.length - 1)];
      next?.scrollIntoView({ inline: "start", block: "nearest", behavior: reduced() ? "auto" : "smooth" });
    }

    /** Draw the line along the rail up to the centre of the active (or hovered) card. */
    draw(animate = true) {
      const cards = this.cards();
      const { svg } = this;
      this.edge();
      if (!cards.length) { svg.style.opacity = "0"; return; }
      svg.style.opacity = "";
      const box = this.rail.getBoundingClientRect();
      const pad = parseFloat(getComputedStyle(this.track).paddingLeft) || 0;
      // Scrolled all the way along, the trail reaches the last card.
      const atEnd = this.rail.classList.contains("at-end") && !this.rail.classList.contains("at-start");
      const i = this.hover >= 0 ? this.hover : atEnd ? cards.length - 1 : this.active;
      cards.forEach((c, k) => c.classList.toggle("active", k === i));
      const card = cards[clamp(i, 0, cards.length - 1)].getBoundingClientRect();
      const y = 7;
      const x0 = pad;
      const x1 = Math.max(x0 + 1, box.width - pad);
      const target = clamp(card.left + card.width / 2 - box.left, x0, x1);
      const track = svg.querySelector(".track");
      const ink = svg.querySelector(".ink");
      const knot = svg.querySelector(".knot");
      for (const line of [track, ink]) {
        line.setAttribute("x1", x0); line.setAttribute("x2", x1);
        line.setAttribute("y1", y); line.setAttribute("y2", y);
      }
      const length = x1 - x0;
      ink.style.strokeDasharray = `${length} ${length}`;
      if (!animate || reduced()) {
        ink.style.transition = knot.style.transition = "none";
        void svg.getBoundingClientRect();
      }
      ink.style.strokeDashoffset = String(length - (target - x0));
      knot.style.transform = `translate(${target}px, ${y}px)`;
      if (!animate || reduced()) requestAnimationFrame(() => { ink.style.transition = knot.style.transition = ""; });
    }

    destroy() { this.resize?.disconnect(); }
  }

  /** Draw a line from a citation chip to its source card inside `host` (position: relative). */
  function connect(host, from, to) {
    disconnect(host, true);
    if (!host || !from || !to) return null;
    const box = host.getBoundingClientRect();
    const a = from.getBoundingClientRect();
    const b = to.getBoundingClientRect();
    const below = b.top >= a.bottom;
    const sx = a.left + a.width / 2 - box.left;
    const sy = (below ? a.bottom + 2 : a.top - 2) - box.top;
    const ex = clamp(a.left + a.width / 2, b.left + 18, b.right - 18) - box.left;
    const ey = (below ? b.top - 3 : b.bottom + 3) - box.top;
    const bend = Math.max(24, Math.abs(ey - sy) * 0.5) * (below ? 1 : -1);
    const d = `M ${sx} ${sy} C ${sx} ${sy + bend}, ${ex} ${ey - bend}, ${ex} ${ey}`;
    host.insertAdjacentHTML("beforeend", `<svg class="connector" aria-hidden="true"><path d="${d}"/><circle cx="${sx}" cy="${sy}" r="2.5"/><circle class="end" cx="${ex}" cy="${ey}" r="3.5"/></svg>`);
    const svg = host.lastElementChild;
    const len = Math.ceil(svg.querySelector("path").getTotalLength());
    svg.style.setProperty("--len", String(len));
    return svg;
  }

  function disconnect(host, now = false) {
    host?.querySelectorAll(":scope > .connector").forEach((svg) => {
      if (now || reduced()) { svg.remove(); return; }
      svg.classList.add("out");
      setTimeout(() => svg.remove(), 160);
    });
  }

  // ================================================================ 5. inertial wheel

  const WheelMath = {
    /** Index under the selection band for a scroll offset. */
    nearest(offset, row, count) {
      return clamp(Math.round(offset / row), 0, Math.max(0, count - 1));
    },
    /** Where a flick at `velocity` (px/ms) comes to rest with iOS-style deceleration. */
    project(offset, velocity, rate = 0.995) {
      return offset + (velocity * rate) / (1 - rate);
    },
    /** Resistance past either end: the further out, the less it moves. */
    rubber(offset, min, max, dim = 120) {
      const band = (x) => (1 - 1 / ((x * 0.55) / dim + 1)) * dim;
      if (offset < min) return min - band(min - offset);
      if (offset > max) return max + band(offset - max);
      return offset;
    },
    /** Drum angle (radians) of item i at an offset; |angle| > π/2 is behind the drum. */
    angle(i, offset, row, radius) {
      return (i * row - offset) / radius;
    },
    /** One step of a damped spring toward `target` (x px, v px/s, dt s). */
    step({ x, v }, target, dt, stiffness = 230, damping = 28) {
      const a = -stiffness * (x - target) - damping * v;
      v += a * dt;
      x += v * dt;
      return { x, v };
    },
    settled({ x, v }, target) {
      return Math.abs(x - target) < 0.25 && Math.abs(v) < 6;
    },
    /** Mouse-wheel notches step a row each; trackpads scroll freely (pixels). */
    wheelDelta(e, row) {
      const px = e.deltaMode === 1 ? e.deltaY * row : e.deltaMode === 2 ? e.deltaY * row * 5 : e.deltaY;
      const notch = e.deltaMode !== 0 || Math.abs(e.deltaY) >= 50;
      // One event can carry several notches when the wheel spins fast (100 px or 3 lines each).
      const per = e.deltaMode === 1 ? 3 : e.deltaMode === 2 ? 1 : 100;
      const steps = notch ? Math.min(5, Math.max(1, Math.round(Math.abs(e.deltaY) / per))) : 0;
      return { px, notch, dir: Math.sign(px), steps };
    },
  };

  // When the page last scrolled by a wheel outside any wheel picker.
  let pageWheelAt = -Infinity;
  doc?.addEventListener("wheel", (e) => {
    if (!e.target.closest?.(".wheel")) pageWheelAt = e.timeStamp;
  }, { capture: true, passive: true });

  class Wheel {
    /** items: [{label, value, hint?}]. onChange(item, index) fires once the wheel settles on a new item. */
    constructor(el, { items = [], index = 0, onChange, label = "" } = {}) {
      this.el = el;
      this.onChange = onChange;
      this.id = uid("wheel");
      el.classList.add("wheel");
      el.tabIndex = 0;
      el.setAttribute("role", "listbox");
      el.setAttribute("aria-orientation", "vertical");
      if (label) el.setAttribute("aria-label", label);
      this.raf = 0;
      this.state = { x: 0, v: 0 };
      this.target = 0;
      this.setItems(items, index);
      this.bind();
    }

    metrics() {
      const cs = getComputedStyle(this.el);
      this.row = parseFloat(cs.getPropertyValue("--row")) || 32;
      this.radius = parseFloat(cs.getPropertyValue("--radius")) || 80;
    }

    setItems(items, index = 0) {
      this.items = items;
      this.el.innerHTML = '<div class="wheel-band" aria-hidden="true"></div>' + items.map((it, i) =>
        `<div class="wheel-item" role="option" id="${this.id}-${i}" data-i="${i}" aria-selected="false" title="${esc(it.title || it.label)}"><span>${esc(it.label)}</span>${it.hint ? `<small>${esc(it.hint)}</small>` : ""}</div>`).join("");
      this.nodes = [...this.el.querySelectorAll(".wheel-item")];
      this.metrics();
      this.index = clamp(index, 0, Math.max(0, items.length - 1));
      this.state = { x: this.index * this.row, v: 0 };
      this.target = this.state.x;
      this.render();
    }

    /** Move to an item without telling onChange (settings read from the backend). */
    setIndex(index, { animate = false, silent = true } = {}) {
      index = clamp(index, 0, this.items.length - 1);
      if (silent) this.index = index;
      if (animate) this.animateTo(index * this.row);
      else { cancelAnimationFrame(this.raf); this.raf = 0; this.state = { x: index * this.row, v: 0 }; this.render(); }
    }

    get max() { return Math.max(0, (this.items.length - 1) * this.row); }

    render() {
      const { row, radius } = this;
      const x = this.state.x;
      const at = WheelMath.nearest(x, row, this.items.length);
      this.nodes.forEach((node, i) => {
        const a = WheelMath.angle(i, x, row, radius);
        if (Math.abs(a) > 1.5) { node.style.visibility = "hidden"; return; }
        node.style.visibility = "";
        node.style.transform = `rotateX(${-a}rad)`;
        node.style.opacity = String(Math.max(0, Math.cos(a)) ** 0.7);
        node.setAttribute("aria-selected", String(i === at));
      });
      if (this.nodes[at]) this.el.setAttribute("aria-activedescendant", this.nodes[at].id);
    }

    animateTo(target, velocity = 0) {
      this.target = clamp(target, 0, this.max);
      if (reduced()) {
        this.state = { x: this.target, v: 0 };
        this.render();
        this.commit();
        return;
      }
      this.state.v = velocity;
      if (this.raf) return;
      let last = performance.now();
      const tick = (now) => {
        const dt = Math.min(0.032, (now - last) / 1000);
        last = now;
        // Two half steps keep the spring stable at low frame rates.
        this.state = WheelMath.step(WheelMath.step(this.state, this.target, dt / 2), this.target, dt / 2);
        if (WheelMath.settled(this.state, this.target)) {
          this.state = { x: this.target, v: 0 };
          this.raf = 0;
          this.render();
          this.commit();
          return;
        }
        this.render();
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }

    stop() {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.state.v = 0;
    }

    commit() {
      const i = WheelMath.nearest(this.state.x, this.row, this.items.length);
      if (i === this.index) return;
      this.index = i;
      this.onChange?.(this.items[i], i);
    }

    stepBy(n) {
      const base = WheelMath.nearest(this.raf ? this.target : this.state.x, this.row, this.items.length);
      this.animateTo((base + n) * this.row);
    }

    bind() {
      const el = this.el;
      let idle = 0;
      let lastWheel = 0;
      el.addEventListener("wheel", (e) => {
        // A page scroll that started elsewhere and passes over the wheel keeps scrolling the page.
        if (e.timeStamp - pageWheelAt < 350) { pageWheelAt = e.timeStamp; return; }
        e.preventDefault();
        const { px, notch, dir, steps } = WheelMath.wheelDelta(e, this.row);
        if (notch) { this.stepBy(dir * steps); return; }
        // Trackpad: follow the fingers, then snap with the speed they had.
        this.stop();
        const now = performance.now();
        const speed = now - lastWheel > 0 ? px / Math.max(8, now - lastWheel) : 0;
        lastWheel = now;
        this.state.x = WheelMath.rubber(this.state.x + px, 0, this.max);
        this.render();
        clearTimeout(idle);
        idle = setTimeout(() => {
          const rest = WheelMath.project(this.state.x, speed * 0.5, 0.99);
          this.animateTo(WheelMath.nearest(rest, this.row, this.items.length) * this.row, speed * 400);
        }, 90);
      }, { passive: false });

      let drag = null;
      el.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        this.stop();
        drag = { y: e.clientY, x0: this.state.x, moved: false, samples: [{ t: e.timeStamp, y: e.clientY }], id: e.pointerId };
      });
      el.addEventListener("pointermove", (e) => {
        if (!drag || e.pointerId !== drag.id) return;
        const dy = drag.y - e.clientY;
        if (!drag.moved && Math.abs(dy) > 3) {
          drag.moved = true;
          el.setPointerCapture(e.pointerId);
          el.classList.add("dragging");
        }
        if (!drag.moved) return;
        this.state.x = WheelMath.rubber(drag.x0 + dy, 0, this.max);
        drag.samples.push({ t: e.timeStamp, y: e.clientY });
        while (drag.samples.length > 2 && e.timeStamp - drag.samples[0].t > 100) drag.samples.shift();
        this.render();
      });
      const end = (e) => {
        if (!drag || e.pointerId !== drag.id) return;
        const d = drag;
        drag = null;
        el.classList.remove("dragging");
        if (!d.moved) {
          const item = e.target.closest?.(".wheel-item");
          if (item) this.animateTo(Number(item.dataset.i) * this.row);
          el.focus({ preventScroll: true });
          return;
        }
        const first = d.samples[0];
        const lastS = d.samples[d.samples.length - 1];
        const dt = Math.max(1, lastS.t - first.t);
        const velocity = (first.y - lastS.y) / dt; // px/ms, + = toward later items
        const rest = WheelMath.project(this.state.x, velocity);
        this.animateTo(WheelMath.nearest(rest, this.row, this.items.length) * this.row, velocity * 1000);
      };
      el.addEventListener("pointerup", end);
      el.addEventListener("pointercancel", end);

      el.addEventListener("keydown", (e) => {
        const n = this.items.length;
        const moves = { ArrowDown: 1, ArrowUp: -1, PageDown: 4, PageUp: -4 };
        if (e.key in moves) this.stepBy(moves[e.key]);
        else if (e.key === "Home") this.animateTo(0);
        else if (e.key === "End") this.animateTo((n - 1) * this.row);
        else return;
        e.preventDefault();
      });
    }
  }

  // ================================================================ 6. circular reveal

  /** Run `update` (a DOM change) as a view transition that grows a circle from the click. */
  async function circleTransition(source, update, { duration = 620 } = {}) {
    if (!doc?.startViewTransition || reduced()) { await update(); return; }
    const { x, y } = pointOf(source);
    const r = Math.hypot(Math.max(x, root.innerWidth - x), Math.max(y, root.innerHeight - y));
    const html = doc.documentElement;
    html.classList.add("vt-circle");
    const t = doc.startViewTransition(() => update());
    try {
      await t.ready;
      html.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
        { duration, easing: "cubic-bezier(.4, 0, .15, 1)", pseudoElement: "::view-transition-new(root)" },
      );
      await t.finished;
    } catch (_) {
      // Skipped (another transition started, or the update threw): the DOM is already updated.
    } finally {
      html.classList.remove("vt-circle");
    }
  }

  /** Grow `el` out of a circle centred on the click (a sheet opening from the card clicked). */
  function revealFrom(el, source, { duration = 560 } = {}) {
    if (!el?.animate || reduced()) return null;
    const rect = el.getBoundingClientRect();
    const { x, y } = pointOf(source);
    const cx = x - rect.left;
    const cy = y - rect.top;
    const r = Math.hypot(Math.max(Math.abs(cx), Math.abs(rect.width - cx)), Math.max(Math.abs(cy), Math.abs(rect.height - cy)));
    return el.animate(
      { clipPath: [`circle(0px at ${cx}px ${cy}px)`, `circle(${r}px at ${cx}px ${cy}px)`] },
      { duration, easing: "cubic-bezier(.4, 0, .15, 1)" },
    );
  }

  // ================================================================ small controls

  /** Segmented control made of radio inputs: keeps the sliding thumb on the checked one. */
  function segmented(el) {
    const sync = () => {
      const options = [...el.querySelectorAll("input[type=radio], button")].filter((o) => !o.closest("[hidden]"));
      const i = options.findIndex((o) => o.checked || o.classList.contains("active"));
      el.style.setProperty("--n", String(options.length));
      el.style.setProperty("--i", String(Math.max(0, i)));
      el.classList.toggle("no-thumb", i < 0);
    };
    if (!el._segSync) {
      el.addEventListener("change", sync);
      el.addEventListener("click", () => requestAnimationFrame(sync));
      el._segSync = sync;
    }
    sync();
    return sync;
  }

  function syncSegments(scope = doc) {
    scope.querySelectorAll(".seg").forEach((el) => segmented(el));
  }

  /** Change a number with the old value rolling out and the new one rolling in. */
  function roll(el, value) {
    const text = String(value);
    const current = el.querySelector("span:not(.out-up):not(.out-down)");
    if (current?.textContent === text) return;
    const up = !current || Number(value) >= Number(current.textContent);
    el.classList.add("roll");
    const next = doc.createElement("span");
    next.textContent = text;
    if (current && !reduced()) {
      next.className = up ? "in-up" : "in-down";
      el.append(next);
      current.className = up ? "out-up" : "out-down";
      setTimeout(() => current.remove(), 420);
      requestAnimationFrame(() => requestAnimationFrame(() => { next.className = ""; }));
    } else {
      el.replaceChildren(next);
    }
  }

  /** Large title that hands over to the frosted toolbar once it scrolls under it. */
  function largeTitle(page) {
    const title = page.querySelector(".large-title");
    if (!title) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      // Measured on screen, so a page that is hidden right now keeps its state.
      const bar = page.querySelector(".page-bar");
      const t = title.getBoundingClientRect();
      if (!t.height || !bar) return;
      page.classList.toggle("scrolled", t.bottom < bar.getBoundingClientRect().bottom + 4);
    };
    page.addEventListener("scroll", () => { if (!frame) frame = requestAnimationFrame(update); }, { passive: true });
    new ResizeObserver(() => update()).observe(page);
    update();
  }

  /** Copy text; falls back to a hidden textarea where the clipboard API is refused. */
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (_) {
      const area = doc.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
      doc.body.append(area);
      area.select();
      const ok = doc.execCommand("copy");
      area.remove();
      if (!ok) throw new Error("clipboard refused");
    }
  }

  /** Apple-style alert; resolves true for the confirming button, false otherwise. */
  /** `groups` ([{ title, items }]) adds a list that scrolls inside the sheet, with the title
      above it and `note` and the buttons below always in view; `subtitle` goes under the title. */
  function confirmSheet({ title, message = "", confirm = "确定", cancel = "取消", danger = false, subtitle = "", groups = null, note = "" } = {}) {
    return new Promise((resolve) => {
      const before = doc.activeElement;
      const scrim = doc.createElement("div");
      scrim.className = "alert-scrim";
      const id = uid("alert");
      const list = groups ? `<div class="alert-notes" id="${id}-m" tabindex="0">${groups.map((g) =>
          `${g.title ? `<h4>${esc(g.title)}</h4>` : ""}${g.items.length ? `<ul>${g.items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}`).join("")}</div>` : "";
      scrim.innerHTML = `<div class="alert${groups ? " with-notes" : ""}" role="alertdialog" aria-modal="true" aria-labelledby="${id}-t" aria-describedby="${id}-m">
          <h3 id="${id}-t">${esc(title)}</h3>${subtitle ? `<p class="alert-sub">${esc(subtitle)}</p>` : ""}${message ? `<p id="${id}-m">${esc(message)}</p>` : ""}${list}
          ${note ? `<p class="alert-note">${esc(note)}</p>` : ""}
          <div class="alert-actions">
            <button type="button" class="btn" data-answer="no">${esc(cancel)}</button>
            <button type="button" class="btn primary${danger ? " danger" : ""}" data-answer="yes">${esc(confirm)}</button>
          </div>
        </div>`;
      const done = (answer) => {
        doc.removeEventListener("keydown", onKey, true);
        scrim.classList.add("closing");
        setTimeout(() => scrim.remove(), reduced() ? 0 : 150);
        before?.focus?.({ preventScroll: true });
        resolve(answer);
      };
      const onKey = (e) => {
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(false); }
        else if (e.key === "Tab") {
          const buttons = [...scrim.querySelectorAll("button")];
          const i = buttons.indexOf(doc.activeElement);
          e.preventDefault();
          buttons[(i + (e.shiftKey ? -1 : 1) + buttons.length) % buttons.length].focus();
        }
      };
      scrim.addEventListener("click", (e) => {
        const b = e.target.closest("[data-answer]");
        if (b) done(b.dataset.answer === "yes");
        else if (e.target === scrim) done(false);
      });
      doc.addEventListener("keydown", onKey, true);
      doc.body.append(scrim);
      scrim.querySelector(danger ? '[data-answer="no"]' : '[data-answer="yes"]').focus();
    });
  }

  /** Read-only facts in the alert style: a title, label / value rows (empty values left out),
      a grey note, and one button. Resolves when closed. */
  function infoSheet({ title, rows = [], note = "", close = "好" } = {}) {
    return new Promise((resolve) => {
      const before = doc.activeElement;
      const scrim = doc.createElement("div");
      scrim.className = "alert-scrim";
      const id = uid("info");
      const items = rows.filter(([, value]) => value);
      scrim.innerHTML = `<div class="alert info-sheet" role="dialog" aria-modal="true" aria-labelledby="${id}-t">
          <h3 id="${id}-t">${esc(title)}</h3>
          ${items.length ? `<dl class="alert-dl">${items.map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join("")}</dl>` : ""}
          ${note ? `<p class="alert-note">${esc(note)}</p>` : ""}
          <div class="alert-actions"><button type="button" class="btn primary" data-answer="close">${esc(close)}</button></div>
        </div>`;
      const done = () => {
        doc.removeEventListener("keydown", onKey, true);
        scrim.classList.add("closing");
        setTimeout(() => scrim.remove(), reduced() ? 0 : 150);
        before?.focus?.({ preventScroll: true });
        resolve();
      };
      const onKey = (e) => {
        if (e.key === "Escape" || e.key === "Enter") { e.preventDefault(); e.stopPropagation(); done(); }
        else if (e.key === "Tab") e.preventDefault();
      };
      scrim.addEventListener("click", (e) => {
        if (e.target.closest("[data-answer]") || e.target === scrim) done();
      });
      doc.addEventListener("keydown", onKey, true);
      doc.body.append(scrim);
      scrim.querySelector("[data-answer]").focus();
    });
  }

  /** A choice between a few options, in the same alert style: one row per option (icon,
      label, one short line; `warn` lines are orange, `disabled` ones cannot be picked), an
      optional "remember" checkbox and two buttons. Resolves {value, remember}; the cancel
      button gives {value: cancelValue} when one is set (e.g. "不用了" = "off"), else null,
      and so do Esc and a click outside (always null: nothing was decided). */
  function chooseSheet({
    title, message = "", options = [], value = "", remember = "", confirm = "确定", cancel = "取消", cancelValue = null,
  } = {}) {
    return new Promise((resolve) => {
      const before = doc.activeElement;
      const scrim = doc.createElement("div");
      scrim.className = "alert-scrim";
      const id = uid("choose");
      const first = options.find((o) => !o.disabled);
      const selected = options.some((o) => o.value === value && !o.disabled) ? value : first?.value;
      const icon = (name) => (name && root.Icons ? root.Icons.svg(name) : "");
      scrim.innerHTML = `<div class="alert choose" role="dialog" aria-modal="true" aria-labelledby="${id}-t"${message ? ` aria-describedby="${id}-m"` : ""}>
          <h3 id="${id}-t">${esc(title)}</h3>${message ? `<p id="${id}-m">${esc(message)}</p>` : ""}
          <div class="choices" role="radiogroup" aria-labelledby="${id}-t">
            ${options.map((o) => `<label class="choice${o.disabled ? " disabled" : ""}">
              <input type="radio" name="${id}" value="${esc(o.value)}"${o.value === selected ? " checked" : ""}${o.disabled ? " disabled" : ""}>
              <span class="choice-icon">${icon(o.icon)}</span>
              <span class="choice-main"><strong>${esc(o.label)}</strong>${o.line ? `<small${o.warn ? ' class="warn"' : ""}>${esc(o.line)}</small>` : ""}</span>
              <span class="choice-mark">${icon("check")}</span>
            </label>`).join("")}
          </div>
          ${remember ? `<label class="choice-remember"><input type="checkbox" class="check">${esc(remember)}</label>` : ""}
          <div class="alert-actions">
            <button type="button" class="btn" data-answer="no">${esc(cancel)}</button>
            <button type="button" class="btn primary" data-answer="yes">${esc(confirm)}</button>
          </div>
        </div>`;
      const done = (answer) => {
        doc.removeEventListener("keydown", onKey, true);
        scrim.classList.add("closing");
        setTimeout(() => scrim.remove(), reduced() ? 0 : 150);
        before?.focus?.({ preventScroll: true });
        resolve(answer);
      };
      const picked = () => ({
        value: scrim.querySelector(`input[name="${id}"]:checked`)?.value ?? null,
        remember: !!scrim.querySelector(".choice-remember input")?.checked,
      });
      const onKey = (e) => {
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); }
        else if (e.key === "Enter" && !e.target.closest?.("[data-answer]")) { e.preventDefault(); done(picked()); }
        else if (e.key === "Tab") {
          // Keep focus inside: the chosen option (arrow keys move between options), the box, the buttons.
          const stops = [scrim.querySelector(`input[name="${id}"]:checked`), scrim.querySelector(".choice-remember input"),
            ...scrim.querySelectorAll("[data-answer]")].filter(Boolean);
          const i = stops.indexOf(doc.activeElement);
          e.preventDefault();
          stops[(i + (e.shiftKey ? -1 : 1) + stops.length) % stops.length].focus();
        }
      };
      scrim.addEventListener("click", (e) => {
        const b = e.target.closest("[data-answer]");
        if (b) done(b.dataset.answer === "yes" ? picked() : cancelValue != null ? { value: cancelValue, remember: false } : null);
        else if (e.target === scrim) done(null);
      });
      doc.addEventListener("keydown", onKey, true);
      doc.body.append(scrim);
      (scrim.querySelector(`input[name="${id}"]:checked`) || scrim.querySelector('[data-answer="yes"]')).focus();
    });
  }

  root.UI = {
    reduced, esc, pointOf,
    runDone, prepareDone,
    Selection, setCheck,
    deckOrder, Deck,
    SnapRail, connect, disconnect,
    WheelMath, Wheel,
    circleTransition, revealFrom,
    segmented, syncSegments, roll, largeTitle, confirmSheet, chooseSheet, infoSheet, copyText,
  };
})(typeof window !== "undefined" ? window : globalThis);
