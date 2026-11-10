"use strict";

/* One line-icon set for the whole UI (main window and quick-search window): 24 px grid,
   1.7 px round strokes, drawn to sit beside SF Symbols' regular weight. The sprite is
   injected once; Icons.svg("trash") returns markup that uses it. */
(function (root) {
  const P = {
    sidebar: '<rect x="3" y="4.5" width="18" height="15" rx="3.5"/><path d="M9 4.5v15M5.5 8h1.5M5.5 11h1.5"/>',
    compose: '<path d="M11.5 4.5H7A2.5 2.5 0 0 0 4.5 7v10A2.5 2.5 0 0 0 7 19.5h10a2.5 2.5 0 0 0 2.5-2.5v-4.5"/><path d="M17.6 3.9a1.8 1.8 0 0 1 2.5 2.5l-7.4 7.4-3.2.8.8-3.2z"/>',
    chat: '<path d="M12 4.5c4.7 0 8.5 3.1 8.5 7s-3.8 7-8.5 7c-1 0-2-.1-2.9-.4L5 19.5l1.2-3.4C4.5 14.8 3.5 13.2 3.5 11.5c0-3.9 3.8-7 8.5-7z"/>',
    books: '<rect x="3.5" y="4.5" width="4.5" height="15" rx="1.2"/><rect x="9.5" y="4.5" width="4.5" height="15" rx="1.2"/><path d="M15.6 6.4l2.6-.7a1 1 0 0 1 1.2.7l2.8 10.7a1 1 0 0 1-.7 1.2l-2.6.7a1 1 0 0 1-1.2-.7L14.9 7.6a1 1 0 0 1 .7-1.2z"/><path d="M5.2 8h1.1M11.2 8h1.1"/>',
    sliders: '<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/>',
    gear: '<path d="M9.87 5.12L10.13 2.79L13.87 2.79L14.13 5.12A7.2 7.2 0 0 1 15.36 5.63L17.19 4.17L19.83 6.81L18.37 8.64A7.2 7.2 0 0 1 18.88 9.87L21.21 10.13L21.21 13.87L18.88 14.13A7.2 7.2 0 0 1 18.37 15.36L19.83 17.19L17.19 19.83L15.36 18.37A7.2 7.2 0 0 1 14.13 18.88L13.87 21.21L10.13 21.21L9.87 18.88A7.2 7.2 0 0 1 8.64 18.37L6.81 19.83L4.17 17.19L5.63 15.36A7.2 7.2 0 0 1 5.12 14.13L2.79 13.87L2.79 10.13L5.12 9.87A7.2 7.2 0 0 1 5.63 8.64L4.17 6.81L6.81 4.17L8.64 5.63A7.2 7.2 0 0 1 9.87 5.12Z"/><circle cx="12" cy="12" r="3.2"/>',
    paw: '<ellipse cx="12" cy="15.6" rx="4" ry="3.4" fill="currentColor" stroke="none"/><ellipse cx="6.6" cy="10.6" rx="1.7" ry="2.2" fill="currentColor" stroke="none"/><ellipse cx="17.4" cy="10.6" rx="1.7" ry="2.2" fill="currentColor" stroke="none"/><ellipse cx="9.7" cy="6.9" rx="1.7" ry="2.2" fill="currentColor" stroke="none"/><ellipse cx="14.3" cy="6.9" rx="1.7" ry="2.2" fill="currentColor" stroke="none"/>',
    search: '<circle cx="10.8" cy="10.8" r="6.3"/><path d="M15.5 15.5l4.5 4.5"/>',
    folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.7l2 2h7.3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    folderPlus: '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.7l2 2h7.3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/><path d="M12 10.5v6M9 13.5h6"/>',
    folderOpen: '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.7l2 2h7.3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/><path d="M12 16.5v-5M9.6 13.6L12 11.2l2.4 2.4"/>',
    doc: '<path d="M7 3.5h6.5L18.5 8.5v11a1.5 1.5 0 0 1-1.5 1.5H7a1.5 1.5 0 0 1-1.5-1.5v-14.5A1.5 1.5 0 0 1 7 3.5z"/><path d="M13.5 3.5v5h5"/>',
    docPlus: '<path d="M7 3.5h6.5L18.5 8.5v11a1.5 1.5 0 0 1-1.5 1.5H7a1.5 1.5 0 0 1-1.5-1.5v-14.5A1.5 1.5 0 0 1 7 3.5z"/><path d="M13.5 3.5v5h5M12 11.5v6M9 14.5h6"/>',
    image: '<rect x="3.5" y="5" width="17" height="14" rx="2.5"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17l4.6-4.4a1.3 1.3 0 0 1 1.8 0L14 16l2-1.9a1.3 1.3 0 0 1 1.8 0L20 16.2"/>',
    cloud: '<path d="M7.5 18.5h9.8a3.7 3.7 0 0 0 .6-7.35A5.5 5.5 0 0 0 7.3 9.6a4.45 4.45 0 0 0 .2 8.9z"/><path d="M12 15.5v-5M9.8 12.6L12 10.4l2.2 2.2"/>',
    tag: '<path d="M4 5.5v5.1a1.5 1.5 0 0 0 .44 1.06l7.9 7.9a1.5 1.5 0 0 0 2.12 0l5.1-5.1a1.5 1.5 0 0 0 0-2.12l-7.9-7.9A1.5 1.5 0 0 0 10.6 4H5.5A1.5 1.5 0 0 0 4 5.5z"/><circle cx="8.3" cy="8.3" r="1.3"/>',
    docText: '<path d="M7 3.5h6.5L18.5 8.5v11a1.5 1.5 0 0 1-1.5 1.5H7a1.5 1.5 0 0 1-1.5-1.5v-14.5A1.5 1.5 0 0 1 7 3.5z"/><path d="M13.5 3.5v5h5M9 13h6M9 16.5h4"/>',
    trash: '<path d="M4.5 7h15M9.5 7V5.2c0-.7.5-1.2 1.2-1.2h2.6c.7 0 1.2.5 1.2 1.2V7M6.5 7l.9 12.1c.1.8.7 1.4 1.5 1.4h6.2c.8 0 1.4-.6 1.5-1.4L17.5 7M10.2 11v5.5M13.8 11v5.5"/>',
    refresh: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4.5v4h-4"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    chevron: '<path d="M9.5 6l6 6-6 6"/>',
    chevronLeft: '<path d="M14.5 6l-6 6 6 6"/>',
    chevronDown: '<path d="M6 9.5l6 6 6-6"/>',
    up: '<path d="M12 19V5.5M5.5 12l6.5-6.5 6.5 6.5"/>',
    stop: '<rect x="7" y="7" width="10" height="10" rx="2.2" fill="currentColor" stroke="none"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.8v1.7M12 19.5v1.7M2.8 12h1.7M19.5 12h1.7M5.5 5.5l1.2 1.2M17.3 17.3l1.2 1.2M5.5 18.5l1.2-1.2M17.3 6.7l1.2-1.2"/>',
    moon: '<path d="M19.5 14.6A7.8 7.8 0 0 1 9.4 4.5a7.8 7.8 0 1 0 10.1 10.1z"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8v.01"/>',
    more: '<circle cx="5.5" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="18.5" cy="12" r="1.5" fill="currentColor" stroke="none"/>',
    copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2.5"/><path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2"/>',
    eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    eyeOff: '<path d="M2.5 12S6 5.5 12 5.5c1.9 0 3.5.7 4.9 1.6M21.5 12s-3.5 6.5-9.5 6.5c-1.9 0-3.5-.7-4.9-1.6"/><path d="M9.9 14.1A3 3 0 0 1 14.1 9.9M4.5 19.5l15-15"/>',
    key: '<circle cx="8" cy="15" r="4"/><path d="M11 12l8.5-8.5M16.5 6.5l2.5 2.5M14 9l2 2"/>',
    lock: '<rect x="5" y="10.5" width="14" height="10" rx="2.5"/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5"/>',
    keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/><path d="M6 10h.01M9.3 10h.01M12.6 10h.01M15.9 10h.01M7.5 14h9"/>',
    clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    scope: '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3"/>',
    alert: '<path d="M10.3 4.2L2.9 17.3A2 2 0 0 0 4.6 20.3h14.8a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z"/><path d="M12 9.5v4.5M12 17v.01"/>',
    open: '<path d="M13.5 4.5h6v6M19.5 4.5l-8 8"/><path d="M18 14v3.5a2 2 0 0 1-2 2H6.5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2H10"/>',
    followup: '<path d="M6 5v6.5a3 3 0 0 0 3 3h9.5"/><path d="M15 11l3.5 3.5L15 18"/>',
    shield: '<path d="M12 3.5l7 2.8v5.2c0 4.4-3 7.7-7 9-4-1.3-7-4.6-7-9V6.3z"/><path d="M8.8 12.2l2.2 2.2 4.3-4.6"/>',
    download: '<path d="M12 4v11.5M7 10.5l5 5 5-5M4.5 20h15"/>',
    upload: '<path d="M12 15.5V4M7 9l5-5 5 5M4.5 15.5v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2.5"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    shuffle: '<path d="M4 17h3.3c1.3 0 2.5-.7 3.2-1.8l3-5.4c.7-1.1 1.9-1.8 3.2-1.8H20M17.5 5.5L20 8l-2.5 2.5M4 7h3.3c1.2 0 2.3.6 3 1.5M20 16h-3.3c-1.2 0-2.3-.6-3-1.5M17.5 13.5L20 16l-2.5 2.5"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    reader: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M12 4.5v15M6.5 8.5h3M6.5 11.5h3M14.5 8.5h3M14.5 11.5h3"/>',
    sparkle: '<path d="M12 3.5c.6 3.9 2.6 5.9 6.5 6.5-3.9.6-5.9 2.6-6.5 6.5-.6-3.9-2.6-5.9-6.5-6.5 3.9-.6 5.9-2.6 6.5-6.5zM18.5 15.5c.3 1.6 1 2.3 2.5 2.5-1.5.2-2.2.9-2.5 2.5-.3-1.6-1-2.3-2.5-2.5 1.5-.2 2.2-.9 2.5-2.5z"/>',
  };

  const NS = "http://www.w3.org/2000/svg";

  function inject(doc = root.document) {
    if (!doc || doc.getElementById("icon-sprite")) return;
    const symbols = Object.entries(P).map(([name, body]) => `<symbol id="i-${name}" viewBox="0 0 24 24">${body}</symbol>`).join("");
    const holder = doc.createElement("div");
    holder.innerHTML = `<svg id="icon-sprite" xmlns="${NS}" aria-hidden="true" style="position:absolute;width:0;height:0;overflow:hidden">${symbols}</svg>`;
    doc.body.prepend(holder.firstChild);
  }

  /** Markup for one icon; extra classes go on the <svg>. */
  function svg(name, cls = "") {
    return `<svg class="ic${cls ? " " + cls : ""}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  }

  /** The brand mark: an accent squircle holding a page with a fold. */
  const BRAND = `<span class="brand-tile">${mascot()}</span>`;

  /** 寻寻, the retriever puppy that fetches answers from your files. Parts carry classes so
      CSS can make it blink (always) and sniff with wiggling ears (while it works). */
  function mascot(cls = "") {
    return `<svg class="mascot${cls ? " " + cls : ""}" viewBox="0 0 48 48" aria-hidden="true">
      <g class="m-body">
        <path class="m-ear m-ear-l" d="M13.5 13.5c-4.6-.8-8.6 3.2-8.4 9.6.2 5.2 2.4 8.9 5.6 9.1 2.6.2 3.6-2.6 3.9-6.6.3-4.2.6-8.6-1.1-12.1z"/>
        <path class="m-ear m-ear-r" d="M34.5 13.5c4.6-.8 8.6 3.2 8.4 9.6-.2 5.2-2.4 8.9-5.6 9.1-2.6.2-3.6-2.6-3.9-6.6-.3-4.2-.6-8.6 1.1-12.1z"/>
        <path class="m-head" d="M24 9.5c8.6 0 13.6 6.2 13.6 14.3 0 8.9-6.1 15.2-13.6 15.2S10.4 32.7 10.4 23.8C10.4 15.7 15.4 9.5 24 9.5z"/>
        <ellipse class="m-muzzle" cx="24" cy="30.4" rx="7.4" ry="5.6"/>
        <ellipse class="m-cheek" cx="15.6" cy="28.2" rx="2.4" ry="1.5"/>
        <ellipse class="m-cheek" cx="32.4" cy="28.2" rx="2.4" ry="1.5"/>
        <g class="m-eyes"><circle cx="18.6" cy="22.6" r="2.1"/><circle cx="29.4" cy="22.6" r="2.1"/></g>
        <circle class="m-shine" cx="19.3" cy="21.9" r=".7"/><circle class="m-shine" cx="30.1" cy="21.9" r=".7"/>
        <path class="m-nose" d="M21.6 27.1c0-1.1 1.1-1.6 2.4-1.6s2.4.5 2.4 1.6c0 1-1.2 2-2.4 2s-2.4-1-2.4-2z"/>
        <path class="m-mouth" d="M24 29.2v1.4M24 30.6c-.8 1.1-2.2 1.3-3.1.6M24 30.6c.8 1.1 2.2 1.3 3.1.6"/>
      </g>
    </svg>`;
  }

  root.Icons = { svg, inject, mascot, names: Object.keys(P), BRAND, NAME: "寻寻" };
  if (root.document?.body) inject();
})(typeof window !== "undefined" ? window : globalThis);
