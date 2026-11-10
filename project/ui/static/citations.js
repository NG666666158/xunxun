"use strict";

/* Inline citations of an answer, used by app.js: "[来源：X｜位置：Y]" markers in every form
   the backend's fact check accepts (CITATION and evidence_for in
   project/rag_agent/answering.py), each matched to one of the answer's source cards.
   Kept free of the page so tests can run it under Node. */
(function (root) {
  // The "位置：" label is optional: "[来源：差旅制度.docx]", "[来源：X.docx | 未标注位置]".
  const MARKER = /[\[【]\s*来源\s*[:：]\s*([^\]】]+?)\s*(?:[｜|，,;；]\s*位置\s*[:：]\s*([^\]】]+?)\s*)?[\]】]/g;
  const SEPARATOR = /\s*[｜|，,;；]\s*/;
  // A marker still being typed at the end of a streaming draft.
  const OPEN_TAIL = /[\[【]\s*来源[^\]】]*$/;

  const loose = (text) => String(text ?? "").replace(/\s+/g, "").toLowerCase();

  /** [source, locator] of one marker. A locator written without its label stays in the
      source text and is split off there, unless that text names a known source. */
  function parse(source, locator, cards) {
    source = String(source ?? "").trim();
    locator = String(locator ?? "").trim();
    if (!locator && !cards.some((c) => loose(c.source) === loose(source))) {
      const cut = source.match(SEPARATOR);
      if (cut) return [source.slice(0, cut.index), source.slice(cut.index + cut[0].length)];
    }
    return [source, locator];
  }

  /** Index of the card a citation points to, or -1: same source and locator (spacing and
      case ignored), else the source's first card. A file name the model shortened
      matches the cards whose name contains it. */
  function match(cards, source, locator) {
    const indexes = (test) => cards.flatMap((c, i) => (test(c) ? [i] : []));
    let same = indexes((c) => loose(c.source) === loose(source));
    if (!same.length && loose(source)) same = indexes((c) => loose(c.source).includes(loose(source)));
    if (!same.length) return -1;
    const exact = locator ? same.find((i) => loose(cards[i].locator) === loose(locator)) : undefined;
    return exact ?? same[0];
  }

  /** The answer with each marker replaced by "\u0000k\u0000" (k counts markers), the
      card number of each marker, and the cards numbered cited first, in order of
      appearance. A citation of a passage that was not retrieved gets a card of its own. */
  function link(answer, citations) {
    const cards = (citations || []).map((c) => ({
      source: c.source || "", locator: c.locator || "", doc_id: c.doc_id || "",
      status: c.status || "", quote: c.quote || "", cited: false,
    }));
    const retrieved = cards.slice();
    const tokens = [];
    const text = String(answer ?? "").replace(MARKER, (_, rawSource, rawLocator) => {
      const [source, locator] = parse(rawSource, rawLocator, retrieved);
      let i = match(retrieved, source, locator);
      if (i < 0) {
        i = cards.findIndex((c, j) => j >= retrieved.length && loose(c.source) === loose(source) && loose(c.locator) === loose(locator));
        if (i < 0) i = cards.push({ source, locator, doc_id: "", status: "", quote: "", cited: false }) - 1;
      }
      cards[i].cited = true;
      tokens.push(i);
      return `\u0000${tokens.length - 1}\u0000`;
    });

    const order = [];
    tokens.forEach((i) => { if (!order.includes(i)) order.push(i); });
    cards.forEach((_, i) => { if (!order.includes(i)) order.push(i); });
    const numberOf = new Map(order.map((i, n) => [i, n + 1]));
    return {
      text,
      marks: tokens.map((i) => numberOf.get(i)),
      sources: order.map((i) => ({ ...cards[i], n: numberOf.get(i) })),
    };
  }

  /** A streaming draft without its markers, complete or still being typed. */
  function strip(answer) {
    return String(answer ?? "").replace(MARKER, "").replace(OPEN_TAIL, "");
  }

  root.Citations = { MARKER, parse, match, link, strip };
})(typeof window !== "undefined" ? window : globalThis);
