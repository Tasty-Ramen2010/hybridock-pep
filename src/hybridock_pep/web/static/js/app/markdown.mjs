// markdown.mjs — a small, safe Markdown reader for the in-app guide. It produces plain data (an AST), never HTML, so
// nothing in a guide file can inject markup: guide.mjs turns the AST into DOM nodes with textContent only.
//
// Supported (what docs/GUIDE.md uses): # / ## / ### headings (an optional {#custom-id} on the line), paragraphs,
// - and 1. lists (one nesting level), ``` fenced code, | tables |, > quotes (callouts), --- rules, and inline
// **bold**, *italic*, `code`, [links](url) and ![images](src). Anything else is shown as ordinary text.

export const slugify = (text) => text.toLowerCase().replace(/`/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');

const INLINE = /(`[^`]+`)|(!\[[^\]]*\]\([^)\s]+\))|(\[[^\]]+\]\([^)\s]+\))|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)/;

/** "Some **bold** and `code`" → [{t:'text',v}, {t:'strong',c:[…]}, {t:'code',v}] */
export function parseInline(text) {
  const out = [];
  let rest = text;
  while (rest) {
    const m = INLINE.exec(rest);
    if (!m) { out.push({ t: 'text', v: rest }); break; }
    if (m.index) out.push({ t: 'text', v: rest.slice(0, m.index) });
    const tok = m[0];
    if (m[1]) out.push({ t: 'code', v: tok.slice(1, -1) });
    else if (m[2]) { const [, alt, src] = /^!\[([^\]]*)\]\(([^)\s]+)\)$/.exec(tok); out.push({ t: 'img', alt, src }); }
    else if (m[3]) { const [, label, href] = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(tok); out.push({ t: 'link', href, c: parseInline(label) }); }
    else if (m[4]) out.push({ t: 'strong', c: parseInline(tok.slice(2, -2)) });
    else out.push({ t: 'em', c: parseInline(tok.slice(1, -1)) });
    rest = rest.slice(m.index + tok.length);
  }
  return out;
}

/** Only these links are followed; anything else (javascript:, data:, …) becomes plain text in the page. */
export function safeHref(href) {
  if (/^(https?:|mailto:|#)/i.test(href)) return href;
  if (/^[a-z][a-z0-9+.-]*:/i.test(href)) return null; // some other scheme
  return href; // a relative path
}

const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
const isTableRule = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l) && l.includes('-');
const splitRow = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
const listMarker = (l) => /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(l);

export function parseMarkdown(src) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let i = 0;
  const usedIds = new Set();
  const uniqueId = (id) => { let n = id || 'section', k = 2; while (usedIds.has(n)) n = `${id}-${k++}`; usedIds.add(n); return n; };

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    // fenced code
    const fence = /^```(\S*)\s*$/.exec(line);
    if (fence) {
      const body = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push({ type: 'code', lang: fence[1], text: body.join('\n') });
      continue;
    }
    // heading, with an optional {#id}
    const head = /^(#{1,4})\s+(.*?)(?:\s*\{#([a-z0-9-]+)\})?\s*$/.exec(line);
    if (head) {
      const text = head[2];
      blocks.push({ type: 'heading', level: head[1].length, id: uniqueId(head[3] || slugify(text)), text, inlines: parseInline(text) });
      i++;
      continue;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { blocks.push({ type: 'hr' }); i++; continue; }

    // table
    if (isTableRow(line) && i + 1 < lines.length && isTableRule(lines[i + 1])) {
      const header = splitRow(line).map(parseInline);
      i += 2;
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) rows.push(splitRow(lines[i++]).map(parseInline));
      blocks.push({ type: 'table', head: header, rows });
      continue;
    }
    // quote (a callout)
    if (/^>\s?/.test(line)) {
      const inner = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) inner.push(lines[i++].replace(/^>\s?/, ''));
      blocks.push({ type: 'quote', blocks: parseMarkdown(inner.join('\n')).blocks });
      continue;
    }
    // list (items may carry indented continuation lines and one nested list)
    const lm = listMarker(line);
    if (lm) {
      const ordered = /\d/.test(lm[2]);
      const baseIndent = lm[1].length;
      const items = [];
      while (i < lines.length) {
        const m = listMarker(lines[i]);
        if (!m || m[1].length < baseIndent || (m[1].length === baseIndent && /\d/.test(m[2]) !== ordered)) break;
        if (m[1].length > baseIndent) break; // belongs to the previous item (handled below)
        const item = { inlines: parseInline(m[3]), children: [] };
        i++;
        const nested = [];
        while (i < lines.length && (/^\s+\S/.test(lines[i]) || !lines[i].trim())) {
          if (!lines[i].trim()) { if (i + 1 < lines.length && /^\s+\S/.test(lines[i + 1])) { nested.push(''); i++; continue; } break; }
          nested.push(lines[i].replace(/^ {2,4}/, '')); i++;
        }
        if (nested.length) {
          const sub = parseMarkdown(nested.join('\n')).blocks;
          // a plain continuation line joins the item's text; lists and other blocks nest under it
          if (sub[0]?.type === 'p') item.inlines = item.inlines.concat([{ t: 'text', v: ' ' }], sub.shift().inlines);
          item.children = sub;
        }
        items.push(item);
      }
      blocks.push({ type: ordered ? 'ol' : 'ul', items });
      continue;
    }
    // paragraph: consecutive non-blank lines that start nothing else
    const para = [line.trim()];
    i++;
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|>\s?|\s*\|.*\|\s*$|\s*(-{3,}|\*{3,})\s*$)/.test(lines[i]) && !listMarker(lines[i])) para.push(lines[i++].trim());
    blocks.push({ type: 'p', inlines: parseInline(para.join(' ')) });
  }
  return { blocks };
}

/** The headings to list in a table of contents: {level 2 and 3, id, text}. */
export const outline = (blocks) => blocks.filter((b) => b.type === 'heading' && (b.level === 2 || b.level === 3)).map(({ level, id, text }) => ({ level, id, text: text.replace(/[`*]/g, '') }));
