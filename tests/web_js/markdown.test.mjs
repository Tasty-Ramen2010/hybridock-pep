import test from 'node:test';
import assert from 'node:assert/strict';
import { parseMarkdown, parseInline, safeHref, slugify, outline } from '../../src/hybridock_pep/web/static/js/app/markdown.mjs';

const first = (md) => parseMarkdown(md).blocks[0];

test('headings get an id from their text, or from {#custom-id}, and ids never repeat', () => {
  const { blocks } = parseMarkdown('# Title\n\n## Get started {#start}\n\n## Get started\n\n## Get started');
  assert.deepEqual(blocks.map((b) => b.id), ['title', 'start', 'get-started', 'get-started-2']);
  assert.equal(blocks[1].text, 'Get started');
  assert.equal(slugify('Reading your `ΔG` result!'), 'reading-your-g-result');
});

test('inline: bold, italic, code, links and images, nested and in order', () => {
  const t = parseInline('Use **the `--seed` flag** or *not*, see [the guide](#faq) ![shot](img/a.png)');
  assert.deepEqual(t.map((x) => x.t), ['text', 'strong', 'text', 'em', 'text', 'link', 'text', 'img']);
  assert.equal(t[1].c[1].t, 'code');
  assert.equal(t[5].href, '#faq');
  assert.deepEqual([t[7].alt, t[7].src], ['shot', 'img/a.png']);
});

test('markup characters inside code stay literal', () => {
  assert.deepEqual(parseInline('`a **b** [c](d)`'), [{ t: 'code', v: 'a **b** [c](d)' }]);
});

test('raw HTML is never interpreted: it stays as text', () => {
  const p = first('<script>alert(1)</script> and <img src=x onerror=alert(1)>');
  assert.equal(p.type, 'p');
  assert.ok(p.inlines.every((n) => n.t === 'text'));
  assert.ok(p.inlines.map((n) => n.v).join('').includes('<script>'));
});

test('only http(s), mailto, #anchors and relative links are followed', () => {
  assert.equal(safeHref('https://example.org/x'), 'https://example.org/x');
  assert.equal(safeHref('#faq'), '#faq');
  assert.equal(safeHref('img/a.png'), 'img/a.png');
  assert.equal(safeHref('javascript:alert(1)'), null);
  assert.equal(safeHref('data:text/html;base64,AAAA'), null);
  assert.equal(safeHref('JaVaScRiPt:alert(1)'), null);
});

test('lists: bullets, numbers, and one nested level', () => {
  const { blocks } = parseMarkdown('- one\n- two\n  - inner a\n  - inner b\n- three\n\n1. first\n2. second');
  assert.equal(blocks[0].type, 'ul');
  assert.equal(blocks[0].items.length, 3);
  assert.equal(blocks[0].items[1].children[0].type, 'ul');
  assert.equal(blocks[0].items[1].children[0].items.length, 2);
  assert.equal(blocks[1].type, 'ol');
  assert.equal(blocks[1].items.length, 2);
});

test('a list item can continue on the next indented line', () => {
  const b = first('1. Press the button\n   and wait for the result\n2. Done');
  assert.equal(b.items.length, 2);
  assert.match(b.items[0].inlines.map((n) => n.v || '').join(''), /Press the button and wait/);
});

test('fenced code keeps every line verbatim, including blank ones and markup', () => {
  const b = first('```bash\nhybridock-pep serve\n\n# comment **not bold**\n```');
  assert.deepEqual([b.type, b.lang], ['code', 'bash']);
  assert.equal(b.text, 'hybridock-pep serve\n\n# comment **not bold**');
});

test('tables: header, rule, rows, and inline formatting in cells', () => {
  const b = first('| Flag | Meaning |\n| --- | --- |\n| `--seed` | **random** seed |\n| `--box` | size |');
  assert.equal(b.type, 'table');
  assert.equal(b.head.length, 2);
  assert.equal(b.rows.length, 2);
  assert.equal(b.rows[0][0][0].t, 'code');
  assert.equal(b.rows[0][1][0].t, 'strong');
});

test('a quote becomes a callout holding its own blocks', () => {
  const b = first('> **Tip:** type the letters.\n> Second line.');
  assert.equal(b.type, 'quote');
  assert.equal(b.blocks[0].type, 'p');
  assert.equal(b.blocks[0].inlines[0].t, 'strong');
});

test('paragraphs join their lines; a heading or list ends one', () => {
  const { blocks } = parseMarkdown('line one\nline two\n## Next\n- item');
  assert.equal(blocks.map((b) => b.type).join(), 'p,heading,ul');
  assert.equal(blocks[0].inlines[0].v, 'line one line two');
});

test('the outline lists level 2 and 3 headings only', () => {
  const { blocks } = parseMarkdown('# Title\n## A\n### B\n#### C\n## D');
  assert.deepEqual(outline(blocks).map((o) => [o.level, o.text]), [[2, 'A'], [3, 'B'], [2, 'D']]);
});

test('Windows line endings and an empty document are fine', () => {
  assert.equal(parseMarkdown('# A\r\n\r\ntext\r\n').blocks.length, 2);
  assert.deepEqual(parseMarkdown('').blocks, []);
});
