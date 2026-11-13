// Reads HTML on stdin, writes Turndown 7.2.5 Markdown (library defaults) on stdout. With --readability, the page is
// first reduced by @mozilla/readability 0.6.0 (jsdom), then converted. Usage: node turndown_cli.js [--readability] URL
const TurndownService = require('turndown');
const { JSDOM } = require('jsdom');
const { Readability } = require('@mozilla/readability');
let html = '';
process.stdin.on('data', d => html += d);
process.stdin.on('end', () => {
  const args = process.argv.slice(2);
  const url = args.find(a => !a.startsWith('--')) || 'http://127.0.0.1/';
  let body = html;
  if (args.includes('--readability')) {
    const dom = new JSDOM(html, { url });
    const art = new Readability(dom.window.document).parse();
    body = art && art.content ? art.content : '';
  }
  process.stdout.write(new TurndownService().turndown(body));
});
