/* Convert manuscript/paper.md to manuscript/paper.docx.

   The manuscript is generated Markdown with a small, known subset of syntax: headings,
   paragraphs, bold / italic / code spans, bullet and numbered lists, pipe tables and
   images. This script maps that subset to a Word document:

     - A4, 2.5 cm margins, Times New Roman 11 pt, page numbers in the footer
     - the appendix (wide tables) in a landscape section
     - Unicode sub/superscripts (R², cᵢₖ, Σ⁻¹) become real Word sub/superscripts
     - tables with a repeated header row; figures scaled to the text width

       cd tools/docx_export && npm install && node export_docx.js
*/
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, Footer, HeadingLevel, ImageRun, LevelFormat, Packer,
  PageNumber, PageOrientation, Paragraph, ShadingType, Table, TableCell, TableRow, TextRun,
  VerticalAlign, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const SRC = path.join(ROOT, "manuscript", "paper.md");
const OUT = path.join(ROOT, "manuscript", "paper.docx");
const FONT = "Times New Roman";
const MONO = "Consolas";
const PAGE = { w: 11906, h: 16838, margin: 1418 };           // A4 in DXA, 2.5 cm margins
const WIDTH_PORTRAIT = PAGE.w - 2 * PAGE.margin;              // 9070
const WIDTH_LANDSCAPE = PAGE.h - 2 * PAGE.margin;             // 14002

// Unicode sub/superscripts -> plain character + Word vertical alignment
const SUP = { "²": "2", "³": "3", "¹": "1", "⁻": "−", "ᵀ": "T", "ᵈ": "d" };
const SUB = { "ᵢ": "i", "ₖ": "k", "ⱼ": "j" };

/* ------------------------------------------------------------------ inline markup */
// A bare web address (for example an ORCID iD) becomes a clickable hyperlink.
function inlineRuns(text, base = {}) {
  const parts = text.split(/(https?:\/\/[^\s)\]]+[^\s)\].,;:])/);
  if (parts.length === 1) return styledRuns(text, base);
  return parts.flatMap((part, i) => (i % 2 === 1
    ? [new ExternalHyperlink({ link: part, children: [new TextRun({ text: part, font: FONT, size: base.size,
        color: "0563C1", underline: {} })] })]
    : styledRuns(part, base)));
}

function styledRuns(text, base = {}) {
  const runs = [];
  let bold = false, italic = false, code = false, buf = "";
  const flush = (extra = {}) => {
    if (!buf) return;
    runs.push(new TextRun({ text: buf, font: code ? MONO : FONT, bold: bold || base.bold,
      italics: italic || base.italics, size: code && base.size ? base.size - 1 : base.size, ...extra }));
    buf = "";
  };
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "`") { flush(); code = !code; continue; }
    if (!code && ch === "*") {
      flush();
      if (text[i + 1] === "*") { bold = !bold; i++; } else { italic = !italic; }
      continue;
    }
    if (!code && (SUP[ch] || SUB[ch])) {
      flush();
      buf = SUP[ch] || SUB[ch];
      flush(SUP[ch] ? { superScript: true } : { subScript: true });
      continue;
    }
    buf += ch;
  }
  flush();
  return runs;
}

/* ------------------------------------------------------------------------- tables */
const isNumeric = (s) => /^[−+\-]?\d[\d.,]*(e[−+\-]?\d+)?%?$/i.test(s.trim()) || s.trim() === "";

// Typographic minus for negative numbers and exponents only (never the hyphen in "PCA-80%"),
// and a word joiner after the en dash of a numeric range so that "0.21–0.47" does not wrap.
const cellText = (s) => s
  .replace(/(^|[\s\[(,])-(?=\d)/g, "$1−")
  .replace(/(\de)-(?=\d)/g, "$1−")
  .replace(/(\d)–(?=\d)/g, "$1–⁠");

function buildTable(lines, totalWidth) {
  const rows = lines.filter((l, i) => i !== 1).map((l) => l.replace(/^\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim()));
  const ncol = rows[0].length;
  const size = ncol >= 11 ? 13 : ncol >= 8 ? 15 : ncol >= 6 ? 17 : 19;      // half-points
  // column widths proportional to the longest cell (capped), with a floor
  const cap = ncol >= 9 ? 28 : 42;                            // long text columns wrap; short ones must not
  const len = Array.from({ length: ncol }, (_, j) => Math.min(cap, Math.max(4, ...rows.map((r) => (r[j] || "").length))));
  const numericCol = Array.from({ length: ncol }, (_, j) => rows.slice(1).every((r) => isNumeric(r[j] || "")));
  const weight = len.map((l, j) => (numericCol[j] ? Math.max(l, 5) + 1.5 : l <= 12 ? l + 5.5 : l + 3));
  const sum = weight.reduce((a, b) => a + b, 0);
  const widths = weight.map((w) => Math.floor((w / sum) * totalWidth));
  widths[0] += totalWidth - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (r, ri) => new TableRow({
    tableHeader: ri === 0, cantSplit: true,
    children: Array.from({ length: ncol }, (_, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
      margins: { top: 30, bottom: 30, left: 60, right: 60 },
      shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
      children: [new Paragraph({
        alignment: ri > 0 && numericCol[j] ? AlignmentType.RIGHT : AlignmentType.LEFT,
        spacing: { before: 0, after: 0, line: 240 },
        children: inlineRuns(cellText(r[j] || ""), { size, bold: ri === 0 }),
      })],
    })),
  });
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: widths, rows: rows.map(mk) });
}

/* ------------------------------------------------------------------------- images */
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function buildImage(rel, totalWidthDxa) {
  const file = path.resolve(path.dirname(SRC), rel);
  const { w, h, data } = pngSize(file);
  const maxW = Math.round((totalWidthDxa / 1440) * 96);      // text width in px at 96 dpi
  const maxH = Math.round(8.4 * 96);
  let width = maxW, height = Math.round((maxW * h) / w);
  if (height > maxH) { height = maxH; width = Math.round((maxH * w) / h); }
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
    children: [new ImageRun({ type: "png", data, transformation: { width, height } })] });
}

/* ----------------------------------------------------------------- block structure */
function convert(md) {
  const lines = md.split(/\r?\n/);
  const sections = [{ landscape: false, children: [] }];
  const numbering = [];
  let cur = sections[0];
  let listId = 0;
  let titleBlock = true;                                    // author lines under the title are centred
  const width = () => (cur.landscape ? WIDTH_LANDSCAPE : WIDTH_PORTRAIT);
  const para = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts,
    children: inlineRuns(text, opts.run || {}) });

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim() || /^---+$/.test(line.trim())) continue;

    if (line.startsWith("# ")) {
      cur.children.push(new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER,
        children: inlineRuns(line.slice(2)) }));
    } else if (line.startsWith("## ")) {
      const title = line.slice(3);
      titleBlock = false;
      if (/^Appendix/.test(title)) {                         // wide tables: landscape section
        cur = { landscape: true, children: [] };
        sections.push(cur);
      }
      cur.children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: inlineRuns(title) }));
    } else if (line.startsWith("### ")) {
      cur.children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: inlineRuns(line.slice(4)) }));
    } else if (line.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      i--;
      cur.children.push(buildTable(block, width()));
      cur.children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    } else if (/^!\[.*\]\(.*\)$/.test(line.trim())) {
      cur.children.push(buildImage(line.trim().match(/\((.*)\)$/)[1], width()));
    } else if (/^- /.test(line)) {
      while (i < lines.length && /^- /.test(lines[i])) {
        cur.children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 276 },
          children: inlineRuns(lines[i].slice(2)) }));
        i++;
      }
      i--;
    } else if (/^\d+\. /.test(line)) {
      const ref = `num${++listId}`;                           // a new reference restarts at 1
      numbering.push(ref);
      while (i < lines.length && /^\d+\. /.test(lines[i])) {
        cur.children.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 80, line: 276 },
          children: inlineRuns(lines[i].replace(/^\d+\. /, "")) }));
        i++;
      }
      i--;
    } else if (/^\*(Figure|Table) /.test(line) && line.trim().endsWith("*")) {
      cur.children.push(para(line.trim().slice(1, -1), { alignment: AlignmentType.LEFT, spacing: { after: 200, line: 252 },
        run: { italics: true, size: 19 } }));
    } else if (/^\*\*Table [A-Z]?\d+\./.test(line)) {
      cur.children.push(para(line, { keepNext: true, spacing: { before: 160, after: 80, line: 252 }, run: { size: 20 } }));
    } else {
      cur.children.push(para(line, titleBlock ? { alignment: AlignmentType.CENTER, spacing: { after: 80, line: 264 } }
        : { alignment: AlignmentType.JUSTIFIED }));
    }
  }
  return { sections, numbering };
}

/* -------------------------------------------------------------------- the document */
const { sections, numbering } = convert(fs.readFileSync(SRC, "utf8"));
const heading = (size, before, after) => ({ run: { font: FONT, size, bold: true, color: "000000" },
  paragraph: { spacing: { before, after }, keepNext: true } });
const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] });

const doc = new Document({
  title: "Which Similarity Measure for Similar-Learner Retrieval?",
  description: "Manuscript draft generated from manuscript/paper.md",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
      title: { run: { font: FONT, size: 32, bold: true, color: "000000" }, paragraph: { spacing: { before: 0, after: 240 }, keepNext: true } },
      heading1: heading(26, 320, 140),
      heading2: heading(23, 240, 100),
    },
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 500, hanging: 280 } } } }] },
      ...numbering.map((reference) => ({ reference, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 320 } } } }] })),
    ],
  },
  sections: sections.map((s) => ({
    properties: { page: { size: { width: PAGE.w, height: PAGE.h, orientation: s.landscape ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT },
      margin: { top: PAGE.margin, bottom: PAGE.margin, left: PAGE.margin, right: PAGE.margin } } },
    footers: { default: footer() },
    children: s.children,
  })),
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  const n = sections.reduce((a, s) => a + s.children.length, 0);
  console.log(`wrote ${path.relative(ROOT, OUT)}  (${(buf.length / 1e6).toFixed(1)} MB, ${n} blocks, ${sections.length} sections)`);
});
