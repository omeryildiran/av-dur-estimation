const fs = require("fs");
const path = require("path");
const sharp = require("/Users/ofy204/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");

const WIDTH = 1494;
const HEIGHT = 500;
const OUT_SVG = path.join(__dirname, "crossmodalTimeline.svg");
const OUT_PNG = path.join(__dirname, "crossmodalTimeline.png");

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random) {
  const u = Math.max(random(), 1e-9);
  const v = Math.max(random(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function pointsToPath(points) {
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
}

function noisePath(x0, x1, y, amp, n, seed) {
  const random = mulberry32(seed);
  const raw = Array.from({ length: n }, () => normal(random));
  const smoothed = raw.map((_, i) => {
    let sum = 0;
    let count = 0;
    for (let k = -2; k <= 2; k += 1) {
      const idx = i + k;
      if (idx >= 0 && idx < raw.length) {
        sum += raw[idx];
        count += 1;
      }
    }
    return sum / count;
  });
  const fadeN = Math.max(3, Math.floor(n / 14));
  const pts = smoothed.map((value, i) => {
    let fade = 1;
    if (i < fadeN) fade = 0.05 + (0.95 * i) / fadeN;
    if (i > n - fadeN - 1) fade = 0.05 + (0.95 * (n - 1 - i)) / fadeN;
    const x = x0 + ((x1 - x0) * i) / (n - 1);
    return [x, y + value * amp * fade];
  });
  return pointsToPath(pts);
}

function circle(cx, cy, r, fill, stroke = "#5f696e", strokeWidth = 3) {
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
}

function speaker(x, y) {
  const body = [
    [x - 24, y - 11],
    [x - 10, y - 11],
    [x + 4, y - 24],
    [x + 4, y + 24],
    [x - 10, y + 11],
    [x - 24, y + 11],
  ]
    .map((p) => p.join(","))
    .join(" ");
  return `
    <polygon points="${body}" fill="#101820"/>
    <path d="M${x + 7} ${y - 18} Q${x + 30} ${y} ${x + 7} ${y + 18}" fill="none" stroke="#101820" stroke-width="2.2"/>
    <path d="M${x + 13} ${y - 29} Q${x + 51} ${y} ${x + 13} ${y + 29}" fill="none" stroke="#101820" stroke-width="2.2"/>`;
}

function eye(x, y) {
  return `
    <path d="M${x - 35} ${y} Q${x} ${y - 24} ${x + 35} ${y} Q${x} ${y + 24} ${x - 35} ${y} Z" fill="white" stroke="#4a555c" stroke-width="3"/>
    <circle cx="${x}" cy="${y}" r="10" fill="#2b2b2b"/>`;
}

function bracket(x0, x1, y, label) {
  return `
    <path d="M${x0} ${y} L${x1} ${y} M${x0} ${y} L${x0} ${y - 18} M${x1} ${y} L${x1} ${y - 18}" fill="none" stroke="#697176" stroke-width="2.2"/>
    <text x="${(x0 + x1) / 2}" y="${y + 38}" text-anchor="middle" class="small">${label}</text>`;
}

async function main() {
  const bounds = [386, 674, 936, 1222];
  const centers = [276, 530, 804, 1078, 1324];
  const yAudio = 148;
  const yVisual = 286;
  const yLabel = 70;
  const yDuration = 370;
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: #444; }
    .small { font-size: 30px; }
    .top { font-size: 30px; }
    .label { font-size: 30px; fill: #111; }
    .duration { font-size: 30px; fill: #444; }
  </style>
  <rect width="100%" height="100%" fill="white"/>
  ${bounds.map((x) => `<line x1="${x}" y1="52" x2="${x}" y2="338" stroke="#999" stroke-width="2" stroke-dasharray="8 12"/>`).join("\n  ")}
  ${["pre", "interval 1", "ISI", "interval 2", "post"].map((label, i) => `<text x="${centers[i]}" y="32" text-anchor="middle" class="top">${label}</text>`).join("\n  ")}
  ${speaker(82, yAudio)}
  ${eye(82, yVisual)}
  <line x1="164" y1="${yAudio}" x2="1430" y2="${yAudio}" stroke="#4d565b" stroke-width="3"/>
  <path d="${noisePath(164, 1430, yAudio, 21, 1000, 41)}" fill="none" stroke="#5a6267" stroke-width="1.4" stroke-linecap="round"/>
  <path d="${noisePath(972, 1180, yAudio, 48, 280, 81)}" fill="none" stroke="#111" stroke-width="2.4" stroke-linecap="round"/>
  <line x1="164" y1="${yVisual}" x2="1430" y2="${yVisual}" stroke="#4d565b" stroke-width="3"/>
  ${centers.map((x) => circle(x, yVisual, 32, "white")).join("\n  ")}
  ${circle(530, yVisual, 46, "#050505", "#050505", 2)}
  <text x="530" y="${yLabel}" text-anchor="middle" dominant-baseline="middle" class="label">standard</text>
  <text x="1078" y="${yLabel}" text-anchor="middle" dominant-baseline="middle" class="label">test</text>
  <text x="530" y="${yDuration}" text-anchor="middle" dominant-baseline="middle" class="duration">500 ms</text>
  <text x="1078" y="${yDuration}" text-anchor="middle" dominant-baseline="middle" class="duration">variable</text>
  ${bracket(bounds[0], bounds[1], 416, "visual standard")}
  ${bracket(bounds[1], bounds[2], 416, "ISI")}
  ${bracket(bounds[2], bounds[3], 416, "auditory test")}
</svg>`;

  fs.writeFileSync(OUT_SVG, svg);
  await sharp(Buffer.from(svg)).png().toFile(OUT_PNG);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
