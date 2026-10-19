import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/ofy204/Documents/my_repos/av-dur-estimation";
const buildDir = path.join(workspaceDir, ".codex-build/social-slides");
const outputDir = path.join(workspaceDir, "output/social-slides");
const skillDir = "/Users/ofy204/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const runtimePython = "/Users/ofy204/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const font = resolvePresentationFont();
const W = 1080;
const H = 1080;
const C = {
  paper: "#F6F0E6",
  ink: "#142A36",
  muted: "#5B6870",
  red: "#D84A3A",
  blue: "#267EB5",
  teal: "#287C78",
  white: "#FFFFFF",
  paleBlue: "#DCECF3",
  paleRed: "#F2DDD7",
  line: "#CFC8BC",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addRect(slide, left, top, width, height, fill, radius = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    borderRadius: radius || undefined,
    position: { left, top, width, height },
    fill,
    line: { fill: "none", width: 0 },
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: font,
    fontSize: opts.fontSize ?? 42,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    autoFit: "none",
    horizontalAlignment: opts.align ?? "left",
    verticalAlignment: opts.vAlign ?? "top",
  };
  return box;
}

function addImage(slide, bytes, left, top, width, height, alt, crop) {
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: crop ? "cover" : "contain",
    ...(crop ? { crop } : {}),
    position: { left, top, width, height },
  });
}

function baseSlide(number, accent = C.blue) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addRect(slide, 0, 0, W, 18, accent);
  addText(slide, `${number}/6`, 820, 1008, 180, 34, {
    fontSize: 22, color: C.muted, align: "right", vAlign: "middle",
  });
  return slide;
}

const taskImg = new Uint8Array(await fs.readFile(path.join(workspaceDir, "ms_latex/assets/figures/bimodal_exp_scheme.png")));
const resultImg = new Uint8Array(await fs.readFile(path.join(buildDir, "assets/result.png")));
const modelsImg = new Uint8Array(await fs.readFile(path.join(buildDir, "assets/models.png")));

// 1. Hook
{
  const slide = baseSlide(1, C.red);
  addText(slide, "00:00.500", 74, 100, 930, 150, {
    fontSize: 112, bold: true, color: C.blue, vAlign: "middle",
  });
  addRect(slide, 76, 286, 140, 12, C.red, 6);
  addText(slide, "When sight and sound\ndisagree about time", 72, 338, 920, 250, {
    fontSize: 72, bold: true,
  });
  addText(slide, "Can visual duration change how long a sound seems?", 76, 690, 810, 130, {
    fontSize: 40, color: C.muted,
  });
  addText(slide, "Duration judgments with conflicting audiovisual cues", 76, 930, 850, 38, {
    fontSize: 24, color: C.muted,
  });
  slide.speakerNotes.textFrame.setText("Source: Yildiran, Ni, and Landy, Duration judgments with conflicting audiovisual cues, bioRxiv preprint.");
}

// 2. Scientific question
{
  const slide = baseSlide(2, C.teal);
  addText(slide, "The question", 72, 72, 850, 88, { fontSize: 62, bold: true });
  addText(slide, "Small conflict", 84, 250, 390, 54, { fontSize: 34, bold: true, color: C.teal });
  addText(slide, "The senses may belong\nto one event", 84, 324, 410, 132, { fontSize: 42, bold: true });
  addText(slide, "Integrate", 84, 500, 400, 88, { fontSize: 68, bold: true, color: C.teal });
  addRect(slide, 538, 212, 3, 500, C.line);
  addText(slide, "Large conflict", 600, 250, 390, 54, { fontSize: 34, bold: true, color: C.red });
  addText(slide, "Do the senses still seem\nto share a cause?", 600, 324, 410, 132, { fontSize: 42, bold: true });
  addText(slide, "Segregate?", 600, 500, 400, 88, { fontSize: 68, bold: true, color: C.red });
  addText(slide, "Causal inference predicts less integration as the conflict grows.", 84, 794, 890, 120, {
    fontSize: 40, color: C.muted,
  });
  slide.speakerNotes.textFrame.setText("Source: Introduction and causal-inference model sections of the manuscript.");
}

// 3. Experiment
{
  const slide = baseSlide(3, C.blue);
  addText(slide, "Audiovisual duration conflict", 72, 70, 940, 85, { fontSize: 60, bold: true });
  addText(slide, "Participants judged the sound. The visual duration varied.", 74, 164, 900, 70, {
    fontSize: 34, color: C.muted,
  });
  addRect(slide, 62, 270, 956, 522, C.white, 22);
  addImage(slide, taskImg, 88, 300, 904, 465, "Schematic of the bimodal audiovisual conflict experiment");
  addText(slide, "7 conflict levels", 78, 844, 330, 48, { fontSize: 34, bold: true, color: C.red });
  addText(slide, "−250 to +250 ms", 78, 900, 380, 60, { fontSize: 44, bold: true });
  addText(slide, "2 auditory noise levels", 600, 844, 400, 48, { fontSize: 34, bold: true, color: C.blue, align: "right" });
  addText(slide, "clear sound  /  noisy sound", 560, 902, 440, 54, { fontSize: 34, color: C.ink, align: "right" });
  slide.speakerNotes.textFrame.setText("Source: Manuscript Figure: Audiovisual conflict in the bimodal experimental task. Seven conflicts from -250 to +250 ms and two auditory-noise conditions.");
}

// 4. Behavioral finding
{
  const slide = baseSlide(4, C.red);
  addText(slide, "Vision pulled auditory judgments", 72, 70, 940, 85, { fontSize: 58, bold: true });
  addText(slide, "The pull grew with cue conflict and became much stronger when sound was noisy.", 74, 158, 915, 92, {
    fontSize: 35, color: C.muted,
  });
  addRect(slide, 60, 274, 960, 610, C.white, 22);
  addImage(slide, resultImg, 78, 298, 924, 566, "PSE shifts across audiovisual conflicts under low and high auditory noise");
  addText(slide, "Less reliable sound gave vision more influence.", 76, 922, 900, 56, {
    fontSize: 37, bold: true, color: C.blue,
  });
  slide.speakerNotes.textFrame.setText("Source: Manuscript Figure: Effect of audiovisual conflict on perceived auditory duration. Low-noise mean slope 0.148 ms/ms; high-noise mean slope 0.505 ms/ms.");
}

// 5. Model overlap
{
  const slide = baseSlide(5, C.teal);
  addText(slide, "Three models, almost one prediction", 72, 70, 940, 142, { fontSize: 54, bold: true });
  addText(slide, "Forced fusion, causal inference, and cue switching produced nearly overlapping shifts.", 74, 210, 920, 92, {
    fontSize: 35, color: C.muted,
  });
  addRect(slide, 60, 322, 960, 550, C.white, 22);
  addImage(slide, modelsImg, 74, 338, 930, 512, "Measured PSE shifts and predictions from three computational models");
  addText(slide, "The predicted causal-inference bend was weak within the tested range.", 76, 904, 920, 78, {
    fontSize: 36, bold: true, color: C.teal,
  });
  slide.speakerNotes.textFrame.setText("Source: Manuscript Figure: Measured and predicted PSE shifts. All three models produce overlapping, approximately linear predictions across the tested conflict range.");
}

// 6. Takeaway
{
  const slide = baseSlide(6, C.blue);
  addText(slide, "What did the models say?", 72, 70, 940, 85, { fontSize: 60, bold: true });
  addText(slide, "8 / 11", 72, 220, 430, 150, { fontSize: 126, bold: true, color: C.blue });
  addText(slide, "participants were best fit by\nprobabilistic cue switching", 82, 385, 600, 130, {
    fontSize: 39, bold: true,
  });
  addRect(slide, 72, 570, 930, 3, C.line);
  addText(slide, "But the differences among models were small.", 72, 630, 920, 70, {
    fontSize: 43, bold: true, color: C.red,
  });
  addText(slide, "High sensory noise made forced fusion and causal inference especially difficult to tell apart.", 72, 728, 900, 120, {
    fontSize: 38, color: C.ink,
  });
  addText(slide, "No clear breakdown of integration within the tested conflict range", 72, 902, 900, 58, {
    fontSize: 32, bold: true, color: C.teal,
  });
  slide.speakerNotes.textFrame.setText("Source: Model fitting, model comparison, and conclusions sections of the manuscript. Cue switching had the lowest AIC for 8 of 11 participants, while model-performance differences remained small.");
}

const draftPath = path.join(buildDir, "social-slides-draft.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draftPath);

for (let i = 0; i < presentation.slides.items.length; i++) {
  const preview = await presentation.export({ slide: presentation.slides.items[i], format: "png", scale: 1 });
  const outPath = path.join(outputDir, `av-duration-${String(i + 1).padStart(2, "0")}.png`);
  await fs.writeFile(outPath, new Uint8Array(await preview.arrayBuffer()));
}

const finalPath = path.join(outputDir, "av-duration-social-slides-v2.pptx");
await finalizePresentation({
  workspaceDir,
  candidatePath: draftPath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "10287000,10287000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "av-duration-social-slides-v2.validation.json"),
});

console.log(JSON.stringify({ font, finalPath, outputDir }, null, 2));
