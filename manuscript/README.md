# Manuscript

The working copy of the article. It started on 20 Nov 2026 as a copy of `ms_latex/main_jov.tex` (the JoV version from 20 Aug). `ms_latex/` stays as it was, as the record of that version.

| Path | What goes there |
|---|---|
| `main.tex` | the article |
| `references.bib` | the references (copied from `ms_latex/elife-sample.bib`) |
| `jov.cls`, `jovcite.sty` | the JoV style files |
| `assets/figures/` | figures the article uses (so far only the 23 that `main.tex` includes) |
| `assets_tmp/` | temporary space: candidate figures, drafts, anything not yet in the article |
| `notes/lab_meetings/` | notes and figures for lab meetings |
| `notes/personal/` | your own notes |

Build (BasicTeX):

```bash
cd manuscript && latexmk -pdf main.tex
```

LaTeX build files are ignored by git. `main.pdf` is not ignored, so you can decide whether to commit it.
