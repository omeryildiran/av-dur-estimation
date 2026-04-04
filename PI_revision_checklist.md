# PI Revision Checklist

## Done in `av_dur_article.tex`

- Normalized manuscript caption references from `(A)/(B)` style to `A/B` in edited captions.
- Reworded several captions to use consistent condition names like `low-noise auditory` and `high-noise auditory`.
- Updated psychometric caption language to use response-style phrasing such as `p("test longer")` / `p("auditory test longer")`.
- Replaced forced `[H]` figure placement with `[tbp]` for the main figure environments that were likely causing awkward section-header/figure ordering and blank-space issues.
- Dropped the appendix subsection on the Gaussian sensory-noise model and log-linear mismatch observer.
- Removed the `Log-Linear Mismatch` row from the appendix parameter table.
- Updated the parameter-recovery caption so it no longer depends on row labels `A-E`.

## Still needs figure-file editing or regeneration

- Figure 1: fix panel-label formatting and alignment; verify whether any green waveform appears outside the intended stimulus durations.
- Figure 2: axis-label wording, capitalization, and legend naming; consider linear y-axis if desired by PI.
- Figure 4: x-axis wording (`low-noise`, `high-noise`), y-axis wording with quotation marks, and `vs.` punctuation.
- Figure 6: keep panel titles as condition names, but change y-axis wording to `p("Test longer")`.
- Figure 7: likely keep current panel titles, but panel-label formatting may still need standardization.
- Figure 9: panel-title capitalization/alignment; repeated vertical labels; `across participants` lower-case; `Model type`; add hyphen in `sensory-noise`.
- Figure 10: likely remove column titles and `A/B` labels from the artwork.
- Figures 11 and 12: convert y-axis units from `s` to `ms`; drop log-linear mismatch if it still appears in those figures.
- Figure 13: restructure headers so low/high auditory-noise titles span the two-column groups; drop `A/B` subtitles from the artwork.
- Figure 14: panel-title wording should match `Low auditory noise` / `High auditory noise`; scientific check still needed on the `sqrt(2)` question.
- Figure 15: remove row labels `A-E`; enlarge row titles, `Estimated value`, `Ground-truth value`, and parameter headers; add missing parameter names above columns 5 and 6.
- Figure 16: delete the left table if it is just the right table multiplied by 100.

## One unresolved scientific check

- Figure 14 / optimal-integration comparison: confirm whether both `\sigma_{opt}` and `\sigma_{AV}` were derived with the `\sqrt{2}` factor appropriate for a 2AFC comparison task.
