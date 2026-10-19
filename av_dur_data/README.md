# Audiovisual duration estimation: trial-level data

One row per trial. All durations are in seconds.

## Files

| File | Experiment |
|---|---|
| `all_visual.csv` | Unimodal visual duration discrimination |
| `all_auditory.csv` | Unimodal auditory duration discrimination (two background-noise levels) |
| `all_crossmodal.csv` | Cross-modal comparison: auditory test vs. visual standard |
| `all_bimodal.csv` | Main audiovisual cue-conflict experiment |
| `per_participant/Pxx_<experiment>.csv` | The same data, split by participant |

## Participants

Thirteen participants were recruited (P01–P13). The eleven analyzed in the paper are **P01–P04 and P07–P13**, and only they appear in these files. Two were excluded:

- **P05**: their cross-modal bias was too large to calibrate (about 560 ms toward the visual duration in the high-noise auditory condition), so they did not complete the main experiment.
- **P06**: excluded because of a procedural difference in an early version of the main experiment.

## Key variables

All tasks are two-interval forced choice ("Which interval was longer?").

- `order`: the interval that held the **test** stimulus (1 = first, 2 = second). The standard was in the other interval.
- `responses`: the interval the participant judged longer (1 = first, left arrow; 2 = second, right arrow).
- **Test judged longer** = `responses == order`. This is the response variable for psychometric fits.
- `is_correct`: whether the response was correct, given the nominal test and standard durations.
- `standardDur`, `testDurS`: standard and test durations. `deltaDurS = testDurS − standardDur`, and `delta_dur_percents = deltaDurS / standardDur`.
- `current_stair`: the staircase that set the test duration on that trial. `lapse_rate` marks easy trials with a large test–standard difference, included to estimate the lapse rate.
- `audNoise`: amplitude of the background noise relative to the auditory signal. 0.1 = low noise and 1.2 = high noise. The value 99 in the visual file means "not applicable".

### Cross-modal task

- The test is the **auditory** stimulus and the standard is the **visual** stimulus (500 ms).
- `order` = 1 means auditory first (A→V), and `order` = 2 means visual first (V→A).
- Order alternated across mini-blocks of 10 trials. Each mini-block was announced on screen before it started.

### Main (bimodal) experiment

- `conflictDur`: visual minus auditory duration in the **standard** interval. It has seven nominal levels: −0.25, −0.167, −0.083, 0, +0.083, +0.167 and +0.25 s.
  - Stored values reflect display-frame timing and can differ from the nominal level by up to 4 ms.
  - To group trials by condition, snap each value to the nearest nominal level.
- `VisualPSE`: the participant-specific bias correction added to the visual component of both intervals. It was estimated from the cross-modal task.
- **Excluded trials.** Sessions run before 11 June 2025 mis-timed the visual test whenever it was presented in the second interval (`order` = 2).
  - As in our analyses, this file includes only trials in which the logged visual durations of both intervals (`recordedDurVisualTest`, `recordedDurVisualStandard`) were within 17 ms of their intended values.
  - Participants affected by this ran additional sessions to replace the excluded trials. This is why trial counts differ across participants (nominally 2,156).

### Unused columns in the auditory file

The auditory task shared presentation code with the audiovisual tasks, but no visual stimulus and no post-cue were displayed in it. These columns therefore carry no information in `all_auditory.csv`:

- `conflictDur` (always 0)
- `modalityPostCue` (assigned at random but never shown)
- `VisualPSE`
- `recorded*Visual*`

