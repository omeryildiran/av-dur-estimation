# Audiovisual duration experiment, version 2

A self-contained rewrite of the four experiments (unimodal auditory, unimodal visual, cross-modal, main audiovisual). It uses the same stimuli and procedures as the 2025 data collection ("v1"), with every known v1 problem fixed. One launcher runs everything, and the main experiment's bias correction is fitted automatically.

## Running a participant

```bash
cd av_dur_experiment_v2
python run_experiment.py            # in the PsychoPy environment (PsychoPy 2024.2, Python 3.10)
```

Enter the participant ID (e.g. `P14`) and press the task buttons in order:

1. **Unimodal auditory** (308 trials)
2. **Unimodal visual** (154 trials)
3. **Cross-modal** (388 trials)
4. **Main experiment**: three blocks of 616–924 trials, each about an hour.
   - The launcher picks the participant's next block. Block order is counterbalanced by participant number over all six orders.
   - Before the first block, the launcher fits the cross-modal bias correction, shows the values and any warnings, and asks you to confirm.
   - Later blocks reuse that same bias file.

Other launcher options:

- **Practice** runs a short version of a task (easy trials with feedback). Practice data are saved separately and never analysed.
- **Simulate** runs a task with a simulated observer and no window, to test the setup on any computer.
- **Text menu:** if Tk is missing, `python run_experiment.py --cli` gives a text menu.
- **Single task:** one task can be started directly, e.g. `python -m avdur.run_task --task crossmodal --pid P14`.

## Before the first participant

- **Display.** `DEFAULT_MONITOR = "v1_profile"` reproduces the v1 pixel geometry. Every v1 script used MacBook-Air numbers, which on the same lab display give a disk radius of 42.7 px. For a true 3° disk on a different display, add a profile with measured width, height and distance.
- **Refresh rate.** Durations are computed at 120 Hz. The session stops if the measured rate is more than 1 Hz off, whereas v1 silently used whatever rate it measured.
- **Audio.** The PsychToolbox audio backend must load. On the laptop's `psychopy310` environment it currently fails (`PsychHID … libusb`); `conda install -n psychopy310 -c conda-forge libusb` should fix it.
- **Audiovisual sync.** Measure it once with a photodiode on the disk and a microphone at the headphones. Put the audio-minus-video delay in `AUDIO_ONSET_OFFSET_S`.
- **Tests.** Run `python tests/test_pipeline.py`. It runs two simulated participants through everything and checks the design, the timing, the bias recovery and the audio levels.

## What changed from v1, and why

| v1 problem | v2 |
|---|---|
| Second-interval visual flash anchored to the first flash's offset, so it was displaced from its sound by ½ conflict + ½ bias correction | Each flash is centred on its own sound and extended symmetrically ([avdur/schedule.py](avdur/schedule.py)). The tests check the centring for every trial. |
| Visual test mis-timed when it came second (code versions of 2–11 June 2025) | Same fix; the logged flip times are checked against the intended durations. |
| Order drawn by a coin flip on each trial; make-up sessions unbalanced | Exactly balanced test-first and test-second trials within every staircase and condition ([avdur/design.py](avdur/design.py)). |
| Two staircases shared by all conditions of a session; conditions in a fixed cycle | Two staircases per condition, as the paper describes; all trials shuffled. |
| Cross-modal staircase `1U1D` moved away from threshold | Rule names can only express converging staircases; that slot is now a 1-down-1-up starting at the top. |
| Bias correction fitted in a notebook and typed in by hand (dt's low-noise value was wrong; values left over from earlier participants; one make-up with the two noise levels swapped) | Fitted from the participant's own cross-modal data, confirmed in the launcher, saved as a file, logged in every row, and reused for all main blocks. |
| Fit with lapse ≤ 0.2 and μ bounds that some fits hit | Lognormal fit with lapse ≤ 0.1, multi-start, a bootstrap CI and quality warnings. v1's linear method is reported alongside for comparison. |
| Sound started from inside the frame loop, not scheduled | Sound scheduled on the trial's first flip: `play(when=win.getFutureFlipTime(clock='ptb'))`. |
| Logged loop times, not flip times; unimodal visual timing not logged at all | Actual flip timestamps for every onset and offset in every task, plus dropped frames and the frame-0 flip error. |
| Peak normalisation gave ±0.75 dB of random level jitter per trial; cross-modal tests 0.4–1.3 dB louder than main-experiment tests | Fixed gains calibrated to v1's average level and SNR (`AUDIO_LEVEL_MODE = "fixed"`; `"peak"` reproduces v1 exactly). |
| `prefs.general['audioLatencyMode']` (ignored by PsychoPy) | `prefs.hardware['audioLatencyMode']`, set to PsychoPy's default (3), which is what v1 effectively used. |
| `visualStim.lineWidht = 5` typo | `DISK_LINE_WIDTH_PX = 1.5`, PsychoPy's default, i.e. what v1 participants saw. |
| Session order described as counterbalanced, though B1 came last for 9 of 10 participants | Latin-square block order by participant number (`BLOCK_ORDERS`). |
| Settings spread over many scripts; testing edits left in place | One settings file ([experiment_settings.py](experiment_settings.py)); code version (git commit plus source checksum) and every setting saved with each session. |

Kept from v1:
- standard 500 ms;
- noise levels 0.1 and 1.2;
- signal band-pass 150–775 Hz and background 10–610 Hz;
- burst intensity 5 with 5 ms ramps;
- pre-, inter- and post-stimulus durations;
- staircase steps (0.16, halved at the first two reversals);
- lapse trials;
- trial counts;
- breaks every 30 trials;
- screens and response keys.

The filter cutoffs are those of every real data collection since commit `c7ff22e1` (24 April 2025). Earlier values were pilot tuning.

## Data layout

```
data/<PID>/<task>/<PID>_<task>[_<block>]_<date-time>.csv    one row per trial, written to disk after every trial
data/<PID>/<task>/<same name>.json                          session metadata: settings, code version, frame rate,
                                                            bias file, seed, completed flag
data/<PID>/bias/<PID>_bias_<date-time>.json                 confirmed cross-modal corrections (+ v1-method values)
data/<PID>/practice/...                                     practice runs
```

The column names follow v1 (`testDurS`, `audNoise`, `order`, `responses`, `conflictDur`, `VisualPSE`, `recordedDurVisualTest`, …). New columns:
- the frame schedule (`audio_interval1`, `visual_interval1`, …);
- intended visual durations;
- `chose_test`;
- `n_dropped_frames`, `flip0_error_s` and `audio_onset_minus_flip0_s`;
- the audio seed and level mode.

An interrupted session keeps its trials; its metadata says `completed: false`, and the analysis uses completed sessions only.

## Analysis

The notebooks in [analysis/](analysis/) use the repository's `plot_style.py`:

- `01_unimodal_fits.ipynb`: auditory and visual psychometric functions, σ (and σ/√2 for the models), a free-μ check, and a visual timing check.
- `02_crossmodal_bias.ipynb`: cross-modal fits, the corrections with CIs and the v1 method, and a check that the main blocks used the confirmed values.
- `03_main_experiment_check.ipynb`: order balance, timing and centring checks, PSE vs. conflict per noise level, and PSE by presentation order.

Until real data exist, they show the simulated participants written by `tests/test_pipeline.py` to `data_simulated/`. `tests/calibrate_audio_levels.py` reproduces the fixed-level audio constants.
