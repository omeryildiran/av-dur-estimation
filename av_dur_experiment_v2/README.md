# Audiovisual duration experiment, version 2

A self-contained rewrite of the four experiments (unimodal auditory, unimodal visual, cross-modal, main audiovisual). It uses the same stimuli and procedures as the 2025 data collection ("v1"), with every known v1 problem fixed. One launcher runs everything, and the main experiment's bias correction is fitted automatically.

## Running a participant

```bash
cd av_dur_experiment_v2
python run_experiment.py            # in the PsychoPy environment (PsychoPy 2024.2, Python 3.10)
```

Choose the monitor profile, enter the participant ID (`P14`; `p14` or `14` also become `P14`), and press the task buttons in order:

1. **Unimodal auditory** (308 trials)
2. **Unimodal visual** (154 trials)
3. **Cross-modal** (388 trials)
4. **Main experiment**: three blocks of 616–924 trials, each about an hour.
   - The launcher picks the participant's next block. Block order is counterbalanced by participant number over all six orders.
   - Before the first block, the launcher fits the cross-modal bias correction, shows the values and any warnings, and asks you to confirm.
   - Later blocks reuse that same bias file.

### Blocks 1–3 in one session

The **1-3 in one session** button (`python -m avdur.run_blocks --pid P14`) runs the unimodal auditory, unimodal visual and cross-modal blocks one after another in a single window. That is about 31 minutes of trials, or 40 with breaks. With **Practice** ticked, a short practice comes before each block.

- **Saving and resuming:** each block is saved as its own session, exactly as from its own button. Blocks already completed are skipped, so after a stop, press the button again to run the rest.
- **Live progress:** every 30 trials the terminal shows the trials done and time left, easy-trial accuracy, median RT, dropped frames and every staircase's current level.
- **Fit after each block:** while the participant rests (at least `BLOCK_REST_MIN_S`, 10 s), the block is fitted in a separate process, so trial timing is unaffected. The terminal then shows σ, PSE and lapse rate per condition next to v1's range of σ, plus the visual correction for the cross-modal block. A plot is saved to `data/<PID>/fits/`, and all plots open at the end.
- **Checks (`SCREENING_CHECKS`):**
  - low-noise auditory: σ ≤ 0.6, and easy trials at least 75% correct;
  - visual: σ ≤ 2.0, and easy trials at least 70% correct.
  - Plots show P(test judged longer) against test duration in ms.
  - These are about 1.5 times beyond v1's worst included participant.
  - High noise and the cross-modal block are reported but not checked. In v1, 2 of 11 participants were at the σ bound (3.0) at high noise.
  - The checks never look at the PSE or the bias.
- **Fit viewer:** if the PC has a second screen, the launcher opens a "Fits" window on the screen that isn't the participant's, before the experiment window. It shows each new plot as soon as it is saved: after every block of the combined session, after any single task, and after each main block, where it plots PSE shift against conflict per noise level with the slope. It updates without taking the focus from the experiment window, and the arrow buttons go back to older plots. With a single screen, the plots open at the end instead.
- **If a check fails:** the participant sees "Please wait for the experimenter". **C** continues; **Escape** ends the session. The fit, the warnings and your decision are saved in the block's metadata (`block_fit`, `screening`).

When a session ends complete, its files are copied to `data_solid/` (see [Data layout](#data-layout)).

Other launcher options:

- **Practice** runs a short version of a task (easy trials with feedback). Practice data are saved separately and never analysed.
- **Simulate** runs a task with a simulated observer and no window, to test the code on any computer.
- **Autotest** runs a task on the real display and sound card with automatic answers. Data go to `data_test/`, and a timing report appears at the end. See [Autotest](#autotest).
- **Text menu:** if Tk is missing, `python run_experiment.py --cli` gives a text menu.
- **Single task:** one task can be started directly, e.g. `python -m avdur.run_task --task crossmodal --pid P14`.

## Autotest

The autotest checks that every trial runs fluently on the actual setup. Use it after any change to the code, the computer or the sound hardware, and before the first participant of a day (quick version).

- **Whole experiment:** press **Autotest the whole experiment**, or run `python -m avdur.autotest --monitor lab_crt`. It runs the four tasks in order, including all three main blocks, with the ID `P99` unless you enter another.
  - The full run takes about 2 hours.
  - With **Quick** ticked (`--quick`), it runs only the first 30 trials of each task, about 7 minutes.
- **One task:** tick **Autotest** and press that task's button.
- **Answers:** they come from the simulated observer, so the staircases move as they would with a participant. Each answer is given at a random time 0.05–0.8 s after the prompt appears. Message screens continue by themselves.
- **Main blocks:** they use a bias correction fitted from the autotest's own cross-modal run. In a quick run, which is too short to fit, they use fixed values (`AUTOTEST_BIAS_S`).
- **Stopping:** press Escape during a trial.
- **The report** is shown at the end and saved as `data_test/<PID>/<PID>_autotest_report_<date-time>.txt`. For every session it gives:
  - trials run, and whether the session completed;
  - dropped frames and the longest frame;
  - how far the frame-0 flip fell from its prediction;
  - each visual flash's recorded length against its intended frames;
  - the sound onset relative to the frame-0 flip (PTB's estimate);
  - sound buffer underruns.

  It ends with a list of problems, or "no problems found". `python -m avdur.autotest --report-only --pid P99` reports on earlier autotest sessions.

## Before the first participant

- **Monitor profile.** Each profile fixes the screen and the refresh rate at which every duration is computed:

  | Profile | Use | Screen | Refresh | Data folder |
  |---|---|---|---|---|
  | `lab_crt` (default) | real participants: Sony GDM-5402 CRT at 800 × 800 | 1 | 120 Hz | `data/` |
  | `office_test` | trying it out on the Mac mini with the Dell S2722DZ (max 75 Hz) | 0 | 75 Hz | `data_test/` |
  | `laptop_test` | trying it out on a laptop set to a fixed 60 Hz | 0 | 60 Hz | `data_test/` |

  Test profiles never write to `data/`, and the launcher marks them.
- **Which screen.** The lab PC has two monitors: the CRT for the participant and one for the experimenter.
  - **How it's found:** PsychoPy's screen numbers need not match Windows' display numbers, so `lab_crt` finds the CRT by its resolution (`screen_px = (800, 800)`) wherever it is in the list. If no screen is at 800 × 800, the session stops at start and lists the screens it found.
  - **Choosing by hand:** the launcher's Screen menu (next to the profile) overrides this, and so does `--screen N` on the command line. **Identify screens** shows each screen's number on that monitor for 3 s.
  - **Logged:** the chosen screen and how it was chosen are printed at start and saved with the session (`screen_index`, `screen_choice`, `screens_found`).
- **Stimulus size.** The disk (radius 42.7 px) and the fixation cross (22.3 px) are fixed in pixels, so they are exactly what v1 participants saw. Every v1 script converted degrees with MacBook-Air numbers, whatever the display.
  - On the lab CRT the 800 × 800 picture is about 29 cm high, and the viewing distance is about 60 cm. The disk is therefore 3.1 cm, or 2.96° wide, which matches the paper's 3°.
  - Each session logs this (`disk_diameter_deg`) from `picture_height_cm` and `viewing_distance_cm` in the profile.
  - **Measure the picture width at 800 × 800** and enter it as `picture_width_cm`. The pixels are square only if the width is also about 29 cm. If the picture fills the tube's width (38–40 cm), the disk is an ellipse 1.3–1.4 times wider than tall.
- **Display check.** At start, the presenter times 180 of its own flips and requires the profile's rate (±1 Hz) with no more than 5% irregular frames. On failure it shows what it measured, e.g. "this display runs at about 75 Hz, choose the 'office_test' profile". A test profile can continue past irregular frames with C; the lab profile cannot. This replaces PsychoPy's `getActualFrameRate`, which returns no value at all ("cannot be measured") when frames are not perfectly regular. v1 silently used whatever rate it measured.
- **Failed starts.** A session that stops before its first trial (failed display or audio check) creates no files. It adds one line to `failed_starts.jsonl` in the data folder, with the measured refresh numbers.
- **Sound card.** The experiment plays through the output chosen in System Settings > Sound, unless `AUDIO_OUTPUT_DEVICE` (or a profile's `audio_device`) names a device; part of the name is enough, e.g. `"Speakers (Realtek"`.
  - **The PsychoPy bug this replaces.** PsychoPy 2024.2 does not use the selected output when left to itself. It takes the first device in PortAudio's list. On the office Mac that is the Dell monitor's DisplayPort audio, which caused "Failed to start PortAudio audio device".
  - **Opened at start.** The sound card is opened before the first trial. If it fails, or does not run at 48 kHz, the session stops with the list of available outputs, and no data files are written.
  - **Logged.** The device, its output latency, and how many frames ahead each sound is scheduled are printed at start and saved with the session.
  - **In the lab:** check the printed device name the first time. The lab autotest of 20 Oct 2026 played through "Headphones (Realtek(R) Audio)". Realtek often gives that name to the front jack whatever is plugged into it, so check that the two speakers are connected to that jack. If not, select the right output or pin it with the profile's `audio_device`.
- **Speakers and sound level.** The sounds are played through the two loudspeakers, identical in both channels.
  - Keep the speakers in the v1 positions, symmetric about the screen, so the sound seems to come from the disk.
  - Mark or tape the speaker volume knob and the Windows/Realtek volume.
  - Before the first participant, measure at the participant's head position with an A-weighted sound level meter (v1 used a Brüel & Kjær 2240, LAeq). Target the paper's levels: about 70.5 dB SPL during the test and standard intervals, 57.4 dB (low noise) and 67.5 dB (high noise) between them. `FIXED_VOLUME` reproduces v1's digital level, so matching these values means the speaker chain is set as in v1. Recheck now and then.
  - **On the Windows lab PC:**
    - In Control Panel > Sound > (the output the speakers are plugged into) > Properties, set Advanced > Default Format to 48000 Hz and turn off audio enhancements. With aggressive low latency, PTB uses WASAPI, which needs the device at the experiment's rate; the session stops at start if the rate differs.
    - The libusb/PortAudio preloading is macOS-only; the Windows psychtoolbox wheel ships its own libraries.
- **PsychToolbox.** The presenter needs PTB for both audio and the keyboard. The pip build of psychtoolbox 3.0.19 on macOS expects libusb and PortAudio to be loaded already. Without them it fails to load, PsychoPy falls back silently, and you get the `'KeyboardDevice' object has no attribute '_buffers'` error at exit. The presenter now preloads both libraries from the environment (the `psychopy310` env has them from conda-forge). If they are missing, it stops with an install hint.
- **Audiovisual sync.** Measure it once with a photodiode on the disk and a microphone at the participant's head position. That includes the sound's travel time from the speakers, about 1.7 ms at 60 cm. Put the audio-minus-video delay in `AUDIO_ONSET_OFFSET_S`.
- **Tests.** Two scripts check the code without a display or sound card. Run the [autotest](#autotest) as well, on the real setup.
  - `python tests/test_pipeline.py` runs two simulated participants through everything. It checks the design, the timing, the bias recovery, the audio levels, which sessions reach `data_solid/`, and the autotest report.
  - `python tests/test_presenter_logic.py` checks the PsychoPy presenter with fake window, keyboard and sound objects. It covers response handling, sound scheduling, autotest answers and the choice of output device.

## What changed from v1, and why

| v1 problem | v2 |
|---|---|
| Second-interval visual flash anchored to the first flash's offset, so it was displaced from its sound by ½ conflict + ½ bias correction | Each flash is centred on its own sound and extended symmetrically ([avdur/schedule.py](avdur/schedule.py)). The tests check the centring for every trial. |
| Visual test mis-timed when it came second (code versions of 2–11 June 2025) | Same fix; the logged flip times are checked against the intended durations. |
| Order drawn by a coin flip on each trial; make-up sessions unbalanced | Exactly balanced test-first and test-second trials within every staircase and condition ([avdur/design.py](avdur/design.py)). |
| Two staircases shared by all conditions of a session; conditions in a fixed cycle | Two staircases per condition, as the paper describes; all trials shuffled. |
| Cross-modal staircase `1U1D` moved away from threshold | Rule names can only express converging staircases; that slot is now a second 1-down-1-up. |
| From 2 June 2025 every staircase started at the bottom of the range (in the main experiment, at the floor raised by the bias correction). Before that the unimodal and cross-modal staircases started at ±0.65; make-ups started near the middle. | Every staircase starts on the side opposite the point it converges to, so it crosses the PSE on its way: 2D1U and 3D1U at the floor, 1D2U and 1D3U at the ceiling. The two cross-modal 1D1U staircases start one at each end, randomly per condition. In simulation this put 75–98% of staircase trials within ±1σ of the PSE (random starts: 56–94%) and made the PSE more precise in the unimodal tasks. The start is logged in `stair_start` and in the session metadata. |
| Bias correction fitted in a notebook and typed in by hand (dt's low-noise value was wrong; values left over from earlier participants; one make-up with the two noise levels swapped) | Fitted from the participant's own cross-modal data, confirmed in the launcher, saved as a file, logged in every row, and reused for all main blocks. |
| Fit with lapse ≤ 0.2 and μ bounds that some fits hit | Lognormal fit with lapse ≤ 0.1, multi-start, a bootstrap CI and quality warnings. v1's linear method is reported alongside for comparison. |
| Sound started from inside the frame loop, not scheduled | Each trial starts with a sync flip. The sound is then scheduled for the frame-0 flip, at least the sound card's output latency + 15 ms ahead, and PTB's estimate of the actual onset is logged against that flip. PsychoPy's flip prediction uses the measured frame period: PsychoPy assumes 60 Hz when its own refresh measurement fails. |
| Output device left to PsychoPy (2024.2 takes the first device PortAudio lists) | The output chosen in System Settings, or a named device. It is opened and checked before the first trial, and the device is logged with each session. |
| 0.1 s wait before every trial and after every stimulus; keys ignored until a prompt appeared; 5 s minimum breaks | Same logic with shorter waits: 30 ms before each trial and 30 ms after the stimulus (sound and flashes) has ended. Then the prompt appears on every trial. Keys pressed before it are discarded, and the response time counts from its first flip. Breaks every 30 trials last at least 2.5 s. At least 200 ms of noise always precedes any sound burst or flash. A day of letting answers come before the prompt (19–20 Oct 2026) was dropped, because participants could answer before seeing the prompt. |
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
data/failed_starts.jsonl                                    sessions that stopped before the first trial
data_solid/ ...                                             same layout: copies of complete real sessions + bias files
data_test/ ...                                              same layout, test monitor profiles and autotests
data_simulated/ ...                                         same layout, simulated observers (tests, Simulate)
```

`data/` keeps every real run as recorded. `data_solid/` receives a copy of a session when it ends complete. Practice runs, stopped sessions, test displays, autotests and simulations are never copied. Confirmed bias files are copied too.

The analysis reads `data_solid/`. To exclude a session, delete its copy there; the original stays in `data/`.

The column names follow v1 (`testDurS`, `audNoise`, `order`, `responses`, `conflictDur`, `VisualPSE`, `recordedDurVisualTest`, …). `response_rts` is measured from the prompt's first flip, as in v1. New columns:
- `prompt_shown` (always 1) and `answer_allowed_from_s` (prompt onset, from the first frame of the trial);
- the frame schedule (`audio_interval1`, `visual_interval1`, …);
- intended visual durations;
- `chose_test` and `stair_start` (`floor` or `ceiling`);
- `n_dropped_frames`, `flip0_error_s`, `audio_start_minus_flip0_s` (PTB's estimate of the sound onset minus the frame-0 flip) and `audio_xruns` (sound buffer underruns);
- the audio seed and level mode.

An interrupted session keeps its trials; its metadata says `completed: false`, and the analysis uses completed sessions only.

## Analysis

The notebooks in [analysis/](analysis/) use the repository's `plot_style.py`:

- `01_unimodal_fits.ipynb`: auditory and visual psychometric functions, σ (and σ/√2 for the models), a free-μ check, and a visual timing check.
- `02_crossmodal_bias.ipynb`: cross-modal fits, the corrections with CIs and the v1 method, and a check that the main blocks used the confirmed values.
- `03_main_experiment_check.ipynb`: order balance, timing and centring checks, PSE vs. conflict per noise level, and PSE by presentation order.
- `04_staircases.ipynb`: staircase progression for each session. It plots test duration against trial number, one line per staircase and one panel per condition, marking answers, reversals, easy trials, the standard and the fitted PSE. A table lists each staircase's start, reversals and mean level after the second reversal.

They read `data_solid/`. Until real data exist, they show the simulated participants written by `tests/test_pipeline.py` to `data_simulated/`. `tests/calibrate_audio_levels.py` reproduces the fixed-level audio constants.
