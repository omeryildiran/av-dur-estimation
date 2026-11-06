"""Every parameter of the v2 experiment. Change values here and nowhere else.

Values marked "v1" reproduce the 2025 data collection. Values marked "v2" are deliberate
changes; README.md explains each one.
"""

import math

EXPERIMENT_NAME = "av_duration_v2"

# ── Display ───────────────────────────────────────────────────────────────────
# A profile gives the window, the screen, and the refresh rate at which every duration is
# computed (checked at start). The picture size and viewing distance are used only to log
# the true visual angle of the stimuli. Profiles marked testing_only are for trying the
# experiment on other machines: they run at that display's own rate and save to data_test/,
# so they never mix with real data. An optional "audio_device" overrides AUDIO_OUTPUT_DEVICE.
# "screen_px" finds the participant's screen by its resolution (avdur/screens.py), because
# PsychoPy's screen numbers need not match Windows' display numbers; "screen" is the fallback,
# and the launcher's Screen menu (--screen) overrides both.
MONITOR_PROFILES = {
    # Lab: Sony GDM-5402 (21" Trinitron CRT, tube picture about 40 x 31 cm), set to 800 x 800
    # at 120 Hz, which shows a picture about 29 cm high. Paper: about 60 cm away, chin rest.
    "lab_crt": dict(window_px=800, screen=1, screen_px=(800, 800), fullscreen=True, frame_rate_hz=120.0,
                    picture_height_cm=29.0, picture_width_cm=None,     # width: measure (square pixels if ~29 cm)
                    viewing_distance_cm=60.0),
                    # audio_device="...",  # pin the output the two speakers are plugged into (see README)
    "office_test": dict(window_px=800, screen=0, fullscreen=True, frame_rate_hz=75.0,
                        testing_only=True),                            # Mac mini + Dell S2722DZ
    "laptop_test": dict(window_px=800, screen=0, fullscreen=True, frame_rate_hz=60.0,
                        testing_only=True,                             # MacBook Air M3 13" (fixed 60 Hz panel)
                        pace_flips=True),   # its flips don't wait for the refresh (presenter_psychopy.pace_flips)
}
DEFAULT_MONITOR = "lab_crt"

EXPECTED_FRAME_RATE_HZ = MONITOR_PROFILES[DEFAULT_MONITOR]["frame_rate_hz"]   # the lab rate (120 Hz)
FRAME_RATE_TOLERANCE_HZ = 1.0       # v2: measured rate must be within this of the profile's rate
MAX_IRREGULAR_FRAMES = 0.05         # v2: share of frame intervals more than 1.5 ms off the median
BACKGROUND_COLOR = "gray"           # v1

# Stimulus sizes are fixed in pixels. Every v1 script converted degrees to pixels with
# MacBook-Air numbers (800 px, 25 x 20 cm, 40 cm away), whatever the display, so these are
# the pixel sizes v1 participants saw. On the lab CRT (29 cm per 800 px, 60 cm away) the
# disk is 3.1 cm, i.e. 2.96 deg wide, matching the paper's 3 deg.
DISK_RADIUS_DEG = 1.5               # v1 nominal: 3 deg disk
DISK_RADIUS_PX = DISK_RADIUS_DEG / (math.degrees(math.atan2(10.0, 40.0)) / 400)   # v1 dva_to_pix.py: 42.75 px
FIXATION_HEIGHT_PX = 1.0 * 40.0 * 0.017455 * 800 / 25.0    # v1 deg2pix(1, monitor): 22.34 px
DISK_LINE_WIDTH_PX = 1.5            # v1 wrote `lineWidht = 5` (typo), so PsychoPy's default 1.5 px was shown
TEXT_HEIGHT_PX = 30                 # v1

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000                 # v1
AUDIO_LATENCY_MODE = 3              # v1 set prefs.general (ignored by PsychoPy), so the default 3 was used
# v2: the sound card is chosen explicitly and opened before the first trial. Left to itself,
# PsychoPy 2024.2 plays through the FIRST output device PortAudio lists (on the office Mac,
# the Dell monitor's DisplayPort audio), not the output chosen in System Settings > Sound.
AUDIO_OUTPUT_DEVICE = None          # None: the Mac's current output (System Settings > Sound);
                                    # or (part of) a device name, e.g. "Speakers (Realtek"
AUDIO_SCHEDULE_MARGIN_S = 0.015     # v2: sound scheduled at least output latency + this before frame 0
                                    # (5 ms let 23% of sounds start 0.5-3.5 ms late on the lab PC's Realtek
                                    # output, 10 ms latency, Windows; 2026-10-20)
SYNC_FLIPS = 2                      # v2: flips before scheduling frame 0. The first flip after a pause can
                                    # return off the refresh (seen on macOS: frame 0 up to 11.8 ms early on
                                    # 6% of trials, always the first trial after a message screen)
VOLUME = 0.11                       # v1
AUDIO_ONSET_OFFSET_S = 0.0          # v2: measured audio-minus-video delay (photodiode + microphone)
NOISE_LEVELS = (0.1, 1.2)           # v1: background noise relative to the signal peak (v1 comparisons, tests)
# v2: both noise levels are fixed for every participant. From 1 Nov 2026 the high level is 1.0
# (v1: 1.2). From 28 Oct to 1 Nov 2026 it was set per participant by the auditory noise calibration
# (avdur/noise_calibration.py, NOISE_CALIBRATION; P101-P106), dropped with the PI: in loud noise many
# listeners fail with the test first but not second, so the calibration misjudged the level (P103,
# P105), took 12 min and left each participant at another level. Simulated (analysis/overnight_sims/
# fixed_noise_sim.py), model recovery does not depend on the high-noise sigma between 0.4 and 3.0;
# 1.0 keeps the high/low contrast (v1 sigma at 1.2 ranged from 0.42 to chance) while sparing the worst
# listeners a little. The participant is excluded if the high-noise sigma reaches the fit bound
# (SCREENING_CHECKS). The calibration stays as an optional launcher button, just in case; nothing
# uses its result unless the experimenter passes it as --noise-high. SPL at 1.0 still to be measured (B&K).
NOISE_LOW = 0.1
NOISE_HIGH = 1.0                    # every participant, every task with sound (--noise-high overrides)
NOISE_HIGH_DEFAULT = 1.2            # v1's level: only simulations and tests that build sessions directly
NOISE_CALIBRATION = dict(
    target_sigma=0.75,              # (0.85 briefly on 26 Oct 2026, not used for any participant.) Model simulations: 0.7-0.85 separates fusion, causal inference and
                                    # cue switching best (switching is best separated near sigma_v, v1 median 0.62)
    target_correct=0.7071,          # 2-down-1-up on correct answers (2 correct: louder noise; 1 error: softer)
    rules=("2D1U", "3D1U"),         # staircases A, B on correct answers. 3D1U (79.4% correct, lower noise) from 30 Oct
                                    # 2026, so the sigma(noise) fit sees two performance levels (both 2D1U before)
    trials_per_staircase=100,       # two interleaved staircases (216 trials with the catch trials, ~12 min; 60 gave ~6 min)
    start_noise=(0.5, 0.5),         # (0.3 and 1.6 until 28 Oct 2026)
    bounds=(0.1, 1.2),              # cap at v1's level (2.0 until 28 Oct 2026). In loud noise most listeners
                                    # are near chance with the test first but stay sharp with it second; the
                                    # pooled % correct then barely changes between 1 and 2, so the staircases
                                    # drifted to 1.7-1.8 (P103, P105) where the unimodal sigma was 1.6-2.3
                                    # (analysis/noise_calibration_order_check.py)
    init_step_ln=0.4, step_factor=0.5, n_step_changes=1,   # steps in ln(noise): 0.4, then 0.2
    # Simulated (true levels 0.3-1.5, shallow and steep sigma-noise slopes): the sigma reached is
    # within 14% of the target in half of the runs and within 34% in 90% at 60 trials (40 trials: 15% / 49%).
    catch_trials=16,                # easy trials at NOISE_LOW: attention check, and they pin the fit's lapse rate; 4 per
                                    # test (50 / 950 ms) x order (6 until 30 Oct 2026; without them P105's lapse went to
                                    # 0.1 and the level up 13%)
)
SIGNAL_BAND_HZ = (150, 775)         # v1: unchanged since commit c7ff22e1 (24 Apr 2025), used for all data
BACKGROUND_BAND_HZ = (10, 610)      # v1: same commit
FILTER_ORDER = 4                    # v1: Butterworth, causal sosfilt
BURST_INTENSITY = 5                 # v1
BURST_RISE_S = 0.005                # v1
HAMMING_WINDOW = True               # v1: PsychoPy's default ramp on the whole sound

# v2: fixed levels instead of v1's per-trial peak normalisation ("peak" reproduces v1 exactly).
# Calibrated with tests/calibrate_audio_levels.py so the average burst level and the SNR equal v1.
AUDIO_LEVEL_MODE = "fixed"
FIXED_BACKGROUND_SCALE = 4.7259     # background SD per unit noise level, relative to the noise floor
FIXED_OUTPUT_GAIN = {0.1: 0.21133, 1.2: 0.14382}   # per noise level: v1's mean burst level, loudest trial at 0.95 FS
FIXED_VOLUME = 0.1653               # played level equals v1's (v1: -11.6 dBFS bursts at volume 0.11)

# ── Trial timing ──────────────────────────────────────────────────────────────
STANDARD_DUR_S = 0.5                # v1
# v2 (26 Oct 2026): gaps shortened from v1's 0.20-0.45 / 0.40-0.90 / 0.20-0.45 s to 0.20-0.35 /
# 0.30-0.60 / 0.15-0.30 s, 0.35 s less per trial (about 12%). 3 Nov 2026 (user): pre and post
# 0.10-0.45 s, ISI unchanged (pre as long as before on average, post 0.05 s longer). The gaps only
# add noise around the sounds: make_schedule never shortens a gap and never changes a sound or a
# flash. When an extended flash (bias correction, conflict) needs room, it shifts that gap's whole
# random range up by just enough, so every gap stays random. Until 3 Nov 2026 it raised short draws
# to the minimum instead, which fixed a gap at large conflicts (pre 425 ms at +450 ms, standard
# first). tests/test_schedule.py checks this for every task.
PRE_RANGE_S = (0.10, 0.45)          # noise before interval 1 (at least MIN_LEAD_S before any event)
ISI_RANGE_S = (0.30, 0.60)          # between the two intervals
POST_RANGE_S = (0.10, 0.45)         # noise after interval 2 (masks its offset), then the prompt
MIN_VISUAL_GAP_S = 0.10             # v2: minimum blank between and after visual flashes (0.05 until 26 Oct 2026)
MIN_VISUAL_DUR_S = 0.025            # v2: shortest visual flash allowed after the bias correction
# The prompt appears on every trial, POST_STIM_WAIT_S after the stimulus (sound and flashes)
# has ended; keys pressed earlier are discarded and response_rts counts from the prompt, as in
# v1. v1 waited 0.1 s at both points; v2 waits 30 ms (2026-10-20: answering before the prompt,
# allowed for one day, let answers come before the participant saw the prompt).
PRE_TRIAL_WAIT_S = 0.03
POST_STIM_WAIT_S = 0.03
MIN_LEAD_S = PRE_RANGE_S[0]         # no event earlier in a trial: the shortest pre (0.20 until 3 Nov 2026)
BREAK_EVERY_N_TRIALS = 30           # v1
BREAK_MIN_WAIT_S = 2.5              # v2 (v1: 5 s)
ORDER_CUE_MIN_WAIT_S = 1.0          # cross-modal "A -> V / V -> A" cue: keys pressed earlier are ignored (P104 skipped it)
FEEDBACK_DUR_S = 0.5                # v1 (practice only)
RESPONSE_KEYS = {"left": 1, "right": 2}   # v1: 1 = first interval longer, 2 = second
QUIT_KEY = "escape"
PRACTICE_TRIALS = 20                # v2: easy trials with feedback

# ── Combined session: unimodal auditory, unimodal visual, cross-modal in one window ──
# After each block the psychometric functions are fitted in a background process while the
# participant rests; the fit, a plot and checks appear on the terminal. If a check fails, the
# next screen asks for the experimenter (C continues, Escape ends); the decision is logged.
# Checks concern attention/precision only, never the PSE or the bias. Thresholds: about 1.5x
# beyond the worst v1 participant (v1 low-noise auditory sigma 0.23-0.39, easy trials 86-100%
# correct; visual sigma 0.36-1.72, easy 79-100%). High noise is reported, not checked: in v1
# 2 of 11 participants were at the sigma bound (3.0) there, and easy trials were 57-93% correct.
COMBINED_SESSION_TASKS = ("unimodal_auditory", "unimodal_visual")   # cross-modal dropped 28 Oct, calibration 1 Nov 2026
BLOCK_REST_MIN_S = 10.0              # rest between blocks (the fit runs meanwhile)
ORDER_GAP_WARNING = 0.25             # warn if % correct with the test first vs second differs more (chance SD ~0.08)
SCREENING_CHECKS = {
    "unimodal_auditory": {0.1: dict(max_sigma=0.6, min_easy_correct=0.75),
                          "high": dict(max_sigma=2.9)},       # NOISE_HIGH. At the fit bound (3.0): exclude from the
                                                              # main experiment (1 Nov 2026; fixed_noise_sim.py)
    "unimodal_visual": {None: dict(max_sigma=2.0, min_easy_correct=0.70)},
}
V1_SIGMA_RANGE = {                   # (min, median, max) over the 11 v1 participants, lognormal fit, lapse <= 0.1
    ("unimodal_auditory", 0.1): (0.23, 0.29, 0.39), ("unimodal_auditory", 1.2): (0.42, 1.88, 3.00),
    ("unimodal_visual", None): (0.36, 0.62, 1.72),
    ("crossmodal", 0.1): (0.11, 0.39, 1.06), ("crossmodal", 1.2): (0.30, 0.78, 1.62),
}

# ── Autotest (automatic answers on the real display and sound card; data_test/ only) ──
# Answers come from the simulated observer (so staircases move as with a real participant)
# at a random time after the prompt. Instruction, break and end screens continue by themselves.
AUTOTEST_RT_RANGE_S = (0.05, 0.80)     # after the prompt
AUTOTEST_SCREEN_S = 0.3
AUTOTEST_QUICK_TRIALS = 30          # "quick" autotest: first trials of each task only
AUTOTEST_BIAS_S = {0.1: 0.10, 1.2: -0.05}   # main-block corrections when no full cross-modal autotest exists

# ── Staircases (v1 rules) ─────────────────────────────────────────────────────
# Rule "xDyU": x "test longer" answers in a row step the test down, y "standard longer"
# answers in a row step it up. Levels are delta_dur_percents = test/standard - 1.
# The step halves at the first two reversals, then stays fixed.
# v2: every staircase starts on the side opposite the point it converges to, so it crosses
# the PSE on its way: 2D1U and 3D1U (above 50% "test longer") at the floor, 1D2U and 1D3U
# at the ceiling; the two cross-modal 1D1U (50%) one at each end, random per condition.
# Simulated: more trials near the PSE and a more precise PSE than random or same-side starts
# (unimodal, sigma 0.6: PSE SD 33-38 vs 39-42 ms), sigma slightly less precise. v1 started
# every staircase at the floor (from 2 June 2025).
# Long tests (4 Nov 2026, user). Until then everything stopped at 0.9 (950 ms, as v1): only 1.9x the
# standard against 10x below it, so the long "easy" trials were not easy for shallow listeners, the top of
# the curve (lapse vs sigma) was poorly pinned, and at +450 ms conflicts full visual capture put the PSE at
# the cap. Now the staircases that start at the top start at start_top = 0.9 (950 ms) and can climb to a
# safety cap of 5.0 (3000 ms, 6x the standard), which only stops runaways (a guessing listener pushes a
# 2D1U staircase up). QUEST+ candidates and the long easy trials go up to TEST_CEILING = 3.0 (2000 ms,
# log 1.39; 1500 ms on 4 Nov): about 92% correct even at sigma 1, where v1's 950 ms gave 74%.
# Main experiment parts (6 Nov 2026, user): parts 2 and 3 continue the staircases at their levels but go
# back to the first step (80 ms), halved again at that part's first two reversals, so a staircase can follow
# a listener who changed between days. Simulated (part 1 day 1, parts 2-3 day 2; PSE +80 ms or sigma x1.5 or
# x0.67 on day 2): sigma error 31-34% vs 35-38% at sigma 0.8, otherwise the same; restarting each part was
# slightly worse everywhere.
STAIRCASE = dict(init_step=0.16, step_factor=0.5, n_step_changes=2, start_top=0.9, ceiling=5.0)
RESET_STEP_EACH_PART = True
TEST_CEILING = 3.0

# QUEST+ (avdur/quest.py): a task whose staircases are ("QUEST",) gets one QUEST+ per condition
# instead (trials_per_staircase trials, order balanced, plus the easy trials, which also inform
# its lapse). It targets mu and sigma, with the lapse as a nuisance parameter. Grids as in the
# simulations (analysis/quest_unimodal_simulation.py); candidate tests are whole frames between
# the task's floor and TEST_CEILING, n_levels of them evenly spaced in log duration.
# Easy (lapse) trials, from 28 Oct 2026: per session and noise level, max(minimum, share x the adaptive
# trials at that noise level), half at each end (lapse_levels), spread evenly over the conditions at
# that noise level, order balanced. The fits share one lapse rate over conditions, so the count is per
# noise level, not per condition (before: lapse_trials_per_level per level and condition).
EASY_TRIALS = dict(share=0.10, minimum=20)

QUEST = dict(target="mu_sigma", mu_range=(-1.0, 1.0), n_mu=81, sigma_range=(0.05, 2.5), n_sigma=25,
             lapse_max=0.1, n_lapse=5, n_levels=60)

# Unimodal blocks: how the test durations are chosen. The launcher (or --method) chooses per session;
# the choice is saved in the session metadata (design_override). Simulated with 154 trials per condition
# (quest_simulation_results/unimodal_summary.csv, median over observers): log sigma error QUEST+ 0.19,
# staircases 0.28; PSE error QUEST+ 55 ms, staircases 50 ms. The joint model fit takes sigma_A and
# sigma_V from these blocks, so QUEST+ was the default from 3 Nov 2026. From 4 Nov 2026 (user) the
# staircases are the default again: less precise for sigma, but the v1 method and easy to explain.
# From 5 Nov 2026 (user): two staircases of 70 per condition instead of four of 35, 2D1U from the floor
# (converges at 70.7% "test longer") and 1D3U from 950 ms (20.6%). Simulated (140 staircase + 20 easy
# trials, 300 observers per cell): sigma error 20-30%, as good as 4 x 35 or slightly better (aud sigma
# 0.6: 23% vs 26%), PSE error the same; 3D1U + 1D2U about the same; 2D1U + 1D2U worse (24-36%).
UNIMODAL_METHODS = {
    "quest": dict(staircases=("QUEST",), trials_per_staircase=150),          # + 20 easy per noise level
    # v1's steps, opposite-side starts, balanced orders. 4 x 35 (2D1U, 1D2U, 3D1U, 1D3U) until 5 Nov 2026
    "staircase": dict(staircases=("2D1U", "1D3U"), trials_per_staircase=70),
}
UNIMODAL_METHOD = "staircase"       # default (QUEST+ 3-4 Nov 2026)
UNIMODAL_METHOD_LABELS = {"quest": "QUEST+", "staircase": "Staircases (2 x 70, default)"}

TASKS = {
    "noise_calibration": dict(
        label="Auditory noise calibration (sets the high noise level)",
        noise_levels=(None,), conflicts=(0.0,), staircases=("A", "B"), floor=-0.9,
    ),
    "unimodal_auditory": dict(
        label="Unimodal auditory",
        noise_levels=NOISE_LEVELS, conflicts=(0.0,),
        # UNIMODAL_METHODS: the staircases 2 x 70 (2D1U, 1D3U) + 20 easy per noise level (320 trials, ~17-22 min; default), or
        # QUEST+ 150 + 20 easy (340 trials, ~17-24 min). History: v2 staircases (4 x 35 + 14 easy, 308)
        # until 27 Oct 2026; QUEST+ 90 + 14 easy from 27 Oct, 100 + 20 from 28 Oct, 150 + 20 from 29 Oct;
        # the choice between the two in the launcher from 3 Nov 2026.
        **UNIMODAL_METHODS[UNIMODAL_METHOD],
        floor=-0.9, lapse_levels=(-0.9, TEST_CEILING),
    ),
    "unimodal_visual": dict(
        label="Unimodal visual",
        noise_levels=(None,), conflicts=(0.0,),
        # UNIMODAL_METHODS, as the auditory block: the staircases 2 x 70 + 20 easy (160 trials; default) or
        # QUEST+ 150 + 20 easy (170 trials, ~9-12 min). History: staircases (154) until 27 Oct 2026; QUEST+ 70 + 14 easy
        # from 27 Oct, 100 from 28 Oct, 150 + 20 from 29 Oct; the choice in the launcher from 3 Nov 2026.
        **UNIMODAL_METHODS[UNIMODAL_METHOD],
        floor=-0.9, lapse_levels=(-0.8, TEST_CEILING),
    ),
    "crossmodal": dict(
        label="Cross-modal (auditory test vs. visual standard)",
        noise_levels=NOISE_LEVELS, conflicts=(0.0,),
        # v1 used 1D1U, 1U1D, 2D1U, 1D2U; its 1U1D rule moved away from threshold.
        # v2 replaces it with a second 1D1U (pairs: the two 1D1U, then 2D1U/1D2U).
        staircases=("1D1U", "1D1U_b", "2D1U", "1D2U"), trials_per_staircase=45,  # v1: 388 trials
        floor=-0.9, lapse_levels=(-0.9, TEST_CEILING),
        order_block_size=10,          # v1: order alternates in pre-cued mini-blocks
    ),
    "main": dict(
        label="Main audiovisual experiment",
        noise_levels=NOISE_LEVELS,
        # From 28 Oct 2026 (overnight_sims report): 9 conflicts -450 ... +450 ms (whole frames at 120 Hz),
        # all interleaved, in two parts; part 2 continues part 1's staircases. No bias correction: the bias
        # is modelled. From 29 Oct 2026, 25 trials per staircase per part (50 per staircase, 100 per condition,
        # plus 46 easy per noise level and part: 992 trials per part, ~50-60 min); 16 until then (636 per part).
        # From 3 Nov 2026 three parts of 17 trials per staircase (51 per staircase, 102 per condition; 612 + 64 easy
        # = 676 trials per part, ~35-47 min at the pilots' 3.1-4.2 s per trial), so the experiment fits in two
        # sessions of up to 1.5 h: unimodal blocks + part 1, then parts 2 and 3. Each part continues the earlier
        # parts' staircases, so the split does not change them.
        # Before: 7 conflicts -250 ... +250 ms in blocks B1-B3, 70 trials per staircase, corrected visual.
        # Rules from 5 Nov 2026 (user): 2D1U from the floor and 1D3U from 950 ms per condition (2D1U + 1D2U
        # until then). Simulated (9 conflicts x 102 trials + 96 easy per noise level, 100 observers): sigma error
        # 26-35% vs 30-42% with 1D2U, PSE error about the same (22-23 ms at sigma 0.3). v1's two staircases
        # shared by all conditions: fine for 0 to +/-250 ms but at +/-450 sigma error 41-56%, PSE error 33-90 ms.
        # Each staircase has 8-15 reversals over its 51 trials (first by trial 5-15).
        staircases=("2D1U", "1D3U"), trials_per_staircase=17,
        floor=-0.95, lapse_levels=("floor", TEST_CEILING),
        # Conflicts from 4 Nov 2026 (user): v1's 7 (0, +/-83, +/-167, +/-250 ms) plus the extremes +/-450 ms,
        # still 9 and whole frames at 120 Hz (10, 20, 30, 54 frames). 28 Oct-4 Nov: 0, +/-117, +/-225, +/-333, +/-450.
        blocks={part: (0.0, -0.0833, 0.0833, -0.1667, 0.1667, -0.25, 0.25, -0.45, 0.45)
                for part in ("part1", "part2", "part3")},
    ),
}

# v2: main-experiment block order, assigned by the participant number (P01 -> first row, ...)
BLOCK_ORDERS = [("part1", "part2", "part3")]   # from 3 Nov 2026 (two parts from 28 Oct; before: the 6 orders of B1-B3)

# ── Cross-modal task at several standards (v2, 2026-10-23) ─────────────────────
# Is the audiovisual duration bias a constant offset or does it scale with duration? With
# enabled=True the cross-modal task runs one block per standard: 500 ms first, with the full
# design (it alone sets the main experiment's correction), then the others with a shorter
# design, in an order that alternates by participant number. --single-standard (or the
# launcher checkbox) runs the 500 ms block only; --all-standards runs them all.
# n_participants: only the first n participants to complete an extra standard get them (a
# participant who has one keeps getting the rest); None: everyone. The bias grew with the
# standard in P101 and P104 (26 Oct 2026), so 3 participants are enough to describe it.
# Dropped 28 Oct 2026 with the whole cross-modal task (decided with the PI): the bias is modelled in
# the main experiment instead (ln v_hat = ln v - c - d ln(v / 0.5)).
CROSSMODAL_STANDARDS = dict(
    enabled=False,
    n_participants=3,
    standards=(0.3, 0.8),           # in addition to STANDARD_DUR_S
    extra_design=dict(staircases=("1D1U", "1D1U_b"), trials_per_staircase=40, lapse_trials_per_level=3),   # 30 until P104 (132 trials)
)

# ── Cross-modal bias correction (v2: fitted automatically) ────────────────────
# The correction is the visual duration added to every visual flash in the main
# experiment (VisualPSE in the data files), split equally over onset and offset.
BIAS = dict(
    method="lognormal",           # "lognormal" (log-ratio PF) or "linear_v1" (v1 notebook method)
    lapse_max=0.1,
    min_trials=300,
    max_abs_correction_s=0.35,
    max_ci_halfwidth_s=0.06,
    n_bootstrap=200,
)
