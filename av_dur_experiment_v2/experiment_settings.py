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
MONITOR_PROFILES = {
    # Lab: Sony GDM-5402 (21" Trinitron CRT, tube picture about 40 x 31 cm), set to 800 x 800
    # at 120 Hz, which shows a picture about 29 cm high. Paper: about 60 cm away, chin rest.
    "lab_crt": dict(window_px=800, screen=1, fullscreen=True, frame_rate_hz=120.0,
                    picture_height_cm=29.0, picture_width_cm=None,     # width: measure (square pixels if ~29 cm)
                    viewing_distance_cm=60.0),
    "office_test": dict(window_px=800, screen=0, fullscreen=True, frame_rate_hz=75.0,
                        testing_only=True),                            # Mac mini + Dell S2722DZ
    "laptop_test": dict(window_px=800, screen=0, fullscreen=True, frame_rate_hz=60.0,
                        testing_only=True),                            # laptop set to 60 Hz
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
                                    # or (part of) a device name, e.g. "External Headphones"
AUDIO_SCHEDULE_MARGIN_S = 0.005     # v2: sound scheduled at least output latency + this before frame 0
VOLUME = 0.11                       # v1
AUDIO_ONSET_OFFSET_S = 0.0          # v2: measured audio-minus-video delay (photodiode + microphone)
NOISE_LEVELS = (0.1, 1.2)           # v1: background noise relative to the signal peak
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
PRE_RANGE_S = (0.20, 0.45)          # v1
ISI_RANGE_S = (0.40, 0.90)          # v1
POST_RANGE_S = (0.20, 0.45)         # v1
MIN_VISUAL_GAP_S = 0.05             # v2: minimum blank between and after visual flashes
MIN_VISUAL_DUR_S = 0.025            # v2: shortest visual flash allowed after the bias correction
# v2: the next trial starts right after the response (v1 waited 0.1 s before each trial and
# 0.1 s after each stimulus). The pre-stimulus noise is the preparation time, so nothing
# (sound burst or visual flash) may start earlier than MIN_LEAD_S after the noise begins.
PRE_TRIAL_WAIT_S = 0.0
MIN_LEAD_S = 0.20                   # = the shortest v1 pre-stimulus noise
# v2: answers are accepted from the end of the second interval (sound and flash both over);
# the rest of the noise still plays in full, and the prompt appears only if no answer came
# before the sound ended. v1 ignored keys until the prompt, shown 0.1 s after the sound.
# response_rts is therefore measured from the end of the second interval.
BREAK_EVERY_N_TRIALS = 30           # v1
BREAK_MIN_WAIT_S = 5.0              # v1
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
COMBINED_SESSION_TASKS = ("unimodal_auditory", "unimodal_visual", "crossmodal")
BLOCK_REST_MIN_S = 10.0              # rest between blocks (the fit runs meanwhile)
SCREENING_CHECKS = {
    "unimodal_auditory": {0.1: dict(max_sigma=0.6, min_easy_correct=0.75)},
    "unimodal_visual": {None: dict(max_sigma=2.0, min_easy_correct=0.70)},
}
V1_SIGMA_RANGE = {                   # (min, median, max) over the 11 v1 participants, lognormal fit, lapse <= 0.1
    ("unimodal_auditory", 0.1): (0.23, 0.29, 0.39), ("unimodal_auditory", 1.2): (0.42, 1.88, 3.00),
    ("unimodal_visual", None): (0.36, 0.62, 1.72),
    ("crossmodal", 0.1): (0.11, 0.39, 1.06), ("crossmodal", 1.2): (0.30, 0.78, 1.62),
}

# ── Autotest (automatic answers on the real display and sound card; data_test/ only) ──
# Answers come from the simulated observer (so staircases move as with a real participant)
# at a random time after the second interval: some during the remaining noise, some after
# the prompt. Instruction, break and end screens continue by themselves.
AUTOTEST_RT_RANGE_S = (0.05, 0.80)
AUTOTEST_SCREEN_S = 0.3
AUTOTEST_QUICK_TRIALS = 30          # "quick" autotest: first trials of each task only
AUTOTEST_BIAS_S = {0.1: 0.10, 1.2: -0.05}   # main-block corrections when no full cross-modal autotest exists

# ── Staircases (v1 rules) ─────────────────────────────────────────────────────
# Rule "xDyU": x "test longer" answers in a row step the test down, y "standard longer"
# answers in a row step it up. Levels are delta_dur_percents = test/standard - 1.
# The step halves at the first two reversals, then stays fixed.
# v2: each task lists its staircases in pairs. In every condition one staircase of each pair
# starts at the ceiling and the other at the floor; which one is random per condition,
# balanced over the session's conditions. v1 started every staircase at the floor.
STAIRCASE = dict(init_step=0.16, step_factor=0.5, n_step_changes=2, ceiling=0.9)

TASKS = {
    "unimodal_auditory": dict(
        label="Unimodal auditory",
        noise_levels=NOISE_LEVELS, conflicts=(0.0,),
        staircases=("2D1U", "1D2U", "3D1U", "1D3U"), trials_per_staircase=35,   # v1: 308 trials
        floor=-0.9, lapse_levels=(-0.9, 0.9), lapse_trials_per_level=7,
    ),
    "unimodal_visual": dict(
        label="Unimodal visual",
        noise_levels=(None,), conflicts=(0.0,),
        staircases=("2D1U", "1D2U", "3D1U", "1D3U"), trials_per_staircase=35,   # v1: 154 trials
        floor=-0.9, lapse_levels=(-0.8, 0.8), lapse_trials_per_level=7,
    ),
    "crossmodal": dict(
        label="Cross-modal (auditory test vs. visual standard)",
        noise_levels=NOISE_LEVELS, conflicts=(0.0,),
        # v1 used 1D1U, 1U1D, 2D1U, 1D2U; its 1U1D rule moved away from threshold.
        # v2 replaces it with a second 1D1U (pairs: the two 1D1U, then 2D1U/1D2U).
        staircases=("1D1U", "1D1U_b", "2D1U", "1D2U"), trials_per_staircase=45,  # v1: 388 trials
        floor=-0.9, lapse_levels=(-0.9, 0.9), lapse_trials_per_level=7,
        order_block_size=10,          # v1: order alternates in pre-cued mini-blocks
    ),
    "main": dict(
        label="Main audiovisual experiment",
        noise_levels=NOISE_LEVELS,
        staircases=("2D1U", "1D2U"), trials_per_staircase=70,   # v2: per condition, as the paper describes
        floor=-0.95, lapse_levels=("floor", 0.9), lapse_trials_per_level=7,
        blocks={"B1": (0.0, -0.167, 0.25), "B2": (-0.083, 0.167), "B3": (-0.25, 0.083)},
    ),
}

# v2: main-experiment block order, assigned by the participant number (P01 -> first row, ...)
BLOCK_ORDERS = [
    ("B1", "B2", "B3"), ("B1", "B3", "B2"), ("B2", "B1", "B3"),
    ("B2", "B3", "B1"), ("B3", "B1", "B2"), ("B3", "B2", "B1"),
]

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
