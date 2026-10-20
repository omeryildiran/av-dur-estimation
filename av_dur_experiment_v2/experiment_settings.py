"""Every parameter of the v2 experiment. Change values here and nowhere else.

Values marked "v1" reproduce the 2025 data collection. Values marked "v2" are deliberate
changes; README.md explains each one.
"""

EXPERIMENT_NAME = "av_duration_v2"

# ── Display ───────────────────────────────────────────────────────────────────
# A profile gives the window, the physical geometry used to convert degrees to pixels, and
# the refresh rate at which every duration is computed (the session checks it at start).
# "v1_profile" is the lab setup: the geometry every v1 script used (MacBook-Air numbers,
# disk radius 42.7 px) on the 120 Hz lab display (screen 1). Profiles marked testing_only
# are for trying the experiment on other machines: they keep the v1 pixel geometry, run at
# that display's own rate, and save to data_test/ so they never mix with real data.
MONITOR_PROFILES = {
    "v1_profile": dict(window_px=800, width_cm=25.0, height_cm=20.0, distance_cm=40.0,
                       screen=1, fullscreen=True, frame_rate_hz=120.0),
    "office_test": dict(window_px=800, width_cm=25.0, height_cm=20.0, distance_cm=40.0,
                        screen=0, fullscreen=True, frame_rate_hz=75.0, testing_only=True),   # Mac mini + Dell S2722DZ
    "laptop_test": dict(window_px=800, width_cm=25.0, height_cm=20.0, distance_cm=40.0,
                        screen=0, fullscreen=True, frame_rate_hz=60.0, testing_only=True),   # laptop set to 60 Hz
}
DEFAULT_MONITOR = "v1_profile"

EXPECTED_FRAME_RATE_HZ = MONITOR_PROFILES[DEFAULT_MONITOR]["frame_rate_hz"]   # the lab rate (120 Hz)
FRAME_RATE_TOLERANCE_HZ = 1.0       # v2: measured rate must be within this of the profile's rate
MAX_IRREGULAR_FRAMES = 0.05         # v2: share of frame intervals more than 1.5 ms off the median
BACKGROUND_COLOR = "gray"           # v1
DISK_RADIUS_DEG = 1.5               # v1: 3 deg disk
DISK_LINE_WIDTH_PX = 1.5            # v1 wrote `lineWidht = 5` (typo), so PsychoPy's default 1.5 px was shown
TEXT_HEIGHT_PX = 30                 # v1

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000                 # v1
AUDIO_LATENCY_MODE = 3              # v1 set prefs.general (ignored by PsychoPy), so the default 3 was used
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

# ── Staircases (v1 rules) ─────────────────────────────────────────────────────
# Rule "xDyU": x "test longer" answers in a row step the test down, y "standard longer"
# answers in a row step it up. Levels are delta_dur_percents = test/standard - 1.
# The step halves at the first two reversals, then stays fixed.
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
        # v2 replaces it with a 1D1U that starts at the top of the range.
        staircases=("1D1U", "1D1U_top", "2D1U", "1D2U"), trials_per_staircase=45,  # v1: 388 trials
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
