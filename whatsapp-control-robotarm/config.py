"""User-editable settings. Everything you'd normally want to tweak lives here."""

# --- Serial ---------------------------------------------------------------
# CH9102 chip = the M5Stack Basic on top of the myCobot 280.
PORT = "COM10"
BAUD = 115200

# --- Access control -------------------------------------------------------
# The bot runs as a linked device on account A. Commands come from account B, so
# they arrive as normal incoming messages and this stays off. Set it True only if
# you also want to drive the arm by messaging A from A itself (Note to Self).
ALLOW_SELF_COMMANDS = False

# Every reply the bot sends is prefixed with this, and any incoming message
# starting with it is ignored. Without that, the bot's own replies come back as
# sent-by-you messages and it starts talking to itself.
BOT_PREFIX = "[arm] "

# ONLY these numbers can drive the arm. International format, no +, no spaces.
# Leave empty and the bot refuses every command (fail closed) until you fill it.
# Send yourself a message and check the console log to learn your own number.
# Entries may be a phone number OR a LID (WhatsApp's privacy id, a long number
# unrelated to the phone number). Incoming messages are matched against both
# forms, so whichever the console logs will work.
ALLOWED_SENDERS = [
    # "441234567890",   # <- the phone you send FROM, international format, no +
    # "123456789012345", # <- ...or the LID the console logs for it, if different
]

# --- Motion ---------------------------------------------------------------
# How long to wait for the arm to report it reached a target before giving up and
# moving on. pymycobot's sync_send_angles blocks for this whole duration if the
# servos settle just outside its arrival tolerance -- which happens routinely at
# all-zeros -- so a too-generous value shows up as the NEXT command seeming slow.
MOVE_TIMEOUT = 8

# Skip a move entirely if every joint is already this close (degrees) to the
# target. Stops "home" from costing a full timeout when you're already home.
ARRIVAL_TOLERANCE = 2.0

DEFAULT_SPEED = 40          # 1-100, deliberately gentle
MAX_SPEED = 70              # ceiling, even if someone messages "speed 100"
STEP_MM = 30                # default distance for "up" / "left" with no number
MAX_STEP_MM = 120           # ceiling for a single relative move

# Joint limits for the myCobot 280 M5 (degrees), from Elephant Robotics' spec:
# J1-J5 are -165~+165. J6 is listed as -175~+175 on one spec page and -179~+179
# on another, so we take the conservative one.
JOINT_LIMITS = [
    (-165, 165),   # J1
    (-165, 165),   # J2
    (-165, 165),   # J3
    (-165, 165),   # J4
    (-165, 165),   # J5
    (-175, 175),   # J6
]

# --- Named poses ----------------------------------------------------------
# "go pickup" -> these six joint angles. Teach them by hand: run `python teach.py`,
# physically pose the arm, and it prints the angles to paste in here.
PRESETS = {
    "home":    [0, 0, 0, 0, 0, 0],
    "ready":   [0, -30, -60, 0, 0, 0],
    "pickup":  [-45, -40, -50, -10, 0, 0],
    "dropoff": [45, -40, -50, -10, 0, 0],
}

# --- Routines -------------------------------------------------------------
# Named sequences of poses, played back in order. Each step is
# ([j1..j6], speed) -- speed is per-step so moves can be snappy or slow.
# Angles are clamped to JOINT_LIMITS on the way out, and `stop` aborts a routine
# between steps. Teach your own with teach.py and paste them in.
ROUTINES = {
    "dance": [
        ([  0, -20, -20,   0,   0,   0], 50),   # stand up
        ([ 40, -30, -25,   0,  30,   0], 70),   # lean right
        ([-40, -30, -25,   0, -30,   0], 70),   # lean left
        ([ 40, -30, -25,   0,  30,   0], 80),   # right again, quicker
        ([-40, -30, -25,   0, -30,   0], 80),
        ([  0, -10, -45,   0,  50,   0], 70),   # bob down
        ([  0, -45, -10,   0, -20,   0], 70),   # bob up
        ([  0, -20, -20,   0,   0,  60], 70),   # twist wrist
        ([  0, -20, -20,   0,   0, -60], 70),
        ([  0,   0,   0,   0,   0,   0], 50),   # home, take a bow
    ],
    "wave": [
        ([  0, -40, -40,   0,  40,   0], 50),   # raise up
        ([  0, -40, -40,  30,  40,   0], 80),
        ([  0, -40, -40, -30,  40,   0], 80),
        ([  0, -40, -40,  30,  40,   0], 80),
        ([  0, -40, -40, -30,  40,   0], 80),
        ([  0,   0,   0,   0,   0,   0], 50),
    ],
    "nod": [
        ([  0, -20, -20,   0,   0,   0], 50),
        ([  0, -20, -20,  40,   0,   0], 70),
        ([  0, -20, -20, -10,   0,   0], 70),
        ([  0, -20, -20,  40,   0,   0], 70),
        ([  0,   0,   0,   0,   0,   0], 50),
    ],
}

# --- Pick and place -------------------------------------------------------
# The cube sits at a fixed spot. The sequence is:
#
#   1. move to START_ANGLES          (approach pose, gripper open)
#   2. descend to PICK_Z             (straight down, x/y unchanged)
#   3. close the gripper             (grab the cube)
#   4. ascend back to travel height
#   5. slide sideways to PLACE_Y     (over the box, x/z unchanged)
#   6. descend to PLACE_Z
#   7. open the gripper              (drop it in the box)
#   8. ascend back to travel height
#
# Steps 2 and onward are Cartesian: only the named axis changes, so the wrist
# orientation from START_ANGLES is preserved throughout.
PICK_PLACE_TAUGHT = True

PICK_PLACE = {
    "start_angles": [0, -40, 0, -45, 0, 45],

    "pick_z":   100,    # absolute z (mm) to descend to when gripping
    "place_y":   60,    # absolute y (mm) of the box
    "place_z":  100,    # absolute z (mm) to descend to before releasing.
                        # If your box has walls, raise this so the cube clears
                        # the rim and drops in -- start high and lower it.

    "travel_z": None,   # z used for the two ascents. None = whatever z the arm
                        # is at after START_ANGLES, read live. Set a number to
                        # force a specific travel height.
}

PICK_PLACE_SPEED = 30       # deliberately slower than free motion
GRIPPER_SETTLE = 1.5        # seconds to let the gripper finish opening/closing

# --- LED ------------------------------------------------------------------
# The Atom's RGB LED. Cycled one colour per step while a routine plays, then
# returned to IDLE_COLOR. Set ROUTINE_COLORS = [] to disable the effect.
IDLE_COLOR = (0, 255, 0)        # green: connected and waiting

ROUTINE_COLORS = [
    (255,   0,   0),    # red
    (255, 120,   0),    # orange
    (255, 255,   0),    # yellow
    (  0, 255,   0),    # green
    (  0, 255, 255),    # cyan
    (  0,   0, 255),    # blue
    (160,   0, 255),    # violet
    (255,   0, 180),    # magenta
]

# --- Gripper --------------------------------------------------------------
GRIPPER_OPEN = 100          # 0 = fully closed, 100 = fully open
GRIPPER_CLOSED = 10
GRIPPER_SPEED = 50
