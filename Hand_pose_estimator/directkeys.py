import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# Arrow key scan codes
right_pressed = 0x4D
left_pressed = 0x4B
up_pressed = 0x48
down_pressed = 0x50

PUL = ctypes.POINTER(ctypes.c_ulong)

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# Arrow keys are "extended" keys, so they need the extended flag
EXTENDED = {0x4D, 0x4B, 0x48, 0x50}


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


def _send(hexKeyCode, flags):
    if hexKeyCode in EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, flags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def PressKey(hexKeyCode):
    _send(hexKeyCode, KEYEVENTF_SCANCODE)


def ReleaseKey(hexKeyCode):
    _send(hexKeyCode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)