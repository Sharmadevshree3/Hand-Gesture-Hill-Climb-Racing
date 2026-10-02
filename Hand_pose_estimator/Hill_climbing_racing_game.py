# SECTION 1: IMPORTS
import cv2                  # OpenCV: reads the webcam and shows the window
import mediapipe as mp      # MediaPipe: detects the hand and its 21 landmark points
import time                 # Used for a short delay at the start (time.sleep)

# These come from YOUR directkeys.py file (it must be in the same folder).
# right_pressed / left_pressed : key codes for the Right and Left arrow keys
# PressKey(code)               : pretends the keyboard key is pushed down
# ReleaseKey(code)             : pretends the keyboard key is let go
from directkeys import right_pressed, left_pressed, PressKey, ReleaseKey



# SECTION 2: GAME KEY SETTINGS
# Give the keys meaningful names so the code below reads easily.
# In Hill Climb Racing: Left arrow = brake, Right arrow = gas (accelerator).
brake_key = left_pressed
accelerator_key = right_pressed



# SECTION 3: MEDIAPIPE SETUP
# mp.solutions.drawing_utils = helper that draws the dots and lines
# of the hand skeleton on top of the image.
mp_draw = mp.solutions.drawing_utils

# mp.solutions.hands = the hand detection module.
# It finds a hand in an image and gives 21 landmark points per hand.
mp_hand = mp.solutions.hands

# The 21 landmarks are numbered 0 to 20. These are the numbers of the
# 5 FINGERTIPS:
#   4  = thumb tip
#   8  = index finger tip
#   12 = middle finger tip
#   16 = ring finger tip
#   20 = pinky tip
# (Landmark 0 is the wrist. Each finger has 4 points from base to tip.)
tipIds = [4, 8, 12, 16, 20]


# SECTION 4: WEBCAM SETUP
# Open the default webcam. 0 = first camera, 1 = second camera, etc.
video = cv2.VideoCapture(0)


# SECTION 5: KEY-HOLDING HELPER FUNCTION
# This variable remembers which key is held down right now.
# None means no key is being held.
current_key = None


def set_key(new_key):
    """
    Makes sure ONLY the wanted key is held down.
    new_key = brake_key, accelerator_key, or None (release everything).
    It only acts when the wanted key changes. Otherwise it would send
    thousands of key presses per second, because it is called every frame.
    """
    global current_key  # we want to change the variable defined outside

    # Same key as before? Nothing to do, so stop here.
    if new_key == current_key:
        return

    # A different key was being held, so let it go first.
    if current_key is not None:
        ReleaseKey(current_key)

    # Now push down the new key (if there is one).
    if new_key is not None:
        PressKey(new_key)

    # Remember the new state for the next frame.
    current_key = new_key


# SECTION 6: START DELAY
# Waits 10 seconds so you can click on the game window.
# Keys are sent to whichever window is active, so the game must be focused.
time.sleep(10.0)



# SECTION 7: MAIN PROGRAM (HAND DETECTION LOOP)
# This creates the hand detector. "with" closes it automatically at the end.
#
# min_detection_confidence = how sure MediaPipe must be (0 to 1) before it
#     says "yes, this is a hand". Higher = fewer false detections.
# min_tracking_confidence  = once a hand is found, how sure it must be to
#     keep following it. If it drops below, MediaPipe searches again.
# max_num_hands = 1 -> only track one hand, which is faster and avoids confusion.
with mp_hand.Hands(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5,
                   max_num_hands=1) as hands:

    # This loop runs once per camera frame, again and again, until you quit.
    while True:

        # 7.1 READ A FRAME FROM THE CAMERA
        # ret   = True if the frame was read successfully
        # image = the frame itself (a numpy array, in BGR color order)
        ret, image = video.read()
        if not ret:
            break  # camera failed or was disconnected, so stop the loop

        # 7.2 PREPARE THE IMAGE FOR MEDIAPIPE
        # OpenCV uses BGR color order, but MediaPipe expects RGB.
        # If you skip this conversion, detection works poorly.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Marking the image as read-only lets MediaPipe use it without
        # making a copy. This is a small speed optimization.
        image.flags.writeable = False

        # THE KEY MEDIAPIPE LINE: find the hand in this frame.
        # "results" now holds the landmarks of every detected hand.
        results = hands.process(image)

        # Make the image editable again, because we want to draw on it.
        image.flags.writeable = True

        # Convert back to BGR, because cv2.imshow() expects BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 7.3 CONVERT LANDMARKS INTO A SIMPLE LIST
        # lmList will hold one [id, x, y] entry per landmark, for example:
        # [[0, 320, 400], [1, 300, 380], ... , [20, 350, 250]]
        # It stays empty if no hand is visible.
        lmList = []

        # results.multi_hand_landmarks is None when no hand is found,
        # so this "if" only runs when at least one hand is detected.
        if results.multi_hand_landmarks:

            # Loop over every detected hand (we allowed only 1 above).
            for hand_landmark in results.multi_hand_landmarks:

                # Take the first hand's landmarks.
                myHands = results.multi_hand_landmarks[0]

                # enumerate gives us the number (id) AND the point (lm) together.
                # id = 0 to 20 (which landmark), lm = that landmark's position.
                for id, lm in enumerate(myHands.landmark):

                    # image.shape gives (height, width, channels).
                    h, w, c = image.shape

                    # IMPORTANT: MediaPipe gives positions as FRACTIONS of the
                    # image size, between 0.0 and 1.0 (for example x = 0.5 is
                    # the middle). We multiply by the width and height to get
                    # real PIXEL positions that OpenCV can use.
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Save the id with its pixel position.
                    lmList.append([id, cx, cy])

                # Draw the dots and connecting lines of the hand on the image.
                # HAND_CONNECTIONS tells it which dots to join with lines.
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

        
        # 7.4 DECIDE WHICH KEY SHOULD BE PRESSED
        # Start by assuming "no key". It changes only if we recognise a gesture.
        wanted_key = None

        # Only count fingers if a hand was actually detected.
        if len(lmList) != 0:

            # fingers will hold 5 values (1 = finger up, 0 = finger down),
            # in the order: thumb, index, middle, ring, pinky.
            fingers = []

            # ---- THUMB ----
            # The thumb folds sideways, not up and down, so we compare the
            # X (horizontal) positions. lmList[id][1] is x, lmList[id][2] is y.
            # tipIds[0] = 4  (thumb tip)
            # tipIds[0] - 1 = 3  (the joint just below the tip)
            # If the tip is further right than that joint, the thumb is open.
            # NOTE: this works for a RIGHT hand with the palm facing the
            # camera. For the opposite direction, change > to <.
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)   # thumb is open
            else:
                fingers.append(0)   # thumb is closed

            # ---- THE OTHER FOUR FINGERS ----
            # id goes 1, 2, 3, 4 -> index, middle, ring, pinky.
            for id in range(1, 5):

                # These fingers move up and down, so we compare Y positions.
                # In images, Y = 0 is the TOP, so a smaller Y means higher up.
                # tipIds[id]     = the fingertip
                # tipIds[id] - 2 = the middle joint of the same finger
                # If the tip is ABOVE the middle joint, the finger is up.
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)   # finger is up
                else:
                    fingers.append(0)   # finger is folded

            # Count how many fingers are up (how many 1s are in the list).
            total = fingers.count(1)

            # ---- DECIDE THE ACTION ----
            if total == 0:
                # FIST (no fingers up) means BRAKE.
                # Draw a green box with the text "BRAKE" on the screen.
                cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 5)
                wanted_key = brake_key

            elif total == 5:
                # OPEN PALM (all 5 fingers up) means GAS.
                # Draw a green box with the text "GAS" on the screen.
                cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, " GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 5)
                wanted_key = accelerator_key

            # Any other number of fingers keeps wanted_key as None,
            # so both keys are released (the car just coasts).

        # 7.5 SEND THE KEY TO THE GAME
        # Presses/releases keys as needed. If wanted_key is None (no hand or
        # a different gesture), this releases whatever key was being held.
        set_key(wanted_key)

        # 7.6 SHOW THE CAMERA WINDOW
        cv2.imshow("Frame", image)

        # Wait 1 ms for a key press. Press 'q' in the camera window to quit.
        k = cv2.waitKey(1)
        if k == ord('q'):
            break



# SECTION 8: CLEAN UP
# Release any key still held, so it doesn't stay stuck down in Windows.
set_key(None)

# Free the webcam and close the OpenCV window.
video.release()
cv2.destroyAllWindows()