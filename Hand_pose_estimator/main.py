import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

video = cv2.VideoCapture(0)

with mp_hand.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, image = video.read()
        if not ret:
            break

        # MediaPipe needs RGB, OpenCV gives BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw landmarks if a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hand.HAND_CONNECTIONS
                )

        cv2.imshow("frame", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()