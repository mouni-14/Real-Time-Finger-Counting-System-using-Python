import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        total_fingers = 0

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Identify if it's Left or Right hand
                hand_label = result.multi_handedness[hand_idx].classification[0].label  # 'Left' or 'Right'

                # Get landmark positions
                landmarks = hand_landmarks.landmark
                h, w, _ = frame.shape
                landmark_positions = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                fingers = []

                # Correct thumb logic based on hand
                if hand_label == "Right":
                    fingers.append(1 if landmark_positions[4][0] > landmark_positions[3][0] else 0)
                else:  # Left hand (reverse logic)
                    fingers.append(1 if landmark_positions[4][0] < landmark_positions[3][0] else 0)

                # Other 4 fingers
                for tip in [8, 12, 16, 20]:
                    fingers.append(1 if landmark_positions[tip][1] < landmark_positions[tip - 2][1] else 0)

                total_fingers += fingers.count(1)

        # Show combined count
        cv2.putText(frame, f"Total Fingers: {total_fingers}", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

        cv2.imshow("Finger Counting 1-10", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()