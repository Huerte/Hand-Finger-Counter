import cv2 # Use to open web cam
import mediapipe as mp


mp_hand = mp.solutions.hands
hands = mp_hand.Hands()
mp_drawing_utils = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0) # The '0' parameter is use for default cam and '1' for external cam

while cam.isOpened():

    success, img = cam.read()

    if not success:
        break

    # upside down -> 0
    # mirrored -> 1
    # both -> -1
    img = cv2.flip(img, 1) # This will invert the cam shot

    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for hand_landmark, multi_handedness in zip(hand_landmarks, result.multi_handedness):
            mp_drawing_utils.draw_landmarks(
                img,
                hand_landmark,
                mp_hand.HAND_CONNECTIONS,

                # Optional Styling
                mp_drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2), # For red dots (optional)
                mp_drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)  # For green connected lines (optional)
            )

            label = multi_handedness.classification[0].label
        
            # See README.md for finger landmark ID reference
            thumb = hand_landmark.landmark[4].x < hand_landmark.landmark[3].x
            index = hand_landmark.landmark[8].y < hand_landmark.landmark[6].y
            middle = hand_landmark.landmark[12].y < hand_landmark.landmark[10].y
            ring = hand_landmark.landmark[16].y < hand_landmark.landmark[14].y
            pinky = hand_landmark.landmark[20].y < hand_landmark.landmark[18].y

            if label == 'Left':
                thumb = not thumb
        
            fingers_up = [thumb, index, middle, ring, pinky]
            count = fingers_up.count(True)

            x = int(hand_landmark.landmark[0].x * img.shape[1])
            y = int(hand_landmark.landmark[0].y * img.shape[0]) - 20

            cv2.putText(img, f'{label}: {count}', (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Hand Detector', img)

    if cv2.waitKey(1) == 27: # ESC key for exit
        break


cam.release()
cv2.destroyAllWindows()