import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
SPEC_HAND_LANDMARKS = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.THUMB_CMC,
                       mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                       mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                       mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP,
                       mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_MCP,
                       mp_hands.HandLandmark.PINKY_TIP]

# For static images:
IMAGE_FILES = ["/media/work/Workspace/Projects/GestureDetection/src/neural/dataset/paper/0a3UtNzl5Ll3sq8K.png"]

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def detect_hand_pretrained(file):
    output = []
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for hand_landmarks in results.multi_hand_landmarks:
      for point in SPEC_HAND_LANDMARKS:
        point_x = round(hand_landmarks.landmark[point].x, 3)
        point_y = round(hand_landmarks.landmark[point].y, 3)

        output.append([point_y, point_x])
    
    return output