import cv2
import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands  

count = 0
position = None

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose, \
    mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera")
            break
        
        image = cv2.resize(image, (1080, 720))
        image = cv2.flip(image, 1) 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        pose_result = pose.process(image_rgb)
        hands_result = hands.process(image_rgb)

        lmlist = [] 

        if pose_result.pose_landmarks:
            h, w, _ = image.shape  

            mp_drawing.draw_landmarks(
                image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for id, lm in enumerate(pose_result.pose_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h) 
                lmlist.append([id, x, y])

        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        if len(lmlist) != 0:
            if ((lmlist[12][2] - lmlist[14][2]) >= 15 and (lmlist[11][2] - lmlist[13][2]) >= 15):
                position = "down"
            if ((lmlist[12][2] - lmlist[14][2]) <= 5 and (lmlist[11][2] - lmlist[13][2]) <= 5) and position == "down":
                position = "up"
                count += 1
                print(f"Push-ups count: {count}")

        cv2.imshow("Push-up Detection", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
