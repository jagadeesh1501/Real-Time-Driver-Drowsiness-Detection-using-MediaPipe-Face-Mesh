import cv2
import mediapipe as mp
import math
import winsound
import threading

# ---------- FUNCTIONS ----------

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def eye_aspect_ratio(landmarks, eye_points, w, h):
    points = []

    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        points.append((x, y))

    vertical1 = euclidean_distance(points[1], points[5])
    vertical2 = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear


def play_alarm():
    for _ in range(5):
        winsound.Beep(1500, 250)
        winsound.Beep(1000, 250)

# ---------- MEDIAPIPE ----------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------- EYE LANDMARKS ----------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------- VARIABLES ----------

blink_count = 0
closed_frames = 0

EAR_THRESHOLD = 0.22
DROWSY_FRAMES = 20

alarm_on = False

# ---------- CAMERA ----------

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            continue

        # Flip first (mirror view)
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                # Draw Face Mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )

                landmarks = face_landmarks.landmark

                left_ear = eye_aspect_ratio(
                    landmarks,
                    LEFT_EYE,
                    w,
                    h
                )

                right_ear = eye_aspect_ratio(
                    landmarks,
                    RIGHT_EYE,
                    w,
                    h
                )

                avg_ear = (left_ear + right_ear) / 2

                # ---------- BLINK DETECTION ----------

                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1

                else:
                    if closed_frames > 2:
                        blink_count += 1

                    closed_frames = 0
                    alarm_on = False

                # ---------- DROWSINESS DETECTION ----------

                if closed_frames > DROWSY_FRAMES:

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    if not alarm_on:
                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

                        alarm_on = True

                # ---------- BLINK COUNT ----------

                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

                # ---------- EAR VALUE ----------

                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()