import cv2, math, time
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
TIPS, PIPS = [8, 12, 16, 20], [6, 10, 14, 18]


def poly(n, r=1.0, start=0.0):
    return np.array([[math.cos(start + 2 * math.pi * i / n),
                      math.sin(start + 2 * math.pi * i / n)] for i in range(n)]) * r


SHAPES = {"Square": poly(4, 1.414, math.pi / 4),
          "Triangle": poly(3, 1.2, -math.pi / 2),
          "Hexagon": poly(6)}
SHAPE_NAMES = list(SHAPES)
COLORS = [(255, 120, 60), (60, 200, 60), (60, 60, 255), (0, 220, 255), (255, 60, 200)]
BASE = 70


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_fingers(hand, label):
    lm = hand.landmark
    f = [1 if (lm[4].x < lm[3].x) == (label == "Right") else 0]
    for t, p in zip(TIPS, PIPS):
        f.append(1 if lm[t].y < lm[p].y else 0)
    return f


def get_gesture(f):
    return {(0, 0, 0, 0, 0): "Fist", (1, 1, 1, 1, 1): "Open Palm",
            (0, 1, 0, 0, 0): "Pointing", (0, 1, 1, 0, 0): "Peace",
            (1, 0, 0, 0, 0): "Thumbs Up"}.get(tuple(f), "Unknown")


def hand_info(hand, label, w, h):
    lm = hand.landmark
    p4 = (lm[4].x * w, lm[4].y * h)
    p8 = (lm[8].x * w, lm[8].y * h)
    size = dist((lm[0].x * w, lm[0].y * h), (lm[9].x * w, lm[9].y * h))
    mid = ((p4[0] + p8[0]) / 2, (p4[1] + p8[1]) / 2)
    pinch = dist(p4, p8) / size < 0.35
    return {"mid": mid, "pinch": pinch,
            "gesture": get_gesture(get_fingers(hand, label))}


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Object state
center = np.array([480.0, 360.0])
scale, angle = 1.0, 0.0
shape_i, color_i = 0, 0
grabbed, two_hand = False, False
offset = np.zeros(2)
d0 = a0 = s0 = ang0 = 0.0
last_action = 0
mode, gesture_txt = "Idle", "None"
paused, last_pause = False, 0

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam could not be opened")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        infos = []
        if res.multi_hand_landmarks:
            for hd, hn in zip(res.multi_hand_landmarks, res.multi_handedness):
                mp_draw.draw_landmarks(frame, hd, mp_hands.HAND_CONNECTIONS)
                infos.append(hand_info(hd, hn.classification[0].label, w, h))

        pinching = [i for i in infos if i["pinch"]]
        mode, gesture_txt = "Idle", infos[0]["gesture"] if infos else "None"

        # Pause toggle (Open Palm)
        if infos and infos[0]["gesture"] == "Open Palm" and time.time() - last_pause > 1.5:
            paused = not paused
            last_pause = time.time()

        if paused:
            grabbed, two_hand = False, False
            mode = "PAUSED (show Open Palm to resume)"

        elif len(pinching) == 2:  # Scale + Rotate
            grabbed = False
            m1, m2 = pinching[0]["mid"], pinching[1]["mid"]
            d = max(dist(m1, m2), 1)
            a = math.atan2(m2[1] - m1[1], m2[0] - m1[0])
            if not two_hand:
                two_hand, d0, a0, s0, ang0 = True, d, a, scale, angle
            scale = float(np.clip(s0 * d / d0, 0.3, 4.0))
            angle = ang0 + (a - a0)
            mode = "Scale + Rotate"
            cv2.line(frame, (int(m1[0]), int(m1[1])), (int(m2[0]), int(m2[1])), (0, 255, 255), 2)

        else:
            two_hand = False
            if len(pinching) == 1:  # Move
                m = np.array(pinching[0]["mid"])
                if not grabbed and dist(m, center) < BASE * scale * 1.5:
                    grabbed, offset = True, center - m
                if grabbed:
                    center = center * 0.6 + (m + offset) * 0.4
                    mode = "Moving"
                else:
                    mode = "Pinch on the object to grab it"
            else:
                grabbed = False
                # Gesture actions (cooldown 1 sec)
                if infos and time.time() - last_action > 1:
                    g = infos[0]["gesture"]
                    if g == "Peace":
                        color_i = (color_i + 1) % len(COLORS)
                        last_action = time.time()
                    elif g == "Fist":
                        shape_i = (shape_i + 1) % len(SHAPES)
                        last_action = time.time()
                    elif g == "Thumbs Up":
                        center, scale, angle = np.array([w / 2, h / 2]), 1.0, 0.0
                        last_action = time.time()

        # Graphics transformation: scale -> rotate -> translate
        pts = SHAPES[SHAPE_NAMES[shape_i]] * BASE * scale
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        pts = (pts @ R.T + center).astype(np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], COLORS[color_i])
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.polylines(frame, [pts], True,
                      (0, 255, 255) if (grabbed or two_hand) else (255, 255, 255), 3)

        for i in infos:  # pinch feedback
            cv2.circle(frame, (int(i["mid"][0]), int(i["mid"][1])), 12,
                       (0, 255, 0) if i["pinch"] else (0, 0, 255), -1)

        # Feedback panel
        cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 0), -1)
        cv2.putText(frame, f"Mode: {mode}   Gesture: {gesture_txt}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Shape: {SHAPE_NAMES[shape_i]}  Scale: {scale:.2f}  Angle: {math.degrees(angle):.0f}",
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Pinch=Move | 2 hands pinch=Scale/Rotate | Palm=Pause | Peace=Color | Fist=Shape | ThumbsUp=Reset | q=Quit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Error / status feedback
        if not infos:
            cv2.putText(frame, "No hand detected - show your hand to the camera",
                        (15, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        if paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 90, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)

        cv2.imshow("GestureX - Gesture Controlled Application", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()