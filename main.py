import math
import cv2
import mediapipe as mp


class DragAndDropCircle:
    def __init__(self, px, py, radius=50, color_index=0, brush_thickness=10):
        self.radius = radius
        self.color_index = color_index
        self.px = px
        self.py = py
        self.brush_thickness = brush_thickness  # новая переменная для толщины кисточки
        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255),
                       (0, 255, 255), (0, 0, 0)]

    def update(self, new_px, new_py):
        cx, cy = self.px, self.py
        if math.hypot(new_px - cx, new_py - cy) > 50:
            self.px, self.py = new_px, new_py

    def get_color(self):
        return self.colors[self.color_index % len(self.colors)]


colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255),
                       (0, 255, 255), (0, 0, 0)]
color_changed = False
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
circles = []
color_index = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        myHand = results.multi_hand_landmarks[0]
        h, w, s = img.shape

        for i in [0, 4, 8, 12, 16, 20]:
            cx, cy = int(myHand.landmark[i].x * w), int(myHand.landmark[i].y * h)
            cv2.circle(img, (cx, cy), 7, (255, 255, 255), cv2.FILLED)

        x1 = myHand.landmark[8].x * w
        x2 = myHand.landmark[4].x * w
        y1 = myHand.landmark[8].y * h
        y2 = myHand.landmark[4].y * h

        x3 = myHand.landmark[16].x * w
        x4 = x2
        y3 = myHand.landmark[16].y * h
        y4 = y2

        x5 = myHand.landmark[12].x * w
        x6 = x2
        y5 = myHand.landmark[12].y * h
        y6 = y2

        length = math.hypot(abs(x1 - x5), abs(y1 - y5))
        length1 = math.hypot(abs(x3 - x4), abs(y3 - y4))
        length2 = math.hypot(abs(x5 - x6), abs(y5 - y6))

        if length < 40:
            circle = DragAndDropCircle(x1, y1, color_index=color_index)
            circles.append(circle)

        elif length1 < 40:
            circles = []

        elif length2 < 40:
            if not color_changed:
                color_index += 1
                color_changed = True
        else:
            color_changed = False

    if circles:
        distance_between_fingers = math.hypot(x1 - x2, y1 - y2)
        circles[-1].brush_thickness = int(distance_between_fingers / 2)

    for circle in circles:
        cv2.circle(img, (int(circle.px), int(circle.py)), circle.radius, circle.get_color(),
                   circle.brush_thickness)  # используйте brush_thickness

        # Отобразите текущий выбранный цвет в углу экрана
    current_color = colors[color_index % len(colors)]
    cv2.rectangle(img, (10, 10), (60, 60), current_color, -1)
    cv2.putText(img, "Current Color", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for circle in circles:
        cv2.circle(img, (int(circle.px), int(circle.py)), circle.radius, circle.get_color(), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break


