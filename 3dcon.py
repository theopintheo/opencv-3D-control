import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# === Global State Variables === #
class State:
    zoom_level = -6.0
    rotation_x = 0.0
    rotation_y = 0.0
    last_x = None
    last_y = None
    initial_distance = None

# === Mediapipe Init === #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

# === Draw a 3D Cube === #
def draw_cube():
    glBegin(GL_QUADS)
    # Top (Y+)
    glColor3f(1, 0, 0)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)

    # Bottom (Y-)
    glColor3f(0, 1, 0)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)

    # Front (Z+)
    glColor3f(0, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)

    # Back (Z-)
    glColor3f(1, 1, 0)
    glVertex3f(1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)

    # Left (X-)
    glColor3f(1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)

    # Right (X+)
    glColor3f(0, 1, 1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, -1, -1)
    glEnd()

# === Display Function (OpenGL + Camera Input) === #
def display():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, State.zoom_level)
    glRotatef(State.rotation_x, 1.0, 0.0, 0.0)
    glRotatef(State.rotation_y, 0.0, 1.0, 0.0)
    draw_cube()

    # === Gesture Logic === #
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        x1 = int(lm[4].x * w)   # Thumb tip
        y1 = int(lm[4].y * h)
        x2 = int(lm[8].x * w)   # Index tip
        y2 = int(lm[8].y * h)

        cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)

        # Pinch distance (zoom control)
        distance = np.hypot(x2 - x1, y2 - y1)
        if State.initial_distance is None:
            State.initial_distance = distance
        else:
            delta = (distance - State.initial_distance) / 100
            State.zoom_level -= delta
            State.zoom_level = np.clip(State.zoom_level, -15, -3)
            State.initial_distance = distance

        # Wrist movement (rotation)
        wrist = lm[0]
        current_x, current_y = wrist.x, wrist.y

        if State.last_x is not None:
            dx = current_x - State.last_x
            dy = current_y - State.last_y

            if abs(dx) > 0.01:
                State.rotation_y += dx * 100
            if abs(dy) > 0.01:
                State.rotation_x += dy * 100

        State.last_x = current_x
        State.last_y = current_y

    cv2.imshow("Webcam Feed", frame)
    glutSwapBuffers()

# === Window Reshape Function === #
def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, float(w) / float(h), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# === OpenGL Init and Main Loop === #
def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"3D Object AR Control")
    glEnable(GL_DEPTH_TEST)
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutReshapeFunc(reshape)
    glutMainLoop()

if __name__ == "__main__":
    main()
