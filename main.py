import cv2 as cv
import utils
import classes.GazeController as GazeController
from classes.GazeController import Direction
import numpy as np

##################### tak zaznacze zmiany :)
direction_to_letter = {
    (Direction.LEFT, Direction.LEFT, Direction.LEFT): 'a',
    (Direction.LEFT, Direction.LEFT, Direction.RIGHT): 'b',
    (Direction.LEFT, Direction.RIGHT, Direction.LEFT): 'c',
    (Direction.LEFT, Direction.RIGHT, Direction.RIGHT): 'd',
    (Direction.LEFT, Direction.UP, Direction.LEFT): 'e',
    (Direction.LEFT, Direction.UP, Direction.RIGHT): 'f',
    (Direction.LEFT, Direction.DOWN, Direction.LEFT): 'g',
    (Direction.LEFT, Direction.DOWN, Direction.RIGHT): 'h',
    (Direction.RIGHT, Direction.LEFT, Direction.LEFT): 'i',
    (Direction.RIGHT, Direction.LEFT, Direction.RIGHT): 'j',
    (Direction.RIGHT, Direction.RIGHT, Direction.LEFT): 'k',
    (Direction.RIGHT, Direction.RIGHT, Direction.RIGHT): 'l',
    (Direction.RIGHT, Direction.UP, Direction.LEFT): 'm',
    (Direction.RIGHT, Direction.UP, Direction.RIGHT): 'n',
    (Direction.RIGHT, Direction.DOWN, Direction.LEFT): 'o',
    (Direction.RIGHT, Direction.DOWN, Direction.RIGHT): 'p',
    (Direction.UP, Direction.LEFT, Direction.LEFT): 'q',
    (Direction.UP, Direction.LEFT, Direction.RIGHT): 'r',
    (Direction.UP, Direction.RIGHT, Direction.LEFT): 's',
    (Direction.UP, Direction.RIGHT, Direction.RIGHT): 't',
    (Direction.UP, Direction.UP, Direction.LEFT): 'u',
    (Direction.UP, Direction.UP, Direction.RIGHT): 'v',
    (Direction.UP, Direction.DOWN, Direction.LEFT): 'w',
    (Direction.UP, Direction.DOWN, Direction.RIGHT): 'x',
    (Direction.DOWN, Direction.LEFT, Direction.LEFT): 'y',
    (Direction.DOWN, Direction.LEFT, Direction.RIGHT): 'z',
    (Direction.DOWN, Direction.RIGHT, Direction.LEFT): '!',
    (Direction.DOWN, Direction.RIGHT, Direction.RIGHT): '@',
    (Direction.DOWN, Direction.UP, Direction.LEFT): '#',
    (Direction.DOWN, Direction.UP, Direction.RIGHT): '#',
    (Direction.DOWN, Direction.DOWN, Direction.LEFT): '$',
    (Direction.DOWN, Direction.DOWN, Direction.RIGHT): ' ',
    # Dodaj więcej mapowań, jeśli potrzebujesz
}


FONTS = cv.FONT_HERSHEY_COMPLEX
color = [utils.YELLOW, utils.PINK]
direction = ["NO", "LEFT", "RIGHT", "UP", "DOWN", "CENTER"]

camera = cv.VideoCapture(1)
gaze_controller = GazeController.GazeController()
result = 1
do_calibration = False
reset_calibration = False
zoom = 1.0

cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

################################
# Inicjalizacja tablicy do przechowywania kierunków patrzenia
direction_history = []
previous_direction = Direction.CENTER
typed_text = ""


##############################
def draw_text_area(frame):
    # Ustawienie parametrów obszaru tekstu
    text_area_height = 50
    text_area_color = (0, 0, 0)
    text_area_thickness = -1  # Ustawienie na -1, aby wypełnić obszar kolorem

    # Pobranie rozmiarów ramki
    frame_height, frame_width, _ = frame.shape

    # Wyznaczenie współrzędnych obszaru tekstu
    text_area_top_left = (100, frame_height-200)
    text_area_bottom_right = (frame_width - 100, frame_height - 150)

    # Narysowanie obszaru tekstu
    cv.rectangle(frame, text_area_top_left, text_area_bottom_right, text_area_color, text_area_thickness)
    cv.putText(frame, typed_text, (text_area_top_left[0] + 10, text_area_top_left[1] + 40), FONTS, 1, (255, 255, 255),
               2)


while result is not None:
    ret, frameBeforeFlip = camera.read()  # getting frame from camera
    if not ret:
        break
    #  resizing frame
    frame = cv.flip(frameBeforeFlip, 1)
    frame = cv.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)

    #moving_point_frame = np.zeros((500, 500, 3), dtype=np.uint8)

    result = gaze_controller.calculate(frame=frame, do_calibration=do_calibration, reset_calibration=reset_calibration)
    #result = gaze_controller.calculate(frame=frame, do_calibration=do_calibration, reset_calibration=reset_calibration,
    #                                   moving_point_frame=moving_point_frame)

    do_calibration = False
    reset_calibration = False
    utils.colorBackgroundText(result["frame"], f'gaze_direction: {direction[result["gaze_direction"].value]}',
                              FONTS, 1.0, (250, 680), 2, color[0], color[1], 8, 8)

    height, width, _ = result["frame"].shape
    half_height = height // 2
    half_width = width // 2


    ################################ tutaj kropki przy kalibracji
    calibration_cnt = result["calibration_cnt"]
    if calibration_cnt == 0:
        cv.circle(result["frame"], (10, half_height), 5, (0, 255, 0), -1)  # Kropka na lewo
    if calibration_cnt == 1:
        cv.circle(result["frame"], (width - 10, half_height), 5, (0, 255, 0), -1)  # Kropka na prawo
    if calibration_cnt == 2:
        cv.circle(result["frame"], (half_width, 10), 5, (0, 255, 0), -1)  # Kropka na górze
    if calibration_cnt == 3:
        cv.circle(result["frame"], (half_width, height - 10), 5, (0, 255, 0), -1)  # Kropka na dole


   #########################a tutaj cala mechanika zapisywania kierunkow do tablicy, porownywanie z zmapowanymi literkami i wpisanie do pola
    ###kierunki 0 i 5 są pomijane (CENTER i NO)
    # Sprawdzenie czy kierunek patrzenia jest różny od "CENTER"
    if result["gaze_direction"] != Direction.CENTER and result["gaze_direction"] != Direction.NO_DIRECTION:
        # Zapisywanie kierunków patrzenia do tablicy (z uwzględnieniem warunku)
        if result["gaze_direction"] != previous_direction:
            # direction_history.pop(0)
            direction_history.append(result["gaze_direction"])
            previous_direction = result["gaze_direction"]
        elif previous_direction == Direction.CENTER:
            # direction_history.pop(0)
            direction_history.append(result["gaze_direction"])
            previous_direction = result["gaze_direction"]
    else:
        previous_direction = result["gaze_direction"]
        if previous_direction == Direction.NO_DIRECTION:
            previous_direction = Direction.CENTER

    # Sprawdzenie czy tablica została zapełniona
    if len(direction_history) == 3:
        # Wyświetlenie zawartości tablicy
        print("Kierunki patrzenia:", direction_history)

        # Przekształcenie tuple z kierunkami w literę na podstawie mapowania
        key = tuple(direction_history)
        if key in direction_to_letter:
            typed_text += direction_to_letter[key]
        else:
            typed_text += '?'  # Jeśli nie ma dopasowania, wpisz znak zapytania


        # Wyczyszczenie tablicy
        direction_history = []
        # previous_direction = None
    print(previous_direction)
    print(direction_history)
    draw_text_area(result["frame"])


    # Wyświetlenie ramki
    cv.imshow('frame', result["frame"])
    # cv.imshow('moving_point_frame', result["moving_point_frame"])
    # cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('c') or key == ord('C'):
        do_calibration = True
    if key == ord('r') or key == ord('R'):
        reset_calibration = True
    if key == ord('q') or key == ord('Q'):
        break
    if key == ord('8') and zoom < 6.0:
        zoom += 0.25
        zoom = gaze_controller.change_zoom(zoom)
    if key == ord('2') and zoom > 1.0:
        zoom -= 0.25
        zoom = gaze_controller.change_zoom(zoom)
    ############################ jak klikniesz z to czysci cale pole tekstu, rpzwoazanie tymczasowe, dla wygody testow
    if key == ord('z') or key == ord('Z'):
        typed_text = ''
    if key == ord('d') or key == ord('D'):
        debug_test = 1

cv.destroyAllWindows()
camera.release()