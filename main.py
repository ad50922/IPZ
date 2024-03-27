import cv2 as cv
import utils
import GazeController as GazeController

##################### tak zaznacze zmiany :)
direction_to_letter = {
    (1, 1, 1): 'a',
    (1, 1, 2): 'b',
    (1, 2, 1): 'c',
    (1, 2, 2): 'd',
    (1, 3, 1): 'e',
    (1, 3, 2): 'f',
    (1, 4, 1): 'g',
    (1, 4, 2): 'h',
    (2, 1, 1): 'i',
    (2, 1, 2): 'j',
    (2, 2, 1): 'k',
    (2, 2, 2): 'l',
    (2, 3, 1): 'm',
    (2, 3, 2): 'n',
    (2, 4, 1): 'o',
    (2, 4, 2): 'p',
    (3, 1, 1): 'q',
    (3, 1, 2): 'r',
    (3, 2, 1): 's',
    (3, 2, 2): 't',
    (3, 3, 1): 'u',
    (3, 3, 2): 'v',
    (3, 4, 1): 'w',
    (3, 4, 2): 'x',
    (4, 1, 1): 'y',
    (4, 1, 2): 'z',
    (4, 2, 1): '!',
    (4, 2, 2): '@',
    (4, 3, 1): '#',
    (4, 3, 2): '#',
    (4, 4, 1): '$',
    (4, 4, 2): ' ',
    # Dodaj więcej mapowań, jeśli potrzebujesz
}


FONTS = cv.FONT_HERSHEY_COMPLEX
color = [utils.YELLOW, utils.PINK]
direction = ["NO", "LEFT", "RIGHT", "UP", "DOWN", "CENTER"]

camera = cv.VideoCapture(0)
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
previous_direction = 5
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
    result = gaze_controller.calculate(frame=frame, do_calibration=do_calibration, reset_calibration=reset_calibration)

    do_calibration = False
    reset_calibration = False
    utils.colorBackgroundText(result["frame"], f'gaze_direction: {direction[result["gaze_direction"]]}',
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
    if result["gaze_direction"] != 5 and result["gaze_direction"] != 0:
        # Zapisywanie kierunków patrzenia do tablicy (z uwzględnieniem warunku)
        if result["gaze_direction"] != previous_direction:
            # direction_history.pop(0)
            direction_history.append(result["gaze_direction"])
            previous_direction = result["gaze_direction"]
        elif previous_direction == 5:
            # direction_history.pop(0)
            direction_history.append(result["gaze_direction"])
            previous_direction = result["gaze_direction"]
    else:
        previous_direction = result["gaze_direction"]
        if previous_direction == 0:
            previous_direction = 5

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
