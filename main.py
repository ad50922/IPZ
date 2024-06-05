import cv2 as cv
import utils
import classes.GazeController as GazeController

##################### tak zaznacze zmiany :)
direction_to_letter = {
    (1, 1, 1): 'a',
    (1, 1, 3): 'b',
    (1, 1, 2): 'c',
    # (1, 4) back
    (1, 3, 1): 'd',
    (1, 3, 3): 'e',
    (1, 3, 2): 'f',
    # (1, 4) back
    (1, 2, 1): 'g',
    (1, 2, 3): 'h',
    (1, 2, 2): 'i',
    # (1, 4) back

    (3, 1, 1): 'j',
    (3, 1, 3): 'k',
    (3, 1, 2): 'l',
    # (1, 4) back
    (3, 3, 1): 'm',
    (3, 3, 3): 'n',
    (3, 3, 2): 'o',
    # (1, 4) backq
    (3, 2, 1): 'p',
    (3, 2, 3): 'q',
    (3, 2, 2): 'r',
    # (1, 4) back


    (2, 1, 1): 's',
    (2, 1, 3): 't',
    (2, 1, 2): 'u',
    # (1, 2, 4) back
    (2, 3, 1): 'v',
    (2, 3, 3): 'w',
    (2, 3, 2): 'x',
    # (1, 2, 4) back
    (2, 2, 1): 'y',
    (2, 2, 3): 'z',
    (2, 2, 2): ',',
    # (1, 2, 4) back

    # (4) spacja // obslugiwana w głownej funkcji
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
previous_direction = 5
typed_text = ''


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

def draw_text_on_frame(frame):
    # Sprawdzenie zawartości tablicy direction_history i rysowanie odpowiedniej litery
    if len(direction_history) == 0:
        # Litery po lewo
        cv.putText(frame, '    e', (400, frame.shape[0] // 2 - 75), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, '   d f', (400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, ' b     h ', (400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, 'a c   g i', (400, frame.shape[0] // 2 + 75), FONTS, 1, (255, 255, 255), 2)
        # Litery u gory
        cv.putText(frame, '    n ', (frame.shape[1] // 2 , 225), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, '   m o', (frame.shape[1] // 2 , 275), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, ' k     q ', (frame.shape[1] // 2 , 325), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, 'j l   p r', (frame.shape[1] // 2 , 375), FONTS, 1, (255, 255, 255), 2)
        # litery po prawo
        cv.putText(frame, '    w', (frame.shape[1] - 400, frame.shape[0] // 2 -75), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, '   v x', (frame.shape[1] - 400, frame.shape[0] // 2 -25), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, ' t     z', (frame.shape[1] - 400, frame.shape[0] // 2 +25), FONTS, 1, (255, 255, 255), 2)
        cv.putText(frame, 's u   y ,', (frame.shape[1] - 400, frame.shape[0] // 2 +75), FONTS, 1, (255, 255, 255), 2)
        # Litery na dole
        cv.putText(frame, 'spacja', (frame.shape[1] // 2, frame.shape[0] - 325), FONTS, 1, (255, 255, 255), 2)

    elif len(direction_history) == 1:
        # Sprawdzenie wartości w direction_history
        value = direction_history[0]
        if value == 1:   # lewo
            cv.putText(frame, ' b', (400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'a c', (400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, ' e ', (frame.shape[1] // 2 , 300 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'd f', (frame.shape[1] // 2 , 300 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, ' h ', (frame.shape[1] - 400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'g i', (frame.shape[1] - 400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - 300), FONTS, 1, (255, 255, 255), 2)
        elif value == 2: # prawo
            cv.putText(frame, ' t ', (400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 's u', (400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, ' w ', (frame.shape[1] // 2, frame.shape[0] // 4 ), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'v x', (frame.shape[1] // 2, frame.shape[0] // 4 + 50), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, ' z ', (frame.shape[1] - 400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'y ,', (frame.shape[1] - 400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                       (255, 255, 255), 2)
        elif value == 3: # gora
            cv.putText(frame, ' k ', (400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'j l', (400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, ' n ', (frame.shape[1] // 2, frame.shape[0] // 4 ), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'm o', (frame.shape[1] // 2, frame.shape[0] // 4 + 50), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, ' q ', (frame.shape[1] - 400, frame.shape[0] // 2 - 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'p r', (frame.shape[1] - 400, frame.shape[0] // 2 + 25), FONTS, 1, (255, 255, 255), 2)
            cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                       (255, 255, 255), 2)
        elif value == 4: # dół
            pass
    elif len(direction_history) == 2:
        value0 = direction_history[0]
        value1 = direction_history[1]
        if value0 == 1:   # lewo
            if value1 == 1:   #lewo
                cv.putText(frame, 'a', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'b', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'c', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 2: # prawo
                cv.putText(frame, 'g', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'h', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'i', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 3: # gora
                cv.putText(frame, 'd', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'e', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'f', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 4: # dol
                pass # back
        elif value0 == 2:   # prawo
            if value1 == 1:   #lewo
                cv.putText(frame, 's', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 't', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'u', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 2: # prawo
                cv.putText(frame, 'y', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'z', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, ',', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 3: # gora
                cv.putText(frame, 'v', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'w', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'x', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 4: # dol
                pass # back
        elif value0 == 3:   # gora
            if value1 == 1:   #lewo
                cv.putText(frame, 'j', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'k', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'l', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 2: # prawo
                cv.putText(frame, 'p', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'q', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'r', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 3: # gora
                cv.putText(frame, 'm', (400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'n', (frame.shape[1] // 2, frame.shape[0] // 4 + 25), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'o', (frame.shape[1] - 400, frame.shape[0] // 2), FONTS, 1, (255, 255, 255), 2)
                cv.putText(frame, 'back', (frame.shape[1] // 2, frame.shape[0] - frame.shape[0] // 4 - 25), FONTS, 1,
                           (255, 255, 255), 2)
            elif value1 == 4: # dol
                pass #back

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
    utils.colorBackgroundText(result["frame"], f'gaze_direction: {direction[result["gaze_direction"].value]}',
                              FONTS, 1.0, (200, 100), 2, color[0], color[1], 8, 8)


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
    if result["gaze_direction"].value != 5 and result["gaze_direction"].value != 0:
        # Zapisywanie kierunków patrzenia do tablicy (z uwzględnieniem warunku)
        if result["gaze_direction"].value != previous_direction:
            # direction_history.pop(0)
            direction_history.append(result["gaze_direction"].value)
            previous_direction = result["gaze_direction"].value
        elif previous_direction == 5:
            # direction_history.pop(0)
            direction_history.append(result["gaze_direction"].value)
            previous_direction = result["gaze_direction"].value
    else:
        previous_direction = result["gaze_direction"].value
        if previous_direction == 0:
            previous_direction = 5

    # spacja. jezeli kierunek to doł wyzeruj tablice
    if direction_history and direction_history[0] == 4:
        typed_text += ' '
        direction_history = []

    if len(direction_history) >= 2 and direction_history[1] == 4:
        direction_history.pop()
        direction_history.pop()

    if len(direction_history) >= 3 and direction_history[2] == 4:
        direction_history.pop()
        direction_history.pop()

    # dodanie kursora

    # Sprawdzenie czy tablica została zapełniona
    if len(direction_history) == 3:
        # Wyświetlenie zawartości tablicy
        print("Kierunki patrzenia:", direction_history)

        # Przekształcenie tuple z kierunkami w literę na podstawie mapowania
        key = tuple(direction_history)
        print(key)
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
    draw_text_on_frame(result["frame"])


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