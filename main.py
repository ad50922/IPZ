import cv2 as cv
import classes.GazeController as GazeController


camera = cv.VideoCapture(1)
gaze_controller = GazeController.GazeController()
result = 1
do_calibration = False
reset_calibration = False
zoom = 1.0
while result is not None:
    ret, frameBeforeFlip = camera.read()  # getting frame from camera
    if not ret:
        break
    #  resizing frame
    frame = cv.flip(frameBeforeFlip, 1)
    frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    result = gaze_controller.Calculate(frame=frame, do_calibration=do_calibration, reset_calibration=reset_calibration)

    do_calibration = False
    reset_calibration = False
    cv.imshow('frame', result["frame"])
    # cv.imshow('frame', frame)
    key = cv.waitKey(2)
    if key == ord('c') or key == ord('C'):
        do_calibration = True
    if key == ord('r') or key == ord('R'):
        reset_calibration = True
    if key == ord('q') or key == ord('Q'):
        break
    if key == ord('8') and zoom < 6.0:
        zoom += 0.25
        zoom = gaze_controller.ChangeZoom(zoom)
    if key == ord('2') and zoom > 1.0:
        zoom -= 0.25
        zoom = gaze_controller.ChangeZoom(zoom)
    if key == ord('d') or key == ord('D'):
        debug_test = 1

cv.destroyAllWindows()
camera.release()