import cv2 as cv
import utils
import classes.GazeController as GazeController
import numpy as np
from enum import Enum

FONTS = cv.FONT_HERSHEY_COMPLEX
color = [utils.YELLOW, utils.PINK]

camera = cv.VideoCapture(0)
gaze_controller = GazeController.GazeController()
result = 1
do_calibration = False
reset_calibration = False
zoom = 1.0
moving_point_frame = np.zeros((500, 500, 3), dtype=np.uint8)
while result is not None:
    ret, frameBeforeFlip = camera.read()  # getting frame from camera
    if not ret:
        break
    #  resizing frame
    frame = cv.flip(frameBeforeFlip, 1)
    #frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    moving_point_frame *= 0
    result = gaze_controller.calculate(frame=frame, do_calibration=do_calibration, reset_calibration=reset_calibration,
                                       moving_point_frame=moving_point_frame)

    do_calibration = False
    reset_calibration = False
    utils.colorBackgroundText(result["frame"], f'gaze_direction: {result["gaze_direction"].name}',
                              FONTS, 0.6, (160, 450), 2, color[0], color[1], 8, 8)
    cv.imshow('frame', result["frame"])
    cv.imshow('moving_point_frame', moving_point_frame)
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
    if key == ord('d') or key == ord('D'):
        debug_test = 1

cv.destroyAllWindows()
camera.release()
