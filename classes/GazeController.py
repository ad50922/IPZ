import cv2 as cv
import mediapipe as mp
import time
import utils
import math
import numpy as np

# variables
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_CENTER_LINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
        39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LEFT_PUPIL_POINT = 468
LEFT_IRIS = [469, 470, 471, 472]
LEFT_KEY_POINTS = [362, 263, 9, 8]  # lewo, prawo, góra, dół

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_PUPIL_POINT = 473
RIGHT_IRIS = [474, 475, 476, 477]
RIGHT_KEY_POINTS = [33, 133, 9, 8]  # lewo, prawo, góra, dół


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y,z), (x,y,z)....]
    mesh_coord = [[int(point.x * img_width), int(point.y * img_height),
                  int(point.z * img_width)] for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p[:2], 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point[:2]
    x1, y1 = point1[:2]
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Distance ratio of point1->point2 euclidean distance
# to point3->point4 distance
def distanceRatio(point1, point2, point3, point4):
    temp1 = np.subtract(point1, point2)
    temp2 = np.subtract(point3, point4)
    d1 = np.sqrt(np.dot(np.transpose(temp1), temp1))
    d2 = np.sqrt(np.dot(np.transpose(temp2), temp2))
    if d2 == 0:
        d2 = 0.001
    if d2 == np.nan:
        print("what, why")
    return d1 / d2


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)

    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # getting the dimension of image
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes
    return cropped_right, cropped_left


# Eyes Postion Estimator
def positionEstimator(cropped_eye):
    # getting height and width of eye
    h, w = cropped_eye.shape

    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with
    piece = int(w / 3)

    # slicing the eyes into three parts
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # calling pixel counter function
    eye_position, color = pixelCounter(left_piece, center_piece, right_piece)   #jako ze zrobilem na kamerze lustrzane odbicie, w tym miejscu zamienilem kolejnoscia left z right i jest git

    return eye_position, color


# creating pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part] #strzelam ze tutaj bedzie trzeba dodac logike gora/dol dodatkowo

    # getting the index of max values in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:                               #i potem pewnie tutaj dodoac
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


class GazeController:
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                      refine_landmarks=True)

    def __init__(self):
        self._start_time = time.time()
        self._frame_counter = 0
        self._calibration_cnt = 0
        self._is_calibrated = False
        self._left_eye_calibration = []
        self._right_eye_calibration = []
        self._reference_face_3d_coords = []
        self._previous_center = None
        self._zoom = 1.0
        return

    def Reset(self):
        self._start_time = time.time()
        self._frame_counter = 0
        self._calibration_cnt = 0
        self._is_calibrated = False
        self._left_eye_calibration = []
        self._right_eye_calibration = []
        self._reference_face_3d_coords = []
        self._previous_center = None
        self._zoom = 1.0
        return

    def ChangeZoom(self, new_zoom):
        if new_zoom < 1.0:
            new_zoom = 1.0
        elif new_zoom > 6.0:
            new_zoom = 6.0
        self._zoom = new_zoom
        self._previous_center = None
        return new_zoom

    def Calculate(self, frame, do_calibration = False, reset_calibration = False):
        self._frame_counter += 1  # frame counter
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Długości wycinanej ramki
        y_offset = int(frame_height / (self._zoom * 2.0))  # tutaj wartości można zmieniać
        x_offset = int(frame_width / (self._zoom * 2.0))

        is_face = False
        gaze_direction = 0
        are_closed = False
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            is_face = True
            for detections in results.detections:
                # test = results.detections

                bounding_box = detections.location_data.relative_bounding_box
                y_center = int(((2 * bounding_box.ymin + bounding_box.height) / 2) * frame_height)
                x_center = int(((2 * bounding_box.xmin + bounding_box.width) / 2) * frame_width)
                # mp.solutions.drawing_utils.draw_detection(frame, results.detections[0])

                # Rozwiązanie problemu wychodzenia współrzędnych wycinanej ramki poza obraz
                diff = y_center - y_offset
                if diff < 0:
                    y_center -= diff
                diff = x_center - x_offset
                if diff < 0:
                    x_center -= diff
                sum = y_center + y_offset
                if sum > frame_height:
                    y_center -= sum - frame_height
                sum = x_center + x_offset
                if sum > frame_width:
                    x_center -= sum - frame_width

                current_center = (y_center, x_center)
                if self._previous_center is None:
                    break
                elif euclideanDistance(current_center, self._previous_center) < 80:
                    # Zmniejszenie trzęsienia obrazu
                    current_center = (int((current_center[0] + 6 * self._previous_center[0]) / 7),
                                      int((current_center[1] + 6 * self._previous_center[1]) / 7))
                    break

            self._previous_center = current_center
            # Wycinanie ramki
            frame = frame[current_center[0] - y_offset: current_center[0] + y_offset,
                          current_center[1] - x_offset: current_center[1] + x_offset]
            frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_CUBIC)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks and is_face:
            # test = results.multi_face_landmarks

            # Getting face mesh coordinates
            mesh_3d_coords = landmarksDetection(frame, results, False)
            mesh_2d_coords = [p[:2] for p in mesh_3d_coords]

            # Blink calculations
            ratio = blinkRatio(frame, mesh_2d_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                      utils.YELLOW)
            global CEF_COUNTER
            global TOTAL_BLINKS
            if ratio > 5.5:
                CEF_COUNTER += 1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                utils.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2,
                                          utils.YELLOW,
                                          pad_x=6, pad_y=6, )
                are_closed = True
            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
            # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
            utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

            # Drawing on frame
            right_eye_coords = [mesh_2d_coords[p] for p in RIGHT_EYE]
            left_eye_coords = [mesh_2d_coords[p] for p in LEFT_EYE]
            cv.polylines(frame, [np.array(left_eye_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array(right_eye_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)

            right_iris_coords = [mesh_2d_coords[p] for p in LEFT_IRIS]
            left_iris_coords = [mesh_2d_coords[p] for p in RIGHT_IRIS]
            cv.polylines(frame, [np.array(right_iris_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array(left_iris_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)

            right_pupil_coords = mesh_2d_coords[LEFT_PUPIL_POINT]
            left_pupil_coords = mesh_2d_coords[RIGHT_PUPIL_POINT]
            cv.circle(frame, right_pupil_coords, 1, utils.GREEN)
            cv.circle(frame, left_pupil_coords, 1, utils.GREEN)

            right_mean_coords = np.asarray([np.mean([mesh_2d_coords[p] for p in [2, 94]], axis=0),
                                            np.mean([mesh_2d_coords[p] for p in [168, 6]], axis=0)],
                                           dtype=np.int16)
            left_mean_coords = right_mean_coords
            for i in left_mean_coords:
                cv.circle(frame, i, 5, utils.RED)
            for i in right_mean_coords:
                cv.circle(frame, i, 5, utils.RED)

            right_key_points_coords = [mesh_2d_coords[p] for p in RIGHT_KEY_POINTS]
            left_key_points_coords = [mesh_2d_coords[p] for p in LEFT_KEY_POINTS]

            # Detecting gaze direction if calibrated
            if self._is_calibrated and not are_closed and not reset_calibration:
                gaze_direction = 5

                #     # Testy estymacji pozycji głowy
                #     # Get 2d Coord
                #     face_2d = np.array(mesh_2d_coords[:468], dtype=np.float64)
                #
                #     face_3d = np.array(reference_face_3d_coords, dtype=np.float64)
                #
                #     focal_length = 1 * frame_width
                #
                #     cam_matrix = np.array([[focal_length, 0, frame_height / 2],
                #                            [0, focal_length, frame_width / 2],
                #                            [0, 0, 1]])
                #     distortion_matrix = np.zeros((4, 1), dtype=np.float64)
                #
                #     success, rotation_vec, translation_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                #
                #     # getting rotational of face
                #     rmat, jac = cv.Rodrigues(rotation_vec)
                #
                #     angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)
                #
                #     x = angles[0]
                #     y = angles[1]
                #     z = angles[2]
                #     print((x, y, z))
                #
                #     R = np.asarray([[np.cos(x)*np.cos(y), np.cos(x)*np.sin(y)*np.sin(z)-np.sin(x)*np.cos(z), np.cos(x)*np.sin(y)*np.cos(z)+np.sin(x)*np.sin(z)],
                #                     [np.sin(x)*np.cos(y), np.sin(x)*np.sin(y)*np.sin(z)+np.cos(x)*np.cos(z), np.sin(x)*np.sin(y)*np.cos(z)-np.cos(x)*np.sin(z)],
                #                     [-np.sin(y), np.cos(y)*np.sin(z), np.cos(y)*np.cos(z)]])
                #     mesh_3d_coords = np.asarray([np.dot(R, p) for p in mesh_3d_coords],dtype=int)
                #     [cv.circle(frame, p[:2], 2, (0, 255, 0), -1) for p in mesh_3d_coords]

                right_distance = distanceRatio(right_mean_coords[0][1], right_pupil_coords[1],
                                               right_mean_coords[0][1],
                                               right_mean_coords[1][1])
                if right_distance >= self._right_eye_calibration[2]:
                    eye_position = 'UP'
                    color = [utils.GRAY, utils.YELLOW]
                elif right_distance <= self._right_eye_calibration[3]:
                    eye_position = "DOWN"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'R: {round(right_distance, 2)}, {eye_position}',
                                          FONTS, 1.0, (40, 380), 2, color[0], color[1], 8, 8)

                left_distance = distanceRatio(left_mean_coords[0][1], left_pupil_coords[1],
                                              left_mean_coords[0][1],
                                              left_mean_coords[1][1])
                if left_distance >= self._left_eye_calibration[2]:
                    if eye_position == 'UP':
                        gaze_direction = 3
                    else:
                        eye_position = 'UP'
                    color = [utils.GRAY, utils.YELLOW]
                elif left_distance <= self._left_eye_calibration[3]:
                    if eye_position == 'DOWN':
                        gaze_direction = 4
                    else:
                        eye_position = "DOWN"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'L: {round(left_distance, 2)}, {eye_position}',
                                          FONTS, 1.0, (40, 440), 2, color[0], color[1], 8, 8)

                right_distance = distanceRatio(right_key_points_coords[1], right_pupil_coords,
                                               right_key_points_coords[1],
                                               right_key_points_coords[0])
                if right_distance >= self._right_eye_calibration[0]:
                    eye_position = 'LEFT'
                    color = [utils.GRAY, utils.YELLOW]
                elif right_distance <= self._right_eye_calibration[1]:
                    eye_position = "RIGHT"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'R: {round(right_distance, 2)}, {eye_position}',
                                          FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)

                left_distance = distanceRatio(left_key_points_coords[0], left_pupil_coords,
                                              left_key_points_coords[0],
                                              left_key_points_coords[1])
                if left_distance <= self._left_eye_calibration[0]:
                    if eye_position == 'LEFT':
                        gaze_direction = 1
                    else:
                        eye_position = 'LEFT'
                    color = [utils.GRAY, utils.YELLOW]
                elif left_distance >= self._left_eye_calibration[1]:
                    if eye_position == 'RIGHT':
                        gaze_direction = 2
                    else:
                        eye_position = "RIGHT"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'L: {round(left_distance, 2)}, {eye_position}',
                                          FONTS, 1.0, (40, 280), 2, color[0], color[1], 8, 8)

        # Calculating frames per seconds FPS
        end_time = time.time() - self._start_time
        fps = self._frame_counter / end_time
        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                         textThickness=2)
        frame = utils.textWithBackground(frame, f'cnt: {self._calibration_cnt}', FONTS, 1.0, (200, 50), bgOpacity=0.9,
                                         textThickness=2)
        frame = utils.textWithBackground(frame, f'shape: {frame.shape}', FONTS, 0.75, (350, 50), bgOpacity=0.9,
                                         textThickness=2)
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)

        # Calibration procedures
        if not self._is_calibrated and do_calibration and results.multi_face_landmarks and is_face:
            # if self._calibration_cnt == 0:
            #     reference_face_3d_coords = mesh_3d_coords[:468]
            if self._calibration_cnt < 2:
                self._right_eye_calibration.append(distanceRatio(right_key_points_coords[1], right_pupil_coords,
                                                   right_key_points_coords[1],
                                                   right_key_points_coords[0]))
                self._left_eye_calibration.append(distanceRatio(left_key_points_coords[0], left_pupil_coords,
                                                                left_key_points_coords[0],
                                                                left_key_points_coords[1]))
            else:
                self._right_eye_calibration.append(distanceRatio(right_mean_coords[0][1], right_pupil_coords[1],
                                                                 right_mean_coords[0][1],
                                                                 right_mean_coords[1][1]))
                self._left_eye_calibration.append(distanceRatio(left_mean_coords[0][1], left_pupil_coords[1],
                                                                left_mean_coords[0][1],
                                                                left_mean_coords[1][1]))
            self._calibration_cnt += 1
            if self._calibration_cnt >= 4:
                self._is_calibrated = True
        if reset_calibration:
            self._is_calibrated = False
            self._calibration_cnt = 0
            self._right_eye_calibration.clear()
            self._left_eye_calibration.clear()
            self._reference_face_3d_coords.clear()

        # Return calculation data as a dictionary object
        return {
            "frame": frame,
            "is_face": is_face,
            "is_calibrated": self._is_calibrated,
            "calibration_cnt": self._calibration_cnt,
            "are_closed": are_closed,
            "gaze_direction": gaze_direction
        }


