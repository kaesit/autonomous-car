import şerit_takip
import levha_tespit
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import utlis

cap = cv2.VideoCapture(0)

img = cap.read()

curveList = []
avgVal = 10


curve = 0
def getLaneCurve(img, display=2):
    imgCopy = img.copy()
    imgResult = img.copy()
    #### STEP 1
    imgThres = utlis.thresholding(img)

    #### STEP 2
    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    #### STEP 3
    middlePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    #### SETP 4
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    #### STEP 5
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt', imgResult)

    #### NORMALIZATION
    curve = curve / 100
    if curve > 1: curve == 1
    if curve < -1: curve == -1

    return curve


def main():
    #levha_tespi#şerit_takip.main_code()
    lower_blue = np.array([85, 100, 70])
    upper_blue = np.array([115, 255, 255])

    intialTrackBarVals = [102, 80, 20, 214]
    utlis.initializeTrackbars(intialTrackBarVals)

    while True:
        # grab the current frame
        (grabbed, img) = cap.read()

        if not grabbed:
            print("No input image")
            break

        img = imutils.resize(img, width=500)
        frameArea = img.shape[0] * img.shape[1]

        # convert color image to HSV color scheme
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define kernel for smoothing
        kernel = np.ones((3, 3), np.uint8)
        # extract binary image with active blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # defite string variable to hold detected sign description
        detectedTrafficSign = None

        # define variables to hold values during loop
        largestArea = 0
        largestRect = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            for cnt in cnts:
                # Rotated Rectangle. Here, bounding rectangle is drawn with minimum area,
                # so it considers the rotation also. The function used is cv2.minAreaRect().
                # It returns a Box2D structure which contains following detals -
                # ( center (x,y), (width, height), angle of rotation ).
                # But to draw this rectangle, we need 4 corners of the rectangle.
                # It is obtained by the function cv2.boxPoints()
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # count euclidian distance for each side of the rectangle
                sideOne = np.linalg.norm(box[0] - box[1])
                sideTwo = np.linalg.norm(box[0] - box[3])
                # count area of the rectangle
                area = sideOne * sideTwo
                # find the largest rectangle within all contours
                if area > largestArea:
                    largestArea = area
                    largestRect = box

        # draw contour of the found rectangle on  the original image
        if largestArea > frameArea * 0.02:
            cv2.drawContours(img, [largestRect], 0, (0, 0, 255), 2)

            # if largestRect is not None:
            # cut and warp interesting area
            warped = four_point_transform(mask, [largestRect][0])

            # show an image if rectangle was found
            # cv2.imshow("Warped", cv2.bitwise_not(warped))

            # use function to detect the sign on the found rectangle

            detectedTrafficSign = identifyTrafficSign(warped)
            print(detectedTrafficSign)

            # write the description of the sign on the original image
            cv2.putText(img, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0),
                        2)

        # show original image
        frameCounter = 0
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        curve = getLaneCurve(img, display=2)
        print(curve)
        cv2.imshow("Original", img)

        # if the q key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF is ord('q'):
            cv2.destroyAllWindows()
            print("Stop programm and close all windows")
            break

gecilenYol = 0


def identifyTrafficSign(image):
     ""
     SIGNS_LOOKUP = {
         (1, 0, 0, 1): 'Saga don',  # turnRight
         (0, 0, 1, 1): 'Sola don',  # turnLeft
         (0, 1, 0, 1): 'Duz ilerle',  # moveStraight
         (1, 0, 1, 1): 'Geri dön',  # turnBack
     }

     THRESHOLD = 150

     image = cv2.bitwise_not (image)
     # (roiH, roiW) = roi.shape
     # subHeight = thresh.shape[0]/10
     # subWidth = thresh.shape[1]/10
     (subHeight, subWidth) = np.divide (image.shape, 10)
     subHeight = int (subHeight)
     subWidth = int (subWidth)

     # mark the ROIs borders on the image
     # cv2.rectangle(image, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
     # cv2.rectangle(image, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
     # cv2.rectangle(image, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
     # cv2.rectangle(image, (3*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

     # substract 4 ROI of the sign thresh image
     leftBlock = image[4 * subHeight:9 * subHeight, subWidth:3 * subWidth]
     centerBlock = image[4 * subHeight:9 * subHeight, 4 * subWidth:6 * subWidth]
     rightBlock = image[4 * subHeight:9 * subHeight, 7 * subWidth:9 * subWidth]
     topBlock = image[2 * subHeight:4 * subHeight, 3 * subWidth:7 * subWidth]

     # we now track the fraction of each ROI
     leftFraction = np.sum (leftBlock) / (leftBlock.shape[0] * leftBlock.shape[1])
     centerFraction = np.sum (centerBlock) / (centerBlock.shape[0] * centerBlock.shape[1])
     rightFraction = np.sum (rightBlock) / (rightBlock.shape[0] * rightBlock.shape[1])
     topFraction = np.sum (topBlock) / (topBlock.shape[0] * topBlock.shape[1])

     segments = (leftFraction, centerFraction, rightFraction, topFraction)
     segments = tuple (1 if segment > THRESHOLD else 0 for segment in segments)

     cv2.imshow ("Warped", image)

     if segments in SIGNS_LOOKUP:
         return SIGNS_LOOKUP[segments]

     # SAĞA DÖNMEK RASPBERRY Pİ İÇİN ŞART KOŞULUMUZ

     # RASPBERRY PI KODLARI BURAYA

     elif SIGNS_LOOKUP[(1, 0, 0, 1)]:
         if curve >= 10:
             if gecilenYol >= 1:
                 GPIO.output(in1,GPIO.HIGH)
                 GPIO.output(in2,GPIO.LOW)
                 print("forward")
             else:
                 gecilenYol += 1

     if SIGNS_LOOKUP[(0, 0, 1, 1)]:
         if curve >= 10:
             if gecilenYol >= 1:
                 return ("Sola dönülüyor")
             else:
                 gecilenYol += 1
     elif SIGNS_LOOKUP[(0, 1, 0, 1)]:
        return ("Ileri")
     else:
         return None


if __name__ == '__main__':
    main()
