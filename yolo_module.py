# This is the custom module with pre-trained YOLO object detector
import numpy as np
import cv2
import time

class Social():
    # all initilaizations with loading of the coco model and weights
    def __init__(self):
        self.weights = "./yolo-coco/yolov3.weights"
        self.config = "./yolo-coco/yolo.cfg"
        self.labelsPath = "./yolo-coco/coco.names"

    # checking the distance between pairs of centroids of rectangles
    def Check_distance(self, a, b):
        dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
        calibration = (a[1] + b[1]) / 2
        if 0 < dist < 0.25 * calibration:
            return True
        else:
            return False
    # setting up the lables and processing the file to get classes from coco-names
    def Setup(self):
        self.LABELS = open(self.labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Processing the exact frame for the calculation of the centroid after detecting the persons and locating them in the same frame
    def ImageProcess(self, image):

        (H, W) = (None, None)
        frame = image.copy()
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        starttime = time.time()

        # This is detection using SSD-Net
        layerOutputs = self.net.forward(self.ln)
        stoptime = time.time()
        print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime - starttime)))
        confidences = []
        outline = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                maxi_class = np.argmax(scores)
                confidence = scores[maxi_class]
                if self.LABELS[maxi_class] == "person":
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        outline.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))

        box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)

        if len(box_line) > 0:
            flat_box = box_line.flatten()
            pairs = []
            center = []
            status = []
            for i in flat_box:
                (x, y) = (outline[i][0], outline[i][1])
                (w, h) = (outline[i][2], outline[i][3])
                center.append([int(x + w / 2), int(y + h / 2)])
                status.append(False)

            for i in range(len(center)):
                for j in range(len(center)):
                    close = self.Check_distance(center[i], center[j])

                    if close:
                        pairs.append([center[i], center[j]])
                        status[i] = True
                        status[j] = True
            index = 0

            for i in flat_box:
                (x, y) = (outline[i][0], outline[i][1])
                (w, h) = (outline[i][2], outline[i][3])
                if status[index] == True:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                elif status[index] == False:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                index += 1
            # Drawing line between centroids if the boxes are closed to each other
            for h in pairs:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        processedImg = frame.copy()
        return processedImg     # Returning the processed frame after drawing the boxes around the persons and indicating if the rules violated
