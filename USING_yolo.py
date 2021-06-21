########### Social Distancing Detector #########################

#########################################################################
#   Author : Sachin Lodhi                                               #
#   Track  : Computer Vision and Internet of Things                     #
#   Batch  : June 2021                                                  #
#   Assignment 4 : Social Distance                                      #
#   Graduate Rotational Internship Program,                             #
#   The Spark Foundation, June 2021                                     #
#                                                                       #
#########################################################################


import cv2
import imutils
import time
import yolo_module as ym



if __name__ == '__main__':
    sdObj = ym.Social()
    create = None
    frameno = 0
    filename = "vids/people3_2.mp4" # Supply the input video
    yolo = "yolo-coco/"
    opname = "output_videos/output_of_" + filename.split('/')[1][:-4] + '.mp4'  #saving video with custom filename
    cap = cv2.VideoCapture(filename)

    time1 = time.time()
    while (True):

        ret, frame = cap.read()
        if not ret:
            break
        current_img = frame.copy()
        current_img = imutils.resize(current_img, width=480)
        video = current_img.shape
        frameno += 1

        if (frameno % 2 == 0 or frameno == 1):
            sdObj.Setup()
            Frame = sdObj.ImageProcess(current_img)
            if create is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                create = cv2.VideoWriter(opname, fourcc, 30, (Frame.shape[1], Frame.shape[0]), True)
        create.write(Frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    time2 = time.time()
    print("Completed. Total Time Taken: {} minutes".format((time2 - time1) / 60))

    cap.release()
    cv2.destroyAllWindows()


# LEts Run it

# I am using CPU not GPU to process the frames so it would take a long time to process even 4 seconds video.

# That is very slow process... You can employ GPU for same task for faster results
# That was the input video

# Okay letscheck what did it produce

# So we can see that it detects the violation