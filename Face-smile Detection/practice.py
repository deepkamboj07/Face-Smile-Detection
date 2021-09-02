import cv2
face_cas=cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')
smile_cas=cv2.CascadeClassifier('haarcascade\haarcascade_smile.xml')

def detectFace(grey, frame):
    flag=2
    face=face_cas.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        flag=0

        roi_grey=grey[y:y+h, x:x+w]
        smile=smile_cas.detectMultiScale(roi_grey,1.3,22)
        for(sx,sy,sw,sh) in smile:
            flag=1


    if flag==0:
        cv2.putText(frame,
                        'You are not Smiling',
                        (50, 50),
                        font, 1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_4)

    elif flag==1:
        cv2.putText(frame,
                'You are Smiling',
                (50, 50),
                font, 1,
                (0, 255, 0),
                2,
                cv2.LINE_4)

    return frame

video_capture=cv2.VideoCapture(0)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))


font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    _,frame=video_capture.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detectFace(grey,frame)



    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
video_capture.release()


cv2.destroyAllWindows()
