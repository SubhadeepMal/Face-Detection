#Face & Eye detection using openCV

import cv2 as cv

cascade_classifier= cv.CascadeClassifier(r"C:\Users\ADMIN\haarcascade_frontalface_alt.xml")
cascade_classifier2= cv.CascadeClassifier(r"C:\Users\ADMIN\haarcascade_eye_tree_eyeglasses.xml")

# For image:
img=cv.imread(r"C:\Users\ADMIN\Pictures\Saved Pictures\picc.jpg")


gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces= cascade_classifier.detectMultiScale(gray, 1.1, 4)
c=0
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    roi_gray= gray[y:y+h, x:x+w]
    roi_color= img[y:y+h, x:x+w]

    c=c+1

    eyes= cascade_classifier2.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

m=str(c)+"Faces Detected"
cv.putText(img, m ,(50,50), cv.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 2)
cv.imshow('itachi',img)

cv.waitKey()

#print(c," faces detected")

# For video:
capture= cv.VideoCapture(0)

while True:
    isTrue, frame=capture.read()
    
    gray_video= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_video= cascade_classifier.detectMultiScale(gray_video, 1.1, 4)
    k=0
    for (x, y, w, h) in faces_video:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        roi_gray= gray[y:y+h, x:x+w]
        roi_color= frame[y:y+h, x:x+w]
        k=k+1
        eyes= cascade_classifier2.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
    
    l=str(k)+"Faces Detected"   
    cv.putText(frame, l ,(50,50,), cv.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 2)
     
    cv.imshow('Video',frame)

    if cv.waitKey(1)& 0xFF==ord('d'):
        break
#print(k," faces detected")
capture.release()        
