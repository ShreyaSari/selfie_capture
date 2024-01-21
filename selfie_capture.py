import cv2

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("D:\\New folder\\haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("D:\\New folder\\haarcascade_smile.xml")

if faceCascade.empty():
    raise Exception("Error loading face cascade classifier XML file")

# Rest of your code, including calling the detectMultiScale function
while True:
    success,img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg,1.1,4)
    cnt=1
    keyPressed = cv2.waitKey(1)

    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),3)
        smiles = smileCascade.detectMultiScale(grayImg,1.8,15)
        for x,y,w,h in smiles:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5)
            print("Image "+str(cnt)+"Saved")
            path=r"C:\Users\SARI SHREYA\OneDrive\Desktop\known_faces"+str(cnt)+'.jpg'
            cv2.imwrite(path,img)
            cnt +=1
            if(cnt>=15):    
                break
                
    cv2.imshow('live video',img)
    if(keyPressed & 0xFF==ord('q')):
        break

video.release()                                  
cv2.destroyAllWindows()