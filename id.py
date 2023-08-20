import cv2

# Load the Haar cascades for face and ID card detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id_cascade = cv2.CascadeClassifier('haarcascade_id.xml')

# Load the image and convert it to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Loop through each face and check if an ID card is present
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    ids = id_cascade.detectMultiScale(roi_gray)
    for (ix,iy,iw,ih) in ids:
        cv2.rectangle(roi_color,(ix,iy),(ix+iw,iy+ih),(0,255,0),2)

# Display the image with the detected faces and ID cards
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()