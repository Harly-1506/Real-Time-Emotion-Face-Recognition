from typing_extensions import final
import tensorflow as tf
# from tensorflow import keras
import cv2
import numpy as np
import os

new_model = tf.keras.models.load_model('Camera.h5')

path = "D:\Python_Face_Emotion\haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set the rectangle background to white
rectangle_bgr = (255, 255, 255)
# make a black image
img = np.zeros((500, 500))
# set some text
text = "some text in a box!"
# get chiều rộng và chiều dài của text box
(text_width, text_height) = cv2.getTextSize(
    text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
# make the coord of the box with a small padding of two pixel
box_coords = ((text_offset_x, text_offset_y)), (text_offset_x +
                                                text_width + 2, text_offset_y - text_height-2)
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font,
            fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# check cam có mở hay không
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("không mở cam được")


while (True):
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    # face_roi = 0
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Dont see your face")

        else:
            for(ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]

                final_image = cv2.resize(face_roi, (224, 224))
                final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                final_image = np.expand_dims(final_image, axis=-1)
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image/255.0
                print("Predicting...")
                font = cv2.FONT_HERSHEY_SIMPLEX

                Predictions = new_model.predict(final_image)

                font_scale = 1.5
                font = cv2.FONT_HERSHEY_PLAIN

                if(np.argmax(Predictions) == 0):
                    status = "Neutral"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

                elif(np.argmax(Predictions) == 1):
                    status = "Happiness"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

                elif (np.argmax(Predictions) == 2):
                    status = "Surprise"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

                elif (np.argmax(Predictions) == 3):
                    status = "Sadness"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

                elif (np.argmax(Predictions) == 4):
                    status = "Anger"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

                elif (np.argmax(Predictions) == 5):
                    status = "Disgust"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

                elif (np.argmax(Predictions) == 6):
                    status = "Fear"
                    accuracy = np.max(Predictions) * 100
                    accuracy = int(accuracy)
                    accuracy = str(accuracy) + "% "
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # vẽ khung màu đen
                    cv2.rectangle(frame, (x1, x1),
                                  (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # add tẽtx
                    cv2.putText(frame, accuracy + status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, status, (100, 150), font,
                                3, (0, 0, 255), 2, cv2.LINE_4)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    cv2.imshow("Python_Face_Emotion", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# predict face recognition with cv2
# import cv2
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing import image
