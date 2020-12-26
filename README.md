# Face Expression Prediction

This is a jupyter notebook for recognizing live facial expressions using *Tensorflow* and *opencv*.

Firstly Tensorflow is used to train the model. For training and testing purpose the dataset which was used is available here :- https://www.kaggle.com/aadityasinghal/facial-expression-dataset

After that we created a **CNN** Model with firstl three convolutional layers after that a flatten layerand after that two fully connected layer. Finally the model was compiled.
The visual model architecture looks like as given below :-

![Model architecture](https://github.com/Aaditya1978/Face_Expression_Prediction/blob/main/Images/model.png?raw=true)

The model training was started. It was ran for 15 epochs and got an accuracy of nearly 68%.

After that the model was saved in json format.

The code below is used to load the model by using json file and also to load the weights by using .h5 file.
After that the prediction is made. and based on the prediction the particular emotion from the list is returned.
```python
class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

```

The code below takes the video frames and does the preprocessing and finally calls the code above to do the prediction and then returns the frame with the boxes bounded and also the expression.

## An Important note to change the input of the video replace 0 with the path of the video file you want to predict. Here 0 is written to access the webcam of your laptop. 
```python
import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0) # An important note -> to hange the input replace 0 with the video file path you want to add

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        return fr
```
Finally we call the last function which calls the above videocamera class and shows the video with prediction till the key **q** is pressed.

The output when a video is given to it is given below :- 

![Model Output Prediction](https://github.com/Aaditya1978/Face_Expression_Prediction/blob/main/Images/Facial-Expression-Recognization.gif)


# If you like my work and gained knowledge. Please do follow me and Please star ⭐ my Repo 😉😉.
