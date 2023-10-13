import io
import os
import PySimpleGUI as sg
from PIL import Image
import pydicom
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
file_types = [("All files (*.*)", "*.*"),("JPEG (*.jpg)", "*.jpg")]               
def main():
    layout = [
        [
            sg.Text("Parkinson's Disease Prediction", size=(30, 1), justification='center', font='Helvetica 20')
        ],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
        [
            sg.Button("Predict"),
        ],
        [
            sg.Button("Exit", size=(5, 1)),
        ],
        [sg.Image(key="-IMAGE-")],
    ]
    window = sg.Window("Parkinson's Disease Early Prediction", layout,margins=(150, 100))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            print(filename)
            if os.path.exists(filename):
                img = pydicom.dcmread(filename)
                img_array = img.pixel_array
                img_array = cv.resize(img_array,(512,512))
                new_image = img.pixel_array.astype(float)
                scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
                scaled_image = np.uint8(scaled_image)
                final_image = Image.fromarray(scaled_image)
                image = final_image
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
        if event == "Predict":
            filename = values["-FILE-"]
            #print(filename)
            print("Prediction")
            img = pydicom.dcmread(filename)
            img_array = img.pixel_array
            img_array = cv.resize(img_array,(512,512))
            saved_model  = "./savedModel/parkinsons_detection_cnn.hdf5"
            model = keras.models.load_model(saved_model)
            model_out = model.predict(img_array.reshape(1,512,512))[0]
            if np.argmax(model_out) == 1:
                str_label ='Parkinsons'
                sg.Popup('Parkinsons Detected') 
            else:
                str_label ='No Parkinsons'
                sg.Popup('No Parkinsons Detected') 
            print("The predicted result is ", str_label.decode())
    window.close()
if __name__ == "__main__":
    main()