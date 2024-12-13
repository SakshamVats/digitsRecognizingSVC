import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split      

def display_digits(digits, number_shown = 5):
    """Displays digit data"""
    _, axs = plt.subplots(nrows=1, ncols=number_shown, figsize=(10,3))
    for ax, image, label in zip(axs, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Training: {label}")
    plt.show()
        
def svc_model(digits):
    """Defines, trains and evaluates a Support Vector Classifier"""
    n_samples = len(digits.images)
    data = digits.images.reshape(n_samples, -1)
    clf = svm.SVC(gamma=0.001)
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    #print(predicted)
    print(
            f"Classification report for classifier {clf}: \n"
            f"{metrics.classification_report(y_test, predicted)}\n"
          )
    return  clf

class DigitApp:
    def __init__(self, root, clf):
        self.root = root
        self.root.title("Draw a digit")
        self.clf = clf

        self.canvas = tk.Canvas(root, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        mouse_left_drag = "<B1-Motion>"
        self.canvas.bind(mouse_left_drag, self.paint)

        self.image = np.ones((200, 200), dtype=np.uint8) * 255
        tk.Button(root, text="Predict",  command=self.predict).grid(row=1, column=0, columnspan=2)
        tk.Button(root, text="Clear", command=self.clear).grid(row=1, column=2, columnspan=2)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8 #brush radius

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        cv2.circle(self.image, (x,y),  r, (0,), thickness=-1)

    def clear(self):
        self.canvas.delete("all")
        self.image = np.ones((200,200), dtype=np.uint8) * 255

    def predict(self):
        resized_image = cv2.resize(self.image, (8,8), interpolation=cv2.INTER_AREA)
        processed_image = 16 - (resized_image // 16)
        processed_image = processed_image.astype(np.int64)
        flattened_image = processed_image.reshape(1, -1)
        prediction = self.clf.predict(flattened_image)[0]

        messagebox.showinfo("Prediction", f"The predicted digit is: {prediction}")

def main():
    """Implements model flow"""
    digits = datasets.load_digits()
    display_digits(digits, 6)
    
    clf = svc_model(digits)
    
    root = tk.Tk()
    app = DigitApp(root, clf)
    root.mainloop()

main()
    
