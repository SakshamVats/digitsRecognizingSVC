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

def main():
    """Implements model flow"""
    digits = datasets.load_digits()
    display_digits(digits, 6)
    svc_model(digits)

main()
    
