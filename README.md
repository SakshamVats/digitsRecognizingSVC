# 🧠 Neural Network Implementation using NumPy  

A simple **feedforward neural network** built from scratch using **NumPy**. The model is trained to classify images of **cats, dogs, and mice** using randomly generated feature data.  

---

## 📌 Features  
- **Customizable Parameters**: Epochs, learning rate, and architecture settings are defined as global variables.  
- **One-Hot Encoding**: Converts categorical labels into one-hot vectors for multi-class classification.  
- **Hidden Layer**: 4 neurons in a single hidden layer.  
- **Output Layer**: 3 neurons for classification (Cat, Dog, Mouse).  
- **Optimized Training**: Uses **Softmax activation** for output and **Sigmoid activation** for hidden layers.  
- **Randomized Initialization**: Weights and biases are initialized randomly.  

---

## 📂 Project Structure  
- **`neural_net.py`** → Main script implementing the neural network.  
- **Generated Data** → Simulated dataset representing cat, dog, and mouse images.  
- **Training Process** → Implements **feedforward propagation, backpropagation, and gradient descent**.  
- **Plotting Option** → Scatter plot visualization of generated data.  

---

## 🛠️ Installation & Usage  

### 🔧 Prerequisites  
Ensure you have **Python 3.x** and the required libraries installed:  
```sh
pip install numpy matplotlib
'''

📊 Visualizing Data

Uncomment the following lines in neural_net.py to plot the dataset before training:

plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)
plt.show()

🌟 If you find this project useful, consider giving it a ⭐ Star on GitHub!
