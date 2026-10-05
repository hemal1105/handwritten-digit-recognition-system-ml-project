import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from tensorflow import keras            # used only to download/load MNIST
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the MNIST dataset (60,000 train + 10,000 test images of 28x28)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)

# 2. Visualise a few samples
plt.figure(figsize=(10, 4))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. Pre-processing: scale to [0,1] and flatten 28x28 -> 784 features
#    (Logistic Regression needs a 2-D table: samples x features)
x_train = (x_train / 255.0).reshape(-1, 784)
x_test = (x_test / 255.0).reshape(-1, 784)
print("Flattened train shape:", x_train.shape)   # (60000, 784)

# 4. Build and train the Logistic Regression model
#    multi-class (softmax / multinomial) with the lbfgs solver
model = LogisticRegression(max_iter=200, solver='lbfgs')
print("\nTraining Logistic Regression... (may take 1-2 minutes)")
start = time.time()
model.fit(x_train, y_train)
print(f"Training finished in {time.time() - start:.1f} seconds")

# 5. Evaluate on the unseen test set
y_pred = model.predict(x_test)
print(f"\nTrain Accuracy: {model.score(x_train, y_train) * 100:.2f}%")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# 7. Visualise the learned weights of each digit (784 weights -> 28x28 image)
plt.figure(figsize=(12, 5))
for d in range(10):
    plt.subplot(2, 5, d + 1)
    plt.imshow(model.coef_[d].reshape(28, 28), cmap='coolwarm')
    plt.title(f"Digit {d}")
    plt.axis('off')
plt.suptitle("Learned weights per class (red = supports the digit, blue = against)")
plt.tight_layout()
plt.show()

# 8. Sample predictions with probabilities
probs = model.predict_proba(x_test[:10])
for i in range(10):
    print(f"Image {i+1}: True = {y_test[i]}, Predicted = {y_pred[i]}, Confidence = {probs[i].max()*100:.1f}%")

# 9. Save the trained model
joblib.dump(model, 'mnist_logreg_model.pkl')
print("\nModel saved as mnist_logreg_model.pkl")
