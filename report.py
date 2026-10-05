import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. LOAD MNIST DATASET
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training data:", x_train.shape)
print("Testing data :", x_test.shape)


# --------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten 28x28 image into 784 features
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("After flattening:")
print("Training:", x_train.shape)
print("Testing :", x_test.shape)


# --------------------------------------------------
# 3. CREATE LOGISTIC REGRESSION MODEL
# --------------------------------------------------

model = LogisticRegression(
    max_iter=200,
    solver='lbfgs'
)

print("\nTraining model...")

model.fit(x_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 4. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(x_test)


# --------------------------------------------------
# 5. ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy : {accuracy * 100:.2f}%")


# --------------------------------------------------
# 6. PRECISION
# --------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

print(f"Precision: {precision * 100:.2f}%")


# --------------------------------------------------
# 7. RECALL
# --------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

print(f"Recall   : {recall * 100:.2f}%")


# --------------------------------------------------
# 8. F1 SCORE
# --------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print(f"F1 Score : {f1 * 100:.2f}%")


# --------------------------------------------------
# 9. CLASSIFICATION REPORT
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MNIST Confusion Matrix")

plt.show()


# --------------------------------------------------
# 11. SAMPLE PREDICTIONS
# --------------------------------------------------

print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(10):

    probability = model.predict_proba(
        x_test[i].reshape(1, -1)
    )

    confidence = np.max(probability)

    print(
        f"Image {i+1}: "
        f"Actual = {y_test[i]}, "
        f"Predicted = {y_pred[i]}, "
        f"Probability = {confidence * 100:.2f}%"
    )


# --------------------------------------------------
# 12. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "mnist_logreg_model.pkl"
)

print("\nModel saved successfully!")