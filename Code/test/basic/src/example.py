# Step 1: Import Libraries ---------------------------------------------------
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Step 2: Create a Synthetic Dataset -----------------------------------------

# Set random seed for reproducibility
np.random.seed(0)
# Number of samples per class
n_samples_per_class = 50

# Generate synthetic data for three classes
class_1 = np.random.randn(n_samples_per_class, 2) + np.array([2, 2])
class_2 = np.random.randn(n_samples_per_class, 2) + np.array([-2, -2])
class_3 = np.random.randn(n_samples_per_class, 2) + np.array([-2, 2])

# Combine the data
X = np.vstack((class_1, class_2, class_3))
y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)

# Plot the data
plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 0')
plt.scatter(class_2[:, 0], class_2[:, 1], label='Class 1')
plt.scatter(class_3[:, 0], class_3[:, 1], label='Class 2')
plt.legend()
plt.title('Synthetic Dataset')
plt.show()


# Step 3: Prepare the Dictionary and Data ------------------------------------

# Split the data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# Normalize the data
X_train_full = normalize(X_train_full)
X_test = normalize(X_test)


# Step 4: Implement Sparse Representation ------------------------------------

# Create dictionaries for each class
D_classes = {}
classes = np.unique(y_train_full)
for cls in classes:
    D_classes[cls] = X_train_full[y_train_full == cls].T  # Transpose for computation

# Concatenate Dictionaries to Form the Overall Dictionary
# Overall dictionary
D = np.hstack([D_classes[cls] for cls in classes])


# Sparse Coding Using Orthogonal Matching Pursuit (OMP) - We'll use OMP to find the sparse coefficients.
# Initialize the OMP model
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10, normalize=False)

# List to store predicted classes
y_pred = []

# For each test sample
for i in range(X_test.shape[0]):
    y_i = X_test[i]

    # Fit the model to find sparse coefficients
    omp.fit(D, y_i)

    # Get the sparse coefficient vector
    x_sparse = omp.coef_

    # Reconstruct the test sample using class-specific dictionaries
    residuals = []
    start = 0
    for cls in classes:
        D_cls = D_classes[cls]
        n_atoms = D_cls.shape[1]
        x_cls = x_sparse[start:start + n_atoms]
        y_reconstructed = D_cls @ x_cls

        # Calculate the residual (reconstruction error)
        residual = np.linalg.norm(y_i - y_reconstructed)
        residuals.append(residual)

        start += n_atoms

    # Assign the class with the smallest residual
    y_pred.append(np.argmin(residuals))

# Step 6: Evaluate the Model -------------------------------------------------
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SRC Classification Accuracy: {accuracy * 100:.2f}%")


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.title('Confusion Matrix')
plt.show()


