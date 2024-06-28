# GRAM SCHMIDT ORTHOGONALIZATION, SVD, LEAST SQUARE APPROXIMATION

## Gram-Schmidt Orthogonalization Based Feature Selection

### Introduction Specificities:
Gram-Schmidt Orthogonalization (GSO) is a mathematical technique used in feature selection to create a new set of features that are uncorrelated (orthogonal) to each other while capturing most of the information from the original features. This helps improve machine learning model performance and reduces computational cost by dealing with a smaller, more informative feature set.

### The Basic Concept Overview:
Imagine you have a bunch of arrows (features) pointing in different directions. Feature selection with GSO aims to find a new set of arrows that represent the same space but are perpendicular (orthogonal) to each other. These new, uncorrelated features will capture the essential information from the originals without redundancy.

### The Process Flow:
Here's the process:
1. Start with the first original feature vector.
2. Project all subsequent features onto the space spanned by the already chosen features and remove that projection. This ensures the new feature is independent of the previous ones.
3. Normalize the remaining vector to get a unit-length new feature.
4. Repeat steps 2 and 3 for all remaining features.

By the end, you have a set of orthogonal features that effectively capture the important information from the originals.

### Mathematical Overview 

![image](https://github.com/MirshaMorningstar/SVD_Gram-Schmidt-Orthogonalization_Least-Square-Approximation/assets/84216040/dacdcfad-cac2-41cb-9333-374beeafd33c)

Let's represent your original features as a set of vectors x_1, x_2, ..., x_n in an n-dimensional space. Here are the relevant formulas involved in GSO for feature selection:

**1. Projection:** The projection of vector x_i onto the space spanned by

vectors x_1, ..., x_(j-1) is:
proj_(x_1, ..., x_(j-1)) (x_i) = ((x_i . x_1) / ||x_1||^2) * x_1 + ... + ((x_i . x_(j-1)) / ||x_(j-1)||^2) * x_(j-1)

where . denotes the dot product and ||x|| represents the vector's magnitude.

**2. Removing the projection:** To get the part of x_i orthogonal to the existing features:

x_i' = x_i - proj_(x_1, ..., x_(j-1)) (x_i)

**3. Normalization (optional):** Normalize x_i' to get a unit-length vector (useful for some algorithms):

x_j = x_i' / ||x_i' ||

Here, x_j represents the new, j-th orthogonal feature vector.

By iteratively applying these equations, you obtain a set of orthogonal feature
vectors that can be used in your machine learning model.

### Breast Cancer Wisconsin (Diagnostic) Data Set

#### About Dataset:
The UCI Breast Cancer Wisconsin (Diagnostic) Dataset is a well-known dataset used for machine learning and statistical modeling, particularly for binary classification tasks.

#### Overview:
- Dataset Name: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Number of Instances: 569
- Number of Attributes: 30 numerical features (plus 1 target variable)
- Task: Classification (Benign vs. Malignant)

#### Attributes Information:
The dataset consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The features are divided into three main categories: mean, standard error, and worst (largest) value. For each cell nucleus, the following features are provided:
- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

### Pythonic Code:

#### Step 1: Load and preprocess the dataset
First, we'll load the Breast Cancer dataset and select 15 features.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Select 15 features
selected_features = data.feature_names[:15]
X_selected = X[:, :15]

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_selected)

# Convert to DataFrame for better visualization
df = pd.DataFrame(X_normalized, columns=selected_features)
df['target'] = y
print(df.head())
```

#### Step 2: Perform Gram-Schmidt Orthogonalization
Next, we implement the Gram-Schmidt process to orthogonalize the feature vectors.

```python
def gram_schmidt(X):
    Q = np.zeros_like(X)
    for i in range(X.shape[1]):
        qi = X[:, i]
        for j in range(i):
            qj = Q[:, j]
            qi -= np.dot(qi, qj) * qj
        qi /= np.linalg.norm(qi)
        Q[:, i] = qi
    return Q

# Apply Gram-Schmidt process
Q = gram_schmidt(X_normalized)

# Convert to DataFrame for visualization
df_orthogonal = pd.DataFrame(Q, columns=selected_features)
df_orthogonal['target'] = y
print(df_orthogonal.head())
```

#### Step 3: Measure Orthogonality
We measure the orthogonality of the transformed features.

```python
def measure_orthogonality(Q):
    orthogonality_matrix = np.dot(Q.T, Q)
    off_diagonal_elements = orthogonality_matrix - np.diag(np.diagonal(orthogonality_matrix))
    orthogonality_score = np.sum(np.abs(off_diagonal_elements))
    return orthogonality_score

# Measure orthogonality
orthogonality_score = measure_orthogonality(Q)
print(f'Orthogonality Score: {orthogonality_score}')
```

#### Step 4: Evaluate the Selected Features
We use a classifier to evaluate the performance of the selected features.

```python
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(Q, y, test_size=0.1, random_state=42)

# Train a Logistic Regression classifier
clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with Orthogonalized Features: {accuracy}')
```

#### Step 5: Generate Relevant Plots
We will generate three plots:
- Correlation Matrix of Original Features
- Correlation Matrix of Orthogonalized Features
- Comparison of Model Performance with Original vs Orthogonalized Features.

```python
# Plot 1: Correlation Matrix of Original Features
plt.figure(figsize=(10, 8))
sns.heatmap(df[selected_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Original Features')
plt.show()

# Plot 2: Correlation Matrix of Orthogonalized Features
plt.figure(figsize=(10, 8))
sns.heatmap(df_orthogonal[selected_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Orthogonalized Features')
plt.show()

# Plot 3: Model Performance Comparison
# Evaluate the model on original features
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
clf_orig = LogisticRegression(max_iter=10000)
clf_orig.fit(X_train_orig, y_train_orig)
y_pred_orig = clf_orig.predict(X_test_orig)
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)

# Bar plot comparison
accuracy_data = pd.DataFrame({
    'Features': ['Original', 'Orthogonalized'],
    'Accuracy': [accuracy_orig, accuracy]
})
plt.figure(figsize=(8, 6))
sns.barplot(x='Features', y='Accuracy', data=accuracy_data)
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.show()
```
### Outputs Generated 

![image](https://github.com/MirshaMorningstar/SVD_Gram-Schmidt-Orthogonalization_Least-Square-Approximation/assets/84216040/91f01233-1c61-4f95-ab13-4b53fbfb439f)

![image](https://github.com/MirshaMorningstar/SVD_Gram-Schmidt-Orthogonalization_Least-Square-Approximation/assets/84216040/2c4ca692-eabf-49ea-b40b-5e76178ade62)

#### Inference:
Therefore, we can notice that after the orthogonalization of the considered dataset and performing Classification on the selected "Top K" Attributes of the resultant dataset, there has been a drastic improvement in the Accuracy Performance of the Classification Model from a mere 0.68 to a much appreciated 0.93 %. This strengthens our knowledge on the effects of orthogonal transformation on the data.

