#Week 6 Activity 1: SVM Classification - IRIS Dataset

##DATASET
- File: Iris.csv
- Rows: 150 samples
- Classes: Iris-setosa, Iris-versicolor, Iris-virginica
- Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

##STEPS
- Load: reads Iris.csv and previews the data
- Clean: drops Id column, removes duplicates, checks missing values
- Visualise: two scatter plots side by side
    - Left: Petal Length vs Petal Width (Iris Petals)
    - Right: Sepal Length vs Sepal Width (Iris Sepals)
- Prepare: 80/20 train-test split, features scaled with StandardScaler
- Train: SVC with linear kernel fitted on training data
- Evaluate: accuracy, precision, recall, and confusion matrix on test data

##TRAIN TEST SPLIT
- test_size = 0.2
- 80% training = 117 samples
- 20% testing = 30 samples

##RESULTS

1) Console Output

<img width="943" height="582" alt="Screenshot 2026-05-17 122147" src="https://github.com/user-attachments/assets/553e4d20-c9c2-48f1-aded-2709dd9bd962" />

2) Scatter Chart

<img width="1200" height="500" alt="scatter_plot" src="https://github.com/user-attachments/assets/53e06c3c-4806-4921-b486-b6f656d4c33a" />

##HOW TO RUN
- Install: pip install pandas matplotlib scikit-learn
- Run: python W6A1.py
- Note: Iris.csv must be in the same folder as W6A1.py
