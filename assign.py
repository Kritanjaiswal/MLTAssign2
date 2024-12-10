import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score

# Replace the file paths with the correct paths to your datasets
demographics = pd.read_csv('NHANES_Demographics.csv')
medical = pd.read_csv('NHANES_Medical_Conditions.csv')


# Select relevant columns
demographics = demographics[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
medical = medical[['SEQN', 'MCQ160E']]  # Assuming 'MCQ160E' indicates heart issues

# Merge datasets on SEQN
data = pd.merge(demographics, medical, on='SEQN')

# Rename columns for clarity
data.columns = ['SEQN', 'Age', 'Gender', 'Heart_Issues']

# Filter data for age ranges and encode Gender
data['Gender'] = data['Gender'].map({1: 'Male', 2: 'Female'})
data = data.dropna()  # Drop rows with missing values

X = data[['Age', 'Gender']]
y = data['Heart_Issues']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'K-NN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'ANN': MLPClassifier(max_iter=1000)
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

# Evaluate all models
results = {}
for name, model in models.items():
    accuracy, precision = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': accuracy, 'Precision': precision}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)


# Group by Age and Gender
age_gender_distribution = data.groupby(['Age', 'Gender']).size().unstack().fillna(0)
print(age_gender_distribution)

# Group by Age and Heart Issues
age_heart_issues_distribution = data.groupby(['Age', 'Heart_Issues']).size().unstack().fillna(0)
print(age_heart_issues_distribution)
