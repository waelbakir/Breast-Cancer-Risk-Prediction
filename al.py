import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Define column names
columns = ['menopaus', 'agegrp', 'density', 'race', 'Hispanic', 'bmi', 'agefirst',
           'nrelbc', 'brstproc', 'lastmamm', 'surgmeno', 'hrt', 'invasive',
           'cancer', 'training', 'count']

# Load data
file_path = "risk.txt"
data = pd.read_csv(file_path, delim_whitespace=True, names=columns)

# Exclude 'cancer' column from encoding
target = data["cancer"]
features = data.drop(columns=["cancer"])

# Encode categorical variables
encoded_features = pd.get_dummies(features, drop_first=True)

# Add the target column back
encoded_data = pd.concat([encoded_features, target], axis=1)

# Separate features and target
X = encoded_data.drop(columns=["cancer"])
y = encoded_data["cancer"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest model with balanced class weights
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, "cancer_risk_model.pkl")
print("Model saved as cancer_risk_model.pkl")

# Save the column names for preprocessing
joblib.dump(list(X.columns), "model_columns.pkl")
print("Feature columns saved as 'model_columns.pkl'")

# Now for user input and prediction
import joblib

# Define columns used in the dataset
columns = ['menopaus', 'agegrp', 'density', 'race', 'Hispanic', 'bmi', 'agefirst',
           'nrelbc', 'brstproc', 'lastmamm', 'surgmeno', 'hrt', 'invasive',
           'training', 'count']

# Load the trained model
model = joblib.load("cancer_risk_model.pkl")

# Collect user input
def get_user_input():
    user_data = {
        "menopaus": int(input("Enter menopausal status (0 = pre, 1 = post): ").strip()),
        "agegrp": int(input("Enter age group (1-10): ").strip()),
        "density": int(input("Enter breast density (1-4): ").strip()),
        "race": int(input("Enter race (1 = white, 2 = Asian, 3 = black, etc.): ").strip()),
        "Hispanic": int(input("Are you Hispanic? (0 = No, 1 = Yes): ").strip()),
        "bmi": float(input("Enter BMI (e.g., 24.5): ").strip()),
        "agefirst": int(input("Enter age at first birth (0 = <30, 1 = >=30, 2 = Nulliparous): ").strip()),
        "nrelbc": int(input("Number of relatives with breast cancer (0-2): ").strip()),
        "brstproc": int(input("Breast procedure? (0 = No, 1 = Yes): ").strip()),
        "lastmamm": int(input("Last mammogram (0 = Negative, 1 = False Positive): ").strip()),
        "surgmeno": int(input("Surgical menopause? (0 = No, 1 = Yes): ").strip()),
        "hrt": int(input("Hormone replacement therapy? (0 = No, 1 = Yes): ").strip()),
        "invasive": int(input("History of invasive breast cancer? (0 = No, 1 = Yes): ").strip()),
        "training": int(input("Training set (0 = No, 1 = Yes): ").strip()),
        "count": int(input("Count (numeric value): ").strip()),
    }
    return pd.DataFrame([user_data])

# Preprocess user data
def preprocess_input(data):
    # Convert categorical values to match the trained model
    data = pd.get_dummies(data, drop_first=True)

    # Ensure all columns from training are present
    model_columns = joblib.load("model_columns.pkl")
    for col in model_columns:
        if col not in data.columns:
            data[col] = 0

    # Ensure same column order
    data = data[model_columns]
    return data

# Run the prediction
user_input = get_user_input()
processed_input = preprocess_input(user_input)

# Get raw probability
prediction_proba = model.predict_proba(processed_input)[0][1]

# Adjust decision threshold
threshold = 0.3  # Example: set threshold to 30% for predicting high-risk cancer
prediction = (prediction_proba > threshold).astype(int)

# Display the result
print(f"Raw probability of cancer: {prediction_proba * 100:.2f}%")
print(f"Adjusted prediction (threshold = {threshold}): {'HIGH' if prediction == 1 else 'LOW'}")
