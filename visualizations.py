import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# File Path and Data Preparation
file_path = "risk.txt"
columns = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", "agefirst", 
           "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt", "invasive", "cancer", 
           "training", "count"]

data = pd.read_csv(file_path, delim_whitespace=True, names=columns)
data = data.replace(9, pd.NA)
data = data.dropna(subset=["count"])  # Drop rows with missing count

weighted_data = data.loc[data.index.repeat(data["count"])]
categorical_cols = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", 
                    "agefirst", "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt", 
                    "invasive", "cancer", "training"]
weighted_data[categorical_cols] = weighted_data[categorical_cols].astype("category")

# Fill missing values in specific columns with "Unknown"
weighted_data["agegrp"] = weighted_data["agegrp"].cat.add_categories("Unknown").fillna("Unknown")
weighted_data["density"] = weighted_data["density"].cat.add_categories("Unknown").fillna("Unknown")
weighted_data["bmi"] = weighted_data["bmi"].cat.add_categories("Unknown").fillna("Unknown")

# Proportion of cancer cases by hormone therapy usage
hrt_cancer = pd.crosstab(weighted_data["hrt"], weighted_data["cancer"], normalize="index")

sns.heatmap(hrt_cancer, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Proportion of Cancer Cases by Hormone Therapy Usage")
plt.xlabel("Cancer (0 = No, 1 = Yes)")
plt.ylabel("Hormone Therapy (0 = No, 1 = Yes)")
plt.show()



# Visualization 1: Distribution of Age Groups
sns.countplot(x="agegrp", data=weighted_data)
plt.title("Distribution of Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Frequency")
plt.show()

# Visualization 2: Distribution of Breast Density
sns.countplot(x="density", data=weighted_data)
plt.title("Distribution of Breast Density")
plt.xlabel("Breast Density (BI-RADS codes)")
plt.ylabel("Frequency")
plt.show()

# Visualization 3: Distribution of BMI Categories
sns.countplot(x="bmi", data=weighted_data)
plt.title("Distribution of BMI Categories")
plt.xlabel("BMI Categories")
plt.ylabel("Frequency")
plt.show()

# Visualization 4: Heatmap of Race vs. Cancer Diagnosis
pivot_table = pd.crosstab(weighted_data["race"], weighted_data["cancer"], normalize="index")
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Proportion of Cancer Diagnosis by Race")
plt.xlabel("Cancer Diagnosis")
plt.ylabel("Race")
plt.tight_layout()
plt.show()

# Machine Learning Integration

#1 Encoding Categorical Data
categorical_cols.remove("cancer")  # Remove 'cancer' from the list of categorical columns
encoded_data = pd.get_dummies(weighted_data, columns=categorical_cols, drop_first=True)

#2 Define Features and Target
X = encoded_data.drop(columns=["cancer"])  # Ensure 'cancer' is present in the DataFrame
y = weighted_data["cancer"].astype(int)    # Use the original 'cancer' column as target


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#debug
print(weighted_data.columns)
print(encoded_data.columns)
print("X shape:", X.shape)
print("y shape:", y.shape)

# 5. Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 Feature Importances:\n", importances.head(10))

# Optional: Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()




