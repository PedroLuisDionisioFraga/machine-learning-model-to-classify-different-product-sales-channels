import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 2 - Reading the CSV
"""
Here, the code reads data from a file named wholesale.csv, which contains information about customer transactions. It then displays the first few rows to give an idea of what the data looks like.
"""
df = pd.read_csv("wholesale.csv", encoding="utf-8")
print(df.head())

# 3 - Conversion
"""
The data contains words like "HoReCa" and "Retail" to describe different sales channels. These words are converted into numbers (0 for HoReCa, 1 for Retail) because the machine learning model works with numbers rather than text.
"""
df["Channel"] = df["Channel"].replace({"HoReCa": 0, "Retail": 1})
df["Region"] = df["Region"].replace({"Lisbon": 0, "Oporto": 1, "Other": 2})

# 4 - Reorder
df = df[['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen', 'Channel']]
print(df.head())

# 5 - Splitting into 80% and 20%
train, test = train_test_split(df, test_size=0.2, random_state=42)
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

# 6 - Choosing a classifier
"""
The code separates the data into features (like how much was spent on milk or groceries) and labels (whether the transaction was HoReCa or Retail)
"""

# 7 - Train and classify the model
# Separate features and labels
"""
The model is tested using the test data, and the predictions it makes are compared to the actual outcomes. The performance of the model is then evaluated and printed.
"""
X_train = train.drop("Channel", axis=1)
y_train = train["Channel"]
X_test = test.drop("Channel", axis=1)
y_test = test["Channel"]
# Instantiate the classifier
classifier = DecisionTreeClassifier()
# Train the model
classifier.fit(X_train, y_train)
# Classify the test set samples
predictions = classifier.predict(X_test)

# 8 - Display evaluation metrics
print("Evaluation Metrics:\n")
print(classification_report(y_test, predictions))

# 9 - Create user input function
def classify_transaction():
  print("Enter the details of the transaction:")
  region = int(input("Which region is it from? (0 for Lisbon, 1 for Oporto, 2 for Other): "))
  fresh = float(input("What is the annual spending on fresh products (in u.m.)? "))
  milk = float(input("What is the annual spending on milk products (in u.m.)? "))
  grocery = float(input("What is the annual spending on grocery products (in u.m.)? "))
  frozen = float(input("What is the annual spending on frozen products (in u.m.)? "))
  detergents_paper = float(input("What is the annual spending on detergents and paper products (in u.m.)? "))
  delicatessen = float(input("What is the annual spending on delicatessen products (in u.m.)? "))

  input_data = pd.DataFrame([[region, fresh, milk, grocery, frozen, detergents_paper, delicatessen]],
                            columns=['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen'])

  prediction = classifier.predict(input_data)

  channel_mapping = {0: "HoReCa", 1: "Retail"}
  predicted_channel = channel_mapping[prediction[0]]

  print(f"The channel of the transaction is: {predicted_channel}")

#classify_transaction()

def test_model_with_new_data():
    # Define new transaction data
    new_transaction = pd.DataFrame([[1, 3000, 5000, 8000, 1200, 1500, 1000]],
                                   columns=['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen'])

    # Make a prediction using the trained model
    prediction = classifier.predict(new_transaction)

    # Map the prediction back to the channel
    channel_mapping = {0: "HoReCa", 1: "Retail"}
    predicted_channel = channel_mapping[prediction[0]]

    print(f"The predicted channel for the transaction is: {predicted_channel}")

# Call the test function
test_model_with_new_data()

