import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Update the path to the dataset file
file_path = r"C:\Users\pc\Downloads\smsspamcollection.tsv"

try:
    # Read the data into a DataFrame
    data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])

    # Print the first few rows of the dataset
    print("Sample data:")
    print(data.head())

    # Check if 'message' and 'label' columns are empty
    if data.empty or data['message'].isnull().all() or data['label'].isnull().all():
        raise ValueError("The dataset is empty or contains no valid data.")

    # Check the distribution of labels
    print("\nLabel distribution:")
    print(data['label'].value_counts())

    # Map labels to binary values
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Drop rows where 'message' or 'label' is NaN
    data.dropna(subset=['message', 'label'], inplace=True)

    # Check if there are enough samples to split
    if len(data) < 2:
        raise ValueError("Not enough samples to split into training and test sets.")

    # Split the data into features and labels
    X = data['message']
    y = data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that combines TF-IDF vectorization and Naive Bayes classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
        ('clf', MultinomialNB())       # Classify using Naive Bayes
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
