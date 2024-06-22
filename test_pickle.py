import pickle

# Try to load the classifier.pkl file
try:
    with open('classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

