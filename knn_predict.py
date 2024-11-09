from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

class PredictBestLabel:
    def __init__(self):
        # Load data and initialize necessary components
        self.df = pd.read_csv('data.csv')
        self.score = []
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.vectorizer = TfidfVectorizer()
        
        # Convert scores and process columns immediately
        self.convert_to_score()
        self.process_columns()
    
    def convert_to_score(self):
        # Calculate scores based on physical symptoms
        scoring = {'Never': 0, 'Sometimes': 1, 'Often': 2, 'Very often': 3}
        scores = []

        for i in range(self.df.shape[0]):
            cnt = sum(scoring.get(self.df.get(col)[i], 0) for col in self.df.columns)
            
            # Assign labels based on the count
            if 0 <= cnt < 6:
                scores.append('Low')
            elif 6 <= cnt < 12:
                scores.append('Moderate')
            elif 12 <= cnt < 18:
                scores.append('High')
            else:
                scores.append('Very high')

        self.score = scores
    
    def process_columns(self):
        # Drop unnecessary columns
        self.df = self.df.drop(['Timestamp', ' [Aches and pains]', ' [Chest pain or heart pounding]', 
                                ' [Exhaustion or trouble sleeping]', ' [Headaches, dizziness or shaking]', 
                                ' [High blood pressure]', ' [Muscle tension or jaw clenching]', 
                                ' [Stomach or digestive problems]', ' [Weakened immune system]'], axis=1)

    def flatten_to_prepare(self):
        # Flatten strings and vectorize
        all_strings = self.df.values.flatten()
        all_vectors = self.vectorizer.fit_transform(all_strings).toarray()
        
        # Combine vectors for each row
        vector_coordinates = np.array([
            np.hstack(all_vectors[i*5:(i+1)*5]) for i in range(len(self.df))
        ])
        
        return vector_coordinates

    def train(self):
        # Prepare vector coordinates and train the model
        vector_coordinates = self.flatten_to_prepare()
        self.knn.fit(vector_coordinates, self.score)

    def predict(self, sample_data):
        # Ensure the model is trained
        self.train()
        
        # Process sample data for prediction
        sample_vector = self.vectorizer.transform(sample_data).toarray()
        combined_sample_vector = np.hstack(sample_vector)
        
        # Make a prediction
        prediction = self.knn.predict([combined_sample_vector])
        return prediction[0]

# Example usage:
res = PredictBestLabel()

sample_data = ['Bug', 'Satan', 'God', 'Lungs', 'Colorful bug']
prediction = res.predict(sample_data)
print(f"Predicted label for the sample response: {prediction}")
