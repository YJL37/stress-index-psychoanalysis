from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Step 1: Take 30 responses, each with 5 string inputs
responses = []
print("Enter 30 responses, each comprised of 5 string inputs:")

for i in range(30):
    response = []
    print(f"\nResponse {i+1}:")
    for j in range(5):
        response.append(input(f"  String {j+1}: "))
    responses.append(response)

# Step 2: Flatten all string inputs to convert them into vector values
all_strings = [string for response in responses for string in response]

# Use TF-IDF to convert strings into vectors
vectorizer = TfidfVectorizer()
all_vectors = vectorizer.fit_transform(all_strings).toarray()

# Step 3: Reshape the vectors into 30 rows with each row being a combined vector coordinate
# Each response (5 strings) now corresponds to a single vector
vector_coordinates = np.array([np.hstack(all_vectors[i*5:(i+1)*5]) for i in range(30)])

# Dummy labels for KNN (for grouping); modify as needed based on the task
y = np.random.randint(0, 3, 30)  # Example: 3 possible groups

# Step 4: Use KNN to group the vector coordinates
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(vector_coordinates, y)

# Step 5: Test the KNN classifier on one of the responses
sample_index = 0
prediction = knn.predict([vector_coordinates[sample_index]])

print(f"\nPredicted class for Response {sample_index + 1}: {prediction[0]}")
