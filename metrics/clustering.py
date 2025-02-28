import numpy as np
from sklearn.metrics import silhouette_score

# Example sequence and labels
sequence = ['a5', 'x5', 'a1', 'x2', 'a3', 'x1', 'x3', 'a2', 'x4', 'x6']
labels = np.array([1, 1, 2, 1, 1, 2, 2, 2, 1, 2])

# Function to calculate the distance as specified
def calculate_custom_distance(sequence, labels):
    n = len(sequence)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Examine path from i to j
            path_labels = labels[i:j+1]
            start_label = labels[i]

            if np.all(path_labels == start_label):
                # All labels in path are the same as the start
                distance = 0
            else:
                # Count label changes excluding changes back to start_label
                unique_labels, counts = np.unique(path_labels, return_counts=True)
                changes = 0
                last_label = path_labels[0]
                for k in range(1, len(path_labels)):
                    if path_labels[k] != path_labels[k-1] and path_labels[k] != start_label:
                        changes += 1
                distance = changes

            # Set distance in both directions
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

# Calculate the custom distance matrix
distance_matrix = calculate_custom_distance(sequence, labels)
print("Custom Distance Matrix:")
print(distance_matrix)

# Calculate the silhouette score using the custom distance matrix
score = silhouette_score(distance_matrix, labels, metric='precomputed')
print(f"Silhouette Score: {score}")
