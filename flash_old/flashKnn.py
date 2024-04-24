from sklearn.neighbors import NearestNeighbors

def knn_clustering(embeddings, k):
    # Create a NearestNeighbors object
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embeddings)

    # Get the cluster assignments for each embedding
    distances, indices = nn.kneighbors(embeddings)

    # Create a dictionary to store the clusters
    clusters = {}
    for i, neighbors in enumerate(indices):
        cluster_id = i
        clusters[cluster_id] = neighbors.tolist()

    return clusters

def query_neighbors(embeddings, query_embedding, n_neighbors):
    # Create a NearestNeighbors object
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embeddings)

    # Find the nearest neighbors for the query embedding
    distances, indices = nn.kneighbors([query_embedding])

    # Get the distances and indices of the neighbors
    neighbor_distances = distances[0]
    neighbor_indices = indices[0]

    return neighbor_indices, neighbor_distances

# Example usage
word_embeddings = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6],
    [0.5, 0.6, 0.7]
]

num_clusters = 2
clusters = knn_clustering(word_embeddings, num_clusters)

print("Clusters:")
for cluster_id, members in clusters.items():
    print(f"Cluster {cluster_id}: {members}")

query_embedding = [0.25, 0.35, 0.45]
num_neighbors = 3
neighbor_indices, neighbor_distances = query_neighbors(word_embeddings, query_embedding, num_neighbors)

print("\nQuery Results:")
for i, (index, distance) in enumerate(zip(neighbor_indices, neighbor_distances)):
    print(f"Neighbor {i+1}: Index = {index}, Distance = {distance:.4f}")

    