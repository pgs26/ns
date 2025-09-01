#PROGRAMMING ASSIGNMENT 3
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Generate power-law degree sequence
def generate_power_law_sequence(N, gamma):
    degrees = np.random.pareto(gamma - 1, N).astype(int) + 1  # Minimum degree 1
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    return degrees

# Simulate attack and track giant component size
def simulate_attack(G, criterion, fractions):
    G_copy = G.copy()
    N = G_copy.number_of_nodes()
    sizes = []

    if criterion == 'degree':
        nodes_sorted = sorted(G_copy.nodes(), key=lambda x: G_copy.degree(x), reverse=True)
    else:  # clustering
        clustering = nx.clustering(G_copy)
        nodes_sorted = sorted(G_copy.nodes(), key=lambda x: clustering[x], reverse=True)

    for f in fractions:
        num_remove = int(f * N)
        G_temp = G_copy.copy()
        G_temp.remove_nodes_from(nodes_sorted[:num_remove])
        if G_temp.number_of_nodes() == 0:
            sizes.append(0)
        else:
            largest_cc = max(nx.connected_components(G_temp), key=len, default=set())
            sizes.append(len(largest_cc) / N)

    return sizes

# Plot giant component sizes
def plot_giant_component(fractions, sizes_degree, sizes_clustering, title):
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, sizes_degree, label='Degree Attack', marker='o')
    plt.plot(fractions, sizes_clustering, label='Clustering Attack', marker='s')
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Giant Component Size (Normalized)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    N = 10000
    fractions = np.linspace(0, 0.5, 11)  # 0 to 50% in 5% steps

    # Task 1: Configuration Model (Power-law, γ = 2.5)
    degrees = generate_power_law_sequence(N, gamma=2.5)
    G_config = nx.configuration_model(degrees)
    G_config = nx.Graph(G_config)  # Convert to simple graph
    sizes_degree_config = simulate_attack(G_config, 'degree', fractions)
    sizes_clustering_config = simulate_attack(G_config, 'clustering', fractions)
    print("Configuration Model (γ = 2.5):")
    print(f"Degree Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_degree_config]}")
    print(f"Clustering Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_clustering_config]}")
    plot_giant_component(fractions, sizes_degree_config, sizes_clustering_config,
                        'Giant Component Size vs Fraction Removed (Configuration Model, γ=2.5)')

    # Task 2: Hierarchical Model
    G_hierarchical = nx.powerlaw_cluster_graph(N, m=4, p=0.1)
    sizes_degree_hier = simulate_attack(G_hierarchical, 'degree', fractions)
    sizes_clustering_hier = simulate_attack(G_hierarchical, 'clustering', fractions)
    print("\nHierarchical Model (m=4, p=0.1):")
    print(f"Degree Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_degree_hier]}")
    print(f"Clustering Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_clustering_hier]}")
    plot_giant_component(fractions, sizes_degree_hier, sizes_clustering_hier,
                        'Giant Component Size vs Fraction Removed (Hierarchical Model)')

#Programming Assignment 4
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Generate power-law degree sequence for scale-free network
def generate_scale_free_degree_sequence(N, gamma, avg_degree):
    degrees = np.random.pareto(gamma - 1, N) * N ** (1 / (gamma - 1))
    degrees = degrees.astype(int) + 1  # Ensure positive integers
    current_avg = np.mean(degrees)
    degrees = (degrees * (avg_degree / current_avg)).astype(int)
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    return degrees

# Simulate sandpile model
def simulate_sandpile(G, steps=10):
    buckets = {node: G.degree(node) for node in G.nodes()}
    grains = {node: 0 for node in G.nodes()}
    avalanche_sizes = []

    for _ in range(steps):
        node = np.random.choice(list(G.nodes()))
        grains[node] += 1

        unstable = deque([node]) if grains[node] >= buckets[node] else deque()
        current_avalanche_size = 0

        while unstable:
            current_node = unstable.popleft()
            if grains[current_node] >= buckets[current_node]:
                current_avalanche_size += 1
                num_grains = grains[current_node]
                grains[current_node] = 0
                neighbors = list(G.neighbors(current_node))
                if neighbors:
                    grains_per_neighbor = num_grains // len(neighbors)
                    for neighbor in neighbors:
                        grains[neighbor] += grains_per_neighbor
                        if grains[neighbor] >= buckets[neighbor]:
                            unstable.append(neighbor)

        if current_avalanche_size > 0:
            avalanche_sizes.append(current_avalanche_size)

    return avalanche_sizes

# Plot avalanche size distribution
def plot_avalanche_distribution(avalanche_sizes, title):
    plt.figure(figsize=(8, 6))
    plt.hist(avalanche_sizes, bins=30, log=True, density=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Avalanche Size')
    plt.ylabel('Frequency (Log Scale)')
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.show()

# Main execution
if __name__ == "__main__":
    N = 50  # Reduced from 1000
    avg_degree = 2

    # Erdős-Rényi network
    p = avg_degree / (N - 1)
    G_er = nx.erdos_renyi_graph(N, p)
    avalanche_sizes_er = simulate_sandpile(G_er, steps=1000)  # Reduced steps
    print(f"Erdős-Rényi: Mean avalanche size = {np.mean(avalanche_sizes_er):.2f}")
    plot_avalanche_distribution(avalanche_sizes_er,
                              f'Sandpile Avalanche Distribution (Erdős-Rényi, N={N}, <k>={avg_degree})')

    # Scale-free network
    degrees = generate_scale_free_degree_sequence(N, gamma=2.5, avg_degree=avg_degree)
    G_sf = nx.configuration_model(degrees)
    G_sf = nx.Graph(G_sf)  # Convert to simple graph
    avalanche_sizes_sf = simulate_sandpile(G_sf, steps=10)  # Reduced steps
    print(f"Scale-Free: Mean avalanche size = {np.mean(avalanche_sizes_sf):.2f}")
    plot_avalanche_distribution(avalanche_sizes_sf,
                              f'Sandpile Avalanche Distribution (Scale-Free, N={N}, <k>={avg_degree})')

#sample code with GML file

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Function to compute network metrics
def compute_metrics(G, name):
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)

    # Average shortest path length (for the largest connected component)
    if nx.is_directed(G):
        G_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    else:
        G_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    avg_path_length = nx.average_shortest_path_length(G_sub)

    # Average degree
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)

    print(f"\nMetrics for {name}:")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"Average Shortest Path Length: {avg_path_length:.4f}")
    print(f"Average Degree: {avg_degree:.4f}")
    return avg_clustering, avg_path_length, avg_degree, degrees

# Function to visualize network and degree distribution
def visualize_network(G, name, degrees):
    # Create figure with two subplots: network layout and degree distribution
    plt.figure(figsize=(12, 5))

    # Network layout (spring layout)
    plt.subplot(121)
    pos = nx.spring_layout(G, seed=42)  # Consistent layout for reproducibility
    nx.draw(G, pos, node_size=50, node_color='skyblue', edge_color='gray', with_labels=False)
    plt.title(f"{name} Layout")

    # Degree distribution histogram
    plt.subplot(122)
    plt.hist(degrees, bins=20, density=True, color='salmon', edgecolor='black')
    plt.title(f"{name} Degree Distribution")
    plt.xlabel('Degree')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()

# Load the dataset from a GML file
try:
    G_data = nx.read_gml('network.gml')
    print("Loaded network from GML file")
    avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G_data, "Input Network")
    visualize_network(G_data, "Input Network", degrees_data)
except FileNotFoundError:
    print("GML file not found. Please provide a valid GML file path.")
    # Fallback: Create a sample graph for demonstration
    G_data = nx.karate_club_graph()
    print("Using Karate Club graph as fallback")
    avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G_data, "Karate Club Network")
    visualize_network(G_data, "Karate Club Network", degrees_data)

# Number of nodes for generated networks
n = G_data.number_of_nodes()
# Approximate number of edges for realistic comparisons
m = G_data.number_of_edges()
# Estimate edge probability for Erdős-Rényi
p = (2 * m) / (n * (n - 1)) if not nx.is_directed(G_data) else m / (n * (n - 1))

# 1. Erdős-Rényi Graph
G_er = nx.erdos_renyi_graph(n, p)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G_er, "Erdős-Rényi Graph")
visualize_network(G_er, "Erdős-Rényi Graph", degrees_er)

# 2. Watts-Strogatz Model
k = int(np.mean([d for n, d in G_data.degree()]))
G_ws = nx.watts_strogatz_graph(n, k, 0.1)
avg_clust_ws, avg_path_ws, avg_deg_ws, degrees_ws = compute_metrics(G_ws, "Watts-Strogatz Graph")
visualize_network(G_ws, "Watts-Strogatz Graph", degrees_ws)

# 3. Scale-Free Network (Barabási-Albert model)
m_ba = max(1, int(m / n)) # Ensure at least 1 edge
G_sf = nx.barabasi_albert_graph(n, m_ba)
avg_clust_sf, avg_path_sf, avg_deg_sf, degrees_sf = compute_metrics(G_sf, "Scale-Free Network")
visualize_network(G_sf, "Scale-Free Network", degrees_sf)
