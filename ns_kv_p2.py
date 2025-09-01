"""## Read from file"""

import networkx as nx

# Unweighted edge list
G = nx.read_edgelist("network.txt", nodetype=int)

# Weighted edge list
Gw = nx.read_edgelist("network.txt", nodetype=int, data=(("weight", float),))

G = nx.read_adjlist("network.txt", nodetype=int)

#adj matrix
A = np.loadtxt("network.txt", dtype=int)
G = nx.from_numpy_array(A)

G = nx.read_gml("network.txt")

import csv

G = nx.Graph()
with open("network.txt") as f:
    reader = csv.reader(f, delimiter=",")  # change delimiter if needed
    for row in reader:
        u, v = row[0], row[1]
        G.add_edge(u, v)

"""## Generate using power-law"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

N = 103
avg_k = 2

# Generate ER graph
p = avg_k / (N - 1)
G_er = nx.erdos_renyi_graph(N, p, seed=42)

# Generate scale-free network using power-law degree sequence
gamma = 3  # exponent for power-law P(k) ~ k^-gamma
min_degree = 1

# Generate power-law degree sequence
degree_sequence = np.random.zipf(gamma, N)
# Ensure all degrees >= min_degree and sum is even
degree_sequence = np.maximum(degree_sequence, min_degree)
if sum(degree_sequence) % 2 != 0:
    degree_sequence[0] += 1

# Generate graph using configuration model
G_sf = nx.configuration_model(degree_sequence, seed=42)
G_sf = nx.Graph(G_sf)  # remove parallel edges
G_sf.remove_edges_from(nx.selfloop_edges(G_sf))

# Plot degree distributions
plt.figure(figsize=(10,5))

# ER degree distribution
degrees_er = [d for n, d in G_er.degree()]
plt.hist(degrees_er, bins=range(max(degrees_er)+2), alpha=0.6, label='ER', color='skyblue')

# Scale-free degree distribution
degrees_sf = [d for n, d in G_sf.degree()]
plt.hist(degrees_sf, bins=range(max(degrees_sf)+2), alpha=0.6, label='Power-law SF', color='orange')

plt.xlabel("Degree k")
plt.ylabel("Frequency")
plt.title("Degree Distributions")
plt.legend()
plt.show()

"""## Comparing Threshold

"""

import networkx as nx
import numpy as np

N = 103
avg_k = 2

# ER graph
p_er = avg_k / (N-1)
G_er = nx.erdos_renyi_graph(N, p_er)

# Power-law network
gamma = 3
degree_sequence_pl = np.random.zipf(gamma, N)
degree_sequence_pl = np.maximum(degree_sequence_pl, 1)
if sum(degree_sequence_pl) % 2 != 0:
    degree_sequence_pl[0] += 1
G_pl = nx.configuration_model(degree_sequence_pl)
G_pl = nx.Graph(G_pl)
G_pl.remove_edges_from(nx.selfloop_edges(G_pl))

# Uniform degree network
degree_sequence_uniform = np.random.randint(1, 5, N)  # degrees 1-4
if sum(degree_sequence_uniform) % 2 != 0:
    degree_sequence_uniform[0] += 1
G_uniform = nx.configuration_model(degree_sequence_uniform)
G_uniform = nx.Graph(G_uniform)
G_uniform.remove_edges_from(nx.selfloop_edges(G_uniform))

# Normal degree network
mean_k = avg_k
std_k = 1
degree_sequence_normal = np.random.normal(mean_k, std_k, N).astype(int)
degree_sequence_normal = np.clip(degree_sequence_normal, 1, N-1)
if sum(degree_sequence_normal) % 2 != 0:
    degree_sequence_normal[0] += 1
G_normal = nx.configuration_model(degree_sequence_normal)
G_normal = nx.Graph(G_normal)
G_normal.remove_edges_from(nx.selfloop_edges(G_normal))

def critical_threshold(G):
    degrees = np.array([d for n, d in G.degree()])
    k_avg = degrees.mean()
    k2_avg = (degrees**2).mean()
    return k_avg / (k2_avg - k_avg)

pc_er = critical_threshold(G_er)
pc_pl = critical_threshold(G_pl)
pc_uniform = critical_threshold(G_uniform)
pc_normal = critical_threshold(G_normal)

print("Critical thresholds:")
print(f"ER network: {pc_er:.4f}")
print(f"Power-law network: {pc_pl:.4f}")
print(f"Uniform degree network: {pc_uniform:.4f}")
print(f"Normal degree network: {pc_normal:.4f}")

"""## Similar to Wattz schtrogatz"""

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

N = 100
m = 2
p_rewire = 0.1

G = nx.cycle_graph(5)


for new_node in range(5, N):
    G.add_node(new_node)

    # Compute degrees and total degree
    degrees = np.array([G.degree(n) for n in G.nodes()])
    total_deg = degrees.sum()

    # Preferential attachment: select m nodes proportional to degree
    targets = set()
    while len(targets) < m:
        rand_node = np.random.choice(G.nodes(), p=degrees/total_deg)
        if rand_node != new_node:
            targets.add(rand_node)

    # Connect new node
    for t in targets:
        G.add_edge(new_node, t)

        # Step 3: Rewire with probability p_rewire
        if random.random() < p_rewire:
            G.remove_edge(new_node, t)
            possible_nodes = list(set(G.nodes()) - {new_node})
            new_target = random.choice(possible_nodes)
            G.add_edge(new_node, new_target)

# --- Analysis ---
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Average clustering coefficient:", nx.average_clustering(G))

# Diameter of largest component
largest_cc_nodes = max(nx.connected_components(G), key=len)
largest_cc = G.subgraph(largest_cc_nodes)
print("Diameter (largest component):", nx.diameter(largest_cc))


# --- Degree Distribution ---
degrees = [d for n, d in G.degree()]
plt.figure(figsize=(8,5))
plt.hist(degrees, bins=range(max(degrees)+1), alpha=0.7, color='orange')
plt.xlabel("Degree k")
plt.ylabel("Number of nodes")
plt.title("Degree Distribution of Hybrid Network")
plt.show()

# --- Visualize network ---
plt.figure(figsize=(8,8))
nx.draw_spring(G, node_size=50, node_color='skyblue', edge_color='gray')
plt.title("Hybrid Network Visualization")
plt.show()

"""## Separate communities"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 120       # total nodes
C = 3         # number of communities
n0 = 5        # initial nodes per community
m = 2         # edges per new node
p_inter = 0.05  # probability of inter-community edge

G = nx.Graph()
community_map = {}  # node -> community

# Step 1: Create seed communities
node_id = 0
for c in range(C):
    H = nx.erdos_renyi_graph(n0, 0.5)
    mapping = {n: n + node_id for n in H.nodes()}
    H = nx.relabel_nodes(H, mapping)
    G = nx.compose(G, H)
    for n in H.nodes():
        community_map[n] = c
    node_id += n0

# Step 2: Add new nodes
while G.number_of_nodes() < N:
    new_node = node_id
    G.add_node(new_node)
    node_id += 1

    # Pick a community randomly
    c = random.randint(0, C-1)

    # Preferential attachment inside community
    community_nodes = [n for n in G.nodes() if community_map[n] == c]
    degrees = np.array([G.degree(n) for n in community_nodes])
    total_deg = degrees.sum()

    targets = set()
    while len(targets) < m:
        if total_deg == 0:
            targets.add(random.choice(community_nodes))
        else:
            t = np.random.choice(community_nodes, p=degrees/total_deg)
            targets.add(t)

    # Connect new node
    for t in targets:
        if random.random() < p_inter:
            # Rewire to another community
            other_com = random.choice([i for i in range(C) if i != c])
            other_nodes = [n for n in G.nodes() if community_map[n] == other_com]
            t = random.choice(other_nodes)
        G.add_edge(new_node, t)

    community_map[new_node] = c

# --- Analysis ---
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Average clustering coefficient:", nx.average_clustering(G))

# Visualize communities
pos = nx.spring_layout(G, seed=42)
colors = [community_map[n] for n in G.nodes()]
plt.figure(figsize=(8,8))
nx.draw_networkx(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=100, with_labels=False)
plt.title("Community Network with Preferential Attachment")
plt.show()

"""## Communirty detection"""

import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

# Load graph from GML
G = nx.read_gml("graph.gml")

# Compute communities using Louvain
partition = community_louvain.best_partition(G)

# Number of communities
num_communities = len(set(partition.values()))
print("Number of communities:", num_communities)

# Modularity
mod = community_louvain.modularity(partition, G)
print("Modularity:", mod)

# Visualization
pos = nx.spring_layout(G)
colors = [partition[n] for n in G.nodes()]
nx.draw(G, pos, node_color=colors, cmap=plt.cm.tab20, with_labels=False, node_size=50)
plt.show()

"""## Phase Transition"""

import networkx as nx
import matplotlib.pyplot as plt

N = 1000
k_values = [0.5 + 0.05*i for i in range(20)]  # <k> from 0.5 to 1.45
largest_component_fraction = []

for k_avg in k_values:
    p = k_avg / (N-1)
    G = nx.erdos_renyi_graph(N, p)

    if len(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_component_fraction.append(len(largest_cc)/N)
    else:
        largest_component_fraction.append(0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(k_values, largest_component_fraction, marker='o')
plt.axvline(1, color='red', linestyle='--', label="Critical <k>=1")
plt.xlabel("Average degree <k>")
plt.ylabel("Fraction of nodes in largest component")
plt.title("Phase Transition in ER Networks")
plt.legend()
plt.grid(True)
plt.show()
