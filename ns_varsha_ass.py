# -----------------------
# 5) ASSIGNMENT-SPECIFIC SIMULATIONS
# -----------------------
# 5.1 Attack simulations (Programming Assignment 3)
def simulate_attack(G, criterion='degree', fractions=np.linspace(0,0.5,11), directed=False):
    """Simulate removal of fraction f of nodes ranked by criterion (degree or clustering).
       Returns list of normalized giant component sizes for each fraction f.
    """
    G0 = G.copy()
    N = G0.number_of_nodes()
    sizes = []
    if criterion == 'degree':
        if directed:
            # use total degree for ranking
            nodes_sorted = sorted(G0.nodes(), key=lambda n: (G0.in_degree(n) + G0.out_degree(n)), reverse=True)
        else:
            nodes_sorted = sorted(G0.nodes(), key=lambda n: G0.degree(n), reverse=True)
    elif criterion == 'clustering':
        cl = nx.clustering(G0.to_undirected()) if directed else nx.clustering(G0)
        nodes_sorted = sorted(G0.nodes(), key=lambda n: cl.get(n, 0), reverse=True)
    else:
        raise ValueError("criterion must be 'degree' or 'clustering'")

    for f in fractions:
        k = int(np.round(f * N))
        Gtemp = G0.copy()
        remove_nodes = nodes_sorted[:k]
        Gtemp.remove_nodes_from(remove_nodes)
        if Gtemp.number_of_nodes() == 0:
            sizes.append(0.0)
        else:
            if is_directed_graph(Gtemp):
                comp = max(nx.weakly_connected_components(Gtemp), key=len) if nx.number_weakly_connected_components(Gtemp) > 0 else set()
            else:
                comp = max(nx.connected_components(Gtemp), key=len) if nx.number_connected_components(Gtemp) > 0 else set()
            sizes.append(len(comp) / N)
    return sizes

# 5.2 Sandpile simulation (Programming Assignment 4)
def simulate_sandpile(G, steps=2000):
    """Sandpile avalanche model: bucket capacity = node degree.
       Returns list of avalanche sizes (number of toppled nodes) per perturbation.
    """
    # Work with a simple graph structure for neighbors
    G_use = G.to_undirected() if is_directed_graph(G) else G
    buckets = {n: max(1, G_use.degree(n)) for n in G_use.nodes()}  # bucket size at least 1
    grains = {n: 0 for n in G_use.nodes()}
    avalanches = []

    nodes_list = list(G_use.nodes())
    for t in range(steps):
        i = random.choice(nodes_list)
        grains[i] += 1
        unstable = deque([i]) if grains[i] >= buckets[i] else deque()
        toppled = set()
        while unstable:
            u = unstable.popleft()
            if grains[u] >= buckets[u]:
                toppled.add(u)
                num = grains[u]
                grains[u] = 0
                neighbors = list(G_use.neighbors(u))
                if len(neighbors) == 0:
                    continue
                # distribute equally (integer division) and random remainder distribution
                per = num // len(neighbors)
                rem = num % len(neighbors)
                for nb in neighbors:
                    grains[nb] += per
                # distribute remainder to random neighbors
                for r in range(rem):
                    nb = random.choice(neighbors)
                    grains[nb] += 1
                for nb in neighbors:
                    if grains[nb] >= buckets[nb] and nb not in toppled:
                        unstable.append(nb)
        if len(toppled) > 0:
            avalanches.append(len(toppled))
    return avalanches
