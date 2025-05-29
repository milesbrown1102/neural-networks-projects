import matplotlib.pyplot as plt
import networkx as nx

def visualize_student_subgraph(graph, student_id):
    neighbors = list(graph.neighbors(student_id))
    edges = [(student_id, n) for n in neighbors]

    for n in neighbors:
        for second_neighbor in graph.neighbors(n):
            edges.append((n, second_neighbor))

    subG = graph.edge_subgraph(edges).copy()
    subG.add_node(student_id, type="student")

    node_colors = []
    for node in subG.nodes():
        ntype = graph.nodes[node].get("type", "")
        node_colors.append({
            "student": "skyblue",
            "video": "green",
            "quiz": "orange",
            "question": "lightgrey"
        }.get(ntype, "white"))

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subG, seed=42)
    nx.draw(subG, pos, with_labels=True, node_color=node_colors, edge_color='gray', font_size=8)
    edge_labels = nx.get_edge_attributes(subG, 'relation')
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f"Knowledge Graph Subgraph for {student_id}")
    plt.show()
