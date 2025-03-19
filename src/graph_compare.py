import json
import graphviz


# ------------------------------ FUNCTIONS -------------------------------------
def load_triplets_from_json(filename):
    """Wczytuje trójki z pliku JSON."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['triplets']

def create_graph(triplets, graph_name, common_nodes=None, file1_nodes=None, file2_nodes=None,file1_color="lightblue", file2_color="lightgreen"):
    """Tworzy graf z podanych trójek, oznaczając wspólne węzły."""
    dot = graphviz.Digraph(comment=graph_name)
    dot.attr('node', shape='box')
    dot.attr('edge', fontsize='10')

    file1_triplets = set()
    for t in file1_nodes:
        subject_name = t['subject']['name']
        file1_triplets.add(subject_name)
        object_name = t['object']['name']
        file1_triplets.add(object_name)

    file2_triplets = set()
    for t in file2_nodes:
        subject_name = t['subject']['name']
        file2_triplets.add(subject_name)
        object_name = t['object']['name']
        file2_triplets.add(object_name)

    znane = []

    for triplet in triplets:
        subject_name = triplet['subject']['name']
        predicate_name = triplet['predicate']['name']
        if predicate_name in ["instanceOf", "instance_of"]:
            continue
        object_name = triplet['object']['name']

        if (subject_name, predicate_name, object_name) in znane:
            continue
        else:
            znane.append((subject_name, predicate_name, object_name))

        subject_label = subject_name
        object_label = object_name

        # Ustawienie koloru węzła
        subject_style = {}
        object_style = {}

        if common_nodes:
            if subject_name not in common_nodes and subject_name in file1_triplets:
                subject_style["style"] = "filled"
                subject_style["fillcolor"] = f"{file1_color}"

            elif subject_name not in common_nodes and subject_name in file2_triplets:
                subject_style["style"] = "filled"
                subject_style["fillcolor"] = f"{file2_color}"

            if object_name not in common_nodes and object_name in file1_triplets:
                object_style["style"] = "filled"
                object_style["fillcolor"] = f"{file1_color}"

            elif object_name not in common_nodes and object_name in file2_triplets:
                object_style["style"] = "filled"
                object_style["fillcolor"] = f"{file2_color}"

        dot.node(subject_name, label=subject_label, **subject_style)
        dot.node(object_name, label=object_label, **object_style)
        dot.edge(subject_name, object_name, label=predicate_name)

    return dot


def find_common_nodes(triplets1, triplets2):
    """Znajduje wspólne węzły (po nazwie) w dwóch listach trójek."""
    nodes1 = set()
    for t in triplets1:
        nodes1.add(t['subject']['name'])
        nodes1.add(t['object']['name'])

    nodes2 = set()
    for t in triplets2:
        nodes2.add(t['subject']['name'])
        nodes2.add(t['object']['name'])

    return nodes1.intersection(nodes2)


def main():
    # --- Wczytanie danych z plików JSON ---
    file1 = "Rossi_Piotr_gpt4o-mini.json"
    file2 = "Rossi_Piotr_cot_1.json"

    triplets1 = load_triplets_from_json(file1)
    triplets2 = load_triplets_from_json(file2)

    # --- Znalezienie wspólnych węzłów ---
    common_nodes = find_common_nodes(triplets1, triplets2)
    print(f"Wspólne węzły: {common_nodes}")

    # --- przygotowanie grafu ---
    all_triplets = triplets1 + triplets2  # proste połączenie list.
    graph = create_graph(all_triplets, "Combined Graph", common_nodes, file1_nodes=triplets1, file2_nodes=triplets2)
    graph_u = graph.unflatten(stagger=4)

    # --- Zapisanie i wyrenderowanie grafu ---
    graph_u.render('combined_graph', view=True, format='png') # format png, pdf


if __name__ == "__main__":
    main()
