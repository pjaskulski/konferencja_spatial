""" graf na podstawie wyników """
from pathlib import Path
from graphviz import Digraph


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":

    # wczytanie wyników
    data_file = Path('..') / "output" / "wynik_gpt_4o_parts.txt"
    with open(data_file, 'r', encoding='utf-8') as f:
        results = f.readlines()
        results = [x.strip() for x in results]

    relations = []
    relation = {}

    for line in results:
        if line == "---":
            relations.append(relation)
            relation = {}
        elif line.startswith("subject  :"):
            line = line.replace("subject  :","").strip()
            tmp = line.split("' ")
            for item in tmp:
                tmp2 = item.split("=")
                if tmp2[0].strip() == "name":
                    relation["subject"] = tmp2[1].strip().strip("'")
                elif tmp2[0].strip() == "type":
                    relation["subject_type"] = tmp2[1].strip().strip("'")
                elif tmp2[0].strip() == "description":
                    relation["subject_description"] = tmp2[1].strip().strip("'")
        elif line.startswith("predicate:"):
            line = line.replace("predicate:","").strip()
            relation["predicate"] = line
        elif line.startswith("object   :"):
            line = line.replace("object   :","").strip()
            tmp = line.split("' ")
            for item in tmp:
                tmp2 = item.split("=")
                if tmp2[0].strip() == "name":
                    relation["object"] = tmp2[1].strip().strip("'")
                elif tmp2[0].strip() == "type":
                    relation["object_type"] = tmp2[1].strip().strip("'")
                elif tmp2[0].strip() == "description":
                    relation["object_description"] = tmp2[1].strip().strip("'")

    if relation:
        relations.append(relation)

    dot = Digraph(comment="Knowledge Graph")
    dot.engine = "dot"

    # Add nodes
    for node in relations:
        dot.node(name=node["subject"]+ f' \n({node["subject_type"]})', label=node["subject"]+ f' \n({node["subject_type"]})', color="black")

    # Add edges
    for edge in relations:
        dot.edge(
            tail_name=str(edge["subject"]) + f' \n({edge["subject_type"]})',
            head_name=str(edge["object"])+ f' \n({edge["object_type"]})',
            label=edge["predicate"],
            color="green",
        )

    # Render the graph
    dot_u = dot.unflatten(stagger=3)
    dot_u.render("knowledge_graph_4o_parts.gv", view=True)
