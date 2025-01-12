""" graf na podstawie wyników """
import json
from pathlib import Path
from graphviz import Digraph


# ----------------------------- FUNCTIONS --------------------------------------
def read_json(input_file:str):
    """ wczytanie danych z pliku json """
    input_path = Path('..') / "output" / input_file
    with open(input_path, 'r', encoding='utf-8') as f_in:
        json_data = json.load(f_in)

    return json_data['triplets']


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":

    # wczytanie wyników
    biogram_file = "Adam_Dabrowski_KG.json"
    relations = read_json(input_file=biogram_file)

    # graf
    dot = Digraph(comment="Knowledge Graph")
    dot.engine = "dot"

    # Add nodes
    for node in relations:
        subject = node["subject"]["name"]
        if "wikihum" in node["subject"]:
            subject += f'\n[{node["subject"]["wikihum"].replace("[","").replace("]","")}]'

        dot.node(name=subject, label=subject, color="black")

    # Add edges
    for edge in relations:
        predicate = edge["predicate"]["name"]
        if "wikihum" in edge["predicate"]:
            predicate += f'\n[{edge["predicate"]["wikihum"].replace("[","").replace("]","")}]'
        e_subject = edge["subject"]["name"]
        if "wikihum" in edge["subject"]:
            e_subject += f'\n[{edge["subject"]["wikihum"].replace("[","").replace("]","")}]'

        e_object = ""
        if "name" in edge["object"]:
            e_object = edge["object"]["name"]
        if "wikihum" in edge["object"]:
            e_object += f'\n[{edge["object"]["wikihum"].replace("[","").replace("]","")}]'

        dot.edge(
            tail_name=e_subject,
            head_name=e_object,
            label=predicate,
            color="green",
        )

    # Render the graph
    output_file = biogram_file.replace(".json", ".gv")
    dot_u = dot.unflatten(stagger=3)
    dot_u.render(filename=output_file, view=True)
