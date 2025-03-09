""" graf na podstawie wyników """
import os
import json
from pathlib import Path
from graphviz import Digraph


# ----------------------------- FUNCTIONS --------------------------------------
def read_json(input_file:str):
    """ wczytanie danych z pliku json """
    input_path = Path('..') / "output_dataset_link" / input_file
    with open(input_path, 'r', encoding='utf-8') as f_in:
        json_data = json.load(f_in)

    return json_data['triplets']


def visualize_knowledge_graph(kg:list, filename:str):
    """ zapis grafu w pliku pdf """
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for triplet in kg:
        if triplet.get("subject",{}).get("description", None):
            node_name = f'{triplet["subject"]["name"]}\n[{triplet["subject"]["wikihum"]}]\n({triplet["subject"]["description"]})'

            node_label = f'{triplet["subject"]["name"]}\n[{triplet["subject"]["wikihum"]}]\n({triplet["subject"]["description"]})'
        else:
            node_name = f'{triplet["subject"]["name"]}\n[{triplet["subject"]["wikihum"]}]'
            node_label = f'{triplet["subject"]["name"]}\n[{triplet["subject"]["wikihum"]}]'

        dot.node(name=node_name, label=node_label, color="black")

    # Add edgee
    for triplet in kg:
        if triplet.get("subject",{}).get("description", ""):
            subject_name = f'{triplet["subject"]["name"]}\n[{triplet["subject"]["wikihum"]}]\n({triplet["subject"]["description"]})'
        else:
            subject_name = f'{triplet["subject"]["name"]}\n[{triplet["subject"]["wikihum"]}]'

        if triplet.get("object",{}).get("description", None):
            if triplet.get("object", {}).get("type", None) == "data":
                object_name = f'{triplet["object"]["name"]}\n({triplet["object"]["description"]})'
            else:
                object_name = f'{triplet["object"]["name"]}\n[{triplet["object"]["wikihum"]}]\n({triplet["object"]["description"]})'
        else:
            if triplet.get("object", {}).get("type", None) == "data":
                object_name = f'{triplet["object"]["name"]}'
            else:
                object_name = f'{triplet["object"]["name"]}\n[{triplet["object"]["wikihum"]}]'

        dot.edge(
            tail_name=subject_name,
            head_name=object_name,
            label=f'{triplet["predicate"]["name"]}\n[{triplet["predicate"]["wikihum"]}]',
            color="green",
        )

    # Render the graph
    output_file = filename.replace('.json','.gv')
    output_path = Path('..') / "output_dataset_pdf" / output_file
    dot_u = dot.unflatten(stagger=4)
    dot_u.render(output_path, view=False)


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":

    # dataset
    data_folder = Path("..") / "output_dataset_link"
    data_file_list = data_folder.glob('*.json')

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)

        # pomijanie jeżeli pdf z grafem już istnieje
        pdf_file = file_name.replace('.json', '.pdf')
        pdf_path = Path("..") / "output_dataset_pdf" / pdf_file
        if os.path.exists(pdf_path):
            continue

        print(file_name)

        # odczytanie pliku json
        relations = read_json(input_file=data_file)
        print(f'size: {len(relations)}')
        # zapis grafu w pliku pdf
        visualize_knowledge_graph(kg=relations, filename=file_name)
