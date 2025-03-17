""" graf na podstawie wyników """
import os
import json
import textwrap
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
    dot.attr('node', shape='box', style='rounded') #,filled', fillcolor='lightgrey')
    dot.attr('edge', fontsize='10')

    # Add nodes
    for triplet in kg:
        # relacje instanceOf są nadmiarowe - typ obiektu zapisany jest w node
        if triplet["predicate"]["name"] == "instanceOf":
            continue
        s_name = triplet["subject"]["name"]
        s_description = triplet.get("subject",{}).get("description", None)
        s_wikihum = triplet["subject"]["wikihum"]
        s_type = triplet["subject"]["type"]

        if s_description:
            s_description = textwrap.fill(s_description, width=25)
            node_name = f'{s_name}\n[{s_wikihum}]\n<{s_type}>\n({s_description})'

            node_label = f'{s_name}\n[{s_wikihum}]\n<{s_type}>\n({s_description})'
        else:
            node_name = f'{s_name}\n[{s_wikihum}]\n<{s_type}>'
            node_label = f'{s_name}\n[{s_wikihum}]\n<{s_type}>'

        dot.node(name=node_name, label=node_label, color="black")

    # Add edgee
    for triplet in kg:
        if triplet["predicate"]["name"] == "instanceOf":
            continue
        s_description = triplet.get("subject",{}).get("description", "")
        s_name = triplet["subject"]["name"]
        s_wikihum = triplet.get("subject",{}).get("wikihum",None)
        s_type = triplet["subject"]["type"]

        if s_description:
            s_description = textwrap.fill(s_description, width=25)
            tail_name = f'{s_name}\n[{s_wikihum}]\n<{s_type}>\n({s_description})'
        else:
            tail_name = f'{s_name}\n[{s_wikihum}]\n<{s_type}>'

        o_description = triplet.get("object",{}).get("description", None)
        o_name = triplet["object"]["name"]
        o_type = triplet.get("object", {}).get("type", None)
        o_wikihum = triplet.get("object",{}).get("wikihum",None)

        if o_description:
            o_description = textwrap.fill(o_description, width=25)
            if  o_type == "data":
                head_name = f'{o_name}\n({s_description})'
            else:
                head_name = f'{o_name}\n[{o_wikihum}]\n<{o_type}>\n({o_description})'
        else:
            if o_type == "data":
                head_name = f'{o_name}'
            else:
                head_name = f'{o_name}\n[{o_wikihum}]\n<{o_type}>'

        dot.edge(
            tail_name=tail_name,
            head_name=head_name,
            label=f'{triplet["predicate"]["name"]} [{triplet["predicate"]["wikihum"]}]',
            color="green",
        )

    # Render the graph
    output_file = filename.replace('.json','.gv')
    output_path = Path('..') / "output_pdf" / output_file
    dot_u = dot.unflatten(stagger=4)
    dot_u.render(output_path, view=False)


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":

    # dataset - dane po identyfikacji i walidacji
    data_folder = Path("..") / "output_link"
    data_file_list = data_folder.glob('*.json')

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)

        # pomijanie jeżeli pdf z grafem już istnieje
        pdf_file = file_name.replace('.json', '.pdf')
        pdf_path = Path("..") / "output_pdf" / pdf_file
        if os.path.exists(pdf_path):
            continue

        print(file_name)

        # odczytanie pliku json
        relations = read_json(input_file=data_file)
        print(f'size: {len(relations)}')
        # zapis grafu w pliku pdf
        visualize_knowledge_graph(kg=relations, filename=file_name)
