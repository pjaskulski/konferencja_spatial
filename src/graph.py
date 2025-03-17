""" generowanie grafów """

import json
import graphviz

def create_graph_from_json(json_file_path, output_pdf_path):
    """
    Tworzy graf z pliku JSON i zapisuje go jako PDF używając Graphviz.

    Args:
        json_file_path: Ścieżka do pliku JSON.
        output_pdf_path: Ścieżka do pliku PDF, gdzie zostanie zapisany graf.
    """

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Błąd: Plik {json_file_path} nie został znaleziony.")
        return
    except json.JSONDecodeError:
        print(f"Błąd: Plik {json_file_path} zawiera błędy składni JSON.")
        return

    dot = graphviz.Digraph(comment='Graf relacji', format='pdf')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey') # Dodanie stylu do węzłów
    dot.attr('edge', fontsize='10')  # Zmniejszenie rozmiaru czcionki krawędzi

    added_nodes = set()  # Zbiór przechowujący dodane już węzły (żeby uniknąć duplikatów)

    for triplet in data['triplets']:
        subject = triplet['subject']
        predicate = triplet['predicate']
        obj = triplet['object']

        # --- Tworzenie etykiety dla węzła Subject ---
        subject_label = f"{subject['name']}\n"
        if 'type' in subject:
            subject_label += f"Typ: {subject['type']}\n"
        if 'wikihum' in subject and subject['wikihum'] != "NEW":  # Dodajemy wikihum, jeśli istnieje i nie jest "NEW"
            subject_label += f"Wikihum: {subject['wikihum']}\n"
        if 'description' in subject and subject['description'] is not None: # Dodajemy opis, o ile istnieje
            subject_label += f"Opis: {subject['description']}"
        subject_label = subject_label.strip()  # Usuwamy zbędne puste linie na końcu

        # --- Tworzenie etykiety dla węzła Object ---
        object_label = f"{obj['name']}\n"
        if obj['type'] == 'data':  # Dla typu 'data', wyświetlamy tylko 'name'
            object_label = obj['name']
        else:
            if 'type' in obj:
                object_label += f"Typ: {obj['type']}\n"
            if 'wikihum' in obj and obj['wikihum'] != "NEW":  # Dodajemy wikihum
                object_label += f"Wikihum: {obj['wikihum']}\n"
            if 'description' in obj and obj['description'] is not None:
                object_label += f"Opis: {obj['description']}"
        object_label = object_label.strip()

        # --- Dodawanie węzłów (zabezpieczenie przed duplikatami) ---
        if subject_label not in added_nodes:
            dot.node(subject_label, label=subject_label)
            added_nodes.add(subject_label)

        if object_label not in added_nodes:
            dot.node(object_label, label=object_label)
            added_nodes.add(object_label)

        # --- Dodawanie krawędzi ---
        dot.edge(subject_label, object_label, label=predicate['name'])


    # --- Renderowanie i zapis do PDF ---
    try:
        dot.render(output_pdf_path, view=False)  # view=False, żeby nie otwierać podglądu
        print(f"Graf został zapisany do pliku: {output_pdf_path}.pdf") # Zmienione, by uwzględniać rozszerzenie .pdf
    except graphviz.backend.ExecutableNotFound:
        print("Błąd: Nie znaleziono programu Graphviz (dot). Upewnij się, że jest zainstalowany i dostępny w PATH.")
    except Exception as e:
        print(f"Wystąpił błąd podczas renderowania grafu: {e}")


# --- Użycie ---
create_graph_from_json('Fiorentini_Wladyslaw.json', 'graf_fiorentini') # Nie podawaj rozszerzenia .pdf w nazwie wyjściowej
