""" knowledge extraction from text """
import os
import json
import time
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
import openai
from openai import OpenAI
from graphviz import Digraph


MODEL_O3 = "o3-mini"
MODEL_GPT_4O_MINI = "gpt-4o-mini"
MODEL_GPT_4O = "gpt-4o"

MODEL = MODEL_GPT_4O

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

predicat_list_with_descripton = [
    {
        "name": "instanceOf",
        "description": "jest to, czym jest podmiot relacji np. osoba, miejscowość, rzeka, instytucja itp.",
    },
    {
        "name": "dateOfBirth",
        "description": "data urodzenia dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany wpisz 00",
    },
    {
        "name": "dateOfDeath",
        "description": "data śmierci dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany wpisz 00",
    },
    {
        "name": "placeOfBirth",
        "description": "miejsce urodzenia dla osób"
    },
    {
        "name": "placeOfDeath",
        "description": "miejsce śmierci dla osób"
    },
    {
        "name": "dateOfBurial",
        "description": "data pochówku (pogrzebu) dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany wpisz 00",
    },
    {
        "name": "placeOfBurial",
        "description": "miejsce pochówku (grobu) dla osób"
    },
    {
        "name": "hasFather",
        "description": "ojciec osoby będącej podmiotem relacji"
    },
    {
        "name": "hasMother",
        "description": "matka osoby będącej podmiotem relacji"
    },
    {
        "name": "hasSon",
        "description": "syn osoby będącej podmiotem relacji"
    },
    {
        "name": "hasDaughter",
        "description": "córka osoby będącej podmiotem relacji"
    },
    {
        "name": "hasBrother",
        "description": "brat osoby będącej podmiotem relacji"
    },
    {
        "name": "hasSister", "description": "siostra osoby będącej podmiotem relacji"
    },
    {
        "name": "hasWife",
        "description": "żona osoby będącej podmiotem relacji"
    },
    {
        "name": "hasHusband",
        "description": "mąż osoby będącej podmiotem relacji"
    },
    {
        "name": "hasUncle",
        "description": "wuj osoby będącej podmiotem relacji"
    },
    {
        "name": "hasAunt",
        "description": "ciotka osoby będącej podmiotem relacji"
    },
    {
        "name": "hasCousin",
        "description": "kuzyn osoby będącej podmiotem relacji"
    },
    {
        "name": "hasGrandfather",
        "description": "dziadek osoby będącej podmiotem relacji",
    },
    {
        "name": "hasGrandmother",
        "description": "babcia osoby będącej podmiotem relacji"
    },
    {
        "name": "significantFigure",
        "description": "znacząca osoba (współpracownik, przyjaciel) dla osoby będącej podmiotem relacji",
    },
    {
        "name": "relatedInstitution",
        "description": "instytucja lub organizacja związana z osobą będącą podmiotem relacji",
    },
    {
        "name": "illegitimatePartner",
        "description": "nieformalny partner osoby będącej podmiotem relacji",
    },
    {
        "name": "hasGrandson",
        "description": "wnuk osoby będącej podmiotem relacji"
    },
    {
        "name": "hasGranddaughter",
        "description": "wnuczka osoby będącej podmiotem relacji",
    },
    {
        "name": "hasOccupation",
        "description": "zawód (wyuczony, wykonywany) osoby będącej podmiotem relacji",
    },
    {
        "name": "positionHeld",
        "description": "stanowisko (praca), urząd zajmowane przez osoby będącej podmiotem relacji",
    },
    {
        "name": "placeOfStudy",
        "description": "szkoła, uczelnia, instytucja edukacyjna w której uczyła się osoba będąca podmiotem relacji",
    },
    {
        "name": "significantLocality",
        "description": "miejsce, miejscowość związana z osobą będącą podmiotem relacji",
    },
    {
        "name": "causeOfDeath",
        "description": "przyczyna śmierci osoby będącej podmiotem relacji",
    },
    {
        "name": "nobleTitle ",
        "description": "tytuł szlachecki lub arystokratyczny (np. (książę, księżna, król, królowa, hrabia, hrabina, baron, baronowa, markiz, markiza itp.) posiadany przez osobę będącą podmiotem relacji",
    },
    {
        "name": "hasAssets",
        "description": "własność (nieruchomości, majątek, firmy) posiadana przez osobę będącą podmiotem relacji",
    },
    {
        "name": "isAuthorOf",
        "description": "dzieło, którego autorem jest osoba będąca podmiotem relacji",
    },
    {
        "name": "significantEvent",
        "description": "znaczące wydarzenie, w którym brała udiał osoba będąca podmiotem relacji"
    },
    {
        "name": "hasAchievement",
        "description": "znaczące osiągnięcia (sukcesy, nagrody) osoby będąca podmiotem relacji"
    },
    {
        "name": "religiousAffiliation",
        "description": "wyznanie religijne osoby będącej podmiotem relacji"
    },
    {
        "name": "hasStepson",
        "description": "przybrany syn osoby będącej podmiotem relacji"
    },
    {
        "name": "hasStepfather",
        "description": "przybrany ojciec osoby będącej podmiotem relacji"
    }
]

predicat_list = ""
for item in predicat_list_with_descripton:
    predicat_list += f'{item["name"]} - {item["description"]}\n'


class ObjectModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania obiektów w tekście. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów.",
    )
    name: str = Field(
        description="nazwa obiektu np. Arnold Kowiński, Kraków, rzeka Wisła, potok, miasto, zbroja, pług. Ten sam obiekt występujący wielokrotnie w analizowanym tekście powinien mieć przypisaną zawsze tą samą nazwę."
    )
    type: str = Field(
        description="typ obiektu np. osoba, miejscowość, obiekt fizjograficzny, obiekt hydrologiczny, przedmiot, instytucja, organizacja, urząd, zawód"
    )
    description: Optional[str] = Field(
        description="Dodatkowy, opcjonalny opis znalezionego obiektu, pochodzący wyłącznie z analizowanego tekstu. Ten sam obiekt występujący wielokrotnie w analizowanym tekście powinien mieć przypisany zawsze ten sam opis."
    )


class RelationModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania relacji między obiektami. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów i ich relacji.",
    )
    subject: ObjectModel = Field(
        description="podmiot relacji np. Bogdan ma brata: Adama Krzepkowskiego (podmiot: Bogdan)"
    )
    predicate: str = Field(
        description=f"nazwa relacji zachodzącej między podmiotem a obiektem np. {predicat_list}"
    )
    object: ObjectModel = Field(
        description="obiekt relacji np. Bogdan ma brata: Adama Krzepkowskiego (obiekt: Adam Krzepkowski)"
    )


class RelationCollection(BaseModel):
    relations: List[RelationModel] = Field(
        description="Lista relacji między obiektami występujacymi w tekście"
    )


def visualize_knowledge_graph(kg: dict, filename:str):
    """ zapis grafu w pliku pdf """
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg["triplets"]:
        if "description" in node["subject"] and node["subject"]["description"]:
            node_name = f'{node["subject"]["name"]}\n({node["subject"]["description"]})'
            #node_label = node.subject.name
        else:
            node_name = node["subject"]["name"]
            #node_label = node.subject.name

        dot.node(name=node_name, label=node_name, color="black")

    # Add edges
    for edge in kg["triplets"]:
        if "description" in edge["subject"] and edge["subject"]["description"]:
            subject_name = f'{edge["subject"]["name"]}\n({edge["subject"]["description"]})'
        else:
            subject_name = edge["subject"]["name"]

        if "description" in edge["object"] and edge["object"]["description"]:
            object_name = f'{edge["object"]["name"]}\n({edge["object"]["description"]})'
        else:
            object_name = edge["object"]["name"]

        dot.edge(
            tail_name=str(subject_name),
            head_name=str(object_name),
            label=edge["predicate"]["name"],
            color="green",
        )

    # Render the graph
    output_file = filename + ".gv"
    output_path = Path('..') / "output_dataset" / output_file
    dot_u = dot.unflatten(stagger=4)
    dot_u.render(output_path, view=False)


def export_to_dictionary(kg: RelationCollection, curr_structure:dict) -> dict:
    """ export wyników w formie słownika """

    # unikanie powtórzeń relacji, jeżeli biogram jest przetwarzany kolejny raz
    # lista relacji jest wypełniana na podstawie poprzednich wyników
    unikalne = []
    if curr_structure:
        for item in curr_structure["triplets"]:
            #subject_description = item["subject"].get("description", None)
            #object_description = item["object"].get("description", None)
            relacja = (
                item["subject"]["name"],
                #subject_description,
                item["predicate"]["name"],
                item["object"]["name"],
                #object_description
            )
            unikalne.append(relacja)
        data = curr_structure["triplets"]
    else:
        data = []

    for item in kg.relations:
        # lista przemyśleń modelu
        for chain in item.chain_of_thought:
            print(f'  - {chain}')

        relacja = (
            item.subject.name,
            #item.subject.description,
            item.predicate,
            item.object.name,
            #item.object.description
        )
        if relacja not in unikalne:
            record = { "subject" : {
                "name": item.subject.name,
                "type": item.subject.type,
                "description": item.subject.description
                },
                "predicate": {
                    "name": item.predicate
                },
                "object": {
                    "name": item.object.name,
                    "type": item.object.type,
                    "description": item.object.description
                }
            }
            data.append(record)
            unikalne.append(relacja)

    return {"triplets": data}


def save_json(out_data:dict, filename:str):
    """ zapis danych w pliku json """
    output_file = filename.replace('.txt', '.json')
    output_path = Path('..') / "output_dataset" / output_file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(out_data, f_out, indent=4, ensure_ascii=False)


def analiza(client, tekst_biogramu:str, llm_model, add_prompt=""):
    """ analiza biogramu """
    system_prompt = """
    Jesteś asystentem historyka badającego biografie postaci historycznych.
    Przeanalizuj podany tekst i zidentyfikuj obiekty występujące w tekście np. osoby,
    miejscowości, rzeki, instytucje, zawody, urzędy. Nazwy osób, miejscowości i innych
    obiektów zapisuj zawsze w formie mianownika. Następnie ustal relacje między tymi
    obiektami np. 'instanceOf', 'hasBrother', 'relatedInstitution', 'causeOfDeath' (dalej podano
    listę możliwych relacji).
    Wszystkie dane powinny być oparte na faktach, możliwe do zweryfikowania na podstawie tekstu,
    a nie na zewnętrznych założeniach. Upewnij się, że wyodrębnione zostały wszystkie
    znaczące obiekty (wszystie osoby, miejscowości) i ich wzajemne relacje z tekstu.
    Odpowiedzi udziel w języku polskim.
    """
    #czy dla znalezionych obiektów będących przedmiotem relacji znaleziono ich relacje np.
    #Kazimierz -> hasFather -> Władysław, Władysław -> nobleTitle -> hrabia.

    user_prompt = f"""
    {add_prompt}
    Tekst do analizy:
    {tekst_biogramu}
    """

    # Extract structured data from natural language
    if llm_model == "o3-mini":
        rezultat = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RelationCollection,
        )
    else:
        rezultat = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RelationCollection,
            temperature=0,
            seed=42,
        )

    return rezultat


def object_analysis(client, data_structure, llm_model) -> dict:
    """ analiza pól description dla obiektów w celu wykrycia dodatkowych relacji """

    data_org = data_structure["triplets"]
    data_new = []

    # unikalne trójki w dotychczasowych danych
    unikalne = []

    # odwrotności pokrewieństw/znajomości
    odwrotnosci = {
        "hasFather": ["hasSon", "hasDaughter"],
        "hasMother": ["hasSon", "hasDaughter"],
        "hasSon": ["hasFather", "hasMother"],
        "hasDaughter": ["hasFather", "hasMother"],
        "hasBrother": ["hasBrother", "hasSister"],
        "hasSister": ["hasBrother", "hasSister"],
        "hasWife": ["hasHusband", "hasWife"],
        "hasHusband": ["hasWife"],
        "significantFigure": ["significantFigure"],
        "hasStepfather": ["hasStepson"]
    }

    for item in data_org:
        relacja = (
            item["subject"]["name"],
            item["predicate"]["name"],
            item["object"]["name"],
        )
        unikalne.append(relacja)

    for element in data_org:
        if "description" in element["object"]:
            description = str(element["object"]["description"]).strip()
            object_name = element["object"]["name"]
            object_type = element["object"]["type"]

            # dodatkowa analiza tylko dla osób
            if object_type != "osoba":
                continue

            # jeżeli jest opis, opis jest dostatecznie długi i zawiera więcej niż 1 wyraz
            if description and len(description) > 15 and description.count(" ") > 1:
                obj_res = analiza(client, f'(dotyczy: {object_name}) {description}', llm_model)
                data = []
                for item in obj_res.relations:
                    # lista przemyśleń modelu
                    for chain in item.chain_of_thought:
                        print(f'  - {chain}')

                    record = { "subject" : {
                        "name": object_name,
                        "type": object_type,
                        "description": description
                        },
                        "predicate": {
                            "name": item.predicate
                        },
                        "object": {
                            "name": item.object.name,
                            "type": item.object.type,
                            "description": item.object.description
                        }
                    }

                    relacja = (
                        record["subject"]["name"],
                        record["predicate"]["name"],
                        record["object"]["name"],
                    )

                    # tylko nieznane relacje
                    if relacja not in unikalne:
                        # test czy to nie jest odwrotność istniejącej relacji
                        odwrotnosc = False
                        if record["predicate"]["name"] in odwrotnosci:
                            lista = odwrotnosci[record["predicate"]["name"]]
                            for l in lista:
                                odwrocona_relacja = (record["object"]["name"],
                                                     l,
                                                     record["subject"]["name"])
                                if odwrocona_relacja in unikalne:
                                    odwrotnosc = True
                                    break

                        if not odwrotnosc:
                            data.append(record)
                            unikalne.append(relacja)

                if data:
                    data_new += data


    if data_new:
        data_org += data_new
        data_structure["triplets"] = data_org

    return data_structure


def prepare_additional_prompt(struktura:dict) -> str:
    """ przygotowanie dodatkowego tektu do user_prompt """
    result = ""

    data = struktura["triplets"]
    if len(data) == 0:
        return result

    result = "Lista już znalezionych relacji: \n w formacie:\n subject (description) -> predicate -> object (description) \n\n"
    for element in data:
        if "description" in element["subject"] and element["subject"]["description"]:
            result += element["subject"]["name"] + '(' + element["subject"]["description"] + ')' + ' -> '
        else:
            result += element["subject"]["name"] + ' -> '

        result += element["predicate"]["name"] + ' -> '

        if "description" in element["object"] and element["object"]["description"]:
            result += element["object"]["name"] + '(' + element["object"]["description"] + ')\n'
        else:
            result += element["object"]["name"] + '\n'

    result += '\nWyszukaj brakujące relacje według podanych niżej poleceń. Pamiętaj by te same obiekty miały przypisane te same nazwy i opisy, tak by można je było poprawnie zidentyfikować.'
    return result


# --------------------------- MAIN ---------------------------------------------
if __name__ == '__main__':

    client = instructor.from_openai(OpenAI())

    # dataset
    data_folder = Path("..") / "dataset"
    data_file_list = data_folder.glob('*.txt')

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)

        # pomijanie biogramów już przetworzonych
        json_file = file_name.replace('.txt', '.json')
        json_path = Path('..') / "output_dataset" / json_file
        if os.path.exists(json_path):
            continue

        print(file_name)

        # wczytanie tekstu testowego biogramu
        data_file = Path("..") / "dataset" / file_name
        with open(data_file, "r", encoding="utf-8") as f:
            biogram = f.read()

        # do przetestowania:
        # -- przetwarzanie mniejszych części tekstu - podział na grupy zdań i przetwarzanie ich osobno

        # możliwość wielokrotnego przetwarzania biogramu przez model językowy
        # z zapisem unikalnych relacji
        struktura = {}
        additional_user_prompt = ""

        number_of_repetitions = 3
        for i in range(1, number_of_repetitions + 1):
            if i > 1:
                additional_user_prompt = prepare_additional_prompt(struktura)

            print(f"Analiza tekstu biogramu ({i}) ...")
            res = analiza(client, biogram, MODEL, add_prompt=additional_user_prompt)

            # konwersja wyników do formy słownika (uzupełnienie istniejącego słownika
            # jeżeli biogram jest przetwarzany kolejny raz)
            print("Zapis wyników w formie ustrukturyzowanej")
            struktura = export_to_dictionary(kg=res, curr_structure=struktura)
            time.sleep(1.0)

        # dodatkowa analiza treści pól description w celu wyszukania dodatkowych
        # relacji dla obiektów w dotychczasowych relacjach (tylko dla osób)
        #print("Dodatkowa analiza struktury wyników, wyszukiwanie dodatkowych relacji...")
        #struktura = object_analysis(client, struktura, MODEL)

        # zapis wyników w pliku json
        print("Zapis wyników w formacie json...")
        save_json(out_data=struktura, filename=file_name)

        # przygotowanie wizualizacji w formie grafu z zapisem do pdf
        #print("Przygotowanie wizualiacji grafu i zapis w pliku pdf...")
        #visualize_knowledge_graph(kg=struktura, filename=file_name)
