""" knowledge extraction from text - simple method without CoT i description """
import os
import json
import time
from typing import List, Optional
import logging
import logging.config
from pathlib import Path
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
import openai
from openai import OpenAI


MODEL_O3 = "o3-mini"
MODEL_GPT_4O_MINI = "gpt-4o-mini"
MODEL_GPT_4O = "gpt-4o"
MODEL_O1 = "o1"
MODEL_GPT_45 = "gpt-4.5-preview-2025-02-27"

MODEL = MODEL_GPT_4O

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# lista predykatów (właściwości)
with open('predicate.json', 'r', encoding='utf-8') as f:
    predicat_list_with_descripton = json.load(f)

predicat_list_string = ""
for item in predicat_list_with_descripton:
    predicat_list_string += f'{item["name"]} - {item["description"]}\n'

# lista typów obiektów
with open('object.json', 'r', encoding='utf-8') as f:
    object_list_with_descripton = json.load(f)

object_list_string = ""
for item in object_list_with_descripton:
    object_list_string += f'{item["name"]} - {item["description"]}\n'


# ------------------------------ CLASS -----------------------------------------
class ObjectModel(BaseModel):
    name: str = Field(
        description="nazwa obiektu np. Arnold Kowiński, Kraków, rzeka Wisła, potok, miasto, zbroja, pług. Ten sam obiekt występujący wielokrotnie w analizowanym tekście powinien mieć przypisaną zawsze tą samą nazwę."
    )
    type: str = Field(
        description=f"typ obiektu np. {object_list_string}"
    )
    description: Optional[str] = Field(
        description="Dodatkowy, opcjonalny opis znalezionego obiektu, pochodzący wyłącznie z analizowanego tekstu. Ten sam obiekt występujący wielokrotnie w analizowanym tekście powinien mieć przypisany zawsze ten sam opis."
    )


class RelationModel(BaseModel):
    subject: ObjectModel = Field(
        description="podmiot relacji np. Bogdan ma brata: Adama Krzepkowskiego (podmiot: Bogdan)"
    )
    predicate: str = Field(
        description=f"nazwa relacji zachodzącej między podmiotem a obiektem np. {predicat_list_string}"
    )
    object: ObjectModel = Field(
        description="obiekt relacji np. Bogdan ma brata: Adama Krzepkowskiego (obiekt: Adam Krzepkowski)"
    )


class RelationCollection(BaseModel):
    relations: List[RelationModel] = Field(
        description="Lista relacji między obiektami występujacymi w tekście"
    )


# ------------------------------ FUNCTIONS -------------------------------------
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
    output_path = Path('..') / "output_dataset_desc" / output_file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(out_data, f_out, indent=4, ensure_ascii=False)


def analiza(client, tekst_biogramu:str, llm_model, add_prompt=""):
    """ analiza biogramu """
    system_prompt = """
    Jesteś asystentem historyka badającego biografie postaci historycznych.
    Twoim zadaniem jest wydobycie informacji z tekstu w celu zbudowania grafu wiedzy.
    Przeanalizuj podany tekst i zidentyfikuj obiekty występujące w tekście (dalej podano
    listę możliwych typów obiektów). Następnie ustal relacje między tymi
    obiektami (dalej podano listę możliwych relacji). Postaraj się odnaleźć wszystkie relacje
    i obiekty z tekstu. 
    Wszystkie dane powinny być oparte na faktach, możliwe do zweryfikowania na podstawie tekstu,
    a nie na zewnętrznych założeniach. Upewnij się, że wyodrębnione zostały wszystkie
    znaczące obiekty (wszystie osoby, miejscowości) i ich wzajemne relacje z tekstu.
    Odpowiedzi udziel w języku polskim. Nazwy osób, miejscowości i innych
    obiektów zapisuj zawsze w formie mianownika. W przypadku osób podawaj nazwę
    zawsze w kolejności: imię nazwisko. Pomiń osoby, krewnych którzy są anomimowi
    (nie znane jest ani imię ani nazwisko).
    """
    #czy dla znalezionych obiektów będących przedmiotem relacji znaleziono ich relacje np.
    #Kazimierz -> hasFather -> Władysław, Władysław -> nobleTitle -> hrabia.

    user_prompt = f"""
    {add_prompt}
    Tekst do analizy:
    {tekst_biogramu}
    """

    rezultat = None

    # Extract structured data from natural language
    if llm_model == "o3-mini" or llm_model == "o1":
        rezultat = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RelationCollection,
        )
    elif llm_model == MODEL_GPT_4O or llm_model == MODEL_GPT_4O_MINI or llm_model == MODEL_GPT_45:
        rezultat = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RelationCollection,
            temperature=0,
            seed=2,
        )

    return rezultat


# -------------------------------- MAIN ----------------------------------------
if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-4s %(message)s',
        datefmt='%m-%d %H:%M:%S')
    logging.RootLogger.manager.getLogger('httpx').disabled = True

    client = instructor.from_openai(OpenAI())

    # dataset
    data_folder = Path("..") / "dataset"
    data_file_list = data_folder.glob('*.txt')

    for data_file in data_file_list:
        # pomiar czasu wykonania
        start_time = time.time()

        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)

        # pomijanie biogramów już przetworzonych
        json_file = file_name.replace('.txt', '.json')
        json_path = Path('..') / "output_dataset_desc" / json_file
        if os.path.exists(json_path):
            continue

        logging.info(file_name)
        with open('analiza.log', 'a', encoding='utf-8') as f_log:
            f_log.write(f'{file_name}\n')

        # wczytanie tekstu testowego biogramu
        data_file = Path("..") / "dataset" / file_name
        with open(data_file, "r", encoding="utf-8") as f:
            biogram = f.read()

        # do przetestowania:
        # -- przetwarzanie mniejszych części tekstu - podział na grupy zdań
        # i przetwarzanie ich osobno

        # możliwość wielokrotnego przetwarzania biogramu przez model językowy
        # z zapisem unikalnych relacji
        struktura = {}
        additional_user_prompt = ""


        logging.info("Analiza tekstu biogramu ...")
        with open('analiza.log', 'a', encoding='utf-8') as f_log:
            f_log.write("Analiza tekstu biogramu ...\n")
        res = analiza(client, biogram, MODEL, add_prompt=additional_user_prompt)

        # konwersja wyników do formy słownika (uzupełnienie istniejącego słownika
        # jeżeli biogram jest przetwarzany kolejny raz)
        logging.info("Zapis wyników w formie ustrukturyzowanej")
        struktura = export_to_dictionary(kg=res, curr_structure=struktura)
        time.sleep(2.0)

        # zapis wyników w pliku json
        logging.info("Zapis wyników w formacie json...")
        save_json(out_data=struktura, filename=file_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info("Czas wykonania programu: %s s.",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
