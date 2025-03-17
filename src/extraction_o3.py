""" knowledge extraction from text """
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
MODEL = MODEL_O3

CHAIN_OF_THOUGHT = False

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
        # lista przemyśleń modelu
        if CHAIN_OF_THOUGHT:
            for chain in item.chain_of_thought:
                with open('analiza.log', 'a', encoding='utf-8') as f_log:
                    f_log.write(f'  - {chain}\n')
                logging.info('  - %s', chain)

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
    output_path = Path('..') / "output_o3" / output_file
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


def object_analysis(client, data_structure, llm_model) -> dict:
    """ analiza pól description dla obiektów w celu wykrycia dodatkowych relacji """

    data_org = data_structure["triplets"]
    data_new = []

    additional = prepare_additional_prompt(data_structure)

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

    with open('analiza.log', 'a', encoding='utf-8') as f_log:
        f_log.write('Analiza pól "description"\n')

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
                obj_res = analiza(client,
                                  f'(dotyczy: {object_name}) {description}',
                                  llm_model,
                                  add_prompt=additional)
                data = []
                for item in obj_res.relations:
                    # lista przemyśleń modelu
                    if CHAIN_OF_THOUGHT:
                        for chain in item.chain_of_thought:
                            with open('analiza.log', 'a', encoding='utf-8') as f_log:
                                f_log.write(f'  - {chain}\n')
                            logging.info('  - %s', chain)

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


# -------------------------------- MAIN ----------------------------------------
if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-4s %(message)s',
        datefmt='%m-%d %H:%M:%S')
    logging.RootLogger.manager.getLogger('httpx').disabled = True

    logging.info("Model: %s", MODEL)

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
        json_path = Path('..') / "output_o3" / json_file
        if os.path.exists(json_path):
            continue

        if file_name != "Rossi_Piotr.txt":
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

        number_of_repetitions = 1
        for i in range(1, number_of_repetitions + 1):
            if i > 1:
                additional_user_prompt = prepare_additional_prompt(struktura)

            logging.info("Analiza tekstu biogramu (%s) ...", i)
            with open('analiza.log', 'a', encoding='utf-8') as f_log:
                f_log.write(f"Analiza tekstu biogramu ({i}) ...\n")
            res = analiza(client, biogram, MODEL, add_prompt=additional_user_prompt)

            # konwersja wyników do formy słownika (uzupełnienie istniejącego słownika
            # jeżeli biogram jest przetwarzany kolejny raz)
            logging.info("Zapis wyników w formie ustrukturyzowanej")
            struktura = export_to_dictionary(kg=res, curr_structure=struktura)
            time.sleep(2.0)

        # dodatkowa analiza treści pól description w celu wyszukania dodatkowych
        # relacji dla obiektów w dotychczasowych relacjach (tylko dla typu: 'osoba')
        logging.info("Dodatkowa analiza struktury wyników, wyszukiwanie dodatkowych relacji...")
        struktura = object_analysis(client, struktura, MODEL)

        # zapis wyników w pliku json
        logging.info("Zapis wyników w formacie json...")
        save_json(out_data=struktura, filename=file_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info("Czas wykonania programu: %s s.",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
