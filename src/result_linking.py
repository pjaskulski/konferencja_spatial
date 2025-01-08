""" entity linking """
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from wikibaseintegrator import WikibaseIntegrator, wbi_login
from wikibaseintegrator.wbi_config import config as wbi_config
from wikibaseintegrator import wbi_helpers
import openai
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field


# adresy dla API Wikibase (instancja docelowa)
wbi_config['MEDIAWIKI_API_URL'] = 'https://wikihum.lab.dariah.pl/api.php'
wbi_config['SPARQL_ENDPOINT_URL'] = 'https://wikihum.lab.dariah.pl/bigdata/sparql'
wbi_config['WIKIBASE_URL'] = 'https://wikihum.lab.dariah.pl'

# login i hasło ze zmiennych środowiskowych
env_path = Path(".") / ".env_wikihum"
load_dotenv(dotenv_path=env_path)

# OAuth
WIKIDARIAH_CONSUMER_TOKEN = os.environ.get('WIKIDARIAH_CONSUMER_TOKEN')
WIKIDARIAH_CONSUMER_SECRET = os.environ.get('WIKIDARIAH_CONSUMER_SECRET')
WIKIDARIAH_ACCESS_TOKEN = os.environ.get('WIKIDARIAH_ACCESS_TOKEN')
WIKIDARIAH_ACCESS_SECRET = os.environ.get('WIKIDARIAH_ACCESS_SECRET')

MODEL = "gpt-4o-mini"

# api key
env_path_openai = Path(".") / ".env"
load_dotenv(dotenv_path=env_path_openai)
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


wikihum_properties = {
    "instanceOf":"P27",
    "dateOfBirth": "P11",
    "dateOfDeath": "P12",
    "placeOfBirth": "P140",
    "placeOfDeath": "P141",
    "hasFather": "P92",
    "hasMother": "P93",
    "hasSon": "P153",
    "hasDaughter": "P154",
    "hasBrother": "P97",
    "hasSister": "P97",
    "hasWife": "P94",
    "hasHusband": "P94",
    "hasUncle": "P99",
    "hasAunt": "P99",
    "hasCousin": "P99",
    "hasGrandfather": "P91",
    "hasGrandmother": "P91",
    "significantFigure": "P100",
    "relatedInstitution": "P",
    "illegitimatePartner": "P95",
    "hasGrandson": "P110",
    "hasGranddaughter": "P110",
    "hasOccupation": "P106",
    "officeHeld": "P101",
    "positionHeld": "P101"
}

wikihum_elements = {
    "osoba":"Q5",            # human
    "miejscowość":"Q175698", # human settlement
    "kraj":"Q50"             # contry
}

# ------------------------------- CLASSES --------------------------------------
class BestCandidateModel(BaseModel):
    chain_of_thought: List[str] = Field(
        None,
        description="Kroki wyjaśniające prowadzące do ustalenia najlepszego wyboru z listy wyników. Te kroki powinny dostarczać szczegółowego uzasadnienia dlaczego wybrany wynik najbardzej pasuje do wskazanej nazwy i opisu."
    )
    best_fit: str = Field(description="Identyfikator (id) najbardziej pasującego wyniku")


# ------------------------------ FUNCTIONS -------------------------------------
def search_wikihum(nazwa:str) -> list:
    """ wyszukiwanie w wikihum """
    try:
        # Ograniczenie do 10 wyników i języka polskiego
        wyniki = wbi_helpers.search_entities(search_string=nazwa,
                                            max_results=10,
                                            language='pl',
                                            dict_result=True,
                                            strict_language=True)

        if wyniki:
            # Przetwarzanie wyników do bardziej czytelnej formy
            przetworzone_wyniki = []
            for wynik in wyniki:
                przetworzone_wyniki.append({
                    "id": wynik['id'],
                    "label": wynik['label'],
                    "description": wynik['description']
                })
            return przetworzone_wyniki
        else:
            return []  # Zwracamy pustą listę, gdy brak wyników

    except Exception as e:
        print(f"Wystąpił błąd podczas wyszukiwania: {e}")
        return None


def select_item(candidate_list:list, expected:str) -> str:
    """ wybór z listy kandydatów """
    if len(candidate_list) == 1:
        return candidate_list[0]["id"]

    client = instructor.from_openai(OpenAI())

    system_prompt = """
    Jesteś asystentem historyka i specjalistą od baz wiedzy. Z podanej listy wyników wyszukiwania w
    bazie wiedzy wybierz najlepiej pasujący do poszukiwanego elementu (element jest zwykle osobą
    lub miejscowiścią). Zwróć identyfikator najbardziej pasującego wyniku.
    """

    licznik = 1
    warianty = ""
    for element in candidate_list:
        warianty += f'\n\n{licznik}.\nID: {element["id"]}\nNAZWA: {element["label"]}\nOPIS: {element["description"]}\n\n'
        licznik += 1

    user_prompt = f"""
    Poszukiwany element:
    {expected}

    Lista wyników do rozpatrzenia:
    {warianty}
    """

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_model=BestCandidateModel,
        temperature = 0
    )

    return res.best_fit


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":

        # logowanie do instancji wikibase
    login_instance = wbi_login.OAuth1(consumer_token=WIKIDARIAH_CONSUMER_TOKEN,
                                     consumer_secret=WIKIDARIAH_CONSUMER_SECRET,
                                     access_token=WIKIDARIAH_ACCESS_TOKEN,
                                     access_secret=WIKIDARIAH_ACCESS_SECRET,
                                     token_renew_period=14400)

    wbi = WikibaseIntegrator(login=login_instance)

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

    for relation in relations:
        # property
        predicate = relation["predicate"]
        if predicate in wikihum_properties:
            relation["predicate_wikihum_p"] = wikihum_properties[predicate]
        else:
            relation["predicate_wikihum_p"] = "CREATE"

        # instance of
        subject_type = relation["subject_type"]
        if subject_type in wikihum_elements:
            relation["subject_instance_of"] = wikihum_elements[subject_type]

        object_type = relation["object_type"]
        if object_type in wikihum_elements:
            relation["object_instance_of"] = wikihum_elements[object_type]

        znane = []
        # próba identyfikacji osób i miejsc
        if relation["subject_type"] in ["osoba", "miejscowość"]:
            znane.append(relation["subject"])
            candidates = search_wikihum(relation["subject"])
            if candidates:
                dane = f'NAZWA: {relation["subject"]} ({relation["subject_type"]})\nOPIS: {relation["subject_description"]} '
                identified_item = select_item(candidates, dane)
                print(dane)
                print(identified_item)
                print()

        znane = []
        if relation["object_type"] in ["osoba", "miejscowość"] and relation["object"] not in znane:
            znane.append(relation["object"])
            candidates = search_wikihum(relation["object"])
            if candidates:
                dane = f'NAZWA: {relation["object"]} ({relation["object_type"]})\nOPIS: {relation["object_description"]} '
                identified_item = select_item(candidates, dane)
                print(dane)
                print(identified_item)
                print()
