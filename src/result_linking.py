""" entity linking """
import os
import time
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from wikibaseintegrator import WikibaseIntegrator, wbi_login
from wikibaseintegrator.wbi_config import config as wbi_config
from wikibaseintegrator import wbi_helpers
import openai
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim


# adresy dla API Wikibase (instancja docelowa)
wbi_config['MEDIAWIKI_API_URL'] = 'https://wikihum.lab.dariah.pl/api.php'
wbi_config['SPARQL_ENDPOINT_URL'] = 'https://wikihum.lab.dariah.pl/bigdata/sparql'
wbi_config['WIKIBASE_URL'] = 'https://wikihum.lab.dariah.pl'

# login i hasło ze zmiennych środowiskowych
env_path = Path(".") / ".env_wikihum"
load_dotenv(dotenv_path=env_path)

# OAuth (wikihum)
WIKIDARIAH_CONSUMER_TOKEN = os.environ.get('WIKIDARIAH_CONSUMER_TOKEN')
WIKIDARIAH_CONSUMER_SECRET = os.environ.get('WIKIDARIAH_CONSUMER_SECRET')
WIKIDARIAH_ACCESS_TOKEN = os.environ.get('WIKIDARIAH_ACCESS_TOKEN')
WIKIDARIAH_ACCESS_SECRET = os.environ.get('WIKIDARIAH_ACCESS_SECRET')

# wikidata
wbd = WikibaseIntegrator()

# logowanie do WikiHum (instancji wikibase), w zasadzie zbędne do wyszukiwania
login_instance = wbi_login.OAuth1(consumer_token=WIKIDARIAH_CONSUMER_TOKEN,
                                    consumer_secret=WIKIDARIAH_CONSUMER_SECRET,
                                    access_token=WIKIDARIAH_ACCESS_TOKEN,
                                    access_secret=WIKIDARIAH_ACCESS_SECRET,
                                    token_renew_period=14400)

wbi = WikibaseIntegrator(login=login_instance)


# stałe
MODEL = "gpt-4o-mini"
P_INSTANCE_OF = "P27"
P_INSTANCE_OF_WIKIDATA = "P31"
Q_HUMAN_WIKIDATA = "Q5"

# api key (OpenAI)
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
    "kraj":"Q50",            # contry
    "region": "Q48",         # region
    "obiekt fizjograficzny": "Q51", # physiographic object
    "obiekt hydrograficzny": "Q53"  # hydrographic object
}

# ------------------------------- CLASSES --------------------------------------
class BestCandidateModel(BaseModel):
    """ struktura odpowiedzi modelu językowego  - wybór najbardziej pasującego wyniku """
    chain_of_thought: List[str] = Field(
        None,
        description="Kroki wyjaśniające prowadzące do ustalenia najlepszego wyboru z listy wyników. Te kroki powinny dostarczać szczegółowego uzasadnienia dlaczego wybrany wynik najbardzej pasuje do wskazanej nazwy i opisu."
    )
    best_fit: str = Field(description="Identyfikator (id) najbardziej pasującego wyniku, jeżeli pasującego wyniku brak - pusty string")


# ------------------------------ FUNCTIONS -------------------------------------
def search_wikihum(nazwa:str, typ_obiektu:str) -> list:
    """ wyszukiwanie w wikihum """
    try:
        # Ograniczenie do 10 wyników i języka polskiego
        wyniki = wbi_helpers.search_entities(search_string=nazwa,
                                            mediawiki_api_url="https://wikihum.lab.dariah.pl/api.php",
                                            max_results=10,
                                            language='pl',
                                            dict_result=True,
                                            strict_language=True)

        if wyniki:
            # Przetwarzanie wyników do bardziej czytelnej formy
            przetworzone_wyniki = []
            for wynik in wyniki:
                source_item = wbi.item.get(entity_id=wynik['id'])

                # weryfikacja czy typ wyniku (właściwość instance of z wikihum)
                # jest zgodna z oczekiwanym typem obiektu (osoba, miejscowość)
                claims = source_item.claims.get(P_INSTANCE_OF)
                if claims:
                    for claim in claims:
                        instance_of_value = claim.mainsnak.datavalue["value"]["id"]
                        claim_value_item = wbi.item.get(entity_id=instance_of_value)
                        value = claim_value_item.labels.get("pl")
                        if value == "jednostka osadnicza":
                            value = "miejscowość"
                            break

                        if value == "człowiek":
                            value = "osoba"
                            break

                    if value != typ_obiektu:
                        continue

                przetworzone_wyniki.append({
                    "id": wynik['id'],
                    "label": wynik['label'],
                    "description": wynik['description']
                })
            return przetworzone_wyniki

        return []  # Zwracamy pustą listę, gdy brak wyników


    except Exception as e:
        print(f"Wystąpił błąd podczas wyszukiwania: {e}")
        return None


def search_wikidata(nazwa:str, typ_obiektu:str) -> list:
    """ wyszukiwanie w wikidata """

    typy_wikidata = [
        "Q106071744", # former town
        "Q111737893", # private city
        "Q22674925",  # former settlement
        "Q129676371", # rural settlement
        "Q14616455",  # destroyed city
        "Q18466176",  # small city
        "Q2974842",   # lost city
        "Q1549591",   # big city
        "Q3957",      # town
        "Q5084",      # hamlet
        "Q74047",     # ghost town
        "Q350895",    # abandoned village
        "Q515",       # city
        "Q532",       # village
    ]

    try:
        # Ograniczenie do 10 wyników i języka polskiego
        wyniki = wbi_helpers.search_entities(search_string=nazwa,
                                            mediawiki_api_url="https://www.wikidata.org/w/api.php",
                                            max_results=10,
                                            language='pl',
                                            dict_result=True,
                                            strict_language=True,
                                            user_agent="MyWikiDataScript/1.0 (https://www.wikidata.org/wiki/User:MyUsername)")

        if wyniki:
            # Przetwarzanie wyników do bardziej czytelnej formy
            przetworzone_wyniki = []
            for wynik in wyniki:
                source_item = wbd.item.get(entity_id=wynik["id"],
                                        mediawiki_api_url="https://www.wikidata.org/w/api.php",
                                        user_agent="MyWikiDataScript/1.0 (https://www.wikidata.org/wiki/User:MyUsername)")

                # weryfikacja czy typ wyniku (właściwość instance of z wikihum)
                # jest zgodna z oczekiwanym typem obiektu (osoba, miejscowość)
                valid_item_instance = False
                claims = source_item.claims.get(P_INSTANCE_OF_WIKIDATA)
                if claims:
                    for claim in claims:
                        instance_of_value = claim.mainsnak.datavalue["value"]["id"]
                        if typ_obiektu == "miejscowość" and instance_of_value in typy_wikidata:
                            valid_item_instance = True
                            break
                        if typ_obiektu == "osoba" and instance_of_value == Q_HUMAN_WIKIDATA:
                            valid_item_instance = True
                            break

                    # pomijanie wyników niezgodnych
                    if not valid_item_instance:
                        continue

                przetworzone_wyniki.append({
                    "id": wynik['id'],
                    "label": wynik['label'],
                    "description": wynik['description']
                })
            return przetworzone_wyniki

        return []  # Zwracamy pustą listę, gdy brak wyników


    except Exception as e:
       print(f"Wystąpił błąd podczas wyszukiwania: {e}")
       return None


def search_nominatim(nazwa:str) -> tuple:
    """ wyszukiwanie miejscowości w serwisie Nominatim (OpenStreetMap) """

    geolocator = Nominatim(user_agent="WikiHum KG creator")
    location = geolocator.geocode(query=nazwa, namedetails=True, extratags=True)

    # przerwa by nie przekroczyć limitów serwera nominatim
    time.sleep(1.0)

    if location:
        # wyniki tylko z odpowiednio dużą wagą
        importance = location.raw.get('importance')
        if float(importance) >= 0.4:
            nom_id = location.raw.get('place_id', None)
            extra = location.raw.get('extratags', None)
            if extra:
                nom_wikidata = extra.get('wikidata', None)
            else:
                nom_wikidata = None
            nom_name = location.raw.get('display_name', None)

            return nom_id, nom_wikidata, nom_name

    return None, None, None


def select_item(candidate_list:list, expected:str) -> str:
    """ wybór z listy kandydatów """
    if len(candidate_list) == 1:
        return candidate_list[0]["id"]

    client = instructor.from_openai(OpenAI())

    system_prompt = """
    Jesteś asystentem historyka i specjalistą od baz wiedzy. Z podanej listy wyników wyszukiwania w
    bazie wiedzy wybierz najlepiej pasujący do poszukiwanego elementu (element jest zwykle osobą
    lub miejscowiścią). Zwróć identyfikator najbardziej pasującego wyniku lub, jeżeli brak pasujących wyników zwróć pusty string.
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

    # pomiar czasu wykonania
    start_time = time.time()

    # wczytanie wyników pracy modelu LLM (triplety do grafu wiedzy)
    print('Wczytanie danych...')

    input_path = Path('..') / "output" / "Adam_Dabrowski.json"
    with open(input_path, 'r', encoding='utf-8') as f_in:
        json_data = json.load(f_in)
        relations = json_data['triplets']

    znane_subject = {}
    znane_object = {}
    subjects_instance_of = {} # lista unikalnych subjektów
    objects_instance_of = {} # lista unikalnych obiektów

    print("Identyfikacja osób i miejscowości...")

    for relation in relations:
        print(relation["subject"], '->', relation["predicate"], '->', relation["object"])

        # property
        predicate = relation["predicate"]
        if predicate in wikihum_properties:
            relation["predicate_wikihum_p"] = wikihum_properties[predicate]
        else:
            relation["predicate_wikihum_p"] = f'[{predicate}]'

        # instance of dla subject
        subject_type = relation["subject_type"]
        if subject_type in wikihum_elements:
            relation["subject_instance_of"] = wikihum_elements[subject_type]
        else:
            relation["subject_instance_of"] = f'[{subject_type}]'

        # instance of dla object
        object_type = relation["object_type"]
        if object_type in wikihum_elements:
            relation["object_instance_of"] = wikihum_elements[object_type]
        else:
            if object_type != 'data':
                relation["object_instance_of"] = f'[{object_type}]'

        # próba identyfikacji osób i miejsc w subject
        if (relation["subject_type"] in ["osoba", "miejscowość"]
                and relation["subject"] not in znane_subject):
            # wyszukiwanie w wikihum
            candidates = search_wikihum(relation["subject"], relation["subject_type"])
            if candidates:
                dane = f'NAZWA: {relation["subject"]} ({relation["subject_type"]})\nOPIS: {relation["subject_description"]} '
                identified_item = select_item(candidates, dane)
                if identified_item and identified_item.strip() != "":
                    relation["subject_qid"] = identified_item
                    znane_subject[(relation["subject"], relation["subject_type"])] = identified_item
        elif (relation["subject"], relation["subject_type"]) in znane_subject:
            relation["subject_qid"] = znane_subject[(relation["subject"], relation["subject_type"])]
        else:
            # wyszukiwanie w wikidata
            candidates = search_wikidata(relation["subject"], relation["subject_type"])
            if candidates:
                dane = f'NAZWA: {relation["subject"]} ({relation["subject_type"]})\nOPIS: {relation["subject_description"]} '
                identified_item = select_item(candidates, dane)
                if identified_item and identified_item.strip() != "":
                    relation["subject_wikidata"] = identified_item

        # subject: wyszukiwanie w Nominatim (tylko miejscowości)
        if ("subject_qid" not in relation and "subject_wikidata" not in relation
                and relation["subject_type"] == "miejscowość"):
            n_id, n_wikidata, n_name = search_nominatim(relation["subject"])
            if n_id:
                relation["subject_nominatim"] = n_id
                if n_wikidata:
                    relation["subject_wikidata"] = n_wikidata
                if n_name:
                    relation["subject_nominatim_name"] = n_name

        if "subject_qid" not in relation and relation["subject_type"] != "data":
            relation["subject_qid"] = "NEW"

        # próba identyfikacji osób i miejsc w object
        if relation["object_type"] in ["osoba", "miejscowość"] and relation["object"] not in znane_object:
            # wyszukiwanie w wikihum
            candidates = search_wikihum(relation["object"], relation["object_type"])
            if candidates:
                dane = f'NAZWA: {relation["object"]} ({relation["object_type"]})\nOPIS: {relation["object_description"]} '
                identified_item = select_item(candidates, dane)
                if identified_item and identified_item.strip() != "":
                    relation["object_qid"] = identified_item
                    znane_object[(relation["object"], relation["object_type"])] = identified_item
        elif (relation["object"], relation["object_type"]) in znane_object:
            relation["object_qid"] = znane_object[(relation["object"], relation["object_type"])]
        else:
             # wyszukiwanie w wikidata
            candidates = search_wikidata(relation["object"], relation["object_type"])
            if candidates:
                dane = f'NAZWA: {relation["object"]} ({relation["object_type"]})\nOPIS: {relation["object_description"]} '
                identified_item = select_item(candidates, dane)
                if identified_item and identified_item.strip() != "":
                    relation["object_wikidata"] = identified_item

        # object: wyszukiwanie w Nominatim (tylko miejscowości)
        if ("object_qid" not in relation and "object_wikidata" not in relation
                and relation["object_type"] == "miejscowość"):
            n_id, n_wikidata, n_name = search_nominatim(relation["object"])
            if n_id:
                relation["object_nominatim"] = n_id
                if n_wikidata:
                    relation["object_wikidata"] = n_wikidata
                if n_name:
                    relation["object_nominatim_name"] = n_name

        if "object_qid" not in relation and relation["object_type"] != "data":
            relation["object_qid"] = "NEW"


    # zapis wyników w pliku json
    print("Zapis wyników w pliku json...")
    data = []
    for relation in relations:
        # subject
        r_subject = {
                "name": relation["subject"],
                "type": relation["subject_type"],
                "description": relation["subject_description"]
            }
        if "subject_qid" in relation:
            r_subject["wikihum"] = relation["subject_qid"]
        if "subject_wikidata" in relation:
            r_subject["wikidata"] = relation["subject_wikidata"]
        if "subject_nominatim" in relation:
            r_subject["nominatim"] = relation["subject_nominatim"]
        if "subject_nominatim_name" in relation:
            r_subject["nominatim_name"] = relation["subject_nominatim_name"]

        # object
        r_object = {
                "name": relation["object"],
                "type": relation["object_type"],
                "description": relation["object_description"],
            }
        if "object_qid" in relation:
            r_object["wikihum"] = relation["object_qid"]
        if "object_wikidata" in relation:
            r_object["wikidata"] = relation["object_wikidata"]
        if "object_nominatim" in relation:
            r_object["nominatim"] = relation["object_nominatim"]
        if "object_nominatim_name" in relation:
            r_object["nominatim_name"] = relation["object_nominatim_name"]

        # predicate
        r_predicate = {
            "name": relation["predicate"],
            "wikihum": relation["predicate_wikihum_p"]
        }

        # triplet
        item = {
            "subject": r_subject,
            "predicate": r_predicate,
            "object": r_object
        }

        data.append(item)

        if "subject_instance_of" in relation:
            if (relation["subject"], relation["subject_instance_of"]) not in subjects_instance_of:
                item = {
                    "subject": r_subject,
                    "predicate": {"name": "instance_of", "wikihum": P_INSTANCE_OF},
                    "object": {"wikihum": relation["subject_instance_of"]}
                }
                data.append(item)
                subjects_instance_of[(relation["subject"], relation["subject_instance_of"])] = relation["subject_instance_of"]

        if "object_instance_of" in relation:
            if (relation["object"], relation["object_instance_of"]) not in objects_instance_of:
                item = {
                    "subject": r_object,
                    "predicate": {"name": "instance_of", "wikihum": P_INSTANCE_OF},
                    "object": {"wikihum": relation["object_instance_of"]}
                }
                data.append(item)
                objects_instance_of[(relation["object"], relation["object_instance_of"])] = relation["object_instance_of"]


    output = {"triplets": data}

    output_path = Path('..') / "output" / "Adam_Dabrowski_KG.json"
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(output, f_out, indent=4, ensure_ascii=False)


    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
