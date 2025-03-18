""" entity linking """
import os
import time
import json
from pathlib import Path
from typing import List
import logging
import logging.config
from dotenv import load_dotenv
from wikibaseintegrator import WikibaseIntegrator, wbi_login
from wikibaseintegrator.wbi_config import config as wbi_config
from wikibaseintegrator import wbi_helpers
import openai
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim
from ratelimit import limits
from groq import Groq


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
MODEL_LLAMA3 = "llama-3.3-70b-versatile"
MODEL_DEEPSEEK_CHAT = "deepseek-chat" # DeepSeek V3
MODEL_GPT4O_MINI = "gpt-4o-mini"
MODEL = MODEL_GPT4O_MINI

P_INSTANCE_OF = "P27"
P_INSTANCE_OF_WIKIDATA = "P31"
Q_HUMAN_WIKIDATA = "Q5"

INSTANCE_OF = False

# api key (OpenAI)
env_path_openai = Path(".") / ".env"
load_dotenv(dotenv_path=env_path_openai)
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


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

# client OpenAI
client_openai = instructor.from_openai(OpenAI())

# client Groq
client_g = Groq(api_key=GROQ_API_KEY)
client_groq = instructor.from_groq(client_g)

# client DeepSeek
client_deep = instructor.from_openai(
    OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
    mode=instructor.Mode.MD_JSON,
    max_tokens=8000
)


# ------------------------------- CLASSES --------------------------------------
class BestCandidateModel(BaseModel):
    """ struktura odpowiedzi modelu językowego  - wybór najbardziej pasującego wyniku """
    chain_of_thought: List[str] = Field(
        None,
        description="Kroki wyjaśniające prowadzące do ustalenia najlepszego wyboru z listy wyników. Te kroki powinny dostarczać szczegółowego uzasadnienia dlaczego wybrany wynik najbardzej pasuje do wskazanej nazwy i opisu."
    )
    best_fit: str = Field(description="Identyfikator (id) najbardziej pasującego wyniku, jeżeli pasującego wyniku brak - pusty string")


# ------------------------------ FUNCTIONS -------------------------------------
def zamien_kolejnosc(tekst):
    """Zamienia kolejność wyrazów """
    if not isinstance(tekst, str):
      return None

    tekst = tekst.strip()
    slowa = tekst.split()
    if len(slowa) == 2:
        return f"{slowa[1]} {slowa[0]}"
    else:
        return tekst


def search_wikihum(nazwa:str, typ_obiektu:str, opis_obiektu:str) -> list:
    """ wyszukiwanie w wikihum """
    try:
        # Ograniczenie do 10 wyników i języka polskiego
        text_to_search = nazwa

        wyniki = wbi_helpers.search_entities(search_string=text_to_search,
                                            mediawiki_api_url="https://wikihum.lab.dariah.pl/api.php",
                                            max_results=10,
                                            language='pl',
                                            dict_result=True,
                                            strict_language=True)

        if not wyniki:
            text_to_search = zamien_kolejnosc(text_to_search)
            if text_to_search != nazwa: # zamiana skuteczna
                wyniki = wbi_helpers.search_entities(search_string=text_to_search,
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
        logging.error("Wystąpił błąd podczas wyszukiwania: %s", e)
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
        "Q21672098",  # village of Ukraine
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

                # weryfikacja czy typ wyniku (właściwość instance of z wikidata)
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
        logging.error("Wystąpił błąd podczas wyszukiwania: %s", e)
        return None


# by nie przekroczyć limitów serwera nominatim
@limits(calls=20, period=60)
def search_nominatim(nazwa:str) -> tuple:
    """ wyszukiwanie miejscowości w serwisie Nominatim (OpenStreetMap) """

    geolocator = Nominatim(user_agent="WikiHum KG creator")
    location = geolocator.geocode(query=nazwa, namedetails=True, extratags=True)

    if location:
        # wyniki tylko z odpowiednio dużą wagą
        importance = location.raw.get('importance', 0)
        if float(importance) >= 0.4:
            nom_id = location.raw.get('place_id', None)
            extra = location.raw.get('extratags', {})
            nom_wikidata = extra.get('wikidata', None)
            nom_name = location.raw.get('display_name', None)
            return nom_id, nom_wikidata, nom_name

    return None, None, None


def select_item(candidate_list:list, expected:str) -> str:
    """ wybór z listy kandydatów """
    if len(candidate_list) == 1:
        return candidate_list[0]["id"]

    if MODEL == MODEL_LLAMA3:
        client = client_groq
    elif MODEL == MODEL_DEEPSEEK_CHAT:
        client = client_deep
    else:
        client = client_openai

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


def save_json_kg(relations:list, output_file:str):
    """ zapis wyników w pliku json """
    data = []

    subjects_instance_of = {} # lista unikalnych subjektów
    objects_instance_of = {} # lista unikalnych obiektów

    for relation in relations:
        # subject
        r_subject = {
                "name": relation["subject"]["name"],
                "type": relation["subject"]["type"],
                "description": relation["subject"]["description"]
            }
        if "subject_wikihum" in relation:
            r_subject["wikihum"] = relation["subject_wikihum"]
        if "subject_wikidata" in relation:
            r_subject["wikidata"] = relation["subject_wikidata"]
        if "subject_nominatim" in relation:
            r_subject["nominatim"] = relation["subject_nominatim"]
        if "subject_nominatim_name" in relation:
            r_subject["nominatim_name"] = relation["subject_nominatim_name"]

        # object
        r_object = {
                "name": relation["object"]["name"],
                "type": relation["object"]["type"],
                "description": relation["object"]["description"],
            }
        if "object_wikihum" in relation:
            r_object["wikihum"] = relation["object_wikihum"]
        if "object_wikidata" in relation:
            r_object["wikidata"] = relation["object_wikidata"]
        if "object_nominatim" in relation:
            r_object["nominatim"] = relation["object_nominatim"]
        if "object_nominatim_name" in relation:
            r_object["nominatim_name"] = relation["object_nominatim_name"]

        # predicate
        if relation["predicate_wikihum_p"].startswith('['):
            relation["predicate_wikihum_p"] = "NEW"
        r_predicate = {
            "name": relation["predicate"]["name"],
            "wikihum": relation["predicate_wikihum_p"]
        }

        # triplet
        item = {
            "subject": r_subject,
            "predicate": r_predicate,
            "object": r_object
        }

        data.append(item)

        if INSTANCE_OF:
            if "subject_instance_of" in relation:
                if (relation["subject"]["name"], relation["subject_instance_of"]) not in subjects_instance_of:
                    if relation["subject_instance_of"] == "Q5":
                        wikihum_qid = relation["subject_instance_of"]
                        wikihum_name = "człowiek"
                    elif relation["subject_instance_of"] == "Q175698":
                        wikihum_qid = relation["subject_instance_of"]
                        wikihum_name = "jednostka osadnicza"
                    else:
                        tmp = str(relation["subject_instance_of"])
                        if '[' in tmp:
                            tmp = tmp.replace('[','').replace(']','')
                        if tmp.startswith('Q'):
                            wikihum_qid = relation["subject_instance_of"]
                            wikihum_name = "?"
                        else:
                            wikihum_qid = "NEW"
                            wikihum_name = tmp

                    item = {
                        "subject": r_subject,
                        "predicate": {"name": "instanceOf", "wikihum": P_INSTANCE_OF},
                        "object": {"name": wikihum_name, "wikihum": wikihum_qid }
                    }
                    data.append(item)
                    subjects_instance_of[(relation["subject"]["name"], relation["subject_instance_of"])] = relation["subject_instance_of"]

            if "object_instance_of" in relation:
                if (relation["object"]["name"], relation["object_instance_of"]) not in objects_instance_of:
                    if relation["object_instance_of"] == "Q5":
                        wikihum_qid = relation["object_instance_of"]
                        wikihum_name = "człowiek"
                    elif relation["object_instance_of"] == "Q175698":
                        wikihum_qid = relation["object_instance_of"]
                        wikihum_name = "jednostka osadnicza"
                    else:
                        tmp = str(relation["object_instance_of"])
                        if '[' in tmp:
                            tmp = tmp.replace('[','').replace(']','')
                        if tmp.startswith('Q'):
                            wikihum_qid = relation["subject_instance_of"]
                            wikihum_name = "?"
                        else:
                            wikihum_qid = "NEW"
                            wikihum_name = tmp

                    item = {
                        "subject": r_object,
                        "predicate": {"name": "instanceOf", "wikihum": P_INSTANCE_OF},
                        "object": {"name": wikihum_name, "wikihum": wikihum_qid}
                    }
                    data.append(item)
                    objects_instance_of[(relation["object"]["name"], relation["object_instance_of"])] = relation["object_instance_of"]

    output = {"triplets": data}

    output_path = Path('..') / "output_identification_2" / output_file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(output, f_out, indent=4, ensure_ascii=False)


def read_json(input_file:str):
    """ wczytanie danych z pliku json """
    input_path = Path('..') / "output_etap_2" / input_file
    with open(input_path, 'r', encoding='utf-8') as f_in:
        json_data = json.load(f_in)

    return json_data['triplets']


def update_relations(relation:dict, d_subject:tuple, d_object:tuple):
    """ aktualizacja danych w słowniku relation """
    subject_qid, subject_wikidata, subject_nominatim, subject_nominatim_name = d_subject
    object_qid, object_wikidata, object_nominatim, object_nominatim_name = d_object

    if subject_qid:
        relation["subject_wikihum"] = subject_qid
    if subject_wikidata:
        relation["subject_wikidata"] = subject_wikidata
    if subject_nominatim:
        relation["subject_nominatim"] = subject_nominatim
    if subject_nominatim_name:
        relation["subject_nominatim_name"] = subject_nominatim_name

    if object_qid:
        relation["object_wikihum"] = object_qid
    if object_wikidata:
        relation["object_wikidata"] = object_wikidata
    if object_nominatim:
        relation["object_nominatim"] = object_nominatim
    if object_nominatim_name:
        relation["object_nominatim_name"] = object_nominatim_name


def process_element(element:str,
                    element_type:str,
                    element_description:str,
                    znane:dict) -> tuple:
    """ identyfikacja osób i miejscowości w subject """
    element_qid = element_wikidata = element_nominatim = element_nominatim_name = None

    if (element_type in ["osoba", "miejscowość"]
            and element not in znane
            and not element.strip()[0].islower()):
        # wyszukiwanie w wikihum
        candidates = search_wikihum(element, element_type, element_description)
        if candidates:
            dane = f'NAZWA: {element} ({element_type})\nOPIS: {element_description} '
            identified_item = select_item(candidates, dane)
            if identified_item and identified_item.strip() != "":
                element_qid = identified_item
                znane[(element, element_type)] = identified_item
    elif (element, element_type) in znane:
        element_qid = znane[(element, element_type)]
    else:
        # wyszukiwanie w wikidata
        candidates = search_wikidata(element, element_type)
        if candidates:
            dane = f'NAZWA: {element} ({element_type})\nOPIS: {element_description} '
            identified_item = select_item(candidates, dane)
            if identified_item and identified_item.strip() != "":
                element_wikidata = identified_item

    # subject: wyszukiwanie w Nominatim (tylko miejscowości)
    if (not element_qid and not element_wikidata
            and element_type == "miejscowość"):
        n_id, n_wikidata, n_name = search_nominatim(element)
        if n_id:
            element_nominatim = n_id
            if n_wikidata:
                element_wikidata = n_wikidata
            if n_name:
                element_nominatim_name = n_name

    if not element_qid and element_type != "data":
        element_qid = "NEW"

    return element_qid, element_wikidata, element_nominatim, element_nominatim_name


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":

    # pomiar czasu wykonania
    start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-4s %(message)s',
        datefmt='%m-%d %H:%M:%S')
    logging.RootLogger.manager.getLogger('httpx').disabled = True

    # wczytanie wyników pracy modelu LLM (triplety do grafu wiedzy)
    logging.info('Wczytanie danych...')

    # dataset pliki json z tripletami znalezionymi przez model językowy
    # po walidacji automatycznej przez model walidujący
    data_folder = Path("..") / "output_etap_2"
    data_file_list = data_folder.glob('*.json')

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)

        # pomijanie biogramów, które już mają zidentyfikowane dane
        link_path = Path('..') / "output_identification_2" / file_name
        if os.path.exists(link_path):
            continue

        relations = read_json(input_file=file_name)

        znane_subject = {}
        znane_object = {}

        logging.info("Identyfikacja osób i miejscowości...")

        for relation in relations:
            logging.info(relation["subject"]["name"] + ' -> ' + relation["predicate"]["name"] + ' -> ' + relation["object"]["name"])

            # property
            predicate = relation["predicate"]["name"]
            if predicate in wikihum_properties:
                relation["predicate_wikihum_p"] = wikihum_properties[predicate]
            else:
                relation["predicate_wikihum_p"] = f'[{predicate}]'

            if INSTANCE_OF:
                # instance of dla subject
                subject_type = relation["subject"]["type"]
                if subject_type in wikihum_elements:
                    relation["subject_instance_of"] = wikihum_elements[subject_type]
                else:
                    relation["subject_instance_of"] = f'[{subject_type}]'

                # instance of dla object
                object_type = relation["object"]["type"]
                if object_type in wikihum_elements:
                    relation["object_instance_of"] = wikihum_elements[object_type]
                else:
                    if object_type != 'data':
                        relation["object_instance_of"] = f'[{object_type}]'

            # próba identyfikacji osób i miejsc w subject
            data_subject = process_element(element=relation["subject"]["name"],
                                                    element_type=relation["subject"]["type"],
                                                    element_description=relation["subject"]["description"],
                                                    znane=znane_subject)

            # próba identyfikacji osób i miejsc w object
            data_object = process_element(element=relation["object"]["name"],
                                                    element_type=relation["object"]["type"],
                                                    element_description=relation["object"]["description"],
                                                    znane=znane_object)

            update_relations(relation=relation, d_subject=data_subject, d_object=data_object)

        # zapis wyników w pliku json
        logging.info("Zapis wyników w pliku json...")
        save_json_kg(relations=relations, output_file=file_name)

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Czas wykonania programu: %s s.",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
