""" validations """
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
from groq import Groq


MODEL_LLAMA3 = "llama-3.3-70b-versatile"
MODEL_DEEPSEEK_CHAT = "deepseek-chat" # DeepSeek V3
MODEL = MODEL_LLAMA3

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


# lista predykatów (właściwości)
with open('predicate.json', 'r', encoding='utf-8') as f:
    predicat_list_with_descripton = json.load(f)

predykaty = {}
for item in predicat_list_with_descripton:
    predykaty[item["name"]] = item["description"]


# -------------------------------- CLASS ---------------------------------------
class ValidationModel(BaseModel):
    is_true: bool = Field(description="Czy badane stwierdzenie jest prawdziwe czy fałszywe?")


# ------------------------------ FUNCTIONS -------------------------------------
def validate(llm_client, data_structure, llm_model, biogram_text="") -> dict:
    """ walidacja relacji """

    data = data_structure["triplets"]

    for item in data:
        if "reasoning" in item:
            uzasadnienie = item["reasoning"]
        else:
            uzasadnienie = biogram_text

        if "description" in item["object"] and item["object"]["description"]:
            uzasadnienie = item["object"]["description"] + ' ' + uzasadnienie
        if "description" in item["subject"] and item["subject"]["description"]:
            uzasadnienie = item["subject"]["description"] + ' ' + uzasadnienie
        log = item["subject"]["name"] + ' -> ' + item["predicate"]["name"] + ' -> ' + item["object"]["name"]
        if item["predicate"]["name"] in predykaty:
            log = log + '\n(definicja ' + item["predicate"]["name"] + ': ' + predykaty[item["predicate"]["name"]]
        result = validate_if_true(llm_client=llm_client,
                                  stwierdzenie=log,
                                  uzasadnienie=uzasadnienie,
                                  llm_model=llm_model)
        item["result"] = result.is_true

    return {"triplets": data}


def validate_if_true(llm_client, stwierdzenie:str, uzasadnienie:str, llm_model):
    """ analiza biogramu """
    system_prompt = f"""
    Jesteś asystentem historyka badającego biografie postaci historycznych.
    Na podstawie podanych niżej informacji oceń prawdziwość stwierdzenia:
    {stwierdzenie}
    """

    user_prompt = f"""
    Informacje:
    {uzasadnienie}
    """

    rezultat = None

    rezultat = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=ValidationModel,
        temperature=0,
        seed=2,
    )

    return rezultat


def read_json(input_path):
    """ wczytanie danych z pliku json """
    with open(input_path, 'r', encoding='utf-8') as f_in:
        json_data = json.load(f_in)

    return json_data


def save_json(out_data:dict, filename:str):
    """ zapis danych w pliku json """
    output_file = filename.replace('.txt', '.json')
    output_path = Path('..') / "output_validation_2" / output_file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(out_data, f_out, indent=4, ensure_ascii=False)


# -------------------------------- MAIN ----------------------------------------
if __name__ == '__main__':
    # pomiar czasu wykonania
    start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-4s %(message)s',
        datefmt='%m-%d %H:%M:%S')
    logging.RootLogger.manager.getLogger('httpx').disabled = True

    logging.info("Model: %s", MODEL)

    client_g = Groq(api_key=GROQ_API_KEY)
    client_groq = instructor.from_groq(client_g)

    client_deep = instructor.from_openai(
        OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
        mode=instructor.Mode.MD_JSON,
        max_tokens=8000
    )

    data_folder = Path("..") / "output_etap_2"
    data_file_list = data_folder.glob('*.json')

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)
        logging.info(file_name)

        txt_name = file_name.replace('.json', '.txt')
        biogram_path = Path("..") / "data" / txt_name
        with open(biogram_path, 'r', encoding='utf-8') as f:
            biogram = f.read()

        relations = read_json(input_path=data_file)

        # validation LLaMA3
        struktura = validate(llm_client=client_groq,
                             data_structure=relations,
                             llm_model=MODEL_LLAMA3,
                             biogram_text=biogram)
        # validation DeepSeek
        # struktura = validate(llm_client=client_deep, data_structure=struktura, llm_model=MODEL_DEEPSEEK_CHAT)

        logging.info("Zapis wyników w formacie json...")
        save_json(out_data=struktura, filename=file_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Czas wykonania programu: %s s.",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
