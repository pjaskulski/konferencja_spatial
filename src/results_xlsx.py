""" entity linking """
import os
import time
import json
from pathlib import Path
from openpyxl import Workbook
import logging
import logging.config


# --------------------------- FUNCTIONS ----------------------------------------
def read_json(input_file:str):
    """ wczytanie danych z pliku json """
    input_path = Path('..') / "output_dataset_link" / input_file
    with open(input_path, 'r', encoding='utf-8') as f_in:
        json_data = json.load(f_in)

    return json_data['triplets']


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
    data_folder = Path("..") / "output_dataset_link"
    data_file_list = data_folder.glob('*.json')

    xlsx_data = [["Biogram","Subject","Subject description", "QID", "Wikidata/Nominatim",
                  "Predicate [QID]",
                  "Object", "Object description", "QID", "Wikidata/Nominatim"]]

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)

        relations = read_json(input_file=file_name)

        for relation in relations:
            logging.info(relation["subject"]["name"] + ' -> ' + relation["predicate"]["name"] + ' -> ' + relation["object"]["name"])

            # property
            subject_name = relation["subject"]["name"]
            subject_qid = relation.get("subject",{}).get("wikihum", "")
            subject_description = relation.get("subject", {}).get("description", "")
            subject_wikidata = ""
            if subject_qid == "NEW":
                subject_wikidata = relation.get("subject",{}).get("wikidata", "")
            subject_nominatim = ""
            if "type" in relation["subject"] and relation["subject"]["type"] == "miejscowość":
                subject_nominatim = relation.get("subject",{}).get("nominatim", "")

            if relation["predicate"]["name"] == 'instance_of':
                predicate_name = "instanceOf"
            else:
                predicate_name = relation["predicate"]["name"]

            predicate = f'{predicate_name} [{relation["predicate"]["wikihum"]}]'
            object_name = relation["object"]["name"]
            object_qid = relation.get("object", {}).get("wikihum", "")
            object_description = relation.get("object", {}).get("description", "")
            object_wikidata = ""
            if object_qid == "NEW":
                object_wikidata = relation.get("object",{}).get("wikidata", "")
            object_nominatim = ""
            if "type" in relation["object"] and relation["object"]["type"] == "miejscowość":
                object_nominatim = relation.get("object",{}).get("nominatim", "")

            if subject_nominatim:
                subject_wikidata += " / " + str(subject_nominatim)
            if object_nominatim:
                object_wikidata += " / " + str(object_nominatim)

            xlsx_data.append([file_name.replace('.json','.txt'), subject_name, subject_description, subject_qid, subject_wikidata,
                              predicate,
                              object_name, object_description, object_qid, object_wikidata])

    # zapis danych w pliku xlsx
    output_file = Path('..') / 'output_xlsx' / 'psb_triplets.xlsx'

    wb = Workbook()
    ws = wb.active
    ws.title = "PSB_seria_100"

    for item in xlsx_data:
        ws.append(item)

    wb.save(output_file)

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Czas wykonania programu: %s s.",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
