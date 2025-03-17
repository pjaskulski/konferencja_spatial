""" skrypt pomocniczy - lista relacji """
import os
import time
import json
from pathlib import Path
import logging
import logging.config


def read_json(input_path):
    """ wczytanie danych z pliku json """
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
    data_folder = Path("..") / "output_deepseek"
    output_path = Path("..") / "output_deepseek" / "deepseek_Rossi_Piotr.txt"
    data_file_list = data_folder.glob('*.json')

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        file_name = os.path.basename(data_file)
        #if file_name != 'Rossi_Piotr.json':
        #    continue

        print(file_name + '\n')
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(file_name + '\n\n')

        relations = read_json(input_path=data_file)

        for relation in relations:
            log = relation["subject"]["name"] + ' -> ' + relation["predicate"]["name"] + ' -> ' + relation["object"]["name"]
            logging.info(log)
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(log + '\n')
