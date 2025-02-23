""" knowledge extraction from text """
import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import instructor
from instructor import Mode
from instructor import llm_validator
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated
import openai
from openai import OpenAI


MODEL = "gpt-4o-mini"
#MODEL = "gpt-4o"

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


predicat_list_with_descripton = [
    {
        "name": "instanceOf",
        "description": "jest to, czym jest podmiot relacji np. osoba, miejscowość, rzeka, instytucja itp.",
    },
    {
        "name": "dateOfBirth",
        "description": "data urodzenia dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany pomiń je stosując krótki format YYYY-MM lub YYYY",
    },
    {
        "name": "dateOfDeath",
        "description": "data śmierci dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany pomiń je stosując krótki format YYYY-MM lub YYYY",
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
        "description": "data pochówku dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany wpisz 00",
    },
    {
        "name": "placeOfBurial",
        "description": "miejsce pochówku dla osób"
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
        "name": "hasChild",
        "description": "dziecko osoby będącej podmiotem relacji"
    },
    {
        "name": "significantFigure",
        "description": "znacząca osoba (współpracownik, przyjaciel) dla osoby będącej podmiotem relacji",
    },
    {
        "name": "relatedInstitution",
        "description": "instytucja związana z osobą będącą podmiotem relacji",
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
        "name": "officeHeld",
        "description": "urząd sprawowany przez osobę będącą podmiotem relacji",
    },
    {
        "name": "positionHeld",
        "description": "stanowisko (praca) zajmowane przez osoby będącej podmiotem relacji",
    },
    {
        "name": "placeOfStudy",
        "description": "miejsce nauki (miejscowość) osoby będącej podmiotem relacji",
    },
    {
        "name": "placeOfWork",
        "description": "miejsce pracy (miejscowość) osoby będącej podmiotem relacji",
    },
    {
        "name": "placeOfResidence",
        "description": "miejsce pobytu (miejscowość) osoby będącej podmiotem relacji",
    },
    {
        "name": "causeOfDeath",
        "description": "przyczyna śmierci osoby będącej podmiotem relacji",
    },
    {
        "name": "institutionWhereWork",
        "description": "miejsce pracy (instytucja, firma) osoby będącej podmiotem relacji",
    },
    {
        "name": "hasAssets",
        "description": "zasoby (nieruchomości, majątek) posiadane przez osobę będącą podmiotem relacji",
    },
    {
        "name": "isAuthorOf",
        "description": "dzieło, którego autorem jest osoba będąca podmiotem relacji",
    }
]

predicat_list = ""
for item in predicat_list_with_descripton:
    predicat_list += f'{item["name"]} - {item["description"]}\n'

type_list = ["osoba", "miejscowość", "kraj", "góra", "rzeka", "jezioro", "staw",
             "morze", "data", "instytucja", "stanowisko", "zawód", "jednostka wojskowa",
             "obiekt gospodarczy", "dzieło", "tytuł", "wydarzenie", "kwota", "inny obiekt"]

type_list_str = ", ".join(type_list)


# ------------------------------- MODELS ---------------------------------------
class ObjectModel(BaseModel):
    name: str = Field(description="nazwa obiektu np. Arnold Kowiński, Kraków, rzeka Wisła, potok, miasto, zbroja, pług")
    type: str = Field(description="typ obiektu")
    description: Optional[str] = Field(description="Dodatkowy (opcjonalny) opis znalezionego obiektu, pochodzący wyłącznie z analizownego tekstu.")

    @field_validator('type')
    @classmethod
    def type_only_from_list(cls, value:str) -> str:
        """ kontrola pola z typem obiektu """
        if value not in type_list:
            raise ValueError(f"'{value}' - value outside the expected list of types")

        return value

class RelationModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania obiektów i relacji między nimi. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów i relacji."
    )
    subject: ObjectModel = Field(description="podmiot relacji np. Bogdan jest bratem Adama Krzepkowskiego (podmiot: Bogdan).")
    predicate: str = Field(description="nazwa relacji zachodzącej między podmiotem a obiektem.")
    object: ObjectModel = Field(description="obiekt relacji np. Bogdan jest bratem Adama Krzepkowskiego (obiekt: Adam).")

class RelationCollection(BaseModel):
    relations: List[RelationModel] = Field(description="Lista relacji między obiektami występujacymi w tekście")

class ValidateTriple(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do oceny poprawności relacji. Te kroki powinny dostarczać szczegółowego uzasadnienia decyzji o poprawności lub nie."
    )
    result: bool = Field(description="Ocena poprawności analizowanej relacji Prawda/Fałsz (True/False)")


# ---------------------------- FUNCTIONS ---------------------------------------
def show_results(result: RelationCollection):
    """ wyświetlenie wyników """
    licznik = 0
    for rel in result.relations:
        licznik += 1
        print(f'{licznik}.')
        print('Przemyślenia:')
        for cot in rel.chain_of_thought:
            print(cot)
        print()
        print(f'subject  : {rel.subject}')
        print(f'predicate: {rel.predicate}')
        print(f'object   : {rel.object}')
        print('---')


def triple_validation(client, relation_to_test:RelationModel, biogram_to_test:str):
    relation_txt = f'<{relation_to_test.subject.name} -> {relation_to_test.predicate} -> {relation_to_test.object.name}>'

    print(relation_txt)

    sys_prompt = """
    Jesteś pomocnym asystentem historyka, badającym poprawność przedstawionych twierdzeń (relacji)
    <subject -> predicate -> object>
    na podstawie analizy przedstawionego tekstu, zwracasz wartość logiczną True lub False"
    """

    test_prompt = f""""
    Relacja do zweryfikowania:
    {relation_txt}
    na podstawie tekstu:
    {biogram_to_test}
    """

    result = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": test_prompt}
            ],
            response_model=ValidateTriple,
            temperature = 0,
            max_retries=3
        )

    return result.result


# ------------------------------ MAIN ------------------------------------------
if __name__ == "__main__":
    client = instructor.from_openai(OpenAI())
    #client = instructor.from_openai(OpenAI(), mode=Mode.TOOLS_STRICT)

    # wczytanie tekstu testowego biogramu
    data_file = Path('..') / "data" / "Dabrowski_Adam.txt"
    with open(data_file, 'r', encoding='utf-8') as f:
        biogram = f.read()

    system_prompt = f"""
    Jesteś asystentem historyka badającego biografie postaci historycznych.
    Przeanalizuj podany tekst i zidentyfikuj wszystkie obiekty występujące w tekście
    np. {type_list_str}.
    Analiza powinna dotyczyć wszystkich obiektów, nie tylko głównego bohatera biografii.
    Następnie dla każdego znalezionego obiektu ustal relacje z innymi obiektami np.
    {predicat_list}.
    Wszystkie dane powinny być oparte na faktach, możliwe do zweryfikowania na podstawie tekstu,
    a nie na zewnętrznych założeniach. Przeanalizuj tekst ponownie upewniając się,
    że z tekstu wyodrębnione zostały wszystkie znaczące obiekty i ich relacje.

    """

    user_prompt = f"""
    Tekst do analizy:
    {biogram}
    """

    # Extract structured data from natural language
    # do przetestowania:
    #  -- wielokrotne przetwarzanie tego samego tekstu i uwzględnienie unikalnych znalezisk
    #  -- przetwarzanie mniejszych części tekstu - podział na grupy zdań i przetwarzanie ich osobno

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=RelationCollection,
            temperature = 0,
            max_retries=3
        )
    except ValidationError as e:
        print(e)

    # walidacja poprawności znalezionych relacji przez LLM często oznacza
    # poprawne relacje jako fałszywe - do zbadania.

    # for item in res.relations:
    #     valid = triple_validation(client, item, biogram)
    #     if not valid:
    #         res.relations.remove(item)
    #         print(valid, item)
    #     else:
    #         print(valid)

    show_results(res)
