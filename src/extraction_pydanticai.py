""" knowledge extraction from text """
import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import openai


MODEL = "openai:gpt-4o-mini"
#MODEL = "gpt-4o"

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


predicat_list = """
dateOfBirth; dateOfDeath; placeOfBirth; placeOfDeath; dateOfBurial;
placeOfBurial; hasFather; hasMother; hasSon; hasDaughter; hasBrother;
hasSister; hasWife; hasHusband; hasUncle; hasAunt; hasCousin;
hasGrandfather; hasGrandmother; hasChild; significantFigure; relatedInstitution;
illegitimatePartner; hasGrandson; hasGranddaughter; hasOccupation;
officeHeld; positionHeld; placeOfStudy; placeOfWork; placeOfResidence;
causeOfDeath; institutionWhereWork; hasAssets; isAuthorOf
"""


class ObjectModel(BaseModel):
    name: str = Field(..., description="nazwa obiektu np. Arnold Kowiński, Kraków, rzeka Wisła, potok, miasto, zbroja, pług")
    type: str = Field(..., description="typ obiektu np. osoba, miejscowość, obiekt fizjograficzny, obiekt hydrologiczny, przedmiot, instytucja, urząd, zawód")
    description: Optional[str] = Field(..., description="Dodatkowy, opcjonalny opis znalezionego obiektu, pochodzący wyłącznie z analizownego tekstu.")

class RelationModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania obiektów i relacji między nimi. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów i relacji."
    )
    subject: ObjectModel = Field(..., description="podmiot relacji np. Bogdan jest bratem Adama Krzepkowskiego (podmiot: Bogdan)")
    predicate: str = Field(..., description=f"nazwa relacji zachodzącej między podmiotem a obiektem np. {predicat_list}")
    object: ObjectModel = Field(..., description="obiekt relacji np. Bogdan jest bratem Adama Krzepkowskiego (obiekt: Adam)")

class RelationCollection(BaseModel):
    relations: List[RelationModel] = Field(..., description="Lista relacji między obiektami występujacymi w tekście")


# wczytanie tekstu testowego biogramu
data_file = Path('..') / "data" / "Dabrowski_Adam.txt"
with open(data_file, 'r', encoding='utf-8') as f:
    biogram = f.read()

system_prompt = """
Jesteś asystentem historyka badającego biografie postaci historycznych.
Przeanalizuj podany tekst i zidentyfikuj obiekty występujące w tekście np. osoby, miejscowości,
rzeki, instytucje, zawody, urzędy. Następnie ustal relacje między obiektami np. 'jest to', 'ma brata',
'miejsce nauki', 'przyczyna śmierci' (dalej podano listę możliwych relacji).
Wszystkie dane powinny być oparte na faktach, możliwe do zweryfikowania na podstawie tekstu,
a nie na zewnętrznych założeniach. Przeanalizuj tekst ponownie upewniając się,
że wyodrębnione zostały wszystkie znaczące obiekty i ich relacje z tekstu.

"""

user_prompt = f"""
Tekst do analizy:
{biogram}
"""

agent = Agent(model=MODEL,
              result_type=RelationCollection,
              system_prompt=system_prompt)

# Extract structured data from natural language
# do przetestowania:
#  -- wielokrotne przetwarzanie tego samego tekstu i uwzględnienie unikalnych znalezisk
#  -- przetwarzanie mniejszych części tekstu - podział na grupy zdań i przetwarzanie ich osobno

result = agent.run_sync(user_prompt=user_prompt,
                        model_settings={'temperature': 0.0})

for rel in result.data.relations:
    print('Przemyślenia:')
    for cot in rel.chain_of_thought:
        print(cot)
    print()
    print(f'subject  : {rel.subject}')
    print(f'predicate: {rel.predicate}')
    print(f'object   : {rel.object}')
    print('---')

print(result.usage())
