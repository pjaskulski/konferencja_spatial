""" knowledge extraction from text """

import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
import openai
from openai import OpenAI
from graphviz import Digraph


MODEL = "gpt-4o-mini"
# MODEL = "gpt-4o"

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
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

predicat_list_with_descripton = [
    {
        "name": "instanceOf",
        "description": "jest to, czym jest podmiot relacji np. osoba, miejscowość, rzeka, instytucja itp.",
    },
    {
        "name": "dateOfBirth",
        "description": "data urodzenia dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany wpisz 00",
    },
    {
        "name": "dateOfDeath",
        "description": "data śmierci dla osób, w formacie YYYY-MM-DD, jeżeli dzień lub miesiąc jest nieznany wpisz 00",
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


class ObjectModel(BaseModel):
    name: str = Field(
        description="nazwa obiektu np. Arnold Kowiński, Kraków, rzeka Wisła, potok, miasto, zbroja, pług"
    )
    type: str = Field(
        description="typ obiektu np. osoba, miejscowość, obiekt fizjograficzny, obiekt hydrologiczny, przedmiot, instytucja, urząd, zawód"
    )
    description: Optional[str] = Field(
        description="Dodatkowy, opcjonalny opis znalezionego obiektu, pochodzący wyłącznie z analizownego tekstu."
    )


class RelationModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania obiektów i relacji między nimi. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów i relacji.",
    )
    subject: ObjectModel = Field(
        description="podmiot relacji np. Bogdan jest bratem Adama Krzepkowskiego (podmiot: Bogdan)"
    )
    predicate: str = Field(
        description=f"nazwa relacji zachodzącej między podmiotem a obiektem np. {predicat_list}"
    )
    object: ObjectModel = Field(
        description="obiekt relacji np. Bogdan jest bratem Adama Krzepkowskiego (obiekt: Adam)"
    )


class RelationCollection(BaseModel):
    relations: List[RelationModel] = Field(
        description="Lista relacji między obiektami występujacymi w tekście"
    )


def visualize_knowledge_graph(kg: RelationCollection):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.relations:
        dot.node(name=str(node.subject.name), label=node.subject.name, color="black")

    # Add edges
    for edge in kg.relations:
        dot.edge(
            tail_name=str(edge.subject.name),
            head_name=str(edge.object.name),
            label=edge.predicate,
            color="green",
        )

    # Render the graph
    dot_u = dot.unflatten(stagger=4)
    dot_u.render("knowledge_graph.gv", view=True)


client = instructor.from_openai(OpenAI())

# wczytanie tekstu testowego biogramu
data_file = Path("..") / "data" / "Dabrowski_Adam.txt"
with open(data_file, "r", encoding="utf-8") as f:
    biogram = f.read()

system_prompt = """
Jesteś asystentem historyka badającego biografie postaci historycznych.
Przeanalizuj podany tekst i zidentyfikuj obiekty występujące w tekście np. osoby, miejscowości,
rzeki, instytucje, zawody, urzędy. Następnie ustal relacje między tymi obiektami np. 'jest to', 'ma brata',
'miejsce nauki', 'przyczyna śmierci' (dalej podano listę możliwych relacji).
Wszystkie dane powinny być oparte na faktach, możliwe do zweryfikowania na podstawie tekstu,
a nie na zewnętrznych założeniach. Przeanalizuj tekst ponownie upewniając się,
że wyodrębnione zostały wszystkie znaczące obiekty i ich relacje z tekstu.

"""

user_prompt = f"""
Tekst do analizy:
{biogram}
"""

# Extract structured data from natural language
# do przetestowania:
#  -- wielokrotne przetwarzanie tego samego tekstu i uwzględnienie unikalnych znalezisk
#  -- przetwarzanie mniejszych części tekstu - podział na grupy zdań i przetwarzanie ich osobno
res = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_model=RelationCollection,
    temperature=0,
    seed=42,
)

for rel in res.relations:
    print("Przemyślenia:")
    for cot in rel.chain_of_thought:
        print(cot)
    print()
    print(f"subject  : {rel.subject}")
    print(f"predicate: {rel.predicate}")
    print(f"object   : {rel.object}")
    print("---")

visualize_knowledge_graph(res)
