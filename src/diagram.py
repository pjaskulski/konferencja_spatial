""" diagram klas """
from typing import List, Optional
from pydantic import BaseModel, Field
import erdantic as erd


# ------------------------------ CLASS -----------------------------------------
class ObjectModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania obiektów w tekście. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów.",
    )
    name: str = Field(
        description="nazwa obiektu np. Arnold Kowiński, Kraków, rzeka Wisła, potok, miasto, zbroja, pług. Ten sam obiekt występujący wielokrotnie w analizowanym tekście powinien mieć przypisaną zawsze tą samą nazwę."
    )
    type: str = Field(
        description="typ obiektu np."
    )
    description: Optional[str] = Field(
        description="Dodatkowy, opcjonalny opis znalezionego obiektu, pochodzący wyłącznie z analizowanego tekstu. Ten sam obiekt występujący wielokrotnie w analizowanym tekście powinien mieć przypisany zawsze ten sam opis."
    )


class RelationModel(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowania relacji między obiektami. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia obiektów i ich relacji.",
    )
    subject: ObjectModel = Field(
        description="podmiot relacji np. Bogdan ma brata: Adama Krzepkowskiego (podmiot: Bogdan)"
    )
    predicate: str = Field(
        description="nazwa relacji zachodzącej między podmiotem a obiektem np."
    )
    object: ObjectModel = Field(
        description="obiekt relacji np. Bogdan ma brata: Adama Krzepkowskiego (obiekt: Adam Krzepkowski)"
    )


class RelationCollection(BaseModel):
    relations: List[RelationModel] = Field(
        description="Lista relacji między obiektami występujacymi w tekście"
    )

# -------------------------------- MAIN ----------------------------------------
if __name__ == "__main__":
    # Easy one-liner
    erd.draw(RelationCollection, out="diagram.png")
