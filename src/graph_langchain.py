import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
llm_transformer = LLMGraphTransformer(llm=llm)

text = """
Szulc (Szulc-Prątnicki) h. Prątnicki w zakonie Tomasz (ok. 1642 — 1705), dominikanin, przeor, pisarz religijny.
Ur. prawdopodobnie w Lubawie (ziemia chełmińska), był bliskim krewnym Tomasza Jana Szulca-Prątnickiego (zm. 1710), prepozyta i kanonika chełmińskiego, archiprezbitera elbląskiego, proboszcza lubawskiego oraz oficjała i wizytatora diec. chełmińskiej.
Do zakonu dominikanów wstąpił S. w Gdańsku, przyjmując imię Tomasz; w tamtejszym konwencie p. wezw. św. Mikołaja złożył 24 X 1660 śluby zakonne. Ukończył studia filozoficzno-teologiczne w studium zakonnym i ok. r. 1672 uzyskał stopień lektora teologii. Po raz pierwszy został wybrany przeorem gdańskim 11 VIII 1676. Kapit. prowincjalna zebrana 30 IV 1678 w Krakowie zwróciła się do generała zakonu z wnioskiem o bakalaureat dla S-a, podkreślając, że spełnia on «praedicatoris officium in civitate haeretica» już od ośmiu lat. Decyzją generała zakonu z 20 VIII t.r. uzyskał S. stopień prezentata teologii, a w kwietniu 1681 był już bakałarzem. Dn. 10 IX 1682 obrano go przeorem klasztoru w Toruniu. Jesienią 1684 generał zakonu przyznał mu najwyższy w zakonie stopień naukowy magistra teologii. W l. 1683—5 był S. bakałarzem studium generalnego w Krakowie; następnie wrócił do Gdańska i 24 VII 1685 rada prowincji zatwierdziła jego ponowny wybór na przeora gdańskiego. T.r. uczestniczył w obradach kapit. prowincjalnej w klasztorze w Opatowcu, a jako pierwszy definitor w kwietniu 1687 w kapitule w Płocku; był też we wrześniu 1688 na kapitule w Lublinie. Wykładał w szkołach dominikańskich w Gdańsku i Toruniu, prawdopodobnie też we Wrocławiu; w czerwcu t.r. podczas dysputy we wrocławskim kościele św. Wojciecha student Kazimierz Jurgiewicz dedykował S-owi bronione właśnie tezy filozoficzno-teologiczne. Na radzie prowincji w maju 1690 w Warszawie S. został przedstawiony jako jeden z trzech kandydatów na stanowisko regensa krakowskiego studium generale; ostatecznie jednak urząd ten objął Tomasz Mdzewski. Po rezygnacji Rajmunda Kaweckiego, przeora-elekta klasztoru w Chełmnie, rada prowincji wybrała S-a 16 XII t.r. na tamtejszego przeora. W październiku 1691 uczestniczył S. w kapit. prowincji w Krakowie. Wybrany na początku r. 1692 na przeora klasztoru w Płocku, nie przyjął tego stanowiska ze względu na stan zdrowia, lecz już we wrześniu t.r. został ponownie przeorem w Gdańsku. Co najmniej w l. 1692—5 był wikariuszem kontraty pruskiej; od r. 1696 miał tytuł misjonarza apostolskiego. Od maja 1698 sprawował urząd przeora w Toruniu, ale już po kilku miesiącach zrezygnował na rzecz ponownego objęcia przeoratu w Gdańsku. W lipcu 1700 uczestniczył w obradach kapituły w Janowie. Wg źródeł zakonnych odznaczał się pobożnością oraz surowością życia i praktyk religijnych. Dn. 29 IX 1704 złożył wpis w księdze bractwa różańcowego, działającego przy gdańskim klasztorze dominikanów. Podczas pobytu w Gdańsku wydał drukiem trzy tomy, spisane w formie odpowiedzi na pytania, dotyczące tez św. Tomasza z Akwinu pt. Compendium unsyllogisticum totius Summae s. Thomae Aquinatis Doctoris Angelici, pro faciliori captu et memoria omnibus verae sapientiae amatoribus (Gedani 1694—5, wyd. 2, 1697). Zadedykował je kolejno: opatowi mogilskiemu i kanonikowi warmińskimu Piotrowi Rostkowskiemu, kanonikowi pułtuskiemu i kustoszowi kapit. warmińskiej Janowi Jerzemu Kunigkowi oraz ówczesnemu proboszczowi chełmińskiemu i lubawskiemu Tomaszowi SzulcowiPrątnickiemu. S. zmarł jesienią 1705; osiemnastowieczny historyk zakonny Wawrzyniec Teleżyński błędnie wymienił go wśród zmarłych w r. 1706, a dominikanin Franciszek Nowowiejski podał mylnie datę jego zgonu ok. r. 1696.
"""

documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)

#print(f"Nodes:{graph_documents[0].nodes}")
#print(f"Relationships:{graph_documents[0].relationships}")

for n in graph_documents[0].nodes:
    print(n)

print()

for r in graph_documents[0].relationships:
    print(r)

print(graph_documents[0].to_json())