# Opis 

Materiały związane z pracami dotyczącymi wydobywania z biogramów PSB danych dla bazy wiedzy WikiHum oraz z referatem na konferencję Spatial & Digital History network - ESSHC 2025 (Leiden, Netherlands), prezentującym te prace.

## Streszczenie referatu

While computer usage has become commonplace in the humanities, historical data often remains siloed within textual sources rather than residing in structured databases or knowledge bases. This unstructured format presents significant challenges for searching and analysis. Manual data transformation into databases or knowledge bases is a further hurdle due to its time-consuming nature. Our work aimed to bridge this gap by leveraging advancements in Artificial Intelligence (AI) and Natural Language Processing (NLP) for automated text processing. This pipeline facilitates information extraction, identification and linking of proper names (people and locations), and data transformation for seamless integration into a knowledge base (Wikibase). The processing pipeline, built using Python scripts, incorporates large language models, fuzzy string matching, and import methods based on the Wikibase API.  We applied this pipeline to biographies from the Polish Biographical Dictionary dataset (over 27,500 entries). An additional challenge stemmed from the occasional use of archaic language in these early 20th-century biographies.  Beyond efficiency gains through automation, a crucial aspect of our work involved evaluating the quality of the processed data.

## Pliki

 - data - katalog na dane, biogramy postaci historycznych w formacje txt
 - output_etap_1 - pliki json z wynikami pierwszego przetwarzania
 - output_etap_2 - pliki json z wynikami prztwarzania po poprawkach listy właściwości
 - output_identification_1 - pliki json po przeprowadzonej procedurze identyfikacji (pierwsze przetwarzanie)
 - output_identification_2 - pliki json po przeprowadzonej procedurze identyfikacji (po poprawkach)
 - output_pdf_1 - wizualizacja grafów - wyniki pierwszego przetwarzania
 - output_pdf_2 - wizualizacja grafów - po poprawkach
 - src - kod źródłówy skryptów
 - test_output - wyniki testów różnych modeli LLM, testów powtarzalności wyników
 - test_prompt - testowe prompty
 - test_validation - wyniki testów automatycznej walidacji "trójek" przez inny model

## Bibliografia

1. Xiaohan Feng, Xixin Wu and Helen Meng, "Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema"
https://human-interpretable-ai.github.io/assets/pdf/19_Ontology_grounded_Automatic.pdf

2. Khasa Gillani, Klemen Kenda, Erik Novak, Dunja Mladenić, "Knowledge graph Extraction from Textual data using LLM"
https://aile3.ijs.si/dunja/SiKDD2024/Papers/IS2024_-_SIKDD_2024_paper_15.pdf

3. Mohammad Sadeq Abolhasani, Rong Pan, "Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation"
https://arxiv.org/abs/2412.00608

4. Sanaz SAKI NOROUZI, Adrita BARUA, Antrea CHRISTOU, Nikita GAUTAM, Andrew EELLS, Pascal HITZLER, Cogan SHIMIZU, "Ontology Population using LLMs"
https://arxiv.org/abs/2411.01612

5. Roos M. Bakker, Daan L. Di Scala and Maaike H. T. de Boer, "Ontology Learning from Text: an Analysis on LLM Performance"
https://ceur-ws.org/Vol-3874/paper5.pdf

6. Roos M. Bakker, Daan L. Di Scala, "From Text to Knowledge Graph: Comparing Relation Extraction Methods in a Practical Context"
https://ceur-ws.org/Vol-3749/genesy-04.pdf

7. Yifan Ding, Amrit Poudel, Qingkai Zeng, Tim Weningera, Balaji Veeramani, Sanmitra Bhattacharya, "EntGPT: Entity Linking with Generative Large Language Models"
https://arxiv.org/pdf/2402.06738v2

8. Amy Xin, Yunjia Qi, Zijun Yao, Fangwei Zhu, Kaisheng Zeng, Bin Xu, Lei Hou, Juanzi Li, "LLMAEL: Large Language Models are Good Context Augmenters for Entity Linking"
https://arxiv.org/pdf/2407.04020

9. Andrea Papaluca, Daniel Krefl, Sergio J. Rodríguez Méndez, Artem Lensky, Hanna Suominen, "Zero- and Few-Shots Knowledge Graph Triplet Extraction with
Large Language Models"
https://aclanthology.org/2024.kallm-1.2.pdf

10. Cogan Shimizu, Andrew Eells, Seila Gonzalez, Lu Zhou, Pascal Hitzler, Alicia Sheill,
Catherine Foley, Dean Rehberger, "Ontology design facilitating Wikibase integration — and a worked example for historical data"
https://www.sciencedirect.com/science/article/pii/S157082682400009X?via%3Dihub

11. Zilin Xiao, Ming Gong, Jie Wu, Xingyao Zhang, Linjun Shou, Daxin Jiang, "Instructed Language Models with Retrievers Are Powerful Entity Linkers"
https://aclanthology.org/2023.emnlp-main.139.pdf

12. Vamsi Krishna Kommineni, Birgitta König-Ries, Sheeba Samuel, "From human experts to machines: An LLM supported approach to ontology and knowledge graph construction"
https://arxiv.org/abs/2403.08345

13. Hanieh Khorashadizadeh et al., "Research Trends for the Interplay between Large Language Models and Knowledge Graphs"
https://arxiv.org/abs/2406.08223

14. Camila Díaz, Jocelyn Dunstan, Lorena Etcheverry, Antonia Fonck, Alejandro Grez, Domingo Mery, Juan Reutter, and Hugo Rojas,
"Automatic knowledge-graph creation from historical documents: The Chilean dictatorship as a case study"
https://arxiv.org/abs/2408.11975

15. Cogan Shimizu ,Pascal Hitzler, "Accelerating Knowledge Graph and Ontology Engineering with Large Language Models"
https://arxiv.org/abs/2411.09601

16. Tania Litaina, Andreas Soularidis, Georgios Bouchouras, Konstantinos Kotis and Evangelia Kavakli "Towards LLM-based semantic analysis of historical legal documents"
https://ceur-ws.org/Vol-3724/short2.pdf

17. Daniel Garijo, María Poveda-Villalón, Elvira Amador-Domínguez, ZiYuan Wang, Raúl García-Castro and Oscar Corcho "LLMs for Ontology Engineering: A landscape of Tasks and Benchmarking challenges"
https://dgarijo.com/papers/iswc_llm.pdf

18. Cameron R. Wolfe, Using LLMs for Evaluation. LLM-as-a-Judge and other scalable additions to human quality ratings...
https://cameronrwolfe.substack.com/p/llm-as-a-judge

