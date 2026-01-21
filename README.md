Projekt zaliczeniowy – Zastosowania Uczenia Maszynowego
# 1. Informacje ogólne

Nazwa projektu:

`System NLP oceniający jakości odpowiedzi tekstowych w kontekście nauczania programowania`


Autor:
s34596

`Data Science, II stopień, semestr III`


Data oddania projektu:
21.01.2026

# 2. Opis projektu

Projekt buduje system NLP do oceny jakości odpowiedzi tekstowych w kontekście nauczania programowania.
Wykorzystuje dataset Mohler'a, który jest pierwszym publicznie dostepnym i szeroko używanym benchmarkiem pod Automatic Short Answer Grading(ASAG).
Model dostaje pytanie programistyczne oraz odpowiedź studenta i przewiduje jakość odpowiedzi (low/mid/high).

Projekt ma charakter aplikacyjny i badawczy. Może znaleźć zastosowanie m.in. w systemach e-learningowych, automatycznej ocenie odpowiedzi tekstowych czy analizie jakości wyjaśnień generowanych przez LLM.

# 3. Dane

Źródło danych:
HuggingFace

Link do danych:
https://huggingface.co/datasets/nkazi/MohlerASAG

Opis danych: `MohlerASAG, jest pierwszym publicznie dostepnym i szeroko używanym benchmarkiem pod Automatic Short Answer Grading(ASAG)`

liczba próbek: `~2300`

liczba cech / kolumn: `9 + id`

format danych: `.csv` i `.parquet` (po preprocessing)

rodzaj etykiet / klas: 
- `low` - odpowiedzi słabe jakościowo
- `mid` - odpowiedzi przeciętne
- `high` - odpowiedzi wysokiej jakości

licencja: CC-BY 4.0




Uwagi:

- Algorytm `data.py` uwzględnia obsługę wypadków, NaN, czybrakujących wartości, jednak wykorzystany dataset był pięknie przygotowany.
- Przy standardowym ustalaniu kwantyli wyniki  jakości "high". Prawdopodobnie spowodowane było to dyskretnym charakterem ocen jakości dla odpowiedzi, gdzie nie została osiągnięta wartość v > q2. Zamiast kwantyli przyjąłem binning oparty na semantycznej interpretacji po wartościach 'score', co zapewniło stworzenie trzech klas jakości.
- Pełny zbiór danych nie jest przechowywany w repo. Dane są ładowane i przetwarzane lokalnmie, a w repozytoruium znajduje się jedynie struktura katalogów i próbki.


# 4. Cel projektu

- zbudowanie klasyfikatora jakości odpowiedzi tekstowych,
- porównanie trzech podejść modelowych:
1. klasycznego ML (TF-IDF + regresja logistyczna)
2. suecu neuronowej (MLP na cechach TF-IDF)
3. transformerze HF (DistilBERT)
- ocena skutecznośći modeli z użyciem metryk dostosowanych do nierównych klas

    Sprecyzuj cel biznesowy lub badawczy projektu:

Celem projektu jest zbudowanie i porównanie modeli uczenia maszynowego do automatycznej oceny jakości odpowiedzi programistycznych na podstawie treści pytania oraz odpowiedzi ucznia. 
Projekt koncentruje się na analizie wartości dydaktycznej odpowiedzi, a nie wyłącznie na ich poprawności formalnej.

    Co ma robić model?

Model ma klasyfikować odpowiedzi tekstowe do ejdnej z trzech klas jakościowych:

- `low` - odpowiedzi słabe jakościowo
- `mid` - odpowiedzi przeciętne
- `high` - odpowiedzi wysokiej jakości

    Jakie pytanie ma odpowiadać lub jaką klasyfikację przeprowadzać?

Czy i w jakim stopniu różne podejścia modelowe (klasyczny ML, NN, Transformer) są w stanie skuytecznie ocenić jakość odpwoiedzi programistycznuych na podstawie tekstu oraz jakie są ograniczenia poszczególnyuch reprezentacji językowych.

    Jakie decyzje lub wnioski można z projektu wyciągnąć?
    
    Na podstawie wyników projektu można:

- ocenić przydatność klasycznych metod NLP (TF-IDF) w zadaniach edukacyjnych,

- określić, czy modele neuronowe i transformerowe poprawiają rozpoznawanie jakości odpowiedzi,

- wskazać, które klasy odpowiedzi są najtrudniejsze do rozróżnienia,

- wyciągnąć wnioski dotyczące dalszego rozwoju systemów automatycznej oceny nauczania.

Szczególną uwagę poświęciłem **macro-F1**, jako kluczowej metryce jakości.


# 5. Struktura projektu

Projekt składa się z czterech głównych etapów, każdy w osobnym i samodzielnym notatniku .ipynb:

`1_EDA.ipynb` 	- Analiza danych, wizualizacje, wnioski

`2_Preprocessing_Features.ipynb` 	- Czyszczenie danych, preprocessing, inżynieria cech

`3_Models_Training.ipynb` 	- Trening modeli klasycznego ML, sieci neuronowej i transformera

`4_Evaluation.ipynb` 	- Ewaluacja, porównanie modeli, wizualizacje wyników

# 6. Modele
### 6.1 Model klasyczny ML

**Algorytm:**  
Regresja logistyczna

**Krótki opis działania:**  
Model wykorzystuje reprezentację TF-IDF (unigramy i bigramy) do zamiany tekstu na wektory cech, a następnie klasyfikuje odpowiedzi do jednej z trzech klas jakościowych. 
Zastosowałem ważenie klas w celu kompensacji nierównomiernego rozkładu (przewaga odpowiedzi klasy `high`).

**Wyniki / metryki (walidacja):**
- Accuracy: ~0.68  
- Macro-F1: ~0.55  
- Weighted-F1: ~0.70  

---

### 6.2 Sieć neuronowa zbudowana od zera

**Architektura:**  
Wielowarstwowy perceptron (MLP)

**Liczba warstw / neuronów:**  
- warstwa wejściowa: wektor TF-IDF  
- 1 warstwa ukryta (256 neuronów)  
- warstwa wyjściowa (3 klasy)

**Funkcje aktywacji i optymalizator:**  
- funkcja aktywacji: ReLU  
- optymalizator: Adam  
- funkcja straty: ważona entropia krzyżowa  
- zastosowano dropout i early stopping

**Wyniki (walidacja):**
- Accuracy: ~0.72  
- Macro-F1: ~0.56  
- Weighted-F1: ~0.72  

---

### 6.3 Model transformerowy (fine-tuning)

**Nazwa modelu:**  
DistilBERT (`distilbert-base-uncased`)

**Zastosowana biblioteka:**  
HuggingFace Transformers + PyTorch

**Zakres dostosowania:**  
- fine-tuning całego modelu (end-to-end),  
- długość sekwencji `max_length = 128` (dobrana empirycznie),  
- trening na GPU z użyciem mixed precision.

**Wyniki (walidacja):**
- Accuracy: ~0.76  
- Macro-F1: ~0.54  
- Weighted-F1: ~0.74  

---

# 7. Ewaluacja

Użyte metryki:

- accuracy
- precision
- recall
- F1-score (macro-F1 jako metryka główna)

### Porównanie modeli:
| Model | Metryka główna | Wynik | Uwagi |
|------|----------------|-------|-------|
| Klasyczny ML | Macro-F1 | ~0.55 | stabilny baseline |
| Sieć neuronowa | Macro-F1 | ~0.56 | najlepszy wynik macro |
| Transformer | Macro-F1 | ~0.54 | najwyższa skuteczność globalna |

### Wizualizacje:
- macierz pomyłek,
- krzywa ROC,
- krzywa uczenia (learning curve).
Wizualizacje:

    Macierz pomyłek
    Krzywa ROC
    Learning curve

# 8. Wnioski i podsumowanie

- Najlepsze wyniki globalne (accuracy, weighted-F1) uzyskał model transformerowy.
- Najwyższy wynik macro-F1 osiągnęła sieć neuronowa MLP.
- Największą trudnością okazała się klasyfikacja klasy pośredniej (`mid`), charakteryzującej się niejednoznacznością semantyczną.
- Ograniczeniem projektu jest niewielka liczność danych oraz miękka definicja klas jakościowych.
- Projekt może zostać rozszerzony o większe zbiory danych, precyzyjniejsze etykiety oraz modele dedykowane systemom edukacyjnym.

---

# 9. Struktura repozytorium

    ZUM/
    │
    ├── data/
    │ ├── raw/
    │ ├── processed/
    │ └── sample/
    │
    ├── notebooks/
    │ ├── 1_EDA.ipynb
    │ ├── 2_Preprocessing_Features.ipynb
    │ ├── 3_Models_Training.ipynb
    │ └── 4_Evaluation.ipynb
    │
    ├── models/
    ├── results/
    │
    ├──src
      ├── data.py
    │
    ├── README.md
    ├── requirements.txt
    └── .gitignore

# 10. Technologia i biblioteki

    Python 3.13.1
    NumPy, Pandas, Matplotlib, Plotly, Seaborn
    scikit-learn
    PyTorch
    HF Transformers
    pathlib, collections, typing

Zależności projektu zostały opisane w pliku `requirements.txt`.  
W przypadku pracy z GPU wymagane jest dodatkowo środowisko CUDA zgodne z wersją PyTorch.
    

# 11. Licencja projektu

Projekt udostępniony na licencji:
MIT License

Źródło danych: zgodnie z licencją wskazaną w sekcji Dane.
