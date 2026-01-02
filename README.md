# fx-analyzer

Narzędzie do analizy trendów na rynkach walutowych Forex.

## Struktura projektu
- `src/` - katalog źródłowy z modułami programu
- `src/ingestion.py` - pobranie i przygotowanie danych
- `src/features.py` - dodanie cech, normalizacja danych
- `src/model.py` - przygotowanie tensorów, parametrów modelu, trening, ewaluacja
- `src/strategy.py` - obliczenia wskaźników, symulacja strategii, obliczenia transakcji, wizualizacja
- `src/utils.py` - funkcje pomocnicze
- `main.py` - główny punkt wejścia aplikacji

## Użycie
Główny skrypt uruchamiający pełną ścieżkę analizy:
```bash
python main.py
```

## Zastrzeżenie
Oprogramowanie służy wyłącznie do celów edukacyjnych i badawczych. Treści generowane przez program oraz kod źródłowy nie stanowią porady inwestycyjnej ani rekomendacji zakupu lub sprzedaży jakichkolwiek instrumentów finansowych. Handel na rynku Forex wiąże się ze znacznym ryzykiem utraty kapitału. Autor nie ponosi żadnej odpowiedzialności za decyzje inwestycyjne oraz ewentualne straty finansowe poniesione w wyniku korzystania z tego narzędzia.
