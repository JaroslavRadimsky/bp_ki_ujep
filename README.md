# Automatizované hodnocení RTG snímků krční páteře

Tento repozitář slouží jako doprovodný materiál k bakalářské práci zaměřené na
automatizované hodnocení RTG snímků krční páteře pomocí metod hlubokého učení.

Repozitář obsahuje zdrojové kódy, grafické výstupy, výsledky experimentů a text
práce.

## Struktura repozitáře

```text
.
├── code/        # Zdrojové kódy pro trénování, inferenci a vyhodnocení
├── images/      # Grafy, obrázky a vizualizace výsledků
└── README.md    # Popis repozitáře
```

## Obsah

Ve složce `code/` jsou uloženy skripty a pomocné moduly použité při implementaci
obou porovnávaných přístupů. Patří sem zejména části pro přípravu dat, trénování
modelů, inferenci, postprocessing, výpočet metrik a analýzu výsledků.

Složka `images/` obsahuje obrázky a grafické výstupy používané pro prezentaci
výsledků práce. Jedná se například o grafy chyb, porovnání metod, vizualizace
predikovaných bodů nebo ukázky výstupů segmentačního a detekčního přístupu.
