# Trabajo_Fin_Master

Resumen de experimentos NER en **Distemist** y **Proctemist**.

## Estructura

```text
Datasets/      -> datos crudos y textos (Distemist y Proctemist)
Encoders/      -> notebooks de experimentos con modelos encoder
Decoders/      -> notebooks de experimentos con modelos decoder
Scripts/       -> utilidades de conversión y preproceso
```

## Evaluación estricta

Los **F1 estrictos** reportados han sido verificados con las librerías oficiales:

- https://github.com/TeMU-BSC/distemist_evaluation_library
- https://github.com/TeMU-BSC/medprocner_evaluation_library

## Datasets (contenido)

En `Datasets/` se incluye, para cada tarea:

- `txt/` o `text_files/`: textos clínicos sobre los que se realiza la evaluación final.
- `*_tsv*_test*.tsv`: gold standard en TSV para comparación por offsets.
- `*_train.jsonl` y `*_test.jsonl`: datos adaptados para modelos encoder.

Script usado para la conversión:

- `Scripts/distemist_to_bio.py`: convierte los textos y anotaciones originales en TSV a JSONL, formato adecuado para entrenar modelos encoder.

Estructura de cada registro JSONL (encoder):

```json
{
	"text": "Texto clínico completo...",
	"tokens": ["token1", "token2", "..."],
	"ner_tags": [2, 2, 0, 1, 2]
}
```

Donde `ner_tags` sigue codificación BIO por etiqueta de tarea (p. ej. `B-`, `I-`, `O`).

## Resultados por modelo

Parámetros usados (comunes a todos los modelos): `ep=20`, `bs=16`, `lr=8.516e-5`, `drop=0.1`, `wd=0.1844`, `warmup=0.1`, `vote=0.5`.

| Tarea | Notebook | Modelo base | Parámetros | Folds x seeds (=ensemble) | Precision | Recall | F-score |
|---|---|---|---|---:|---:|---:|---:|
| Distemist | `bsc_79.47.ipynb` | `PlanTL-GOB-ES/bsc-bio-ehr-es` | ~125M | 5 x 2 = 10 | 0.8101 | 0.7798 | 0.7947 |
| Distemist | `lcampillos_79.70.ipynb` | `lcampillos/roberta-es-clinical-trials-ner` | ~125M | 5 x 2 = 10 | 0.8150 | 0.7798 | 0.7970 |
| Distemist | `iic-rigoberta-2-0_77.01.ipynb` | `IIC/RigoBERTa-2.0` | ~560M | 3 x 1 = 3 | 0.8066 | 0.7367 | 0.7701 |
| Distemist | `xlm-roberta_76.58.ipynb` | `FacebookAI/xlm-roberta-large` | ~560M | 3 x 1 = 3 | 0.8031 | 0.7317 | 0.7658 |
| Distemist | `iic-rigoberta-clinical_76.58.ipynb` | `IIC/RigoBERTa-Clinical` | ~560M | 3 x 1 = 3 | 0.8023 | 0.7325 | 0.7658 |
| Proctemist | `lcampillos-roberta-es-clinical-trials-ner_80.85.ipynb` | `lcampillos/roberta-es-clinical-trials-ner` | ~125M | 5 x 2 = 10 | 0.8191 | 0.7982 | 0.8085 |
| Proctemist | `bsc-bio-ehr-es_80.33.ipynb` | `PlanTL-GOB-ES/bsc-bio-ehr-es` | ~125M | 5 x 2 = 10 | 0.8176 | 0.7894 | 0.8033 |
| Proctemist | `iic-rigoberta-2-0_79.20.ipynb` | `IIC/RigoBERTa-2.0` | ~560M | 3 x 1 = 3 | 0.8127 | 0.7722 | 0.7920 |
| Proctemist | `iic-rigoberta-clinical_79.04.ipynb` | `IIC/RigoBERTa-Clinical` | ~560M | 3 x 1 = 3 | 0.8210 | 0.7620 | 0.7904 |
| Proctemist | `facebookai-xlm-roberta-large_78.11.ipynb` | `FacebookAI/xlm-roberta-large` | ~560M | 3 x 1 = 3 | 0.8233 | 0.7430 | 0.7811 |

## Referencia competiciones

Como referencia oficial:

- Ganador Distemist: **0,777** de F-score.
- Ganador MedProcNER: **0,798** de F-score.

Enlaces:

- https://temu.bsc.es/distemist/
- https://temu.bsc.es/medprocner/
