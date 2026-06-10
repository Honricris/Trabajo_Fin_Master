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

### Encoders

Parámetros usados (comunes a todos los modelos): `ep=20`, `bs=16`, `lr=8.516e-5`, `drop=0.1`, `wd=0.1844`, `warmup=0.1`, `vote=0.5`.

#### Distemist (encoders)
Los tiempos de inferencia se calcularon sobre los 250 archivos del conjunto de test. El tiempo total corresponde a la suma de todos los ensembles; el tiempo por modelo en el ensemble es `tiempo_total_del_ensemble / número_de_modelos_del_ensemble` y el tiempo medio por archivo es `tiempo_por_modelo_en_el_ensemble / 250`.

| Tarea | Notebook | Modelo base | Parámetros | Folds x seeds (=ensemble) | Precision | Recall | F-score | Tiempo medio/archivo (test) | Tiempo por modelo en el ensemble | Tiempo total del ensemble |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Distemist | `bsc-bio-ehr-es_7940.ipynb` | `PlanTL-GOB-ES/bsc-bio-ehr-es` | ~125M | 5 x 2 = 10 | 0.8088 | 0.7798 | 0.7940 | 0.21 s | 53.7 s | 537.4 s |
| Distemist | `lcampillos_7965.ipynb` | `lcampillos/roberta-es-clinical-trials-ner` | ~125M | 5 x 2 = 10 | 0.8159 | 0.7779 | 0.7965 | 0.22 s | 55.5 s | 554.9 s |
| Distemist | `rigoberta_7756.ipynb` | `IIC/RigoBERTa-2.0` | ~560M | 3 x 1 = 3 | 0.8058 | 0.7475 | 0.7756 | 0.38 s | 95.5 s | 286.4 s |
| Distemist | `xlm-roberta_7626.ipynb` | `FacebookAI/xlm-roberta-large` | ~560M | 3 x 1 = 3 | 0.8018 | 0.7271 | 0.7626 | 0.39 s | 96.8 s | 290.3 s |
| Distemist | `rigoberta-clinical_7650.ipynb` | `IIC/RigoBERTa-Clinical` | ~560M | 3 x 1 = 3 | 0.8037 | 0.7298 | 0.7650 | 0.41 s | 102.9 s | 308.8 s |

#### Proctemist (encoders)
| Tarea | Notebook | Modelo base | Parámetros | Folds x seeds (=ensemble) | Precision | Recall | F-score | Tiempo medio/archivo (test) | Tiempo por modelo en el ensemble | Tiempo total del ensemble |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proctemist | `lcampillos_8014.ipynb` | `lcampillos/roberta-es-clinical-trials-ner` | ~125M | 5 x 2 = 10 | 0.8026 | 0.8002 | 0.8014 | 0.22 s | 53.9 s | 538.7 s |
| Proctemist | `bsc-bio-ehr-es_8048.ipynb` | `PlanTL-GOB-ES/bsc-bio-ehr-es` | ~125M | 5 x 2 = 10 | 0.8157 | 0.7941 | 0.8048 | 0.21 s | 52.8 s | 527.6 s |
| Proctemist | `rigoberta_7896.ipynb` | `IIC/RigoBERTa-2.0` | ~560M | 3 x 1 = 3 | 0.8105 | 0.7698 | 0.7896 | 0.39 s | 97.1 s | 291.4 s |
| Proctemist | `rigoberta-clinical_7928.ipynb` | `IIC/RigoBERTa-Clinical` | ~560M | 3 x 1 = 3 | 0.8125 | 0.7739 | 0.7928 | 0.39 s | 97.8 s | 293.5 s |
| Proctemist | `xlm-roberta_7822.ipynb` | `FacebookAI/xlm-roberta-large` | ~560M | 3 x 1 = 3 | 0.8037 | 0.7617 | 0.7822 | 0.38 s | 95.7 s | 287.1 s |

---

### Decoders

Todos los modelos decoder se entrenaron con **QLoRA** (4-bit, LoRA r=64, alpha=128) usando la librería `unsloth`. Parámetros comunes: `ep=2`, `MAX_SEQ_LENGTH=4096`, `MAX_NEW_TOKENS=300`. La inferencia se realiza sobre los mismos 250 archivos del conjunto de test. Los tiempos son directamente el tiempo medio por archivo (el paradigma generativo no usa ensemble).

Los **parámetros totales** corresponden al conteo del modelo base cargado en 4-bit (`sum(p.numel() for p in model.parameters())`). Los **parámetros entrenables** son los adaptadores LoRA añadidos sobre el modelo congelado.

#### Distemist (decoders)

| Tarea | Notebook | Modelo base | Parámetros totales | Parámetros entrenables | Precision | Recall | F-score | Tiempo medio/archivo (test) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Distemist | `ministral-3-8b-instruct-sft_79_25_.ipynb` | `unsloth/Ministral-3-8B-Instruct-2512` | 5,403,199,488 | 215,613,440 (3.84%) | 0.8090 | 0.7766 | 0.7925 | 22.63 s |
| Distemist | `BioMistral-7B-Dare_77_03_.ipynb` | `BioMistral/BioMistral-7B-DARE` | 3,752,071,168 | 167,772,160 (4.28%) | 0.7848 | 0.7564 | 0.7703 | 13.99 s |
| Distemist | `Meta-Llama-3_1-8B-Instruct_76_18_.ipynb` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 4,540,600,320 | 167,772,160 (3.56%) | 0.7835 | 0.7412 | 0.7618 | 14.02 s |
| Distemist | `Qwen3-8B_75_98_.ipynb` | `Qwen/Qwen3-8B` | 5,191,808,000 | 174,587,904 (3.25%) | 0.7739 | 0.7461 | 0.7598 | 18.96 s |
| Distemist | `mistral-7b-instruct_75_69_.ipynb` | `mistralai/Mistral-7B-Instruct-v0.3` | 3,758,362,624 | 167,772,160 (4.27%) | 0.7663 | 0.7477 | 0.7569 | 17.30 s |
| Distemist | `salamandra-7b-instruct_71_76_.ipynb` | `BSC-LT/salamandra-7b-instruct` | 4,932,767,744 | 147,324,928 (2.90%) | 0.6940 | 0.7429 | 0.7176 | 12.77 s |

#### Proctemist (decoders)

| Tarea | Notebook | Modelo base | Parámetros totales | Parámetros entrenables | Precision | Recall | F-score | Tiempo medio/archivo (test) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Proctemist | `Ministral-3-8B-Instruct-2512_79_05_.ipynb` | `unsloth/Ministral-3-8B-Instruct-2512` | 5,403,199,488 | 215,613,440 (3.84%) | 0.7839 | 0.7971 | 0.7905 | 48.23 s |
| Proctemist | `Meta-Llama-3_1-8B-Instruct_78_22_.ipynb` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 4,540,600,320 | 167,772,160 (3.56%) | 0.7858 | 0.7786 | 0.7822 | 15.28 s |
| Proctemist | `BioMistral-7B-DARE_77_99_.ipynb` | `BioMistral/BioMistral-7B-DARE` | 3,752,071,168 | 167,772,160 (4.28%) | 0.7823 | 0.7775 | 0.7799 | 16.14 s |
| Proctemist | `Qwen3-8B_77_30_.ipynb` | `Qwen/Qwen3-8B` | 5,191,808,000 | 174,587,904 (3.25%) | 0.7760 | 0.7700 | 0.7730 | 19.36 s |
| Proctemist | `Mistral-7b-instruct-v0_3_77_17_.ipynb` | `mistralai/Mistral-7B-Instruct-v0.3` | 3,758,362,624 | 167,772,160 (4.27%) | 0.7514 | 0.7930 | 0.7717 | 17.68 s |
| Proctemist | `salamandra-7b-instruct_75_47_.ipynb` | `BSC-LT/salamandra-7b-instruct` | 4,932,767,744 | 147,324,928 (2.90%) | 0.7440 | 0.7656 | 0.7547 | 14.33 s |

## Referencia competiciones
Como referencia oficial:
- Ganador Distemist: **0,777** de F-score.
- Ganador Proctemist: **0,798** de F-score.

Enlaces:
- https://temu.bsc.es/distemist/
- https://temu.bsc.es/medprocner/
