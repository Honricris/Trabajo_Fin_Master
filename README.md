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

---

## Resultados por modelo

### Encoders

Hiperparámetros usados (comunes a todos los modelos): `ep=20`, `bs=16`, `lr=8.516e-5`, `drop=0.1`, `wd=0.1844`, `warmup=0.1`, `vote=0.5`.

#### Distemist (encoders)
Los tiempos de inferencia se calcularon sobre los 250 archivos del conjunto de test. El tiempo total corresponde a la suma de todos los ensembles; el tiempo por modelo en el ensemble es `tiempo_total_del_ensemble / número_de_modelos_del_ensemble` y el tiempo medio por archivo es `tiempo_por_modelo_en_el_ensemble / 250`.

| Tarea | Notebook | Modelo base | Parámetros | Folds x seeds (=ensemble) | Precision | Recall | F-score | Tiempo medio/archivo (test) | Tiempo por modelo en el ensemble | Tiempo total del ensemble |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Distemist | `bsc-bio-ehr-es_7940.ipynb` | `PlanTL-GOB-ES/bsc-bio-ehr-es` | ~125M | 5 x 2 = 10 | 0.8088 | 0.7798 | 0.7940 | 0.21 s | 53.7 s | 537.4 s |
| Distemist | `lcampillos_7965.ipynb` | `lcampillos/roberta-es-clinical-trials-ner` | ~125M | 5 x 2 = 10 | 0.8159 | 0.7779 | 0.7965 | 0.22 s | 55.5 s | 554.9 s |
| Distemist | `rigoberta_7756.ipynb` | `IIC/RigoBERTa-2.0` | ~560M | 3 x 1 = 3 | 0.8058 | 0.7475 | 0.7756 | 0.38 s | 95.5 s | 286.4 s |
| Distemist | `xlm-roberta_7626.ipynb` | `FacebookAI/xlm-roberta-large` | ~560M | 3 x 1 = 3 | 0.8018 | 0.7271 | 0.7626 | 0.39 s | 96.8 s | 290.3 s |
| Distemist | `rigoberta-clinical_7650.ipynb` | `IIC/RigoBERTa-Clinical` | ~560M | 3 x 1 = 3 | 0.8037 | 0.7298 | 0.7650 | 0.41 s | 102.9 s | 308.8 s |

#### Proctemist (encoders)

| Tarea | Notebook | Modelo base | Parámetros | Folds x seeds (=ensemble) | Precision | Recall | F-score | Tiempo medio/archivo (test) | Tiempo por modelo en el ensemble | Tiempo total del ensemble |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proctemist | `lcampillos_8014.ipynb` | `lcampillos/roberta-es-clinical-trials-ner` | ~125M | 5 x 2 = 10 | 0.8026 | 0.8002 | 0.8014 | 0.22 s | 53.9 s | 538.7 s |
| Proctemist | `bsc-bio-ehr-es_8048.ipynb` | `PlanTL-GOB-ES/bsc-bio-ehr-es` | ~125M | 5 x 2 = 10 | 0.8157 | 0.7941 | 0.8048 | 0.21 s | 52.8 s | 527.6 s |
| Proctemist | `rigoberta_7896.ipynb` | `IIC/RigoBERTa-2.0` | ~560M | 3 x 1 = 3 | 0.8105 | 0.7698 | 0.7896 | 0.39 s | 97.1 s | 291.4 s |
| Proctemist | `rigoberta-clinical_7928.ipynb` | `IIC/RigoBERTa-Clinical` | ~560M | 3 x 1 = 3 | 0.8125 | 0.7739 | 0.7928 | 0.39 s | 97.8 s | 293.5 s |
| Proctemist | `xlm-roberta_7822.ipynb` | `FacebookAI/xlm-roberta-large` | ~560M | 3 x 1 = 3 | 0.8037 | 0.7617 | 0.7822 | 0.38 s | 95.7 s | 287.1 s |

---

### Decoders

#### Open-source (QLoRA / unsloth)

Todos los modelos open-source decoder se entrenaron con **QLoRA** (4-bit) usando la librería `unsloth`. Hiperparámetros comunes: `LoRA r=64`, `LoRA alpha=128`, `LoRA dropout=0`, `target_modules="all-linear"`, `epochs=2`, `batch_size=1`, `gradient_accumulation_steps=32` (batch efectivo=32), `learning_rate=3e-5`, `lr_scheduler="linear"`, `warmup_ratio=0.05`, `weight_decay=0.01`, `max_grad_norm=0.3`, `optimizer="paged_adamw_8bit"`, `MAX_SEQ_LENGTH=4096`, `MAX_NEW_TOKENS=300`, `seed=3407`. La inferencia se realiza sobre los mismos 250 archivos del conjunto de test. Los tiempos son directamente el tiempo medio por archivo (el paradigma generativo no usa ensemble).

Los **parámetros totales** corresponden al conteo del modelo base cargado en 4-bit (`sum(p.numel() for p in model.parameters())`). Los **parámetros entrenables** son los adaptadores LoRA añadidos sobre el modelo congelado.

Las columnas `F1 0-shot` y `F1 5-shot` corresponden al mismo modelo evaluado sin entrenamiento (mismo formato de prompt e inferencia, sin QLoRA), en zero-shot y con 5 ejemplos en el prompt respectivamente. `ΔF1` es la ganancia del ajuste fino con QLoRA respecto al mejor de los dos baselines (0-shot o 5-shot).

##### Distemist (decoders open-source)

| Tarea | Notebook | Modelo base | Parámetros totales | Parámetros entrenables | Precision | Recall | F-score | F1 0-shot | F1 5-shot | ΔF1 | Tiempo medio/archivo (test) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Distemist | `ministral-3-8b-instruct-sft_79_25_.ipynb` | `unsloth/Ministral-3-8B-Instruct-2512` | 9.13B | 215.6M (2.36%) | 0.8090 | 0.7766 | 0.7925 | 0.3934 | 0.4335 | +0.3590 | 22.63 s |
| Distemist | `BioMistral-7B-Dare_77_03_.ipynb` | `BioMistral/BioMistral-7B-DARE` | 7.41B | 167.8M (2.26%) | 0.7848 | 0.7564 | 0.7703 | 0.0970 | 0.1949 | +0.5754 | 13.99 s |
| Distemist | `Meta-Llama-3_1-8B-Instruct_76_18_.ipynb` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8.20B | 167.8M (2.05%) | 0.7835 | 0.7412 | 0.7618 | 0.2787 | 0.3620 | +0.3998 | 14.02 s |
| Distemist | `Qwen3-8B_75_98_.ipynb` | `Qwen/Qwen3-8B` | 8.37B | 174.6M (2.09%) | 0.7739 | 0.7461 | 0.7598 | 0.3013 | 0.2937 | +0.4661 | 18.96 s |
| Distemist | `mistral-7b-instruct_75_69_.ipynb` | `mistralai/Mistral-7B-Instruct-v0.3` | 7.42B | 167.8M (2.26%) | 0.7663 | 0.7477 | 0.7569 | 0.2911 | 0.2175 | +0.5394 | 17.30 s |
| Distemist | `salamandra-7b-instruct_71_76_.ipynb` | `BSC-LT/salamandra-7b-instruct` | 7.92B | 147.3M (1.86%) | 0.6940 | 0.7429 | 0.7176 | 0.2093 | 0.1948 | +0.5228 | 12.77 s |

##### Proctemist (decoders open-source)

La columna `F1 0-shot` corresponde al mismo modelo evaluado sin entrenamiento (mismo formato de prompt e inferencia, sin QLoRA). `ΔF1` es la ganancia del ajuste fino con QLoRA respecto al baseline zero-shot.

| Tarea | Notebook | Modelo base | Parámetros totales | Parámetros entrenables | Precision | Recall | F-score | F1 0-shot | ΔF1 | Tiempo medio/archivo (test) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proctemist | `Ministral-3-8B-Instruct-2512_79_05_.ipynb` | `unsloth/Ministral-3-8B-Instruct-2512` | 9.13B | 215.6M (2.36%) | 0.7839 | 0.7971 | 0.7905 | 0.4604 | +0.3301 | 48.23 s |
| Proctemist | `Meta-Llama-3_1-8B-Instruct_78_22_.ipynb` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8.20B | 167.8M (2.05%) | 0.7858 | 0.7786 | 0.7822 | 0.2139 | +0.5683 | 15.28 s |
| Proctemist | `BioMistral-7B-DARE_77_99_.ipynb` | `BioMistral/BioMistral-7B-DARE` | 7.41B | 167.8M (2.26%) | 0.7823 | 0.7775 | 0.7799 | 0.0481 | +0.7318 | 16.14 s |
| Proctemist | `Qwen3-8B_77_30_.ipynb` | `Qwen/Qwen3-8B` | 8.37B | 174.6M (2.09%) | 0.7760 | 0.7700 | 0.7730 | 0.4145 | +0.3585 | 19.36 s |
| Proctemist | `Mistral-7b-instruct-v0_3_77_17_.ipynb` | `mistralai/Mistral-7B-Instruct-v0.3` | 7.42B | 167.8M (2.26%) | 0.7514 | 0.7930 | 0.7717 | 0.3091 | +0.4626 | 17.68 s |
| Proctemist | `salamandra-7b-instruct_75_47_.ipynb` | `BSC-LT/salamandra-7b-instruct` | 7.92B | 147.3M (1.86%) | 0.7440 | 0.7656 | 0.7547 | 0.1425 | +0.6122 | 14.33 s |

---

#### GPT-4.1-mini (OpenAI Fine-tuning API)

Modelo propietario evaluado en **Distemist** en tres escenarios: zero-shot, 5-shot y ajuste fino supervisado vía la API de fine-tuning de OpenAI (sin LoRA; el ajuste fino lo gestiona íntegramente la plataforma de OpenAI). Modelo base: `gpt-4.1-mini-2025-04-14`, `epochs=2`, sin validation split (100 % de los datos de entrenamiento usados para train). Modelo fine-tuneado resultante: `ft:gpt-4.1-mini-2025-04-14:personal::DpOhbhTB`.

Los parámetros totales y entrenables no se reportan al tratarse de un modelo de acceso cerrado vía API.

| Tarea | Notebook | Modelo base | Precision | Recall | F-score | F1 0-shot | F1 5-shot | ΔF1 (vs 5-shot) | Tiempo medio/archivo (test) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Distemist | `GPT-4_1-mini_79_45_.ipynb` | `gpt-4.1-mini-2025-04-14` | 0.8015 | 0.7876 | 0.7945 | 0.4803 | 0.5177 | +0.2768 | 43.13 s |

---

## Referencia competiciones
Como referencia oficial:
- Ganador Distemist: **0.777** de F-score.
- Ganador Proctemist: **0.798** de F-score.

Enlaces:
- https://temu.bsc.es/distemist/
- https://temu.bsc.es/medprocner/
