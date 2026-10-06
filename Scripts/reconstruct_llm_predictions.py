#!/usr/bin/env python3
"""Rebuild and validate the final TSVs from checkpoints and notebook logs.

Run with: python Scripts/reconstruct_llm_predictions.py
"""

import ast
import csv
import json
import re
from pathlib import Path


JOBS = (
    {
        "task": "Distemist",
        "dataset": "distemist",
        "checkpoint": "Decoders/Distemist/SFT/checkpoints/inference_checkpointLLamaDistemist.json",
        "notebook": "Decoders/Distemist/SFT/Meta-Llama-3.1-8B-Instruct(76.18).ipynb",
        "label": "ENFERMEDAD",
        "columns": ("filename", "mark", "label", "off0", "off1", "span"),
        "start": "off0",
        "end": "off1",
        "surface": "span",
        "gold": "distemist_subtrack1_test_mentions.tsv",
        "output": "Decoders/Distemist/SFT/checkpoints/meta-llama-3.1-8b-instruct.tsv",
    },
    {
        "task": "Proctemist",
        "dataset": "proctemist",
        "checkpoint": "Decoders/Proctemist/SFT/checkpoints/inference_checkpointLLamaProctemist.json",
        "notebook": "Decoders/Proctemist/SFT/Meta-Llama-3.1-8B-Instruct(78.22).ipynb",
        "label": "PROCEDIMIENTO",
        "columns": ("filename", "ann_id", "label", "start_span", "end_span", "text"),
        "start": "start_span",
        "end": "end_span",
        "surface": "text",
        "gold": "medprocner_tsv_test_subtask1.tsv",
        "output": "Decoders/Proctemist/SFT/checkpoints/meta-llama-3.1-8b-instruct.tsv",
    },
    {
        "task": "Proctemist",
        "dataset": "proctemist",
        "checkpoint": "Decoders/Proctemist/SFT/checkpoints/inference_checkpointBioMistralProctemist.json",
        "notebook": "Decoders/Proctemist/SFT/BioMistral-7B-DARE(77.99).ipynb",
        "label": "PROCEDIMIENTO",
        "columns": ("filename", "ann_id", "label", "start_span", "end_span", "text"),
        "start": "start_span",
        "end": "end_span",
        "surface": "text",
        "gold": "medprocner_tsv_test_subtask1.tsv",
        "output": "Decoders/Proctemist/SFT/checkpoints/biomistral-7b-dare.tsv",
    },
)

CHUNK = re.compile(r"^\s*\[chunk (\d+)/(\d+) \| chars (\d+)-(\d+)\]$")
RAW = re.compile(r"^\s*RAW: (.*)$")
SKIP = re.compile(r"^\[(\d+)/(\d+)\] (.+?) — ya procesado, saltando$")
DONE = re.compile(r"^\[(\d+)/(\d+)\] (.+?) \| [\d.]+s \| (\d+) chunks \| (\d+) menciones$")


def stream_text(cell):
    return "".join(
        "".join(output.get("text", []))
        if isinstance(output.get("text"), list)
        else output.get("text", "")
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def notebook_cells(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    cells = notebook.get("cells", [])
    inference = next(
        (cell for cell in cells if "CHECKPOINT_INPUT" in "".join(cell.get("source", []))),
        None,
    )
    evaluation = next(
        (cell for cell in cells if "def prf" in "".join(cell.get("source", []))),
        None,
    )
    if inference is None or evaluation is None:
        raise ValueError(f"No encuentro las celdas de inferencia y evaluación en {notebook_path.name}")
    metric_match = re.search(
        r"ESTRICTA.*?F1=([\d.]+)\s+\(TP=(\d+) FP=(\d+) FN=(\d+)\)",
        stream_text(evaluation),
        re.S,
    )
    if metric_match is None:
        raise ValueError(f"La salida de evaluación no contiene F1 estricto en {notebook_path.name}")
    metrics = (int(metric_match[2]), int(metric_match[3]), int(metric_match[4]), float(metric_match[1]))
    return stream_text(inference), metrics


def parse_inference_log(log, checkpoint_done, test_ids, notebook_name):
    resumed = re.search(r"Retomando desde checkpoint .*?: (\d+) archivos ya procesados", log)
    if resumed is None or int(resumed[1]) != len(checkpoint_done):
        raise ValueError(f"Checkpoint y log no coinciden en {notebook_name}")

    pending = []
    current_chunk = None
    skipped = []
    recovered = {}
    status_indexes = []

    for line in log.splitlines():
        match = CHUNK.match(line)
        if match:
            if current_chunk is not None:
                raise ValueError(f"Falta RAW antes del siguiente chunk en {notebook_name}")
            current_chunk = tuple(map(int, match.groups()))
            continue

        match = RAW.match(line)
        if match:
            if current_chunk is None:
                raise ValueError(f"RAW sin chunk asociado en {notebook_name}")
            result = ast.literal_eval(match[1])
            if not isinstance(result, str):
                raise ValueError(f"RAW no es texto en {notebook_name}")
            pending.append((*current_chunk, result))
            current_chunk = None
            continue

        match = SKIP.match(line)
        if match:
            index, total = int(match[1]), int(match[2])
            filename = match[3].removesuffix(".txt")
            skipped.append(filename)
            status_indexes.append((index, total, filename))
            continue

        match = DONE.match(line)
        if match:
            index, total = int(match[1]), int(match[2])
            filename = match[3].removesuffix(".txt")
            chunks, mentions = int(match[4]), int(match[5])
            if current_chunk is not None or len(pending) != chunks:
                raise ValueError(f"N.º de chunks incompleto para {filename} en {notebook_name}")
            if [item[0] for item in pending] != list(range(1, chunks + 1)) or any(item[1] != chunks for item in pending):
                raise ValueError(f"Secuencia de chunks inválida para {filename} en {notebook_name}")
            if filename in recovered:
                raise ValueError(f"Documento repetido en el log: {filename}")
            recovered[filename] = (pending, mentions)
            pending = []
            status_indexes.append((index, total, filename))

    if current_chunk is not None or pending:
        raise ValueError(f"Log de inferencia truncado en {notebook_name}")
    if set(skipped) != checkpoint_done or len(skipped) != len(checkpoint_done):
        raise ValueError(f"Documentos omitidos no coinciden con done_files en {notebook_name}")
    if set(recovered) & checkpoint_done:
        raise ValueError(f"El log repite documentos del checkpoint en {notebook_name}")
    if checkpoint_done | set(recovered) != test_ids:
        raise ValueError(f"La reconstrucción no cubre exactamente los textos de test en {notebook_name}")
    if {name for _, _, name in status_indexes} != test_ids or len(status_indexes) != len(test_ids):
        raise ValueError(f"La ejecución no registra exactamente todos los textos de test en {notebook_name}")
    if {total for _, total, _ in status_indexes} != {len(test_ids)} or {i for i, _, _ in status_indexes} != set(range(1, len(test_ids) + 1)):
        raise ValueError(f"Índices de ejecución incompletos en {notebook_name}")
    return recovered


def rows_from_chunks(filename, chunks, text, job):
    mentions = []
    for _, _, start, end, result in chunks:
        if start > end or end > len(text):
            raise ValueError(f"Offsets de chunk fuera del texto: {filename}")
        cursors = {}
        for mention in (line.strip() for line in result.splitlines() if line.strip()):
            pattern = re.escape(mention)
            cursor = cursors.get(mention, start)
            match = re.search(pattern, text[cursor:end], re.IGNORECASE)
            if match:
                offset_start, offset_end = cursor + match.start(), cursor + match.end()
            else:
                match = re.search(pattern, text[start:end], re.IGNORECASE)
                if not match:
                    continue
                offset_start, offset_end = start + match.start(), start + match.end()
            cursors[mention] = offset_end
            mentions.append((text[offset_start:offset_end], offset_start, offset_end))

    unique = []
    seen = set()
    for surface, start, end in sorted(mentions, key=lambda item: item[1]):
        if (start, end) not in seen:
            unique.append((surface, start, end))
            seen.add((start, end))
    id_column = "mark" if job["task"] == "Distemist" else "ann_id"
    start_column, end_column, surface_column = job["start"], job["end"], job["surface"]
    return [
        {
            "filename": filename,
            id_column: f"T{index}",
            "label": job["label"],
            start_column: start,
            end_column: end,
            surface_column: surface,
        }
        for index, (surface, start, end) in enumerate(unique, 1)
    ]


def validate_rows(rows, texts, job):
    start_column, end_column, surface_column = job["start"], job["end"], job["surface"]
    id_column = "mark" if job["task"] == "Distemist" else "ann_id"
    seen = set()
    by_file = {}
    for row in rows:
        filename = row["filename"]
        if filename not in texts or row["label"] != job["label"]:
            raise ValueError(f"Dataset o etiqueta incorrectos en {job['task']}: {filename}")
        start, end = int(row[start_column]), int(row[end_column])
        key = (filename, start, end)
        if key in seen:
            raise ValueError(f"Predicción duplicada: {key}")
        seen.add(key)
        text = texts[filename]
        if start < 0 or end <= start or end > len(text) or text[start:end] != row[surface_column]:
            raise ValueError(f"Offset/texto inconsistente en {filename}: {start}:{end}")
        by_file.setdefault(filename, []).append(row)

    for filename, file_rows in by_file.items():
        file_rows.sort(key=lambda row: int(row[start_column]))
        if [row[id_column] for row in file_rows] != [f"T{i}" for i in range(1, len(file_rows) + 1)]:
            raise ValueError(f"IDs de anotación no secuenciales en {filename}")


def strict_metrics(rows, gold_path, job):
    predicted_files = {row["filename"] for row in rows}
    start_column, end_column = job["start"], job["end"]
    predicted = {
        (row["filename"], row["label"], int(row[start_column]), int(row[end_column]))
        for row in rows
    }
    with gold_path.open(encoding="utf-8-sig", newline="") as file:
        gold = {
            (row["filename"], row["label"], int(row[start_column]), int(row[end_column]))
            for row in csv.DictReader(file, delimiter="\t")
            if row["filename"] in predicted_files
        }
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, round(f1, 4)


def reconstruct(job, repo_root):
    checkpoint = json.loads((repo_root / job["checkpoint"]).read_text(encoding="utf-8-sig"))
    done = checkpoint.get("done_files")
    pred_rows = checkpoint.get("pred_rows")
    if not isinstance(done, list) or not isinstance(pred_rows, list) or len(set(done)) != len(done):
        raise ValueError(f"Checkpoint inválido: {job['checkpoint']}")
    if len(checkpoint.get("inference_times", [])) != len(done):
        raise ValueError(f"inference_times y done_files no coinciden: {job['checkpoint']}")

    text_dir = repo_root / "Datasets" / job["dataset"] / "text_files_test"
    text_paths = sorted(text_dir.glob("*.txt"))
    if not text_paths:
        raise ValueError(f"No encuentro textos de test en {text_dir}")
    test_ids = {path.stem for path in text_paths}
    checkpoint_done = set(done)
    if not checkpoint_done <= test_ids:
        raise ValueError(f"Checkpoint contiene IDs ajenos al test: {job['checkpoint']}")

    log, notebook_metrics = notebook_cells(repo_root / job["notebook"])
    recovered = parse_inference_log(log, checkpoint_done, test_ids, job["notebook"])
    rows = []
    for row in pred_rows:
        if row.get("filename") not in checkpoint_done or not set(job["columns"]) <= row.keys():
            raise ValueError(f"Fila inválida en {job['checkpoint']}")
        normalized = {column: row[column] for column in job["columns"]}
        normalized[job["start"]] = int(normalized[job["start"]])
        normalized[job["end"]] = int(normalized[job["end"]])
        rows.append(normalized)

    texts = {path.stem: path.read_text(encoding="utf-8") for path in text_paths}
    for filename, (chunks, expected_mentions) in recovered.items():
        file_rows = rows_from_chunks(
            filename,
            chunks,
            texts[filename],
            job,
        )
        if len(file_rows) != expected_mentions:
            raise ValueError(f"RAW y total de menciones no coinciden en {filename}")
        rows.extend(file_rows)

    order = {path.stem: index for index, path in enumerate(text_paths)}
    rows.sort(key=lambda row: (order[row["filename"]], int(row[job["start"]])))
    validate_rows(rows, texts, job)

    gold_path = repo_root / "Datasets" / job["dataset"] / job["gold"]
    actual_metrics = strict_metrics(rows, gold_path, job)
    if actual_metrics[:3] != notebook_metrics[:3] or actual_metrics[3] != notebook_metrics[3]:
        raise ValueError(
            f"Evaluación distinta para {job['notebook']}: "
            f"obtenida={actual_metrics}, notebook={notebook_metrics}"
        )
    print(
        f"{job['task']} / {job['checkpoint']}: {len(rows)} filas, "
        f"{len(checkpoint_done)} checkpoint + {len(recovered)} documentos del log, "
        f"F1={actual_metrics[3]:.4f} (TP={actual_metrics[0]} FP={actual_metrics[1]} FN={actual_metrics[2]})"
    )
    return job, rows


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results = [reconstruct(job, repo_root) for job in JOBS]

    for job, rows in results:
        output = repo_root / job["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=job["columns"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Escrito: {output.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
