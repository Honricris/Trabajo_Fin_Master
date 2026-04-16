import pandas as pd
import os
import re

def create_hf_dataset():
    tsv_path = "ProcteMist/medprocner_data/medprocner_test/tsv/medprocner_tsv_test_subtask1.tsv"
    txt_dir = "ProcteMist/medprocner_data/medprocner_test/txt"

    df_annotations = pd.read_csv(tsv_path, sep="\t")
    hf_dataset = []

    # Detect column names for start/end
    if 'off0' in df_annotations.columns and 'off1' in df_annotations.columns:
        start_col = 'off0'
        end_col = 'off1'
        print(f"[DEBUG] Usando columnas: off0/off1 en el TSV: {tsv_path}")
    elif 'start_span' in df_annotations.columns and 'end_span' in df_annotations.columns:
        start_col = 'start_span'
        end_col = 'end_span'
        print(f"[DEBUG] Usando columnas: start_span/end_span en el TSV: {tsv_path}")
    else:
        raise ValueError("TSV file must contain either 'off0'/'off1' or 'start_span'/'end_span' columns.")

    for filename in df_annotations['filename'].unique():
        txt_path = os.path.join(txt_dir, f"{filename}.txt")

        if not os.path.exists(txt_path):
            continue

        print(f"[DEBUG] Procesando archivo de texto: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()

        file_annotations = df_annotations[df_annotations['filename'] == filename].sort_values(start_col)

        tokens = []
        ner_tags = []

        for match in re.finditer(r'\w+|[^\w\s]', text):
            tok_start = match.start()
            tok_end = match.end()

            tokens.append(match.group())
            tag = 2

            for _, row in file_annotations.iterrows():
                ent_start = int(row[start_col])
                ent_end = int(row[end_col])

                if tok_start >= ent_start and tok_end <= ent_end:
                    tag = 0 if tok_start == ent_start else 1
                    break

            ner_tags.append(tag)

        hf_dataset.append({
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags
        })

    df_final = pd.DataFrame(hf_dataset)
    df_final.to_json("proctemist_test.jsonl", orient="records", lines=True)
    print("Dataset successfully saved as proctemist_test.jsonl")

if __name__ == "__main__":
    create_hf_dataset()
