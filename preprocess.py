import io
import os

import pandas as pd

from config import DATA_DIR


def integrate_data(data_filenames):
    return pd.concat(
        [pd.read_csv(os.path.join(DATA_DIR, filename), sep=";", engine="python") for filename in data_filenames]
    ).drop_duplicates().reset_index(drop=True).rename(columns={"description ": "description"})


def fill_missing_values(df):
    return df


def write_embedding_files(labels, embedded_tensors, path=DATA_DIR):
  out_v = io.open(os.path.join(path, 'vecs.tsv'), 'w', encoding='utf-8')
  out_m = io.open(os.path.join(path, 'meta.tsv'), 'w', encoding='utf-8')
  vectors = embedded_tensors.numpy()
  for message, vector in zip(labels, vectors):
    out_m.write(message + "\n")
    out_v.write('\t'.join([str(x) for x in vector]) + "\n")
  out_v.close()
  out_m.close()
