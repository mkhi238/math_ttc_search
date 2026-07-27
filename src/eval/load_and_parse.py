from datasets import load_dataset
import pandas as pd
from math_verify import parse

def make_gsm8k_parser(df, key, substring, ret_col, dtype):
  df[ret_col] = df[key].str.split(substring).str[-1].str.lstrip()
  df[ret_col] = df[ret_col].str.replace(",", "")
  df[ret_col] = df[ret_col].astype(dtype)
  return df

def make_math_parser(df, key, ret_col):
  df = df[~df['problem'].str.contains(r'\[asy\]', regex = True, na = False)].reset_index(drop = True)
  df[ret_col] = df[key].apply(lambda x: parse(f"${x}$"))

  return df
