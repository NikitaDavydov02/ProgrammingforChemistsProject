import numpy as np
import pandas as pd
import math
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from plotly.subplots import make_subplots

df = pd.read_csv('DATASET.csv')
print(df.head())

#Заменяем знаки в цифровых данных
df = df.replace(',', '.', regex=True)

# Функция для преобразования строк в float, исключая определенные столбцы
def convert_to_float(df, exclude_cols):
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = df[col].astype(float)
    return df

exclude = ['smiles', 'CMPD_CHEMBLID']
df = convert_to_float(df, exclude)

#Удаление ненужных данных(дубликатов и пустых столбцов)
df=df.drop(columns=['Unnamed: 0'])
df = df.drop_duplicates()

# Группировка и расчет количества повторений
repeat_df =df[['smiles', 'CMPD_CHEMBLID']].groupby('smiles').agg('count')
repeat_df[repeat_df['CMPD_CHEMBLID'] >= 2]

df = df.drop_duplicates (subset=['smiles', 'CMPD_CHEMBLID'])
df = df.loc[:, df.any()]
print(dict(df.isnull().sum()))

df = df.drop(columns = ['PEOE_VSA9','PEOE_VSA10','CMPD_CHEMBLID'])
df = df.dropna(subset=['BCUT2D_MWHI'])

#Заполнение пропущенных данных с помощью библиотеки RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors


def calculate_logp(smiles):
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.Crippen.MolLogP(mol)
    return None

assert math.isclose(calculate_logp('CN1:C(CN2CCN(C3:C:C:C(Cl):C:C:3)CC2):N:C2:C:C:C:C:C:2:1'),3.5489), "Тест не пройден!"

df['MolLogP'] = df['MolLogP'].fillna(df['smiles'].apply(calculate_logp))

def calculate_MaxPartialCharge(smiles):
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MaxPartialCharge(mol)
    return None

assert math.isclose(calculate_MaxPartialCharge('CN1:C(CN2CCN(C3:C:C:C(Cl):C:C:3)CC2):N:C2:C:C:C:C:C:2:1'), 0.1233429033), "Тест не пройден!"

df['MaxPartialCharge'] = df['MaxPartialCharge'].fillna(df['smiles'].apply(calculate_MaxPartialCharge))

def calculate_MinPartialCharge(smiles):
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MinPartialCharge(mol)
    return None

assert math.isclose(calculate_MinPartialCharge('CN1:C(CN2CCN(C3:C:C:C(Cl):C:C:3)CC2):N:C2:C:C:C:C:C:2:1'), -0.3689644732), "Тест не пройден!"

df['MinPartialCharge'] = df['MinPartialCharge'].fillna(df['smiles'].apply(calculate_MinPartialCharge))

def calculate_MaxAbsPartialCharge(smiles):
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MaxAbsPartialCharge(mol)
    return None

assert math.isclose(calculate_MaxAbsPartialCharge('CN1:C(CN2CCN(C3:C:C:C(Cl):C:C:3)CC2):N:C2:C:C:C:C:C:2:1'), 0.3689644732), "Тест не пройден!"

df['MaxAbsPartialCharge'] = df['MaxAbsPartialCharge'].fillna(df['smiles'].apply(calculate_MaxAbsPartialCharge))

def calculate_MinAbsPartialCharge(smiles):
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MinAbsPartialCharge(mol)
    return None

assert math.isclose(calculate_MinAbsPartialCharge('CN1:C(CN2CCN(C3:C:C:C(Cl):C:C:3)CC2):N:C2:C:C:C:C:C:2:1'), 0.1233429033), "Тест не пройден!"

df['MinAbsPartialCharge'] = df['MinAbsPartialCharge'].fillna(df['smiles'].apply(calculate_MinAbsPartialCharge))

print(df)

print(dict(df.isnull().sum()))

df.to_csv('df198.csv', index=True)

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=32)

val_df.to_csv('val_df.csv', index=True)
train_df.to_csv('train_df.csv', index=True)