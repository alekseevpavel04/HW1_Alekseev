from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import re

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

# Это функция для предсказания на одном объекте
def predict_item(item: Item) -> float:
    df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
    df_test = pd.DataFrame([item.dict()], columns=item.dict().keys())
    def process_torque_column(column):
        def classify_torque_type(row):
            row = str(row)
            if re.search(r'\d+\s*-\s*\d+\s*rpm', row):
                return 'range_rpm'
            elif re.search(r'\d+\s*(Nm|kgm)@', row):
                return 'value_torque'
            elif re.search(r'\d+@', row):
                return 'value_rate'
            elif re.search(r'\d+\s*(kgm@|rpm)', row):
                return 'units_last'
            else:
                return 'unknown'

        def extract_max_torque_rpm(row, torque_type):
            if torque_type == 'range_rpm':
                rpm_str = re.findall(r'\d+', row)
                try:
                    rpm_values = [float(val.replace(',', '')) for val in rpm_str]
                    return max(rpm_values)
                except (ValueError, IndexError):
                    pass
            elif torque_type == 'value_torque':
                rpm_str = re.findall(r'\d+', row)
                try:
                    rpm_value = float(rpm_str[-1].replace(',', ''))
                    return rpm_value
                except (ValueError, IndexError):
                    pass
            elif torque_type == 'value_rate':
                rpm_str = re.findall(r'[\d,.]+', row)
                try:
                    rpm_value = float(rpm_str[-1].replace(',', '').replace('@', ''))
                    return rpm_value
                except (ValueError, IndexError):
                    pass
            elif torque_type == 'units_last':
                rpm_str = re.findall(r'\d+', row)
                try:
                    rpm_value = float(rpm_str[-1].replace(',', ''))
                    return rpm_value
                except (ValueError, IndexError):
                    pass
            return None

        processed_data = column.astype(str).apply(lambda x: extract_max_torque_rpm(x, classify_torque_type(x)))
        return processed_data

    def extract_torque(column):
        def extract_value(row):
            row = str(row)
            match = re.search(r'([\d.]+)', row)
            if match:
                value = float(match.group(1))
                if '@' in row:
                    return value
                else:
                    return value * 9.80665
            else:
                return None

        torque_values = column.astype(str).apply(extract_value)
        return torque_values

    df_test['mileage'] = pd.to_numeric(df_test['mileage'].str.extract(r'([0-9.]+)', expand=False))
    df_test['engine'] = pd.to_numeric(df_test['engine'].str.extract(r'([0-9.]+)', expand=False))
    df_test['max_power'] = pd.to_numeric(df_test['max_power'].str.extract(r'([0-9.]+)', expand=False))
    df_test['max_torque_rpm'] = process_torque_column(df_test['torque'])
    df_test['torque'] = extract_torque(df_test['torque'])

    # Загрузка данных из файла column_modes
    column_modes = pd.read_csv('column_modes.csv')
    # Проход по столбцам df_test для замены пропущенных значений
    for column in df_test.columns:
        # Проверка, есть ли пропущенные значения в столбце
        if df_test[column].isnull().any():
            # Если есть пропуски, заменяем их на соответствующее значение из column_modes
            if column in column_modes.columns and not pd.isnull(column_modes[column].iloc[0]):
                mode_value = column_modes[column].iloc[0]
                df_test[column].fillna(mode_value, inplace=True)

    df_test["year2"] = df_test["year"] ** 2
    df_test = df_test.drop("selling_price", axis=1)
    df_test['name'] = df_test['name'].str.split().str.get(0)

    # Заменяем редкие (как и раньше)
    df_train_processed = df_train
    df_train['name'] = df_train['name'].str.split().str.get(0)
    columns_to_check = ["name", "fuel", "seller_type", "transmission", "owner", "seats"]

    for i in range(len(columns_to_check)):
        # Получение уникальных значений из столбца типа object в df_train_processed
        unique_values_train = df_train_processed[columns_to_check[i]].unique()
        # Замена значений в df_test, которых нет в df_train_processed, на "Rare"
        df_test[columns_to_check[i]] = df_test[columns_to_check[i]].apply(
            lambda x: x if x in unique_values_train else 'Rare')

    df_test = df_test.apply(pd.to_numeric, errors='ignore')
    df_test["m/m"] = df_test["max_power"] / (100 / df_test["mileage"])
    df_test["seats"] = df_test["seats"].astype(int).astype(str)
    df_test = pd.get_dummies(df_test)

    # Загрузка df_train_processed
    df_train_processed = pd.read_csv('df_train_processed.csv').drop("selling_price",
                                                                    axis=1)  # укажите путь к вашему файлу df_train_processed
    # Получение столбцов, которые есть в df_train_processed, но отсутствуют в df_test
    missing_columns = [col for col in df_train_processed.columns if col not in df_test.columns]
    # Добавление отсутствующих столбцов в df_test и заполнение их нулями
    for col in missing_columns:
        df_test[col] = 0

        # Ensure the order of column in the test set is in the same order than in train set
    df_test = df_test[df_train_processed.columns]

    # Загрузка модели из файла
    with open('model.pkl', 'rb') as file:
        model6 = pickle.load(file)
    pred6 = np.exp(model6.predict(df_test))

    prediction = pred6

    return prediction

# Это функция для предсказания на коллекции объектов
def predict_items(items: Items) -> List[float]:
    predictions = []
    for item in items.objects:
        prediction = predict_item(item)
        predictions.append(prediction)
    return predictions

# Роут для предсказания на одном объекте
@app.post("/predict_item")
def get_predict_item(item: Item) -> float:

    return predict_item(item)

# Роут для предсказания на коллекции объектов
@app.post("/predict_items")
def get_predict_items(items: Items) -> List[float]:

    return predict_items(items)
