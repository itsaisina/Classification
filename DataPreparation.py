import re
import string
from typing import Tuple

import numpy as np
import pandas as pd
import pymorphy2
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, filepath: str, text_column: str, label_column: str):
        """
        Конструктор класса DataLoader.

        :param filepath: путь к файлу данных.
        :param text_column: название колонки с текстом в файле данных.
        :param label_column: название колонки с метками классов в файле данных.
        """
        self.filepath = filepath
        self.text_column = text_column
        self.label_column = label_column

    def load_data(self) -> pd.DataFrame | None:
        """
        Функция загрузки данных из файла.

        :return: DataFrame с данными, загруженными из файла.
        """
        try:
            data = pd.read_excel(self.filepath, engine='openpyxl')
            print(f"Data loaded successfully from {self.filepath}")
            return data
        except Exception as e:
            print(f"An error occurred while loading data from {self.filepath}:\n{e}")
            return None


class TextPreprocessor:
    def __init__(self, data: pd.DataFrame, text_column: str):
        """
        Конструктор класса TextPreprocessor.

        :param data: DataFrame с данными для предобработки.
        :param text_column: название колонки с текстом для предобработки.
        """
        self.data = data
        self.text_column = text_column
        self.stop_words = set(stopwords.words('russian'))
        self.morph = pymorphy2.MorphAnalyzer()

    def remove_punctuation(self, text: str) -> str:
        """
        Функция для удаления знаков пунктуации из текста.

        :param text: текст для обработки.
        :return: текст без знаков пунктуации.
        """
        if isinstance(text, str):
            return text.translate(str.maketrans('', '', string.punctuation))
        else:
            return ""

    def to_lower(self, text: str) -> str:
        """
        Функция для преобразования текста к нижнему регистру.

        :param text: текст для обработки.
        :return: текст в нижнем регистре.
        """
        if isinstance(text, str):
            return text.lower()
        else:
            return ""

    def remove_stopwords(self, text: str) -> str:
        """
        Функция для удаления стоп-слов из текста.

        :param text: текст для обработки.
        :return: текст без стоп-слов.
        """
        if isinstance(text, str):
            return ' '.join([word for word in word_tokenize(text) if word not in self.stop_words])
        else:
            return ""

    def remove_digits(self, text: str) -> str:
        """
        Функция для удаления цифр из текста.

        :param text: текст для обработки.
        :return: текст без цифр.
        """
        if isinstance(text, str):
            return re.sub(r'\d+', '', text)
        else:
            return ""

    def lemmatize(self, text: str) -> str:
        """
        Функция для лемматизации текста.

        :param text: текст для обработки.
        :return: лемматизированный текст.
        """
        if isinstance(text, str):
            words = word_tokenize(text)
            lemmatized_words = [self.morph.parse(word)[0].normal_form for word in words]
            return ' '.join(lemmatized_words)
        else:
            return ""

    def preprocess(self) -> pd.DataFrame:
        """
        Функция для предобработки данных: применяются все вышеописанные функции.

        :return: DataFrame с предобработанными данными.
        """
        self.data[self.text_column] = self.data[self.text_column].fillna("")
        self.data[self.text_column] = self.data[self.text_column].astype(str)
        self.data[self.text_column] = self.data[self.text_column].apply(self.remove_digits)
        self.data[self.text_column] = self.data[self.text_column].apply(self.remove_punctuation)
        self.data[self.text_column] = self.data[self.text_column].apply(self.to_lower)
        self.data[self.text_column] = self.data[self.text_column].apply(self.remove_stopwords)
        self.data[self.text_column] = self.data[self.text_column].apply(self.lemmatize)
        return self.data


class TextVectorizer:
    def __init__(self, text_data: pd.Series):
        """
        Конструктор класса TextVectorizer.

        :param text_data: текстовые данные для векторизации.
        """
        self.text_data = text_data
        self.vectorizer = TfidfVectorizer()

    def vectorize_text(self) -> Tuple[csr_matrix, TfidfVectorizer]:
        """
        Функция для векторизации текстовых данных с использованием TF-IDF.

        :return: векторизованные данные и объект vectorizer, обученный на данных.
        """
        vectorized_data = self.vectorizer.fit_transform(self.text_data)
        return vectorized_data, self.vectorizer


class LabelEncoderWrapper:
    def __init__(self, data_frame: pd.DataFrame, label_column: str):
        """
        Конструктор класса LabelEncoderWrapper.

        :param data_frame: DataFrame с данными.
        :param label_column: название колонки с метками классов.
        """
        self.data_frame = data_frame
        self.label_column = label_column
        self.encoder = LabelEncoder()

    def transform(self) -> Tuple[np.ndarray, LabelEncoder]:
        """
        Функция для кодирования меток классов.

        :return: закодированные метки классов и объект encoder, обученный на данных.
        """
        self.data_frame[self.label_column] = self.data_frame[self.label_column].fillna('')
        encoded_labels = self.encoder.fit_transform(self.data_frame[self.label_column])
        return encoded_labels, self.encoder


class DataSplitter:
    def __init__(self,
                 features: csr_matrix,
                 labels: np.ndarray,
                 test_size: float = 0.2,
                 val_size: float = 0.25,
                 random_state: int = 42):
        """
        Конструктор класса DataSplitter.

        :param features: признаки для разделения на обучающую, валидационную и тестовую выборки.
        :param labels: метки классов для разделения на обучающую, валидационную и тестовую выборки.
        :param test_size: размер тестовой выборки.
        :param val_size: размер валидационной выборки.
        :param random_state: состояние генератора случайных чисел.
        """
        self.features = features
        self.labels = labels
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self) -> Tuple[csr_matrix, csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """
        Функция для разделения данных на обучающую, валидационную и тестовую выборки.

        :return: обучающую, валидационную и тестовую выборки.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=self.val_size,
                                                          random_state=self.random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test


data_loader = DataLoader(filepath="FilteredDataNewNew.xlsx", text_column="Текст открытки", label_column="Тег")
data = data_loader.load_data()

text_preprocessor = TextPreprocessor(data=data, text_column="Текст открытки")
preprocessed_data = text_preprocessor.preprocess()

text_vectorizer = TextVectorizer(preprocessed_data["Текст открытки"])
vectorized_data, fitted_vectorizer = text_vectorizer.vectorize_text()

n_unique_tokens = len(fitted_vectorizer.vocabulary_)
print(f"Общее количество уникальных токенов после предобработки: {n_unique_tokens}")

label_encoder = LabelEncoderWrapper(data_frame=preprocessed_data, label_column="Тег")
encoded_labels, le = label_encoder.transform()
print(le.classes_)

data_splitter = DataSplitter(features=vectorized_data, labels=encoded_labels)
X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split_data()
