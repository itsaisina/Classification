from scipy.sparse import csr_matrix
from numpy import ndarray
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from DataPreparation import X_train, y_val, y_train, X_val

classifiers = {
    "Logistic Regression": LogisticRegression(C=100,
                                              penalty='l2',
                                              solver='liblinear',
                                              random_state=42),
    "Random Forest": RandomForestClassifier(max_depth=None,
                                            n_estimators=700,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            random_state=42),
    "SVM": SVC(random_state=42),
    "XGB": XGBClassifier(colsample_bytree=0.5,
                         learning_rate=0.1,
                         max_depth=8,
                         n_estimators=100,
                         subsample=0.7,
                         random_state=42)
}


def train_model(model, X_train: csr_matrix, y_train: ndarray):
    """
    Функция для обучения модели на данных.

    :param model: модель для обучения.
    :param X_train: матрица признаков для обучения.
    :param y_train: целевой вектор для обучения.
    :return: обученная модель.
    """
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    model.fit(X_train_smote, y_train_smote)
    return model


def evaluate_model(model, X_val: csr_matrix, y_val: ndarray) -> None:
    """
    Функция для оценки модели на валидационных данных.

    :param model: модель для оценки.
    :param X_val: матрица признаков для валидации.
    :param y_val: целевой вектор для валидации.
    :return: None
    """
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)
    print(f'Accuracy on validation set: {accuracy}\n')
    print(report)


def cross_validate_model(model, X_train: csr_matrix, y_train: ndarray) -> None:
    """
    Функция для кросс-валидации модели на обучающих данных.

    :param model: модель для кросс-валидации.
    :param X_train: матрица признаков для обучения.
    :param y_train: целевой вектор для обучения.
    :return: None
    """
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
    print(f'Cross-validation Accuracy: {scores.mean()}')


for name, classifier in classifiers.items():
    print(f"Model: {name}")
    model = train_model(classifier, X_train, y_train)
    evaluate_model(model, X_val, y_val)
    cross_validate_model(model, X_train, y_train)
    print("-" * 60)
