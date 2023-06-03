from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from seaborn import heatmap
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from numpy import ndarray

from DataPreparation import X_test, y_test
from ModelTraining import classifiers, train_model, X_train, y_train


def evaluate_model_test(model, X_test: csr_matrix, y_test: ndarray) -> None:
    """
    Функция для оценки модели на тестовых данных.

    :param model: модель для оценки.
    :param X_test: матрица признаков для тестирования.
    :param y_test: целевой вектор для тестирования.
    :return: None
    """
    y_test_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    confusion = confusion_matrix(y_test, y_test_pred)

    print(f'Accuracy on test set: {accuracy}\n')
    print('Classification report:\n', report)
    print('Confusion matrix:\n', confusion)
    heatmap(confusion, annot=True, fmt='d')
    plt.show()


for name, classifier in classifiers.items():
    print(f"Model: {name}")
    model = train_model(classifier, X_train, y_train)
    evaluate_model_test(model, X_test, y_test)
    print("-" * 60)
