# Импортируем необходимые библиотеки
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Загружаем набор данных Iris
iris = load_iris()
X = iris.data
y = iris.target

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем случайный лес с 100 деревьями
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучаем модель на обучающем наборе данных
clf.fit(X_train, y_train)

# Делаем предсказания на тестовом наборе данных
predictions = clf.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Выводим отчет о классификации
report = classification_report(y_test, predictions, target_names=iris.target_names)
print(report)
