# Импортируем библиотеки
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Создаем набор данных
data = [
    [25, 50000, 1],
    [30, 70000, 0],
    [35, 90000, 1],
    [20, 30000, 0],
    [28, 60000, 1]
]

# Разделяем признаки (X) и метки (y)
X = [[row[0], row[1]] for row in data]
y = [row[2] for row in data]

# Разделяем данные на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем решающее дерево
clf = DecisionTreeClassifier()

# Обучаем модель
clf.fit(X_train, y_train)

# Делаем предсказания
predictions = clf.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
