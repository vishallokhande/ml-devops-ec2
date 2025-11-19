# src/train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_FILE = 'model.pkl'

def train_and_save():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    print('Train score:', clf.score(X_train, y_train))
    print('Test score:', clf.score(X_test, y_test))
    joblib.dump(clf, MODEL_FILE)
    print(f'Saved model to {MODEL_FILE}')

if __name__ == '__main__':
    train_and_save()
