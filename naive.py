from preprocessing import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
import joblib 

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))


print("\nNaive Bayes Report:")
print(classification_report(y_test, nb_pred))

joblib.dump(nb_model, 'nb_model.pkl')
