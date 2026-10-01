import pickle
import numpy as np
import pandas as pd

ohe_sex = pickle.load(open('models/ohe_sex.pkl','rb'))
ohe_embarked = pickle.load(open('models/ohe_embarked.pkl','rb'))
clf = pickle.load(open('models/clf.pkl','rb'))

# Assume user input
# Pclass/gender/age/SibSp/Parch/Fare/Embarked
test_input = pd.DataFrame([
    [2, 'male', 31.0, 0, 0, 10.5, 'S']
], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

test_input_sex = ohe_sex.transform(test_input[['Sex']])
test_input_embarked = ohe_embarked.transform(test_input[['Embarked']])
test_input_age = test_input[['Age']].values

test_input_transformed = np.concatenate((
    test_input[['Pclass', 'SibSp', 'Parch', 'Fare']].values,
    test_input_age,
    test_input_sex,
    test_input_embarked
), axis=1)

print(clf.predict(test_input_transformed))