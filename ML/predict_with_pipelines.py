import pickle
import pandas as pd

pipe = pickle.load(open('models/pipe.pkl','rb'))

# Assume user input
# columns order matches the original training data
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

test_input2 = pd.DataFrame([
    [2, 'male', 31.0, 0, 0, 10.5, 'S']
], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

print(pipe.predict(test_input2))