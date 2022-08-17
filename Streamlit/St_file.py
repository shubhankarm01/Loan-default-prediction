import pandas as pd
import streamlit as st
import pickle

st.title("Loan default prediction")
    
# Import input data for the model
df = pd.read_csv('df.csv')
st.write('## Model data' , df)

# Input transformed data
df_ml = pd.read_csv('df_ml.csv')
X = df_ml.iloc[:, :-1]
Y = df_ml.iloc[:, -1]

# Oversampling of minority class for imbalanced data
from imblearn.over_sampling import ADASYN
oversample = ADASYN(sampling_strategy = 0.8)
x, y = oversample.fit_resample(X, Y)   
# print(Y.value_counts(), '\n\n', y.value_counts())

# Splitting of data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, stratify = y, random_state = 6)

# Function for training of the model
def model_train():
    from xgboost import XGBClassifier

    xgb = XGBClassifier(max_depth = max_depth, colsample_bytree = colsample_bytree, gamma = gamma, learning_rate = learning_rate, reg_lambda = reg_lambda, subsample = subsample, random_state = 42)

    xgb.fit(x_train, y_train)
    
    import pickle
    pickle_out = open("xgb.pkl", mode = 'wb')
    pickle.dump(xgb, pickle_out)
    pickle_out.close()
    
    return xgb
    
# Input option for model parameters
st.write('---')
st.write('Parametrs for XGBoost model')
max_depth = st.slider('max_depth', 3, 10, step = 1)
learning_rate = st.slider('learning_rate', 0.05, 0.5, step = 0.05)
gamma = st.slider('gamma', 0.0, 1.0, step = 0.1)
reg_lambda = st.slider('reg_lambda', 0, 10, step = 1)
subsample = st.slider('subsample', 0.1, 1.0, step = 0.01)
colsample_bytree = st.slider('colsample_bytree', 0.1, 1.0, step = 0.05)


if st.button("Train"):
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    xgb = model_train()
    
    y_pred = xgb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) 
    confusion_matrix = confusion_matrix(y_test, y_pred)
    
    st.write('accuracy: ', accuracy)
    st.write('Confusion metrics: ', confusion_matrix)


# Input data for prediction    
st.write('---')
st.write('Input the data for prediction')
amount = st.number_input('amount')
duration = st.number_input('duration')
payments = st.number_input('payments')
DID = st.number_input('DID')
Average_order_amount = st.number_input('Average_order_amount')
Average_trans_amount = st.number_input('Average_trans_amount')
Average_trans_balance = st.number_input('Average_trans_balance')
No_transaction = st.number_input('No_transaction')
Card_type = st.selectbox('Card_type', ("No", "classic", "gold", "junior"))
No_inhabitants = st.number_input('No_inhabitants')
Average_salary = st.number_input('Average_salary')
Average_unemployment_rate = st.number_input('Average_unemployment_rate')
Average_crime_rate = st.number_input('Average_crime_rate')
gender = st.selectbox('gender', ("female", "male"))
Owner_age = st.number_input('Owner_age')
Same_district = st.selectbox('Same_district', ("1", "0"))


# Transforming input data
cat_columns = []
num_columns = []
bool_columns = []

for i in df.columns:
    if df[i].dtype == 'object':
        cat_columns.append(i)
    else:
        if df[i].dtype =='bool':
            bool_columns.append(i)
        else:
            num_columns.append(i)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

column_transformer = ColumnTransformer([('num_trans', MinMaxScaler(), num_columns),
                                        ('cat_trans', OneHotEncoder(), cat_columns)], 
                                       remainder = 'passthrough'
                                      )
column_transformer.fit(df)


st.write('---')
model_type = st.selectbox('Model type', ("Trained model", "Optimized model"))

if st.button("predict"):
    df2 = [amount, duration, payments, DID, Average_order_amount, Average_trans_amount, Average_trans_balance, No_transaction, Card_type, No_inhabitants, Average_salary, Average_unemployment_rate, Average_crime_rate, gender, Owner_age, Same_district]

    df2 = pd.DataFrame(df2, df.columns)
    df2 = df2.T

    df_ml = pd.DataFrame(column_transformer.transform(df2))
    df_ml.columns = column_transformer.get_feature_names_out()

    st.write('model input', df_ml)
    
    if model_type == 'Trained model':
        pickle_in = open('xgb.pkl', 'rb')
        xgb = pickle.load(pickle_in)
    elif model_type == 'Optimized model':
        pickle_in = open('xgb_optimized.pkl', 'rb')
        xgb = pickle.load(pickle_in)

    prediction = xgb.predict(df_ml.values)
    # prediction = xgb.predict(df_ml)

    if prediction == 0:
        result = "Non-default"
    else:
        result = "Default"

    st.write('### Result: ', result)
    
    
# df2 = [185544, 36, 5154.00, 293, 3670.400, 6633.824561, 22717.736842, 57, 'No', 124605, 8772, 4.835, 0.024124, 'male', 708.0, 1]