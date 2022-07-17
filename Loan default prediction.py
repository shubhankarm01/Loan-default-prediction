#!/usr/bin/env python
# coding: utf-8

# # For connecting and retrieving query results from MYSQL

# In[1]:


# pip install mysql-connector-python


# In[2]:


import mysql.connector
import pandas as pd

class mysqlconnector:
    
    def __init__(self, database = 'test'):
        try:
            connection = mysql.connector.connect(host = 'localhost',
                                                user = 'root',
                                                password = '',
                                                use_pure = True,
                                                database = database
                                                )
            if connection.is_connected:
                db_info = connection.get_server_info()
                print('connected to MYSQL server version', db_info)
                print('You are connected to the database:', database)
                self.connection = connection
        except Exception as e:
            print('Error while connecting to MYSQL', e)
            
    def execute(self, query, header = False):
        cursor = self.connection.cursor(buffered = True)
        cursor.execute(query)
        
        try:
            record = cursor.fetchall()
            if header:
                header = [i[0] for i in cursor.description]
                return {'header': header, 'record': record}
            else:
                return record
        except:
            pass
        
    def to_df(self, query):
        result = self.execute(query, header = True)
        df = pd.DataFrame(result['record'])
        df.columns = result['header']
        return df


# In[3]:


db = mysqlconnector('bank')


# # EDA

# In[5]:


query = 'SELECT * FROM Loan JOIN Account USING(account_id);'
df = db.to_df(query)


# In[6]:


df.sample(10)


# In[7]:


# df_good for food loans
df_good = df[df['status'].isin(['A','C'])]
df_good.shape


# In[8]:


# df_bad for bad loans
df_bad = df[df['status'].isin(['B', 'D'])]
df_bad.shape


# In[9]:


import matplotlib.pyplot as plt

# To plot histogram of the loan amount
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df_good['amount'].hist(bins = 25, ax = ax1, label = 'good loans', color = 'grey')
df_bad['amount'].hist(bins =  25, ax = ax2, label = 'bad loans', color = 'red')

ax1.legend()
ax2.legend()


# In[9]:


# To plot histogram of the loan duration

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df_good['duration'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df_bad['duration'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend(loc = 9)
ax2.legend()


# In[10]:


# To plot histogram of the loan monthly installment

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df_good['payments'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df_bad['payments'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[11]:


# To plot histogram of the loan location

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df_good['district_id'].hist(ax = ax1, label = 'good_loan', color = 'grey', bins = 25)
df_bad['district_id'].hist(ax = ax2, label = 'bad_loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[10]:


# To rename columns date column in Loan and Account tables

query = """ALTER TABLE Loan 
        CHANGE COLUMN `date` Loan_date DATE;"""

df1 = db.to_df(query)


# In[11]:


query = 'SELECT * FROM Loan;'

df1 = db.to_df(query)
df1.head()


# In[ ]:


query = 'ALTER TABLE Account CHANGE COLUMN `date` Account_date DATE;'
db.to_df(query)


# In[12]:


query = 'SELECT * FROM Loan JOIN Account USING(account_id)'
df = db.to_df(query)
df.head()


# In[13]:


df1 = df.copy()
# df1.rename(columns = {'date': 'loan_dates'})
df1.columns.duplicated()


# In[14]:


# feature creation from date of account opening and date of applying loan (DID : Dfference In Dates)

df['DID'] = df['Loan_date'] - df['Account_date']
df_good = df[df['status'].isin(['A', 'C'])]
df_bad = df[df['status'].isin(['B', 'D'])]


# In[15]:


# Converts timedelta64 to float64 as timedelta was not being able to use for plotting histogram
df_good['DID'].astype('timedelta64[D]')


# In[16]:


# Another way of converting the timedelta64 to the int64 for plotting histogram
df_good['DID'].dt.days


# In[17]:


# To plot the histogram of the time difference between opening the account and the taking the loan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
df_good['DID'].astype('timedelta64[D]').hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df_bad['DID'].astype('timedelta64[D]').hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# ## Joining district along with the loan and account tables

# In[20]:


query = """SELECT * FROM Loan JOIN Account USING(account_id) JOIN District USING(district_id)"""

df1 = db.to_df(query)


# In[22]:


df1.sample(10)


# In[23]:


df1 = df1.rename(columns = {'A4': 'No_inhabitants', 'A11': 'Average_salary', 'A14': 'No_entrepreneur_per1000'})

df1['Average_unemployment_rate'] = df1[['A12', 'A13']].mean(axis = 1)
df1['Average_crime_rate'] = df1[['A15', 'A16']].mean(axis = 1)/df1['No_inhabitants']

df1.head()


# In[24]:


df1.columns


# In[25]:


df1_good = df1[df1['status'].isin(['A', 'C'])]
df1_bad = df1[df1['status'].isin(['B', 'D'])]


# In[26]:


# Histograme plot for Number of inhabitants

fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (15,5))

df1_good['No_inhabitants'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df1_bad['No_inhabitants'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[28]:


# Histogram plot for average salary

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df1_good['Average_salary'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df1_bad['Average_salary'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[29]:


# Histogram to plot for average unemployment rate 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df1_good['Average_unemployment_rate'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df1_bad['Average_unemployment_rate'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[30]:


df1['Average_unemployment_rate']


# In[33]:


# Histogram for average crime rate

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))

df1_good['Average_crime_rate'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df1_bad['Average_crime_rate'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# ## Matching order tabel with loan table

# In[29]:


query = 'SELECT * FROM Loan;'

df2 = db.to_df(query)


# In[30]:


df2.head()


# In[31]:


query = """
        SELECT account_id, amount Order_amount
        FROM `Order`
        WHERE account_id in (SELECT account_id FROM Loan);
        """

df3 = db.to_df(query)


# In[32]:


df3.head()


# In[33]:


# Datatype of Order_amount was 'Object' before changing it to 'float'

df3['Order_amount'] = df3['Order_amount'].astype('float')


# In[34]:


df3.groupby('account_id').mean()


# In[35]:


df4 = df2.set_index('account_id').join(df3.groupby('account_id').mean())
df4.head()

# or use pd.merge(df2, df3.groupby('account_id').mean(), on = 'account_id', how = 'outer').set_index('account_id')


# In[36]:


df4 = df4.rename(columns = {'Order_amount': 'Average order_amount'})
df4.head()


# In[39]:


df4_good = df4[df4['status'].isin(['A', 'C'])]
df4_bad = df4[df4['status'].isin(['B', 'D'])]


# In[76]:


# histogram plot for average order amount

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
df4_good['Average order_amount'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df4_bad['Average order_amount'].hist(ax= ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# ## Combining loan table with transaction table

# In[80]:


query = """
        SELECT * 
        FROM Loan;"""

df5 = db.to_df(query)


# In[81]:


df5.head()


# In[83]:


query = """
        SELECT account_id, amount Trans_amount, balance Trans_balance
        FROM Trans WHERE account_id IN (SELECT account_id FROM Loan);
        """

df6 = db.to_df(query)


# In[91]:


df6.head()


# In[100]:


No_transaction = df6.groupby('account_id').count().iloc[:,1]
No_transaction.name = 'No_transaction'
No_transaction


# In[101]:


df6 = df6.groupby('account_id').mean()
df6.columns = ['Average_trans_amount', 'Average_trans_balance']
df6.head()


# In[108]:


df7 = df5.set_index('account_id').join(df6).join(No_transaction)
df7.sample(5)


# In[109]:


print(df5.shape, df6.shape, No_transaction.shape, df7.shape)


# In[110]:


df7_good = df7[df7['status'].isin(['A', 'C'])]
df7_bad = df7[df7['status'].isin(['B', 'D'])]


# In[112]:


# Histogram plot for average transaction amount

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

df7_good['Average_trans_amount'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df7_bad['Average_trans_amount'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[114]:


# Histogram plot for average balance after transaction

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

df7_good['Average_trans_balance'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df7_bad['Average_trans_balance'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# In[117]:


# Histogram for number of transaction

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

df7_good['No_transaction'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df7_bad['No_transaction'].hist(ax = ax2, label = 'bad loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# ## Combining Loan tabel with Credit card tabel

# In[8]:


query = """
        SELECT account_id, card_id, Card.type Card_type, status
        FROM Loan JOIN Desposition USING(account_id) LEFT JOIN Card USING(disp_id)
        """

# JOIN operation with USING gives only those rows in which the column mention in the USING matches.

df8 = db.to_df(query)
df8.sample(5)


# In[9]:


df8_card = df8[~df8['card_id'].isna()]

# FROM Loan JOIN Desposition USING(account_id) JOIN Card USING(disp_id); can also be used but it was not working


# In[10]:


# query = """
#         SELECT account_id, card_id, Card.type Card_type, status
#         FROM Loan JOIN Desposition USING(account_id) JOIN Card USING(disp_id);
#         """

# df9 = db.to_df(query)
# df9.sample(5)


# In[11]:


df8_good = df8[df8['status'].isin(['A', 'C'])]
df8_bad = df8[df8['status'].isin(['B', 'D'])]


# In[12]:


# Barchart for credit card type

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5)) 

df8_good[['Card_type', 'account_id']].groupby('Card_type', dropna = False).count().plot.bar(ax = ax1, color = 'grey', label = 'good loan', legend = False)
df8_bad[['Card_type', 'account_id']].groupby('Card_type', dropna = False).count().plot.bar(ax = ax2, color = 'red', label = 'bad loan', legend = False)


# ## Combing Account table and Client Table

# In[3]:


# Taking aprrox. 6 mins to run

query = """
        SELECT * 
        FROM Loan JOIN Desposition USING(account_id) 
        JOIN Client USING(Client_id) 
        JOIN District USING(district_id);
        """
df9 = db.to_df(query)


# In[4]:


df9.head()


# In[5]:


query = """
         SELECT account_id, district_id Acc_dist_id
         FROM Loan JOIN Account USING(account_id);
         """

df10 = db.to_df(query)


# In[6]:


df10.head()


# In[98]:


df11 = df9.set_index('account_id').join(df10.set_index('account_id'))

# or
# df12 = pd.merge(df9, df10, on = 'account_id', how = 'inner').set_index('account_id')


# In[99]:


df11.head()


# In[100]:


df11['type'].unique()


# In[101]:


df11 = df11[df11['type'] == 'OWNER']


# In[102]:


print(df11.shape, df11.index.unique().shape)


# In[103]:


df11['Same_district'] = df11['district_id'] == df11['Acc_dist_id']


# In[104]:


df11.head()


# In[105]:


df11['Loan_date'] = df11['Loan_date'].astype('datetime64')
df11['birth_date'] = df11['birth_date'].astype('datetime64')
df11['Owner_age'] = df11['Loan_date'] - df11['birth_date']


# In[106]:


df11['Owner_age']


# In[107]:


# To convert days value of timedelta64 type into year values of float64 type

import numpy as np

df11['Owner_age'] = df11['Owner_age']/np.timedelta64(1, 'Y')


# In[108]:


df11 = df11.reset_index()
df11_good = df11[df11['status'].isin(['A', 'C'])]
df11_bad = df11[df11['status'].isin(['B', 'D'])]


# In[109]:


df11_good.head()


# In[110]:


# Bar chart for client district and account district
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

df11_good[['account_id', 'Same_district']].groupby('Same_district').count().plot.bar(ax = ax1, color = 'grey', legend = False)
df11_bad[['account_id', 'Same_district']].groupby('Same_district').count().plot.bar(ax = ax2, color = 'red', legend = False)


# In[111]:


# Barchart for gender

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

df11_good[['gender', 'account_id']].groupby('gender').count().plot.bar(ax = ax1, color = 'grey', legend = False)
df11_bad[['gender', 'account_id']].groupby('gender').count().plot.bar(ax= ax2, color = 'red', legend = False)


# In[112]:


# histogram for owner age

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

df11_good['Owner_age'].hist(ax = ax1, label = 'good loan', color = 'grey', bins = 25)
df11_bad['Owner_age'].hist(ax = ax2, label = 'bad_loan', color = 'red', bins = 25)

ax1.legend()
ax2.legend()


# # Finalizing Features

# In[4]:


query = """
        SELECT account_id, amount, duration, payments, Loan_date, status, Account_date, A4, A11, A12, A13, A14, A15, A16
        FROM Loan JOIN Account USING(account_id) JOIN District USING(district_id)
        """

df = db.to_df(query)


# In[5]:


df.head()


# In[6]:


df.set_index('account_id', inplace = True)


# In[7]:


df['DID'] = (df['Loan_date'] - df['Account_date']).dt.days
df = df.rename(columns = {'A4': 'No_inhabitants', 'A11': 'Average_salary'})
df['Average_unemployment_rate'] = df[['A12', 'A13']].mean(axis = 1)
df['Average_crime_rate'] = df[['A15', 'A16']].mean(axis = 1)/df['No_inhabitants']


# In[8]:


query = """
        SELECT account_id, amount Order_amount
        FROM `Order`
        WHERE account_id IN(SELECT account_id FROM Loan);
        """

df2 = db.to_df(query)


# In[9]:


df2['Order_amount'] = df2['Order_amount'].astype('float')
df = df.join(df2.groupby('account_id').mean())
df = df.rename(columns = {'Order_amount': 'Average_order_amount'})


# In[10]:


df.head()


# In[11]:


query = """
        SELECT account_id, amount Trans_amount, balance Trans_balance
        FROM Trans
        WHERE account_id IN (SELECT account_id FROM Loan);
        """

df3 = db.to_df(query)


# In[12]:


df3.head()


# In[13]:


No_transaction = df3.groupby('account_id').count().iloc[: , 1]


# In[14]:


No_transaction.name = 'No_transaction'


# In[15]:


df3 = df3.groupby('account_id').mean()
df3.columns = ['Average_trans_amount', 'Average_trans_balance']


# In[16]:


df = df.join(df3).join(No_transaction)


# In[17]:


df.head()


# In[18]:


query = """
        SELECT account_id, Card.type Card_type
        FROM Loan JOIN Desposition USING(account_id) LEFT JOIN Card USING(disp_id)
        WHERE Desposition.type = 'OWNER';
        """

df4 = db.to_df(query)


# In[19]:


df4.head()


# In[20]:


df = df.join(df4.set_index('account_id'), how = 'left')


# In[21]:


df['Card_type'].fillna('No', inplace = True)


# In[22]:


df.head()


# In[23]:


query = """
        SELECT account_id, Loan_date, Account.district_id Account_dist_id, Client.district_id Client_dist_id, gender, birth_date
        FROM Loan JOIN Account USING(account_id) JOIN Desposition USING(account_id) JOIN Client USING(Client_id) WHERE Desposition.type = 'Owner';
        """

df5 = db.to_df(query)


# In[24]:


df5.head()


# In[25]:


df5['Same_district'] = df5['Account_dist_id'] == df5['Client_dist_id']
df5['Owner_age'] = (df5['Loan_date'] - df5['birth_date']).astype('timedelta64[M]')


# In[26]:


df = df.join(df5.set_index('account_id')[['Same_district', 'gender', 'Owner_age']])


# In[27]:


df.head()


# In[28]:


df.shape


# In[29]:


df['Default'] = df['status'].isin(['A', 'D'])


# In[30]:


df = df[['amount', 'duration', 'payments', 'DID', 'Average_order_amount',
         'Average_trans_amount', 'Average_trans_balance', 'No_transaction', 
         'Card_type', 'No_inhabitants', 'Average_salary', 'Average_unemployment_rate', 
         'Average_crime_rate', 'gender', 'Owner_age', 'Same_district', 'Default']]


# In[31]:


df.sample(10)


# In[32]:


df.shape


# In[33]:


df.columns


# ## Correlations

# In[34]:


df.corr()


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


plt.figure(figsize = (10,8))
sns.heatmap(df.corr(), cmap = 'RdYlGn', annot = True, annot_kws = {'fontsize': 7.5}, linewidths = 0.2)


# ## Transformation

# In[37]:


import pandas as pd


# In[38]:


for i in df.columns:
    print(i , df[i].dtype)


# In[39]:


df['payments'] = df['payments'].astype('float')


# In[40]:


df2 = df.copy()


# In[41]:


# To seperate numerical and categorical data

cat_columns = []
num_columns = []
bool_columns = []

for i in df2.columns:
    if df2[i].dtype == 'object':
        cat_columns.append(i)
    else:
        if df2[i].dtype =='bool':
            bool_columns.append(i)
        else:
            num_columns.append(i)
        
print(cat_columns,'\n\n', num_columns, '\n\n', bool_columns)


# In[42]:


df2 = pd.get_dummies(df2)


# In[43]:


df2.head()


# In[44]:


df2.columns.tolist()


# In[45]:


df2.iloc[:,-6:]


# In[46]:


from sklearn.preprocessing import OneHotEncoder


# In[47]:


df3 = df.copy()


# In[48]:


# To seperate numerical and categorical data

cat_columns = []
num_columns = []
bool_columns = []

for i in df3.columns:
    if df3[i].dtype == 'object':
        cat_columns.append(i)
    else:
        if df3[i].dtype =='bool':
            bool_columns.append(i)
        else:
            num_columns.append(i)
        
print(cat_columns,'\n\n', num_columns, '\n\n', bool_columns)


# In[49]:


OHE = OneHotEncoder()

OHE.fit_transform(df3[cat_columns])


# In[50]:


from sklearn.compose import ColumnTransformer

# cat_cols = []
# num_cols = []

# for i in df3.columns:
#     if df3[i].dtype == 'object' or df3[i].dtype =='bool':
#         cat_cols.append(i)
#     else:
#         num_cols.append(i)

# print(cat_cols,'\n\n', num_cols)

ColumnTransformer([('cat', OneHotEncoder(), cat_columns)]).fit_transform(df3)


# In[51]:


pd.get_dummies(df3[cat_columns]).iloc[:, :]


# ### Profile report

# In[52]:


# from pandas_profiling import ProfileReport

# profile = ProfileReport(df)
# profile.to_file(output_file = 'Profile_report.html')


# In[53]:


# # To veiw profile report in notebook
# profile.to_notebook_iframe()


# In[54]:


from sklearn.preprocessing import MinMaxScaler


# In[55]:


df_num = pd.DataFrame(MinMaxScaler().fit_transform(df[num_columns]))


# In[56]:


df_num.head()


# In[57]:


df_cat = pd.get_dummies(df[cat_columns]).reset_index(drop = True)


# In[58]:


df_cat.head()


# In[59]:


df_bool = df[bool_columns].reset_index(drop = True)
df_bool.head()


# In[60]:


# Dataframe for machine learning

df_ml = df_num.join(df_cat).join(df_bool)


# In[61]:


df_ml


# In[154]:


from sklearn.compose import ColumnTransformer


# In[155]:


column_transformer = ColumnTransformer([('num_trans', MinMaxScaler(), num_columns),
                                        ('cat_trans', OneHotEncoder(), cat_columns)], 
                                       remainder = 'passthrough'
                                      )


# In[156]:


df_ml = pd.DataFrame(column_transformer.fit_transform(df3))
df_ml.head()


# In[157]:


print(len(df3.columns), df3.columns, '\n\n')
print(len(df_ml.columns), df_ml.columns)


# In[158]:


# To add feature names to the transformed dataframe

df_ml.columns = column_transformer.get_feature_names_out()
df_ml.head()


# ## Upscalling minority class data (default cases)

# In[159]:


# Library package for upscalling imbalanced data

# pip install imbalanced-learn


# In[160]:


df_ml_2 = df_ml.copy()


# In[161]:


X = df_ml_2.iloc[:, :-1]
Y = df_ml_2.iloc[:, -1]


# In[162]:


from imblearn.over_sampling import ADASYN


# In[174]:


oversample = ADASYN(sampling_strategy = 0.8)


# In[175]:


x, y = oversample.fit_resample(X, Y)


# In[176]:


print(Y.value_counts(), '\n\n', y.value_counts())


# In[177]:


355/434


# # Modelling

# In[178]:


from sklearn.model_selection import train_test_split


# In[179]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, stratify = y, random_state = 6)


# ### KNN model

# In[180]:


from sklearn.neighbors import KNeighborsClassifier


# In[181]:


knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)


# In[183]:


y_pred = knn.predict(x_test)


# In[184]:


from sklearn.metrics import confusion_matrix, classification_report


# In[185]:


confusion_matrix(y_test, y_pred)


# In[186]:


class_report = classification_report(y_test, y_pred)
print(class_report)


# ### Linear regression model

# In[187]:


from sklearn.linear_model import LogisticRegression


# In[188]:


lr = LogisticRegression()
lr.fit(x_train, y_train)


# In[189]:


y_pred = lr.predict(x_test)


# In[190]:


confusion_matrix(y_test, y_pred)


# In[191]:


class_report = classification_report(y_test, y_pred)
print(class_report)


# ### Cross-validation and Kfold

# In[192]:


from sklearn.model_selection import cross_val_score

cross_val_score(lr, x_train, y_train, cv = 5, scoring = 'accuracy')


# In[203]:


# gives prection of each element in the input when it was in test split in cross-validation

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(lr, x, y, cv = 5)
confusion_matrix(y, y_pred)


# In[202]:


from sklearn.metrics import accuracy_score


# In[204]:


accuracy_score(y, y_pred)


# In[196]:


from sklearn.model_selection import StratifiedKFold


# In[197]:


kfold = StratifiedKFold(n_splits = 5, shuffle = True)

for train_index, test_index in kfold.split(x, y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print(confusion_matrix(y_test, y_pred), '\n')


# ### GridsearchCV

# In[198]:


from sklearn. model_selection import GridSearchCV


# In[199]:


# hyperparameters tuning

params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8]}
          
clf = GridSearchCV(KNeighborsClassifier(), param_grid = params, 
                   cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 6), 
                   scoring = 'f1')
          
clf.fit(x_train, y_train)
          
print(clf.best_params_)


# In[200]:


clf.cv_results_


# In[205]:


def ml(model, folds, params = {}):
    
    global clf
    
    clf = GridSearchCV(model, param_grid = params, 
                       cv = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 6))
    
    clf.fit(x_train, y_train)


# In[206]:


ml(LogisticRegression(),5)                       

clf.cv_results_


# ### Random Forest model

# In[207]:


from sklearn.ensemble import RandomForestClassifier


# In[211]:


params = {'max_depth': [10, 20, 25, 30, 35, 40],
          'criterion': ['gini', 'entropy']}

ml(RandomForestClassifier(), 3, params)


# In[212]:


clf.best_score_


# In[213]:


clf.best_params_


# In[225]:


clf2 = RandomForestClassifier(criterion = 'entropy', max_depth = 25)
clf2.fit(x_train, y_train)
y_pred = clf2.predict(x_test)
accuracy_score(y_test, y_pred)


# ## Threshold tunning

# In[143]:


lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(classification_report(y_test, y_pred))


# In[145]:


# Probability for the predicting each instance as 1

pred_prob_fr_posit = lr.predict_proba(x_test)[:,1]


# ### ROC_AUC curve

# In[103]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[104]:


roc_auc_score(y_test, pred_prob_fr_posit)


# In[105]:


fpr, tpr, threshold = roc_curve(y_test, pred_prob_fr_posit)


# In[107]:


sensitivity = tpr
specivity = 1- fpr

gmean = (sensitivity*specivity)**0.5


# In[109]:


threshold[gmean.argmax()]


# In[108]:


import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.plot(fpr[gmean.argmax()], tpr[gmean.argmax()], marker = 'o')


# In[110]:


# Youden's J statistics

Y_stat = tpr-fpr
threshold[Y_stat.argmax()]


# In[111]:


plt.plot(fpr, tpr)
plt.plot(fpr[Y_stat.argmax()], tpr[Y_stat.argmax()], marker = 'o')


# ### Precision- Recall curve

# In[112]:


from sklearn.metrics import precision_recall_curve


# In[115]:


precision, recall, threshold = precision_recall_curve(y_test, pred_prob_fr_posit)

fscore = 2*precision*recall/(precision+recall)


# In[117]:


fscore.argmax()


# In[116]:


plt.plot(precision, recall)
plt.plot(precision[fscore.argmax()], recall[fscore.argmax()], marker = 'o')


# ### Checking scores at different thresolds for the model

# In[131]:


import numpy as np

threshold = np.arange(0, 1.1, 0.1)
threshold


# In[132]:


from sklearn.metrics import accuracy_score, f1_score


# In[135]:


lr.fit(x_train, y_train)

# To check score on different thresholds
# lr.predict_proba(x_test)[:, 1] >= i).astype('int')

for i in threshold:
    print(confusion_matrix(y_test, (lr.predict_proba(x_test)[:, 1] >= i).astype('int')))


# ### XGboost model

# In[287]:


# pip install xgboost


# In[288]:


from xgboost import XGBClassifier


# In[289]:


# Running basic models on the training set

for i in [KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), XGBClassifier()]:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    print(f1_score(y_test, y_pred), i)


# In[290]:


xgb = XGBClassifier()


# In[291]:


xgb.fit(x_train, y_train)


# In[292]:


y_pred = xgb.predict(x_test)

confusion_matrix(y_test, y_pred)


# In[299]:


params = {'max_depth': [4, 5, 6, 7, 8],
         'learning_rate': [0.05, 0.01, 0.1],
         'gamma': [0, 0.25, 1],
         'reg_lambda': [0, 1, 10],
         'subsample': [0.5, 0.75, 1],
         'colsample_bytree': [0.5, 0.75, 1]}

ml(XGBClassifier(), 4, params)


# In[300]:


clf.cv_results_


# In[301]:


clf.best_params_


# In[302]:


xgb = XGBClassifier(max_depth = 5, colsample_bytree = 1, gamma = 0, learning_rate = 0.1, reg_lambda = 10, subsample = 0.75)


# In[303]:


xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
confusion_matrix(y_test, y_pred)


# In[307]:


xgb = XGBClassifier(max_depth = 7)
xgb.fit(x_train, y_train)

# To check score on different thresholds
threshold = np.arange(0.1, 0.9, 0.1)

for i in threshold:
    print(f1_score(y_test, (xgb.predict_proba(x_test)[:, 1] >= i).astype('int')), i)


# In[ ]:




