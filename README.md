Loan default prediction

The objective of this self-project was to automate the classification of a given loan as either default or non-default, which then can be used for assistance in loan disbursal or loan amount recovery. 

At first, the data was stored in a local MySQL server to simulate the data retrieval process employed in the companies. The connection was set up in the python workspace for accessing data from the database using SQL queries. Then, a detailed EDA was performed to filter the features to be used for Machine Learning modelling. Different features were visually explored by comparing good and bad loan classes using Matplotlib. After finalizing the features, the data was further analyzed and transformed as per the requirement. Categorical data were One-Hot encoded, and numerical data were scaled using Min-max scaler.

The processed data was split into train and test datasets for the modelling. The minority class was upscaled with ADASYN to compensate for the imbalance in the dataset. At first, different classification models with default parameters such as KNN, logistic regression, Random Forest and XGBoost were fitted on the training dataset to pick the apt one for tuning. Metrics such as accuracy, precision, recall and confusion metrics were used for the evaluation of the models. XGBoost outperformed the other models; hence it was further tuned for the improvements. Techniques such as cross-validation, gridsearch and threshold tuning were employed to further improve the model's classification capabilities.

At last, the final model was saved as a pickle file and a simple webapp was developed using Streamlit. In the webapp, interactive options for the model's parameter were provided to train own model. An interactive field for model input data was given to run it on either the tuned or trained model.

![Alt text](https://github.com/shubhankarm01/Loan-default-prediction/blob/main/Project%20technical%20architecture.jpg?raw=true "Project technical architecture")
