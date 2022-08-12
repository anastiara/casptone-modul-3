# Bank Marketing Cmapaign
by Anastiara Adina Restu - JCDS 1702 002

## Problem Statement

Marketing campaign is a strategy to enhance business. One of the approach to sell the products/services is by telemarketing. 
Telemarketing is a marketing approach operationalized through a contact center. 
Representatives from the contact center will contact the customers by phone offer the products. The cost of telemarketing is variative, based on per hour or per lead. 
This approach is commonly done by banking service companies to offer their products, for example long-term deposit. 
To determine whether a campaign is successful or not is based on how many customers decide to subscribe the long-term deposit. 

However, if we talk about the cost of the telemarketing, sometimes it can be costly if the marketing 
team just call all the customers without knowing that they might be not interested to subscribe to the 
long-term deposit also waste time as well. Besides, this strategy makes the customers uncomfortable, especially the ones who don't 
want to subscribe to our product. To overcome with the campaign cost waste and customers' complaint, it is needed a prediction on how many
customers are more likely to subscribe the deposit so we can calculate the campaign cost from the prediction.

## Objectives
To achieve the effiency of the telemarketing cost, we need to predict deposit subscription on our customers based on the available data. 
From the prediction, we can calculate the telemarketing cost before we do prediction and after the prediction.

## Dataset
The dataset is obtained from [this dataset](https://drive.google.com/file/d/1PQTTWgITANg5Av-1Ot28KCIHVyFaCmUK/view) about direct marketing campaigns of a Portuguese banking institution and based from [[1]](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset). In further analysis and modeling, the currency of this dataset will be explained in EUR.

## Analytic Approach
### Modeling
Before we do our modeling, we need to do data cleaning followed by EDA (explanatory data analysis) on our data to get insights about our customers details. Then, we split our data with train set 80% and test set 20%. Next, we will train and test our data with some machine model learning such as Decision Tree, Logistic Regression, KNN, Random Forest, AdaBoost, Gradient Boost, and XGB. After getting the top 3 best performance score from the benchmark models, we do hyperparameter tuning for each model to improve the model performance. Last, we then train and test our model alongside with the cost prediction to see what model predicts the most efficient marketing cost.

### Marketing Cost Calculation
To be able understand our prediction, we can use confusion matrix to explain the customers' category

![](https://github.com/anastiara/casptone-modul-3/blob/main/Images/2022-08-08-20-25-29.png)

TARGET :
- Positive class : 'yes' for subscribe deposit
- Negative class : 'no' for not subscribe deposit

Confusion Matrix term
- TP: number of customers who are **PREDICTED SUBSCRIBED** deposit are **ACTUALLY SUBSCRIBED**
- TN: number of customers who are **PREDICTED NOT SUBSCRIBED** deposit are **ACTUALLY NOT SUBSCRIBED**
- FP: number of customers who are **PREDICTED SUBSCRIBED** deposit is **ACTUALLY NOT SUBSCRIBED**
- FN: number of customers who are **PREDICTED NOT SUBSCRIBED** deposit is **ACTUALLY SUBSCRIBED**


In this case, the marketing team knows that [[2]](https://github.com/goncaloggomes/cost-prediction/blob/master/ML_fullproject_bankmktcampaign.ipynb):
- For each customer predicted subscribed to our deposit but actually they don't subscribe, the bank will cost 2000 EUR
- For each customer predicted not subscribed to our deposit but actually they subscribe, the bank will cost 500 EUR

Both cases have the consquence of wasting money and our goal will not be achieved. However, the consequences of False Positive will costly more. So, we will calculate those cases based on our prediction through confusion matrix from each model and try to reduce the FP numbers as low as possible in order to make the marketing cost more efficient (cheaper).

## Metrics Evaluation

There are two metrics in this case

1. ROC-AUC score: used to determine whether a model is good or not to distinguish positive class and negative class
2. Cost Reduction: used to determine a model is able to make the efficient cost. The higher cost reduction, the better performance model to make the efficient cost.

## Feature Description

**Bank client data:**

1. age: (numeric)
2. job: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. housing: has housing loan? (categorical: 'no','yes','unknown')
4. loan: has personal loan? (categorical: 'no','yes','unknown')
5. balance: Balance of the individual.

**Related with the last contact of the current campaign:**

6. contact: contact communication type (categorical: 'cellular','telephone','unknown')
7. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

**Other attributes:**

8. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
9. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted))
10. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

**Output variable (desired target):**

11. deposit : as the client subscribed a term deposit? (binary: 'yes','no')

## EDA Insights
1. From our cleaned dataset, now our `age` customers range from 18-95 years old and customers' `balance` range from 0-4017 EUR.
2. Our customers tend to deposit if they are contacted only once, if we contact them more than once, they possibly will not subscribe to our deposit product. Also, if they are never contacted before (based on `pdays` feature), they also tend to subscribe.
3. Most customers who subscribe the deposit products whose jobs are management, student, and retired. We should expand our approach to student and retired customers because we have too much approached on management customers. 
4. The optimal months to contact our customers are in Febrary, April, September, October, and December.
5. Most of our customers who subscribe to our deposit product are the ones who don't have housing loan and personal loan.
6. Customers whose poutcome status is success also will subscribe to our deposit product.
7. Contact feature doesn't give any insights about our customers' subscription. However, we will keep this feature for modeling process
8. `campaign` feature will not be used from modeling because this feature is not obtained before campaign
9. Our data is balanced. So for the modeling we don't have to do data balancing.

## Modeling
In this part, we will do our modeling for the dataset. The steps are:
- Data Preprocessing
    - Data splitting (train and test set)
    - Data Encoding and Scaling
- Modeling
    - Cross Validation of Benchmark Models
    - Predict to Test Set
    - Choose Best 3 models based on the performance score (accuracy and ROC AUC)
    - Hyperparameter Tuning the 3 models
    - Predict to Test set and calculate the marketing cost
- Choose the best model with the lowest marketing cost
- Recommendation

### Data Preprocessing
*Data Splitting: Splitting Dataset into training set (80%) and test set (20%)
*Data Encoding and Scaling: 
  * OneHotEncoder = `loan`, `housing`,`contact`,`poutcome`. This is done  because the unique values of these features are only 2-3 values. So, it will be more effiecient if I encode these features with OneHotEncoder
  *  BinaryEncoder = `job`,`month`. This is done because their unique vales are more than 3 values. Eventough this feature will hard to explain, but at least it will make the computation faster.
  * StandardScaler = This scaling feature is used to scale the numeric features. This scaler standardizes a feature by substracting the mean and then scaling to unit variance. The feature `age` and `balance` will be scaled by this.

### Define Benchmark Model
To obtain the best model, we will compare these classification models to know what model perform the best. There are KNN, Logistic Regression, Decision Tree, Random Forest, Ada Boost, Gradient Boost, and XGB.

*Cross Validation Result

Cross Validation is performed to avoid overfitting of a model and determine what models have good performances based on the mean and standard value. Cross Validation is a resampling method that uses different portions of the data to test and train a model on different iterations.

![Training Benchmark Model](https://github.com/anastiara/casptone-modul-3/blob/main/Images/Screenshot%202022-08-12%20202446.png)
![Testing Benchmark Model](https://github.com/anastiara/casptone-modul-3/blob/main/Images/Screenshot%202022-08-12%20202503.png)

From the benchkmark model performance score, we have our top models based on ROC-AUC score and precision score: Random Forest, Gradient Boost and Ada Boost. Therefore, we are going to choose Ranfom Forest, Gradient Boost and Ada Boost to enhance their performance by tuning then we also calculate the cost based on FP and FN score.

## Marketing Cost Calculation
Before we are doing hyperparameter tuning and predict to the new tuned models, we need to calculate the marketing cost if no model applied to the case. To do that, we simulate two cases: 
- if customers where considered subscribed to the deposit product but actually don't subscribe to the deposit product
- if customers where considered not subscribed to the deposit product but actually subscribe to the deposit product

Then, we calculate the cost of FP and FN based the information mentioned and compare which total cost will more costly. We want to prove that our stance about the first case will more costly. To do this, we combine the calculation from [[2]](https://github.com/goncaloggomes/cost-prediction/blob/master/ML_fullproject_bankmktcampaign.ipynb) and [[3]](https://www.kdnuggets.com/2018/10/confusion-matrices-quantify-cost-being-wrong.html) and modify them based on our dataset.

- if customers where considered subscribed to the deposit product but actually don't subscribe to the deposit product will cost 1041.7 EUR
- if customers where considered not subscribed to the deposit product but actually subscribe to the deposit product wiil cost 239.58n EUR

From our observation above, we know that customers where considered subscribed to the deposit product but actually don't subscribe to the deposit product will more expensive. It proves our problem statement. Next step is we will look for the models who can lower the False Positive numbers so that we can find the most efficient marketing cost.

## Hyperparameter Tuning
### Random Forest
- Best Parameters:{'scaler': StandardScaler(), 'method__n_estimators': 200, 'method__min_samples_split': 10, 'method__min_samples_leaf': 4, 'method__max_depth': 60, 'method__bootstrap': True}
- Best train ROC AUC score: 0.7547888602402649
- Best test ROC AUC score : 0.7437538002321596
- Marketing cost predicted by model : 304.02 EUR Average per customer
- the cost difference by model is 737.68 EUR average savings per customer
- Cost Reduction from the cost without model is 70.82%



### AdaBoost
- Best Parameters:{'scaler': StandardScaler(), 'method__n_estimators': 96, 'method__learning_rate': 0.1}
- Best train ROC AUC score: 0.7321735490156287
- Best test ROC AUC score : 0.72823733716581
- Marketing cost predicted by model : 306.29 EUR Average per customer
- the cost difference by model is  735.41 EUR average savings per customer
- Cost Reduction from the cost without model is 70.6%

### Gradient Boost
- Best Parameters:{'scaler': StandardScaler(), 'method__n_estimators': 400, 'method__min_samples_split': 200, 'method__min_samples_leaf': 70, 'method__max_depth': 11, 'method__learning_rate': 0.01}
- Best train ROC AUC score: 0.7511258979354815
- Best test ROC AUC score : 0.7391820057855656
- Marketing cost predicted by model : 276.72 EUR Average per customer
- the cost difference by model is  764.97 EUR average savings per customer
- Cost Reduction from the cost without model is 73.44%

### ROC-AUC
![ROC-AUC Curves](https://github.com/anastiara/casptone-modul-3/blob/main/Images/ROCAUC.png)

### Final Results
![Final Result](https://github.com/anastiara/casptone-modul-3/blob/main/Images/final.png)

From the hyperparameter tuning and prediction of marketing cost calculation above, we got Gradient Boost as our best model to predict the cost effieciency. Gradient Boost shows that its total cost is the lowest with the highest cost saving. It seems like best ROC-AUC score is Random Forest model, but the cost calculation from the model is not as good as Gradient Boost. In this case, we try to lower the FP numbers and we know that it is better to use Precision as our metrics. We get the best score but since our data is quite balanced, we are not sure that using precision as main metrics will be good enough. To overcome with this, we only care about the confusion matrix that each model produces and our main goal of doing prediction is to lower the FP number.

We use ROC-AUC metrics to only see how good our models to distinguish between positive and negative classes. Based on this [source](https://github.com/goncaloggomes/cost-prediction/blob/master/ML_fullproject_bankmktcampaign.ipynb), with the academic scoring system as

- 0.9 - 1 = excellent (A)
- 0.8 - 0.9 = Good (B)
- 0.7 - 0.8 = reasonable (C)
- 0.6 - 0.7 = weak (D)
- 0.5 - 0.6 = terrible (E)

Overall, our models' ROC-AUC score are at reasonable level, meaning that our models can classify the positive classes and negative classes.

### Feature Importances

## Conclusion and Recommendation
*Marketing Cost Efficiency

Based on our modeling, we have compared Gradient boost and Ada Boost models. Our goal here is to reduce FP number since its consequence is costly to the bank. So, we compared the models also with the confusion matrix to obtain the FP and FN results based on the models. The results show that Gradient Boost is the best performance with this dataset because it has the lowest total marketing cost (276.62 EUR) and highest saving cost (764.97 EUR) from the baseline total cost 1401.7 EUR.

*Recommendation

Things that should be prioritized by the bank:
- Target the deposit product to wider prospective customers such as students and retired customers, since the marketing team doesn't contact them much. If we still approach management customers, the chance of not getting new subscribers is higher because from our dataset, the difference between no deposit vs deposit of management customers is relatively small.
- Based on our data, the customers who don't have housing loan and with relatively high balance also will tend to subscribe to our deposit product.
- The campaign might be better in February, April, September, October,and December since the in those month we can get many customers to subscribe the deposit product.
- The customers that have high balance and the the poutcome is success will tend to subscribe the deposit
- Having a questionnaire after contacting the customers is a good way to know our customers so that we can get better insights about our customers and better prediction.
