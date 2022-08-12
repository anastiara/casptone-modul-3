# Bank Marketing
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

![](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.dataschool.io%2Fcontent%2Fimages%2F2015%2F01%2Fconfusion_matrix2.png&imgrefurl=https%3A%2F%2Fwww.dataschool.io%2Fsimple-guide-to-confusion-matrix-terminology%2F&tbnid=ZH8M9hQ__uqDXM&vet=12ahUKEwj0xpPCqcH5AhVMNLcAHQ1XDtAQMygDegUIARDfAQ..i&docid=j5uYZDebx1HmNM&w=487&h=278&q=confusion%20matrix&ved=2ahUKEwj0xpPCqcH5AhVMNLcAHQ1XDtAQMygDegUIARDfAQ)

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
