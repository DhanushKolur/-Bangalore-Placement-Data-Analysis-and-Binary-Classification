# -Bangalore-Placement-Data-Analysis-and-Binary-Classification

 Placement Data Analysis
Page | 1
A PROJECT REPORT ON
“Bangalore Placement Data Analysis
and Binary Classification”
 DHANUSH K
 TLS21A2554
 Placement Data Analysis
Page | 2
INDEX
Topic Page no
Abstract 3
Introduction 4
Discussion on Tasks 5-6
Python Code 7-35
Conclusion 36
 Placement Data Analysis
Page | 3
ABSTRACT
Placement of students in appropriate jobs is very important to college 
recruitment or placement committee of a university. It is crucial to identify 
parameters and discover trends that improve a student's chances of getting 
a suitable job. The placement team selects students based on company 
criteria like education history and CGPA. After that the students have to 
clear the company's evaluation levels, namely aptitude, technical and 
personal interview to get the job. This would provide insight into improving 
the overall placement process and highlight areas that need attention. In an 
educational institution, the success of its imbibing model is usually 
measured using the career opportunities of the graduates. Hence, the 
placement data has an important relevance for the future plan and growth. 
Quite a good amount of information can be gained by the entire stakeholder 
by carefully looking at this information. In educational institutes a huge 
amount of data is being generated. The produced data does not provide 
enough information which obscures important details of data that may help 
in better understanding of available data and its utility. If the data is analyzed 
efficiently, it can provide many insights, specific information regarding 
various facets of data which can be useful in a multiple ways. Analysis of 
data plays a very important role in understanding of information from a given 
set of data. Analysis of data can be performed using various data mining 
algorithms which help them to take decisions or arrive at a conclusion with 
the help of available data
 Placement Data Analysis
Page | 4
INTRODUCTION
The college placement community thrives on students finding suitable jobs 
and recruiters finding students that add value to their company. Every year, 
thousands of engineering students sit for college placements find jobs. The 
college administration helps students navigate these waters by providing 
the necessary tools. They provide information and train students to become 
eligible for jobs. It is paramount that this process yields results that add 
value to the student and the company. Thus, we analysed the placement 
data of a university to gain insights into process and areas that need focus. 
This study targeted to find the most impactful companies which can be get 
the higher position in the priority list of campus placement. It is 
inconvenient to track the trends or patterns of the results and the 
placements, making it more difficult to change the policies or any other 
change required for improvement. This happens due to varying data 
formats or faulty data entry or improper data updating. Analysis and 
prediction often becomes an arduous task in absence of the right mining 
techniques and technology. Prediction of students' performance is very 
important in academic environments. In present day scenario where 
education has been privatized and where there is ceaseless competition 
amongst students and institutes, there is a need to be more organized and 
have the ability to make sound decisions and make constructive changes. 
Due to these challenges, the management is meeting the diverse needs of 
utilizing a decision support system for facing increased complexity in 
academic processes and college policies to keep their institution at a 
respectable position in the contest. This system helps in continuous 
improvements in operational strategies based on accurate, timely and 
consistent information.
 Placement Data Analysis
Page | 5
TASKS
• DATA ACQUISITION AND CLEANING
• DATA VISUALIZATION
• DATA MODELLING
• TESTING
• COMPARISON AND MEASUREMENT
 Placement Data Analysis
Page | 6
DATA ACQUISITION AND CLEANING
This data set consists of Placement data of students in a XYZ campus. 
It includes secondary and higher secondary school percentage and 
specialization. It also includes degree specialization, type and Work 
experience and salary offers to the placed students. 
Input variables based on physicochemical tests:
• sl_no 
• gender 
• ssc_p 
• ssc_b 
• hsc_p 
• hsc_b 
• hsc_s 
• degree_p 
• degree_t 
• workex 
• etest_p
• specialisation
• mba_p
Output variable based on sensory data:
• status (Placed or Not Placed)
 Placement Data Analysis
Page | 7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
data = pd.read_csv("data/Placement_Data_Full_Class.csv")
#Remove Serial Number
data.drop("sl_no", axis=1, inplace=True)
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
 Data Analysis and Machine Learning on Campus 
Placement Data
Explatory Data Analysis
Prediction of wheather student gets placed or not (Binary Classification) 
Determining characteristics affecting placement
Predition of Salary secured by a student (Regression) 
Determining characteristics affecting salary
Common Questions
Does GPA affect placement?
Does Higher Secondary School's Percentage still affect campus placement? 
Is work experience required for securing good job?
What factor affect the salary?
Let's find out
Library Imports
Loading Data
Exploratory Data Analysis
Pandas Profiler's Interactive Report
 Placement Data Analysis
Page | 8
data.gender.value_counts()
# Almost double
Summarize dataset: 0%| | 0/27 [00:00<?, ?it/s]
Generate report structure: 0%| | 0/1 [00:00<?, ?it/s] 
Render widgets: 0%| | 0/1 [00:00<?, ?it/s]
VBox(children=(Tab(children=(Tab(children=(GridBox(children=(VBox(children=(G 
ridspecLayout(children=(HTML(valu…
67 Missing values in Salary for students who didn't get placed. NaN Value needs to be filled.
Data is not scaled. Salary column ranges from 200k-940k, rest of numerical columns are percentages. 300k 
at 75th Percentile goes all the way up to 940k max, in Salary (high skewnwss). Thus, outliers at high salary 
end.
Exploring Data by each Features
Feature: Gender
Does gender affect placements?
Out[8]:
M 139
F 76
Name: gender, dtype: int64
/Users/local/miniconda3/envs/bayesian/lib/python3.8/site-packages/seaborn/_de 
corators.py:36: FutureWarning: Pass the following variable as a keyword arg:
x. From version 0.12, the only valid positional argument will be `data`, and
passing other arguments without an explicit keyword will result in an error 
o r misinterpretation.
warnings.warn(
sns.countplot("gender", hue="status", data=data) 
plt.show()
profile.to_widgets()
 Placement Data Analysis
Page | 9
plt.figure(figsize =(18,6))
sns.boxplot("salary", "gender", data=data) 
plt.show()
Insights
We have samples of 139 Male studets and 76 Female students.
30 Female and 40 Male students are not placed. Male students have comparatively higher placemets. 
More outliers on Male -> Male students are getting high CTC jobs.
Male students are offered slightly greater salary than female on an average.
#This plot ignores NaN values for salary, igoring students whoare not placed
sns.kdeplot(data.salary[ 
data.gender=="M"]) 
sns.kdeplot(data.salary[ 
data.gender=="F"]) plt.legend(["Male", 
"Female"])
plt.xlabel("Salary 
(100k)") plt.show()
 Placement Data Analysis
Page | 10
#Kernel-Density Plot
sns.kdeplot(data.ssc_p[ data.status=="Placed"])
sns.kdeplot(data.ssc_p[ data.status=="Not Placed"]) 
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Secondary Education Percentage") 
plt.show()
sns.countplot("ssc_b", hue="status", data=data) 
plt.show()
Does Secondary Education affect placements?
All students with Secondary Education Percentage above 90% are placed
All students with Secondary Education Percentage below 50% are not-placed
Students with good Secondary Education Percentage are placed on average.
 Placement Data Analysis
Page | 11
plt.figure(figsize =(18,6))
sns.boxplot("salary", "ssc_b", data=data) 
plt.show()
sns.lineplot("ssc_p", "salary", hue="ssc_b", data=data) 
plt.show()
• Board Of Education does not affect Placement Status much
Outliers on both, but students from Central Board are getting the highly paid jobs.
No specific pattern (correlation) between Secondary Education Percentage and Salary. 
Board of Education is Not Affecting Salary
Feature: hsc_p (Higher Secondary Education percentage), hsc_b (Board 
Of Education), hsc_s (Specialization in Higher Secondary Education)
 Placement Data Analysis
Page | 12
#Kernel-Density Plot
sns.kdeplot(data.hsc_p[ data.status=="Placed"])
sns.kdeplot(data.hsc_p[ data.status=="Not Placed"]) 
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Higher Secondary Education Percentage") 
plt.show()
sns.countplot("hsc_b", hue="status",
data=data) 
plt.show()
Overlap here too. More placements for percentage above 65%
Straight drop below 60 in placements -> Perntage must be atleast 60 for chance of being placed
 Placement Data Analysis
Page | 13
sns.countplot("hsc_s", hue="status",
data=data) plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "hsc_b", data=data) 
plt.show()
sns.lineplot("hsc_p", "salary", hue="hsc_b", data=data) 
plt.show()
Education Board again, doesn't affect placement status much
We have very less students with Arts specialization.
Around 2:1 placed:unplaced ratio for both Science and Commerse students
Outliers on both, board doesn't affect getting highly paid jobs. Highest paid job was obtailed by student from 
Central Board though.
 Placement Data Analysis
Page | 14
plt.figure(figsize =(18,6))
sns.boxplot("salary", "hsc_s", 
data=data) 
plt.show()
 High salary from both Central and Other.
High salary for both high and low percentage. 
Thus, both these feature doesnot affect salary.
We can't really say for sure due to only few samples of students with Arts Major, but they aren't getting 
good salaries.
 Commerse students have slightly better placement status.
 Placement Data Analysis
Page | 15
Student with Art Specialization surprisingly have comparatively low salary
Feature: degree_p (Degree Percentage), degree_t (Under Graduation 
Degree Field)
Does Under Graduate affect placements?
Overlap here too. But More placements for percentage 
above 65. UG Percentage least 50% to get placement
#Kernel-Density Plot
sns.kdeplot(data.degree_p[ data.status=="Placed"])
sns.kdeplot(data.degree_p[data.status=="NotPlaced) 
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Under Graduate 
Percentage") 
plt.show()
sns.lineplot("hsc_p", "salary", hue="hsc_s", data=data) 
plt.show()
 Placement Data Analysis
Page | 16
sns.countplot("degree_t", hue="status",
data=data) plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "degree_t", 
data=data) 
plt.show()
We have very less students with "Other". We cant make decision from few cases. 
Around 2:1 placed:unplaced ratio for both Science and Commerse students
Science&Tech students getting more salary on average
Management stidents are getting more highly paid dream jobs.
 Placement Data Analysis
Page | 17
Percentage does not seem to affect salary.
Commerce&Mgmt students occasionally get dream placements with high salary
Feature: workex (Work Experience)
Does Work Experience affect placements?
This affects Placement. Very few students with work experience not getting placed
sns.lineplot("degree_p", "salary", hue="degree_t", data=data) 
plt.show()
sns.countplot("workex", hue="status", data=data) 
plt.show()
 Placement Data Analysis
Page | 18
Outliers (High salary than average) on bith end but students with experience getting dream jobs
Average salary as well as base salary high for students with work experience.
Feature: etest_p (Employability test percentage)
High overlap -> It does not affect placement status much
More "Not Placed" on percentage 50-70 range and more placed on 80% percentage range
#Kernel-Density Plot
sns.kdeplot(data.etest_p[ data.status=="Placed"])
sns.kdeplot(data.etest_p[ data.status=="Not Placed"]) 
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Employability testpercentage") 
plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "workex", data=data) 
plt.show()
 Placement Data Analysis
Page | 19
sns.lineplot("etest_p", "salary", data=data) 
plt.show()
sns.countplot("specialisation", hue="status", data=data) 
plt.show()
This feature surprisingly does not affect placements and salary much
Feature: specialisation (Post Graduate Specialization)
In [30]:
This feature affects Placement status.
Comparitively very low not-placed students in Mkt&Fin Section
 Placement Data Analysis
Page | 20
sns.lineplot("mba_p", "salary", data=data) 
plt.show()
*More Highly Paid Jobs for Mkt&Fin students *
Feature: mba_p (MBA percentage)
Does MBA Percentage affect placements?
sns.boxplot("mba_p", "status", data=data) 
plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "specialisation", data=data) 
plt.show()
 Placement Data Analysis
Page | 21
data.drop(['ssc_b','hsc_b'], axis=1, inplace=True)
data.dtypes
# We have to encode gender,hsc_s, degree_t, workex, specialisation and status
MBA Percentage also deos not affect salary much
Feature Selection
Using Only following features (Ignoring Board of Education -> they didnt seem to have much effect)
Gender
Secondary Education percentage
Higher Secondary Education Percentsge 
Specialization in Higher Secondary Education 
Under Graduate Dergree Percentage
Under Graduation Degree Field 
Work Experience
Employability test percentage 
Specialization
MBA Percentage
Will compute feature importance later on.
Data Pre-Processing
Feature Encoding
Out[35]:
 Placement Data Analysis
Page | 22
#Lets make a copy of data, before we proceeed with specific problems
data_clf = data.copy() 
data_reg = data.copy()
# Library imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
gender object
ssc_p float64
hsc_p float64
hsc_s object
degree_p float64
degree_t object
workex object
etest_p float64 
specialisation object 
mba_p float64
status object
salary float64 
dtype: object
Problem Statement
Predicting If Students gets placed or not (Binary Classification Problem) 
Predicting Salary of Student (Regression Problem)
Binary Classification Problem
Decision Tree Based Models
Using Decision Tree based Algorithm does not require feature scaling, and works great also in 
presence of categorical columns without ONE_HOT Encoding
data["gender"] = data.gender.map({"M":0,"F":1})
data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2}) 
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["status"] = data.status.map({"Not Placed":0, "Placed":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
 Placement Data Analysis
Page | 23
# Seperating Features and Target
X = data_clf[['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex','etest_ 
y = data_clf['status']
dtree = DecisionTreeClassifier(criterion='entropy') 
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#Using Random Forest Algorithm
random_forest = RandomForestClassifier(n_estimators=100) 
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
accuracy_score(y_test, y_pred)
Dropping Salary Feature
Filling 0s for salary of students who didn't get placements would be bad idea as it would mean student 
gets placement if he earns salary.
Out[42]:
0.7230769230769231
precisi
on
rec
all
f1-
score
supp
ort
0 0.44 0.80 0.57 15
1 0.92 0.70 0.80 50
accuracy 0.72 65
macro
avg
0.68 0.75 0.68 65
weighted 
avg
0.81 0.72 0.74 65
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 Placement Data Analysis
Page | 24
rows = list(X.columns)
imp = pd.DataFrame(np.zeros(6*len(rows)).reshape(2*len(rows), 3)) 
imp.columns = ["Classifier", "Feature", "Importance"]
#Add Rows
for index in range(0, 2*len(rows), 2):
imp.iloc[index] = ["DecisionTree", rows[index//2], (100*dtree.feature_importances_[inde 
imp.iloc[index + 1] = ["RandomForest", rows[index//2], (100*random_forest.feature_impor
Out[45]:
0.7692307692307693
precisi
on
rec
all
f1-
score
supp
ort
0 0.50 0.73 0.59 15
1 0.91 0.78 0.84 50
accuracy 0.77 65
macro
avg
0.70 0.76 0.72 65
weighted 
avg
0.81 0.77 0.78 65
Feature Importance (Percentage)
Tree based algorithms can be used to compute feature 
importance Checking feature importance obtained from these:
plt.figure(figsize=(15,5))
sns.barplot("Feature", "Importance", hue="Classifier", data=imp) 
plt.title("Computed Feature Importance")
plt.show()
print(classification_report(y_test, y_pred))
 Placement Data Analysis
Page | 25
#One-Hot Encoding
X = pd.get_dummies(X)
colmunn_names = X.columns.to_list()
hsc_s -> Specialization in Higher Secondary Education
degree_t -> Under Graduation(Degree type)- Field of degree 
education specialisation -> Post Graduation(MBA)- Specialization
Field of study does not seem to affect much
Optionally we can remove these least important features and re-clssify data.
Binary Classification with Logistic 
Regression One Hot Encoding
Encoding Categorical Featu
X["specialisation"] = pd.Categorical(X.specialisation.map({0:"Mkt&HR", 
1:"Mkt&Fin"}))
:
Feature Scaling
 Placement Data Analysis
Page | 26
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression() 
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)
accuracy_score(y_test, y_pred)
Percentages are on scale 0-100
Categorical Features are on range 0-1 (By one hot encoding)
High Scale for Salary -> Salary is heavily skewed too -> SkLearn has RobustScaler which might work well 
here
Scaling Everything between 0 and 1 (This wont affect one-hot encoded values)
:
:
:
:
Out[55]:
0.8615384615384616
 Placement Data Analysis
Page | 27
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(logistic_reg).fit(X_test, y_test) 
eli5.show_weights(perm)
plt.figure(figsize=(30, 10))
plt.bar(colmunn_names , perm.feature_importances_std_ * 100) 
plt.show()
precisi
on
rec
all
f1-
score
supp
ort
0 0.84 0.73 0.78 22
1 0.87 0.93 0.90 43
accuracy 0.86 65
macro
avg
0.86 0.83 0.84 65
weighted 
avg
0.86 0.86 0.86 65
Computating Feature importance by Mean Decrease Accuracy (MDA)
Since Logistic Regression performed well, Lets run another method for determining fearure 
importance here.
From Feature Importance of Tree-based Algorithms and MDA we can conclude that:
Academic performance affects placement (All percentages had importantance) 
Work Experience Effects Placement
Gender and Specialization in Commerse (in higher-seondary and undergraduate) also has effect on 
placements.
print(classification_report(y_test, y_pred))
 Placement Data Analysis
Page | 28
#dropping NaNs (in Salary)
data_reg.dropna(inplace=True)
#dropping Status = "Placed" column
data_reg.drop("status", axis=1, inplace=True)
data_reg.head()
#Seperating Depencent and Independent Vaiiables
y = data_reg["salary"] #Dependent Variable
X = data_reg.drop("salary", axis=1) 
column_names = X.columns.values
Prediction of Salary (Regression Analysis)
Data Preprocessing
:
Out[60]:
ge
nd
er
ss
c_
p
hsc_p h
s
c
_
s
degr
ee_p
degr
ee_t
workex etest
_p
specialis
ation
m
ba
_p
0 0 67.
00
91.00 0 58.00 1 0 55.0 0 58.8
0
1 0 79.
33
78.33 1 77.48 1 1 86.5 1 66.2
8
2 0 65.
00
68.00 2 64.00 0 0 75.0 1 57.8
0
4 0 85.
80
73.60 0 73.30 0 0 96.8 1 55.5
0
7 0 82.
00
64.00 1 66.00 1 1 67.0 1 62.1
4
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
 Placement Data Analysis
Page | 29
#Scalizing between 0-1 (Normalization)
X_scaled = MinMaxScaler().fit_transform(X)
Feature Selection
** Not all features are significant. Thus, let's perform a feature selection procedure**
 Placement Data Analysis
Page | 30
#Selecting outliers
y[y > 400000]
# 9 records
Determining Least Significant Variable
The least significant variable is a variable which:
has the highest p-value
Removing it reduces R2 to lowest value compared to other features 
Removing it has least increment in residuals-sum-of-squares (RSS)
Outliers' Removal
Feature Selecton cannot perform well in presence of outliers. Lets identy and remove outliers before proceding
It is clear that very few students have salary greater than 400,000 (hence outliers)
Out[64]:
4 425000.0
39 411000.0
53 450000.0
77 500000.0
#PDF ofSalary 
sns.kdeplot(y) 
plt.show()
 Placement Data Analysis
Page | 31
#Removing these Records from data
X_scaled = X_scaled[y < 400000]
y = y[y < 400000]
95 420000.0
11
9
940000.0
15
0
690000.0
16
3
500000.0
17
4
500000.0
17
7
650000.0
Na
m
e:
salary, dtype: 
float64
:
1. termining Least Significant Variable by R2 Score
#PDF of Salary without outliers. Still skewed though
sns.kdeplot(y) 
plt.show()
 Placement Data Analysis
Page | 32
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# Lets see the top 5 most significant features
top_n = 5
sfs.get_metric_dict()[top_n]
Out[74]:
{'feature_idx': (0, 3, 5, 7, 9),
'cv_scores': array([-0.09997564, -0.11551795, 0.14652782, 0.19241391, -0.1
9535134,
-0.13235138, -0.03896556, 0.3116134 , 0.13836643, 0.07020936]),
'avg_score': 0.027696903219017098,
'feature_names': ('0', '3', '5', '7', '9'),
'ci_bound': 0.11802751969012418,
'std_dev': 0.15891404557580255,
'std_err': 0.052971348525267505
linreg = LinearRegression()
sfs = SFS(linreg, k_features=1, forward=False, scoring='r2',cv=10) 
sfs = sfs.fit(X_scaled, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Backward Elimination') 
plt.grid()
plt.show()
#From Plot its clear that, many features actually decrease the performance
 Placement Data Analysis
Page | 33
#Select these Features only
X_selected = X_scaled[: ,top_n_indices] 
lin_reg = LinearRegression()
lin_reg.fit(X_selected, y)
y_pred = lin_reg.predict(X_selected)
print(f"R2 Score: {r2_score(y, y_pred)}")
print(f"MAE: {mean_absolute_error(y, y_pred)}")
Most Significant 5 Features: gender
hsc_s
degree_t 
etest_p 
mba_p
R2 Score: 0.1101660718969637
MAE: 30630.128295211565
This is the best I could do with Linear Regression
2. Determining Least Significant Variable by P-Value
If the base model gives 0.7 R2 score and the model without a feature gives 0.75 R2 score, we 
cannot conclude that feature makes the difference, as the score may vary in another trial; in 10 
trials the R2 score might change in +/- 0.05. However, if model only varies in +/- 0.01, we can 
then say that removing a feature made the model better.
Our null hypothesis is that there is no difference between the two samples of R2 scores.
P-value is the probability that you would arrive at the same results as the null hypothesis. 
One of the most commonly used p-value is 0.05. If the calculated p-value turns out to be less 
than 0.05, the null hypothesis is considered to be false, or nullified (hence the name null 
hypothesis). And if the value is greater than 0.05, the null hypothesis is considered to be true.
For a feature, a small p-value indicates that it is unlikely we will observe a relationship 
between the predictor (feature) and response (salary in our case) variables due to chance.
Thus, we start with all features. We compute the P-values. We eliminate frature with highest 
p-value until p- values of all features reach below threshold: 0.05.
 Placement Data Analysis
Page | 34
Out[78]:
OLS Regression Results
Dep. Variable: y R-squared: 0.123
Model: OLS Adj. R-squared: 0.052
Method: Least Squares F-statistic: 1.722
Date: Sat, 15 May 2021 Prob (F-statistic): 0.0829
Time: 18:54:08 Log-Likelihood: -1608.4
No. Observations: 134 AIC: 3239.
Df Residuals: 123
Df Model: 10
Covariance Type: nonrobust
BIC: 3271.
coef std err t P>|t| [0.025 0.975]
const 2.625e+05 1.28e+04 20.498 0.000 2.37e+05 2.88e+05
gender -1.784e+04 8299.998 -2.149 0.034 -3.43e+04 -1406.775
ssc_p -116.6148 2.04e+04 -0.006 0.995 -4.04e+04 4.02e+04
hsc_p -1.842e+04 2.13e+04 -0.864 0.389 -6.06e+04 2.38e+04
hsc_s -2.775e+04 1.58e+04 -1.761 0.081 -5.9e+04 3444.983
degree_p -9885.6991 2.25e+04 -0.438 0.662 -5.45e+04 3.47e+04
degree_t 3.947e+04 1.69e+04 2.340 0.021 6077.584 7.29e+04
workex -7748.2212 7673.070 -1.010 0.315 -2.29e+04 7440.151
etest_p 1.839e+04 1.43e+04 1.286 0.201 -9906.447 4.67e+04
specialisatio
n
2457.2424 8013.710 0.307 0.760 -1.34e+04 1.83e+04
mba_p 3.704e+04 2.11e+04 1.756 0.082 -4717.648 7.88e+04
Omnibus: 10.852 Durbin-Watson: 1.965
Prob(Omnib
us):
0.004 Jarque-Bera (JB): 11.041
Skew: 0.661 Prob(JB): 0.00400
Kurtosis: 3.477 Cond. No. 12.9
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# Step 1: With all Features 
model = sm.OLS(y, X_scaled) 
results = model.fit()
results.summary()
 Placement Data Analysis
Page | 35
Out[79]:
 OLS Regression Results
Dep. Variable: y R-squared: 0.123
Model: OLS Adj. R-squared: 0.059
Method: Least Squares F-statistic: 1.929
Date: Sat, 15 May 2021 Prob (F-statistic): 0.0536
Time: 18:54:08 Log-Likelihood: -1608.4
No. Observations: 134 AIC: 3237.
Df Residuals: 124
Df Model: 9
Covariance Type: nonrobust
BIC: 3266.
coef std err t P>|t| [0.025 0.975]
const 2.625e+05 1.2e+04 21.888 0.000 2.39e+05 2.86e+05
gender -1.784e+04 8177.285 -2.182 0.031 -3.4e+04 -1657.933
hsc_p -1.845e+04 2.08e+04 -0.888 0.376 -
5.96e+04
2.27e+04
hsc_s -2.775e+04 1.57e+04 -1.768 0.080 -
5.88e+04
3315.162
degree_p -9910.8456 2.2e+04 -0.450 0.653 -
5.35e+04
3.37e+04
degree_t 3.946e+04 1.66e+04 2.379 0.019 6636.236 7.23e+04
workex -7750.7686 7629.205 -1.016 0.312 -
2.29e+04
7349.566
etest_p 1.837e+04 1.4e+04 1.313 0.192 -
9329.141
4.61e+04
specialisatio
n
2457.7237 7980.893 0.308 0.759 -
1.33e+04
1.83e+04
mba_p 3.702e+04 2.07e+04 1.785 0.077 -
4031.562
7.81e+04
Omnibus: 10.856 Durbin-Watson: 1.965
Prob(Omnib
us):
0.004 Jarque-Bera (JB): 11.044
Skew: 0.662 Prob(JB): 0.00400
Kurtosis: 3.477 Cond. No. 12.1
# Identify max P-value (P>|t|) column 
# Feature ssc_p has 0.995
#drop ssc_p
X_scaled = X_scaled.drop('ssc_p', axis=1) 
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()
 Placement Data Analysis
Page | 36
Conclusion
This data set consists of Placement data of students in a XYZ campus. It includes 
secondary and higher secondary school percentage and specialization. It also includes 
degree specialization, type and Work experience and salary offers to the placed students. 
The college placement community thrives on students finding suitable jobs and recruiters 
finding students that add value to their company. Every year, thousands of engineering 
students sit for college placements find jobs. The college administration helps students 
navigate these waters by providing the necessary tools. 
They provide information and train students to become eligible for jobs. It is paramount 
that this process yields results that add value to the student and the company. Thus, we 
analyzed the placement data of a university to gain insights into process and areas that 
need focus. This study targeted to find the most impactful companies which can be get 
the higher position in the priority list of campus placement. It is inconvenient to track the 
trends or patterns of the results and the placements, making it more difficult to change the 
policies or any other change required for improvement.
