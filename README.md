# AI-ML 
<br>
***Notes, URL, Summary***
Repo for Artificial Intelligence and Machine Learning projects and information for research and academic purposes. <br>
<br>


## Intro to AI and ML
Artificial intelligence (AI) refers to the use of technology to build computers (and technology solutions) that mimic human intelligence / cognitive abilities. Machine Learning (ML) is a subset of AI that alows a machine to learn and improve from experience.
<br>
### Resources
<br>
A good place to start is the Artificial Intelligence: A Modern Approach by Russell and Norvig book https://aima.cs.berkeley.edu/  <br>
<br>
There are many good books and resources available including: <br>
Artificial intelligence fundamentals: https://www.ibm.com/design/ai/fundamentals  <br>
Artificial intelligence (AI) vs. machine learning (ML): https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning<br>
Essential Math for Data Science: https://www.oreilly.com/library/view/essential-math-for/9781098102920/<br>
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow:  https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/  <br>
Inside Deep Learning: https://www.manning.com/books/inside-deep-learning <br>
Introduction to Generative AI: introductory level microlearning course https://www.cloudskillsboost.google/course_templates/536 <br>
Machine Learning: https://www.w3schools.com/python/python_ml_getting_started.asp <br>
Pandas Workout: https://www.manning.com/books/pandas-workout  <br>
Python for Data Analysis: https://wesmckinney.com/book/ <br>
Structuring Machine Learning Projects: https://w3steps.com/en/structuring-machine-learning-projects-outline/  <br>
<br>

## Application 
We may create an application to include a user interface for viewing results such as virtual assistants, web apps, chatbox etc. <br>
Lightweight micro-framework for web apps: https://flask.palletsprojects.com/en/3.0.x/ <br>
Python web framework: https://www.djangoproject.com/ <br>
<br>

## Data 
<br>

### CRISP-DM
<br>
CRISP-DN is an open standard process model for common data mining approaches. <br> 
OSEMN Data Science Life Cycle: https://www.datascience-pm.com/osemn/ <br>
Cross-industry standard process for data mining (CRISP-DM): https://www.datascience-pm.com/wp-content/uploads/2021/08/CRISP-DM-for-Data-Science.pdf
<br> <br>

**Steps** <br>
from https://www.datascience-pm.com/crisp-dm-2/
<br>
![CRISP-DM.png](images%2FCRISP-DM.png)
<br>It has six sequential phases: <br>
Business understanding – What does the business need? <br>
Data understanding – What data do we have / need? Is it clean? <br>
Data preparation – How do we organize the data for modeling? <br>
Modeling – What modeling techniques should we apply? <br>
Evaluation – Which model best meets the business objectives? <br>
Deployment – How do stakeholders access the results? <br>
<br>
here is how I used it in a project, check it out at: https://github.com/johennie/assg5_coupons
<br>

### OSEMN (Obtain, Scrub, Explore, Model, and iNterpret) framework 
<br>
It is a framework and more focused on data and data quality. <br>
OSEMN Data Science Life Cycle: https://www.datascience-pm.com/osemn/ <br>
<br>

**Steps** <br>
Obtain: make sure we have access to the data, figure out if there is a cost to it or do we need to set up a system to capture the data. <br>
Scrub: make sure it is quality data, and if needed clean it up, document it, and load it. <br>
Explore: create visualization, understand the data, look at the raw data as well as identify similar characteristics. Once the data is obtained, we would need to make sure visualizations are possible with the data available. <br>
Model: identify what and how the different data points will be used. Evaluate the performance and minimize the cost function. <br>
iNterpret: verify that non-technical objectives have been achieved. <br>
<br>

## Free datasets
 <br>
IEEE data port: https://ieee-dataport.org/datasets <br>
UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/ <br>
U.S. Government's Open Data: https://data.gov/  <br>
Kaggle: https://www.kaggle.com/datasets  <br>
Washington University Supreme Court Database: http://scdb.wustl.edu/  <br>
HuggingFace: https://huggingface.co/datasets  <br>
 <br>

## Python
 <br>
Python Tutorial: https://www.w3schools.com/python/default.asp
<br>

Machine Learning with Python: https://developer.ibm.com/languages/python/courses/  <br>
 <br>

### Libraries
 <br>
pandas: https://pandas.pydata.org/  <br>
<br>
collections: implements specialized container datatypes   https://docs.python.org/3/library/collections.html <br>
fuzzywuzzy: fuzzy string matching https://pypi.org/project/fuzzywuzzy/ <br>
IPython: provides a rich toolkit for an interactive Python shell https://pypi.org/project/ipython/ <br>
Matplotlib: Visualization with Python https://matplotlib.org/  <br>
numpy: fundamental package for scientific computing with Python https://numpy.org/  <br>
re: Regular expression operations https://docs.python.org/3/library/re.html  <br>
scikit-learn: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble <br>
scipy: provides algorithms for optimization, integration, interpolation, statistics etc  https://scipy.org/ <br>
seaborn: statistical data visualization  https://seaborn.pydata.org/  <br>
sklearn: Machine Learning in Python https://scikit-learn.org/stable/
<br>
 
# ML
<br>

**Steps** <br>
1. Business understanding <br>
1.1. Outcomes and Predictions <br>
2. Data understanding <br>
2.1. Data Acquisition <br>
2.2. Exploratory Data Analysis  (EDA) <br>
 - and Visualization <br>
    a good libray to start with is: https://docs.profiling.ydata.ai/latest/ <br>
    here is how I used it in a project, check it out at: https://github.com/johennie/tropical_cyclone_tracks/blob/main/notebooks/tropical_storm.ipynb <br>
3. Data preparation <br>
   - drop unnecessary columns <br>
   - check for duplicates <br>
   - check and address Nan, nulls, inaccurate data <br>
   - check and address outliers <br>
   - examine correlations <br>
   - scale and normalize data for investigation <br>
   here is how I used it in a project, check it out at: https://github.com/johennie/what_drives_a_car_price/blob/main/notebooks/prompt_II.ipynb <br>
   - <br>
4. Modeling <br>
4.1. Split data into train test split <br>
4.2. Define and select the model(s) <br>
4.3. Fit, predict, and collect metrics to evaluate the model(s) <br>
<br>

## Visualization
Data Visualization Techniques <br>
•	Histograms: For understanding the distribution of numerical features. <br>
•	Box Plots: For detecting outliers and understanding the spread of numerical features. <br>
•	Bar Plots: For visualizing the distribution of categorical features. <br>
•	Count Plots: For showing the frequency of categories in categorical features. <br>
•	Scatter Plots: For visualizing relationships between numerical features. <br>
•	Pair Plots: For visualizing pairwise relationships in the dataset. <br>
•	Heatmap: For visualizing correlations between features. <br>
•	Pivot Tables: For summarizing data and exploring relationships between different categorical features.<br>
<br>
Types of Data Plots and How to Create Them in Python: https://www.datacamp.com/tutorial/types-of-data-plots-and-how-to-create-them-in-python <br>
Top 50 matplotlib Visualizations: https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/ <br>
<br>


## Cluster analysis and Principal Component Analysis (PCA)
<br>![clustering.png](images%2Fclustering.png) <br>
<br>
Assigns related data into clusters / groups for processing used used for generalization, data compression, and privacy.
<br>
10 Clustering Algorithms With Python: https://machinelearningmastery.com/clustering-algorithms-with-python/
<br> 
Statistical technique to reduce data features to those essential while maintining the variance and properties of the original data. <br>
<br> 
NOTE: to get an object w the columns that make up each PC : pca_loadings_analysis.apply(lambda s: s.nlargest(5).index.tolist(), axis=1)
<br>

## Classification
 <br>
Classification in Machine Learning: An Introduction
 https://www.datacamp.com/blog/classification-machine-learning
<br>
Classification: ROC Curve and AUC https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc  <br>
 <br>
 

## Hierarchical Clustering
 <br>
Hierarchical Clustering in Machine Learning 
 https://www.geeksforgeeks.org/hierarchical-clustering/
<br>
Machine Learning - Hierarchical Clustering https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp  <br>
 <br>

## Cross Validation CV
<br>What is Cross Validation?
https://docs.aws.amazon.com/machine-learning/latest/dg/cross-validation.html <br>
<br>
cross-validation is used to evaluate an estimator (model) performance and
sklearn.model_selection.cross_val_score can be used to evaluate cross-validation scores
some info: https://scikit-learn.org/stable/modules/cross_validation.html
Machine Learning - Cross Validation https://www.w3schools.com/python/python_ml_cross_validation.asp  <br>
 <br>

## Regression and Multiple Regression
Regression is used to predict continuous values and in multiple regression multiple independent variables are used to predict the dependent prediction y.
<br>
What is regression? https://www.seldon.io/machine-learning-regression-explained <br>
<br>
What is Multiple Regression?
https://www.sciencedirect.com/topics/psychology/multiple-regression <br>
Machine Learning - Multiple Regression https://www.w3schools.com/python/python_ml_multiple_regression.asp  <br>
 <br>

## Grid Search and Hyperparameters
A technique to try different values to select the best score.
<br>
Grid Search in Python from scratch— Hyperparameter tuning: https://towardsdatascience.com/grid-search-in-python-from-scratch-hyperparameter-tuning-3cca8443727b <br>
Machine Learning - Grid Search: https://www.w3schools.com/python/python_ml_grid_search.asp  <br>
Statistical comparison of models using grid search: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html  <br>
<br>
example: <br>
1. define pipeline<br>
pipeline = Pipeline(steps=[...])<br>
preprocessor = ColumnTransformer(transformers=[...])<br>
2. split the data<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)<br>
3. fit and predict<br>
grid_search = GridSearchCV(pipeline, params, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)<br>
start = time()<br>
grid_search.fit(X_train, y_train)<br>
stop = time() <br>
4. calculate metrics<br>
...<br>

![gridsearch_metrics.png](images%2Fgridsearch_metrics.png)

<br>
here is how I used it in a project, check it out at: https://github.com/johennie/bank_marketing/blob/main/notebooks/jh_prompt_III.ipynb

<br>

## Linear Regression
<br>Algorithm that defines a linear relationship between the features X (independent variables) and the prediction y (dependent variable)
<br>
![linear_regression_example.png](images%2Flinear_regression_example.png)
<br>
What is Linear Regression?
 https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-linear-regression/ <br>
Linear regression python code example:<br>
from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import LinearRegression <br>
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error <br>
... <br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model <br>
model = LinearRegression() <br>
#Fit, predict, and collect metrics to evaluate the model <br>
model.fit(X_train, y_train) <br>
y_pred = model.predict(X_test) <br>
median_absolute_error(y_train, y_pred) <br>
mean_squared_error(y_train, y_pred) <br>
mean_absolute_error(y_train, y_pred) <br>
<br>
here is how I used it in a project, check it out at: https://github.com/johennie/what_drives_a_car_price/blob/main/notebooks/prompt_II.ipynb <br>
<br>
Machine Learning - Linear Regression https://www.w3schools.com/python/python_ml_linear_regression.asp  <br>
<br>


## Ridge Regression / L2 regularization
<br>Used for the analysis of multicollinearity in multiple regression data. Useful when multicollinearity is present in a dataset.
<br>![ridge_regression_example.png](images%2Fridge_regression_example.png)
<br>
What is ridge regression?
 https://www.ibm.com/topics/ridge-regression <br>
Ridge regression python code example:<br>
from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import Ridge <br>
from sklearn.metrics import r2_score <br>
... <br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model <br>
model = Ridge(alpha=0.1) <br>
#Fit, predict, and collect metrics to evaluate the model <br>
model.fit(X_train, y_train) <br>
y_pred = model.predict(X_test) <br>
ridger2 = r2_score(y_test, y_pred) <br>
<br>
here is how I used it in a project, check it out at: https://github.com/johennie/what_drives_a_car_price/blob/main/notebooks/prompt_II.ipynb <br>
<br>

## Least absolute shrinkage and selection operator (LASSO) Regression  
<br>
Enhance the prediction accuracy by performing variable selection and regularization.
<br>

![lasso_example.png](images%2Flasso_example.png)
<br>
what is Lasso regression?
 https://www.sciencedirect.com/topics/psychology/lasso-regression <br>
Lasso regression python code example:<br>
from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import Lasso <br>
from sklearn.metrics import r2_score, mean_absolute_error <br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model <br>
model = Lasso(alpha=0.1) <br>
#Fit, predict, and collect metrics to evaluate the model <br>
model.fit(X_train, y_train) <br>
y_pred = model.predict(X_test) <br>
lasso_coefficients = model.coef_ <br>
lasso_r2 = r2_score (y_test, y_pred)  <br>
mae = mean_absolute_error(y_train, y_pred) <br>
<br>
here is how I used it in a project, check it out at: https://github.com/johennie/what_drives_a_car_price/blob/main/notebooks/prompt_II.ipynb <br>

<br>
<br>

## Polynomial Regression
<br>
A linear regression model that defines the relationship between independent features X and dependent prediction y as a polynomial of nth degree.
<br>
What is Polynomial Regression?
 https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/ <br>
Machine Learning - Polynomial Regression https://www.w3schools.com/python/python_ml_polynomial_regression.asp  <br>
 <br>

## RidgeCV 
<br><br>![grid_search_ridge.png](images%2Fgrid_search_ridge.png)
What is Ridge Regression?
https://www.publichealth.columbia.edu/research/population-health-methods/ridge-regression <br>
RidgeCV python code example:<br>
from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import RidgeCV <br>
from sklearn.metrics import mean_absolute_error <br>
from sklearn.compose import TransformedTargetRegressor <br>
from sklearn.pipeline import make_pipeline <br>
from sklearn.inspection import permutation_importance <br>
... <br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model <br>
alphas = np.logspace(-10, 10, 21)  #alpha values to be chosen from by cross-validation <br>
model = make_pipeline( <br>
    preprocessor, <br>
    TransformedTargetRegressor( <br>
        regressor=RidgeCV(alphas=alphas_2), <br>
        func=np.log10, <br>
        inverse_func=sp.special.exp10,         <br>
    ),    <br>
) <br>
#Fit, predict, and collect metrics to evaluate the model <br>
model.fit(X_train, y_train) <br>
y_pred = model.predict(X_test) <br>
ridge_2_r = permutation_importance(model, X_test, y_test,n_repeats=30, random_state=0) <br>
mae = mean_absolute_error(y_train, y_pred) <br>
<br>
here is how I used it in a project, check it out at: https://github.com/johennie/what_drives_a_car_price/blob/main/notebooks/prompt_II.ipynb <br>
 <br>

## K-nearest Neighbors
 <br>
What is the k-nearest neighbors (KNN) algorithm?
https://www.ibm.com/topics/knn 
<br>
sklearn.neighbors.KNeighborsClassifier https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  <br>
 <br>
KNN python code example:<br>
from sklearn.datasets import make_classification <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.neighbors import KNeighborsClassifier <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.preprocessing import OneHotEncoder <br>
from sklearn.pipeline import make_pipeline <br>
from sklearn.compose import ColumnTransformer <br>
from sklearn.metrics import accuracy_score, classification_report <br>
... <br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model and pipeline <br>
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', KNeighborsClassifier())]<br>
start = time()<br>
knn_pipeline.fit(X_train, y_train)<br>
stop = time()<br>
y_pred = knn_pipeline.predict(X_test)<br>
...<br>
<br>

## Logistic Regression
<br>
What is Logistic Regression?
https://www.ibm.com/topics/logistic-regression <br>
Logistic regression python code example:<br>
from sklearn.datasets import make_classification <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import LogisticRegression <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.preprocessing import OneHotEncoder <br>
from sklearn.pipeline import make_pipeline <br>
from sklearn.compose import ColumnTransformer <br>
from sklearn.metrics import accuracy_score, classification_report <br>
... <br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model and pipeline <br>
model = LogisticRegression(multi_class='ovr', solver='liblinear') <br>
preprocessor = ColumnTransformer( <br>
    transformers=[ <br>
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), <br>
        ... <br>
    ]) <br>
pipeline = make_pipeline(preprocessor, model) <br>
#Fit, predict, and collect metrics to evaluate the model <br>
pipeline.fit(X_train, y_train) <br>
y_pred = pipeline.predict(X_test) <br>
accuracy = accuracy_score(y_test, y_pred) <br>
report = classification_report(y_test, y_pred) <br>

 <br>
Machine Learning - Logistic Regression https://www.w3schools.com/python/python_ml_logistic_regression.asp  <br>
 <br>

## Coefficients
Models with Coefficients (coef_ attribute):
<br>LogisticRegression
<br>SVM (Support Vector Machine)
<br>GridSearch with LogisticRegression
<br>GridSearch with SVM
<br>Models without Coefficients (coef_ attribute):
<br>RandomForestClassifier
<br>KNN (K-Nearest Neighbors)
<br>DecisionTreeClassifier
<br>GridSearch with RandomForestClassifier
<br>GridSearch with KNN
<br>GridSearch with DecisionTree
<br>feature_names = logreg_pipeline.named_steps['preprocessor'].get_feature_names_out()
<br>coefs = pd.DataFrame(
<br>    logreg_pipeline.named_steps['model'].coef_.flatten(), 
<br>    columns=["Coefficient values"], 
<br>    index=feature_names
<br>)
<br>
<br>coefs.plot.barh(figsize=(9, 12))
<br>titlestr = "Logistic Regression Coefficients \n age seems to have a small impact here \n while the target is impacted significantly by the month and duration"
<br>plt.title(titlestr)
<br>plt.xlabel("Coefficient values")
<br>
<br>plt.axvline(x=0, color=".5")
<br>plt.tight_layout()
<br>plt.subplots_adjust(left=0.3)
<br>plt.show()
<br>
![features_2.png](images%2Ffeatures_2.png)
<br>
## Decision Tree
used for classification and regression, the goal of the model is to predict the value of the dependent variable y.<br>
 <br>
What is a Decision Tree?
 https://www.ibm.com/topics/decision-trees <br>
Machine Learning - Decision Tree https://www.w3schools.com/python/python_ml_decision_tree.asp  <br>
 <br>
https://scikit-learn.org/stable/modules/tree.html <br>
<br>
## DecisionTreeClassifier
<br>
What is decision tree?
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html <br>
<br>
Decision Tree python code example:<br>
<br>
from sklearn.datasets import make_classification <br>
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split<br>
...<br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model <br>
dtc_model = DecisionTreeClassifier() <br>
#Fit, predict, and collect metrics to evaluate the model <br>
dtc_model.fit(X_train, y_train) <br>
y_pred = dtc_model.predict(X_test) <br>
<br>
here is how I used it in a project, check it out at: https://github.com/johennie/bank_marketing/blob/main/notebooks/jh_prompt_III.ipynb
 <br>

## Random Forest Classifier
<br>
What is random forest?
https://www.ibm.com/topics/random-forest
<br>
Reandom forest python code example:<br>
<br>
from sklearn.datasets import make_classification <br>
from sklearn.ensemble import RandomForestClassifier <br>
from sklearn.model_selection import train_test_split<br>
...<br>
#Split the data <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) <br>
#Define the model <br>
rf_model = RandomForestClassifier() <br>
#Fit, predict, and collect metrics to evaluate the model <br>
rf_model = RandomForestClassifier() <br>
rf_model.fit(X_train, y_train) <br>
y_pred_rf = rf_model.predict(X_test) <br>
y_proba_rf = rf_model.predict_proba(X_test)[:, 1] <br>
 <br>
scikit-learn random forest classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
 <br>
RandomForestClassifier 
is a set of decision trees from random selected subset of the training data and uses averaging to improve the predictive accuracy and control over-fitting. 
<br>
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html <br>
 <br>

## C-Support Vector Classification
it can determine the penalty for misclassified data points during the training process. <br>
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
<br>
