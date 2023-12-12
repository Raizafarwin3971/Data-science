
import pandas as pd
import os
import numpy as np
from pandas import read_csv
import matplotlib 
import sklearn as sk
import plotly.graph_objects as go
import webbrowser
from sklearn import preprocessing
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Normalizer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from pandas.plotting import scatter_matrix
import seaborn as sns

#importing the models we need for ensembling

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

stroke_headers=['id','gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg-glucose_level','bmi','smoking_status']
df1=read_csv(r"C:\Users\raiza\Documents\BSc\Data Science\healthcare-dataset-stroke-data.csv")
print(df1.head())

print(df1.shape)

# lets apply Bivariate Analysis to show the relationship between variables: Correlation Matrix
corr=df1.corr()
print(corr)

correlation_matrix = df1.corr()
plt.figure(figsize=(10, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


#univariate analysis
# Line chart of stroke ocuurence across different ages
plt.figure(figsize=(10, 6))
sns.lineplot(x='age', y='stroke', data=df1, marker='o')
plt.title('Trend of stroke occurence (Unprocessed data)')
plt.xlabel('Age')
plt.ylabel('Stroke occurence')
plt.show()

df1.hist(figsize=(10, 8))
plt.suptitle('Histograms of variables in unprocessed dataset', y=1.02)
plt.show()


df2=df1.drop(['id','gender','ever_married','work_type','Residence_type'],axis='columns')
df2.head()
print(df2.shape)


print(df2.head())

print(df2.isnull().sum())

df3 = df2.copy()
median_value = df3['bmi'].median()
df3['bmi'].fillna(median_value, inplace=True)

print(df3.isnull().sum())
print(df3.head())

df4=df3.copy()
print(len(df4.smoking_status.unique()))#checking how many different values are there in smoking status
#since the feature is less enough, it is not required to do dimensionality reduction
#so now let us check for outliers in the dataset

numerical_columns = df4.select_dtypes(include=np.number)
z_scores = np.abs((numerical_columns - numerical_columns.mean()) / numerical_columns.std())
threshold = 3
outliers = z_scores > threshold
print(outliers.sum())

#declaring new headers for the dataset

new_stroke_headers=['age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke', 'smoking_status_unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
#lets now apply simpleimputer function to remove outliers i.e replacing missing values with median value

columns_with_missing_values = ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
imputer = SimpleImputer(strategy='median')
df4[columns_with_missing_values] = imputer.fit_transform(df4[columns_with_missing_values])

print(df4.isnull().sum())
print(df4.dtypes)
#lets now apply one hot encoding for smoking status column in order to convert its categorical values to numerical values

categorical_columns = ['smoking_status']
df_categorical = df4[categorical_columns]
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(df_categorical)
df_encoded = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(input_features=categorical_columns))
df_final = pd.concat([df4.drop(columns=categorical_columns), df_encoded], axis=1)
print(df_final.head()) # One-hot encoding converted 'smoking_status' into binary columns for each category present in the original column. Each category becomes a new column with binary values (0 or 1), and the value is 1 for the category that corresponds to the original value of the categorical variable, and 0 for all other categories.

# lets now normalize the data so that all features are on a similar scale, preventing some features from dominating others based solely on their magnitude
columns_to_scale = ['avg_glucose_level', 'bmi']
scaler = MinMaxScaler()
df_final[columns_to_scale] = scaler.fit_transform(df_final[columns_to_scale])
print(df_final.head())

#using normalizer for feature scaling
df_final_normalizer=Normalizer(norm='l1').fit(df_final)
my_normalised_data=df_final_normalizer.transform(df_final)


set_printoptions(precision=2)
print("\n my normalized data:\n", my_normalised_data [0:3])

#standardazition
data_array=df_final.values

data_scaler=StandardScaler().fit(data_array)
data_rescaled=data_scaler.transform(data_array)

set_printoptions(precision=2)
print("\n my standardized data:\n", data_rescaled [765:770])

#lets visualize the new dataframe 
df_final.hist(figsize=(10, 8))
plt.suptitle('Histograms of Variables in final preprocessed dataset', y=1.02)
plt.show()


#get the data correlation
df_final_correlations = df_final.corr()

corr_fig = plt.figure()
axises=corr_fig.add_subplot(111)
axcorr = axises.matshow(df_final_correlations, vmin=-1, vmax=1)

corr_fig.colorbar(axcorr)  
ticks = np.arange(0, 10, 1)  

axises.set_xticks(ticks)
axises.set_yticks(ticks)
axises.set_xticklabels(df_final_correlations.columns, rotation=45, ha='right')
axises.set_yticklabels(df_final_correlations.columns)

plt.show()


# Scatter plot of age vs. avg_glucose_level
plt.figure(figsize=(8, 6))
plt.scatter(df1['age'], df1['avg_glucose_level'], c=df1['stroke'], cmap='coolwarm', alpha=0.7)
plt.title('Age vs. Average Glucose Level (Unpreprocessed)')
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.colorbar(label='Stroke')
plt.show()

# Scatter plot of bmi vs. avg_glucose_level
plt.figure(figsize=(8, 6))
plt.scatter(df1['bmi'], df1['avg_glucose_level'], c=df1['stroke'], cmap='coolwarm', alpha=0.7)
plt.title('BMI vs. Average Glucose Level (Unpreprocessed)')
plt.xlabel('BMI')
plt.ylabel('Average Glucose Level')
plt.colorbar(label='Stroke')
plt.show()

# Scatter plot of age vs. bmi
plt.figure(figsize=(8, 6))
plt.scatter(df1['age'], df1['bmi'], c=df1['stroke'], cmap='coolwarm', alpha=0.7)
plt.title('Age vs. BMI (Unpreprocessed)')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.colorbar(label='Stroke')
plt.show()


#distribution of stroke across heart stroke 
heart_disease_stroke = df1.groupby('heart_disease')['stroke'].mean() * 100
plt.figure(figsize=(8, 6))
sns.barplot(x=heart_disease_stroke.index, y=heart_disease_stroke.values, palette='coolwarm')
plt.xlabel('Heart Disease Status')
plt.ylabel('Percentage of Stroke Occurrence')
plt.title('Effect of Heart Disease on Stroke')
plt.xticks(ticks=[0, 1], labels=['No Heart Disease', 'Heart Disease'])
plt.ylim(0, 15)  # Set the y-axis limits for better visualization
plt.show()

#print(df1.info)

#model fitting

    #split train and test data
    
train_headers=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
target_header=['stroke']


X=df_final[train_headers]
y = df_final[target_header].values.ravel()


print(y.shape)
print(X.shape)


#print the training set, x
#print(x)
#print(y)
SEED=1
#split the data into train and test-split 
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

#checking no. of train and test data
print('\n The total of training dataset', X_train.shape)
print('\n The total of test dataset', X_test.shape)
print('---------------------------------------------------')

#instantiate the model
max_depth_limit = 3
model=DecisionTreeClassifier(random_state=SEED)

#cross validation is performed to split the train data in order to train and evaluate model 10 times
k = 10 #give no. of folds to k

#hyperparameter and values
classifier_hypara=dict()
classifier_hypara['max_depth']=[2,3,4,6,8,10]
classifier_hypara['min_samples_split']=[2,4,6,8,9]
classifier_hypara['min_samples_leaf']=[0.05,0.1,0.5,1]
classifier_hypara['criterion']=['gini','entropy']

kf = KFold(n_splits=k)

#doing gridseaerch to fit the grid
classifier_grid=GridSearchCV(model,classifier_hypara,scoring='accuracy',n_jobs=-1, cv=kf)
classifier_grid_fit=classifier_grid.fit(X_train, y_train)

# cross-validation and get the accuracy scores for each fold
scores = cross_val_score(model, X_train, y_train, cv=kf)

# Printing the accuracy scores for each fold
print("Accuracy scores for each fold:", scores)

# Calculating the mean and standard deviation of the accuracy scores
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print("Mean accuracy:", mean_accuracy)
print("Standard deviation of accuracy:", std_accuracy)

#hyperparameter tuning results
print('best hyperparameters: %s' % classifier_grid_fit.best_params_)
print('best max_depth=', classifier_grid_fit.best_estimator_.get_params()['max_depth'])
print('best min_samples_split=', classifier_grid_fit.best_estimator_.get_params()['min_samples_leaf'])
print('best min_samples_leaf=', classifier_grid_fit.best_estimator_.get_params()['min_samples_leaf'])
print('best criterion=',classifier_grid_fit.best_estimator_.get_params()['criterion'])

#print best hyperparameyters
print('\nSuggested best hyperparameters: \n', classifier_grid_fit.best_estimator_.get_params())
print('best score: %s {:.3f}\n'.format(classifier_grid_fit.best_score_))

# let us visualize 
train_accuracy = []
test_accuracy = []
 

max_depth_values = range(1, 21)
 

for max_depth in max_depth_values:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
 
    # Calculating accuracy on training data
    train_accuracy.append(model.score(X_train, y_train))
 
    # Calculating accuracy on testing data
    test_accuracy.append(model.score(X_test, y_test))
    
#train the model to fit
model.fit(X_train, y_train)

#now lets predict the model
y_pred_train=model.predict(X_train)

#now let's predict the model
y_pred=model.predict(X_test)
print(y_pred)
 
# Plot the accuracy chart
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, train_accuracy, label='Training Accuracy')
plt.plot(max_depth_values, test_accuracy, label='Testing Accuracy')
plt.xlabel('Maximum Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy vs. Maximum Depth')
plt.legend()
plt.xticks(max_depth_values)
plt.show()

#plot the decsion tree model
tree_classif_label=['2', '4']
fig=plt.figure(figsize=(30,20))
tree.plot_tree(model, feature_names=train_headers, class_names=tree_classif_label, filled=True,
             rounded=True, fontsize=14)


#let's limit decision tree model in order analyze
tree_classif_label = ['2', '4']
max_depth_limit = 3
fig = plt.figure(figsize=(30, 20))
tree.plot_tree(model, feature_names=train_headers, class_names=tree_classif_label, filled=True,
               rounded=True, fontsize=14, max_depth=max_depth_limit)



pred_one = model.predict([[30, 0, 0, 93.25, 30]])
print(pred_one)

#print(X_test)

#train set accuracy
model_acc_train=accuracy_score(y_train, y_pred_train)
print('\nModel accuracy on train set: {:.2f}\n'.format(model_acc_train))

#test set accuracy
model_acc_test=accuracy_score(y_test, y_pred)
print('\nModel accuracy on test set: {:.2f}\n'.format(model_acc_test))

#confusion matrix
matrix_info=confusion_matrix(y_test, y_pred)
print('\nconfusion matrix for decision tree\n',matrix_info, '\n')

#put the classification report for decsion tree
class_report=classification_report(y_test, y_pred)
print('\nclassification report for decision tree\n', class_report,'\n')
#in conclusion, my model is overfitted to the training set. it is poorly predicting stroke

lr=LogisticRegression(solver='saga', max_iter=1000, random_state=SEED)

lr_hypara = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'solver': ['liblinear', 'lbfgs', 'saga'],  
    'max_iter': [100, 200, 300] 
}
#hyperparameter tuning
classifier_hypara = dict()
classifier_hypara['max_depth'] = [2, 3, 4, 6, 8, 10]
classifier_hypara['min_samples_split'] = [2, 4, 6, 8, 9]
classifier_hypara['min_samples_leaf'] = [0.05, 0.1, 0.5, 1]
classifier_hypara['criterion'] = ['gini', 'entropy']

knc=KNN()
#hyperparameter tuning
knc_param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

classifier_list=[('Logistic Regression:', lr),('K Nearest Neighbours:',knc), ('Decision Tree :',model)]


#instantiate the cv
kf = KFold(n_splits=k)

for clsf_name, clsf in classifier_list:
    
    #now fit the model
    clsf.fit(X_train, y_train)
    
    #compute the array containing the 10-folds CV MSEs
    scores_clsf=-cross_val_score(clsf, X_train, y_train, cv=kf)
    print("\nCross Val mean: {:.3f}(std: {:.3f})".format(scores.mean()*-1,scores.std()),end="\n")
    
    #predict and calculate the accuracy on test set for each model
    y_pred_test_clsf=clsf.predict(X_test)
    print("\n {:s} Test:{:.3f}".format(clsf_name, accuracy_score(y_test, y_pred_test_clsf)),'\n')
    
    #predict and calculate the accuracy on train data for each model
    y_pred_train_clsf=clsf.predict(X_train)
    print('\n {:s}:{:.3f}'.format(clsf_name, accuracy_score(y_train, y_pred_train_clsf)), '\n')
    
    
#instantiate the vc
vc=VotingClassifier(estimators=classifier_list)

#fit vc to the training set 
vc.fit(X_train, y_train)

scores_vc=-cross_val_score(vc, X_train, y_train, cv=10)
print("\nCross Val mean: {:.3f}(std: {:.3f})".format(scores.mean()*-1,scores.std()),end="\n")

#now lets predict the label for training set
y_pred_train_vc=vc.predict(X_train)
#now lets print the evaluation score for the vc
print('\n voting classifier train{:.3f}'.format(accuracy_score(y_train, y_pred_train_vc)), '\n')


# Perform GridSearchCV for Logistic Regression
lr_grid = GridSearchCV(lr, lr_hypara, scoring='accuracy', n_jobs=-1, cv=kf)
lr_grid_fit = lr_grid.fit(X_train, y_train)

# Print best hyperparameters for Logistic Regression
print('Best hyperparameters for Logistic Regression:', lr_grid_fit.best_params_)
print('Best C:', lr_grid_fit.best_estimator_.get_params()['C'])
print('Best solver:', lr_grid_fit.best_estimator_.get_params()['solver'])
print('Best max_iter:', lr_grid_fit.best_estimator_.get_params()['max_iter'])
print('Best score:', lr_grid_fit.best_score_)

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UndefinedMetricWarning)

#hyperparamter tuning for KNN
knc_grid = GridSearchCV(knc, knc_param_grid, scoring='accuracy', cv=kf, n_jobs=-1)
knc_grid.fit(X_train, y_train)

print('\nBest hyperparameters for K Nearest Neighbors:', knc_grid.best_params_)
print('Best score:', knc_grid.best_score_)

#classification report for KNN
y_pred_knn = knc.predict(X_test)
print("\nClassification Report for K Nearest Neighbors:")
print(classification_report(y_test, y_pred_knn))

# classification report for logistic reg
y_pred_lr = lr_grid_fit.predict(X_test)
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

# Predictions from Decision Tree model
y_pred_tree = model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Predictions from Logistic Regression model
lr_grid_fit.best_estimator_.fit(X_train, y_train)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Predictions from K Nearest Neighbors model
y_pred_train_lr = lr_grid_fit.best_estimator_.predict(X_train)
y_pred_test_lr = lr_grid_fit.best_estimator_.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

accuracy_train_lr = accuracy_score(y_train, y_pred_train_lr)
accuracy_test_lr = accuracy_score(y_test, y_pred_test_lr)

# Plot training and test accuracies on log regression
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, train_accuracy, label='Training Accuracy')
plt.plot(max_depth_values, test_accuracy, label='Testing Accuracy')
plt.axhline(y=accuracy_train_lr, color='r', linestyle='--', label='Logistic Regression Train Accuracy')
plt.axhline(y=accuracy_test_lr, color='g', linestyle='--', label='Logistic Regression Test Accuracy')
plt.xlabel('Maximum Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Decision Tree vs. Logistic Regression Accuracy Comparison')
plt.legend()
plt.xticks(max_depth_values)
plt.show()

# Print the accuracy scores for each model
print("Accuracy of Decision Tree model:", accuracy_tree)
print("Accuracy of Logistic Regression model:", accuracy_lr)
print("Accuracy of K Nearest Neighbors model:", accuracy_knn)


#compraison of 3 models accruacy
#accuracy_scores = {'Decision Tree': [], 'Logistic Regression': [], 'K Nearest Neighbors': []}


#for max_depth in max_depth_values:
    # Decision Tree 
#    model_dt = DecisionTreeClassifier(max_depth=max_depth)
#    model_dt.fit(X_train, y_train)
#    y_pred_dt = model_dt.predict(X_test)
#    accuracy_dt = accuracy_score(y_test, y_pred_dt)
#    accuracy_scores['Decision Tree'].append(accuracy_dt)

    # Logistic Regression 
#    model_lr = LogisticRegression(solver='saga', max_iter=100, random_state=SEED)
#    model_lr_grid = GridSearchCV(model_lr, lr_hypara, scoring='accuracy', n_jobs=-1, cv=kf)
#    model_lr_grid.fit(X_train, y_train)
#    y_pred_lr = model_lr_grid.predict(X_test)
#    accuracy_lr = accuracy_score(y_test, y_pred_lr)
#    accuracy_scores['Logistic Regression'].append(accuracy_lr)

    # K Nearest Neighbors
#    model_knn = KNN()
#    model_knn_grid = GridSearchCV(model_knn, knc_param_grid, scoring='accuracy', cv=kf, n_jobs=-1)
#    model_knn_grid.fit(X_train, y_train)
#    y_pred_knn = model_knn_grid.predict(X_test)
#    accuracy_knn = accuracy_score(y_test, y_pred_knn)
#    accuracy_scores['K Nearest Neighbors'].append(accuracy_knn)


#plt.figure(figsize=(10, 6))
#for model_name, accuracy_list in accuracy_scores.items():
 #   plt.plot(max_depth_values, accuracy_list, label=model_name)

#plt.xlabel('Maximum Depth of Decision Tree')
#plt.ylabel('Accuracy')
#plt.title('Models Accuracy Comparison')
#plt.legend()
#plt.xticks(max_depth_values)
#plt.show()

# comparison of accuracy scores of all 3 models
accuracy_scores = {
    'Decision Tree': accuracy_tree,
    'Logistic Regression': accuracy_lr,
    'K Nearest Neighbors': accuracy_knn
}

plt.figure(figsize=(8, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['m', 'crimson', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.ylim(0.0, 1.0) 
plt.show()


# lets calculate summary statistics(count, mean,std, min, max values)
summary_stats = df_final.describe()
print(summary_stats)


# Scatter plot for age vs. heart disease
plt.figure(figsize=(8, 6))
plt.scatter(df_final['age'], df_final['heart_disease'], c=df_final['stroke'], cmap='coolwarm', alpha=0.7)
plt.title('Age vs. heart disease status')
plt.xlabel('Age')
plt.ylabel('heart disease status')
plt.colorbar(label='Stroke')
plt.show()

# Calculate correlation matrix
correlation_matrix = df_final.corr()
print(corr)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of the cleaned data')
plt.show()


#print(df_final.columns)
#pie chart
plt.figure(figsize=(6, 6))
df_final['stroke'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['sienna', 'lightcoral'])
plt.title('Distribution of Stroke')
plt.ylabel('')
plt.show()



