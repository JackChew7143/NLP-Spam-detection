# NLP-Spam-detection
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.


The first step to use XGBoost model is to import its library from xgboost according to the 
figure above. The XGBClassifier module from xgboost library would be the main method to build 
the model itself.
First of all, create instance of XGBClassifier() and assign it to variable xgb. The xgb will 
be fit the training data using fit() function including training data features(x_train_trainsf) and
training target variable (y_train).

Based on the evaluation results, the default model accuracy is 0.98, which means that the 
total number of samples correctly classified by the model across all samples is 98%, which is very 
good performance. Furthermore, the default precision is 0.97 which means that the model can
escape the false positive. In addition, the default recall is 0.9 which means that it can predict well 
on positive samples across all the true positive samples (true positive+ false negative). For 
confusion matrix, the only 3 samples model misclassify as class 0 (ham) but belong to class 
1(spam)- false positive and only 15 samples model misclassify as class 1 (spam) but actually 
belong to class 0 (spam)-false negative.

From the result above, the researcher can observe that the default model able to 
differentiate well on true positive (positive samples) and false positive(negative samples) as AUC 
score is 0.98

Given the high computational and time cost required for the grid search to try each 
hyperparameter combination, the random search method was chosen as an alternative solution for 
performing hyperparameters, as it can reduce the high computational cost by randomly selecting 
combinations of hyperparameters values for testing (Chen, 2021). 

First of all, the hyperparameter space will be defined at a broader scope, as it is different 
from grid search attempts to perform exhaustive searches on all combinations as shown in Figure
46. Define a wide distribution of learning rates (0-0.3 with interval 0.05), values ranging from 1 
to 10 at intervals of 1 for gamma, reg_lambda, and max_depth.

RandomizedSearchCV will search for local optima of hyperparameter combinations in 50 
runs and estimate the performance of models with different folds using 4 cross-validations to 
ensure that accuracy is well generalized in unseen data, as shown in Figure 47.
Figure 47: Cross-Validation Object for XGBoost
The best hyperparameters within 50 runs are lambda =6, max_depth =6, learning_rate 
=0.25 and gamma =1 on training data as shown in Figure 48.
Figure 48: Best Parameters after Tuning for XGBoost
Applying the best hyperparameters to the test data results in improved overall performance 
compared to the default models with parameter learning rate = 0.3, gamma=0, reg_lambda=1, 
and max_depth =6

By changing the default parameters to the best hyperparameters, it can maximize model 
performance in terms of accuracy, recall , precision, F1 score, and reduce number false negative 
confusion matrix. The most obvious improvements were F1 and the confusion matrix, with the F1 
score changing from 0.937 to 0.941, recall score changing from 0.90 to 0.91 that represent improve 
performance in predicting positive instances out of all actual positive instances and the confusion 
matrix 15 being a false negative (FN) down to 14 false negatives.
One of the reasons the hyperparameter set can maximize model performance is because it 
implements a relatively conservative model to shrink the leaf weights by setting higher
reg_lambda L2 regularization values, a maximum tree depth of 6 to make it a less aggressive 
learner, and a learning rate of 0.25 to prevent learning too much from the previous tree, all these 
combinations eventually result in tree pruning, preventing overfitting,
