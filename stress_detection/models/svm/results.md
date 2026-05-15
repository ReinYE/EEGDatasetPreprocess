## epocx_relative_power_hjorth_features:
5 relative power bands * 2 Hjorth features
+ 5 ratio features
= 15 features / channel

Absolute band power sequence shape: (120, 25, 12, 5)
Relative band power sequence shape: (120, 25, 12, 5)
Meaning: n_trials, n_secs, n_channels, n_bands
Input EEG data shape: (120, 25, 12, 128)
Output relative-power Hjorth features shape: (2520, 180)
Features per channel: 15
Power window: 5
Step: 1
Number of windows per trial: 21
NaN count: 0
Inf count: 0
features shape: (2520, 180)
number of feature names: 180
label shape: (2520,)
Feature DataFrame shape: (2520, 181)

First 20 feature names:
['POW.AF3.theta_relative_power.hjorth_mobility', 'POW.AF3.theta_relative_power.hjorth_complexity', 'POW.AF3.alpha_relative_power.hjorth_mobility', 'POW.AF3.alpha_relative_power.hjorth_complexity', 'POW.AF3.betaL_relative_power.hjorth_mobility', 'POW.AF3.betaL_relative_power.hjorth_complexity', 'POW.AF3.betaH_relative_power.hjorth_mobility', 'POW.AF3.betaH_relative_power.hjorth_complexity', 'POW.AF3.gamma_relative_power.hjorth_mobility', 'POW.AF3.gamma_relative_power.hjorth_complexity', 'POW.AF3.relative_power.beta_alpha', 'POW.AF3.relative_power.beta_alpha_theta', 'POW.AF3.relative_power.theta_beta', 'POW.AF3.relative_power.theta_alpha_beta', 'POW.AF3.relative_power.inverse_alpha', 'POW.F7.theta_relative_power.hjorth_mobility', 'POW.F7.theta_relative_power.hjorth_complexity', 'POW.F7.alpha_relative_power.hjorth_mobility', 'POW.F7.alpha_relative_power.hjorth_complexity', 'POW.F7.betaL_relative_power.hjorth_mobility']

Last 20 feature names:
['POW.F8.relative_power.beta_alpha', 'POW.F8.relative_power.beta_alpha_theta', 'POW.F8.relative_power.theta_beta', 'POW.F8.relative_power.theta_alpha_beta', 'POW.F8.relative_power.inverse_alpha', 'POW.AF4.theta_relative_power.hjorth_mobility', 'POW.AF4.theta_relative_power.hjorth_complexity', 'POW.AF4.alpha_relative_power.hjorth_mobility', 'POW.AF4.alpha_relative_power.hjorth_complexity', 'POW.AF4.betaL_relative_power.hjorth_mobility', 'POW.AF4.betaL_relative_power.hjorth_complexity', 'POW.AF4.betaH_relative_power.hjorth_mobility', 'POW.AF4.betaH_relative_power.hjorth_complexity', 'POW.AF4.gamma_relative_power.hjorth_mobility', 'POW.AF4.gamma_relative_power.hjorth_complexity', 'POW.AF4.relative_power.beta_alpha', 'POW.AF4.relative_power.beta_alpha_theta', 'POW.AF4.relative_power.theta_beta', 'POW.AF4.relative_power.theta_alpha_beta', 'POW.AF4.relative_power.inverse_alpha']

n_secs_per_trial: 25
power_window: 5
step: 1
n_windows_per_trial: 21
Data shape: (2520, 180)
Label shape: (2520,)
Label distribution:
0     819
1    1281
2     420
Name: count, dtype: int64
Subject IDs shape: (2520,)
Number of subjects: 40
Expected samples: 2520
Actual data rows: 2520
Actual label rows: 2520
Train subjects: [ 0  1  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26
 27 28 29 30 32 33 35 36 37 39]
Valid subjects: [34 38]
Test subjects: [ 2  3 21 31]
Train/valid overlap: []
Train/test overlap: []
Valid/test overlap: []
x_train: (2142, 180)
x_val: (126, 180)
x_test: (252, 180)
y_train distribution:
0     693
1    1092
2     357
Name: count, dtype: int64
y_val distribution:
0    21
1    63
2    42
Name: count, dtype: int64
y_test distribution:
0    105
1    126
2     21
Name: count, dtype: int64
# train svm (select by f1)
Scaler=standard, PCA=None, components=180, explained_var=None
Scaler=standard, PCA=20, components=20, explained_var=0.4806449788198858
Scaler=standard, PCA=30, components=30, explained_var=0.5738920317305103
Scaler=standard, PCA=50, components=50, explained_var=0.7128481098456553
Scaler=standard, PCA=80, components=80, explained_var=0.8555076038116108
Scaler=standard, PCA=100, components=100, explained_var=0.9165243819199982
Scaler=standard, PCA=0.9, components=94, explained_var=0.900583177139951
Scaler=standard, PCA=0.95, components=116, explained_var=0.9509980394511195
Scaler=minmax, PCA=None, components=180, explained_var=None
Scaler=minmax, PCA=20, components=20, explained_var=0.548193511119898
Scaler=minmax, PCA=30, components=30, explained_var=0.6603998846199168
Scaler=minmax, PCA=50, components=50, explained_var=0.81663401570763
Scaler=minmax, PCA=80, components=80, explained_var=0.9445398879947886
Scaler=minmax, PCA=100, components=100, explained_var=0.97277792805703
Scaler=minmax, PCA=0.9, components=67, explained_var=0.9037350162261929
Scaler=minmax, PCA=0.95, components=83, explained_var=0.9505330566846456
Best validation macro F1: 0.480965548223193
Best validation accuracy: 0.5158730158730159
Best validation balanced accuracy: 0.4920634920634921
Best parameters: {'model_type': 'SVM', 'selection_metric': 'macro_f1', 'scaler': 'standard', 
'pca_n_components': 30, 'actual_pca_components': 30, 'pca_explained_variance': 0.5738920317305103, 
'kernel': 'rbf', 'C': 1, 'gamma': 'scale', 'class_weight': 'balanced', 
'val_accuracy': 0.5158730158730159, 'val_macro_f1': 0.480965548223193, 'val_balanced_accuracy': 0.4920634920634921}
**Test accuracy: 0.3412698412698413
Test macro F1: 0.29324956638880534
Test weighted F1: 0.3654335607901937
Test balanced accuracy: 0.30952380952380953**
Classification report:
              precision    recall  f1-score   support

 mild_stress       0.39      0.29      0.33       105
      stress       0.47      0.40      0.44       126
 high_stress       0.07      0.24      0.11        21

    accuracy                           0.34       252
   macro avg       0.31      0.31      0.29       252
weighted avg       0.41      0.34      0.37       252

Confusion matrix:
[[30 50 25]
 [37 51 38]
 [ 9  7  5]]

# best svm model
**pca: None
scaler: StandardScaler()
param: {'kernel': 'rbf', 'C': 100, 'gamma': 'scale', 'class_weight': 'balanced'}
feature number: 180**
Validation accuracy: 0.3492063492063492
Validation macro F1: 0.2538690476190476
Validation weighted F1: 0.32872023809523804
Validation balanced accuracy: 0.2724867724867725
Validation classification report:
              precision    recall  f1-score   support

 mild_stress       0.07      0.14      0.09        21
      stress       0.49      0.60      0.54        63
 high_stress       0.50      0.07      0.12        42

    accuracy                           0.35       126
   macro avg       0.35      0.27      0.25       126
weighted avg       0.43      0.35      0.33       126

Validation confusion matrix:
[[ 3 18  0]
 [22 38  3]
 [18 21  3]]
**Test accuracy: 0.44047619047619047
Test macro F1: 0.3329222457129434
Test weighted F1: 0.4193093727977449
Test balanced accuracy: 0.34603174603174597
Test classification report:**
              precision    recall  f1-score   support

 mild_stress       0.36      0.23      0.28       105
      stress       0.52      0.67      0.58       126
 high_stress       0.13      0.14      0.14        21

    accuracy                           0.44       252
   macro avg       0.34      0.35      0.33       252
weighted avg       0.42      0.44      0.42       252

Test confusion matrix:
[[24 68 13]
 [35 84  7]
 [ 8 10  3]]


## epocx_relative_power_hjorth_features_plus:
5 relative power bands * 2 Hjorth features
+ 5 ratio features * 2 Hjorth features
= 20 features / channel
Absolute band power sequence shape: (120, 25, 12, 5)
Relative band power sequence shape: (120, 25, 12, 5)
Meaning: n_trials, n_secs, n_channels, n_bands
Input EEG data shape: (120, 25, 12, 128)
Output relative-power Hjorth features shape: (2520, 240)
Features per channel: 20
Power window: 5
Step: 1
Number of windows per trial: 21
NaN count: 0
Inf count: 0
features shape: (2520, 240)
number of feature names: 240
label shape: (2520,)
Feature DataFrame shape: (2520, 241)

First 20 feature names:
['POW.AF3.theta_relative_power.hjorth_mobility', 'POW.AF3.theta_relative_power.hjorth_complexity', 'POW.AF3.alpha_relative_power.hjorth_mobility', 'POW.AF3.alpha_relative_power.hjorth_complexity', 'POW.AF3.betaL_relative_power.hjorth_mobility', 'POW.AF3.betaL_relative_power.hjorth_complexity', 'POW.AF3.betaH_relative_power.hjorth_mobility', 'POW.AF3.betaH_relative_power.hjorth_complexity', 'POW.AF3.gamma_relative_power.hjorth_mobility', 'POW.AF3.gamma_relative_power.hjorth_complexity', 'POW.AF3.beta_alpha_relative_power_ratio.hjorth_mobility', 'POW.AF3.beta_alpha_relative_power_ratio.hjorth_complexity', 'POW.AF3.beta_alpha_theta_relative_power_ratio.hjorth_mobility', 'POW.AF3.beta_alpha_theta_relative_power_ratio.hjorth_complexity', 'POW.AF3.theta_beta_relative_power_ratio.hjorth_mobility', 'POW.AF3.theta_beta_relative_power_ratio.hjorth_complexity', 'POW.AF3.theta_alpha_beta_relative_power_ratio.hjorth_mobility', 'POW.AF3.theta_alpha_beta_relative_power_ratio.hjorth_complexity', 'POW.AF3.inverse_alpha_relative_power_ratio.hjorth_mobility', 'POW.AF3.inverse_alpha_relative_power_ratio.hjorth_complexity']

Last 20 feature names:
['POW.AF4.theta_relative_power.hjorth_mobility', 'POW.AF4.theta_relative_power.hjorth_complexity', 'POW.AF4.alpha_relative_power.hjorth_mobility', 'POW.AF4.alpha_relative_power.hjorth_complexity', 'POW.AF4.betaL_relative_power.hjorth_mobility', 'POW.AF4.betaL_relative_power.hjorth_complexity', 'POW.AF4.betaH_relative_power.hjorth_mobility', 'POW.AF4.betaH_relative_power.hjorth_complexity', 'POW.AF4.gamma_relative_power.hjorth_mobility', 'POW.AF4.gamma_relative_power.hjorth_complexity', 'POW.AF4.beta_alpha_relative_power_ratio.hjorth_mobility', 'POW.AF4.beta_alpha_relative_power_ratio.hjorth_complexity', 'POW.AF4.beta_alpha_theta_relative_power_ratio.hjorth_mobility', 'POW.AF4.beta_alpha_theta_relative_power_ratio.hjorth_complexity', 'POW.AF4.theta_beta_relative_power_ratio.hjorth_mobility', 'POW.AF4.theta_beta_relative_power_ratio.hjorth_complexity', 'POW.AF4.theta_alpha_beta_relative_power_ratio.hjorth_mobility', 'POW.AF4.theta_alpha_beta_relative_power_ratio.hjorth_complexity', 'POW.AF4.inverse_alpha_relative_power_ratio.hjorth_mobility', 'POW.AF4.inverse_alpha_relative_power_ratio.hjorth_complexity']

