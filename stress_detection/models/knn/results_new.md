## epocx_bandpower_hjorth_features:
"Scaler=standard, PCA=0.9, components=99, explained_var=0.9021\n",
"Scaler=standard, PCA=0.95, components=122, explained_var=0.9508\n",
"Scaler=standard, PCA=0.98, components=142, explained_var=0.9807\n",
"Scaler=standard, PCA=20, components=20, explained_var=0.4593\n",
"Scaler=standard, PCA=30, components=30, explained_var=0.5523\n",
"Scaler=standard, PCA=50, components=50, explained_var=0.6934\n",
"Scaler=standard, PCA=80, components=80, explained_var=0.8421\n",
"Scaler=standard, PCA=100, components=100, explained_var=0.9046\n",
"Scaler=minmax, PCA=0.9, components=70, explained_var=0.9026\n",
"Scaler=minmax, PCA=0.95, components=91, explained_var=0.9511\n",
"Scaler=minmax, PCA=0.98, components=121, explained_var=0.9801\n",
"Scaler=minmax, PCA=20, components=20, explained_var=0.5341\n",
"Scaler=minmax, PCA=30, components=30, explained_var=0.6456\n",
"Scaler=minmax, PCA=50, components=50, explained_var=0.8011\n",
"Scaler=minmax, PCA=80, components=80, explained_var=0.9321\n",
"Scaler=minmax, PCA=100, components=100, explained_var=0.9623\n",
"Best validation accuracy: 0.47619047619047616\n",
"Best parameters: {'scaler': 'standard', 'pca_n_components': 30, 'actual_pca_components': 30, 'pca_explained_variance': 0.552334244136798, 'n_neighbors': 28, 'p': 1, 'weights': 'distance'}\n",
"Final train+valid shape before PCA: (2268, 180)\n",
"Final train+valid shape after PCA: (2268, 30)\n",
"Test shape before PCA: (252, 180)\n",
"Test shape after PCA: (252, 30)\n",
"Final PCA explained variance: 0.5504293017479527\n",
"Test accuracy: 0.44841269841269843\n",

"Classification report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      " mild_stress       0.31      0.12      0.18       105\n",
      "      stress       0.49      0.79      0.60       126\n",
      " high_stress       0.00      0.00      0.00        21\n",
      "\n",
      "    accuracy                           0.45       252\n",
      "   macro avg       0.27      0.31      0.26       252\n",
      "weighted avg       0.37      0.45      0.38       252\n",
      "\n",
      "Confusion matrix:\n",
      "[[ 13  88   4]\n",
      " [ 25 100   1]\n",
      " [  4  17   0]]\n",

"Model saved to: models/knn\\knn_pca_bandpower_hjorth_model.joblib\n",
"Saved original feature number: 180\n",
"Saved PCA feature number: 30\n",
"First 10 original features: ['POW.AF3.theta_power.hjorth_mobility', 'POW.AF3.theta_power.hjorth_complexity', 'POW.AF3.alpha_power.hjorth_mobility', 'POW.AF3.alpha_power.hjorth_complexity', 'POW.AF3.betaL_power.hjorth_mobility', 'POW.AF3.betaL_power.hjorth_complexity', 'POW.AF3.betaH_power.hjorth_mobility', 'POW.AF3.betaH_power.hjorth_complexity', 'POW.AF3.gamma_power.hjorth_mobility', 'POW.AF3.gamma_power.hjorth_complexity']\n"


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
# train knn model

--select by best acc.:
Scaler=standard, PCA=0.9, components=94, explained_var=0.9006
Scaler=standard, PCA=0.95, components=116, explained_var=0.9510
Scaler=standard, PCA=0.98, components=135, explained_var=0.9812
Scaler=standard, PCA=20, components=20, explained_var=0.4806
Scaler=standard, PCA=30, components=30, explained_var=0.5739
Scaler=standard, PCA=50, components=50, explained_var=0.7128
Scaler=standard, PCA=80, components=80, explained_var=0.8555
Scaler=standard, PCA=100, components=100, explained_var=0.9165
Scaler=minmax, PCA=0.9, components=67, explained_var=0.9037
Scaler=minmax, PCA=0.95, components=83, explained_var=0.9505
Scaler=minmax, PCA=0.98, components=109, explained_var=0.9804
Scaler=minmax, PCA=20, components=20, explained_var=0.5482
Scaler=minmax, PCA=30, components=30, explained_var=0.6604
Scaler=minmax, PCA=50, components=50, explained_var=0.8166
Scaler=minmax, PCA=80, components=80, explained_var=0.9445
Scaler=minmax, PCA=100, components=100, explained_var=0.9728
Best validation accuracy: 0.4444444444444444
Best parameters: {'scaler': 'minmax', 'pca_n_components': 100, 'actual_pca_components': 100, 'pca_explained_variance': 0.97277792805703, 'n_neighbors': 30, 'p': 1, 'weights': 'distance'}
Final train+valid shape before PCA: (2268, 180)
Final train+valid shape after PCA: (2268, 100)
Test shape before PCA: (252, 180)
Test shape after PCA: (252, 100)
Final PCA explained variance: 0.9727254986121571
Test accuracy: 0.4722222222222222
Classification report:
              precision    recall  f1-score   support

 mild_stress       0.32      0.07      0.11       105
      stress       0.49      0.89      0.63       126
 high_stress       0.00      0.00      0.00        21

    accuracy                           0.47       252
   macro avg       0.27      0.32      0.25       252
weighted avg       0.38      0.47      0.36       252

Confusion matrix:
[[  7  97   1]
 [ 13 112   1]
 [  2  19   0]]
Model saved to: models/knn\knn_pca_bandpower_hjorth_model.joblib
Saved original feature number: 180
Saved PCA feature number: 100
First 10 original features: ['POW.AF3.theta_relative_power.hjorth_mobility', 'POW.AF3.theta_relative_power.hjorth_complexity', 'POW.AF3.alpha_relative_power.hjorth_mobility', 'POW.AF3.alpha_relative_power.hjorth_complexity', 'POW.AF3.betaL_relative_power.hjorth_mobility', 'POW.AF3.betaL_relative_power.hjorth_complexity', 'POW.AF3.betaH_relative_power.hjorth_mobility', 'POW.AF3.betaH_relative_power.hjorth_complexity', 'POW.AF3.gamma_relative_power.hjorth_mobility', 'POW.AF3.gamma_relative_power.hjorth_complexity']

--select by best macro-f1:

Scaler=standard, PCA=0.9, components=94, explained_var=0.9006
Scaler=standard, PCA=0.95, components=116, explained_var=0.9510
Scaler=standard, PCA=0.98, components=135, explained_var=0.9812
Scaler=standard, PCA=20, components=20, explained_var=0.4806
Scaler=standard, PCA=30, components=30, explained_var=0.5739
Scaler=standard, PCA=50, components=50, explained_var=0.7128
Scaler=standard, PCA=80, components=80, explained_var=0.8555
Scaler=standard, PCA=100, components=100, explained_var=0.9165
Scaler=minmax, PCA=0.9, components=67, explained_var=0.9037
Scaler=minmax, PCA=0.95, components=83, explained_var=0.9505
Scaler=minmax, PCA=0.98, components=109, explained_var=0.9804
Scaler=minmax, PCA=20, components=20, explained_var=0.5482
Scaler=minmax, PCA=30, components=30, explained_var=0.6604
Scaler=minmax, PCA=50, components=50, explained_var=0.8166
Scaler=minmax, PCA=80, components=80, explained_var=0.9445
Scaler=minmax, PCA=100, components=100, explained_var=0.9728
Best validation macro F1: 0.3902233061796523
Best validation accuracy: 0.42857142857142855
Best validation balanced accuracy: 0.3941798941798942
Best parameters: {'selection_metric': 'macro_f1', 'scaler': 'minmax', 
'pca_n_components': 30, 'actual_pca_components': 30, 'pca_explained_variance': 0.6603998846199168, 
'n_neighbors': 1, 'p': 1, 'weights': 'uniform', 
'val_accuracy': 0.42857142857142855, 'val_macro_f1': 0.3902233061796523, 'val_balanced_accuracy': 0.3941798941798942}
**Test accuracy: 0.4007936507936508
Test macro F1: 0.33128373000225386
Test weighted F1: 0.40027479595543786
Test balanced accuracy: 0.3592592592592592**
Classification report:
              precision    recall  f1-score   support

 mild_stress       0.35      0.23      0.28       105
      stress       0.52      0.56      0.54       126
 high_stress       0.13      0.29      0.18        21

    accuracy                           0.40       252
   macro avg       0.33      0.36      0.33       252
weighted avg       0.42      0.40      0.40       252

Confusion matrix:
[[24 57 24]
 [38 71 17]
 [ 6  9  6]]
Model saved to: models/knn\knn_pca_relative_bandpower_hjorth_model.joblib
Saved original feature number: 180
Saved PCA feature number: 30
First 10 original features: ['POW.AF3.theta_relative_power.hjorth_mobility', 'POW.AF3.theta_relative_power.hjorth_complexity', 'POW.AF3.alpha_relative_power.hjorth_mobility', 'POW.AF3.alpha_relative_power.hjorth_complexity', 'POW.AF3.betaL_relative_power.hjorth_mobility', 'POW.AF3.betaL_relative_power.hjorth_complexity', 'POW.AF3.betaH_relative_power.hjorth_mobility', 'POW.AF3.betaH_relative_power.hjorth_complexity', 'POW.AF3.gamma_relative_power.hjorth_mobility', 'POW.AF3.gamma_relative_power.hjorth_complexity']

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


# train knn (select by best acc.)
n_secs_per_trial: 25
power_window: 5
step: 1
n_windows_per_trial: 21
Data shape: (2520, 240)
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
x_train: (2142, 240)
x_val: (126, 240)
x_test: (252, 240)
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
Scaler=standard, PCA=0.9, components=134, explained_var=0.9018
Scaler=standard, PCA=0.95, components=165, explained_var=0.9507
Scaler=standard, PCA=0.98, components=194, explained_var=0.9801
Scaler=standard, PCA=20, components=20, explained_var=0.3895
Scaler=standard, PCA=30, components=30, explained_var=0.4829
Scaler=standard, PCA=50, components=50, explained_var=0.6202
Scaler=standard, PCA=80, components=80, explained_var=0.7540
Scaler=standard, PCA=100, components=100, explained_var=0.8189
Scaler=minmax, PCA=0.9, components=82, explained_var=0.9003
Scaler=minmax, PCA=0.95, components=114, explained_var=0.9507
Scaler=minmax, PCA=0.98, components=155, explained_var=0.9801
Scaler=minmax, PCA=20, components=20, explained_var=0.5263
Scaler=minmax, PCA=30, components=30, explained_var=0.6391
Scaler=minmax, PCA=50, components=50, explained_var=0.7869
Scaler=minmax, PCA=80, components=80, explained_var=0.8955
Scaler=minmax, PCA=100, components=100, explained_var=0.9330
Best validation accuracy: 0.49206349206349204
Best parameters: {'scaler': 'standard', 'pca_n_components': 20, 'actual_pca_components': 20, 'pca_explained_variance': 0.389489851504399, 'n_neighbors': 27, 'p': 2, 'weights': 'distance'}
Final train+valid shape before PCA: (2268, 240)
Final train+valid shape after PCA: (2268, 20)
Test shape before PCA: (252, 240)
Test shape after PCA: (252, 20)
Final PCA explained variance: 0.38746286669491325
Test accuracy: 0.45634920634920634
Classification report:
              precision    recall  f1-score   support

 mild_stress       0.36      0.13      0.19       105
      stress       0.48      0.80      0.60       126
 high_stress       0.00      0.00      0.00        21

    accuracy                           0.46       252
   macro avg       0.28      0.31      0.26       252
weighted avg       0.39      0.46      0.38       252

Confusion matrix:
[[ 14  91   0]
 [ 24 101   1]
 [  1  20   0]]
Model saved to: models/knn\knn_pca_hjorth_plus_model.joblib
Saved original feature number: 240
Saved PCA feature number: 20
First 10 original features: ['POW.AF3.theta_relative_power.hjorth_mobility', 'POW.AF3.theta_relative_power.hjorth_complexity', 'POW.AF3.alpha_relative_power.hjorth_mobility', 'POW.AF3.alpha_relative_power.hjorth_complexity', 'POW.AF3.betaL_relative_power.hjorth_mobility', 'POW.AF3.betaL_relative_power.hjorth_complexity', 'POW.AF3.betaH_relative_power.hjorth_mobility', 'POW.AF3.betaH_relative_power.hjorth_complexity', 'POW.AF3.gamma_relative_power.hjorth_mobility', 'POW.AF3.gamma_relative_power.hjorth_complexity']