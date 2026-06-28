from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from tabpfn import TabPFNClassifier
from imblearn.combine import SMOTEENN

df = pd.read_csv(r'Global_SPCN_trainset.csv')
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote_enn = SMOTEENN(random_state=42, n_jobs=24, sampling_strategy=0.5)
X_train, y_train = smote_enn.fit_resample(X_train, y_train)
X_train.to_csv('X_train_global_3_5.csv', index=False)
y_train.to_csv('y_train_global_3_5.csv', index=False)
X_test.to_csv('X_test_global_3_5.csv', index=False)
y_test.to_csv('y_test_global_3_5.csv', index=False)
model = TabPFNClassifier(model_path="tabpfn-v2.5-classifier-v2.5_default.ckpt", 
                         ignore_pretraining_limits=True,
                         device="cuda")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred) 
roc_auc = roc_auc_score(y_test, y_pred_proba) 
conf_matrix = confusion_matrix(y_test, y_pred)
print('Model Evaluation Metrics:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

df_external = pd.read_csv(r'Global_SPCN_testset.csv')
X_external = df_external.drop(columns=['label'])
y_external = df_external['label']
y_external_pred = model.predict(X_external)
y_external_pred_proba = model.predict_proba(X_external)[:, 1]
accuracy_external = accuracy_score(y_external, y_external_pred)
precision_external = precision_score(y_external, y_external_pred)
recall_external = recall_score(y_external, y_external_pred)
f1_external = f1_score(y_external, y_external_pred)
roc_auc_external = roc_auc_score(y_external, y_external_pred_proba)
conf_matrix_external = confusion_matrix(y_external, y_external_pred)
print('External Test Set Evaluation Metrics:')
print(f'Accuracy: {accuracy_external:.4f}')
print(f'Precision: {precision_external:.4f}')
print(f'Recall: {recall_external:.4f}')
print(f'F1 Score: {f1_external:.4f}')
print(f'ROC-AUC: {roc_auc_external:.4f}')
print('Confusion Matrix:')
print(conf_matrix_external)

joblib.dump(model, 'fkp_tabpfn_model.pkl')