import numpy as np
import pandas as pd
import torch
import tqdm
from tsai.models.MINIROCKET import *
from sklearn.linear_model import RidgeClassifier
from model.bert_encoder import bert_enc
from model.feature import data_preprocess2, data_preprocess3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from model.model import *
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score,roc_auc_score, accuracy_score

# Function to calculate sensitivity and specificity from a confusion matrix
def calculate_sensitivity_specificity(confusion_matrix):
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]

    sensitivity = TP / (TP + FN)  # True Positive Rate
    specificity = TN / (TN + FP)  # True Negative Rate

    return sensitivity, specificity

"""
Stage 2: Data preprocessing and model training
"""
device = torch.device('cuda:0')
save_path = 'preprocess_data_path'

# Load patient data from npz file
npz_file = np.load(save_path + 'combined_.npz', allow_pickle=True)
patient_data = npz_file['X']
labels = npz_file['y']

train_data, train_label = data_preprocess3(patient_data, labels)
train_data = train_data.astype(np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.1, random_state=42)

torch.cuda.set_device(0)
mrf = MiniRocketFeatures(c_in=train_data.shape[1], seq_len=train_data.shape[2]).to(device)
mrf.fit(train_data)
X_tfm = get_minirocket_features(train_data, mrf)

"""
Extract semantic information using BERT
"""
path = "text_patient.csv"
df = pd.read_csv(path)
sentences = []

# Generate sentence descriptions for patients based on their medical information
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    sentence = f"The body mass index is {row['BMI']:.2f},"
    if 18.5 <= row["BMI"] < 23.9:
        sentence += " which is normal weight."
    elif row["BMI"] < 18.5:
        sentence += " which indicates underweight."
    elif 23.9 <= row["BMI"] < 27.5:
        sentence += " which indicates overweight."
    elif 27.5 <= row["BMI"] < 32.5:
        sentence += " which indicates mild obesity."
    elif 32.5 <= row["BMI"] < 37.5:
        sentence += " which indicates moderate obesity."
    else:
        sentence += " which indicates severe obesity."

    if row["hypertension"] == 'yes':
        sentence += " The patient has hypertension."
    if row["diabetes mellitus"] == 'yes':
        sentence += " The patient has diabetes mellitus."
    if row["hyperlipoidemia"] == 'yes':
        sentence += " The patient has hyperlipoidemia."
    if row["heart disease"] == 'yes':
        sentence += " The patient has heart disease."
    if sentence.endswith("."):
        sentence = sentence[:-1] + f", and the age is {row['age']}."
    else:
        sentence += f" The age is {row['age']}."

    sentences.append(sentence)

# Encode sentences to get embeddings
sentence_embeddings = [bert_enc(sentence) for sentence in tqdm.tqdm(sentences)]
sentence_embeddings = torch.stack(sentence_embeddings)

# Combine features for classification
X_combined = np.concatenate([X_tfm, np.expand_dims(np.array(sentence_embeddings), axis=-1)], axis=1)



"""
classification
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier

# Split the combined data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_combined, train_label, test_size=0.3829, random_state=42)
print('Shape of the test set before reshaping:', X_test.shape)

# Reshape the training and test datasets as required by the classifiers
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print('Shape of the test set after reshaping:', X_test.shape)

# Train a Logistic Regression classifier
clf = LogisticRegression(random_state=0, max_iter=3000).fit(X_train, y_train)

# Train a Logistic Regression with Cross-Validation to find optimal regularization strength
clfcv = LogisticRegressionCV(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=3000).fit(X_train, y_train)

# Train a Ridge Classifier
clf_rig = RidgeClassifier(random_state=0).fit(X_train, y_train)

# Evaluate classifiers
score = clf.score(X_test, y_test)  # Accuracy of Logistic Regression
score1 = clfcv.score(X_test, y_test)  # Accuracy of Logistic Regression with CV
score2 = clf_rig.score(X_test, y_test)  # Accuracy of Ridge Classifier

# Make predictions using the trained classifiers
pre_LR = clf.predict(X_test)  # Predictions from Logistic Regression
pre_LRCV = clfcv.predict(X_test)  # Predictions from Logistic Regression with CV
pre_Rig = clf_rig.predict(X_test)  # Predictions from Ridge Classifier

# Optionally, print scores to verify the performance of the classifiers
print(f"Accuracy of Logistic Regression: {score}")
print(f"Accuracy of Logistic Regression with CV: {score1}")
print(f"Accuracy of Ridge Classifier: {score2}")
