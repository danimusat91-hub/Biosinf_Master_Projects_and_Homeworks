import numpy as np
from sklearn.svm import SVC
from utils import getUA,getWA
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## Train data
X_train = np.load("./Data_filtered_500ms_exercises_Train.npy")
Y_train = np.load("./Labels_filtered_500ms_exercises_Train.npy")


## Val Data
X_val = np.load("./Data_filtered_500ms_exercises_Val.npy")
Y_val = np.load("./Labels_filtered_500ms_exercises_Val.npy")


## Test Data
X_test = np.load("./Data_filtered_500ms_exercises_Test.npy")
Y_test = np.load("./Labels_filtered_500ms_exercises_Test.npy")

# ### Clasifier


# 1. Focused Parameter Lists
# Using logarithmic scales (0.1, 1, 10) is more effective than linear scales
c_list = [0.1, 1, 10]
# We only use RBF (the gold standard) and Linear (the fastest)
kernel_list = ['rbf', 'linear']
# 'scale' is a smart default that handles most cases
gamma_list = ['scale', 0.01, 0.1]

total_combinations = len(c_list) * len(kernel_list) * len(gamma_list)
best_score = 0
best_params = {}
count = 0

print(f"Starting Efficient SVM Search: {total_combinations} combinations.\n")
print(f"{'Iter':<5} | {'Acc':<7} | {'C':<5} | {'Kernel':<8} | {'Gamma':<7}")
print("-" * 45)
best_model = None

for c in c_list:
    for kern in kernel_list:
        for gam in gamma_list:
            # Skip gamma if kernel is linear (gamma is ignored in linear SVM)
            if kern == 'linear' and gam != 'scale':
                continue
                
            count += 1
            model = SVC(C=c, kernel=kern, gamma=gam, random_state=42)
            model.fit(X_train, Y_train)
            
            # 3. Calculate Accuracy on the train set
            OUT_train = model.predict(X_train)
            UA_train = getUA(to_categorical(OUT_train,3), to_categorical(Y_train,3),3)
            WA_train = getWA(to_categorical(OUT_train,3),to_categorical(Y_train,3))
            
            
            # 3. Calculate Accuracy on the val set
            OUT_val = model.predict(X_val)
            UA_val = getUA(to_categorical(OUT_val,3), to_categorical(Y_val,3),3)
            WA_val = getWA(to_categorical(OUT_val,3),to_categorical(Y_val,3))
            
            current_score = UA_val
            
            print(f"{count:<5} | {current_score:.4f}  | {c:<5} | {kern:<8} | {str(gam):<7}")

            if current_score > best_score:
                best_score = current_score
                best_model = model
                best_params = {
                    'C': c,
                    'Kernel': kern,
                    'gamma': gam,
                }


# 5. Final Evaluation for the Documentation
print("\n" + "="*40)
print(f"WINNING PARAMETERS: {best_params}")
print(f"BEST VALIDATION ACCURACY: {best_score:.4f}")
print("="*40)

test_preds = best_model.predict(X_test)
UA_val = getUA(to_categorical(test_preds,3), to_categorical(Y_test,3),3)
print("\nFINAL TEST SET METRICS (RECAP):%d",UA_val)
report = classification_report(Y_test, test_preds)
print(report)

conf_matrix = confusion_matrix(Y_test, test_preds)

# Vizualizați matricea de confuzie
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('MLP Confusion Matrix')
plt.show()
