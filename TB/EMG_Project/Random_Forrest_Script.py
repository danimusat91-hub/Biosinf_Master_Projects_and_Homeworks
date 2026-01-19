import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
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



# 1. Define all parameter ranges to explore
count = 0
n_estimators_list = [100, 300]
max_depth_list = [14, 16, None]
max_features_list = [ 0.5]
min_samples_split_list = [40, 60]
min_samples_leaf_list = [20 , 40]
bootstrap_list = [True]

total_combinations = len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list) * len(max_features_list)
best_accuracy = 0
best_params = {}
best_model = None
best_score = 0

print(f"Starting Efficient SVM Search: {total_combinations} combinations.\n")
print(f"{'n_estimators':<13} | {'depth':<7} | {'criterion':<10} | {'Val Accuracy':<12}")
print("-" * 55)


# 2. Nested loops for comprehensive tuning
for n_est in n_estimators_list:
    for depth in max_depth_list:
        for feat in max_features_list:
            for split in min_samples_split_list:
                for leaf in min_samples_leaf_list:
                    for boot in bootstrap_list:
                        count += 1
                        # Initialize the model with current combination
                        model = RF(
                            n_estimators=n_est,
                            max_depth=depth,
                            max_features=feat,
                            min_samples_split=split,
                            min_samples_leaf=leaf,
                            bootstrap=boot,
                            n_jobs=-1,      # Use all CPU cores for each fit
                        )
                        
                      # Train on your pre-extracted training set
                        model.fit(X_train, Y_train)
                      
                      # 3. Calculate Accuracy on the train set
                        OUT_train = model.predict(X_train)
                        UA_train = getUA(to_categorical(OUT_train,3), to_categorical(Y_train,3),3)
                        WA_train = getWA(to_categorical(OUT_train,3), to_categorical(Y_train,3))
                        train_score = UA_train
                      
                      # 3. Calculate Accuracy on the val set
                        OUT_val = model.predict(X_val)
                        UA_val = getUA(to_categorical(OUT_val,3), to_categorical(Y_val,3),3)
                        WA_val = getWA(to_categorical(OUT_val,3), to_categorical(Y_val,3))
                      
                        current_score = UA_val

                        print("UA_val")
                        print(f"{count:<5} | {current_score:.4f}  | {train_score:.4f}  |{n_est:<5} | {str(depth):<5} | {str(feat):<5} | {split:<5} | {leaf:<5} | {str(boot):<5}")
                        print("WA_val")
                        print(f"{count:<5} | {WA_val:.4f}  | {WA_train:.4f}  |{n_est:<5} | {str(depth):<5} | {str(feat):<5} | {split:<5} | {leaf:<5} | {str(boot):<5}")

                        # Check if this is our best model so far
                        if current_score > best_score:
                            best_score = current_score
                            best_model = model
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'max_features': feat,
                                'min_samples_split': split,
                                'min_samples_leaf': leaf,
                                'bootstrap': boot
                            }


# 5. Final Evaluation for the Documentation
print("\n" + "="*40)
print(f"WINNING PARAMETERS: {best_params}")
print(f"BEST VALIDATION ACCURACY: {best_score:.4f}")
print("="*40)

test_preds = best_model.predict(X_test)
UA_val = getUA(to_categorical(test_preds,3), to_categorical(Y_test,3),3)
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





