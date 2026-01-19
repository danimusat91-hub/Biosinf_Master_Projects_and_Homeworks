from Load_Data import LoadData
from process_Data import processData
import numpy as np
from utils import *
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler



# Load Data

Data_Loader = LoadData("./sEmg_databases")
X, Y = Data_Loader.loadData_armthreeClasses()
X, Y = shuffle(X,Y)

# ################ Parameters ################################

T_obs = 60
Ts = 60/30720
Fs = 512
Window_Size = 0.5 # 256 samples
Overlapping =  0.5


### ################# Preprocess data ###########################

### Filter the signal and normalize it
for count, subject in enumerate(X):
  X[count] = clean_signal_list(subject)



# ################## Plot some of the Signals
plot_emg_analysis(X[4])
plot_emg_analysis(X[5])
plot_emg_analysis(X[6])

############# Ckeck for Asymtery

Data_process_Object = processData(X,Y,Fs,Overlapping,Window_Size)
Health_Status = Data_process_Object.calculateAsymetry("RMS")



############# Window and Test, Validation, Train Separation ##################

## Training set
Data_process_Object = processData(X[0:210],Y[0:210],Fs,Overlapping,Window_Size)
X_train, Y_train = Data_process_Object.extractArmFeatures()
X_train = np.array(X_train,dtype=np.float32)
Y_train = np.array(Y_train)
shape = X_train.shape
data_reshaped = X_train.reshape(-1, shape[-1])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reshaped)
X_train = data_scaled.reshape(shape)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_train = X_train_reshaped
X_train, Y_train = shuffle(X_train,Y_train)
np.save("./Data_filtered_500ms_exercises_Train.npy", X_train)
np.save("./Labels_filtered_500ms_exercises_Train.npy", Y_train)


## Validation set
Data_process_Object = processData(X[210:255],Y[210:255],Fs,Overlapping,Window_Size)
X_val, Y_val = Data_process_Object.extractArmFeatures()
X_val = np.array(X_val,dtype=np.float32)
Y_val = np.array(Y_val)
shape = X_val.shape
data_reshaped = X_val.reshape(-1, shape[-1])
data_scaled = scaler.fit_transform(data_reshaped)
X_val = data_scaled.reshape(shape)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
X_val = X_val_reshaped
X_val, Y_val = shuffle(X_val,Y_val)
np.save("./Data_filtered_500ms_exercises_Val.npy", X_val)
np.save("./Labels_filtered_500ms_exercises_Val.npy", Y_val)

## Test set
Data_process_Object = processData(X[255:303],Y[255:303],Fs,Overlapping,Window_Size)
X_test, Y_test = Data_process_Object.extractArmFeatures()
X_test = np.array(X_test,dtype=np.float32)
Y_test = np.array(Y_test)
shape = X_test.shape
data_reshaped = X_test.reshape(-1, shape[-1])
data_scaled = scaler.fit_transform(data_reshaped)
X_test = data_scaled.reshape(shape)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
X_test = X_test_reshaped
X_test, Y_test = shuffle(X_test,Y_test)
np.save("./Data_filtered_500ms_exercises_Test.npy", X_test)
np.save("./Labels_filtered_500ms_exercises_Test.npy", Y_test)





# METRIX = []
# SKF = KfCV(n_splits=5)
# for idx, (idx_train, idx_val) in enumerate(SKF.split(X_reshaped, Y)):
#     X_train = X_reshaped[idx_train]
#     Y_train = Y[idx_train]
#     Y_val = Y[idx_val]
#     X_val = X_reshaped[idx_val]

#     MODEL = RF(n_estimators=200, max_depth=14) #, min_samples_split=0.1,min_samples_leaf=0.1, max_samples=0.5)
#     MODEL.fit(X_train, Y_train)
#     OUT_train = MODEL.predict(X_train)
#     OUT_val = MODEL.predict(X_val)

#     UA_train = getUA(codeOneHot(OUT_train,3),
#                      codeOneHot(Y_train,3),3)
#     WA_train = getWA(codeOneHot(OUT_train,3),
#                      codeOneHot(Y_train,3))
#     UA_val = getUA(codeOneHot(OUT_val,3), codeOneHot(Y_val,3),3)
#     WA_val = getWA(codeOneHot(OUT_val,3), codeOneHot(Y_val,3))
#     METRIX += [UA_train, WA_train, UA_val, WA_val]

#     print(METRIX)



