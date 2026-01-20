from Load_Data import LoadData
from process_Data import processData
import numpy as np
from utils import clean_signal_list
from sklearn.utils import shuffle 



# Load Data

Data_Loader = LoadData("./sEmg_databases")
X, Y = Data_Loader.loadData_armthreeClasses()
# X, Y = shuffle(X,Y)

# ################ Parameters ################################

T_obs = 60
Ts = 60/30720
Fs = 512
Window_Size = 0.5 # 256 samples
Overlapping =  0.5


### ################# Preprocess data ###########################

### Filter the signal 
for count, subject in enumerate(X):
  X[count] = clean_signal_list(subject)




############# Ckeck for Asymtery

Data_process_Object = processData(X,Y,Fs,Overlapping,Window_Size)
Health_Status = Data_process_Object.calculateAsymetry("RMS")
