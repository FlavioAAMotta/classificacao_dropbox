PATH = "."
LOG_ID = "Pop1"
WINDOWS = 8

import pandas as pd
from functools import reduce

arqPath = PATH + "/target_" + LOG_ID + ".txt"
arqClasses = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")

arqPath = PATH + "/access_" + LOG_ID + ".txt"
arqAcessos = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")

df_class_final = pd.DataFrame( index=arqClasses.index )
df_acc_final = pd.DataFrame( index=arqAcessos.index )

win_start = 1 # start from the second window
for window in range(win_start, WINDOWS+1):
    '''
    #window 0+1: ini = 0    fim = 4
    #window 1+1: ini = 4    fim = 8
    '''
    ini = (window-1)*4
    fim = window*4
    
    df_class = arqClasses.iloc[:, ini:fim]
    df_class = pd.DataFrame( index=df_class.index,
        data=[reduce((lambda x,y: x or y), [int(val) for val in labels]) for namespace, labels in df_class.iterrows()] )
    
    df_class_final = pd.concat( [df_class_final, df_class], axis=1 )
    
    df_acc = arqAcessos.iloc[:, ini:fim]
    df_acc = pd.DataFrame( index=df_acc.index,
        data=[reduce((lambda x,y: x + y), [int(val) for val in accs]) for namespace, accs in df_acc.iterrows()])
        
    df_acc_final = pd.concat( [df_acc_final, df_acc], axis=1 )
    

df_class_final.columns = [i for i in range(WINDOWS)]
arqPath = "./target_" + LOG_ID + "_" + str(WINDOWS) + "_win.txt"
df_class_final.to_csv(arqPath, sep = " ")

df_acc_final.columns = [i for i in range(WINDOWS)]
arqPath = "./access_" + LOG_ID + "_" + str(WINDOWS) + "_win.txt"
df_acc_final.to_csv(arqPath, sep = " ")
