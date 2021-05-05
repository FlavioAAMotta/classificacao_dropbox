#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from collections import defaultdict
from matplotlib import pyplot as plt
from optparse import OptionParser
from functools import reduce
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


#*******************************************************************************
# constant
#*******************************************************************************
## List of classifiers
ONLINE, SVMR, SVML, SVMS, RF, KNN, DCT, LR, VR, VS = [i for i in range(10)] 
CLF_NAMES = ['ONLINE', 'SVMR', 'SVML', 'SVMS', 'RF', 'KNN', 'DCT', 'LR', 'VR', 'VS']

## Price constants
## https://cloud.google.com/storage/pricing#operations-pricing
## https://azure.microsoft.com/en-us/pricing/details/storage/blobs/
## https://aws.amazon.com/s3/pricing
H, W, C = [0, 1, 2]
## Cost Storage {Hot, Warm, Cold} based on AMAZON S3 per GB
CSH, CSW, CSC = [0.0230, 0.0125, 0.0040]
## Cost Operations {Hot, Warm, Cold} based on AMAZON S3 per 1000 operations
COH, COW, COC = [0.0004, 0.0010, 0.0500]
## Cost Retrivals {Hot, Warm, Cold} based on AMAZON S3 per GB
CRH, CRW, CRC = [0.0000, 0.0100, 0.0100]


#*******************************************************************************
# usage
#*******************************************************************************
def process_opt():
    usage = "usage: %prog [options] files\n"
    parser = OptionParser(usage=usage)
    parser.add_option("-l", "--log_id", dest="LOG_ID", help="log site id")
    parser.add_option("-p", "--path", dest="PATH", help="path for input files")
    parser.add_option("-w", "--windows", dest="WINDOWS", help="total of time windows", type="int")
    parser.add_option("-c", "--classifier", dest="CLASSIFIER", default=0, type=int, help="1-9 (default 0-ONLINE)")
    parser.add_option("-t", "--train_lag", dest="TRAIN_LAG", default=1, type=int, help="(default 1)")
    parser.add_option("-s", "--sweight", dest="SWEIGHT", default=0, type=int, help="Sample weight (default 0)")    
    opt, files = parser.parse_args()
    if not( opt.LOG_ID ) or not( opt.PATH ) or not( opt.WINDOWS ):
        parser.print_help()
        sys.exit(1)
    return opt, files


#*******************************************************************************
# Functions
#*******************************************************************************


def obj_cost(vol_gb, acc, obj_class):
    acc_1k = float(acc)/1000.0     # acc prop to 1000
    if obj_class == H:      # hot
        return vol_gb*CSH + acc_1k*COH + vol_gb*acc*CRH
    elif obj_class == W:    # warm
        return vol_gb*CSW + acc_1k*COW + vol_gb*acc*CRW
    else:                   # cold
        return vol_gb*CSC + acc_1k*COC + vol_gb*acc*CRC


def thr_acc(vol_gb, obj_class):
    if obj_class == 'HW':   # hot to warm
        return int(vol_gb*(CSW-CSH)/(COH-COW-vol_gb*1000*(CRW-CRH)))
    else:                   # warm to cold
        return int(vol_gb*(CSC-CSW)/(COW-COC-vol_gb*1000*(CRC-CRW)))


def Acerto_Erros(df, window, log_id, clf_name):

    AQ = EF = EQ = AF = 0
    CAQ = CEF = CEQ = CAF = 0.0
    CH0 = CW0 = CC0 = 0.0
    CH2 = CW2 = CC2 = 0.0
    CH3 = CW3 = CC3 = 0.0    
    QQ0 = QW0 = QF0 = 0
    QQ2 = QW2 = QF2 = 0
    
    for index, row in df.iterrows():
        vol_gb = float(row['vol_bytes'])/(1024.0**3)    # vol per GB
        #vol_gb = 1.0 #DEBUG
        acc_fut = row['acc_fut']
        
        #Limiares de acesso para camadas H-W e W-C
        accThresHW = thr_acc(vol_gb, 'HW')
        
        #Custo optimo
        if ( acc_fut > accThresHW ): # HOT
            CH0 += obj_cost(vol_gb, acc_fut, H)
            QQ0 += 1
        else:                        # WARM
            CW0 += obj_cost(vol_gb, acc_fut, W)
            QW0 += 1
        
        #Desempenho do classificador e custos de erros/acertos
        if (row['label'] == 0 and row['pred'] == 0) : #TN
            AF  += 1
            CAF += obj_cost(vol_gb, acc_fut, W)
        elif(row['label'] == 0 and row['pred'] == 1): #FP 
            EQ  += 1                                  # cost FP:
            CEQ += obj_cost(vol_gb, acc_fut, W)       # - penalty: the cost that should be saved in warm tier
            CEQ += obj_cost(vol_gb, acc_fut, H)       # - accesses in hot tier
        elif(row['label'] == 1 and row['pred'] == 0): #FN
            EF  += 1                                  # cost FN:
            CEF += obj_cost(vol_gb, 1, W)             # - one access to return to hot tier
            CEF += obj_cost(vol_gb, acc_fut-1, H)     # - accesses in hot tier
        elif(row['label'] == 1 and row['pred'] == 1): #TP
            AQ  += 1
            CAQ += obj_cost(vol_gb, acc_fut, H)
                
        #Custo simples sem optimizacao
        CH3 += obj_cost(vol_gb, acc_fut, H)
        CW3 += obj_cost(vol_gb, acc_fut, W)
        CC3 += obj_cost(vol_gb, acc_fut, C)

    opt_cost = CH0 + CW0 + CC0          #Custo0
    pred_cost = CAQ + CAF + CEQ + CEF   #Custo1
    default_cost = CH3                  #Custo3: always hot
    opt_rcs = (default_cost - opt_cost)/default_cost
    pred_rcs = (default_cost - pred_cost)/default_cost
    precision1, recall1, _ = precision_recall_curve(df['label'], df['pred'])    
    print("Custo0", window,df.shape[0],CH0,CW0,CC0)
    print("Quant0", window,df.shape[0],QQ0,QW0,QF0)
    print("Quant1", window,df.shape[0],AQ,AF,EQ,EF)
    print("Custo1", window,df.shape[0],CAQ,CAF,CEQ,CEF)
    print("Custo3", window,df.shape[0],CH3,CW3,CC3)
    print("Perform1 {} {} {} {} {} {}".format(window,df.shape[0], 
                                        accuracy_score(df['label'], df['pred']), 
                                        roc_auc_score(df['label'], df['pred']), 
                                        f1_score(df['label'], df['pred']), 
                                        auc(recall1,precision1)))    
    print("Perform", log_id, clf_name, 
                        window,df.shape[0], accuracy_score(df['label'], df['pred']),
                        roc_auc_score(df['label'], df['pred']),
                        f1_score(df['label'], df['pred']),
                        fbeta_score(df['label'], df['pred'] ,beta=2),
                        auc(recall1,precision1),
                        AQ/(AQ+EQ),  #precision TP/(TP+FP)
                        AQ/(AQ+EF),  #recall TP/(TP+FN)
                        pred_rcs, opt_rcs, pred_cost, opt_cost, default_cost,
                        (AQ+EF), (EQ+AF), AQ, EQ, EF, AF)


def make_estimators():
    estimators = []
    knn = KNeighborsClassifier()
    estimators.append(('KNN',knn))
    dct = DecisionTreeClassifier(max_depth=5)
    estimators.append(('DCT',dct))
    clf = RandomForestClassifier(n_estimators=100, random_state=2)
    estimators.append(('RandomForest',clf))
    logReg = LogisticRegression()
    estimators.append(('LRG',logReg))
    l_svm = svm.SVC(kernel='linear', C=0.00001, probability=True) # Linear Kernel
    estimators.append(('Linear SVM',l_svm))
    g_svm = svm.SVC(kernel='rbf', probability=True)
    estimators.append(('Gaussian SVM',g_svm))
    s_svm = svm.SVC(kernel='sigmoid', probability=True)
    estimators.append(('Sigmoid SVM',s_svm))
    return estimators
    

def set_clf(clf_opt):
    if clf_opt == SVMR:
        return svm.SVC(gamma='auto',tol=0.00001)
    elif clf_opt == SVML:
        return svm.SVC(kernel='linear', C=0.00001, probability=True) # Linear Kernel
    elif clf_opt == SVMS:
        return svm.SVC(kernel='sigmoid', probability=True)
    elif clf_opt == RF:
        return RandomForestClassifier(max_depth=10, n_estimators=50, n_jobs=-1)
        #return RandomForestClassifier(n_estimators=100, random_state=2)
    elif clf_opt == KNN:
        return KNeighborsClassifier()
    elif clf_opt == DCT:
        return DecisionTreeClassifier(max_depth=5)
    elif clf_opt == LR:
        return LogisticRegression()
    elif clf_opt == VR:
        estimators = make_estimators()        
        return VotingClassifier(estimators = estimators, voting ='hard', n_jobs=-1)
    elif clf_opt == VS:
        estimators = make_estimators()
        vs = VotingClassifier(estimators = estimators, voting ='soft', n_jobs=-1)
        return vs   # comment here to rum grid score in soft voting
        comb = itertools.product([0,0.5,1], repeat=7)
        params = []
        for i in list(comb): 
            params.append(list(i))            
        del params[0]
        params = {'weights': params }
        score = make_scorer(fbeta_score, beta=2)
        return GridSearchCV(estimator=vs, param_grid=params, scoring=score, n_jobs=-1)        
    else:
        print("Unknown classifier!")
    sys.exit(1)



#*******************************************************************************
# Main:
#*******************************************************************************
if __name__ == '__main__':
    opt, files = process_opt()
    
    np.random.seed(0)

    name_space = []
    arq = open(opt.PATH + "/nsJanelas_" + opt.LOG_ID + ".txt", 'r')
    for line in arq:
        name_space.append(
            sorted(
                list(
                    map(int, line.split())
                    )
                )
            )

    scaler = MinMaxScaler()

    arqPath = opt.PATH + "/access_" + opt.LOG_ID + ".txt"
    arqAcessos = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")
    #arqAcessos = pd.DataFrame( scaler.fit_transform(arqAcessos) ) # very bad results
    
    arqPath = opt.PATH + "/target_" + opt.LOG_ID + ".txt"
    arqClasses = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")

    arqPath = opt.PATH + "/vol_bytes_" + opt.LOG_ID + ".txt"
    arqVolBytes = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")
    
    ## extra features
    #arqPath = opt.PATH + "/usr_access_" + opt.LOG_ID + ".txt"
    #feature_acc = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")
    
    #arqPath = opt.PATH + "/usr_objects_" + opt.LOG_ID + ".txt"
    #feature_obj = pd.read_csv(arqPath, low_memory=False, sep=" ",  index_col="NameSpace")
    
    #arqPath = opt.PATH + "/usr_session_" + opt.LOG_ID + ".txt"
    #feature_ses = pd.read_csv(arqPath, low_memory=False, sep=" ", index_col="NameSpace")
    
    #arqPath = opt.PATH + "/usr_logins_" + opt.LOG_ID + ".txt"
    #feature_log = pd.read_csv(arqPath, low_memory=False, sep=" ",  index_col="NameSpace")
        
    clf = None  # this is my classifier variable
    
    win_start = win_train = 1 # start from the second window
    for window in range(win_start, opt.WINDOWS+1):
        '''
        #window 0+1: ini_train = 0    fim_train = 4    ini_test = 4    fim_test =  8    ini_eval =  8    fim_eval = 12
        #window 1+1: ini_train = 4    fim_train = 8    ini_test = 8    fim_test = 12    ini_eval = 12    fim_eval = 16
        '''    
        ini_train = (window-1)*4
        fim_train = window*4
        
        ini_test = window*4
        fim_test = (window+1)*4
        
        ini_eval = (window+1)*4
        fim_eval = (window+2)*4
        
        if opt.CLASSIFIER:
            
            if window == win_train:
                print("Window train:", window)
                win_train = window + opt.TRAIN_LAG  # next window for training
            
                #prepare data frame for training
                df_acc_train = arqAcessos.loc[name_space[window-1]].iloc[:,ini_train:fim_train]
                df_vol_train = arqVolBytes.loc[name_space[window-1]].iloc[:,fim_train-1:fim_train]
                df_train = pd.concat( [df_acc_train, df_vol_train], axis=1 )
                df_train.columns = ['acc1','acc2','acc3','acc4','vol']
                
                #prepare class data frame for training the forecast model, i.e., predict the next win
                # keep namespaces (rows window) and use the next time window (cols test)
                df_class = arqClasses.loc[name_space[window-1]].iloc[:,ini_test:fim_test]
                df_class = pd.DataFrame( index=df_class.index,
                    data=[reduce((lambda x,y: x or y), [int(val) for val in labels]) for namespace, labels in df_class.iterrows()] )
                
                #remove instances with zero volume
                idx = df_train.loc[:,'vol'] > 0.0
                df_train = df_train.loc[idx,:]
                df_class = df_class.loc[idx,:]
                                        
                # train the classifier
                balancer = RandomUnderSampler()
                df_train, df_class = balancer.fit_sample(df_train, df_class)
                df_sweight = (np.log(df_train.vol/min(df_train.vol))/np.log(100))+1                # vol are weights of instances for training
                df_train = df_train.loc[:, ['acc1','acc2','acc3','acc4'] ]  # accs are features for training
                clf = set_clf(opt.CLASSIFIER)
                if opt.SWEIGHT:
                    clf.fit(df_train, df_class.values.ravel(), sample_weight=df_sweight)
                else:
                    clf.fit(df_train, df_class.values.ravel())
            
            
            #prepare test data frame
            # increase namespaces (rows window) and use the next time window (cols test)
            df_test = arqAcessos.loc[name_space[window]].iloc[:, ini_test:fim_test]        

            class_pred = clf.predict(df_test)
            
        else: # online algorithm in Liu et al, 2019 (10.1109/ACCESS.2019.2928844), Erradi & Mansouri, 2020 (10.1016/j.jss.2019.110457)
            
            class_pred = arqClasses.loc[name_space[window]].iloc[:, ini_test:fim_test]
            class_pred = pd.DataFrame( index=class_pred.index,
                data=[reduce((lambda x,y: x or y), [int(val) for val in labels]) for namespace, labels in class_pred.iterrows()])
            

        # prepare evaluation data frame using windows test and eval with cols: 
        #       acc_cur, acc_fut, vol_cur, label, pred
        # NOTE cur means the test windows and fut means the eval window (eval = cur+1)
        df_acc_cur = arqAcessos.loc[name_space[window]].iloc[:, ini_test:fim_test]
        df_acc_cur = pd.DataFrame( index=df_acc_cur.index,
            data=[reduce((lambda x,y: x + y), [int(val) for val in accs]) for namespace, accs in df_acc_cur.iterrows()])

        df_vol_cur = arqVolBytes.loc[name_space[window]].iloc[:, fim_test-1:fim_test]
            # take the last column of the test window as volume dataset is cumulative
        
        df_acc_fut = arqAcessos.loc[name_space[window]].iloc[:, ini_eval:fim_eval]
        df_acc_fut = pd.DataFrame( index=df_acc_fut.index,
            data=[reduce((lambda x,y: x + y), [int(val) for val in accs]) for namespace, accs in df_acc_fut.iterrows()])
        
        df_class_fut = arqClasses.loc[name_space[window]].iloc[:, ini_eval:fim_eval]
        df_class_fut = pd.DataFrame( index=df_class_fut.index,
            data=[reduce((lambda x,y: x or y), [int(val) for val in labels]) for namespace, labels in df_class_fut.iterrows()])
        
        df_class_pred = pd.DataFrame( index=df_class_fut.index,
            data=class_pred[:])
        
        df = pd.concat( [df_acc_cur, df_acc_fut, df_vol_cur, df_class_fut, df_class_pred], axis=1 )
        df.columns = ['acc_cur','acc_fut','vol_bytes','label','pred']
        
        #remove instances with zero volume
        idx = df.loc[:,'vol_bytes'] > 0.0
        df = df.loc[idx,:]
        
        Acerto_Erros(df, window, opt.LOG_ID, CLF_NAMES[opt.CLASSIFIER])
