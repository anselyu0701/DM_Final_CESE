########################################################################
#   Course: data mining
#   Copyright (c)   2023.12.14
#   Group: 5
#   Created by  all of members in group 5
########################################################################
#   you should run this code after intalling packages in requirement.txt
########################################################################
#from sklearn import lda pca decision_tree
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import ExtraTreeClassifier as ETC
from sklearn.utils import resample as boostrap
import pandas as pd
import numpy as np

#input csv data
DATA_PATH = './DM_dataset/dataset/'
DATA_NAME = ['CNAE_9','colon','ISOLATE','LSVT','madelon']#'CNAE_9','colon','ISOLATE','LSVT','madelon'
data_info =  pd.read_csv('./DM_dataset/data_info.csv')
def read_data(data_name):
    csv = pd.read_csv(data_name, header=None) #read csv
    csv = csv.sample(frac = 1) #shuffle
    data=csv.to_numpy()
    #split data
    dp =int(0.75 * data.shape[0])
    X_train = data[0:dp, 0:data.shape[1]-1]
    y_train = data[0:dp, -1]
    X_test = data[dp:data.shape[0], 0:data.shape[1]-1]
    y_test = data[dp:data.shape[0], -1]
    return X_train, y_train, X_test, y_test

def CESE(X_train, y_train, X_test, y_test):
    #parameters
    psi = 6
    C = data_info.loc[data_info['name'] == data_name, 'class'].iloc[0]
    L = X_train.shape[1]
    q = psi * (C - 1)
    z = int(L/q)
    #print('C = ',C,' L = ',L,' q = ',q,' z = ',z)

    #use Random Forest to train q important features
    rfc = RFC(n_estimators = z, bootstrap=True, random_state=None, n_jobs=-1)
    rfc.estimator_ = ETC(random_state=None)
    rfc.fit(X_train,y_train)
    #select q best features based on feature importance in each tree
    SSE_idx = []
    for tree in rfc.estimators_:
        feature_importances_ = tree.feature_importances_
        feature_importances_ = np.argsort(feature_importances_)
        feature_importances_ = feature_importances_[::-1]
        di_features_idx = feature_importances_[0:q]
        SSE_idx.append(di_features_idx)
        #print(di_features_idx)

    #use LDA to find the best vector in q features
    #the goal is to get subspace enhanced features
    SSE_data = []
    for features_idx in SSE_idx:
        X_sub = X_train[:,features_idx]
        lda = LDA(n_components = None,)
        data = lda.fit_transform(X_sub,y_train)
        SSE_data.append(data)
    #print(SSE_data[0])

    #apply PCA to SSE_data
    M=3 #3 features in a group to do PCA analysis
    sse_len=int(len(SSE_data)/M)
    PCA_data_train = None
    PCA_transform=[]
    
    #each groups in 3f eatures finds best variance in lower dimension
    for i in range(sse_len):
        pca = PCA(n_components=None, random_state=0)
        data = np.concatenate(SSE_data[i*M:(i+1)*M], axis=1)
        boostrap(data, n_samples=int(data.shape[1]*0.75), random_state=0, stratify=y_train)
        data = pca.fit_transform(data)

        if PCA_data_train is None: PCA_data_train = data
        else: PCA_data_train = np.concatenate([PCA_data_train, data], axis=1)
        #concatenate all of the result of PCA to be a new dataset(PCA_transform)
        PCA_transform.append(pca)

    #apply Transform to test data
    #original test data also need to do above preprocessing, and input new classifiar 
    SSE_data_test = []
    PCA_data_test = None
    #LDA
    for features_idx in SSE_idx:
        X_sub = X_test[:,features_idx]
        lda = LDA(n_components = None,)
        data = lda.fit_transform(X_sub,y_test)
        SSE_data_test.append(data)
    #PCA
    for i in range(sse_len):
        data = np.concatenate(SSE_data_test[i*M:(i+1)*M], axis=1)
        data = PCA_transform[i].transform(data)
        if PCA_data_test is None: PCA_data_test = data
        else: PCA_data_test = np.concatenate([PCA_data_test, data], axis=1)


    final_classifier = [RFC(n_estimators = z, bootstrap=True, random_state=0, n_jobs=-1),#random forest
                    BC(n_estimators = z, bootstrap=True, random_state=0, n_jobs=-1),#baggging
                    DTC(random_state=0),#decision tree
                    ETC(random_state=0)]#extra tree

    
    origin_classifier = [RFC(n_estimators = z, bootstrap=True, random_state=0, n_jobs=-1),
                        BC(n_estimators = z, bootstrap=True, random_state=0, n_jobs=-1),
                        DTC(random_state=0),
                        ETC(random_state=0)]
    
    CESE_result = []
    #CESE means using PCA_transform dataset and normal classifiars to do classification
    # Because paper does not mention what classifiars in this phase, we implement 4 types of classifiars
    for classifier in final_classifier:
        classifier.fit(PCA_data_train,y_train)
        result = classifier.predict(PCA_data_test)
        #calculate accuracy
        accuracy = 0
        for i in range(len(result)):
            if result[i] == y_test[i]: accuracy += 1
        accuracy = accuracy/len(result)
        CESE_result.append(accuracy)

    raw_result = [] #raw means using original dataset and normal classifiars to do classification 
    for classifier in origin_classifier:
        classifier.fit(X_train,y_train)
        result = classifier.predict(X_test)
        #calculate accuracy
        accuracy = 0
        for i in range(len(result)):
            if result[i] == y_test[i]: accuracy += 1
        accuracy = accuracy/len(result)
        raw_result.append(accuracy)
    return CESE_result, raw_result

    
if __name__ == "__main__":
    testing_num = 20
    result={}
    for data_name in DATA_NAME: #run all of dataset
        data_path = DATA_PATH + data_name + '.csv' #read file
        X_train, y_train, X_test, y_test = read_data(data_path)
        avg_CESE_result = [0,0,0,0]
        avg_raw_result = [0,0,0,0]
        for i in range(testing_num):
            print(data_name, ' ', i,flush=True)
            CESE_result ,raw_result= CESE(X_train, y_train, X_test, y_test)
            avg_CESE_result =  np.add(avg_CESE_result, CESE_result)
            avg_raw_result = np.add(avg_raw_result, raw_result)
        #repeat runing 20 times to get average of accuracy    
        result[data_name] = [avg_CESE_result/testing_num, avg_raw_result/testing_num]

    #save result
    df = pd.DataFrame()
    for name in result: #input accuracy of results
        CESE_result = result[name][0]
        raw_result = result[name][1]
        df[name+'_CESE'] = CESE_result
        df[name+'_raw'] = raw_result
    #write into result.csv
    df.to_csv('./result.csv', index=False)
    print('done')



