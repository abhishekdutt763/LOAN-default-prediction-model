import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import os
import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score,roc_curve,precision_score,recall_score

class metrics:
	def __init__(self,cm):
		self.cm=np.array(cm)
		self.classes=len(cm)
	def precision(self,i):
		true_positive=self.cm[i][i]
		sum_axis_0=np.sum(self.cm, axis=0)
		prec=true_positive/(sum_axis_0[i]+0.0)
		return prec

	def recall(self,i):
		true_positive=self.cm[i][i]
		sum_axis_1=np.sum(self.cm, axis=1)
		recall=true_positive/(sum_axis_1[i]+0.0)
		return recall

	def f_measure(self,i):
		f_mes=(2*self.precision(i)*self.recall(i))/(self.precision(i)+self.recall(i))
		return f_mes

def print_confusion_matrix(cm,score,size):
    plt.figure(figsize=(size,size))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)




def evaluation_metrics(cf,z):
    gbt_metrics_test = metrics(cf)
    print ('<<<<<<<<<<<<'+z+' METRICS >>>>>>>>>>>>>')
    print ('accuracy : ',(cf[0][0]+cf[1][1])/np.sum(cf))
    #print '\n'
    print ('=================class_0==================')
    print ('fMeasure_of_class_0 : ',gbt_metrics_test.f_measure(0))
    print ('precision_of_class_0: ',gbt_metrics_test.precision(0))
    print ('recall_of_class_0:    ',gbt_metrics_test.recall(0))

    #print '\n'
    print ('=================class_1==================')
    print ('fMeasure_of_class_1 : ',gbt_metrics_test.f_measure(1))
    print ('precision_of_class_1: ',gbt_metrics_test.precision(1))
    print ('recall_of_class_1:    ',gbt_metrics_test.recall(1))

    print_confusion_matrix(cf,'cf',3)



def choose_threshold(true_y,pred_prob,k):
    
    sorted_true=pd.DataFrame(list(zip(true_y,pred_prob))).sort_values(by=1,ascending=False).reset_index(drop=True)
    max_f1=0
    recall=0
    precision=0
    prob_array=np.sort(1*np.random.random(k))
    for i in prob_array:
        sorted_true[2]=(sorted_true[1]>=i).astype(int)
        temp_f1=f1_score(sorted_true[0],sorted_true[2])
        if(temp_f1>max_f1):
            max_f1=temp_f1
            precision=precision_score(sorted_true[0],sorted_true[2])
            recall=recall_score(sorted_true[0],sorted_true[2])
            best_threshold=i
    return max_f1,precision,recall,best_threshold

################project specific functions


# EXPLORATION FUNCTIONS

def corr(X,Y):
    return np.corrcoef(X,Y)[0][1]

def perc_missing(X):
    return (len(X)-X.count())*100/len(X)


def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        num=file_info.st_size
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x)
            num /= 1024.0    

            
def class_dist(df,class_col):
    df_count=(df
              .groupby(class_col)
              .agg({class_col:'count'}))

    df_count=df_count.rename({class_col:'count'},axis=1)
    df_count['percent']=df_count['count']*100/df_count['count'].sum()
    df_count = df_count.reset_index()
    return df_count

def feature(df,cols):
    d=df[cols]
    return d

def missing(dataset,missing_col,target_col):
    feature_2=copy.deepcopy(dataset[[missing_col,target_col]])
    feature_2['is_null']=feature_2[missing_col].isna()

    f2_missing=feature_2.groupby('is_null').agg({target_col:'mean'}).reset_index()
    f2_missing=f2_missing.rename({target_col:'%default'},axis=1)
    f2_missing['count']=feature_2.groupby('is_null').agg({target_col:'count'})[target_col]
    f2_missing['no_default']=feature_2.groupby('is_null').agg({target_col:'sum'})[target_col]
    return f2_missing

    
def categorical_distribution(dataset,col,target_col):
    f5_bin=dataset.groupby(col).agg({target_col:'count'})
    f5_bin=f5_bin.rename({target_col:'count'},axis=1).reset_index()
    f5_bin['no_default']=np.array(dataset.groupby(col).agg({target_col:'sum'})[target_col])
    f5_bin['%default']=np.array(dataset.groupby(col).agg({target_col:'mean'})[target_col])
    return f5_bin

def continous_distribution(dataset,col,target_col,bin_nos):
    feature_9=copy.deepcopy(dataset[[col,target_col]])
    name=col+'_binned'
    feature_9[name]= pd.cut(x=feature_9[col],bins=bin_nos)

    clage_bins=feature_9.groupby(name).agg({target_col:'count'})
    clage_bins=clage_bins.rename({target_col:'count'},axis=1).reset_index()
    clage_bins['no_default']=np.array(feature_9.groupby(name).agg({target_col:'sum'})[target_col])
    clage_bins['%_default']=np.array(feature_9.groupby(name).agg({target_col:'mean'})[target_col])
    return clage_bins
    


def new_bins(dataset,col,target_col,arr):
    
    def bins(row):
        for i in arr:
            if row>=i[0] and row<=i[1]:
                return (i)

    feature_7=copy.deepcopy(dataset[[col,target_col]])
    feature_7['new_bins']=feature_7[col].apply(bins)
    
    
    derog_bins_new=feature_7.groupby('new_bins').agg({target_col:'count'})
    derog_bins_new=derog_bins_new.rename({target_col:'count'},axis=1).reset_index()
    derog_bins_new['no_default']=np.array(feature_7.groupby('new_bins').agg({target_col:'sum'})[target_col])
    derog_bins_new['%_default']=np.array(feature_7.groupby('new_bins').agg({target_col:'mean'})[target_col])
    return derog_bins_new

### TRANSFORMERS FUNCTIONS

def col_imputer(fit_dataset,transforms_datasets,col,value):
    obj = SimpleImputer(strategy='constant',fill_value=value)
    obj.fit(np.array(fit_dataset[col]).reshape(-1, 1)) 
    
    for i in transforms_datasets:
        i.loc[:,col]=obj.transform(np.array(i[col]).reshape(-1, 1))
    
def is_null_new_column(fit_dataset,transforms_datasets,col):
    name='is_null_'+col
    le = LabelEncoder()
    le.fit(fit_dataset[col].isna())
    
    for i in transforms_datasets:
        i.loc[:,name] = le.transform(i[col].isna())

    
def label_encoder(fit_dataset,transforms_datasets,col):
    le = LabelEncoder()
    le.fit(fit_dataset[col])
    
    for i in transforms_datasets:
        i.loc[:,col] = le.transform(i[col])

def one_hot_encoder(fit_dataset, transforms_datasets, col):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(fit_dataset[[col]])
    categories=enc.categories_
    for i in transforms_datasets:
        temp=enc.transform(i[[col]]).toarray()
        indx=0
        for c in categories[0]:
            name=col+'_'+c
            i.loc[:,name]=temp[:,indx]
            indx+=1
        

def func_on_dataframe(datasets,cols_apply,result_cols,method):
    for ds in datasets:
        for cl in range(len(cols_apply)):
            ds.loc[:,result_cols[cl]]=ds[cols_apply[cl]].apply(method)


