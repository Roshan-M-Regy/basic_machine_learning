# Roshan Mammen Regy
# roshanm.regy@tamu.edu
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import scipy.linalg 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from random import randint

def Q2_ii(data):
    fig,axes = plt.subplots(2,3,figsize=[5,3],dpi=300)
    x = 0
    y = 0
    for i,col1 in enumerate(data.columns[1:-2]):
        if col1 == 'primary_strength':
            continue
        ax = axes[y][x] 
        x += 1
        if x>2:
            x=0
            y+=1
        ax.scatter(data[col1],data['combat_point'],marker='o',s=2,edgecolor='black',linewidth=0.2,color='red')
        ax.set_xlabel(col1,fontsize=5)
        ax.set_ylabel('combat_point',fontsize=5)
        ax.tick_params(labelsize=4)
        ax.set_title('Pearson r = %.3f'%data[col1].corr(data['combat_point']),fontsize=5)
    plt.tight_layout()
    plt.savefig('Q2_ii.pdf',dpi=300)
    plt.show()

def Q2_iii(data):
    fig,axes = plt.subplots(3,5,dpi=300,figsize=[5,3])
    x = 0
    y = 0
    for i,col1 in enumerate(data.columns[1:-2]):
        for j,col2 in enumerate(data.columns[i+2:-2]):
            if col1 == 'primary_strength' or col2 == 'primary_strength':
                continue
            ax = axes[y][x] 
            x += 1
            if x>4:
                x=0
                y+=1
            ax.scatter(data[col1],data[col2],marker='o',s=2,edgecolor='black',linewidth=0.2,color='red')
            ax.set_xlabel(col1,fontsize=5)
            ax.set_ylabel(col2,fontsize=5)
            ax.tick_params(labelsize=4)
            ax.set_title('Pearson r = %.3f'%data[col1].corr(data[col2]),fontsize=5)
    plt.tight_layout()
    plt.savefig('Q2_iii.pdf',dpi=300)
    plt.show()

def Q2_iv(data):
    types,counts = np.unique(data['primary_strength'],return_counts=True)
    onehotmat = np.zeros((len(data['primary_strength']),len(types)))
    for i,val in enumerate(data['primary_strength']):
        for j,typ in enumerate(types):
            if typ==val:
                onehotmat[i,j] = 1
    
    countfile = open('Q2_iv_type_counts.csv','w') 
    countfile.write('#Name, Counts\n')
    for i,typ in enumerate(types):
        countfile.write('%s, %s\n'%(typ,counts[i]))
        data[typ] = onehotmat[:,i]
    newdata = data.drop(['primary_strength'],axis=1)
    newdata.to_csv('Q2_iv_onehotdata.csv',sep=',')
    countfile.close()
    return data

def make_folds(data,nfold):
    print ("Making %s folds ..."%nfold)
    foldsize = int(data.shape[0]/nfold)
    foldindex = [] # List to save sample indices in each test fold 
    leftindex = [] # List to save sample indices in corresponding train fold
    indexlist = list(range(data.shape[0]))
    for fold in range(nfold):
        findex = []
        if len(indexlist)>=foldsize:
            for j in range(foldsize):
                pos = randint(0,len(indexlist)-1)
                findex.append(indexlist[pos])
                indexlist.pop(pos)
        elif len(indexlist) < foldsize and len(indexlist) > 0.0:
            findex = indexlist
        foldindex.append(findex)
        lindex = []
        for i in range(data.shape[0]):
            if i not in findex:
                lindex.append(i)
        leftindex.append(lindex)
    return (leftindex, foldindex)

def Q2_v(data, colnos,nfold):
    print ('Linear Regression')
    print (data.shape)
    y = data['combat_point']
    X = data.drop(columns=['combat_point','name'])
    X.insert(0,'intercept', np.full(data.shape[0],1))
    # Make folds 
    #leftindex, foldindex = make_folds(data, nfold)
    # Train and test linear regression model for each fold
    print ("Training over folds...")
    RSS = np.zeros(nfold)
    mRSS = np.zeros(nfold)
    for fold in range(nfold):
        '''
        Xtest = np.array(X.iloc[foldindex[fold],colnos])
        Ytest = np.array(y.iloc[foldindex[fold]])
        Xtrain = np.array(X.iloc[leftindex[fold],colnos])
        Ytrain = np.array(y.iloc[leftindex[fold]])
     
        np.savetxt('folds/Xtest_%s.txt'%(fold+1),Xtest)
        np.savetxt('folds/Ytest_%s.txt'%(fold+1),Ytest)
        np.savetxt('folds/Xtrain_%s.txt'%(fold+1),Xtrain)
        np.savetxt('folds/Ytrain_%s.txt'%(fold+1),Ytrain) 
        '''
        Xtest = np.loadtxt('folds/Xtest_%s.txt'%(fold+1))[:,colnos]
        Ytest = np.loadtxt('folds/Ytest_%s.txt'%(fold+1))
        Xtrain = np.loadtxt('folds/Xtrain_%s.txt'%(fold+1))[:,colnos]
        Ytrain = np.loadtxt('folds/Ytrain_%s.txt'%(fold+1)) 
        
        xt = np.transpose(Xtrain)        
        xtx = np.dot(xt,Xtrain)
        try:
            xtxinv = np.linalg.inv(xtx)
        except:
            print ('P inverse for fold %s'%(fold+1))
            xtxinv = np.linalg.pinv(xtx)
        xtxinvxt = np.matmul(xtxinv,xt)
        Wstar = (np.matmul(xtxinvxt,Ytrain))
        #print ('W*',Wstar)
        #np.savetxt('folds/Wopt_fold_%s.txt'%(fold+1),Wstar,fmt='%.4f')
        # Compute RSS of each fold
        RSS[fold] = np.power(np.sum(np.power(Ytest - np.matmul(Xtest,Wstar),2)),0.5)
        mRSS[fold] = np.power(np.mean(np.power(Ytest - np.matmul(Xtest,Wstar),2)),0.5)

    
    #np.savetxt('folds/RSS_linear_regression.txt',RSS, fmt='%.4f')
    print ('Avg RSS over folds',RSS.mean())
    print (RSS)
    print ('Avg mRSS over folds',mRSS.mean())
    print (mRSS)

def Q2_vi(nfold,lam):
    print ("l2-norm Regularized Linear Regression")
    RSS = np.zeros(nfold)
    mRSS = np.zeros(nfold)
    for fold in range(nfold):
        #print ('FOLD %s'%(fold+1))
        Xtest = np.loadtxt('folds/Xtest_%s.txt'%(fold+1))
        Ytest = np.loadtxt('folds/Ytest_%s.txt'%(fold+1))
        Xtrain = np.loadtxt('folds/Xtrain_%s.txt'%(fold+1))
        Ytrain = np.loadtxt('folds/Ytrain_%s.txt'%(fold+1)) 
        
        xt = np.transpose(Xtrain)        
        xtx = np.matmul(xt,Xtrain)
        xtx_lamI = xtx+lam*np.identity(xtx.shape[0],dtype=float)
        try:
            xtx_lamIinv = np.linalg.pinv(xtx_lamI)
        except:
            #print ('P inverse for fold %s'%(fold+1))
            xtxinv_lamIinv = np.linalg.pinv(xtx_lamI)
        xtx_lamIinvxt = np.matmul(xtx_lamIinv,xt)
        Wstar = (np.matmul(xtx_lamIinvxt,Ytrain))
        #print ('W*',Wstar)
        np.savetxt('folds/Wopt_fold_%s_lam_%f.txt'%(fold+1,lam),Wstar,fmt='%.4f')
        RSS[fold] = np.power(np.sum(np.power(Ytest - np.matmul(Xtest,Wstar),2)),0.5)
        mRSS[fold] = np.power(np.mean(np.power(Ytest - np.matmul(Xtest,Wstar),2)),0.5)
    
    print ('Lambda = %s'%lam)
    print ('Avg RSS over folds',RSS.mean())
    print ('Avg mRSS over folds',mRSS.mean())
    np.savetxt('folds/RSS_linear_regression_regularized_l2norm_lam_%f.txt'%lam,RSS, fmt='%.4f')
    return (RSS.mean())


def Q2_vii_b(nfold, lam):
    RSS = np.zeros(nfold)
    clf = linear_model.Lasso(alpha=lam)
    for fold in range(nfold):
        print ('FOLD %s'%(fold+1))
        Xtest = np.loadtxt('folds/Xtest_%s.txt'%(fold+1))
        Ytest = np.loadtxt('folds/Ytest_%s.txt'%(fold+1))
        Xtrain = np.loadtxt('folds/Xtrain_%s.txt'%(fold+1))
        Ytrain = np.loadtxt('folds/Ytrain_%s.txt'%(fold+1))
        clf.fit(Xtrain, Ytrain)
        print ('W*')
        print (clf.coef_)
        Wstar = clf.coef_
        np.savetxt('folds/Wopt_fold_%s_lam_%f_l1norm.txt'%(fold+1,lam),Wstar,fmt='%.4f')
        RSS[fold] = np.power(np.sum(np.power(Ytest - np.matmul(Xtest,Wstar),2)),0.5)
    print (RSS.mean())


def Q2_viii(data):
    # Binarize the 'combat_point' data 
    cmbptmean = np.mean(data['combat_point'])
    for i in range(data.shape[0]):
        if data['combat_point'][i] >= cmbptmean:
            data['combat_point'][i] = 1
        else:
            data['combat_point'][i] = 0
    
    # Make 80/20 split for train/test data
    Y = data['combat_point']
    X = data.drop(columns=['combat_point','name'])
    count = int(np.round(0.20*data.shape[0]))
    indexlist = list(range(data.shape[0]))
    testlist = []
    for i in range(count):
        pos = randint(0,len(indexlist)-1)
        testlist.append(indexlist[pos])
        indexlist.pop(pos)
    trainlist = indexlist

    # Run logistic regression
    trainlist = range(count,data.shape[0])
    testlist = range(0,count)
    Ytrain = np.array(Y.iloc[trainlist])
    Xtrain = np.array(X.iloc[trainlist,:])
    Ytest = np.array(Y.iloc[testlist])
    Xtest = np.array(X.iloc[testlist,:])
    logrg = LogisticRegression(penalty = 'none', max_iter = 5000).fit(Xtrain, Ytrain)
    print (logrg.score(Xtest,Ytest))
    return (data, testlist, trainlist)
        

def Q2_ix(data,trainlist, testlist, nfold, lamlist):
    traindata = data.iloc[trainlist,:]
    testdata = data.iloc[testlist,:]
    Y = traindata['combat_point']
    X = traindata.drop(columns=['combat_point','name'])
    testY = testdata['combat_point']
    testX = testdata.drop(columns=['combat_point','name'])
    leftindex, foldindex = make_folds(traindata, nfold)
    for l,lam in enumerate(lamlist):
        acc = np.zeros(nfold)
        testacc = np.zeros(nfold)
        for fold in range(nfold):
            Xtest = np.array(X.iloc[foldindex[fold],:])
            Ytest = np.array(Y.iloc[foldindex[fold]])
            Xtrain = np.array(X.iloc[leftindex[fold],:])
            Ytrain = np.array(Y.iloc[leftindex[fold]])
            logrg_reg = LogisticRegression(penalty='l2', max_iter = 5000, C = 1/lam).fit(Xtrain, Ytrain)
            acc[fold] = logrg_reg.score(Xtest,Ytest)
            testacc[fold] = logrg_reg.score(np.array(testX), np.array(testY))

        print ('Lambda %s, Acc. %s, test Acc. %s'%(lam,np.mean(acc),np.mean(testacc)))
        
    
nfold = 5
data = pd.read_csv('Q2_iv_onehotdata.csv')
'''
#Q2_v(nfold)
lamlist = [0.00001, 0.001, 0.1, 1.0, 10, 100 ,1000, 1000000]
#lamlist = [0.01, 0.1, 0.5, 1.0, 2, 10]
lamlist = np.linspace(0.000001,1000,20)
Acc = np.zeros(len(lamlist))
#lamlist = [1.0]
#for i,lam in enumerate(lamlist):
#    Q2_vii_b(nfold,lam)
Q2_v(data,nfold)

for i,lam in enumerate(lamlist):
    Acc[i] = Q2_vi(nfold,lam)
fig,ax = plt.subplots(1,1,dpi=300,figsize=[3,3])
ax.plot(lamlist, Acc,'--o',markersize=2,color='red',lw=0.5)
ax.set_xlabel(r'$\lambda$', fontsize=6)
ax.set_ylabel('sq. rt. RSS', fontsize=6)
plt.tight_layout()
plt.savefig('lambda_vs_RSS.pdf',dpi=300)
selections = [['defense_value'],['attack_value'],['stamina'],['capture_rate'],['defense_value','attack_value','capture_rate'],['defense_value','attack_value','stamina']]
columns = data.columns
for i,select in enumerate(selections):
    print (select+['combat_point','name'])
    colnos = []
    for j in select:
        for k,col in enumerate(columns):
            if j==col:
                colnos.append(k+1)
    newdata = data[select+['combat_point','name']]
    
    Q2_v(newdata,colnos,nfold)
lamlist = [0.01,1.0,10]
for lam in lamlist:
    Q2_vii_b(nfold,lam)
'''
lamlist = [0.00001, 0.001, 0.1, 1.0, 10, 100 ,1000, 1000000]
bindata, testlist, trainlist = Q2_viii(data)
Q2_ix(bindata, trainlist, testlist, nfold, lamlist)
