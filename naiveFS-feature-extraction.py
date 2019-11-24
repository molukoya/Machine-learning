###  Feature selection using NaiveFS
D = np.array([[-2,-1,0,-1],[-2,-1,1.4,-1],[2,1,-1,1],[-0.2,-0.1,-1,1]])

def select_feature(D):
    DT = D.transpose()
    sorted_correl = np.argsort(abs(np.corrcoef(DT)[:,3]))[::-1]
    label = []
    for i in range((np.size(D,1))-1):
        label.append('Feature %s'% i)
    label.append('Class')
    return [label[sorted_correl[1]],label[sorted_correl[2]]];
