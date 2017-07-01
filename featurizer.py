from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# Numeric featurization

def log_transform(x, epsilon = 1e-4):
    
    """
    Simple log transform of x. 
    
    If min(x) < 0 them x -> x + min(x)
    
    Parameters
    ----------
    
    x : real valued numpy array or pandas series
    
    epsilon : very small value to add to zero, default = 1e-4
    
    
    Returns
    --------
    
    log(x + epsilon)
    
    """
    if x.min() < 0: epsilon += np.abs(x.min())
    return (x.fillna(0).astype(float) + epsilon).apply(np.log)


# Categorical featurization

class count_featurizer:
    
    """ 
    Counts number of occurrences within a given category for 
    some categorical variable x.
    """
    
    def __init__(self):
        self.d = {}
    
    def fit(self, x):
        
        """ x is categorical variable to count. 
        This method should be used on training set only """
        
        self.d = x.value_counts().to_dict()
        
    def transform(self, x):
        
        """Returns value counts of categorical variable."""
        
        return x.apply(lambda x_i: self.d[x_i] if x_i in self.d else 0)
    
    

class badness_featurizer: 
    
    """
    Returns estimate of p(y = 1 | x = g) where 
    
    y : boolean label,
    x : some categorical variable
    g : some category of x
    
    """
    
    def __init__(self):
        self.d = {}
    
    def fit(self, x, y):
        """ 
        Use method on training set only. 
        
        Parameters
        ----------
        
        x : categorical variable
        
        y : Boolean label y == 1 or y == 0 (no multiclass) 
        
        """
        tempdf = pd.DataFrame({'x':x, 'y':y})
        self.d = tempdf.groupby('x').apply(lambda g: g.y.sum()/len(g)).to_dict()
        
    def transform(self, x):
        """ Returns estimate of p(y == 1 | x == g)."""
        
        return x.apply(lambda x_i: self.d[x_i] if x_i in self.d else 0)
    
    
def first_featurizer(group_key, order_key, condition = None):
    
    """ 
    Returns boolean array with values corresponding to whether 
    a example corresponds to the first element of group_key 
    ordered by order_key
    
    Parameters
    ----------
    
    group_key : key to group examples by, i.e. categorical feature
    
    order_key : key that determines which example is first
    
    condition : boolean array to condition df on
    
    """
    
    tempdf = pd.DataFrame({'gkey':group_key, 'okey':order_key})
    
    if condition is None:
        d = tempdf.groupby('gkey').okey.min().to_dict()
    else:
        d = tempdf[condition].groupby('gkey').okey.min().to_dict()
        
    return tempdf.apply(lambda x: x.okey == d[x.gkey] if x.gkey in d else False, 1)


class MultiColLabelEncoder():
    
    """
        Like sklearn's LabelEncoder but capabile of doing multiple columns.
        
        Usage:
        ------
        i
        encoder = MultiColLabelEncoder()
        df_train[cat_var_list] = df_train[cat_var_list].apply(encoder.fit_transform)
        df_pred[cat_var_list] = df_pred[cat_var_list].apply(encoder.transform)
    """
    
    def __init__(self):
        self.les = {}
        self.labels = []
        
    def fit_transform(self, x):
        le = LabelEncoder()
        res = le.fit_transform(x)
        self.les[x.name] = le
        for label in le.classes_:
            label_str = str(x.name)+' '+str(label).replace(' ','_')
            if label_str not in self.labels:
                self.labels.append(label_str)
        return res
    
    def transform(self, x):
        return self.les[x.name].transform(x)
    
    def get_indices(self, labels):
        return [self.labels.index(l) for l in labels]
            

class categorical_feature_selector():
    """
        One hot encoding with feature selection. 
        
        Performs one-of-k hot encoding on a dataframe and removes encodings with 
        low support.
        
        Usage:
        ------
        
        selector = categorical_feature_selector()
        X_train = selector.fit_transform(df_train[cat_var_list])
        X_pred = selector.fit_transform(df_pred[cat_var_list])
    
    """
    
    def __init__(self, supp_thresh = 0.05, skip_list = []):
        self.le = MultiColLabelEncoder()
        self.he = OneHotEncoder()
        self.selected_features = []
        self.skip_list = skip_list
        self.supp_thresh = supp_thresh
        
    def fit_transform(self, X):
        
        self.index = X.index.tolist()
        self.N = float(len(X))
        
        Z = X.apply(self.le.fit_transform)
        
        #Turn each possible value of the categorical features into it's own feature
        #This is called One-hot-encoding. 
        Z = self.he.fit_transform(Z)
        
        for i, name in enumerate(self.le.labels):
            z_i = Z[:,i].toarray().flatten()
            supp = sum(z_i)/self.N
            if supp < self.supp_thresh or name in self.skip_list:
                continue
            self.selected_features.append(name)
        
        #Make list of statistically significant features
        self.selected_feature_indicies = self.le.get_indices(self.selected_features)
        Z_selected = Z[:,self.selected_feature_indicies]
        
        return pd.DataFrame(Z_selected.todense(),
                            columns = self.selected_features,
                            index = self.index).astype(int)
    
    def transform(self, X):
        
        Z = X.apply(self.le.transform)
        
        #Turn each possible value of the categorical features into it's own feature
        #This is called One-hot-encoding. 
        Z = self.he.transform(Z)
        Z_selected = Z[:,self.selected_feature_indicies]
        
        return pd.DataFrame(Z_selected.todense(),
                            columns = self.selected_features,
                            index = self.index)
    
    
    
class rho():
    """
        Transforms categorical feature columns into real ones. 
        
        This transformation (which has no name) 
        calculates the correlation of each categorical label agianst the target y along with
        the pvalues for the training set. Category labels are then replaced with the correlation 
        times 1 - pvalues (to down weight spurious correlations). This transformation is useful 
        when there's a large number of labels and one-of-k hot encoding is not an option. 
        
        Usage: 
        -------
        
        r = rho()
        df_train[cat_var_list] = df_train[cat_var_list].apply(lambda x: r.fit_transform(x,y))
        df_pred[cat_var_list] = df_pred[cat_var_list].apply(r.transform) 
    """
    
    def __init__(self):
        self.rho = {}
    
    def fit_transform(self, x, y): 
        z = LabelEncoder().fit_transform(x).reshape(-1,1)
        Z = OneHotEncoder().fit_transform(z)
        temp_dict = {}
        for i, label in enumerate(self.le.classes_):
            z_i = Z[:,i].toarray().flatten()
            r, p = pearsonr(z_i, y)
            temp_dict[str(label)] = r*(1-p)
        self.rho[x.name] = temp_dict
        return [temp_dict[str(x_i)] for x_i in x]
    
    def transform(self, x):
        return [self.rho[x.name][str(x_i)] for x_i in x]