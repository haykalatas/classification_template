# -*- coding: utf-8 -*-

# Meta informations.
__version__ = '1.3.2'
__author__ = 'Agil Haykal'
__author_email__ = 'agil@datalabs.id'

import re
import traceback
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import scipy.stats.stats as stats
import pandas.core.algorithms as algos
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

################################################################################
#1.1 Load Dataset
################################################################################

def dataset (file_dir, sampling=None, sample_size=None, random_state=None):
    """
    Importing and Sampling Dataset.
    
    Parameters
    ----------
    
    file_dir:     string
        Name of directory of dataset.
        Make sure to fully write the filetype and directory name.

    sampling:     Boolean, default False
        Define whether to sample the data or not.
                
    sample_size:  float, default 0.2
        Proportion of sample size (from 0 to 1).

    random_state: int, default 0
        Seed for the random number generator.

    Returns
    -------
    
    DataFrame:
        The data which will be used.
    
    Printed Information
    -------
    
    (Printed informations will not be assigned inside the variable)
    
    Data Shape:
        Information of row and column number.
        
    Data Column Names and types:
        Information of each column names and types.
    
    """
    # Parameter
    if sampling is None:
        sampling = False

    if sample_size is None:
        sample_size = 0.2
        
    if random_state is None:
        random_state = 0
    
    # Define Dataset
    dataframe = pd.read_csv(file_dir)
    
    #Define Information
    def information (dataframe):
        print('Rows:', dataframe.shape[0],', Columns:', dataframe.shape[1], '\n')
        print('Column Names and Types:\n', dataframe.dtypes)

    # Sampling Dataset
    if sampling == True:
        dataframe = dataframe.sample(frac=sample_size, random_state=random_state)
        information(dataframe)
        return dataframe
    else:
        information(dataframe)
        return dataframe

################################################################################
#1.2 Get Categorical Columns
################################################################################

def get_categorical(dataframe, exception, threshold=None):
    """
    A function that get categorical column names.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which are going to be checked.

    exception:    array, string
        Name of columns which are not included in the process.

    threshold:    int, default 12
        Number of unique values in each column.

    Returns
    -------
    
    Array:
        The function return array of categorical columns' names.
    
    """
    #Parameter
    if threshold is None:
        threshold = 12

    col = [c for c in dataframe.columns if c not in exception]
    numclasses=[]
    for c in col:
        numclasses.append(dataframe[c].nunique())

    limit = threshold
    categorical_variables = list(np.array(col)[np.array(numclasses) < limit])
    return categorical_variables

################################################################################
#1.2 Get Numerical Columns
################################################################################

def get_numerical(dataframe, exception, threshold=None):
    """
    A function that get numerical column names.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which are going to be checked.

    exception:    array, string
        Name of columns which are not included in the process.

    threshold:    int, default 12
        Number of unique values in each column.

    Returns
    -------
    
    Array:
        The function return array of numerical columns' names.
    
    """
    #Parameter
    if threshold is None:
        threshold = 12

    col = [c for c in dataframe.columns if c not in exception]
    numclasses=[]
    for c in col:
        numclasses.append(dataframe[c].nunique())

    limit = threshold
    numerical_variables = list(np.array(col)[np.array(numclasses) > limit])
    return numerical_variables

################################################################################
#2. Checking Values
################################################################################

def check_values (dataframe, target=None):
    """
    Checking missing values and duplicated rows.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which are going to be checked.

    target:       object, default Do Not Drop Target Column
        Drop or not the target column is depend on how sure that column has
        missing values or not.

    Returns
    -------
    
    Table:
        Table consist of column name, total missing values,
        and missing percentage.
    
    Printed Information
    -------
    
    (Printed informations will not be assigned inside the variable)
    
    Duplicated Rows:
        Information of how many duplicated rows.
        
    Missing Values Columns:
        Information of total columns which has empty values.
    
    """
    # Define Function of Missing Value Table
    def missing_table (missing):
        missing['missing_percent'] = missing[missing.columns[1]].apply(lambda x: round(x/dataframe.shape[0] * 100, 2))
        missing = missing[missing.missing_percent !=0].sort_values('missing_percent', ascending=False)
        print('DUPLICATED ROWS')
        print('Number of duplicated rows:', dataframe.duplicated().sum(), '\n\n')
        print('MISSING VALUES')
        print('Number of cols containing missing values:', missing.shape[0])
        return missing
        
    # Missing Values Table
    if target == None:
        missing = dataframe.isnull().sum().to_frame().reset_index()
        table = missing_table(missing)
        return table
    elif target != None:
        missing = dataframe.drop(target,1).isnull().sum().to_frame().reset_index()
        table = missing_table(missing)
        return table
    
################################################################################
#3.1 Feature Engineering - Handling Outliers
################################################################################

def outlier_handling (dataframe, num_cols, upper=None, lower=None):
    """
    Change the value of outliers to the upper/lower bound.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which its outliers values are going to be changed.

    num_cols:     array, string
        Name of the columns that are going to be changed.
        
    upper:        Boolean, default True
        Decide whether going to drag upper outliers to upper bound or not.
        
    lower:        Boolean, default True
        Decide whether going to drag lower outliers to lower bound or not.

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    # Parameter
    if upper is None:
        upper = True
        
    if lower is None:
        lower = True
        
    updated_df = dataframe

    for column in num_cols:
        updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce')
    
    # Define The Calculation of Bounds
    def boundary_values (dataframe, column):
        q1 = dataframe[column].quantile(q=0.25)
        q3 = dataframe[column].quantile(q=0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return lower_bound, upper_bound
    
    # Drag The Upper and Lower Outlier to The Upper Bound
    if (upper == True) & (lower == True):
        for column in num_cols:
            lower_bound, upper_bound = boundary_values(updated_df, column)
            updated_df.loc[updated_df[column] > upper_bound, column] = upper_bound
            updated_df.loc[updated_df[column] < lower_bound, column] = lower_bound
    
    # Drag The Upper Outlier to The Upper Bound
    elif (upper == True) & (lower == False):
        for column in num_cols:
            lower_bound, upper_bound = boundary_values(updated_df, column)
            updated_df.loc[updated_df[column] > upper_bound, column] = upper_bound

    # Drag The Lower Outlier to The Upper Bound
    elif (upper == False) & (lower == True):
        for column in num_cols:
            lower_bound, upper_bound = boundary_values(updated_df, column)
            updated_df.loc[updated_df[column] < lower_bound, column] = lower_bound
        
    # Nothing is changed
    elif (upper == False) & (lower == False):
        print('No outlier is changed here')
    
    # False input
    else:
        print('Please input a proper boolean type (True or False). No outlier is changed here')
    
    return updated_df

################################################################################
#3.2 Feature Engineering - Handling Missing and Duplication
################################################################################

def dupli_missing (dataframe, columns, duplicated=None, missing=None, fill_method=None):
    """
    Change the value of outliers to the upper/lower bound.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which its duplicated rows and missing values
        are going to be changed.

    columns:      array, string
        Name of the columns that are going to be changed.
        REMEMBER TO SEPARATE CATEGORICAL AND NUMERICAL COLUMNS.
        
    duplicated:   Boolean, default True
        Define whether going to handle duplicated or not.

    missing:      Boolean, default True
        Define whether going to handle missing value or not.
        
    fill_method:  string, default mean
        Method of changing the data inside the column (mean, median, mode).

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    # Parameter
    if fill_method is None:
        fill_method = 'mean'

    if duplicated is None:
        fill_method = True

    if missing is None:
        missing = True
    
    # Define Remove Duplication Function
    def remove_duplication(dataframe):
        dataframe.drop_duplicates(keep='first', inplace=True)
        return dataframe
    
    # Filling missing value method
    def methods(dataframe, column, fill_method):
        if fill_method == 'mean':
            method = dataframe[column].mean()
            return method
        elif fill_method == 'median':
            method = dataframe[column].median()
            return method
        elif fill_method == 'mode':
            method = dataframe[column].mode()[0]
            return method
        else:
            print('Please type a proper method (mean, median, mode). No NaN is changed here.')
    
    # Define Handling missing values Function
    def missing_values(dataframe, columns, fill_method):
        for column in columns:
            dataframe.loc[dataframe[column].isnull(), column] = methods(dataframe, column, fill_method)
        return dataframe
    
    # Apply the functions
    if (duplicated == True) & (missing == True):
        dataframe = remove_duplication(dataframe)
        dataframe = missing_values(dataframe, columns, fill_method)
        return dataframe

    elif (duplicated == True) & (missing == False):
        dataframe = remove_duplication(dataframe)
        return dataframe

    elif (duplicated == False) & (missing == True):
        dataframe = missing_values(dataframe, columns, fill_method)
        return dataframe

    else:
        return dataframe

################################################################################
#3.3 Feature Engineering - Information Value and Weight of Evidence
################################################################################

def information_value(dataframe, feat_cols, target_col, weak_predictor=None):
    """
    - Information Value (IV) is Weighted SUM of WOE (Weighted of Evidence)
    - WOE tells predictive power of an independent variable in relation to the dependent variable.

    RULE OF THUMB
          
    < 0.02	    (Useless for prediction)   
    0.02 to 0.1	(Weak predictor)  
    0.1 to 0.3	(Medium predictor)  
    0.3 to 0.5	(Strong predictor)  
    >0.5	    (Suspicious or too good to be true)
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which going to be processed by information value.

    feat_cols:    array, string
        Names of all columns which going to be processed by information value.
        
    target_col:   string
        Name of target column (PLEASE DON'T ADD BRACKET [] TO PREVENT ERROR)

    weak_predictor: Boolean, default False
        False means does not include the weak predictor
        True  means include the weak predictor

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe by WOE and IV.

    """
    # Parameter
    if weak_predictor is None:
        weak_predictor = False

    max_bin = 20
    force_bin = 3
    
    def mono_bin(Y, X, n = max_bin):
        
        df1 = pd.DataFrame({"X": X, "Y": Y})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]
        r = 0
        while np.abs(r) < 1:
            try:
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
                d2 = d1.groupby('Bucket', as_index=True)
                r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                n = n - 1 
            except Exception as e:
                n = n - 1
    
        if len(d2) == 1:
            n = force_bin         
            bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
            if len(np.unique(bins)) == 2:
                bins = np.insert(bins, 0, 1)
                bins[1] = bins[1]-(bins[1]/2)
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
            d2 = d1.groupby('Bucket', as_index=True)
        
        d3 = pd.DataFrame({},index=[])
        d3["MIN_VALUE"] = d2.min().X
        d3["MAX_VALUE"] = d2.max().X
        d3["COUNT"] = d2.count().Y
        d3["EVENT"] = d2.sum().Y
        d3["NONEVENT"] = d2.count().Y - d2.sum().Y
        d3=d3.reset_index(drop=True)
        
        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4,ignore_index=True)
        
        d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["VAR_NAME"] = "VAR"
        d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()
        
        return(d3)
    
    def char_bin(Y, X):
            
        df1 = pd.DataFrame({"X": X, "Y": Y})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]    
        df2 = notmiss.groupby('X',as_index=True)
        
        d3 = pd.DataFrame({},index=[])
        d3["COUNT"] = df2.count().Y
        d3["MIN_VALUE"] = df2.sum().Y.index
        d3["MAX_VALUE"] = d3["MIN_VALUE"]
        d3["EVENT"] = df2.sum().Y
        d3["NONEVENT"] = df2.count().Y - df2.sum().Y
        
        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4,ignore_index=True)
        
        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3.loc[d3["DIST_EVENT"] == 0, ["DIST_EVENT", "DIST_NON_EVENT"]] = 1
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT) * d3.WOE
        
        d3["VAR_NAME"] = "VAR"
        d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()
        d3 = d3.reset_index(drop=True)
        
        return(d3)
    
    def data_vars(df1, target):
        
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
        final = (re.findall(r"[\w']+", vars_name))[-1]
        
        x = df1.dtypes.index
        count = -1
        
        for i in x:
            if i.upper() not in (final.upper()):
                if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                    conv = mono_bin(target, df1[i])
                    conv["VAR_NAME"] = i
                    count = count + 1
                else:
                    conv = char_bin(target, df1[i])
                    conv["VAR_NAME"] = i            
                    count = count + 1
                    
                if count == 0:
                    iv_df = conv
                else:
                    iv_df = iv_df.append(conv,ignore_index=True)
        
        iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
        iv = iv.reset_index()
        return(iv_df,iv)
    
    # Apply the functions
    final_iv, IV = data_vars(dataframe[feat_cols], dataframe[target_col])
    
    IV_data = IV.sort_values('IV', ascending=False)
    
    if weak_predictor == True:
        # Variables with IV > 0.02
        IV_data_filtered = IV_data[IV_data.IV > 0.02] 
    
    else:
        # Variables with IV > 0.1
        IV_data_filtered = IV_data[IV_data.IV > 0.1]

    # List of variables with Selected IV
    final_col_names = IV_data_filtered['VAR_NAME'].tolist()
    
    return dataframe[final_col_names]

################################################################################
#3.3 Feature Engineering - Labeling Categories
################################################################################
    
def label (dataframe, cat_cols):
    """
    Labeling the Categorical strings to integers (Randomly).
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe that has categorical columns.

    cat_cols:     array, string
        Names of all Categorical Columns.

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    le = LabelEncoder()

    for column in cat_cols:
        dataframe[column] = le.fit_transform(dataframe[column].astype(str))
    return dataframe

################################################################################
#3.4 Feature Engineering - Normalization 
################################################################################

def normalization (dataframe, num_cols, standardize=None):
    """
    Normalizing the Numerical values with StandardScaler or MinMaxScaler.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe that has Numerical columns.

    num_cols:     array, int
        Names of all Numerical Columns.
        
    standardize:  Boolean, default True
        Choose method of Normalization.
        StandardScaler = True
        MinMaxScaler   = False

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    # Parameter
    if standardize is None:
        standardize = True

    values  = dataframe.__getitem__(num_cols)
    max_updater = MinMaxScaler().fit_transform(values)
    std_updater = StandardScaler().fit_transform(values)
    

    if standardize == False:
        dataframe.loc.__setitem__((slice(None), (num_cols)), max_updater)
        return dataframe

    else:
        dataframe.loc.__setitem__((slice(None), (num_cols)), std_updater)
        return dataframe

################################################################################
#3.5 Feature Engineering - Derivative Columns
################################################################################

def derivative (dataframe, columns, multiply=None, divide=None):
    """
    Create new column based on multiplication or division.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe that has Numerical columns.

    columns:      array, int
        Names of all Numerical Columns.
        
    multiply:     Boolean, default True
        Create new column based on multiplication.

    divide:       Boolean, default True
        Create new column based on division.

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    # Parameter
    if multiply is None:
        multiply = True
        
    if divide is None:
        divide = True
        
    # Defining multiplication between 2 columns
    def multiplication (dataframe, columns):
        times   = ' X '
        for column1 in columns:
            for column2 in columns:
                dataframe[column1+times+column2] = dataframe[column1] * dataframe[column2]
        return dataframe
    
    # Defining division between 2 columns
    def division (dataframe, columns):
        divided = ' / '
        for column1 in columns:
            for column2 in columns:
                dataframe[column1+divided+column2] = dataframe[column1] / dataframe[column2]
        return dataframe
    
    # Replace the infinite number into 0
    def neutralize (dataframe):
        columns = list(dataframe.columns)
        for column in columns:
            dataframe.loc[dataframe[column] == float('inf'), column] = 0
        return dataframe
    
    updated = dataframe
    
    # Activating multiplication and division
    if (multiply == True) & (divide == True):
        updated = multiplication(updated, columns)
        updated = division(updated, columns)
        updated = neutralize(updated)
        return updated
    
    # Activating multiplication
    elif (multiply == True) & (divide == False):
        updated = multiplication(updated, columns)
        updated = neutralize(updated)
        return updated
    
    # Activating division
    elif (multiply == False) & (divide == True):
        updated = division(updated, columns)
        updated = neutralize(updated)
        return updated
    
    # Activating nothing
    else:
        print('Nothing is Changed Here!')
        return updated
    
################################################################################
#3.6 Feature Engineering - Drop Missing
################################################################################ 

def drop(dataframe, columns=None, drop_threshold=None):
    """
     Drop the columns that have missing value above the threshold.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which its duplicated rows and missing values
        are going to be changed.

    columns:      array, string
        Name of the columns that are going to be changed.
        REMEMBER TO SEPARATE CATEGORICAL AND NUMERICAL COLUMNS.
        
    drop_threshold: float, default 0.5
        Percentage of missing values per total rows (from 0 to 1).
        Example: 0.4 -> means there are 40% of missing data in a column.

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    # Parameter
    if drop_threshold is None:
        drop_threshold = 0.5
        
    if columns is None:
        columns = list(dataframe.columns)
        
    # Total of Rows
    max_row = dataframe.shape[0]

    dropped_column = []

    # Decide Whether drop column or not
    for column in columns:
        missing = dataframe[column].isnull().sum()
        percent = missing / max_row
        if percent >= drop_threshold:
            dropped_column.append(column)
    return dataframe.drop(dropped_column, 1)

################################################################################
#4. Train Test Sample
################################################################################
        
def train_test_preparation (dataframe, target, test_size=None, imbalanced=None, random_state=None):
    """
    Splitting The Dataframe to Train-Test sets and Handling Imbalanced Data.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe of features.

    target:       array
        Array of target.
        
    test_size:    float, default 0.2
        Size of test set.
        
    imbalanced:   Boolean, default False
        Handling the imbalanced data and transform it with SMOTE method.
        
    random_state: float, default 0
        Random Seed of Splitting and Balancing.

    Returns
    -------
    
    X_train:
        Dataframe of Train's Feature.
        
    X_test:
        Dataframe of Test's Feature.

    y_train:
        Array of Train's target.
        
    y_test:
        Array of Test's target.

    """
    # Parameter
    if test_size is None:
        test_size = 0.2
        
    if random_state is None:
        random_state = 0
        
    if imbalanced is None:
        imbalanced = False
 
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target, test_size=test_size, random_state=random_state)
    X_test = X_test.iloc[:, :].values
    X_test = pd.DataFrame(X_test)
    
    # Imbalanced Target
    if imbalanced == True:
        # Handling imbalance data using SMOTE
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_sample(X_train, y_train)
        X_res = pd.DataFrame(X_res)
        return X_res, X_test, y_res, y_test
    else:
        return X_train, X_test, y_train, y_test

################################################################################
#5.1. Evaluation - Confusion Matrix, Log Loss, AUC, and ROC
################################################################################

def evaluation(y_true, y_prediction, X_test, classifier):
    """
    Evaluation of Machine Learning Classification model. The Evaluation consists of:
    Confusion Matrix, Accuracy, Recall, Specificity, Precision, Log Loss, AUC, and ROC.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which are going to be checked.

    target:       object, default Do Not Drop Target Column
        Drop or not the target column is depend on how sure that column has
        missing values or not.

    X_test:       object, DataFrame
        Features of test set.

    classifier:   object
        Result of fitting model with features and target

    Printed Information
    -------
    
    (Printed informations will not be assigned inside the variable)
    
    a. Confusion Matrix
    b. AUC
    c. Accuracy
    d. Recall
    e. Specificity
    f. Precision
    g. Log Loss
    h. ROC Curve
    
    """
    # Define confusion matrix
    cm = confusion_matrix(y_true, y_prediction)
    
    # Calculate the fpr and tpr for all thresholds of the classification
    probs = classifier.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    # True Positive
    TP = cm[1][1]
    # True Negative
    TN = cm[0][0]
    # False Positive
    FP = cm[1][0]
    # False Negative
    FN = cm[0][1]
    
    # Overall, how often is the classifier correct?
    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    # When it's actually yes, how often does it predict yes?
    Recall      = (TP)/(TP + FP)
    # When it's actually no, how often does it predict no?
    Specificity = (TN)/(TN + FN)
    # When it predicts yes, how often is it correct?
    Precision   = (TP)/(TP + FN)
    
    # Print Function
    print(cm)
    print('\nArea Under Curve')
    print('AUC         : %.2f%%' % (roc_auc*100))
    print('\nConfusion Matrix Evaluation')
    print('Accuracy    : %.2f%%' % (Accuracy*100))
    print('Recall      : %.2f%%' % (Recall*100))
    print('Specificity : %.2f%%' % (Specificity*100))
    print('Precision   : %.2f%%' % (Precision*100))
    print('Log Loss    :', round(log_loss(y_true, y_prediction), 3), '\n')

    plt.figure(figsize=(7,6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()