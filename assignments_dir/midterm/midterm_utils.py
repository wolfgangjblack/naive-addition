from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score as accuracy, recall_score as recall, precision_score as precision, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_weight_of_evidence_mapping(df, cols):
    """
    Weight of Evidence (WOE) and Information Value (IV) calculator for categorical variables.
    Given a feature column of categorical type and a column of binary targets, this calculator determines the WOE
    and IV for the feature. It returns a dictionary to map the feature to WOE, a dictionary to map the feature to IV,
    and a summary table of the calculations used.
    
    *Note: To calculate WOE and IV values for NULL/None entries, the dataframe should have all such values replaced with 
    np.nan values via code such as df = df.fillna(value=np.nan)
    
    The WOE is given by ln(distribution of non-events / distribution of events) and can be interpreted as follows:
    -If the category has the same distribution of events and non-events, then the outcome is proportionally distributed 
      across the category, so it is not very informative. The WOE will be 0.
    -If the distribution of events is proportionally low in the category, the ratio in the natural log will be greater
      than 1 so the WOE will be positive.
    -If the distribution of events is proportionally high in the category, the ratio in the natural log will be less
      than 1 so the WOE will be negative.
    
    The IV is given by Sum over all Categories of (distribution of events - distribution of non-events) * WOE, so it will always be >= 0.
    From sources online, interpretation of IV values breaks down as follows:
    
    Information Value (IV) | Variable Predictive Power
    ----------------------------------------------------
            < 0.02         | Generally not useful for prediction - still a candidate for inclusion in the model if uncorrelated to other features in the model
         0.02 to 0.1       | Weaker predictive power
          0.1 to 0.3       | Medium predictive power
          0.3 to 0.5       | Stronger predictive power
            > 0.5          | Incredibly strong predictive power - look to distribute power by looking at other viewpoints of the same field (if max deposits, then look at trend or recency in deposits)
    
    Parameters
    ----------
    df : pandas dataframe
        The dataframe which contains the categorical feature column and the target column. Data within the 
            feature column can be strings or numbers, as long as the count of distinct values is small.
        
    cols = [feature, target]: list of strings
        feature: string 
            String which represents the exact name of the categorical feature column which needs a weight of evidence calculation 
        Sample notation cat_col = 'VistorType' 
        
        target: string
            String which represents the exact name of the target column which will determine the number of events and non-events as per the categorical column
        Sample notation target_col = 'target_flag'
        
    Returns
    -------
    output_dict: dictionary
        This is a dictionary to be used in mapping the feature categories to the weight of evidence.
    
    information_value = summary_df['contribution'].sum()
       #Sum all contributions to get Information Value of the feature. This can be paired with the table above to describe whether the feature is predictive
        """
    #Sort the columns out
    feature = cols[0]
    target_col = cols[-1]
    
    #reduce the df down to the feature/target dv
    df = df[[feature, target_col]]
    
    #initialize a calc df and populate it with category
    summary_df = pd.DataFrame()
    summary_df[feature] = sorted(df[feature].unique())
    
    #sum up the binary target variable per category as 'events'
    summary_df['events'] =  df.groupby(feature)[target_col].sum().to_numpy()
    event_total = summary_df['events'].sum()
    
    #get counts of each category, and use this calculate non-events (counts-events). if 0, this is highly correlated with target
    summary_df['counts'] = df.groupby(feature)[target_col].count().to_numpy()
    summary_df['non_events'] = summary_df['counts'] - summary_df['events']
    non_event_total = summary_df['non_events'].sum()

    #Get percents per event/non-event
    summary_df['pct_event'] = summary_df['events']/event_total*100
    summary_df['pct_non_event'] = summary_df['non_events']/non_event_total*100
    
    #Get ratio of events over non-events
    summary_df['info_odds'] = summary_df['pct_event']/summary_df['pct_non_event']
    
    #Get weight of evidence
    summary_df['weight_of_evidence'] = np.log(summary_df['info_odds'])
    summary_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    #Find the contribution and print total contribution, this is the value >= 0.5
    summary_df['contribution'] = (summary_df['pct_event']-summary_df['pct_non_event'])*summary_df['weight_of_evidence']/100

    print(f"the information value for {feature} is {summary_df['contribution'].sum()}")
    
    return dict(zip(summary_df[feature],summary_df['weight_of_evidence'])), summary_df['contribution'].sum()

def train_log_reg(X_train, y_train, X_test, y_test):
    """trains a logistic regression model after feature engineering/selection has been performed. 
    Note: This will scale data in the pipeline
    
    Inputs: 
        X_train: a pandas DataFrame
            - with numeric or binary features pre-engineered and selected
        y_train: a pandas Series
            - target corresponding to the training data
        X_test: a pandas DataFrame
            - with numeric or binary features pre-engineered and selected
        y_test: a pandas Series
     Returns:
         model: a sklearn model
             - Model that has been fit on scaled data X_train data
        
        Printed Classification report
    """
    
    lr_model = LogisticRegression(class_weight = 'balanced')
    scaler = StandardScaler()
    pipeline = Pipeline(steps = [('scaler', scaler),
                                 ('model', lr_model)])
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)
    print(classification_report(y_test, y_preds))
    return pipeline

def get_pca_with_explainable_variance(scaled_df, threshold = 0.05):
    """This function processes SCALED data into n features via Principal Component Analysis. 
    The resulting dataframe has n features, each of which contains explained variance greater or equal to the 
    threshold value. 

    Inputs:
        scaled_df: pandas DataFrame
            - a df which has been scaled via StandardScaler() or some similar method
        threshold: float
            - a minimum explained variance cutoff that n features must account for via PCA on the scaled_df
    
    Outputs:
        reduced_df: pandas DataFrame
            - a dataFrame with n features, where n is the number of minimum features necessary to capture threshold explained variance in the scaled_df
    """
    n_max = scaled_df.shape[-1]
    min_exp_var = 100
    n  = 1
    threshold = 0.8
    while  (n < n_max):
        pca = PCA(n, whiten = True)
        X_reduced = pca.fit_transform(scaled_df)
        min_exp_var = pca.explained_variance_ratio_[-1]
        total_var = sum(pca.explained_variance_ratio_)
        if min_exp_var < 0.05:
            break
        n += 1
    
    print(f"PCA returned {n} features with {total_var} explained variance of the original dataset")
    return X_reduced

from sklearn.metrics import  silhouette_score

def get_silhouette_score(min_clusters, max_clusters, data):
    for i in range(min_clusters, max_clusters+1):
        cluster = KMeans(i, random_state = 1234)
        preds = cluster.fit_predict(data)
        centers = cluster.cluster_centers_

        score = silhouette_score(data, preds)
        print("For n_clusters = {}, silhouette score is {})".format(i, score))

def get_violin_plots(features_to_plot, n_clusters, dataframe, cluster_col):
    """gets violin plots with rows = len(features_to_plot) and n_cluster columns from dataframe
    
    Inputs:
        features_to_plot: list of str
            - a list of features to plot the violin plots of, generates rows of plots (each row is 1 feature)
            
        n_clusters: int
            - number of clusters we want to see plots for, this is an int and will plot columns per cluster
            
        dataframe: dataframe where all data (features and target) reside, this includes the cluster_label column
        
        cluster_col: the column heading for the cluster_label column
        
    Returns:
        a rows x columns subplot of violin plots
    """
    rows = len(features_to_plot)
    if rows == len(dataframe.columns):
        size = (20, 60)
    else:
        size = (20, 30)
    k = 1
    fig, axs = plt.subplots(nrows=rows, ncols=n_clusters, figsize = size)
    for j in range(rows):
        for i in range(n_clusters):
            ax = plt.subplot(rows, n_clusters, k)
            sns.violinplot(dataframe.loc[dataframe[cluster_col] == i][features_to_plot[j]])
            k+=1 
    plt.show()

def prepare_dataset_for_semi_sup_model(labeled_train_data, labeled_test_data, unlabeled_data):
    """Takes the labeled training, test, and unlabeled training datasets and combines them to be useable in 
    classification or semi-supervised learning models
    
    Inputs:
        labeled_train_data: pandas DataFrame
            - a pandas dataframe with all selected/engineered features and the target
            - expects target of column name: 'Revenue_binary'
            
        labeled_test_data: pandas DataFrame
            - a pandas dataframe with all selected/engineered features and the target
            - expects target of column name: 'Revenue_binary'
            
        unlabeled_train_data: pandas DataFrame
            - a pandas dataframe with all selected/engineered features and the target, target column will be
            ignored
            - expects target of column name: 'Revenue_binary'
            
    Outputs:
        X: np.darray
            - a concatenated np array of the training labeled and unlabeled data, features only
            
        y: np.darray
            - a concatenated np array of the training labeles and an np.array of ones of the unlabeled shape
            
        X_train_labeled: np.darray
            - a np array of the training labeled, features only
            
        y_train_labeled: np.darray
            - a np array of the training labels
            
        X_test: np.darray
            - a np array of the test labeled features only
            
        y_test: np.darray
            - a np array of the test labels
        """
    X_train_labeled = labeled_train_data.drop(columns = ['Revenue_binary'])
    y_train_labeled = labeled_train_data['Revenue_binary']

    X_train_unlabeled = unlabeled_data.drop(columns = ['Revenue_binary'])
    y_train_unlabeled  = np.ones(unlabeled_data.shape[0])

    X_test = labeled_test_data.drop(columns = ['Revenue_binary'])
    y_test = labeled_test_data['Revenue_binary']
    
    X = np.concatenate((X_train_labeled,X_train_unlabeled), axis = 0)
    y = np.concatenate((y_train_labeled, y_train_unlabeled), -1)

    try:
        (X_train_labeled.shape[0] + X_train_unlabeled.shape[0],X_train_unlabeled.shape[-1]) == X.shape
    except:
        print('concatenating labeled and unlabeled data failed due to shape error')
    return X, y, X_train_labeled, y_train_labeled, X_test, y_test

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

def get_preds_semi_sup_model(X, y, X_train_labeled_shape, model_flag = 'ls'):
    """Returns labels for the unlabeled portion of X by running either a LabelSpreading or LabelPropagation
        model. 
    
    Inputs:
         X: np.darray
            - a concatenated np array of the training labeled and unlabeled data, features only
            
        y: np.darray
            - a concatenated np array of the training labeles and an np.array of ones of the unlabeled shape
            
        X_train_labeled_shape: int
            - an int that denotes the X_train_labeled.shape[0] value, this is to tell the model where to predict 
            labels
            
        model_flag = 'ls': str
            - a str variable to denote whether to used LabelSpreading (ls) or LabelPropagation (lp)
        
    Outputs:
        labels: np.array
            - an array of binary labels predicted by model._transduction of shape X-X_train_labeled.shape[0] meant
                to represent the unlabeled data predicted labels
    """
    model_flag = model_flag.lower()
    try:
       ((model_flag == 'ls') |( model_flag == 'lp'))
    except:
        print('warning: model_flag is not recognized - needs to be ls or lr')
    if model_flag == 'ls':
        model = LabelSpreading(kernel = 'knn', alpha = 0.1)
        model.fit(X,y)
    elif model_flag == 'lp':
        model = LabelPropagation(kernel = 'knn')
        model.fit(X,y)
   
    return model.transduction_[ X_train_labeled_shape:]


def get_random_forest_predictions(X_train, y_train, X_test):
    "get random forest predictions"
    rf_model = RandomForestClassifier(n_estimators = 100, max_depth = 5, class_weight = 'balanced')
    rf_model.fit(X_train, y_train)
    return rf_model.predict(X_test)

def compare_supervised_models(labeled_train_data, unlabeled_data, labeled_test_data):
    """Takes the labeled training, test, and unlabeled training datasets and calls
     prepare_dataset_for_semi_sup_model, get_preds_semi_sup_model, and get_random_forest_predictions functions
     to produce a classification score comparing a random forest trained only on labeled training data random
     forest trained on semi-supervised self-labeled data.
    
    Inputs:
        labeled_train_data: pandas DataFrame
            - a pandas dataframe with all selected/engineered features and the target
            - expects target of column name: 'Revenue_binary'
            
        labeled_test_data: pandas DataFrame
            - a pandas dataframe with all selected/engineered features and the target
            - expects target of column name: 'Revenue_binary'
            
        unlabeled_train_data: pandas DataFrame
            - a pandas dataframe with all selected/engineered features and the target, target column will be
            ignored
            - expects target of column name: 'Revenue_binary'
            """
    X, y, X_train_labeled, y_train_labeled, X_test, y_test = prepare_dataset_for_semi_sup_model(labeled_train_data, labeled_test_data, unlabeled_data)
    #get semi-sup predicted labels for only the UNlabeled data
    semi_sup_ls_preds = get_preds_semi_sup_model(X, y, X_train_labeled.shape[0], 'ls')
    semi_sup_lp_preds = get_preds_semi_sup_model(X, y, X_train_labeled.shape[0], 'lp')

    #Train the Random forest and print their classifications

    #first train base on smaller labeled dataset, this is the 'purest' model
    rf_base_pred = get_random_forest_predictions(X_train_labeled, y_train_labeled, X_test)
    print('base model performance without any semi-supervised labeling')
    print(classification_report(y_test, rf_base_pred))

    #Train the label spreading with soft clamping, less sensitive to noise but may ignore some labeled data
    y_train_ls = np.concatenate((y_train_labeled, semi_sup_ls_preds), -1)
    rf_ls_pred = get_random_forest_predictions(X, y_train_ls, X_test)
    print('Semi-Supervised Label Spreading model performance')
    print(classification_report(y_test, rf_ls_pred))

    #Train label prop random forest, used hard clamping
    y_train_lp = np.concatenate((y_train_labeled,semi_sup_lp_preds), -1)
    rf_lp_pred = get_random_forest_predictions(X, y_train_lp, X_test)
    print('base model performance without any semi-supervised labeling')
    print(classification_report(y_test, rf_lp_pred))