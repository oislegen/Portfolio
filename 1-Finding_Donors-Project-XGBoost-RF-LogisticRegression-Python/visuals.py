###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

# Import libraries for data visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
sns.set()


import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score



def analyze_numeric_features(dataset,target):
    """
    Identify and draw histograms for numeric features
    """
    # Define the color and font size for the plot below
    base_color = sns.color_palette()[0]
    # Identify numeric features in the data set
    numeric_columns=dataset.select_dtypes(include=['int','float']).columns.values.tolist()
    numeric_column_number=len(numeric_columns) # Number of numeric features
    axis_length=5 # Base axis length for each graph
    total_y_axis_length=axis_length*numeric_column_number
    f, ax = plt.subplots(numeric_column_number,2,figsize=(2.5*axis_length,total_y_axis_length))
    for i, label in enumerate(numeric_columns):
        extra=(dataset[label].max()-dataset[label].min())/14 
        bin_edges=np.arange(dataset[label].min(), np.ceil(dataset[label].max())+extra, extra)
        bin_idxs=pd.cut(dataset[label], bin_edges, include_lowest=True,labels=False).astype(int)
        pts_per_bin=dataset.groupby(bin_idxs).size()
        count_bins=pd.DataFrame(pts_per_bin[bin_idxs],columns=['count'])
        num_var_weights=np.true_divide(target,count_bins['count'])
        sns.distplot(dataset[label], bins=bin_edges, vertical=True, kde=False, hist_kws={'alpha':1}, ax= ax[i][0]) 
        ax[i,0].set_xlabel('Count')
        ax[i,1].hist(x=dataset[label], bins=bin_edges, weights=num_var_weights, orientation='horizontal', color=base_color)
        ax[i,1].set_xlabel('Mean Income (Income=1 if >=50K, 0 otherwise)') 
    f.tight_layout()
    f.show()   
    return                             


def numeric_scatterplots(dataset):
    """
    Draw scatter plots of all combinations of numeric features
    """
    from itertools import combinations
    base_color = sns.color_palette()[0]
    numeric_columns=dataset.select_dtypes(include=['int','float']).columns.values.tolist()
    numeric_column_number=len(numeric_columns) # Number of numeric features
    axis_length=5 # Base axis length for each graph
    total_y_axis_length=axis_length*len(list(combinations(numeric_columns,2)))
    f, ax = plt.subplots(len(list(combinations(numeric_columns,2))),1,figsize=(1.5*axis_length,total_y_axis_length))
    for i, (x,y) in enumerate(list(combinations(numeric_columns,2))):
        sns.regplot(dataset[x],dataset[y], color=base_color, scatter_kws={'alpha':0.1}, ax=ax[i])   

def analyze_categorical_features(dataset,target):
    """
    Write a function to build bar charts for all categorical variables as well as an adapted barcharts showing how the mean           income changes for each label of the categorical variables
    """
    # Define the color for the plot
    base_color = sns.color_palette()[0]
    # Identify categorical features in the data set
    cat_columns=dataset.select_dtypes(include=['object']).columns.values.tolist()
    cat_column_number=len(cat_columns) # Number of numeric features
    axis_length=5 # Base axis length for each graph
    total_y_axis_length=axis_length*cat_column_number
    f, ax = plt.subplots(cat_column_number,2,figsize=(2.5*axis_length,total_y_axis_length))
    for i, label in enumerate(cat_columns):
        # Get the frequency order from high to low frequency for nominal variables 
        order= dataset[label].value_counts().index
        sns.countplot(data=dataset, y=label, order=order, color=base_color, ax=ax[i][0])
        sns.barplot(x=target, y=dataset[label], order=order, color=base_color, ax=ax[i][1])
        ax[i,1].set_xlabel('Mean Income (Income=1 if >=50K, 0 otherwise)')
    # Change the absolute frequency bar charts to relative frequency
    n_points=dataset.shape[0]
    j=0
    for i in range(len(cat_columns)):   
        xlimit=ax[i,j].get_xlim()[1]
        limit=xlimit/n_points

        # Generate tick mark locations and names
        tick_props = np.arange(0, limit+0.3, 0.2)
        tick_names = ['{:.1f}'.format(v) for v in tick_props]
        ax[i,j].set_xticks(tick_props*n_points)
        ax[i,j].set_xticklabels(tick_names)
        ax[i,j].set_xlabel('Frequency')
        
        # Add annotations
        ax[i,j].get_ylabel()
        cat_counts = dataset[ax[i,j].get_ylabel()].value_counts()
        max_count = dataset[ax[i,j].get_ylabel()].value_counts().max()
        locs= ax[i,j].get_yticks() # get the current tick locations and labels
        labels=ax[i,j].get_yticklabels()
        
        # Loop through each pair of locations and labels
        for loc, label in zip(locs, labels):
            # Get the text property for the label to get the correct count
            count = cat_counts[label.get_text()]
            pct_string = '{:0.1f}%'.format(100*count/n_points)
            # Print the annotation just below the top of the bar
            ax[i,j].text(count+(limit+0.25)*n_points*0.1, loc, pct_string, ha = 'center', color = 'black')

    f.tight_layout()
    f.show() 
    return
        

def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Define the color for the plot
    base_color = sns.color_palette()[0]
    
    # Create figure
    fig = plt.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain','capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = base_color)
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 12, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 12, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
   
    # Define the color and font size for the plot below
    base_color1 = sns.color_palette()[0]
    base_color2 = sns.color_palette()[1]
    base_color3 = sns.color_palette()[2]
    matplotlib.rcParams.update({'font.size': 12})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = [base_color1, base_color2,base_color3]
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Create plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["Sample: 1%", "10%", "100%"])
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    
    #Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()
    plt.show()    

    
def feature_plot(model, importances, X_train, y_train):
    
    # Define the colors for the plot
    base_color1 = sns.color_palette()[0]
    base_color2 = sns.color_palette()[2]
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create the plot
    fig = plt.figure(figsize = (9,5))
    plt.title("{}: Normalized Weights for Five Most Important Features".format(model.__class__.__name__), fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = base_color1, \
          label = "Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = base_color2, \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Features", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  
