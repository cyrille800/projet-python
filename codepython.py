from flask import Flask, redirect, url_for, render_template, request, flash
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import feature_selection
import sklearn
from sklearn.decomposition import TruncatedSVD
import json
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

import tensorflow as tf
from ivis import Ivis

import plotly
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import os

#initialisation de flask
app = Flask(__name__, template_folder='template')

#import dataset
df = pd.read_csv("online_shoppers_intention.csv")
df_default = pd.read_csv("online_shoppers_intention.csv")

# we find the categorical and numerical Features 
labels_count_columns = {}
for name in df.columns.to_list():
    labels_count_columns[name] = len(set(df[name]))
    

sizes = list(labels_count_columns.values())
labels = list(labels_count_columns.keys())

figcategorical, ax1 = plt.subplots(figsize=(11, 8))
figcategorical.subplots_adjust(0.3,0,1,1)

theme = plt.get_cmap('bwr')
ax1.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])


#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:len(list(labels_count_columns.keys()))]

_, _ = ax1.pie(sizes, startangle=90, colors = colors)

ax1.axis('equal')

total = sum(sizes)
plt.legend(
    labels=['%s, %1.1f%%' % (
        l, (float(s) / total) * 100) for l, s in zip(labels, sizes)],
    prop={'size': 11},
    bbox_to_anchor=(0.0, 1)
)

total = sum(sizes)
categorical_feature= []
numerical_feature= []
categorical_feature = [l for l, s in zip(labels, sizes) if (float(s) / total) * 100 <1.4 and l not in ['Administrative', 'Informational', 'ProductRelated']]
numerical_feature = list(filter(lambda x : x not in categorical_feature, labels))

##  to perform calculation operations, we need to transform the categorical variables into numerical variables, for this we will use the label encoder de sklearn
le = preprocessing.LabelEncoder()
for name in categorical_feature:
    df[name] = le.fit_transform(df[name].to_list())+1

df_analyse = df.copy()
y = df["Revenue"]

figplot_cor, ax = plt.subplots(figsize=(15,15)) 

# plotting correlation heatmap
dataplot_cor = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, ax=ax)
seuil = 0.65
orr=df.corr()
tuple_corr = []

for name in labels:
    for k,v in orr[name].to_dict().items():
        if v>seuil and name!=k:
            tuple_corr.append((name, k))
            

tuple_corr = list(set([y for item in tuple_corr for y in item]))
combi = []
for el in combinations(tuple_corr, 2):
    otherInfo= ""
    if (el[0]=="BounceRates" and el[1]=="ExitRates") or (el[0]=="ExitRates" and el[1]=="BounceRates") or (el[0]=="ProductRelated_Duration" and el[1]=="ProductRelated") or (el[0]=="ProductRelated" and el[1]=="ProductRelated_Duration"):
        otherInfo={
            "message": "une droite de regression linéaire est le meilleur moyen pour trouver la valeur manquante d'une de ces variables. NB: ceci n'est qu'une illustration:",
            "image":"https://bioinfo.iric.ca/wpbioinfo/wp-content/uploads/2017/03/lr_example-1.png"}

    else:
        otherInfo={
            "message": "une droite de regression linéaire exponentiel est le meilleur moyen pour trouver la valeur manquante d'une de ces variables. NB: ceci n'est qu'une illustration:",
            "image":"http://zoonek2.free.fr/UNIX/48_R_2004/g538.png"}
    print(el)
    combi.append([el, otherInfo])

for cop in combi:
    figplot_corlinear, ax = plt.subplots(figsize=(15,15)) 
    plt.scatter(df.loc[:,cop[0][0]], df.loc[:,cop[0][1]])
    plt.xlabel(cop[0][0], fontsize=18)
    plt.ylabel(cop[0][1], fontsize=16)
    figplot_corlinear.savefig('./static/images/presentation/corlinear'+cop[0][0]+cop[0][1]+'.png', dpi=400)
    
# Initialise the Scaler
scaler = StandardScaler()
df.loc[:,categorical_feature] = preprocessing.StandardScaler().fit_transform(df.loc[:,categorical_feature])
df.loc[:,numerical_feature] = np.log2(df.loc[:,numerical_feature]+1) 
df["Revenue"] = y-1

#representer la donnée
# supression de la colinéarité
principalComponents = TruncatedSVD(n_components=len(df.loc[:, df.columns != "Revenue"].columns)-1).fit_transform(df.loc[:, df.columns != "Revenue"])
model = Ivis(model='maaten',n_epochs_without_progress = 5, embedding_dims = 3)
if os.path.isdir('saveModelIvis'):
    model = model.load_model("saveModelIvis")
    embdeding = model.transform(principalComponents)
if os.path.isdir('saveModelIvis')==False:
    embdeding = model.fit_transform(principalComponents,df["Revenue"].values)
    model.save_model("saveModelIvis", save_format='h5', overwrite=False)


# Configure the trace.
trace = go.Scatter3d(
    x=embdeding[:,0],  # <-- Put your data instead
    y=embdeding[:,1],  # <-- Put your data instead
    z=embdeding[:,2],  # <-- Put your data instead
    mode='markers',
    marker={
        'size': 3,
        'opacity': 0.8,
        'color': ["#4747A1" if i==1 else "#F3797E" for i in df["Revenue"].values]
    }
)

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
# plotly.offline.iplot(plot_figure)

plot_figure.write_html("./static/pages/plot3d.html")

###################################
data_double = dict([(n, list(df["Revenue"].values).count(n)) for n in set(list(df["Revenue"].values))])
sizes_ = list(data_double.values())
labels_ = list(data_double.keys())

figRevenu, ax1 = plt.subplots(figsize=(6, 5))
figRevenu.subplots_adjust(0.3,0,1,1)


theme = plt.get_cmap('bwr')
ax1.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes_))])


#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:len(list(labels_count_columns.keys()))]

_, _ = ax1.pie(sizes_, startangle=90, colors = colors)

ax1.axis('equal')

total = sum(sizes_)
plt.legend(
    labels=['%s, %1.1f%%' % (
        l, (float(s) / total) * 100) for l, s in zip(labels_, sizes_)],
    prop={'size': 11},
    bbox_to_anchor=(0.0, 1)
)

#boxplot des variables
boxplot, ax1 = plt.subplots(figsize=(11, 17))
sns.boxplot(data=df)


##########################
#fonction de prediction

def select_feature(x_train, y_train,x_test,k='all',method = 'anova'):
    # configure to select all features
    fs= None
    if method == "anova":
        fs = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=k)
    else:
        fs = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=k)
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    x_test_fs = fs.transform(x_test)
    
    return x_train_fs, x_test_fs, fs

def LogisticRegression_method(x_test):
    x_train = df.iloc[:, df.columns != 'Revenue']
    y_train = df['Revenue']
    x_train_fs_anova, x_test_fs_anova, fs_anova = select_feature(x_train, y_train,x_test,k=12,method = 'anova')

    ## anova
    # fit the model
    model_anova = LogisticRegression(solver='liblinear')
    model_anova.fit(x_train_fs_anova, y_train)
    # evaluate the model
    yhat = model_anova.predict(x_test_fs_anova)
    # evaluate predictions
    return yhat

def DecisionTreeClassifier_method(x_test):
    x_train = df.iloc[:, df.columns != 'Revenue']
    y_train = df['Revenue']
    x_train_fs_anova, x_test_fs_anova, fs_anova = select_feature(x_train, y_train,x_test,k=12,method = 'anova')

    ## anova
    # fit the model
    model_anova = DecisionTreeClassifier()
    model_anova.fit(x_train_fs_anova, y_train)
    # evaluate the model
    yhat = model_anova.predict(x_test_fs_anova)
    # evaluate predictions
    return yhat


def xgboost_method(x_test):
    x_train = df.iloc[:, df.columns != 'Revenue']
    y_train = df['Revenue']
    x_train_fs_anova, x_test_fs_anova, fs_anova = select_feature(x_train, y_train,x_test,k=12,method = 'anova')

    ## anova
    # fit the model
    model_anova = xgb.XGBRegressor()
    model_anova.fit(x_train_fs_anova, y_train)
    # evaluate the model
    yhat = model_anova.predict(x_test_fs_anova)
    # evaluate predictions
    return yhat
###########################
@app.route("/")
def index():
    figcategorical.savefig('./static/images/presentation/categoricalSum.png', dpi=400)
    figRevenu.savefig('./static/images/presentation/figRevenu.png')
    return render_template("presentation.html", data = data_double)

@app.route("/presentation")
def presentation():
    figcategorical.savefig('./static/images/presentation/categoricalSum.png', dpi=400)
    figRevenu.savefig('./static/images/presentation/figRevenu.png')
    return render_template("presentation.html", data = data_double)

@app.route("/liaison")
def liaison():
    figplot_cor.savefig('./static/images/presentation/cor.png', dpi=400)
    boxplot.savefig('./static/images/presentation/boxplot.png', dpi=400)
    return render_template("liaison.html", data = combi)

@app.route("/interpretation")
def interpreation():
    return render_template("interpretation.html")

@app.route("/predire",methods = ['POST'])
def predire():
    dict_data = {}
    if request.method == 'POST':
        for el in request.form:
            dict_data[el] = [request.form[el]]

    test = pd.DataFrame(dict_data)
    resul = {
        "LogisticRegression_method" : [False] if LogisticRegression_method(test)==0 else [True],
        "xgboost_method" : [False] if xgboost_method(test)<0.5 else [True],
        "DecisionTreeClassifier_method" :[False] if DecisionTreeClassifier_method(test)==0 else [True],
    }


    dict_var = {}
    for va in categorical_feature:
        dict_var[va] = {
            "type":"categoriel",
            "val": set(df[va])
        }

    for va in numerical_feature:
        dict_var[va] = {
            "type":"value"
        }
    return render_template("prediction.html", data = [dict_var,df.iloc[:5, df.columns != 'Revenue'], pd.DataFrame(resul)])

@app.route("/prediction")
def prediction():
    dict_var = {}
    for va in categorical_feature:
        dict_var[va] = {
            "type":"categoriel",
            "val": set(df[va])
        }

    for va in numerical_feature:
        dict_var[va] = {
            "type":"value"
        }
    return render_template("prediction.html", data = [dict_var,df.iloc[:5, df.columns != 'Revenue']])

if __name__ == "__main__":
  app.run()
