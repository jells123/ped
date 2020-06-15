# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: ped-venv
#     language: python
#     name: ped-venv
# ---

# ### Loading original dataframes

# +
import pandas as pd
import os

pd.set_option("colwidth", None)

PATH = os.path.join('..', 'data')
GB_videos_df = pd.read_csv(os.path.join(PATH, "GB_videos_5p.csv"), sep=";", engine="python")
US_videos_df = pd.read_csv(os.path.join(PATH, "US_videos_5p.csv"), sep=";", engine="python")

df = pd.concat([GB_videos_df, US_videos_df]).drop_duplicates().reset_index(drop=True)
df = df.rename(columns={"description ": "description"})
print(df.shape)
df.head(3) 
# -

# ### Loading `detailed*.json` files with downloaded YT API data

# +
import os
import json

data = []
for filename in os.listdir('../data/'):
    if 'detailed' in filename:
        print(filename)
        with open(os.path.join('..', 'data', filename), "r") as handle:
            data.extend(json.load(handle))

len(data)
# -

# ### Process this data & transform into pandas DataFrame

# +
import pandas as pd

df_dict = {key : [] for key in df.columns}
mapper = {
    "video_id": "id",
    "trending_date": "",
    "title": "snippet.title",
    "channel_title": "snippet.channelTitle",
    "category_id": "snippet.categoryId",
    "publish_time": "snippet.publishedAt",
    "tags": "snippet.tags", # special handling here!
    "views": "statistics.viewCount",
    "likes": "statistics.likeCount",
    "dislikes": "statistics.dislikeCount",
    "comment_count": "statistics.commentCount",
    "thumbnail_link": "snippet.thumbnails.default.url",
    "comments_disabled": "",
    "ratings_disabled": "",
    "video_error_or_removed": "",
    "description": "snippet.description"
}

miss_count = 0
for element in data:
    for key in mapper:
        try:
            if '.' in mapper[key]:
                get_value = element
                for subkey in mapper[key].split('.'):
                    get_value = get_value[subkey]
            elif mapper[key] == "":
                get_value = "?"
            else:
                get_value = element[mapper[key]]

            if isinstance(get_value, list):
                get_value = '|'.join(map(lambda x : '"' + x + '"', get_value))
        except:
            get_value = -1
            print(element['id'])
            print(f"Key {key} is missing!")
            miss_count += 1
            
        df_dict[key].append(get_value)
        
print(f"Missed {miss_count} keys")

pd.options.display.max_columns = None

new_df = pd.DataFrame(df_dict)
new_df.head(5)
# -

# ### Load preprocessed train & test + fix `description_length_newlines` column

# +
train = pd.read_csv(os.path.join(PATH, "aggregated_train_no_embeddings.csv"))
test = pd.read_csv(os.path.join(PATH, "aggregated_test_no_embeddings.csv"))

new_df["description_length_newlines"] = new_df["description"].apply(count_newlines)
new_data = new_df.loc[:, ["video_id", "description_length_newlines"]].groupby("video_id").agg(
    description_length_newlines=("description_length_newlines", "median")
).reset_index().values.tolist()
new_data = dict(new_data)

def f(video_id, is_trending, old_value):
    if is_trending:
        return old_value
    elif video_id in new_data.keys() and not is_trending:
        return new_data.get(video_id)
    else:
        return "?"

train["description_length_newlines"] = train.apply(lambda x : f(x.video_id, x.is_trending, x.description_length_newlines), axis=1)
test["description_length_newlines"] = test.apply(lambda x : f(x.video_id, x.is_trending, x.description_length_newlines), axis=1)
# -

# ### Read best features from jsons and exclude some of them
# > The Youtuber will not have any influence on: views, likes, or potentially: month of publish 
#
#
# > Skipping number of changes in description and title, as for non-trending videos such information is missing

# +
best_attributes = {}
for filename in os.listdir(PATH):
    if "best_all" in filename and "no_embeddings" in filename:
        with open(os.path.join(PATH, filename)) as handle:
            best = json.load(handle)
            best_attributes[filename.split("_")[0]] = best
 
flatten = lambda l: [item for sublist in l for item in sublist]
any_best = set(flatten(list(best_attributes.values())))

SELECT_FEATURES = []
for feature in any_best:
    if any(word in feature for word in ['views', 'likes', 'changes', 'month']):
        continue
    else:
        SELECT_FEATURES.append(feature)
# -

train.loc[:, SELECT_FEATURES].describe()

# +
import numpy as np

y_train = list(map(int, train.loc[:, "is_trending"].values.tolist()))
X_train = np.asarray(train.loc[:, SELECT_FEATURES].values, dtype=np.float64)

y_test = list(map(int, test.loc[:, "is_trending"].values.tolist()))
X_test = np.asarray(test.loc[:, SELECT_FEATURES].values, dtype=np.float64)

X_train[np.isnan(X_train)] = -1.0
X_test[np.isnan(X_test)] = -1.0
# -

# ## Run `GridSearch` with Cross Validation using Logistic Regression classifier
# ### Tune for best `f1-score` possible (micro averaged = weighted by class counts)
# #### Applied `MinMaxScaler()` due to convergence issues - all fetaures are forced to be in range [0, 1], scaled by Min and Max

# +
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.fit_transform(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

grid = {
    "C" : np.logspace(-3, 3, 7), 
    "solver": ['lbfgs', 'saga', 'newton-cg', 'liblinear'],
    "class_weight": ["balanced", None]
}
logreg = LogisticRegression(random_state=20201506, max_iter=2000)
logreg_cv = GridSearchCV(logreg, grid, cv=5, verbose=3, scoring='f1_micro')
logreg_cv.fit(X_train_minmax, y_train)
# -

# #### What are the best params?

# +
print("tuned hyperparameters : (best parameters) ", logreg_cv.best_params_)
print("f1-score (train dataset):", logreg_cv.best_score_)

clf = logreg_cv.best_estimator_
clf
# -

# ### Evaluate on test set
# > The model mostly suffers from False Negatives. Some trending videos are not recognized.
#
# > The other way, there are only few mistakes: 11 False Positives

# +
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

pred = clf.predict(X_test_minmax)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')
f1 = f1_score(y_test, pred, average='micro')

print("accuracy = %.5f, precision = %.5f, recall = %.5f, f1 = %.5f" % (accuracy, precision, recall, f1))

disp = plot_confusion_matrix(clf, X_test_minmax, y_test,
                             display_labels=["non trending", "trending"],
                             cmap=plt.cm.Blues,
                             values_format=''
                            )
disp.ax_.set_title("Logistic Regression Confusion Matrix")
print(disp.confusion_matrix)

# +
pred_proba = clf.predict_proba(X_test_minmax)
proba_with_true = list(zip(pred_proba, y_test))

nontrending_most_certain = sorted(proba_with_true, key=lambda x : -1 * x[0][0])
trending_most_certain = sorted(proba_with_true, key=lambda x : -1 * x[0][1])
# -

# ### Preview model's coefficients per feature

feature_coefficients = list(zip(SELECT_FEATURES, clf.coef_[0].tolist()))
sorted(feature_coefficients, key=lambda x : -x[1])

# +
import shap

shap.initjs()

explainer = shap.LinearExplainer(clf, X_train_minmax, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_minmax)

# +
false_negatives = []
false_positives = []
true_positives = []
true_negatives = []

for idx, (predicted, true) in enumerate(list(zip(pred, y_test))):
    if predicted == true:
        if true == 1:
            true_positives.append(idx)
        else:
            true_negatives.append(idx)
    else:
        if predicted == 1 and true == 0:
            false_positives.append(idx)
        elif predicted == 0 and true == 1:
            false_negatives.append(idx)

len(false_negatives), len(false_positives)
# -

ind = true_positives[0]
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_minmax[ind,:],
    feature_names=SELECT_FEATURES
)

ind = true_negatives[0]
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_minmax[ind,:],
    feature_names=SELECT_FEATURES
)

ind = false_positives[0]
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_minmax[ind,:],
    feature_names=SELECT_FEATURES
)

ind = false_negatives[0]
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_minmax[ind,:],
    feature_names=SELECT_FEATURES
)

shap.summary_plot(shap_values, X_test_minmax, feature_names=SELECT_FEATURES, plot_type="bar")

# +
import seaborn as sns
sns.set()

sns.distplot(train[train["is_trending"]]["edges"])
sns.distplot(train[~train["is_trending"]]["edges"])
# -

sns.distplot(train[train["is_trending"]]["description_length_newlines"])
sns.distplot(train[~train["is_trending"]]["description_length_newlines"])


