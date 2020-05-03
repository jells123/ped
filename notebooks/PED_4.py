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

# ## Read Aggregated Data

# +
import pandas as pd
import numpy as np
import os
import scipy.spatial
import scipy.stats as ss

agg_df = pd.read_csv('aggregated.csv')
print(agg_df.shape)
agg_df.columns
# -

# ## Read simple category_id -> title mapper

# +
import csv

# LOOKS LIKE WORST PYTHON FILE READING CODE :D

categories = {}
with open(os.path.join('..', 'data', 'categories.csv')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            categories[int(row[0])] = row[1]
        line_count += 1
        
    print(f'Processed {line_count} lines.')
    
categories
# -

# ### Apply PCA over those multi-one-hot vectors

# +
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(18, 14)})

title_onehot_feature_columns = list(filter(lambda x : 'title' in x and 'bin' in x, agg_df.columns))
X = agg_df[title_onehot_feature_columns].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_df["category_id"].fillna(0).values)
# -

category_id_indices = agg_df.index[~agg_df["category_id"].isna()].tolist()
plt.scatter(X_pca[category_id_indices, 0], X_pca[category_id_indices, 1], c=agg_df.loc[category_id_indices, "category_id"])

# ## Apply PCA over all columns, normalized by mean and std

# +
agg_df_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] in [np.int64, np.float64]]]
agg_df_not_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] not in [np.int64, np.float64]]]
agg_df_embeddings = agg_df[[cname for cname in agg_df.columns if cname.startswith('embed_')]]

all_numeric_df = agg_df_numeric.reset_index().fillna(-1).drop(columns=['trending_date', 'category_id'])
normalized_df = (all_numeric_df - all_numeric_df.mean()) / all_numeric_df.std()

X = normalized_df.values
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
ax = plt.gca()
# -

# ## Select features based on previous checkpoint's analysis

# +
ANOVA_BEST = ['time_of_day', 'title_length_chars', 'title_length_tokens', 'title_uppercase_ratio', 'title_not_alnum_ratio', 'title_common_chars_count', 'tags_count', 'description_length_chars', 'description_length_tokens', 'description_uppercase_ratio', 'description_url_count', 'description_top_domains_count', 'vehicle_detected', 'animal_detected', 'value_median', 'title_0_bin', 'title_1_bin', 'title_2_bin', 'title_5_bin', 'title_12_bin']

CHI2_BEST = ['likes_median', 'likes_max', 'comments_disabled', 'ratings_disabled', 'month', 'title_changes', 'title_length_chars', 'title_uppercase_ratio', 'title_common_chars_count', 'channel_title_length_tokens', 'tags_count', 'description_length_chars', 'description_length_newlines', 'description_url_count', 'description_top_domains_count', 'description_emojis_counts', 'ocr_length_tokens', 'angry_count', 'fear_count', 'happy_count']

MI_BEST = ['likes_median', 'dislikes_max', 'title_uppercase_ratio', 'title_common_chars_count', 'channel_title_length_chars', 'description_length_newlines', 'description_top_domains_count', 'has_detection', 'person_detected', 'vehicle_detected', 'animal_detected', 'food_detected', 'face_count', 'saturation_median', 'title_1_bin', 'title_2_bin', 'title_5_bin', 'title_7_bin', 'title_8_bin', 'title_13_bin']

RFECV_BEST = ['comments_disabled', 'week_day', 'time_of_day', 'month',
       'title_length_chars', 'title_length_tokens', 'title_uppercase_ratio',
       'title_not_alnum_ratio', 'title_common_chars_count',
       'channel_title_length_chars', 'channel_title_length_tokens',
       'tags_count', 'description_changes', 'description_length_chars',
       'description_length_newlines', 'description_uppercase_ratio',
       'description_url_count', 'description_top_domains_count',
       'person_detected', 'object_detected', 'vehicle_detected',
       'animal_detected', 'food_detected', 'face_count', 'gray_median',
       'hue_median', 'saturation_median', 'value_median', 'ocr_length_tokens',
       'angry_count', 'fear_count', 'happy_count', 'gray_0_bin', 'gray_1_bin',
       'title_0_bin', 'title_1_bin', 'title_2_bin', 'title_3_bin',
       'title_5_bin', 'title_7_bin', 'title_9_bin', 'title_10_bin',
       'title_11_bin', 'title_12_bin', 'title_13_bin', 'title_14_bin',
       'title_16_bin', 'title_17_bin', 'title_18_bin', 'title_19_bin']

N = 20
SELECT_FEATURES = list(set([*ANOVA_BEST[:N], *CHI2_BEST[:N], *MI_BEST[:N], *RFECV_BEST[:N]]))
len(SELECT_FEATURES), len(agg_df.columns)
# -

# ## Apply PCA over SELECTED FEATURES

# +
select_features_df = agg_df_numeric.fillna(0)[SELECT_FEATURES]
normalized_df = (select_features_df - select_features_df.mean()) / select_features_df.std()

X_all = normalized_df.values
y_all = list(map(int, agg_df.fillna(-1).loc[:, "category_id"].values))

pca_all = PCA(n_components=5)
X_pca_all = pca_all.fit_transform(X_all)

import seaborn as sns

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    size='has_category',
    data=pd.DataFrame({
      'c1': X_pca_all[:, 0],
      'c2': X_pca_all[:, 1],
      'category': list(map(lambda x : categories.get(int(x), "undefined"), y_all)),
        'has_category': list(map(lambda x : 1 if x == -1 else 15, y_all))
  })); 

# +
labeled_idx = agg_df.index[~agg_df["category_id"].isna()].tolist()
X = normalized_df.loc[labeled_idx, :].values
y = list(map(int, agg_df.loc[labeled_idx, "category_id"].values))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

import seaborn as sns

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    data=pd.DataFrame({
      'c1': X_pca[:, 0],
      'c2': X_pca[:, 1],
      'category': list(map(lambda x : categories.get(int(x), "undefined"), y)),
  })); 
# -

_ = normalized_df.hist(bins=20)
plt.show()

# ## Distribution of known categories

ax = sns.countplot(
    x="category", 
    data=pd.DataFrame({"category": map(lambda x : categories.get(x),filter(lambda x : x > -1, y_all))})
)

# ## Try: supervised apprroach vs. naive Self Learning Model

# +
import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel

# supervised score 
# basemodel = WQDA() # weighted Quadratic Discriminant Analysis
basemodel = SGDClassifier(loss='log', penalty='l2', random_state=20200501) # scikit logistic regression
basemodel.fit(X, y)
print("supervised log.reg. score", basemodel.score(X, y))

y = np.array(y)
y_all = np.array(y_all)

# # fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X_all, y_all)
print("self-learning log.reg. score", ssmodel.score(X, y))
# -

# ## Label Spreading

# +
from sklearn.semi_supervised import LabelSpreading

# label_spread = LabelSpreading(kernel='knn', alpha=0.8, max_iter=1000)
label_spread = LabelSpreading(kernel='knn', alpha=0.2, max_iter=1000)

label_spread.fit(X_all, y_all)

# +
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred = label_spread.predict(X)
cm = confusion_matrix(y, y_pred, labels=label_spread.classes_)

print(classification_report(y, y_pred))

disp = plot_confusion_matrix(label_spread, X, y,
                                 display_labels=label_spread.classes_,
                                 cmap=plt.cm.Blues)
# -

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    size='correct',
    data=pd.DataFrame({
      'c1': X_pca_all[:, 0],
      'c2': X_pca_all[:, 1],
      'category': list(map(lambda x : categories.get(int(x), "undefined"), 
                          label_spread.predict(X_all))),
        'correct': list(map(
            lambda x : 15 if x[0] == x[1] else 1, zip(y_all, label_spread.predict(X_all))))
  })); 

# ## Entropies

# +
from scipy import stats

# #############################################################################
# Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(label_spread.label_distributions_.T)
print(pred_entropies.shape)

sns.distplot(pred_entropies)
# -

# ### Read original dataframe to reference original titles & tags

# +
path = "../data/"

GB_videos_df = pd.read_csv(path + "/" + "GB_videos_5p.csv", sep=";", engine="python")
US_videos_df = pd.read_csv(path + "/" + "US_videos_5p.csv", sep=";", engine="python")

df = pd.concat([GB_videos_df, US_videos_df]).drop_duplicates().reset_index(drop=True)
df = df.rename(columns={"description ": "description"})
print(df.shape)
df.head(3) 
# -

# ## Least certain

# +
transductions_entropies = list(zip(
    label_spread.transduction_, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : -1*x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        print(select_from_df.loc[:, ["title"]].values[0][0])
        print(select_from_df.loc[:, ["tags"]].values[0][0])
        print()

# -

# ## Most certain

# +
transductions_entropies = list(zip(
    label_spread.transduction_, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        if select_from_df.shape[0] > 0:
            print(select_from_df.loc[:, ["title"]].values[0][0])
            print(select_from_df.loc[:, ["tags"]].values[0][0][:100])
            print()

# -


