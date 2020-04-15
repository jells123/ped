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

# ### Load dataframes with text and visual attributes

# +
import pandas as pd
import os

text_df = pd.read_csv(os.path.join('..', 'data', 'text_attributes_bedzju.csv'))
img_df = pd.read_csv(os.path.join('..', 'data', 'image_attributes_bedzju.csv'))
# -

# #### Text DF preview

print(text_df.shape)
text_df.head()

# #### Visual DF preview

# +
# need to convert 'list strings' into numpy arrays
for cname in img_df.columns:
    if 'histogram' in cname:
        img_df[cname] = img_df[cname].apply(lambda x : np.fromstring(x[1:-1], sep=' '))

print(img_df.shape)
img_df.head()
# -

# ### Join dataframes with visual and text attributes

# +
text_df["image_filename"] = text_df["thumbnail_link"].apply(lambda x : x.replace('/', '').replace(':', '_'))
df = text_df.set_index("image_filename").join(img_df.set_index("image_filename"))
print(df.shape)
print(df.columns)

df = df.reset_index()
df.head()


# -

# ## Perform aggregations

# +
def reduce_histogram(series):
    series = list(filter(lambda x : not isinstance(x, float), series))
    if series:
        return tuple(np.mean(series, axis=0))
    else:
        return tuple(np.zeros(5)-1.0)

def max_with_nans(series):
    result = max(series)
    if np.isnan(result):
        return -1.0
    else:
        return float(result) if isinstance(result, bool) else result

agg_df = df.groupby("video_id").agg(
    trending_date=("trending_date", lambda s : len(set(s))), # how many days this video was trending,
    category_id=("category_id", lambda s : max(s)), # if a category was given, then take it :)
    publish_time=("publish_time", lambda s : max(s)), # we expect only one publish time anyway
    
    views_median=("views", "median"),
    views_max=("views", "max"),
    
    likes_median=("likes", "median"),
    likes_max=("likes", "max"),
    
    dislikes_median=("dislikes", "median"),
    dislikes_max=("dislikes", "max"),

    comments_disabled=("comments_disabled", "mean"),
    ratings_disabled=('ratings_disabled', "mean"),
    video_error_or_removed=('video_error_or_removed', "mean"),
    
    week_day=('week_day', "max"), # we don't expect different values here
    time_of_day=("time_of_day", "max"), # as they come from publish_time column
    month=('month', "max"),
    
    title_changes=("title", lambda s : len(set(s))), # how many different titles did we have?
    title_length_chars=('title_length_chars', "median"),
    title_length_tokens=("title_length_tokens", "median"),
    title_uppercase_ratio=("title_uppercase_ratio", "mean"),
    title_not_alnum_ratio=("title_not_alnum_ratio", "mean"),
    title_common_chars_count=("title_common_chars_count", "median"),
    
    channel_title_length_chars=("channel_title_length_chars", "median"),
    channel_title_length_tokens=("channel_title_length_tokens", "median"),
    
    tags_count=("tags_count", "median"),
    
    description_changes=("description", lambda s : len(set(s))), # how many changes of description?
    description_length_chars=("description_length_chars", "median"),
    description_length_tokens=("description_length_tokens", "median"),
    description_length_newlines=("description_length_newlines", "median"),
    description_uppercase_ratio=("description_uppercase_ratio", "mean"),
    description_url_count=("description_url_count", "median"),
    description_top_domains_count=("description_top_domains_count", "median"),
    
    has_detection=("has_detection", max_with_nans),
    person_detected=("person_detected", max_with_nans),
    object_detected=("object_detected", max_with_nans),
    vehicle_detected=("vehicle_detected", max_with_nans),
    animal_detected=("animal_detected", max_with_nans),
    food_detected=("food_detected", max_with_nans),
    face_count=("face_count", max_with_nans),
    
    gray_histogram=("gray_histogram", reduce_histogram),
    hue_histogram=("hue_histogram", reduce_histogram),
    saturation_histogram=("saturation_histogram", reduce_histogram),
    value_histogram=("value_histogram", reduce_histogram),
    
    gray_median=("gray_median", "median"),
    hue_median=("hue_median", "median"),
    saturation_median=("saturation_median", "median"),
    value_median=("value_median", "median"),
    edges=("edges", "median")
)
agg_df.head()
# -

# ### Extract subsets: numeric columns, non-numeric, histograms and videos with category_id given

# +
agg_df_histograms = agg_df[[cname for cname in agg_df.columns if 'histogram' in cname]]
agg_df_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] in [np.int64, np.float64]]]
agg_df_not_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] not in [np.int64, np.float64]]]

categories_df = agg_df[~agg_df["category_id"].isna()]
categories_df.shape
# -

# ### Analyze features stats
# > In order to do so, we normalize all the values into one range: 0-1, so that the variances are more comparable

# +
normalized_df = (agg_df_numeric - agg_df_numeric.mean()) / agg_df_numeric.std()
normalized_df = (agg_df_numeric - agg_df_numeric.min()) / (agg_df_numeric.max()-agg_df_numeric.min())

stats = normalized_df.describe()
stats
# -

# #### Which columns have the highest variance?

# +
import matplotlib.pyplot as plt

std_deviations = stats.loc["std", :].sort_values(ascending=False) ** 2
std_deviations.plot.bar(figsize=(14, 7), rot=80)
# -

std_deviations[ std_deviations < 0.01 ]

# ## Feature corerlations

# +
import seaborn as sns
sns.set(rc={'figure.figsize':(18, 14)})

corr = agg_df_numeric.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

xxx = ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
# -

# ## Let's go deeper
# ### Visual attributes

# +
visual_words = ['detect', 'face', 'gray', 'hue', 'saturation', 'value', 'edges']
select_columns = [cname for cname in agg_df_numeric.columns if any([word in cname for word in visual_words])]

select_df = agg_df_numeric[select_columns]
select_df = select_df[select_df != -1.0]

corr = select_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
)

xxx = ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
# -

# ## Title, Channel Title + Description attributes

# +
select_columns = [cname for cname in agg_df_numeric.columns if any([word in cname for word in ["title", "description"]])]

select_df = agg_df_numeric[select_columns]
select_df = select_df[select_df != -1.0]

corr = select_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
)

xxx = ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
# -

# ## Print most and least correlated feature for each column

# +
import pandas as pd
import numpy as np

for idx, cname in enumerate(corr.index):
    if cname == 'category_id':
        continue
    
    max_corr = np.max(corr.loc[cname, corr.columns != cname])
    closest_idx = np.argmax(corr.loc[cname, (corr.columns != cname) & (corr.columns != "category_id")])
    print(cname, '-', corr.index[closest_idx], ' : ', max_corr)
    
    min_corr = np.min(corr.loc[cname, corr.columns != cname])
    furthest_idx = np.argmin(corr.loc[cname, (corr.columns != cname) & (corr.columns != "category_id")])
    print(cname, '-', corr.index[furthest_idx], ' : ', min_corr)
    
    print()


# -

# ### "Flatten" histogram values into columns

# +
def transform_histogram_df(df):
    for cname in df.columns:
        if 'histogram' in cname:
            prefix = cname.split('_')[0]
            for i in range(5):
                df[f"{prefix}_{i}_bin"] = df[cname].apply(lambda x : x[i])
            df = df.drop(columns=[cname])
    return df

agg_df_histograms = transform_histogram_df(agg_df[[cname for cname in agg_df.columns if 'histogram' in cname]])
     
# VERY important, remove -1.0s! 
agg_df_histograms = agg_df_histograms[agg_df_histograms["gray_0_bin"] != -1.0]

# +
corr = agg_df_histograms.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

xxx = ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

ax.hlines([0, 5, 10, 15, 20], *ax.get_xlim())
ax.vlines([0, 5, 10, 15, 20], *ax.get_xlim())
# -

# ### Histogram bins variances

# +
normalized_df = (agg_df_histograms - agg_df_histograms.min()) / (agg_df_histograms.max()-agg_df_histograms.min())

stats = normalized_df.describe()

std_deviations = stats.loc["std", :].sort_values(ascending=False) ** 2
std_deviations.plot.bar(figsize=(14, 7), rot=45)
# -

# > Feature selection is performed using ANOVA F measure via the f_classif() function.

# +
print(categories_df.shape)

categories_df_numeric = transform_histogram_df(categories_df)
categories_df_numeric = categories_df_numeric[[cname for idx, cname in enumerate(categories_df_numeric.columns) if categories_df_numeric.dtypes[idx] in [np.int64, np.float64]]]

y = categories_df_numeric["category_id"].values
X = categories_df_numeric.drop(columns=["category_id"]).fillna(-1.0)
X = (X - X.min()) / (X.max()-X.min()+1e-12) # normalize values - how about those that are missing?

X_columns = X.columns
X = X.values
X.shape

# +
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector = SelectKBest(score_func=f_classif, k=15)
fit = selector.fit(X, y)

# summarize scores
print(fit.scores_)
features = fit.transform(X)

cols = selector.get_support(indices=True)
print(list(X_columns[cols]))

X_indices = np.arange(X.shape[-1])
plt.bar(X_indices, -np.log10(selector.pvalues_), tick_label=X_columns)
plt.xticks(rotation=45)


# +
from sklearn.feature_selection import chi2

selector = SelectKBest(score_func=chi2, k=10)
fit = selector.fit(X, y)

# summarize scores
print(fit.scores_)
features = fit.transform(X)

cols = selector.get_support(indices=True)
print(list(X_columns[cols]))

X_indices = np.arange(X.shape[-1])
# plt.bar(X_indices, selector.pvalues_)


# +
from sklearn.feature_selection import mutual_info_classif

selector = SelectKBest(score_func=mutual_info_classif, k=10)
fit = selector.fit(X, y)

# summarize scores
print(fit.scores_)
features = fit.transform(X)

cols = selector.get_support(indices=True)
print(list(X_columns[cols]))

# +
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(n_splits=15, shuffle=True, random_state=15042020),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
