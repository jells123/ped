# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Load dataframes with text and visual attributes

# +
import pandas as pd
import numpy as np
import os
import scipy.spatial
import scipy.stats as ss

FIGURES_DIR=os.path.join('..', 'figures')

# +
text_df = pd.read_csv(os.path.join('..', 'data', 'text_attributes_bedzju.csv'))
img_df = pd.read_csv(os.path.join('..', 'data', 'image_attributes_bedzju.csv'))

img_df_2 = pd.read_csv(os.path.join('..', 'data', 'image_attributes_nawrba.csv'))
text_df_2 = pd.read_csv(os.path.join('..', 'data', 'text_attributes_nawrba.csv'))
# -

# #### Text DF preview

print(text_df.shape, img_df_2.shape, text_df_2.shape)
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

df = pd.concat([text_df, text_df_2, img_df_2], axis=1).set_index("image_filename").join(img_df.set_index("image_filename"))
print(df.shape)
print(df.columns)

df = df.reset_index()
df.head()
# -

list(df[['channel_title_embed', 'transormed_tags_embed', 'thumbnail_ocr_embed']].dtypes)


# +
def cast_to_list(x):
    if x:
        return [float(num) for num in x[1:-1].replace("\n", "").split(" ") if num]
    else:
        return None

for column in ['channel_title_embed', 'transormed_tags_embed', 'thumbnail_ocr_embed', "title_embed"]:
    df[column] = df[column].apply(cast_to_list)
# -

df[['channel_title_embed', 'transormed_tags_embed', 'thumbnail_ocr_embed', "title_embed"]].isnull().describe()


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
    
def reduce_medoid(series):
    series = np.array([row for row in series.to_numpy()])
    dist_matrix = scipy.spatial.distance_matrix(series, series)    
    return tuple(series[np.argmin(dist_matrix.sum(axis=0))])

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
    description_emojis_counts = ('emojis_counts', "median"),
    
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
    edges=("edges", "median"),
    
    ocr_length_tokens=('thumbnail_ocr_length', "median"),
    angry_count=('angry_count', "median"),
    surprise_count=('surprise_count', "median"),
    fear_count=('fear_count', "median"),
    happy_count=('happy_count', "median"),
    
    embed_title=('title_embed', reduce_medoid), 
    embed_channel_title=('channel_title_embed', reduce_medoid),
    embed_transormed_tags=('transormed_tags_embed', reduce_medoid), 
    embed_thumbnail_ocr=('thumbnail_ocr_embed', reduce_medoid),
)
agg_df.head()
# -

# ### Extract subsets: numeric columns, non-numeric, histograms and videos with category_id given

# +
agg_df_histograms = agg_df[[cname for cname in agg_df.columns if 'histogram' in cname]]
agg_df_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] in [np.int64, np.float64]]]
agg_df_not_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] not in [np.int64, np.float64]]]
agg_df_embeddings = agg_df[[cname for cname in agg_df.columns if cname.startswith('embed_')]]

categories_df = agg_df[~agg_df["category_id"].isna()]
categories_df.shape
# -

# ### Analyze features stats
# > In order to do so, we normalize all the values into one range: 0-1, so that the variances are more comparable

# +
# normalized_df = (agg_df_numeric - agg_df_numeric.mean()) / agg_df_numeric.std()

normalized_df = (agg_df_numeric - agg_df_numeric.min()) / (agg_df_numeric.max()-agg_df_numeric.min())

stats = normalized_df.describe()
stats
# -

# #### Which columns have the highest variance?

# +
import matplotlib.pyplot as plt

std_deviations = stats.loc["std", :].sort_values(ascending=False)
std_plot = std_deviations.plot.bar(figsize=(14, 7), rot=80)
std_plot.get_figure().savefig("std_dev.pdf")
std_plot.plot()
# -

std_deviations[ std_deviations < 0.1 ]

# ## Feature corerlations
# > Using `-1.0` to denote missing values will potentially break the usefulness of correlation coef, so in the next heatmaps we split the features by their 'domain' (text or visual), skipping the missing values. This makes new coefficients more relevant.

# +
import seaborn as sns
sns.set(rc={'figure.figsize':(18, 14)})

corr = agg_df_numeric[[
    cname for cname in agg_df_numeric.columns if cname not in ["trending_date", "category_id"]]
].corr()
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

ax.get_figure().savefig(os.path.join(FIGURES_DIR, "corr_all.pdf"))

# ## Let's go deeper
# ### Visual attributes

# +
visual_words = ['detect', 'face', 'gray', 'hue', 'saturation', 'value', 'edges', "ocr_length_tokens", "angry_count", "surprise_count", "fear_count", "happy_count"]
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

ax.get_figure().savefig(os.path.join(FIGURES_DIR, "corr_visual.pdf"))

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

ax.get_figure().savefig(os.path.join(FIGURES_DIR, "corr_desc.pdf"))

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

# ### Embeddings comparison

for column_name in agg_df_embeddings.columns:
    print(column_name)
    agg_df_embeddings[column_name + "_argmax"] = agg_df_embeddings[column_name].copy().apply(np.argmax).copy()

agg_df_embeddings.columns


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


max_embed_column_names = [colname for colname in agg_df_embeddings.columns if colname.endswith("_argmax")]
corr = []
for column_name in max_embed_column_names:
    corr.append([])
    for column_name_2 in max_embed_column_names:
        confusion_matrix = pd.crosstab(agg_df_embeddings[column_name], agg_df_embeddings[column_name_2]).values
        # print(confusion_matrix)
        corr[-1].append(cramers_corrected_stat(confusion_matrix))

corr

# +
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True,
    xticklabels=max_embed_column_names,
    yticklabels=max_embed_column_names
)

xxx = ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

yxx = ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
)
# -

ax.get_figure().savefig(os.path.join(FIGURES_DIR, "corr_embed.pdf"))


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

std_deviations = stats.loc["std", :].sort_values(ascending=False)
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

cols = selecport(indices=True)
print(list(X_columns[coltor.get_sups]))

X_indices = np.arange(X.shape[-1])
plt.bar(X_indices, -np.log10(selector.pvalues_), tick_label=X_columns)
# plt.bar(X_indices, selector.pvalues_, tick_label=X_columns)

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
svc = SVC(kernel="linear", class_weight='balanced')
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
# -

# # Try more features on text data

# ### Test `CountVectorizer` and `TfIdfVectorizer` in terms of distinguishing between categories

# +
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(np.unique(df["title"].values))

def top_tfidf_scores(corpus, n=15):
    # http://stackoverflow.com/questions/16078015/
    MAX_DF = 0.3 if len(corpus) > 10 else 1.0
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100, max_df=MAX_DF, sublinear_tf=True)
    tfidf_result = vectorizer.fit_transform(corpus)
    
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:n]

top_tfidf_scores(np.unique(df["title"].values))


# +
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

titles = np.unique(df["title"].values)
get_top_n_words(titles, n=20)
# -

# ### Top `TITLE` words by counts, per category

for i in df["category_id"].value_counts().keys():
    titles = np.unique(df[df["category_id"] == i]["title"].values)
    print("\n\t>>> CATEGORY ", i, " -> ", len(titles))
    for w in get_top_n_words(titles, n=10):
        print(w)

# ### Top `TITLE` words by Tf IDF score, per category

for i in df["category_id"].value_counts().keys():
    titles = np.unique(df[df["category_id"] == i]["title"].values)
    print("\n\t>>> CATEGORY ", i, " -> ", len(titles))
    for w in top_tfidf_scores(titles, n=10):
        print(w)

# ### Select title-specific words to construct bag-of-words 
# (very small vocabulary)
#
# ### Construct one-hot vectors over this small vocabulary as a new feature

# +
titles_bow = []
N = 30

for i in df["category_id"].value_counts().keys():
    titles = np.unique(df[df["category_id"] == i]["title"].values)
    titles_bow.extend(w[0] for w in get_top_n_words(titles, n=N))
    titles_bow.extend(w[0] for w in top_tfidf_scores(titles, n=N))

titles_bow = list(sorted(set(titles_bow)))

def onehot_encode(x, BOW):
    x_lower = x.lower()
    result = np.zeros(shape=len(BOW), dtype=np.uint8)
    for idx, w in enumerate(BOW):
        if w in x_lower:
            result[idx] += 1
    return result

titles_onehot = []
df["title_onehot"] = df["title"].apply(lambda x : onehot_encode(x, titles_bow))

bow_agg_df = df.groupby("video_id").agg(
    title_onehot=("title_onehot", reduce_histogram),
    category_id=("category_id", lambda s : max(s)),
).reset_index()
bow_agg_df.head()

# +
from sklearn.decomposition import PCA

bow_data = np.stack(bow_agg_df["title_onehot"].values, axis=0)
print(bow_data.shape)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(bow_data)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=bow_agg_df["category_id"].fillna(0).values)
# -

category_id_indices = bow_agg_df.index[~bow_agg_df["category_id"].isna()].tolist()
plt.scatter(X_pca[category_id_indices, 0], X_pca[category_id_indices, 1], c=bow_agg_df.loc[category_id_indices, "category_id"])

# ### Top `TAGS` words by counts, per category

for i in df["category_id"].value_counts().keys():
    titles = np.unique(df[df["category_id"] == i]["tags"].values)
    print("\n\t>>> CATEGORY ", i, " -> ", len(titles))
    for w in get_top_n_words(titles, n=10):
        print(w)

# ### Top `tags` words by TF IDF score, per category

for i in df["category_id"].value_counts().keys():
    tags = np.unique(df[df["category_id"] == i]["tags"].values)
    print("\n\t>>> CATEGORY ", i, " -> ", len(tags))
    for w in top_tfidf_scores(tags, n=10):
        print(w)

# +
tags_bow = []
N = 30

for i in df["category_id"].value_counts().keys():
    tags = np.unique(df[df["category_id"] == i]["tags"].values)
    tags_bow.extend(w[0] for w in get_top_n_words(tags, n=N))
    tags_bow.extend(w[0] for w in top_tfidf_scores(tags, n=N))

tags_bow = list(sorted(set(tags_bow)))

tags_onehot = []
df["tags_onehot"] = df["tags"].apply(lambda x : onehot_encode(x, tags_bow))

bow_agg_df = df.groupby("video_id").agg(
    title_onehot=("title_onehot", reduce_histogram),
    tags_onehot=("tags_onehot", reduce_histogram),
    category_id=("category_id", lambda s : max(s)),
).reset_index()
bow_agg_df.head()

# +
bow_data = np.stack(bow_agg_df["tags_onehot"].values, axis=0)
print(bow_data.shape)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(bow_data)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=bow_agg_df["category_id"].fillna(0).values)
