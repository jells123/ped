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

# ## Read & Aggregate Data

# +
import pandas as pd
import numpy as np
import os
import scipy.spatial
import scipy.stats as ss

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

def cast_to_list(x):
    if x:
        return [float(num) for num in x[1:-1].replace("\n", "").split(" ") if num]
    else:
        return None

FIGURES_DIR=os.path.join('..', 'figures')

text_df = pd.read_csv(os.path.join('..', 'data', 'text_attributes_bedzju.csv'))
img_df = pd.read_csv(os.path.join('..', 'data', 'image_attributes_bedzju.csv'))

img_df_2 = pd.read_csv(os.path.join('..', 'data', 'image_attributes_nawrba.csv'))
text_df_2 = pd.read_csv(os.path.join('..', 'data', 'text_attributes_nawrba.csv'))

print(text_df.shape, img_df.shape)
print(text_df_2.shape, img_df_2.shape)

# need to convert 'list strings' into numpy arrays
for cname in img_df.columns:
    if 'histogram' in cname:
        img_df[cname] = img_df[cname].apply(lambda x : np.fromstring(x[1:-1], sep=' '))

text_df["image_filename"] = text_df["thumbnail_link"].apply(lambda x : x.replace('/', '').replace(':', '_'))

df = pd.concat([text_df, text_df_2, img_df_2], axis=1).set_index("image_filename").join(img_df.set_index("image_filename"))
df = df.reset_index()

for column in ['channel_title_embed', 'transormed_tags_embed', 'thumbnail_ocr_embed', "title_embed"]:
    df[column] = df[column].apply(cast_to_list)

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
    print("\n\t>>> CATEGORY ", categories[i], " -> ", len(titles))
    for w in get_top_n_words(titles, n=10):
        print(w)

# ### Top `TITLE` words by Tf IDF score, per category

for i in df["category_id"].value_counts().keys():
    titles = np.unique(df[df["category_id"] == i]["title"].values)
    print("\n\t>>> CATEGORY ", categories[i], " -> ", len(titles))
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
# -

# ### Apply PCA over those multi-one-hot vectors

# +
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(18, 14)})

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
    print("\n\t>>> CATEGORY ", categories[i], " -> ", len(titles))
    for w in get_top_n_words(titles, n=10):
        print(w)

# ### Top `tags` words by TF IDF score, per category

for i in df["category_id"].value_counts().keys():
    tags = np.unique(df[df["category_id"] == i]["tags"].values)
    print("\n\t>>> CATEGORY ", categories[i], " -> ", len(titles))
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

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=bow_agg_df["category_id"].fillna(0).values)
# -

agg_df_numeric.reset_index().fillna(-1).drop(columns=['video_id', 'trending_date', 'category_id'])

# ## Apply PCA over all columns, normalized by mean and std

# +
agg_df_histograms = agg_df[[cname for cname in agg_df.columns if 'histogram' in cname]]
agg_df_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] in [np.int64, np.float64]]]
agg_df_not_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] not in [np.int64, np.float64]]]
agg_df_embeddings = agg_df[[cname for cname in agg_df.columns if cname.startswith('embed_')]]

all_numeric_df = agg_df_numeric.reset_index().fillna(-1).drop(columns=['video_id', 'trending_date', 'category_id'])
normalized_df = (all_numeric_df - all_numeric_df.mean()) / all_numeric_df.std()

X = normalized_df.values
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
ax = plt.gca()

# -

# ## Select features based on previous checkpoint's analysis

# +
ANOVA_BEST = [
    'likes_max', 
    'time_of_day',
    'title_length_chars', 
    'title_length_tokens', 
    'title_uppercase_ratio', 
    'title_not_alnum_ratio', 
    'title_common_chars_count', 
    'channel_title_length_tokens', 
    'tags_count', 
    'description_length_chars', 
    'description_length_tokens', 
    'description_uppercase_ratio', 
    'description_url_count', 
    'description_top_domains_count', 
    'vehicle_detected', 
    'animal_detected', 
    'value_median', 
    'ocr_length_tokens', 
    'saturation_0_bin', 
    'value_4_bin'
]
CHI2_BEST = [
    'likes_median',
    'likes_max',
    'comments_disabled',
    'ratings_disabled',
    'month',
    'title_changes',
    'title_length_chars',
    'title_uppercase_ratio',
    'title_common_chars_count',
    'channel_title_length_tokens',
    'tags_count',
    'description_length_chars',
    'description_length_newlines',
    'description_url_count',
    'description_top_domains_count',
    'description_emojis_counts',
    'ocr_length_tokens',
    'angry_count',
    'fear_count',
    'happy_count',
]
MI_BEST = [
    'likes_median',
    'dislikes_max',
    'month',
    'title_uppercase_ratio',
    'title_common_chars_count',
    'channel_title_length_chars',
    'description_length_tokens',
    'description_length_newlines',
    'description_uppercase_ratio',
    'description_url_count',
    'has_detection',
    'person_detected',
    'vehicle_detected',
    'food_detected',
    'face_count',
    'saturation_median',
    'gray_0_bin',
    'saturation_0_bin',
    'saturation_3_bin',
    'value_1_bin',
]
RFECV_BEST = [
    'likes_median', 'comments_disabled', 'week_day', 'time_of_day', 'month',
   'title_changes', 'title_length_chars', 'title_length_tokens',
   'title_uppercase_ratio', 'title_not_alnum_ratio',
   'title_common_chars_count', 'channel_title_length_chars',
   'channel_title_length_tokens', 'tags_count', 'description_changes',
   'description_length_chars', 'description_length_tokens',
   'description_length_newlines', 'description_uppercase_ratio',
   'description_url_count', 'description_top_domains_count',
   'has_detection', 'person_detected', 'object_detected',
   'vehicle_detected', 'animal_detected', 'face_count', 'gray_median',
   'hue_median', 'saturation_median', 'value_median', 'ocr_length_tokens',
   'angry_count', 'fear_count', 'happy_count', 'hue_0_bin', 'hue_1_bin',
   'hue_2_bin', 'hue_3_bin', 'hue_4_bin', 'saturation_0_bin',
   'saturation_1_bin', 'value_0_bin', 'value_1_bin', 'value_2_bin'
]

N = 15
SELECT_FEATURES = list(set([*ANOVA_BEST[:N], *CHI2_BEST[:N], *MI_BEST[:N], *RFECV_BEST[:N]]))
len(SELECT_FEATURES), len(agg_df.columns)
# -

# ## Apply PCA over SELECTED FEATURES

# +
select_features_df = agg_df_numeric.fillna(0)[SELECT_FEATURES]
normalized_df = (select_features_df - select_features_df.mean()) / select_features_df.std()

X = normalized_df.values
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

import seaborn as sns

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    size='has_category',
    data=pd.DataFrame({
      'c1': X_pca[:, 0],
      'c2': X_pca[:, 1],
      'category': list(map(lambda x : categories.get(int(x), "undefined"), agg_df_numeric.fillna(0)["category_id"].values)),
        'has_category': list(map(lambda x : 1 if x == -1 else 15, agg_df_numeric.fillna(-1)["category_id"].values))
  })); 
