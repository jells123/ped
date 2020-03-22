# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + colab_type="code" id="RWSJpsyKqHjH" outputId="e65be45c-5485-45a9-ee9f-03e4df37f740" colab={"base_uri": "https://localhost:8080/", "height": 133}
import nltk

nltk.download("punkt")

import os

path = "../data/"
files = os.listdir(path)
files

# + id="DbzcWu-ZnRsQ" colab_type="code" outputId="1fcb3627-9804-480a-aa0f-0c5253263725" colab={"base_uri": "https://localhost:8080/", "height": 324}
import pandas as pd

pd.set_option("colwidth", -1)

GB_videos_df = pd.read_csv(path + "/" + "GB_videos_5p.csv", sep=";", engine="python")
US_videos_df = pd.read_csv(path + "/" + "US_videos_5p.csv", sep=";", engine="python")

df = pd.concat([GB_videos_df, US_videos_df]).drop_duplicates().reset_index(drop=True)
df = df.rename(columns={"description ": "description"})
print(df.shape)
df.head(3)

# + [markdown] id="l8P8ev47QLxX" colab_type="text"
# ## Unwanted attributes
# - We do not need to analyze **views**, **likes**, **dislikes** or **comment_count** as we cannot base the trending guidelines upon such statistics

# + [markdown] id="Yggnu6MUGqIE" colab_type="text"
# ## Check for **missing values**
# Apart from category_id column about which we already know it has values missing, there are other attributes with missing data.
#
# ### Description

# + id="iIPd0TVvFqAG" colab_type="code" outputId="181ade2b-8956-4db3-cc91-5842f68927d3" colab={"base_uri": "https://localhost:8080/", "height": 33}
missing_values_df = df.drop(["category_id"], axis=1)
missing_values_df = missing_values_df[missing_values_df.isnull().any(axis=1)]

for cname in missing_values_df.columns:
    check_nulls = missing_values_df[[cname]].isnull().sum().values[0]
    if check_nulls > 0:
        print("Missing values in column", cname, ":", check_nulls)

# + [markdown] id="S0qEV6gbO41G" colab_type="text"
# There are NaNs in column `description`.
#
# **Solution**: Replace `NaN`s with "no description"

# + id="kYPvhOLo3ucr" colab_type="code" outputId="6e416f75-ba35-4ce4-94b3-126a7645e3b1" colab={"base_uri": "https://localhost:8080/", "height": 117}
df[df["description"].isna()].loc[:, "description"] = "no description"

# + [markdown] id="0qhw5A3Z3tAE" colab_type="text"
# ### Tags
# We can also observe that there can be missing tags, represented as `[none]`. We leave it as it is as no tags is also some kind of an information.

# + id="pqvqxS6kIxp4" colab_type="code" outputId="084bc9f1-d850-4fc8-d64b-c787e273f9d0" colab={"base_uri": "https://localhost:8080/", "height": 33}
df[df["tags"] == "[none]"].shape

# + [markdown] id="Kl53OivLeka_" colab_type="text"
#
# ### Video_id
#
# Some `video_ids` seem corrupted:
# > #NAZWA?

# + id="-oU8cukPeLH4" colab_type="code" outputId="c206ae85-3aa7-48c0-a7d4-b07795836f12" colab={"base_uri": "https://localhost:8080/", "height": 154}
print(
    "Count #NAZWA?:",
    df[df["video_id"].apply(lambda x: any([not char.isalnum() and char not in "-_" for char in x]))]["video_id"].shape,
)
df[df["video_id"].apply(lambda x: any([not char.isalnum() and char not in "-_" for char in x]))][
    ["video_id", "title"]
].head(3)

# + id="ylAw5GTKEaaT" colab_type="code" outputId="9996ec14-4f7c-4ae3-b8c5-dfc583c99488" colab={"base_uri": "https://localhost:8080/", "height": 237}
df[df["description"].apply(lambda x: "\\n" in str(x))]["description"]

# + [markdown] id="gW6EAgaOTOhe" colab_type="text"
# ### Single video - different descriptions and titles ...
#
# Typical `video_id` contains alphanumeric characters, and `-` or `_`, and is 11 characters long.
#
# Grouping by `video_id`, and looking at columns of our interest there are some columns which values are different:
# - trending_date - as expected,
# - description - there are very slight differences between those (few characters diff), so it is definitely not enough to say that given the same title, the video is different. <u>It rather means that the user who uploaded the video, has decided to change a description a few days after the upload.</u>
# - tags - there are very rare cases, but also tags can vary!
# - title - unfortunately, in this column we can also observe changes.
#
# There is also `category_id` which yields two unique values when some of them are NaNs.
#
# After the analysis, we decided that the only column that can distinguish videos between themselves, aside from `video_id`, is `publish_time`.

# + id="koGb2t6cOYjA" colab_type="code" outputId="e131f452-0377-4e65-d514-6b536f60ad09" colab={"base_uri": "https://localhost:8080/", "height": 1000}
for example_video_id in df["video_id"].values[:100]:
    if "NAZWA" not in example_video_id:
        video_id_df = df[df["video_id"] == example_video_id]
        for cname in video_id_df.columns:
            if cname not in ["category_id", "views", "likes", "dislikes", "comment_count"]:
                count_unique = len(video_id_df[cname].unique())
                if count_unique > 1:
                    if cname == "title" or cname == "tags" or cname == "description":
                        print("\nnumber of unique '", cname, "': ", count_unique)
                        print(video_id_df[cname].unique())

# + [markdown] id="WmM9S738XqSh" colab_type="text"
# > We can replace "#NAZWA?" with manually-generated video_ids.

# + id="8CXJd0ztTGRC" colab_type="code" outputId="db9f4b2b-efdb-4b64-80e3-a24efa98049c" colab={"base_uri": "https://localhost:8080/", "height": 197}
corrupted_id_df = df[df["video_id"] == "#NAZWA?"]
for idx, t in enumerate(corrupted_id_df["publish_time"].unique()):
    corrupted_id_df.loc[corrupted_id_df["publish_time"] == t, "video_id"] = f"XXX{idx}"

df.loc[corrupted_id_df.index, :] = corrupted_id_df
df[df["video_id"].apply(lambda x: "XXX" in x)][["video_id", "title", "publish_time"]].head()

# + [markdown] id="gI1hhZuRZPUC" colab_type="text"
# Now with the missing values fixed, we can look at UNIQUE values per column.
# <!--
# There are repeating entries for the same video, but only with different `trending_date`, as already shown.
#
# > We can aggregate those videos into one row, and replace `trending_date` column with:
#
# > - count of different trending_dates
# > - list of exact trending_dates  -->
# ### Can one `video_id` have more than one title?
# > Answer: YES ...
#
# > This shows that user who uploaded a video is able to change its title while it's listed in TRENDING.
#

# + id="Jn_vUSJrbxO7" colab_type="code" outputId="b74dc3ff-6b21-4d83-df69-b8b18245e4a0" colab={"base_uri": "https://localhost:8080/", "height": 167}
df_by_video_id = df.groupby("video_id").agg({"title": lambda x: len(set(x))})
df_by_video_id.sort_values(by="title", ascending=False).head(3)

# + id="ltVrX64weytY" colab_type="code" outputId="a1563f4c-80f1-403d-df0a-de3c79efd1a5" colab={"base_uri": "https://localhost:8080/", "height": 167}
print(df[df["video_id"] == "w4SSZQDFuc8"].title.unique())
print(df[df["video_id"] == "sfMwXjNo3Rs"].title.unique())
print(df[df["video_id"] == "eVoXmDdI6Qg"].title.unique())

# + [markdown] id="gJuyyri1ESHj" colab_type="text"
# ### Analyze distribution of 'category_id'

# + id="D77HMXSWDtmi" colab_type="code" outputId="0e507376-f3d1-4f83-9a3a-d0cad6a1ceeb" colab={"base_uri": "https://localhost:8080/", "height": 564}
from collections import Counter
import numpy as np

categories = df.category_id.values
nans = categories[np.isnan(categories)]
categories = categories[~np.isnan(categories)]
print("NANs:", nans.shape, "not NANs:", categories.shape)

df.hist(column="category_id", bins=int(max(categories)))
Counter(categories.tolist()).most_common(100)

# + [markdown] id="fUeS6cNTkZFz" colab_type="text"
# ## Preview some categories examples
# - category **1** is trailers
# - category **2** is about cars and racing :P
# - category **10** is music videos
# - category **24** is ???

# + id="dh1wGieakbDv" colab_type="code" outputId="de910017-92ad-4a51-b085-4404eab5f540" colab={"base_uri": "https://localhost:8080/", "height": 200}
df[df["category_id"] == 24].head(10)["title"]

# + [markdown] id="ILzJC6_ayRj7" colab_type="text"
# # TEXT Attributes
# - title
# - channel title
# - publish time
# - tags
# - description

# + [markdown] id="9di0iCsJyo6x" colab_type="text"
# ## 1. Title
# Lengths in characters
# > On average, aroubd 50 characters describe the title

# + id="vcIFgnRUQxnx" colab_type="code" outputId="9d8ec612-0be9-4da7-810a-2dd576cedb2d" colab={"base_uri": "https://localhost:8080/", "height": 432}
import seaborn as sns

titles = df["title"].values

lengths = list(map(len, titles))
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)

# + id="UhdbYmaBZKa9" colab_type="code" outputId="6bc0013c-a1bc-46c3-ff55-719d6eb29c1c" colab={"base_uri": "https://localhost:8080/", "height": 50}
print("MAX length:", df.loc[df["title"].apply(len).idxmax(), :]["title"])
print("MIN length:", df.loc[df["title"].apply(len).idxmin(), :]["title"])

# + [markdown] id="cq93_rI8oVaA" colab_type="text"
# ### Non-ascii characters

# + id="eLmzFyXUmsOU" colab_type="code" outputId="127c43f7-6a85-433b-aaff-d2fce0604ee9" colab={"base_uri": "https://localhost:8080/", "height": 617}
count_chars = Counter("".join(titles))
print("Number of unique characters:", len(count_chars.keys()))
non_ascii = [key for key in count_chars if ord(key) > 127]
non_ascii_count = sorted([(key, count_chars[key]) for key in non_ascii], key=lambda x: -1 * x[1])
for pair in non_ascii_count[:35]:
    print(pair, ord(pair[0]))


# + id="hDzhW1cMqV5q" colab_type="code" outputId="8235b5f5-e5f9-42a8-b5e3-262ea9186acc" colab={"base_uri": "https://localhost:8080/", "height": 187}
from sklearn.cluster import KMeans

codes = np.array(list(map(ord, [key for key in count_chars.keys() if ord(key) > 127]))).reshape(-1, 1)

nc = 3
kmeans = KMeans(n_clusters=nc)
kmeans.fit(codes)
y_kmeans = kmeans.predict(codes)

chars_clusters = list(zip(map(chr, codes.reshape(-1)), y_kmeans))
for i in range(nc):
    print("\nCLUSTER #", i)
    print(list(filter(lambda x: x[1] == i, chars_clusters))[:20])


# + [markdown] id="0SqoOgSmlvsA" colab_type="text"
# Lengths in words
# > There are 10 *words* in average describing a video's title

# + id="P36KDA5xyiFL" colab_type="code" outputId="463fcdc8-3f75-4552-9c7e-a5d15481286e" colab={"base_uri": "https://localhost:8080/", "height": 432}
from nltk.tokenize import word_tokenize

lengths = []
for t in titles:
    lengths.append(len(word_tokenize(t)))

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)

# + [markdown] id="JAV-a2GwbRUE" colab_type="text"
# Look for titles without alphabetical characters at all.
# > Turns out there is one title repeating, `435`, with the following trending dates changing

# + id="bD6BaDAkarbr" colab_type="code" outputId="ab3eb9cf-3547-489d-b46d-a600e7569105" colab={"base_uri": "https://localhost:8080/", "height": 197}
df[df["title"].apply(lambda x: True if all([not char.isalpha() for char in x]) else False)].loc[
    :, ["video_id", "title", "channel_title", "trending_date"]
].head(5)

# + [markdown] id="8pfaz2Gqc_fH" colab_type="text"
# Look for titles that are all UPPERCASE

# + id="sx1LdUFpcxw3" colab_type="code" outputId="becb7f93-f1df-4845-fbf1-0bfa550aa72e" colab={"base_uri": "https://localhost:8080/", "height": 214}
upper = df[
    df["title"].apply(lambda x: True if all([char.isupper() or not char.isalpha() for char in x]) else False)
].loc[:, ["video_id", "title", "channel_title", "trending_date"]]
print(upper.shape)
upper.head(5)

# + id="anX1eqiWdE1U" colab_type="code" outputId="af26830a-481d-4fde-fd30-f112518450b2" colab={"base_uri": "https://localhost:8080/", "height": 347}
df["not_alpha_count"] = df["title"].apply(
    lambda x: sum([1 if not char.isalnum() and not char.isspace() else 0 for char in x])
)
df.sort_values(by="not_alpha_count", ascending=False)[["title", "not_alpha_count"]].head(10)

# + [markdown] id="9hhGDugcQZr8" colab_type="text"
# ### Most common words, without preprocessing:
# > One can observe that punctuation is one of the most frequent 'words'

# + id="A9EHQzapP6NG" colab_type="code" colab={}

# -

# ### Embeddings

# +
import io
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

tf.__version__


# -

#  encoder = info.features['text'].encoder
def write_embedding_files(labels, embedded_tensors, path=path):
    out_v = io.open(os.path.join(path, "vecs.tsv"), "w", encoding="utf-8")
    out_m = io.open(os.path.join(path, "meta.tsv"), "w", encoding="utf-8")
    vectors = embedded_tensors.numpy()
    for message, vector in zip(labels, vectors):
        out_m.write(message + "\n")
        out_v.write("\t".join([str(x) for x in vector]) + "\n")
    out_v.close()
    out_m.close()


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# +
messages = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding",
]

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
    print("Embedding[{},...]\n".format(message_embedding_snippet))
# -

unique_titles = np.unique(titles)
