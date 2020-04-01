```python
import nltk
nltk.download("punkt")

import os

path = "../data/"
files = os.listdir(path)
files
```

    [nltk_data] Downloading package punkt to /home/maaslak/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    ['US_category_id.json',
     '_vecs.tsv',
     'filename2text.json',
     'meta.tsv',
     'US_videos_5p.csv',
     'GB_category_id.json',
     'images.zip',
     'errors.txt',
     'title_meta.tsv',
     'vecs.tsv',
     'descriptions.csv',
     'channel_title_meta.tsv',
     'image_attributes.csv',
     'channel_title_vecs.tsv',
     'yolo_detections.json',
     'images',
     'title_vecs.tsv',
     'tags_meta.tsv',
     '_meta.tsv',
     'corrected_thumbnail.json',
     'thumbnail_ocr_vecs.tsv',
     'thumbnail_ocr_meta.tsv',
     'tags_vecs.tsv',
     'GB_videos_5p.csv',
     'errors copy.txt']




```python
import pandas as pd

pd.set_option("colwidth", None)

GB_videos_df = pd.read_csv(path + "/" + "GB_videos_5p.csv", sep=";", engine="python")
US_videos_df = pd.read_csv(path + "/" + "US_videos_5p.csv", sep=";", engine="python")

df = pd.concat([GB_videos_df, US_videos_df]).drop_duplicates().reset_index(drop=True)
df = df.rename(columns={"description ": "description"})
print(df.shape)
df.head(3) 
```

    (78255, 16)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>trending_date</th>
      <th>title</th>
      <th>channel_title</th>
      <th>category_id</th>
      <th>publish_time</th>
      <th>tags</th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>comment_count</th>
      <th>thumbnail_link</th>
      <th>comments_disabled</th>
      <th>ratings_disabled</th>
      <th>video_error_or_removed</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jw1Y-zhQURU</td>
      <td>17.14.11</td>
      <td>John Lewis Christmas Ad 2017 - #MozTheMonster</td>
      <td>John Lewis</td>
      <td>NaN</td>
      <td>2017-11-10T07:38:29.000Z</td>
      <td>christmas|"john lewis christmas"|"john lewis"|"christmas ad"|"mozthemonster"|"christmas 2017"|"christmas ad 2017"|"john lewis christmas advert"|"moz"</td>
      <td>7224515</td>
      <td>55681</td>
      <td>10247</td>
      <td>9479</td>
      <td>https://i.ytimg.com/vi/Jw1Y-zhQURU/default.jpg</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Click here to continue the story and make your own monster:\nhttp://bit.ly/2mboXgj\n\nJoe befriends a noisy Monster under his bed but the two have so much fun together that he can't get to sleep, leaving him tired by day. For Christmas Joe receives a gift to help him finally get a good night’s sleep.\n\nShop the ad\nhttp://bit.ly/2hg04Lc\n\nThe music is Golden Slumbers performed by elbow, the original song was by The Beatles. \nFind the track:\nhttps://Elbow.lnk.to/GoldenSlumbersXS\n\nSubscribe to this channel for regular video updates\nhttp://bit.ly/2eU8MvW\n\nIf you want to hear more from John Lewis:\n\nLike John Lewis on Facebook\nhttp://www.facebook.com/johnlewisretail\n\nFollow John Lewis on Twitter\nhttp://twitter.com/johnlewisretail\n\nFollow John Lewis on Instagram\nhttp://instagram.com/johnlewisretail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3s1rvMFUweQ</td>
      <td>17.14.11</td>
      <td>Taylor Swift: …Ready for It? (Live) - SNL</td>
      <td>Saturday Night Live</td>
      <td>NaN</td>
      <td>2017-11-12T06:24:44.000Z</td>
      <td>SNL|"Saturday Night Live"|"SNL Season 43"|"Episode 1730"|"Tiffany Haddish"|"Taylor Swift"|"Taylor Swift Ready for It"|"s43"|"s43e5"|"episode 5"|"live"|"new york"|"comedy"|"sketch"|"funny"|"hilarious"|"late night"|"host"|"music"|"guest"|"laugh"|"impersonation"|"actor"|"improv"|"musician"|"comedian"|"actress"|"If Loving You Is Wrong"|"Oprah Winfrey"|"OWN"|"Girls Trip"|"The Carmichael Show"|"Keanu"|"Reputation"|"Look What You Made Me Do"|"ready for it?"</td>
      <td>1053632</td>
      <td>25561</td>
      <td>2294</td>
      <td>2757</td>
      <td>https://i.ytimg.com/vi/3s1rvMFUweQ/default.jpg</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Musical guest Taylor Swift performs …Ready for It? on Saturday Night Live.\n\n#SNL #SNL43\n\nGet more SNL: http://www.nbc.com/saturday-night-live\nFull Episodes: http://www.nbc.com/saturday-night-liv...\n\nLike SNL: https://www.facebook.com/snl\nFollow SNL: https://twitter.com/nbcsnl\nSNL Tumblr: http://nbcsnl.tumblr.com/\nSNL Instagram: http://instagram.com/nbcsnl \nSNL Pinterest: http://www.pinterest.com/nbcsnl/</td>
    </tr>
    <tr>
      <th>2</th>
      <td>n1WpP7iowLc</td>
      <td>17.14.11</td>
      <td>Eminem - Walk On Water (Audio) ft. Beyoncé</td>
      <td>EminemVEVO</td>
      <td>NaN</td>
      <td>2017-11-10T17:00:03.000Z</td>
      <td>Eminem|"Walk"|"On"|"Water"|"Aftermath/Shady/Interscope"|"Rap"</td>
      <td>17158579</td>
      <td>787420</td>
      <td>43420</td>
      <td>125882</td>
      <td>https://i.ytimg.com/vi/n1WpP7iowLc/default.jpg</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Eminem's new track Walk on Water ft. Beyoncé is available everywhere: http://shady.sr/WOWEminem \nPlaylist Best of Eminem: https://goo.gl/AquNpo\nSubscribe for more: https://goo.gl/DxCrDV\n\nFor more visit: \nhttp://eminem.com\nhttp://facebook.com/eminem\nhttp://twitter.com/eminem\nhttp://instagram.com/eminem\nhttp://eminem.tumblr.com\nhttp://shadyrecords.com\nhttp://facebook.com/shadyrecords\nhttp://twitter.com/shadyrecords\nhttp://instagram.com/shadyrecords\nhttp://trustshady.tumblr.com\n\nMusic video by Eminem performing Walk On Water. (C) 2017 Aftermath Records\nhttp://vevo.ly/gA7xKt</td>
    </tr>
  </tbody>
</table>
</div>



## Unwanted attributes
- We do not need to analyze **views**, **likes**, **dislikes** or **comment_count** as we cannot base the trending guidelines upon such statistics

## Check for **missing values**
Apart from category_id column about which we already know it has values missing, there are other attributes with missing data.

### Description


```python
missing_values_df = df.drop(["category_id"], axis=1)
missing_values_df = missing_values_df[missing_values_df.isnull().any(axis=1)]

for cname in missing_values_df.columns:
    check_nulls = missing_values_df[[cname]].isnull().sum().values[0]
    if check_nulls > 0:
        print("Missing values in column", cname, ":", check_nulls)
```

    Missing values in column description : 1098


There are NaNs in column `description`.

**Solution**: Replace `NaN`s with "no description"


```python
df.loc[df["description"].isna(), "description"] = "no description"
```

### Tags
We can also observe that there can be missing tags, represented as `[none]`. We leave it as it is as no tags is also some kind of an information.


```python
df[df["tags"] == "[none]"].shape
```




    (3449, 16)




### Video_id

Some `video_ids` seem corrupted:
> #NAZWA?


```python
print(
    "Count #NAZWA?:",
    df[df["video_id"].apply(lambda x: any([not char.isalnum() and char not in "-_" for char in x]))]["video_id"].shape,
)
df[df["video_id"].apply(lambda x: any([not char.isalnum() and char not in "-_" for char in x]))][
    ["video_id", "title"]
].head(3)
```

    Count #NAZWA?: (721,)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>#NAZWA?</td>
      <td>CHRISTMAS HAS GONE TO MY HEAD</td>
    </tr>
    <tr>
      <th>179</th>
      <td>#NAZWA?</td>
      <td>Don Diablo ft. A R I Z O N A - Take Her Place (Official Lyric Video)</td>
    </tr>
    <tr>
      <th>198</th>
      <td>#NAZWA?</td>
      <td>Wendy Williams Faints On Live TV</td>
    </tr>
  </tbody>
</table>
</div>



### Single video - different descriptions and titles ...

Typical `video_id` contains alphanumeric characters, and `-` or `_`, and is 11 characters long.

Grouping by `video_id`, and looking at columns of our interest there are some columns which values are different:
- trending_date - as expected,
- description - there are very slight differences between those (few characters diff), so it is definitely not enough to say that given the same title, the video is different. <u>It rather means that the user who uploaded the video, has decided to change a description a few days after the upload.</u>
- tags - there are very rare cases, but also tags can vary!
- title - unfortunately, in this column we can also observe changes.

There is also `category_id` which yields two unique values when some of them are NaNs.

After the analysis, we decided that the only column that can distinguish videos between themselves, aside from `video_id`, is `publish_time`.


```python
for example_video_id in df["video_id"].values[:5]:
    if "NAZWA" not in example_video_id:
        video_id_df = df[df["video_id"] == example_video_id]
        for cname in video_id_df.columns:
            if cname not in ["category_id", "views", "likes", "dislikes", "comment_count"]:
                count_unique = len(video_id_df[cname].unique())
                if count_unique > 1:
                    if cname == "title" or cname == "tags" or cname == "description":
                        print("\nnumber of unique '", cname, "': ", count_unique, '\n')
                        print(video_id_df[cname].unique())
```

    
    number of unique ' description ':  2 
    
    ["Click here to continue the story and make your own monster:\\nhttp://bit.ly/2mboXgj\\n\\nJoe befriends a noisy Monster under his bed but the two have so much fun together that he can't get to sleep, leaving him tired by day. For Christmas Joe receives a gift to help him finally get a good night’s sleep.\\n\\nShop the ad\\nhttp://bit.ly/2hg04Lc\\n\\nThe music is Golden Slumbers performed by elbow, the original song was by The Beatles. \\nFind the track:\\nhttps://Elbow.lnk.to/GoldenSlumbersXS\\n\\nSubscribe to this channel for regular video updates\\nhttp://bit.ly/2eU8MvW\\n\\nIf you want to hear more from John Lewis:\\n\\nLike John Lewis on Facebook\\nhttp://www.facebook.com/johnlewisretail\\n\\nFollow John Lewis on Twitter\\nhttp://twitter.com/johnlewisretail\\n\\nFollow John Lewis on Instagram\\nhttp://instagram.com/johnlewisretail"
     "Click here to continue the story and make your own monster: http://bit.ly/2mboXgj Joe befriends a noisy Monster under his bed but the two have so much fun together that he can't get to sleep, leaving him tired by day. For Christmas Joe receives a gift to help him finally get a good night’s sleep. Shop the ad http://bit.ly/2hg04Lc The music is Golden Slumbers performed by elbow, the original song was by The Beatles.  Find the track: https://Elbow.lnk.to/GoldenSlumbersXS Subscribe to this channel for regular video updates http://bit.ly/2eU8MvW If you want to hear more from John Lewis: Like John Lewis on Facebook http://www.facebook.com/johnlewisretail Follow John Lewis on Twitter http://twitter.com/johnlewisretail Follow John Lewis on Instagram http://instagram.com/johnlewisretail"]
    
    number of unique ' description ':  3 
    
    ["Eminem's new track Walk on Water ft. Beyoncé is available everywhere: http://shady.sr/WOWEminem \\nPlaylist Best of Eminem: https://goo.gl/AquNpo\\nSubscribe for more: https://goo.gl/DxCrDV\\n\\nFor more visit: \\nhttp://eminem.com\\nhttp://facebook.com/eminem\\nhttp://twitter.com/eminem\\nhttp://instagram.com/eminem\\nhttp://eminem.tumblr.com\\nhttp://shadyrecords.com\\nhttp://facebook.com/shadyrecords\\nhttp://twitter.com/shadyrecords\\nhttp://instagram.com/shadyrecords\\nhttp://trustshady.tumblr.com\\n\\nMusic video by Eminem performing Walk On Water. (C) 2017 Aftermath Records\\nhttp://vevo.ly/gA7xKt"
     "Eminem's new track Walk on Water ft. Beyoncé is available everywhere: http://shady.sr/WOWEminem \\n\\nFor more visit: \\nhttp://eminem.com\\nhttp://facebook.com/eminem\\nhttp://twitter.com/eminem\\nhttp://instagram.com/eminem\\nhttp://eminem.tumblr.com\\nhttp://shadyrecords.com\\nhttp://facebook.com/shadyrecords\\nhttp://twitter.com/shadyrecords\\nhttp://instagram.com/shadyrecords\\nhttp://trustshady.tumblr.com\\n\\nMusic video by Eminem performing Walk On Water. (C) 2017 Aftermath Records\\nhttp://vevo.ly/gA7xKt"
     "Eminem's new track Walk on Water ft. Beyoncé is available everywhere: http://shady.sr/WOWEminem  Playlist Best of Eminem: https://goo.gl/AquNpo Subscribe for more: https://goo.gl/DxCrDV For more visit:  http://eminem.com http://facebook.com/eminem http://twitter.com/eminem http://instagram.com/eminem http://eminem.tumblr.com http://shadyrecords.com http://facebook.com/shadyrecords http://twitter.com/shadyrecords http://instagram.com/shadyrecords http://trustshady.tumblr.com Music video by Eminem performing Walk On Water. (C) 2017 Aftermath Records http://vevo.ly/gA7xKt"]
    
    number of unique ' description ':  2 
    
    ['Salford drew 4-4 against the Class of 92 and Friends at the newly opened The Peninsula Stadium!\\n\\nLike us on Facebook: https://www.facebook.com/SalfordCityFC/ \\nFollow us on Twitter: https://twitter.com/SalfordCityFC\\nFollow us on Instagram: https://www.instagram.com/salfordcityfc/ \\nSubscribe to us on YouTube: https://www.youtube.com/user/SalfordCityTV\\n\\nWebsite: https://salfordcityfc.co.uk'
     'Salford drew 4-4 against the Class of 92 and Friends at the newly opened The Peninsula Stadium!\\n\\nThanks to University of Salford students Andy Bellamy, Zoe Worthington and Hugh Nelson for the filming the replay angles. \\n\\nLike us on Facebook: https://www.facebook.com/SalfordCityFC/ \\nFollow us on Twitter: https://twitter.com/SalfordCityFC\\nFollow us on Instagram: https://www.instagram.com/salfordcityfc/ \\nSubscribe to us on YouTube: https://www.youtube.com/user/SalfordCityTV\\n\\nWebsite: https://salfordcityfc.co.uk']


> We can replace "#NAZWA?" with manually-generated video_ids.


```python
corrupted_id_df = df[df["video_id"] == "#NAZWA?"]
for idx, t in enumerate(corrupted_id_df["publish_time"].unique()):
    corrupted_id_df.loc[corrupted_id_df["publish_time"] == t, "video_id"] = f"XXX{idx}"

df.loc[corrupted_id_df.index, :] = corrupted_id_df

df[df["video_id"].apply(lambda x: "XXX" in x)][["video_id", "title", "publish_time"]].head()
```

    /home/maaslak/PycharmProjects/ped/venv/lib/python3.8/site-packages/pandas/core/indexing.py:966: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>title</th>
      <th>publish_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>XXX0</td>
      <td>CHRISTMAS HAS GONE TO MY HEAD</td>
      <td>2017-11-10T19:19:43.000Z</td>
    </tr>
    <tr>
      <th>179</th>
      <td>XXX1</td>
      <td>Don Diablo ft. A R I Z O N A - Take Her Place (Official Lyric Video)</td>
      <td>2017-11-02T16:34:20.000Z</td>
    </tr>
    <tr>
      <th>198</th>
      <td>XXX2</td>
      <td>Wendy Williams Faints On Live TV</td>
      <td>2017-10-31T15:59:58.000Z</td>
    </tr>
    <tr>
      <th>383</th>
      <td>XXX1</td>
      <td>Don Diablo ft. A R I Z O N A - Take Her Place (Official Lyric Video)</td>
      <td>2017-11-02T16:34:20.000Z</td>
    </tr>
    <tr>
      <th>593</th>
      <td>XXX1</td>
      <td>Don Diablo ft. A R I Z O N A - Take Her Place (Official Lyric Video)</td>
      <td>2017-11-02T16:34:20.000Z</td>
    </tr>
  </tbody>
</table>
</div>



Now with the missing values fixed, we can look at UNIQUE values per column.
<!--
There are repeating entries for the same video, but only with different `trending_date`, as already shown.

> We can aggregate those videos into one row, and replace `trending_date` column with:

> - count of different trending_dates
> - list of exact trending_dates  -->
### Can one `video_id` have more than one title?
> Answer: YES ...

> This shows that user who uploaded a video is able to change its title while it's listed in TRENDING.



```python
df_by_video_id = df.groupby("video_id").agg({"title": lambda x: len(set(x))})
df_by_video_id.sort_values(by="title", ascending=False).head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
    </tr>
    <tr>
      <th>video_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>w4SSZQDFuc8</th>
      <td>3</td>
    </tr>
    <tr>
      <th>sfMwXjNo3Rs</th>
      <td>3</td>
    </tr>
    <tr>
      <th>eVoXmDdI6Qg</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df[df["video_id"] == "w4SSZQDFuc8"].title.unique())
print(df[df["video_id"] == "sfMwXjNo3Rs"].title.unique())
print(df[df["video_id"] == "eVoXmDdI6Qg"].title.unique())
```

    ['Ghost Adventures S15E17 - The Slaughter House [HD 720p]'
     'Ghost Adventures S15E17 - The Slaughter House'
     'Ghost Adventures S15E17 - The Slaughterhouse [HD 720p]']
    ['How Spring Looks Like around the World'
     'What Spring Looks Like around the World'
     'What Spring Looks Like Around The World']
    ['5000 IQ IS REQUIRED TO WATCH'
     '5000 IQ IS REQUIRED TO WATCH /r/iamverysmart/ #4'
     '5000 IQ IS REQUIRED TO WATCH /r/iamverysmart/ #4 [REDDIT REVIEW]']


### Analyze distribution of 'category_id'


```python
from collections import Counter
import numpy as np

categories = df.category_id.values
nans = categories[np.isnan(categories)]
categories = categories[~np.isnan(categories)]
print("NANs:", nans.shape, "not NANs:", categories.shape)

df.hist(column="category_id", bins=int(max(categories)))
Counter(categories.tolist()).most_common()
```

    NANs: (74335,) not NANs: (3920,)





    [(10.0, 946),
     (24.0, 923),
     (22.0, 321),
     (23.0, 292),
     (26.0, 274),
     (1.0, 249),
     (17.0, 243),
     (25.0, 192),
     (28.0, 134),
     (20.0, 121),
     (27.0, 99),
     (15.0, 74),
     (19.0, 24),
     (2.0, 20),
     (29.0, 6),
     (43.0, 2)]




![png](output_19_2.png)


## Preview some categories examples
- category **1** is trailers
- category **2** is about cars and racing :P
- category **10** is music videos
- ...


```python
df[df["category_id"] == 24].head(10)["title"]
```




    76                                              How Are Waves Formed? - Earth Lab
    181                                                HALLOWEEN AFTERMATH AND Q&A 👻🎃
    199            Drop the Mic: Rob Gronkowski vs Gina Rodriguez - FULL BATTLE | TBS
    316    Selena Gomez Singing Wolves and It Ain't Me FULL Instagram Live | Acoustic
    424                                     Taylor Swift: …Ready for It? (Live) - SNL
    457                               Taylor Swift - “New Year’s Day” Fan Performance
    645                                                         How to be an Aquarius
    786                                  [Mnet Present Special] SEVENTEEN - CHANGE UP
    809                                          73 Questions With Liza Koshy | Vogue
    826                                                 Justice League - Movie Review
    Name: title, dtype: object



# TEXT Attributes
- publish time
- title
- channel title
- tags
- description

## 1. Publish Time
A) Years of publish



```python
import dateutil.parser

dates = [dateutil.parser.isoparse(d) for d in df["publish_time"].unique()]
years = [d.year for d in dates]
count_years = Counter(years)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_years.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])
ax.set_xticks([it[0] for it in its])
ax.set_title("Years of publish_time")
```




    Text(0.5, 1.0, 'Years of publish_time')




![png](output_24_1.png)


*B*) Month of publish
> We can see that during some months, there have been much less trending videos than during other ones. In particular, months July to October (inclusive) are very rare.



```python
import datetime 

months = [d.month for d in dates]
count_months = Counter(months)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_months.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([datetime.date(1900, it[0], 1).strftime('%B') for it in its], rotation=30)
ax.set_title("Months of publish_time")
```




    Text(0.5, 1.0, 'Months of publish_time')




![png](output_26_1.png)


C) Weekday
> Conversely to what we expected, most trending videos have not been published on weekend


```python
days = [d.weekday() for d in dates]
count_days = Counter(days)
weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

fig, ax = plt.subplots(figsize=(8, 6))
its =  count_days.items()
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([weekDays[it[0]] for it in its], rotation=30)
ax.set_title("WeekDays of publish_time")
```




    Text(0.5, 1.0, 'WeekDays of publish_time')




![png](output_28_1.png)



```python
# ADD week_day attribute
df["week_day"] = df["publish_time"].apply(lambda x : dateutil.parser.isoparse(x).weekday())
```

D) Hour
> There is a significant increase in trending videos in the middle of the day, between 13:00 and 19:00

> Hours can be divided into four periods in a day:
- 00:00 - 06:00 (time_of_day = 1)
- 06:00 - 12:00 (time_of_day = 2)
- 12:00 - 18:00 (time_of_day = 3)
- 18:00 - 24:00 (time_of_day = 4)

Let's make an attribute out of that.


```python
hours = [d.hour for d in dates]
count_hours = Counter(hours)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_hours.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([f"{it[0]:02d}:00" for it in its], rotation=40)
ax.set_title("Hours of publish_time")
```




    Text(0.5, 1.0, 'Hours of publish_time')




![png](output_31_1.png)



```python
def extract_time_of_day(datestring):
  d = dateutil.parser.isoparse(datestring)
  return d.hour // 6 + 1

# ADD time_of_day attribute
df["time_of_day"] = df["publish_time"].apply(extract_time_of_day)
df[["time_of_day", "publish_time"]].head()

# ## 1. Publish Time
# A) Years of publish
#
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_of_day</th>
      <th>publish_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2017-11-10T07:38:29.000Z</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2017-11-12T06:24:44.000Z</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2017-11-10T17:00:03.000Z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2017-11-13T02:30:38.000Z</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2017-11-13T01:45:13.000Z</td>
    </tr>
  </tbody>
</table>
</div>




```python
import dateutil.parser

dates = [dateutil.parser.isoparse(d) for d in df["publish_time"].unique()]
years = [d.year for d in dates]
count_years = Counter(years)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_years.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])
ax.set_xticks([it[0] for it in its])
ax.set_title("Years of publish_time")
```




    Text(0.5, 1.0, 'Years of publish_time')




![png](output_33_1.png)


*B*) Month of publish
> We can see that during some months, there have been much less trending videos than during other ones. In particular, months July to October (inclusive) are very rare.



```python
import datetime 

months = [d.month for d in dates]
count_months = Counter(months)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_months.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([datetime.date(1900, it[0], 1).strftime('%B') for it in its], rotation=30)
ax.set_title("Months of publish_time")
```




    Text(0.5, 1.0, 'Months of publish_time')




![png](output_35_1.png)



```python
# ADD month attribute
df["month"] = df["publish_time"].apply(lambda x : dateutil.parser.isoparse(x).month)
```

C) Weekday
> Conversely to what we expected, most trending videos have not been published on weekend


```python
days = [d.weekday() for d in dates]
count_days = Counter(days)
weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

fig, ax = plt.subplots(figsize=(8, 6))
its =  count_days.items()
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([weekDays[it[0]] for it in its], rotation=30)
ax.set_title("WeekDays of publish_time")
```




    Text(0.5, 1.0, 'WeekDays of publish_time')




![png](output_38_1.png)



```python
# ADD week_day attribute
df["week_day"] = df["publish_time"].apply(lambda x : dateutil.parser.isoparse(x).weekday())
```

D) Hour
> There is a significant increase in trending videos in the middle of the day, between 13:00 and 19:00

> Hours can be divided into four periods in a day:
- 00:00 - 06:00 (time_of_day = 1)
- 06:00 - 12:00 (time_of_day = 2)
- 12:00 - 18:00 (time_of_day = 3)
- 18:00 - 24:00 (time_of_day = 4)

Let's make an attribute out of that.


```python
hours = [d.hour for d in dates]
count_hours = Counter(hours)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_hours.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([f"{it[0]:02d}:00" for it in its], rotation=40)
ax.set_title("Hours of publish_time")
```




    Text(0.5, 1.0, 'Hours of publish_time')




![png](output_41_1.png)



```python
def extract_time_of_day(datestring):
  d = dateutil.parser.isoparse(datestring)
  return d.hour // 6 + 1

# ADD time_of_day attribute
df["time_of_day"] = df["publish_time"].apply(extract_time_of_day)
df[["time_of_day", "publish_time"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_of_day</th>
      <th>publish_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2017-11-10T07:38:29.000Z</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2017-11-12T06:24:44.000Z</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2017-11-10T17:00:03.000Z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2017-11-13T02:30:38.000Z</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2017-11-13T01:45:13.000Z</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Title

**Lengths in characters**
> On average, around 50 characters describe the title.

> It seems as if there was normal distribution of title length in characters, with a right tail slightly longer.


```python
import seaborn as sns

titles = df["title"].values
lengths = list(map(len, titles))

# ADD title_length_chars attribute
df["title_length_chars"] = df["title"].apply(len)

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
```

           length_statistics
    count       78255.000000
    mean           49.059587
    std            19.742279
    min             3.000000
    25%            34.000000
    50%            47.000000
    75%            61.000000
    max           100.000000





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd8803b0040>




![png](output_44_2.png)



```python
print("MAX length:", df.loc[df["title"].apply(len).idxmax(), :]["title"])
print()
print("MIN length:", df.loc[df["title"].apply(len).idxmin(), :]["title"])
```

    MAX length: President Trump and First Lady Melania Trump participates in NORAD Santa Tracker phone calls. Dec 24
    
    MIN length: 435


**Lengths in words (*tokens*)**
> There are 10 *words* in average describing a video's title.

> There exist titles with only one word, and titles with a maximum of 28 words. 

> The distirbution seems normal, but is not very smooth.


```python
from nltk.tokenize import word_tokenize

lengths = []
for t in titles:
    lengths.append(len(word_tokenize(t)))

# ADD title_length_tokens attribute
df["title_length_tokens"] = df["title"].apply(lambda x : len(word_tokenize(x)))
    
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
```

           length_statistics
    count       78255.000000
    mean            9.999962
    std             4.192119
    min             1.000000
    25%             7.000000
    50%            10.000000
    75%            13.000000
    max            27.000000





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd8805f3e20>




![png](output_47_2.png)


### Upper vs. Lower case
> We can observe a dominating ratio between uppercase letters and overall length of title: most of them have 20% uppercase characters. Right tail shows that higher ratios appear, but not as often as we could expect.


```python
def get_uppercase_ratio(x):
    return sum([1 if char.isalpha() and char.isupper() else 0 for char in x]) / len(x)

# ADD title_uppercase_ratio attribute
df["title_uppercase_ratio"] = df["title"].apply(get_uppercase_ratio)

uppercase_ratio = df["title_uppercase_ratio"]
print(uppercase_ratio.describe())
sns.distplot(uppercase_ratio)
```

    count    78255.000000
    mean         0.220660
    std          0.181672
    min          0.000000
    25%          0.130435
    50%          0.157895
    75%          0.215686
    max          0.947368
    Name: title_uppercase_ratio, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87f1b4f70>




![png](output_49_2.png)


### Non-alphanumeric characters
> Similarily to uppercase ratio, there is a trend in titles that 20% of the characters describing it are neither letters nor digits, but other special characters.


```python
def get_not_alnum_ratio(x):
    return sum([1 if not char.isalnum() else 0 for char in x]) / len(x)


# ADD title_uppercase_ratio attribute
df["title_not_alnum_ratio"] = df["title"].apply(get_not_alnum_ratio)

not_alnum_ratio = df["title_not_alnum_ratio"]
print(not_alnum_ratio.describe())
sns.distplot(not_alnum_ratio)
```

    count    78255.000000
    mean         0.201883
    std          0.048081
    min          0.000000
    25%          0.171429
    50%          0.200000
    75%          0.231884
    max          0.500000
    Name: title_not_alnum_ratio, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87ebca310>




![png](output_51_2.png)


### Top not-alphanumeric characters
> For each often occurring character, we check in how many titles it was observed. Large percentages could suggest one should use this character in the title :)

> We can create binary features which will tell for each title, whether it contains this particular character, or not. However, we don't want our feature vector to grow too much at this point, so instead of binary encoding for each common character, we will count all characters that belong to TOP-N characters. TOP-N will be derived by applying threshold: <u>more than 10% of titles</u>. Furthermore, we skip whitespace characters and assume that number of tokens will reflect this feature well enough.


```python
count_chars = Counter("".join(titles))
print("Number of different characters:", len(count_chars.keys()))

top_chars = count_chars.most_common()
not_alnum = [t for t in top_chars if not t[0].isalnum()]
top_not_alnum = not_alnum[:15]

common_chars = []
for char, count in not_alnum:
    percentage = sum(df["title"].apply(lambda x : char in x)) / df.shape[0] * 100.0
    if percentage > 10.0:
        print(f"'{char}': {count}", ",", round(percentage, 3), "%")
        if not char.isspace():
            common_chars.append(char)
```

    Number of different characters: 734
    ' ': 598207 , 99.672 %
    '-': 33210 , 37.908 %
    '|': 19711 , 19.811 %
    '(': 19213 , 22.736 %
    ')': 19191 , 22.687 %
    '.': 14383 , 13.439 %
    ''': 14304 , 13.953 %
    ':': 9295 , 11.394 %



```python
# ADD title_common_chars_count
df["title_common_chars_count"] = df["title"].apply(lambda x : sum(1 if char in common_chars else 0 for char in x))
sns.distplot(df["title_common_chars_count"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87d576a00>




![png](output_54_1.png)


## 3. Channel title
> For Channel title, we follow very similar analysis to **Title** analysis 

> Channel title is usually less than 20 characters long


```python
import seaborn as sns

titles = df["channel_title"].values
lengths = list(map(len, titles))

# ADD title_length_chars attribute
df["channel_title_length_chars"] = df["channel_title"].apply(len)

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
```

           length_statistics
    count       78255.000000
    mean           12.944272
    std             6.300970
    min             1.000000
    25%             9.000000
    50%            12.000000
    75%            16.000000
    max            49.000000





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87e7668e0>




![png](output_56_2.png)


> Most Channel titles are one or two words long.


```python
from nltk.tokenize import word_tokenize

lengths = []
for t in titles:
    lengths.append(len(word_tokenize(t)))

# ADD title_length_tokens attribute
df["channel_title_length_tokens"] = df["channel_title"].apply(lambda x : len(word_tokenize(x)))
    
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
```

           length_statistics
    count       78255.000000
    mean            1.897400
    std             1.159166
    min             1.000000
    25%             1.000000
    50%             2.000000
    75%             2.000000
    max             9.000000





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd8807c4b20>




![png](output_58_2.png)


-

## 4. Tags

> For Tags, we simply apply **counting**. If **Tags** are `[none]`, we use $-1$ to denote this special value.
> Looking at the distribution of tags counts, we can tell that there is no simple relation such as: the more tags the better.


```python
df["tags_count"] = df["tags"].apply(lambda x : x.count('|') if '|' in x else -1)
sns.distplot(df["tags_count"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87e6d8160>




![png](output_61_1.png)


#### Are there any popular tags?
We apply `unique` in order to find popular tags,as if there was a video which was trending for many days, its tags are recorded multiple times. This would distort the distribution of tags across all the videos.

> Some tags are more common than other ones, however most of them are connected to some fixed category, for example `trailer` - implies that the video content will be a movie trailer.


```python
all_tags_list = df["tags"].apply(lambda x : [tag.lower().replace('"', '') for tag in x.split('|')]).values
print(len(all_tags_list))
all_tags_list = np.unique(all_tags_list)
print(len(all_tags_list))

all_tags = []
all_tags_tokens = []
for atl in all_tags_list:
    all_tags.extend(atl)
    for tag in atl:
        all_tags_tokens.extend(tag.split())
    
print("\nPOPULAR TAGS:")
print(Counter(all_tags).most_common(30))
```

    78255
    8185
    
    POPULAR TAGS:
    [('funny', 917), ('comedy', 788), ('music', 439), ('interview', 379), ('trailer', 322), ('humor', 316), ('video', 315), ('news', 290), ('television', 287), ('2018', 276), ('celebrity', 275), ('comedian', 262), ('talk show', 261), ('pop', 255), ('live', 255), ('show', 239), ('review', 235), ('nbc', 234), ('how to', 233), ('clip', 230), ('celebrities', 225), ('late night', 225), ('entertainment', 212), ('food', 211), ('science', 210), ('sports', 209), ('funny video', 207), ('movie', 202), ('vlog', 200), ('comedic', 199)]



```python
print("POPULAR TAGS TOKENS:")
print(Counter(all_tags_tokens).most_common(30))
```

    POPULAR TAGS TOKENS:
    [('the', 4560), ('video', 2021), ('show', 1685), ('to', 1666), ('music', 1615), ('new', 1569), ('funny', 1560), ('news', 1501), ('of', 1376), ('2018', 1301), ('and', 1184), ('makeup', 1168), ('how', 1127), ('comedy', 1072), ('live', 1069), ('a', 1014), ('trailer', 955), ('in', 838), ('movie', 810), ('star', 782), ('2017', 779), ('interview', 755), ('ellen', 738), ('food', 726), ('youtube', 722), ('best', 708), ('official', 704), ('late', 665), ('game', 649), ('you', 626)]


## 5. Description
**Lengths in characters**

> Average length of description in characters is close to one thousand. However, the standard deviation is also very high, very close to the mean - roughly 845 characters. Median of the distribution is equal to 733 characters.

> Almost no-descriptions also happen: minimum value observed is 1.


```python
descriptions = df["description"].values
lengths = list(map(len, descriptions))

# ADD title_length_chars attribute
df["description_length_chars"] = df["description"].apply(len)

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
```

           length_statistics
    count       78255.000000
    mean          950.783963
    std           826.734702
    min             1.000000
    25%           373.000000
    50%           720.000000
    75%          1291.000000
    max          5260.000000





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87f3a0a90>




![png](output_66_2.png)


**Lengths in tokens**
> On average, there are 156 words (tokens) per description. Maximum observation was 1520, which is a great outlier as the 75th percentile is equal to 212 words. 


```python
lengths = []
for d in descriptions:
    lengths.append(len(word_tokenize(d)))

# ADD title_length_tokens attribute
df["description_length_tokens"] = df["description"].apply(lambda x : len(word_tokenize(x)))
    
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
```

           length_statistics
    count       78255.000000
    mean          156.114536
    std           144.472724
    min             0.000000
    25%            54.000000
    50%           112.000000
    75%           212.000000
    max          1047.000000





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87e6fef10>




![png](output_68_2.png)


**Lengths in lines**
> For description, we decided also to count newlines. This is because youtubers often format description in some way, for example enuemrating links to their social medias in new lines each.

> It turned out that many descriptions have no new line at all. We suspect that sometimes there could have been an error when exporting the data.


```python
def count_newlines(x):
    x = x.replace(r'\\n', r'\n')
    return x.count(r'\n')

# ADD title_length_tokens attribute
df["description_length_newlines"] = df["description"].apply(count_newlines)
    
print(df["description_length_newlines"].describe())
sns.distplot(df["description_length_newlines"])
```

    count    78255.000000
    mean         8.093131
    std         18.830621
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%         10.000000
    max        269.000000
    Name: description_length_newlines, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd875f47520>




![png](output_70_2.png)


**Upper vs. Lower case (again)**
> We take a look onto uppercase letters again, followed by what we discovered in titles.

> For the description attribute, the tendency towards using uppercase letters is smaller. Most ratios don't exceed 10%.


```python
# ADD title_uppercase_ratio attribute
df["description_uppercase_ratio"] = df["description"].apply(get_uppercase_ratio)

uppercase_ratio = df["description_uppercase_ratio"]
print(uppercase_ratio.describe())
sns.distplot(uppercase_ratio)
```

    count    78255.000000
    mean         0.094871
    std          0.061965
    min          0.000000
    25%          0.057082
    50%          0.083333
    75%          0.117503
    max          0.888889
    Name: description_uppercase_ratio, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87eed5700>




![png](output_72_2.png)


**Number of URLs per description**
> We observed that users often leave many links in their descriptions, such as: instagram profile, twitter profile, subscribe redirection url, or event download links. It may be useful to keep track on how many URLs each description has.

> Although some descriptions lack URLs, the median count of URLs is equal to 7 - quite a lot - and the maximum observation is 83 URLs!


```python
import re

best_url_regex = r'(http:\/\/www\.|http:\/\/|https:\/\/www\.|https:\/\/|www\.)?(?P<domain>([a-zA-Z0-9]+\.)+[a-zA-Z0-9]+)\/*[a-zA-Z0-9-_\/\.]*'
df["description_url_count"] = df["description"].apply(lambda x : len(re.findall(best_url_regex, x)))

print(df["description_url_count"].describe())
sns.distplot(df["description_url_count"])
```

    count    78255.000000
    mean         8.139697
    std          8.011052
    min          0.000000
    25%          2.000000
    50%          7.000000
    75%         11.000000
    max         83.000000
    Name: description_url_count, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x7fd87e6adb20>




![png](output_74_2.png)


#### Which URLs are popular?


```python
descriptions = df["description"].unique()

domains = []
for d in descriptions:
    matches = re.findall(best_url_regex, d)
    for m in matches:
        if len(m) > 1:
            domains.append(m[1])

top_domains = Counter(domains).most_common(30)
print(top_domains)
```

    [('bit.ly', 13274), ('youtube.com', 8348), ('twitter.com', 6480), ('instagram.com', 6459), ('facebook.com', 5482), ('goo.gl', 3684), ('smarturl.it', 2787), ('youtu.be', 2778), ('amzn.to', 2069), ('po.st', 708), ('soundcloud.com', 707), ('rstyle.me', 641), ('Twitter.com', 639), ('Facebook.com', 623), ('patreon.com', 620), ('plus.google.com', 526), ('es.pn', 525), ('vevo.ly', 511), ('snapchat.com', 377), ('gmail.com', 315), ('nfl.com', 310), ('apple.co', 304), ('pinterest.com', 303), ('buzzfeed.com', 257), ('on.fb.me', 254), ('twitch.tv', 241), ('nbc.com', 211), ('itunes.apple.com', 207), ('amazon.com', 200), ('spoti.fi', 196)]


#### For the most popular URLs, in how many unique descriptions (in %) do they appear?
> Some URLs appear very often:
- twitter.com
- facebook.com
- instagram.com

We can hypothesise that trending videos include links to popular social medias. It is correlated with the fact that if some youtuber is popular, then she/he already has accounts on many other social medias as well, to reach out to more fans.


```python
percentages = []
for domain, count in top_domains:
    percentage = sum(1 if domain in d else 0 for d in descriptions) / len(descriptions) * 100.0
    percentages.append((domain, percentage))

for domain, count in sorted(percentages, key=lambda x : -x[1]):
    print(f"{domain} : {count:.3f}%")
```

    twitter.com : 49.307%
    facebook.com : 46.035%
    instagram.com : 44.265%
    youtube.com : 33.766%
    bit.ly : 33.093%
    goo.gl : 11.057%
    youtu.be : 9.623%
    smarturl.it : 6.563%
    soundcloud.com : 5.581%
    patreon.com : 5.572%
    vevo.ly : 4.860%
    plus.google.com : 4.455%
    amzn.to : 4.215%
    apple.co : 3.666%
    snapchat.com : 3.503%
    Twitter.com : 3.483%
    Facebook.com : 3.358%
    pinterest.com : 3.012%
    gmail.com : 2.993%
    on.fb.me : 2.444%
    itunes.apple.com : 1.896%
    amazon.com : 1.722%
    twitch.tv : 1.684%
    spoti.fi : 1.655%
    nbc.com : 1.549%
    buzzfeed.com : 1.078%
    es.pn : 1.010%
    po.st : 0.991%
    rstyle.me : 0.423%
    nfl.com : 0.375%


### Non-ascii characters
# `TO DELETE?`


```python
count_chars = Counter("".join(titles))
print("Number of different characters:", len(count_chars.keys()))
non_ascii = [key for key in count_chars if ord(key) > 127]
non_ascii_count = sorted([(key, count_chars[key]) for key in non_ascii], key=lambda x: -1 * x[1])
for pair in non_ascii_count[:15]:
    print(pair, ord(pair[0]))
```

    Number of different characters: 217
    ('é', 457) 233
    ('ン', 178) 12531
    ('ー', 159) 12540
    ('á', 141) 225
    ('이', 140) 51060
    ('и', 138) 1080
    ('원', 128) 50896
    ('더', 128) 45908
    ('케', 128) 52992
    ('–', 118) 8211
    ('式', 116) 24335
    ('チ', 115) 12481
    ('ャ', 115) 12515
    ('ネ', 115) 12493
    ('ル', 115) 12523



```python
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
    print(ord(list(filter(lambda x: x[1] == i, chars_clusters))[0][0]))
```

    
    CLUSTER # 0
    [('원', 0), ('더', 0), ('케', 0), ('이', 0), ('영', 0), ('국', 0), ('남', 0), ('자', 0), ('특', 0), ('한', 0), ('동', 0), ('물', 0), ('채', 0), ('널', 0), ('펜', 0), ('타', 0), ('곤', 0), ('여', 0), ('친', 0), ('구', 0)]
    50896
    
    CLUSTER # 1
    [('é', 1), ('ė', 1), ('–', 1), ('ö', 1), ('ス', 1), ('ク', 1), ('ウ', 1), ('ェ', 1), ('ア', 1), ('・', 1), ('エ', 1), ('ニ', 1), ('ッ', 1), ('ワ', 1), ('ー', 1), ('ナ', 1), ('ブ', 1), ('ラ', 1), ('ザ', 1), ('チ', 1)]
    233
    
    CLUSTER # 2
    [('公', 2), ('式', 2), ('新', 2), ('日', 2), ('本', 2), ('株', 2), ('会', 2), ('社', 2), ('米', 2), ('津', 2), ('玄', 2), ('師', 2), ('東', 2), ('宝', 2), ('春', 2), ('晚', 2), ('郭', 2), ('辰', 2), ('圧', 2), ('倒', 2)]
    20844


### Emojis analysis


```python
def read_emojis_txt(filename):
    emojis_df = pd.read_csv(filename, sep=";", header=None, comment="#")
    print(emojis_df[0].head())
    array = emojis_df[0].to_numpy()
    result = []
    for code in array:
        # range
        code = code.strip()
        if ".." in code:
            begin_code, end_code = code.split("..")
            for code in range(int(begin_code, 16), int(end_code, 16)):
                result.append(chr(code))
            pass
        elif not " " in code:
            result.append(chr(int(code, 16)))
    print(len(result))
    return set(result)
    

emojis = read_emojis_txt("emojis.txt")

# small test
"👻" in emojis
```

    0    231A..231B    
    1    23E9..23EC    
    2    23F0          
    3    23F3          
    4    25FD..25FE    
    Name: 0, dtype: object
    983





    True




```python
def count_emojis(text):
    return sum([1 for char in text if char in emojis])

df["emojis_counts"] = df["description"].apply(count_emojis)
```


```python
print(df["emojis_counts"].describe())
sum(df["emojis_counts"] == 0), sum(df["emojis_counts"] != 0)
```

    count    78255.000000
    mean         0.185036
    std          1.295852
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max         30.000000
    Name: emojis_counts, dtype: float64





    (74450, 3805)



#### There is not much emojis in descriptions, anyway added column into consideration


```python
sns.distplot(df["emojis_counts"][df["emojis_counts"] != 0])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd875079520>




![png](output_87_1.png)


### Embeddings


```python
import io
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

tf.__version__
```




    '2.2.0-rc2'




```python
#  encoder = info.features['text'].encoder
def write_embedding_files(labels, embedded_ndarray, path=path, prefix=""):
    out_v = io.open(os.path.join(path, f"{prefix}_vecs.tsv"), "w", encoding="utf-8")
    out_m = io.open(os.path.join(path, f"{prefix}_meta.tsv"), "w", encoding="utf-8")
    vectors = embedded_ndarray
    for message, vector in zip(labels, vectors):
        out_m.write(message + "\n")
        out_v.write("\t".join([str(x) for x in vector]) + "\n")
    out_v.close()
    out_m.close()
```


```python
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```


```python
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
```

    Message: The quick brown fox jumps over the lazy dog.
    Embedding size: 512
    Embedding[-0.031330183148384094, -0.06338633596897125, -0.0160750150680542,...]
    
    Message: I am a sentence for which I would like to get its embedding
    Embedding size: 512
    Embedding[0.0508086122572422, -0.01652427949011326, 0.015737809240818024,...]
    



```python
unique_titles = np.unique(titles)
write_embedding_files(unique_titles, embed(unique_titles).numpy())
```

Embeding appropriate text columns


```python
df.columns
```




    Index(['video_id', 'trending_date', 'title', 'channel_title', 'category_id',
           'publish_time', 'tags', 'views', 'likes', 'dislikes', 'comment_count',
           'thumbnail_link', 'comments_disabled', 'ratings_disabled',
           'video_error_or_removed', 'description', 'week_day', 'time_of_day',
           'month', 'title_length_chars', 'title_length_tokens',
           'title_uppercase_ratio', 'title_not_alnum_ratio',
           'title_common_chars_count', 'channel_title_length_chars',
           'channel_title_length_tokens', 'tags_count', 'description_length_chars',
           'description_length_tokens', 'description_length_newlines',
           'description_uppercase_ratio', 'description_url_count',
           'emojis_counts'],
          dtype='object')




```python
def calc_embeddings(df, column_names, write_visualizations_files=False):
    for column in column_names:
        # batch_processing
        batch_size = 1000
        input_col = df[column].to_numpy()
        num_it = len(input_col) // batch_size

        result = np.zeros(shape=[len(input_col), 512])
        for i in range(num_it):
            result[batch_size * i: batch_size * (i + 1)]= embed(input_col[batch_size * i: batch_size * (i + 1)]).numpy()
        if len(input_col) % batch_size:
            result[batch_size * i:]= embed(input_col[batch_size * i:]).numpy()
        if write_visualizations_files:
            unique_inputs, unique_indexes = np.unique(input_col, return_index=True) 
            write_embedding_files(unique_inputs, result[unique_indexes], prefix=column)
        df[f"{column}_embed"] = list(result)

calc_embeddings(df, ["title", "channel_title"], False) # , "description" Description doesnt work...
```


```python
np.unique([[0, 1], [0, 1]], axis=0)
```




    array([[0, 1]])




```python
pd.set_option("colwidth", 15)
print(df.head())
pd.set_option("colwidth", None)
```

          video_id trending_date           title   channel_title  category_id  \
    0  Jw1Y-zhQURU      17.14.11  John Lewis ...      John Lewis          NaN   
    1  3s1rvMFUweQ      17.14.11  Taylor Swif...  Saturday Ni...          NaN   
    2  n1WpP7iowLc      17.14.11  Eminem - Wa...      EminemVEVO          NaN   
    3  PUTEiSjKwJU      17.14.11  Goals from ...  Salford Cit...          NaN   
    4  rHwDegptbI4      17.14.11  Dashcam cap...  Cute Girl V...          NaN   
    
         publish_time            tags     views   likes  dislikes  ...  \
    0  2017-11-10T...  christmas|"...   7224515   55681     10247  ...   
    1  2017-11-12T...  SNL|"Saturd...   1053632   25561      2294  ...   
    2  2017-11-10T...  Eminem|"Wal...  17158579  787420     43420  ...   
    3  2017-11-13T...  Salford Cit...     27833     193        12  ...   
    4  2017-11-13T...          [none]      9815      30         2  ...   
    
       channel_title_length_tokens tags_count  description_length_chars  \
    0               2                       8             821             
    1               3                      35             417             
    2               1                       5             594             
    3               4                       8             396             
    4               3                      -1             151             
    
       description_length_tokens  description_length_newlines  \
    0             125                         24                
    1              56                         11                
    2              70                         17                
    3              54                          7                
    4              23                          4                
    
      description_uppercase_ratio  description_url_count  emojis_counts  \
    0        0.047503                           7                     0   
    1        0.095923                           7                     0   
    2        0.048822                          14                     0   
    3        0.070707                           5                     0   
    4        0.039735                           0                     0   
    
          title_embed  channel_title_embed  
    0  [-0.0050918...  [-0.0297106...       
    1  [0.03928868...  [-0.0450098...       
    2  [0.02989614...  [0.01385195...       
    3  [0.03698702...  [0.04244102...       
    4  [-0.0025660...  [-0.0004196...       
    
    [5 rows x 35 columns]


### Tags


```python
def tags_transformer(x):
    return ", ".join(sorted([tag.replace('"', "") for tag in x.split("|")]))

df["transormed_tags"] = df["tags"].apply(tags_transformer)
```


```python
df["transormed_tags"].head()
```




    0                                                                                                                                                                                                                                                                                          christmas, christmas 2017, christmas ad, christmas ad 2017, john lewis, john lewis christmas, john lewis christmas advert, moz, mozthemonster
    1    Episode 1730, Girls Trip, If Loving You Is Wrong, Keanu, Look What You Made Me Do, OWN, Oprah Winfrey, Reputation, SNL, SNL Season 43, Saturday Night Live, Taylor Swift, Taylor Swift Ready for It, The Carmichael Show, Tiffany Haddish, actor, actress, comedian, comedy, episode 5, funny, guest, hilarious, host, impersonation, improv, late night, laugh, live, music, musician, new york, ready for it?, s43, s43e5, sketch
    2                                                                                                                                                                                                                                                                                                                                                                               Aftermath/Shady/Interscope, Eminem, On, Rap, Walk, Water
    3                                                                                                                                                                                                                                                                                            Class of 92, National League, National League North, Non League, Salford, Salford City, Salford City FC, Salford Uni, University of Salford
    4                                                                                                                                                                                                                                                                                                                                                                                                                                 [none]
    Name: transormed_tags, dtype: object




```python
cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
print(df["transormed_tags"][0])
cosine_loss(embed([df["transormed_tags"][0]]), embed(["lewis, christmas, what, none, moz"]))
```

    christmas, christmas 2017, christmas ad, christmas ad 2017, john lewis, john lewis christmas, john lewis christmas advert, moz, mozthemonster





    <tf.Tensor: shape=(), dtype=float32, numpy=-0.5619302>




```python
calc_embeddings(df, ["transormed_tags"], False)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>trending_date</th>
      <th>title</th>
      <th>channel_title</th>
      <th>category_id</th>
      <th>publish_time</th>
      <th>tags</th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>...</th>
      <th>description_length_chars</th>
      <th>description_length_tokens</th>
      <th>description_length_newlines</th>
      <th>description_uppercase_ratio</th>
      <th>description_url_count</th>
      <th>emojis_counts</th>
      <th>title_embed</th>
      <th>channel_title_embed</th>
      <th>transormed_tags</th>
      <th>transormed_tags_embed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jw1Y-zhQURU</td>
      <td>17.14.11</td>
      <td>John Lewis Christmas Ad 2017 - #MozTheMonster</td>
      <td>John Lewis</td>
      <td>NaN</td>
      <td>2017-11-10T07:38:29.000Z</td>
      <td>christmas|"john lewis christmas"|"john lewis"|"christmas ad"|"mozthemonster"|"christmas 2017"|"christmas ad 2017"|"john lewis christmas advert"|"moz"</td>
      <td>7224515</td>
      <td>55681</td>
      <td>10247</td>
      <td>...</td>
      <td>821</td>
      <td>125</td>
      <td>24</td>
      <td>0.047503</td>
      <td>7</td>
      <td>0</td>
      <td>[-0.005091841798275709, -0.03501396253705025, 0.024470044299960136, 0.0068895891308784485, 0.005174224730581045, -0.06355790793895721, -0.046983473002910614, 0.045910727232694626, 0.05336160957813263, -0.018825281411409378, -0.04587670788168907, 0.029513224959373474, 0.032403647899627686, 0.0672135055065155, 0.0296368096023798, -0.04013442248106003, -0.06935736536979675, -0.04193755239248276, 0.03987836465239525, 0.015229777432978153, 0.053287968039512634, -0.06341671198606491, 0.07341840118169785, 0.04851316288113594, -0.048704955726861954, 0.029907770454883575, 0.06315942108631134, -0.04547937959432602, 0.03641752526164055, -0.06284002959728241, -0.012376663275063038, -0.07766798138618469, -0.023182524368166924, -0.019294606521725655, -0.08200432360172272, -0.005668653175234795, 0.02857767976820469, -0.0416874997317791, 0.006396831013262272, 0.05615299940109253, 0.03886755183339119, -0.03949529677629471, 0.0031639421358704567, -0.07644487172365189, -0.05098548159003258, 0.05355275794863701, 0.06952707469463348, -0.0548761822283268, -0.024107636883854866, 0.005905281752347946, -0.01518475916236639, -0.07583069801330566, -0.02141406759619713, 0.012069711461663246, 0.007148789241909981, 0.01728614792227745, 0.05435257777571678, -0.058160778135061264, 0.00534764165058732, 0.04801211878657341, 0.0850357934832573, -0.06823601573705673, -0.08425016701221466, -0.03900540992617607, 0.07315398007631302, 0.0028741206042468548, -0.07161451876163483, -0.054075997322797775, 0.010171223431825638, 0.004802633076906204, -0.03450680524110794, 0.0173561442643404, -0.004887675400823355, -0.012039325200021267, 0.04135529696941376, -0.0031471122056245804, 0.07234080135822296, -0.0008361843065358698, 0.03990431874990463, 0.011781292036175728, 0.07448883354663849, -0.07951249927282333, -0.026723522692918777, 0.07422462105751038, -0.05855011194944382, 0.04041410610079765, 0.05483884736895561, 0.002947035478428006, 0.03914183750748634, 0.0028906376101076603, -0.019768906757235527, 0.005352904088795185, -0.04296785965561867, -0.07942917197942734, -0.04617784544825554, -0.048272352665662766, 0.0038346105720847845, 0.05194224789738655, -0.0089460713788867, 0.03255922347307205, ...]</td>
      <td>[-0.02971063368022442, 0.01896030083298683, 0.06452146917581558, 0.007386431563645601, -0.014871030114591122, 0.03151745721697807, 0.032862283289432526, -0.06464766710996628, 0.05297629535198212, -0.006257033906877041, -0.052753519266843796, 0.06857527792453766, 0.026013806462287903, 0.030575444921851158, 0.008657114580273628, -0.06211378425359726, -0.050779882818460464, 0.0466696135699749, -0.012659633532166481, -0.051646992564201355, 0.0727292150259018, 0.02368530072271824, 0.040839556604623795, 0.06877869367599487, 0.007595873437821865, -0.01914672926068306, 0.012358077801764011, -0.001296880654990673, 0.07297196984291077, -0.0279860720038414, 0.004338265862315893, -0.05435493215918541, 0.03622088208794594, 0.009190460667014122, -0.08247636258602142, -0.042290203273296356, 0.0622430257499218, 0.04769568890333176, 0.014346948824822903, 0.009992622770369053, -0.009260405786335468, -0.06445535272359848, -0.045931536704301834, -0.041634395718574524, -0.033023934811353683, 0.0777331292629242, -0.03209323063492775, 0.02421151101589203, 0.025226900354027748, -0.01982731930911541, -0.017118018120527267, 0.01746981032192707, 0.05225931107997894, -0.03691762685775757, 0.027439948171377182, -0.05051761120557785, 0.06617798656225204, -0.0749455988407135, 0.06174616515636444, -0.03342287614941597, 0.08600513637065887, -0.011145348660647869, -0.08601178228855133, 0.0034721940755844116, 0.014642126858234406, 0.006058812141418457, -0.07403961569070816, -0.070286825299263, 0.034857675433158875, 0.012675217352807522, -0.0237038005143404, 0.06504993140697479, -0.016289621591567993, 0.03177200257778168, 0.027393322438001633, -0.01969473995268345, 0.05861181020736694, -0.041612155735492706, 0.01354082953184843, 0.015224411152303219, 0.012248852290213108, -0.0656319409608841, -0.052470240741968155, 0.07401057332754135, -0.03200593963265419, 0.04625687375664711, -0.017151501029729843, 0.042472656816244125, 0.06161738187074661, -0.04871882125735283, 0.07641908526420593, -0.0717572271823883, 0.012891574762761593, -0.08086429536342621, 0.029427088797092438, -0.001265917206183076, 0.0221976600587368, -0.00753712747246027, -0.03638749569654465, 0.05084517225623131, ...]</td>
      <td>christmas, christmas 2017, christmas ad, christmas ad 2017, john lewis, john lewis christmas, john lewis christmas advert, moz, mozthemonster</td>
      <td>[-0.05650825798511505, -0.05317847430706024, 0.05258147791028023, 0.039779260754585266, 0.04354749247431755, -0.060117628425359726, -0.0625777468085289, -0.005973564460873604, -0.028266672044992447, -0.020562464371323586, -0.044429145753383636, -0.05645556002855301, 0.050593212246894836, 0.029242122545838356, -0.004220325965434313, -0.05428972467780113, -0.06256613880395889, -0.049682099372148514, 0.04003344103693962, 0.01872187666594982, 0.06061515957117081, -0.04106912761926651, 0.0626642182469368, -0.007924015633761883, -0.057614993304014206, -0.0585632361471653, 0.0615537203848362, -0.054378196597099304, 0.05527222156524658, -0.05582942068576813, 0.008606895804405212, -0.06268387287855148, -0.04504987597465515, 0.013298513367772102, -0.06256403028964996, 0.038544222712516785, 0.05497150495648384, -0.06124594435095787, -0.027608778327703476, -0.005496859550476074, 0.030139081180095673, -0.017098726704716682, -0.02012065425515175, -0.059861745685338974, -0.022684702649712563, 0.04180058836936951, 0.06167168170213699, -0.060308925807476044, 0.006226500030606985, -0.02985895797610283, -0.035526398569345474, 0.0021740382071584463, -0.05816883221268654, -0.013349401764571667, 0.03897114470601082, 0.027865413576364517, -0.016270391643047333, -0.045334864407777786, -0.013192042708396912, 0.05493134260177612, 0.0627455785870552, -0.039936959743499756, -0.06271973997354507, -0.018421368673443794, 0.0627325102686882, 0.029843242838978767, -0.04604264348745346, -0.016756277531385422, 0.0016830984968692064, -0.022817028686404228, 0.026293491944670677, 0.05763323977589607, -0.03487137332558632, 0.025404945015907288, 0.058869048953056335, -0.04005227982997894, 0.06160765513777733, 0.01815035566687584, 0.05221434310078621, 0.024125421419739723, 0.06271138042211533, -0.0625627264380455, -0.021374458447098732, 0.06104731559753418, -0.05411669611930847, -0.014183204621076584, 0.043865758925676346, 0.045605774968862534, 0.04281016066670418, 0.0036978102289140224, -0.038636110723018646, -0.03202126920223236, -0.03232667222619057, -0.06249064579606056, 0.032575543969869614, -0.061597809195518494, -0.0006199181079864502, 0.023944716900587082, 0.0370967835187912, 0.03916199505329132, ...]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3s1rvMFUweQ</td>
      <td>17.14.11</td>
      <td>Taylor Swift: …Ready for It? (Live) - SNL</td>
      <td>Saturday Night Live</td>
      <td>NaN</td>
      <td>2017-11-12T06:24:44.000Z</td>
      <td>SNL|"Saturday Night Live"|"SNL Season 43"|"Episode 1730"|"Tiffany Haddish"|"Taylor Swift"|"Taylor Swift Ready for It"|"s43"|"s43e5"|"episode 5"|"live"|"new york"|"comedy"|"sketch"|"funny"|"hilarious"|"late night"|"host"|"music"|"guest"|"laugh"|"impersonation"|"actor"|"improv"|"musician"|"comedian"|"actress"|"If Loving You Is Wrong"|"Oprah Winfrey"|"OWN"|"Girls Trip"|"The Carmichael Show"|"Keanu"|"Reputation"|"Look What You Made Me Do"|"ready for it?"</td>
      <td>1053632</td>
      <td>25561</td>
      <td>2294</td>
      <td>...</td>
      <td>417</td>
      <td>56</td>
      <td>11</td>
      <td>0.095923</td>
      <td>7</td>
      <td>0</td>
      <td>[0.039288681000471115, -0.0069383252412080765, 0.07743072509765625, -0.01942211203277111, 0.05312428995966911, -0.0028869423549622297, 0.04638775810599327, 0.044384103268384933, 0.03059392236173153, 0.006613639649003744, -0.011068971827626228, 0.00919104740023613, 0.053292278200387955, -0.05343255773186684, 0.03634525462985039, -0.048513129353523254, 0.03750867396593094, -0.015446526929736137, 0.05612833797931671, -0.06500312685966492, -0.014453893527388573, -0.03606420010328293, -0.01061958447098732, -0.018049897626042366, -0.07750784605741501, -0.01534215547144413, 0.06372438371181488, 0.05746811255812645, -0.015002687461674213, -0.03925217688083649, 0.06501417607069016, -0.023004893213510513, -0.030894825235009193, 0.06173738092184067, 0.04785357043147087, -0.002629632595926523, 0.009472052566707134, -0.01725994609296322, 0.00021696808107662946, 0.012641284614801407, 0.05169335752725601, -0.010078983381390572, 0.04988082870841026, -0.07005735486745834, 0.05192863568663597, 0.04864175617694855, 0.04869353398680687, 0.05522827431559563, 0.04554182291030884, -0.06367367506027222, 0.034416135400533676, -0.055030930787324905, 0.05641371011734009, 0.045716412365436554, -0.03141511231660843, 0.062297504395246506, 0.05951254069805145, 0.017762284725904465, -0.04456251114606857, -0.03877422213554382, -0.04795736074447632, 0.006995900999754667, -0.07579176127910614, -0.03913901373744011, -0.022477030754089355, -0.004321795888245106, 0.04524711146950722, -0.027794919908046722, 0.05013475567102432, 0.002774585038423538, -0.054281461983919144, -0.054731812328100204, 0.03613528981804848, -0.0013831708347424865, 0.05040498450398445, 0.04843410477042198, 0.029543668031692505, -0.011904478073120117, 0.01042269915342331, -0.005299792159348726, 0.055289436131715775, -0.04908640310168266, -0.05349813401699066, 0.039700236171483994, -0.04910654202103615, 0.017163127660751343, 0.011576134711503983, -0.07438109070062637, 0.0021767555736005306, -0.055947769433259964, 0.029086388647556305, 0.040439825505018234, 0.025563236325979233, 0.03876039385795593, 0.02634475566446781, -0.04828165844082832, 0.004550265613943338, 0.06363709270954132, 0.06521710753440857, 0.037976738065481186, ...]</td>
      <td>[-0.04500989243388176, -0.030348241329193115, 0.009652510285377502, -0.04319463297724724, 0.025847336277365685, 0.027773169800639153, 0.05039365962147713, -0.020356154069304466, 0.015285038389265537, 0.0339357852935791, -0.02819785475730896, 0.04918679967522621, -0.005939324386417866, 0.06555424630641937, 0.008322875015437603, -0.06485544145107269, -0.057676080614328384, -0.007312914356589317, 0.05996441841125488, -0.0009045513579621911, -0.00924558937549591, -0.05297460779547691, -0.009181481786072254, -0.05477023124694824, -0.03142819553613663, -0.03273235261440277, 0.026148954406380653, -0.01909741200506687, -0.03876318037509918, -0.01591571979224682, 0.06344293802976608, 0.0033249864354729652, -0.028302041813731194, 0.04140160232782364, 0.030857054516673088, -0.015369083732366562, 0.02690241113305092, -0.05845757573843002, -0.03316274657845497, -0.004677379038184881, 0.011378520168364048, 0.0004206551529932767, 0.061204805970191956, -0.048175159841775894, 0.009562617167830467, -0.04440831393003464, -0.005825475789606571, 0.06086575612425804, 0.07342815399169922, -0.04465068504214287, 0.02725941501557827, 0.06578392535448074, 0.07968942075967789, 0.057777684181928635, -0.06282427161931992, 0.05197376757860184, -0.007307741325348616, 0.039751097559928894, 0.04879816249012947, -0.018266480416059494, 0.0008445173152722418, 0.04309585690498352, 0.028964433819055557, 0.008395609445869923, -0.07006581127643585, 0.02069663442671299, 0.03957124054431915, 0.06748300045728683, 0.0008160602883435786, -0.002897781552746892, -0.03980320692062378, -0.0831875428557396, -0.001508981571532786, 0.005147710908204317, -0.02334607020020485, 0.04317651316523552, 0.04608432203531265, -0.04094801843166351, 0.04208572953939438, 0.03543587774038315, 0.017287487164139748, 0.007853851653635502, -0.07451321184635162, -0.0476880818605423, -0.06086583435535431, 0.03198852017521858, -0.016333039849996567, -0.03327898681163788, -0.02739619091153145, -0.05148607864975929, 0.025561852380633354, 0.012288378551602364, -0.04722384363412857, -0.02112923562526703, 0.026827946305274963, -0.06971411406993866, -0.014692454598844051, 0.01625482551753521, 0.07690494507551193, 0.04102060943841934, ...]</td>
      <td>Episode 1730, Girls Trip, If Loving You Is Wrong, Keanu, Look What You Made Me Do, OWN, Oprah Winfrey, Reputation, SNL, SNL Season 43, Saturday Night Live, Taylor Swift, Taylor Swift Ready for It, The Carmichael Show, Tiffany Haddish, actor, actress, comedian, comedy, episode 5, funny, guest, hilarious, host, impersonation, improv, late night, laugh, live, music, musician, new york, ready for it?, s43, s43e5, sketch</td>
      <td>[-0.020205063745379448, -0.005373778752982616, 0.062376346439123154, -0.048817023634910583, 0.061378270387649536, 0.001703165122307837, 0.044204577803611755, 0.036956388503313065, -0.0018385027069598436, -0.05510907992720604, -0.02831944264471531, 0.04540479928255081, 0.04520338028669357, -0.0585973896086216, 0.05852344259619713, -0.062435999512672424, 0.03670334443449974, -0.0249053705483675, 0.040320854634046555, -0.009145720861852169, 0.018621018156409264, -0.06069391220808029, 0.0028102307114750147, -0.05496527999639511, -0.062140997499227524, -0.012424508109688759, 0.05313573032617569, 0.039323579519987106, 0.04571380093693733, 0.018427887931466103, 0.05710145831108093, 0.05997803807258606, -0.060429900884628296, 0.015847666189074516, 0.04778526350855827, 0.020346559584140778, 0.05354996398091316, -0.057006362825632095, -0.024869563058018684, 0.03460220620036125, -0.0015790603356435895, -0.04264384135603905, 0.062429916113615036, -0.05005893483757973, 0.06164119020104408, -0.016080273315310478, 0.050015389919281006, 0.033480800688266754, 0.047489505261182785, -0.061426952481269836, 0.05184893310070038, 0.01072894036769867, -0.062376681715250015, -0.041584353893995285, -0.059088535606861115, 0.06205551326274872, 0.021055743098258972, 0.05626756325364113, -0.019665567204356194, -0.017227694392204285, -0.013419737108051777, 0.05022961273789406, -0.06244873255491257, -0.001710905460640788, -0.05890604108572006, 0.011267689988017082, 0.03783414140343666, 0.05497152730822563, 0.02226436696946621, -0.035400036722421646, 0.048356588929891586, 0.057635992765426636, -0.03944024816155434, 0.055929917842149734, 0.005994509439915419, 0.010507735423743725, -0.005501719191670418, 0.019499441608786583, 0.054218996316194534, -0.023847149685025215, 0.06215009093284607, 0.0017742206109687686, 0.004116421099752188, -0.01276802085340023, -0.04916408285498619, 0.021118255332112312, 0.04454842582345009, -0.05979399010539055, -0.061696793884038925, -0.05454361438751221, -0.0594187006354332, 0.05063559114933014, 0.036506231874227524, -0.00861680880188942, 0.06210463121533394, -0.060958150774240494, 0.04726342111825943, 0.052124444395303726, 0.06243916600942612, -0.0159621499478817, ...]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>n1WpP7iowLc</td>
      <td>17.14.11</td>
      <td>Eminem - Walk On Water (Audio) ft. Beyoncé</td>
      <td>EminemVEVO</td>
      <td>NaN</td>
      <td>2017-11-10T17:00:03.000Z</td>
      <td>Eminem|"Walk"|"On"|"Water"|"Aftermath/Shady/Interscope"|"Rap"</td>
      <td>17158579</td>
      <td>787420</td>
      <td>43420</td>
      <td>...</td>
      <td>594</td>
      <td>70</td>
      <td>17</td>
      <td>0.048822</td>
      <td>14</td>
      <td>0</td>
      <td>[0.029896140098571777, 0.04058977961540222, 0.07530400902032852, -0.011584913358092308, 0.05749419704079628, 0.012273832224309444, 0.06639967113733292, 0.008736140094697475, -0.029805218800902367, 0.04477962478995323, -0.013244591653347015, -0.04414362832903862, -0.03752753138542175, -0.004602440167218447, -0.07059207558631897, -0.041936688125133514, 0.0223141610622406, 0.0468781478703022, -0.06045052036643028, 0.013259747996926308, 0.043605878949165344, -0.06849446147680283, -0.04449052736163139, 0.07166385650634766, -0.07530458271503448, 0.03573901206254959, 0.04753456264734268, -0.010642695240676403, 0.02007775753736496, -0.02238660492002964, -0.03562737628817558, 0.013555922545492649, -0.007316420786082745, 0.008989360183477402, 0.04941173270344734, 0.03920209780335426, -0.04711393266916275, 0.04579537734389305, -0.061417534947395325, -0.05454251542687416, 0.07231145352125168, 0.005406425800174475, 0.00913546048104763, -0.0680263414978981, 0.06318400800228119, -0.03877267986536026, -0.024601908400654793, 0.05315447598695755, 7.968435966176912e-05, 0.04043421521782875, -0.06051918491721153, 0.05020271614193916, 0.008208421058952808, -0.04469072073698044, -0.06411804258823395, 0.023431215435266495, 0.039161622524261475, 0.04259210079908371, 0.024790771305561066, -0.03161119297146797, 0.07369571179151535, -0.04335755482316017, -0.07472940534353256, 0.012853311374783516, 0.05570327863097191, -0.015059838071465492, 0.0340401828289032, -0.021925825625658035, -0.03799686208367348, 0.010970202274620533, -0.06958089768886566, 0.03573713079094887, -0.028542401269078255, -0.054691705852746964, 0.054611001163721085, -0.03697994351387024, -0.0044011687859892845, -0.001841368735767901, 0.019966846331954002, -0.003957521636039019, 0.04903498291969299, -0.06121030077338219, -0.011519917286932468, -0.022350864484906197, 0.03127780184149742, 0.053201980888843536, -0.021718349307775497, -0.06479576975107193, -0.05942543223500252, 0.04629041999578476, -0.02548898756504059, 0.05637971684336662, 0.015956846997141838, -0.024788962677121162, -0.06808837503194809, -0.0731789693236351, -0.030380386859178543, 0.0534609779715538, 0.07446646690368652, 0.043548181653022766, ...]</td>
      <td>[0.013851956464350224, 0.00728388549759984, 0.0084955720230937, 0.01165651436895132, -0.023055793717503548, 0.005224643740803003, -0.0009089919622056186, -0.037205129861831665, -0.006044801790267229, 0.010315733030438423, -0.020449699833989143, -0.009321504272520542, 0.044727861881256104, 0.011441812850534916, -0.04154679924249649, 0.023432329297065735, -0.03240537643432617, 0.04136144369840622, -0.015286055393517017, -0.039448752999305725, -0.00111889757681638, 0.05371971055865288, -0.022722851485013962, 0.04105021059513092, 0.007195890415459871, 0.03136885538697243, -0.04844866693019867, -0.05102938041090965, 0.03166762739419937, -0.009417380206286907, 0.07322385162115097, -0.020291881635785103, -0.0004308559582568705, 0.011913212947547436, 0.009354343637824059, 0.021716419607400894, -0.008168552070856094, -0.001825847546570003, 0.07215859740972519, 0.060230180621147156, 0.011684178374707699, -0.03920081630349159, -0.03585235774517059, 0.007656932808458805, 0.03424518182873726, 0.01969398930668831, -0.01686209999024868, 0.03131014481186867, -0.0339854434132576, 0.03493744507431984, -0.007782174274325371, -0.02147519215941429, 0.11034544557332993, 0.08153484761714935, -0.09494765102863312, -0.04610593989491463, 0.010511696338653564, 0.07056226581335068, 0.0541355274617672, -0.05112072452902794, 0.021642236039042473, 0.01349797286093235, -0.014223725534975529, -0.04279382526874542, -0.05685429275035858, 0.04794924333691597, -0.03230881318449974, 0.03042338788509369, -0.020532922819256783, 0.04848095774650574, -0.013970145024359226, -0.007976679131388664, -0.017122633755207062, -0.03262611851096153, 0.010530778206884861, -0.05087989196181297, -0.07659614086151123, -0.05605730786919594, -0.0436423234641552, -0.0409674234688282, -0.010744461789727211, -2.4010771539906273e-06, 0.03767657279968262, -0.03280111029744148, 0.0217526163905859, 0.0016317485133185983, 0.0024999070446938276, 0.0069821784272789955, -0.01410175021737814, -0.0211363323032856, -0.018604077398777008, 0.012337065301835537, -0.07811903953552246, 0.0967676192522049, 0.037091560661792755, 0.08948477357625961, 0.0022305953316390514, -0.02002081274986267, 0.03622571378946304, 0.04549553990364075, ...]</td>
      <td>Aftermath/Shady/Interscope, Eminem, On, Rap, Walk, Water</td>
      <td>[-0.004463671240955591, 0.024410290643572807, 0.08088809251785278, -0.017149770632386208, 0.05679933354258537, 0.034919824451208115, 0.06493070721626282, 0.02048019878566265, -0.02015043795108795, 0.04530903324484825, 0.015453871339559555, -0.05664650723338127, -0.05336378887295723, 0.014271057210862637, -0.06841331720352173, -0.06963545829057693, -0.06363275647163391, 0.04114792123436928, -0.05003443732857704, -0.05897779017686844, -0.03486393392086029, -0.026575656607747078, -0.05943921208381653, 0.03137997165322304, -0.08089898526668549, 0.057519737631082535, 0.0009317283402197063, -0.02799399569630623, -0.025572601705789566, -0.010328219272196293, -0.010511425323784351, 0.05623588338494301, 0.02984848991036415, -0.04686599224805832, 0.03785044327378273, 0.07243958860635757, 0.004593266639858484, -0.04799903556704521, -0.06717906147241592, -0.06297621876001358, 0.07580289989709854, 0.006575665902346373, -0.021998459473252296, -0.028881384059786797, 0.06824729591608047, -0.05505025014281273, 0.016376031562685966, 0.041438426822423935, 0.041503794491291046, -0.007463962305337191, -0.042259860783815384, 0.02956123650074005, -0.039970070123672485, -0.052566707134246826, -0.08060935884714127, 0.01135038211941719, 0.010227360762655735, 0.03243589028716087, 0.04852321743965149, -0.008387748152017593, 0.07910030335187912, 0.017370784655213356, -0.07560126483440399, 0.0573812834918499, 0.029764121398329735, -0.009579939767718315, 0.019128093495965004, 0.0045927222818136215, -0.07058581709861755, 0.024904901161789894, -0.0537908710539341, -0.009367113932967186, 0.003155461046844721, -0.06056452915072441, 0.05320841446518898, 0.01000206358730793, -0.06006282567977905, -0.008936331607401371, 0.001423617242835462, -0.009672070853412151, 0.04053835943341255, -0.037312425673007965, 0.003305489895865321, -0.04136500135064125, -0.020431870594620705, -0.004849726799875498, 0.044872649013996124, -0.06164812296628952, -0.05780842900276184, 0.03975250944495201, -0.05459913611412048, 0.02540465071797371, 0.025393573567271233, 0.023806171491742134, -0.05804258584976196, -0.06632877141237259, 0.01193391252309084, 0.05633135885000229, 0.0788852795958519, 0.02340948022902012, ...]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PUTEiSjKwJU</td>
      <td>17.14.11</td>
      <td>Goals from Salford City vs Class of 92 and Friends at The Peninsula Stadium!</td>
      <td>Salford City Football Club</td>
      <td>NaN</td>
      <td>2017-11-13T02:30:38.000Z</td>
      <td>Salford City FC|"Salford City"|"Salford"|"Class of 92"|"University of Salford"|"Salford Uni"|"Non League"|"National League"|"National League North"</td>
      <td>27833</td>
      <td>193</td>
      <td>12</td>
      <td>...</td>
      <td>396</td>
      <td>54</td>
      <td>7</td>
      <td>0.070707</td>
      <td>5</td>
      <td>0</td>
      <td>[0.03698702156543732, -0.0594879686832428, -0.04617753252387047, 0.05924941599369049, -0.0104559026658535, 0.023427648469805717, -0.03190228343009949, 0.03009907715022564, 0.06581231951713562, -0.014459213241934776, -0.08778245002031326, 0.02836141362786293, -0.0011798253981396556, -0.03302150219678879, 0.010120570659637451, 0.007745935581624508, -0.05532084032893181, 0.05130602419376373, -0.05289115384221077, -0.034825652837753296, 0.08374786376953125, 0.04569961503148079, 0.06970220059156418, -0.002940218895673752, -0.009960008785128593, -0.057443730533123016, -0.0003243562823627144, 0.036261480301618576, -0.005765492562204599, -0.009454104118049145, 0.07183664292097092, -0.012884947471320629, 0.023538777604699135, 0.02685074508190155, -0.07416684925556183, -0.04980602115392685, 0.03654152527451515, 0.04914504289627075, 0.06892668455839157, 0.0033082258887588978, -0.07429755479097366, 0.0023315439466387033, -0.020539747551083565, -0.020803069695830345, -0.08252459764480591, -0.07979041337966919, -0.02323109842836857, -0.04041125252842903, 0.05122034251689911, -0.009983730502426624, 0.08344674110412598, -0.013455945067107677, -0.024891456589102745, 0.04090022295713425, 0.054344091564416885, 0.003364801174029708, 0.03480381891131401, -0.037435658276081085, -0.05120956525206566, -0.031217481940984726, -0.023206796497106552, 0.04847560450434685, 0.06942885369062424, 0.06631314754486084, -0.04078366234898567, 0.04716545343399048, 0.05032725632190704, -0.0313667356967926, 0.0022105828393250704, -0.021269073709845543, -0.05503521114587784, 0.060882408171892166, 0.03073768876492977, 0.0408853143453598, -0.02495325729250908, 0.053000327199697495, -0.0711461752653122, -0.040277622640132904, -0.022556956857442856, -0.019242381677031517, -0.008918838575482368, 0.002092757262289524, 0.017182886600494385, -0.0074582635425031185, 0.02343609184026718, -0.06992794573307037, -0.032047756016254425, 0.04931342601776123, -0.05381666496396065, -0.04093318432569504, 0.0802077203989029, -0.025366412475705147, 0.037299446761608124, -0.05116444081068039, 0.055162087082862854, -0.019451797008514404, 0.017033111304044724, 0.044028084725141525, -0.048829011619091034, 0.013521850109100342, ...]</td>
      <td>[0.042441025376319885, 0.025883154943585396, -0.05898943170905113, 0.022117773070931435, -0.037892524152994156, 0.0350419282913208, -0.03305860236287117, -0.050749994814395905, -0.023662978783249855, -0.07051733136177063, -0.08198216557502747, 0.008870143443346024, 0.016310948878526688, -0.04524674266576767, 0.03313044458627701, -0.04412084445357323, -0.056501518934965134, -0.04225819185376167, -0.005585434380918741, -0.03174729272723198, 0.0640086829662323, 0.07101909071207047, 0.06704406440258026, 0.02086833491921425, 0.01060893852263689, -0.07117141038179398, -0.03411227464675903, -0.034718792885541916, -0.031196631491184235, -0.03897570073604584, 0.07568740844726562, 0.04694294556975365, 0.03256010636687279, 0.06685972958803177, -0.07416535913944244, -0.07937303930521011, 0.019551830366253853, 0.03499039262533188, -0.0579429566860199, 0.017562326043844223, -0.08241857588291168, -0.02611803449690342, 0.03317869082093239, -0.011215313337743282, -0.07825999706983566, -0.05546204000711441, 0.00697778444737196, -0.018224062398076057, 0.06913451105356216, 0.05339794233441353, 0.0834057554602623, 0.0493273064494133, 0.015811430290341377, 0.04955857992172241, 0.06560356169939041, -0.0019372443202883005, 0.03131716325879097, -0.07966896146535873, 0.04686211049556732, 0.005127415526658297, -0.043985847383737564, 0.0021918118000030518, 0.06891078501939774, 0.044389210641384125, 0.002116707619279623, -0.04086007550358772, -0.051362983882427216, 0.0006528186495415866, -0.010188127867877483, -0.0133671248331666, -0.04338754713535309, -0.038707882165908813, 0.06478811800479889, -0.043740857392549515, -0.0008755438611842692, -0.011416207067668438, -0.05209542065858841, -0.024482037872076035, -0.020587189123034477, 0.01550703588873148, -0.035151008516550064, 0.04584329202771187, -0.06491046398878098, -0.06553088873624802, -0.010004233568906784, -0.0038423966616392136, -0.061229657381772995, 0.0818815603852272, -0.050451092422008514, 0.005834414158016443, -0.03193756565451622, 0.023035796359181404, 0.04896792024374008, -0.07464131712913513, 0.07465417683124542, -0.032627321779727936, -0.0357961468398571, 0.023344874382019043, -0.018264612182974815, 0.03916175663471222, ...]</td>
      <td>Class of 92, National League, National League North, Non League, Salford, Salford City, Salford City FC, Salford Uni, University of Salford</td>
      <td>[0.039707839488983154, -0.011713636107742786, -0.05791545659303665, 0.04770888015627861, 0.03181470185518265, 0.05328508839011192, -0.01607429049909115, -0.001659640111029148, 0.04710397496819496, -0.05731940641999245, -0.06522522866725922, 0.05217314884066582, 0.0297729279845953, -0.0486239530146122, 0.03479520604014397, 0.03806500881910324, -0.05731027200818062, -0.05270007252693176, -0.059059564024209976, -0.061430539935827255, 0.04345504194498062, 0.06202523037791252, 0.06335554271936417, 0.01824505627155304, -0.02054085209965706, -0.05946783721446991, 0.03672638535499573, 0.04096197336912155, 0.009790771640837193, -0.016510717570781708, 0.06401395052671432, 0.002679114695638418, 0.03915400430560112, 0.04962301254272461, -0.06423616409301758, -0.06358896195888519, 0.003040169132873416, 0.03354964777827263, -0.05946001410484314, 0.05546940490603447, -0.06514042615890503, 0.011120500974357128, 0.03623559698462486, -0.03429824113845825, -0.0644877478480339, -0.06103963777422905, -0.037172552198171616, -0.06077829375863075, 0.06472190469503403, -0.007247522007673979, 0.06522511690855026, -0.015159706585109234, -0.06038104370236397, 0.061256639659404755, 0.06450220197439194, -0.05078094080090523, 0.01274490263313055, -0.06511644273996353, -0.0484430305659771, -0.013655695132911205, -0.01046711951494217, 0.0290190652012825, 0.05594707652926445, 0.03954501450061798, -0.0318012610077858, 0.043607197701931, 0.025157425552606583, -0.03012699820101261, -0.02542855404317379, 0.021783890202641487, -0.04703648015856743, 0.054587993770837784, 0.06039576232433319, 0.030448216944932938, -0.03169965744018555, 0.013267474249005318, -0.061665840446949005, -0.02269994094967842, -0.039299894124269485, -0.025530975311994553, -0.05936998501420021, 0.04205646365880966, -0.011129285208880901, -0.061346977949142456, -0.02613098919391632, -0.058403026312589645, 0.006819091271609068, 0.06472815573215485, -0.031207310035824776, 0.005518801044672728, -0.047424137592315674, 0.022783497348427773, 0.059576451778411865, -0.06469640880823135, 0.06429097801446915, 0.05567338690161705, -0.014370462857186794, 0.053149301558732986, -0.04761430621147156, -0.033946555107831955, ...]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rHwDegptbI4</td>
      <td>17.14.11</td>
      <td>Dashcam captures truck's near miss with child in Norway</td>
      <td>Cute Girl Videos</td>
      <td>NaN</td>
      <td>2017-11-13T01:45:13.000Z</td>
      <td>[none]</td>
      <td>9815</td>
      <td>30</td>
      <td>2</td>
      <td>...</td>
      <td>151</td>
      <td>23</td>
      <td>4</td>
      <td>0.039735</td>
      <td>0</td>
      <td>0</td>
      <td>[-0.0025660621467977762, 0.07401233911514282, -0.02016867697238922, -0.032766230404376984, 0.004739161115139723, 0.0028290986083447933, -0.05159008875489235, 0.07514362037181854, 0.01689090020954609, -0.04505756497383118, 0.02102091535925865, 0.02912275306880474, -0.05190291255712509, -0.004615333862602711, -0.00935843214392662, -0.005522352177649736, -0.03235519677400589, -0.04392477124929428, 0.05734782665967941, 0.06380341947078705, 0.012576029635965824, 0.011960813775658607, -0.01515203807502985, -0.0709429532289505, 0.004932077135890722, -0.013301782310009003, 0.06492317467927933, 0.05094949156045914, -0.03772549331188202, 0.04234333336353302, -0.07140200585126877, 0.005452999845147133, -0.038494713604450226, -0.07104106992483139, -0.02773682400584221, 0.02308775670826435, -0.01190179493278265, 0.04417320340871811, 0.03690732270479202, -0.015389060601592064, 0.07639655470848083, 0.038921769708395004, 0.03213024511933327, -0.05680273845791817, -0.08018989115953445, -0.021627135574817657, -0.015449725091457367, 0.027553094550967216, -0.05055680871009827, 0.07311416417360306, -0.03970467299222946, -0.07723259925842285, 0.027431756258010864, 0.04807509109377861, 0.06233934685587883, -0.02576061151921749, -0.01126862782984972, -0.01643129251897335, 0.04880543425679207, 0.04754155874252319, -0.010292379185557365, -0.05454007163643837, -0.07095347344875336, 0.0527065210044384, -0.0679069384932518, 0.010347864590585232, 0.05537955090403557, 0.05820365250110626, -0.03480338677763939, -0.0028039184398949146, -0.06998880207538605, 0.021879633888602257, 0.005307199899107218, -0.05716981738805771, 0.040518004447221756, 0.02848421037197113, 0.035762932151556015, 0.05646350234746933, -0.04295828565955162, 0.031230425462126732, 0.08029098808765411, -0.018497798591852188, -0.01725236140191555, -0.011781701818108559, 0.055798083543777466, 0.027781525626778603, 0.0188822690397501, -0.02414855547249317, -0.025852259248495102, 0.0012840857962146401, 0.05099283531308174, 0.07173435389995575, 0.04917353391647339, 0.04366650804877281, -0.013320216909050941, -0.05809509754180908, 0.0004067150584887713, 0.0529300719499588, -0.045498963445425034, 0.06983830034732819, ...]</td>
      <td>[-0.00041962802060879767, 0.056079935282468796, 0.003309292485937476, -0.01591983437538147, -0.0002310668205609545, 0.026458708569407463, -0.036500975489616394, 0.09211715310811996, -0.03740188851952553, 0.026904713362455368, 0.029106922447681427, -0.013477472588419914, -0.05396333709359169, 0.08053042739629745, -0.015074760653078556, -0.06881854683160782, 0.043059080839157104, -0.004291935823857784, -0.0835374966263771, -0.06090927496552467, 0.03551596403121948, -0.03200254216790199, -0.02680218778550625, 0.019048547372221947, -0.08561839163303375, 0.00419315742328763, 0.00351363024674356, 0.014713376760482788, 0.007484192494302988, -0.027515044435858727, -0.005478752311319113, -0.025709832087159157, -0.04514823481440544, 0.05333787202835083, -0.019966188818216324, 0.0769684836268425, -0.03580727055668831, -0.01030927523970604, -0.016082530841231346, 0.021905891597270966, -0.01856503263115883, 0.03877807781100273, -0.06278955936431885, -0.031144866719841957, 0.028654752299189568, 0.016433656215667725, -0.005938854534178972, 0.01444904413074255, -0.06165516376495361, -0.024050652980804443, -0.06462689489126205, 0.05111309885978699, -0.032122742384672165, 0.04322337359189987, -0.010241792537271976, 0.00793943740427494, 0.025951558724045753, -0.005657656583935022, 0.07046889513731003, 0.03426024690270424, -0.08486881852149963, -0.018711794167757034, -0.08713575452566147, 0.04350031167268753, -0.039741866290569305, 0.058366890996694565, -0.045630089938640594, 0.03794680908322334, 0.024325355887413025, 0.008471641689538956, 0.01636556163430214, -0.022395433858036995, 0.004361520521342754, -0.00873649213463068, 0.030795074999332428, 0.0472358763217926, 0.0005029875319451094, 0.06993406265974045, -0.06192780286073685, 0.036878690123558044, -0.00028874765848740935, 0.08607260137796402, -0.057993095368146896, -0.06692071259021759, -0.04345812276005745, 0.02588677778840065, 0.01754787005484104, -0.03583747521042824, -0.05688028410077095, 0.014916334301233292, 0.030903832986950874, 0.07314936071634293, 0.05927427113056183, 0.04532744735479355, -0.04576956853270531, -0.042806487530469894, 0.023542020469903946, 0.05664671212434769, -0.047415245324373245, 0.0598045289516449, ...]</td>
      <td>[none]</td>
      <td>[-0.06890606880187988, -0.06527753174304962, 0.00474488316103816, -0.07200437039136887, -0.049485351890325546, 0.013415690511465073, 0.04432838037610054, -0.005674577318131924, -0.07965438812971115, -0.01933957077562809, 0.015232820063829422, 0.04530401900410652, 0.06058049201965332, -0.07967492192983627, 0.04787122830748558, 0.06382979452610016, -0.02814880758523941, 0.024407487362623215, -0.028790775686502457, 0.027145177125930786, 0.0835169330239296, 0.014171982184052467, -0.019215896725654602, 0.04409782961010933, 0.009876691736280918, 0.04660492017865181, 0.07268918305635452, -0.08183494955301285, -0.049595557153224945, 0.052024513483047485, 0.043357402086257935, -0.002310029696673155, -0.026471830904483795, -0.0655069574713707, -0.021977555006742477, -0.0524870939552784, 0.03003179468214512, 0.01398453488945961, -0.03529636561870575, -0.011309203691780567, -0.03611506521701813, 0.06399687379598618, -0.014639610424637794, 0.042340680956840515, 0.017897890880703926, 0.02180793322622776, -0.07109320163726807, 0.07321067154407501, -0.028136219829320908, -0.00922258011996746, 0.045658595860004425, -0.06511558592319489, 0.04158170521259308, 0.06464574486017227, 0.032119523733854294, 0.04162513092160225, 0.04547373577952385, 0.009046979248523712, 0.033821653574705124, -0.07710089534521103, 0.05110042169690132, -0.017872966825962067, 0.011662641540169716, -0.04036581143736839, -0.036490026861429214, -0.03191787376999855, -0.053871795535087585, -0.06054125726222992, -0.01552205067127943, 0.004773660562932491, -0.06383129954338074, 0.06532156467437744, -0.03741082549095154, 0.08606045693159103, 0.008498778566718102, 0.02185351960361004, 0.004867427051067352, -0.010801794938743114, 0.004058252088725567, -0.02209807001054287, 0.06665069609880447, 0.044320039451122284, 0.06096072494983673, -0.04445991292595863, 0.0001674878440098837, -0.05783943831920624, 0.06412035971879959, 0.004358451813459396, 0.003724749432876706, 0.036491759121418, 0.06395004689693451, 0.0768292173743248, -0.033808741718530655, 0.031769268214702606, -0.0598084032535553, -0.0068783811293542385, 0.03821703791618347, 0.05672299489378929, 0.0057940431870520115, 0.028924213722348213, ...]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78250</th>
      <td>1PhPYr_9zRY</td>
      <td>18.14.06</td>
      <td>BTS Plays With Puppies While Answering Fan Questions</td>
      <td>BuzzFeed Celeb</td>
      <td>NaN</td>
      <td>2018-05-18T16:39:29.000Z</td>
      <td>BuzzFeed|"BuzzFeedVideo"|"Puppy Interview"|"puppy"|"john lennon"|"questions"|"q&amp;A"|"BTS"|"k-pop"|"korean"|"korean music"|"korean pop music"|"fashion"|"BuzzFeedCeleb"|"ARMY's"|"songs"|"music"|"new album"|"funny"|"cute"|"puppies"|"cute puppies"|"Play with puppies"|"animals"|"cute animals"|"celebrities"|"boyband"</td>
      <td>8259128</td>
      <td>645888</td>
      <td>4052</td>
      <td>...</td>
      <td>926</td>
      <td>104</td>
      <td>0</td>
      <td>0.086393</td>
      <td>14</td>
      <td>0</td>
      <td>[0.0570649616420269, 0.02147122099995613, 0.06740880012512207, -0.018862353637814522, -0.0182102769613266, -0.009797926992177963, -0.07669155299663544, -0.06969863921403885, -0.0851852223277092, 0.02224336564540863, 0.021509120240807533, 0.07517463713884354, 0.007152911275625229, -0.027870826423168182, 0.029524346813559532, -0.038321781903505325, 0.03723530098795891, -0.009896883741021156, 0.03296910971403122, -0.017133450135588646, -0.016404692083597183, 0.019578762352466583, 0.0012376252561807632, 0.017266131937503815, -0.06393490731716156, 0.050331201404333115, -0.08265923708677292, -0.007141836918890476, -0.06390765309333801, 0.03931773826479912, 0.07487577944993973, 0.057136062532663345, -0.04237716272473335, -0.03290664032101631, 0.05992288887500763, 0.029842078685760498, -0.031660497188568115, -0.05106304958462715, -0.07533074915409088, -0.05114411190152168, 0.0019825666677206755, 0.04972846060991287, 0.05500771850347519, -0.03487461432814598, -0.06694221496582031, -0.05618264153599739, -0.011328071355819702, -0.03173794597387314, 0.08131728321313858, -0.05629516392946243, 0.07980333268642426, 0.02922305278480053, 0.020643332973122597, -0.0249028243124485, -0.016963617876172066, 0.022181764245033264, 0.011782264336943626, 0.020493507385253906, 0.048941802233457565, -0.07414357364177704, 0.010995319113135338, -0.017464499920606613, -0.08227367699146271, 0.05246816948056221, 0.07768463343381882, -0.018140772357583046, -0.0333237424492836, 0.05866429582238197, -0.055687423795461655, -0.054040223360061646, 0.06205957755446434, -0.0010366433998569846, 0.011341807432472706, 0.02931801602244377, -0.08575393259525299, -0.07209376245737076, -0.020917154848575592, 0.05829404667019844, 0.033830177038908005, 0.032407816499471664, -0.0012152105337008834, -0.021434854716062546, -0.014718976803123951, 0.028023269027471542, -0.05314009264111519, -0.040091898292303085, 0.011122898198664188, 0.0501297265291214, -0.03305118903517723, -0.04305678978562355, 0.014853348024189472, 0.0209577064961195, -0.018857058137655258, 0.0006600423366762698, 0.03984967619180679, -0.042367104440927505, 0.040904317051172256, 0.025200705975294113, -0.036301787942647934, 0.06926164031028748, ...]</td>
      <td>[0.03750678896903992, 0.03273759409785271, 0.06837526708841324, -0.06288312375545502, 0.05601122975349426, 0.03414342552423477, -0.008357541635632515, 0.02672594226896763, -0.07030580937862396, -0.014816165901720524, 0.02299456112086773, -0.045355457812547684, -0.052601296454668045, -0.015154636465013027, -0.05356965959072113, -0.06421088427305222, 0.02682713232934475, -0.06789594143629074, 0.035366613417863846, 0.011175289750099182, 0.003399464301764965, -0.0612909197807312, 0.02991378866136074, 0.0587085522711277, 0.04926060140132904, 0.01892237551510334, -0.06207548826932907, -0.06762561202049255, 0.010347291827201843, 0.031003832817077637, -0.03677903488278389, -0.02235480584204197, 0.0015641630161553621, -0.06303390115499496, -0.022771818563342094, 0.04731515794992447, -0.06857338547706604, 0.03241758048534393, 0.023291517049074173, 0.02699223719537258, -0.06287037581205368, -0.006246437318623066, 0.05126384645700455, -0.040958188474178314, 0.05775001272559166, -0.0305916927754879, -0.026320036500692368, 0.04517752677202225, 0.036644693464040756, 0.023201778531074524, 0.02249392308294773, -0.03200393170118332, -0.041486091911792755, -0.055720675736665726, -0.047899458557367325, -0.06550668179988861, 0.03413037583231926, -0.0006598465261049569, -0.020892875269055367, -0.027671199291944504, -0.016461661085486412, -0.01493016816675663, -0.05212430655956268, -0.02550617605447769, -0.0446760468184948, 0.028759021311998367, -0.021266212686896324, -0.050687242299318314, -0.01639193668961525, -0.0011158538982272148, -0.021788792684674263, 0.014202120713889599, -0.0210060216486454, 0.07060568034648895, 0.06745633482933044, 0.05685338005423546, 0.005458203610032797, 0.06246934086084366, 0.03382781520485878, -0.009461811743676662, 0.04540744051337242, -0.02228349819779396, -0.06615830212831497, -0.06108710169792175, -0.06600356847047806, 0.041082438081502914, 0.037531137466430664, -0.007481304928660393, -0.053329844027757645, -0.05774962529540062, 0.04124879464507103, 0.04330422729253769, -0.022420017048716545, -0.06503181159496307, 0.05685890093445778, 0.022867001593112946, 0.036575619131326675, 0.014296307228505611, 0.055087704211473465, 0.034236978739500046, ...]</td>
      <td>ARMY's, BTS, BuzzFeed, BuzzFeedCeleb, BuzzFeedVideo, Play with puppies, Puppy Interview, animals, boyband, celebrities, cute, cute animals, cute puppies, fashion, funny, john lennon, k-pop, korean, korean music, korean pop music, music, new album, puppies, puppy, q&amp;A, questions, songs</td>
      <td>[-0.03220510110259056, 0.01855790801346302, 0.057570286095142365, 0.005401831120252609, -0.05182057246565819, 0.05528750643134117, -0.05754490941762924, 0.04737330973148346, -0.057564906775951385, 0.016449538990855217, 0.05723552778363228, 0.05756569653749466, -0.047130703926086426, -0.05531245842576027, 0.05059884116053581, -0.05107572302222252, 0.053585268557071686, -0.0388459637761116, 0.002092653652653098, -0.025670770555734634, 0.0016717059770599008, 0.05556531250476837, 0.04292158782482147, 0.054937079548835754, -0.057570286095142365, 0.0546114444732666, -0.057037290185689926, -0.044910892844200134, 0.04680272936820984, -0.018480660393834114, 0.057423755526542664, 0.05437565967440605, -0.04322478920221329, -0.05620614439249039, 0.05067780986428261, 0.056949540972709656, -0.04000924527645111, -0.055098891258239746, 0.030945660546422005, 0.03079262375831604, -0.057148247957229614, 0.030420294031500816, 0.056671142578125, 0.02067043073475361, -0.05755946412682533, -0.057274702936410904, 0.04804683476686478, -0.03234120458364487, 0.055206794291734695, -0.015917815268039703, -0.04928801208734512, 0.04282446205615997, -0.05756865441799164, -0.007574689574539661, -0.05167587473988533, -0.00496162474155426, -0.007869565859436989, 0.05703439563512802, -0.013461493887007236, -0.05751771479845047, -0.0021231728605926037, 0.025647848844528198, -0.05738144740462303, 0.011608000844717026, 0.05756961554288864, 0.0531940683722496, -0.03386968746781349, -0.03429609537124634, -0.053058747202157974, -0.006950557231903076, 0.05421340838074684, 0.056954555213451385, -0.02682354673743248, 0.05584210157394409, -0.012661658227443695, -0.03643282130360603, -0.04456155002117157, 0.05602375790476799, 0.05079958960413933, -0.003002275712788105, 0.05524105951189995, 0.05720800161361694, -0.04952574148774147, -0.056989751756191254, 0.0030353632755577564, -0.0444524921476841, 0.03411564603447914, 0.05314798280596733, -0.057456374168395996, -0.0045377276837825775, -0.029602542519569397, 0.05493979528546333, 0.021675292402505875, 0.015284805558621883, 0.05710224062204361, -0.05718514695763588, 0.05286407470703125, 0.05168459191918373, 0.05749960616230965, 0.04983244091272354, ...]</td>
    </tr>
    <tr>
      <th>78251</th>
      <td>1h7KV2sjUWY</td>
      <td>18.14.06</td>
      <td>True Facts : Ant Mutualism</td>
      <td>zefrank1</td>
      <td>NaN</td>
      <td>2018-05-18T01:00:06.000Z</td>
      <td>[none]</td>
      <td>1064798</td>
      <td>60008</td>
      <td>382</td>
      <td>...</td>
      <td>14</td>
      <td>2</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>[0.07197204232215881, -0.06309523433446884, 0.006990946363657713, -0.030520403757691383, 0.029214225709438324, 0.023781469091773033, -0.04028615728020668, 0.009532208554446697, 0.0029062468092888594, 0.05796515941619873, 0.07912807911634445, 0.06135546788573265, 0.07562721520662308, 0.056051407009363174, 0.045244116336107254, -0.06343888491392136, -0.05036395415663719, 0.006710805464535952, -0.049881428480148315, -0.04592946171760559, -0.01814369671046734, -0.031279511749744415, 0.06350217759609222, 0.03849911317229271, -0.01656702719628811, 0.07505334913730621, -0.07309309393167496, -0.0624534972012043, -0.016457457095384598, -0.011198833584785461, 0.029667051509022713, -0.06931522488594055, 0.01646392233669758, 0.051322825253009796, -0.05754493176937103, 0.06987680494785309, -0.050830282270908356, 0.03437822312116623, -0.012028752826154232, 0.010252725332975388, -0.015788260847330093, -0.04031004011631012, 0.036039359867572784, 0.045648206025362015, -0.07931952178478241, -0.054787613451480865, 0.021735189482569695, 0.026363801211118698, -0.026762567460536957, 0.027682047337293625, 0.0781836286187172, -0.03748435899615288, -0.04840551316738129, -0.03342481330037117, 0.004925432149320841, -0.005142535082995892, 0.07535916566848755, 0.006348432507365942, 0.04754837602376938, 0.03893066570162773, -0.01918644830584526, -0.058656156063079834, -0.032121364027261734, 0.04659596458077431, 0.021819785237312317, -0.00040220460505224764, 0.04611877724528313, 0.050070058554410934, 0.021830104291439056, -0.04386083781719208, -0.02945409156382084, -0.02732713147997856, 0.01835986226797104, -0.011511688120663166, 0.027058295905590057, -0.07248927652835846, -0.07341054081916809, -0.025163967162370682, -0.005189648829400539, 0.053292203694581985, -0.02855270728468895, 0.007570608984678984, 0.03246178478002548, -0.06947032362222672, 0.04558394104242325, 0.03194542974233627, 0.008078036829829216, 0.007779955398291349, -0.06555506587028503, 0.06249631568789482, 0.023188691586256027, -0.06007104739546776, -0.008323372341692448, 0.011942262761294842, 0.00031924142967909575, 0.03410812094807625, 0.028209786862134933, 0.06954393535852432, 0.07761843502521515, 0.02208739146590233, ...]</td>
      <td>[-0.05904030054807663, -0.03966587036848068, 0.046291373670101166, 0.04812397062778473, -0.027457404881715775, 0.022121228277683258, -0.07741555571556091, -0.022804589942097664, 0.01009262353181839, 0.058230020105838776, 0.012752917595207691, 0.009023897349834442, 0.0035175338853150606, 0.05179114267230034, -0.016675885766744614, 0.011633636429905891, -0.027812534943223, -0.040150538086891174, 0.0315440408885479, -0.0005867933505214751, 0.016001634299755096, 0.011057330295443535, -0.03261817246675491, -0.010035021230578423, 0.031010176986455917, 0.011599029414355755, 0.0008746598614379764, -0.020833343267440796, 0.056172192096710205, 0.06400579959154129, -0.0011703965719789267, 0.0009437799453735352, -0.02766498550772667, 0.04365459829568863, 0.024514000862836838, -0.0163163673132658, 0.04071204364299774, 0.027285780757665634, 0.009036600589752197, 0.0014630405930802226, 0.004199848044663668, 0.001103956950828433, -0.03650438413023949, -0.03659118711948395, -0.017628688365221024, -0.04692110791802406, 0.002236389322206378, 0.00445622019469738, 0.04439355432987213, -0.02758820913732052, 0.004827897995710373, 0.09514717757701874, 0.10947562009096146, 0.0905197486281395, -0.016726290807127953, -0.03587675467133522, -0.02286691591143608, -0.017596887424588203, 0.11026337742805481, 0.0387481227517128, -0.01147818099707365, -0.023922229185700417, -0.021128224208950996, 0.028329400345683098, 0.014292297884821892, -0.03179802745580673, -0.12594559788703918, -0.013121319003403187, 0.028817594051361084, 0.05275401100516319, 0.02185330167412758, 0.013401839882135391, -0.05774008855223656, 0.014296409673988819, 0.012512215413153172, 0.07715200632810593, 0.005383542273193598, -0.04241352528333664, 0.005797465797513723, 0.01701459102332592, -0.042707595974206924, 0.017259210348129272, 7.500936044380069e-05, -0.05043512210249901, -0.05295994132757187, -0.015199363231658936, 0.03903813287615776, 0.07273919135332108, -0.028229424729943275, 0.008311056531965733, -0.035439975559711456, 0.0006406796746887267, -0.0015620595077052712, 0.06401623785495758, -0.025534117594361305, 0.0007965399418026209, -0.0032179104164242744, -0.0793919637799263, 0.02272750996053219, -0.011866400018334389, ...]</td>
      <td>[none]</td>
      <td>[-0.06890607625246048, -0.06527753919363022, 0.00474488316103816, -0.07200437784194946, -0.04948535934090614, 0.013415692374110222, 0.04432838037610054, -0.005674577318131924, -0.07965438812971115, -0.01933957263827324, 0.015232820995151997, 0.04530401900410652, 0.06058049201965332, -0.07967492192983627, 0.04787123203277588, 0.06382979452610016, -0.02814880758523941, 0.024407489225268364, -0.028790777549147606, 0.027145180851221085, 0.0835169330239296, 0.014171983115375042, -0.019215896725654602, 0.04409782961010933, 0.009876691736280918, 0.04660492017865181, 0.07268918305635452, -0.08183494210243225, -0.049595560878515244, 0.052024517208337784, 0.043357402086257935, -0.0023100299295037985, -0.026471832767128944, -0.0655069649219513, -0.021977555006742477, -0.052487097680568695, 0.030031796544790268, 0.013984535820782185, -0.03529636561870575, -0.011309203691780567, -0.03611506521701813, 0.06399688124656677, -0.014639611355960369, 0.042340680956840515, 0.017897890880703926, 0.02180793322622776, -0.07109320908784866, 0.07321067899465561, -0.028136219829320908, -0.009222581051290035, 0.045658595860004425, -0.06511558592319489, 0.04158170521259308, 0.06464574486017227, 0.03211952745914459, 0.04162513092160225, 0.04547373577952385, 0.009046979248523712, 0.033821653574705124, -0.07710089534521103, 0.05110042542219162, -0.017872968688607216, 0.011662641540169716, -0.040365807712078094, -0.03649003058671951, -0.03191787749528885, -0.053871799260377884, -0.06054126098752022, -0.015522051602602005, 0.004773661028593779, -0.06383129954338074, 0.06532156467437744, -0.03741082549095154, 0.08606045693159103, 0.008498778566718102, 0.02185351960361004, 0.00486742751672864, -0.010801794938743114, 0.004058252088725567, -0.02209807001054287, 0.06665070354938507, 0.044320039451122284, 0.06096072867512703, -0.04445991292595863, 0.0001674878440098837, -0.057839442044496536, 0.06412035971879959, 0.004358452279120684, 0.0037247496657073498, 0.0364917628467083, 0.0639500543475151, 0.0768292248249054, -0.033808741718530655, 0.031769268214702606, -0.059808406978845596, -0.006878381595015526, 0.03821704164147377, 0.05672299861907959, 0.0057940431870520115, 0.028924213722348213, ...]</td>
    </tr>
    <tr>
      <th>78252</th>
      <td>D6Oy4LfoqsU</td>
      <td>18.14.06</td>
      <td>I GAVE SAFIYA NYGAARD A PERFECT HAIR MAKEOVER BASED ON HER FEATURES: BTS! |bradmondo</td>
      <td>Brad Mondo</td>
      <td>NaN</td>
      <td>2018-05-18T17:34:22.000Z</td>
      <td>I gave safiya nygaard a perfect hair makeover based on her features: bts|"brad mondo"|"safiya and tyler"|"safiya nygaard"|"hair transformation"|"makeover"|"I got a perfect makeover based on my features"|"bts"|"hairdresser reacts"|"before and after"|"hair"|"makeup"|"transformation"|"ANTM"|"what not to wear"|"the ideal haircut and color for your face"|"safiya buzzfeed"|"color for your skin tone"|"haircut for your face shape"|"tutorial"|"balayage"|"hair stylist"|"hair color"|"hair tutorial"</td>
      <td>1066451</td>
      <td>48068</td>
      <td>1032</td>
      <td>...</td>
      <td>775</td>
      <td>111</td>
      <td>0</td>
      <td>0.362581</td>
      <td>15</td>
      <td>1</td>
      <td>[-0.017997004091739655, -0.051434844732284546, 0.05656133592128754, -0.040525585412979126, -0.03240945190191269, -0.005117053631693125, 0.02249749004840851, 0.05801581218838692, -0.01478497963398695, -0.02369924634695053, -0.0351669006049633, 0.07920348644256592, 0.0031386171467602253, 0.04640854895114899, 0.0662703663110733, -0.04356630891561508, -0.004509946331381798, 0.027231544256210327, -0.0073692686855793, -0.01566668227314949, -0.030012020841240883, 0.04967135936021805, 0.006688660476356745, 0.0038547127041965723, 0.03701695054769516, -0.05443023145198822, -0.019318735226988792, 0.0019273923244327307, 0.04794294014573097, 0.0734596997499466, 0.05137232318520546, 0.06204076483845711, -0.007860143668949604, 0.03079802170395851, 0.02363518811762333, 0.03766535595059395, 0.029637720435857773, -0.02071891538798809, -0.07794293761253357, -0.04022327810525894, -0.0005041658878326416, -0.05845768377184868, 0.01693103276193142, 0.0007726628100499511, 0.08349888771772385, -0.058175574988126755, 0.05175330117344856, -0.010425448417663574, 0.0687873512506485, -0.030200719833374023, 0.040840234607458115, -0.08673886209726334, 0.009964103810489178, -0.01755150780081749, -0.0743403434753418, -0.02050899900496006, 0.03965378180146217, 0.03385911136865616, 0.06500114500522614, -0.003585921600461006, -0.05225436016917229, 0.05883392319083214, -0.08588776737451553, 0.013557841069996357, -0.025934645906090736, 0.04667291417717934, 0.03686297684907913, 0.017741398885846138, 0.04222461208701134, -0.05378035455942154, 0.03664292395114899, 0.0445110946893692, 0.016227351501584053, -0.014330116100609303, -0.03377123177051544, -0.0054456009529531, -0.03452055901288986, 0.0457688607275486, -0.0076246666721999645, 0.046889528632164, 0.007604155223816633, 0.09486886113882065, 0.008083527907729149, 0.028815578669309616, -0.0015390361659228802, -0.009568405337631702, 0.030440060421824455, 0.08288165926933289, 0.011202236637473106, -0.006826871540397406, 0.03659041225910187, 0.06229649484157562, 0.02040848694741726, 0.05577074736356735, 0.030217528343200684, -0.06523590534925461, 0.01524685975164175, 0.016518117859959602, -0.026174455881118774, 0.04555497318506241, ...]</td>
      <td>[-0.015779737383127213, -0.023638173937797546, 0.061938658356666565, -0.04531217738986015, -0.044793304055929184, 0.07544543594121933, 0.009773511439561844, -0.0027073491364717484, -0.036846112459897995, 0.01882294937968254, -0.03819310665130615, -0.043420348316431046, 0.018624037504196167, 0.03168131411075592, -0.06662272661924362, -0.052017923444509506, 0.06147656589746475, 0.041035156697034836, -0.029087109491229057, -0.017612682655453682, 0.04153136909008026, -0.04989387467503548, 0.005854757037013769, -0.01176985539495945, -0.005936913657933474, 0.0011312579736113548, -0.013980294577777386, -0.06774824857711792, 0.002196319866925478, 0.06732507795095444, -0.07064113020896912, 0.030424118041992188, 0.06104675680398941, -0.08038116246461868, 0.008219263516366482, 0.061588045209646225, 0.01999610662460327, 0.05962543934583664, 0.04997639358043671, -0.023464374244213104, -0.062613844871521, -0.02040046453475952, -0.01441311277449131, -0.049323614686727524, 0.04249624162912369, 0.022674132138490677, 0.00038173107896000147, 0.06538986414670944, -0.033003069460392, 0.07969614863395691, 0.04610564932227135, 0.04055282473564148, 0.023296106606721878, -0.0690208375453949, 0.046901751309633255, -0.06454647332429886, 0.07482533901929855, 0.004203505348414183, 0.041087422519922256, 0.0066915047354996204, 0.08307303488254547, -0.0204366073012352, -0.08303463459014893, 0.04237411543726921, 0.022047827020287514, 0.06704512238502502, -0.02124827913939953, 0.060952235013246536, -0.06307534128427505, -0.005712941754609346, -0.015528921969234943, 0.05089008808135986, 0.03351714462041855, 0.04728570953011513, 0.0480433851480484, 0.04161694645881653, 0.0523749403655529, -0.004626034758985043, 0.0296102873980999, 0.035852108150720596, -0.00034202897222712636, -0.05773336440324783, -0.060229167342185974, -0.05588453263044357, 0.06798400729894638, 0.04326775670051575, -0.04235665872693062, 0.013578979298472404, 0.010862271301448345, -0.031010625883936882, -0.009783432818949223, -0.0397249273955822, 0.05775132775306702, -0.04368580877780914, 0.022394852712750435, -0.03776960447430611, -0.03303515538573265, 0.03908305987715721, 0.001262756297364831, 0.015289395116269588, ...]</td>
      <td>ANTM, I gave safiya nygaard a perfect hair makeover based on her features: bts, I got a perfect makeover based on my features, balayage, before and after, brad mondo, bts, color for your skin tone, hair, hair color, hair stylist, hair transformation, hair tutorial, haircut for your face shape, hairdresser reacts, makeover, makeup, safiya and tyler, safiya buzzfeed, safiya nygaard, the ideal haircut and color for your face, transformation, tutorial, what not to wear</td>
      <td>[0.02902231365442276, -0.04726419970393181, 0.06168745085597038, -0.03860316053032875, -0.031213462352752686, -0.007660595700144768, 0.015348942019045353, 0.06259454786777496, 0.021763425320386887, 0.01920311339199543, -0.05516306310892105, 0.06298481673002243, -0.011787394993007183, -0.056742534041404724, 0.0071782502345740795, -0.057685572654008865, 0.014010816812515259, -0.012553253211081028, -0.057698093354701996, 0.0178084596991539, -0.004066484980285168, 0.05841166898608208, 0.02185899019241333, 0.009171471931040287, 0.06232905387878418, -0.05997765436768532, -0.05617649853229523, -0.019409596920013428, 0.026364723220467567, 0.060587093234062195, 0.03985673189163208, 0.05432751402258873, -0.019091201946139336, 0.019649101421236992, -0.05677150934934616, 0.05432721972465515, 0.029511943459510803, 0.010816598311066628, -0.006305626127868891, 0.03766431659460068, -0.03754317760467529, -0.053158633410930634, 0.0519489087164402, -0.05425448343157768, 0.06299134343862534, -0.05552997440099716, -0.027156256139278412, -0.055150356143713, 0.05620494484901428, -0.061765603721141815, -0.01659887470304966, 0.02897561341524124, -0.05611240491271019, 0.054567549377679825, -0.06288014352321625, -0.04421564191579819, 0.052784088999032974, 0.060130998492240906, 0.016043664887547493, -0.04428853094577789, -0.05069960281252861, 0.05809597298502922, -0.024052057415246964, 0.05156754329800606, -0.062434788793325424, 0.05358082801103592, 0.03001546673476696, -0.04805425927042961, 0.03039621189236641, 0.020701834931969643, 0.0489230751991272, 0.06231929361820221, 0.05345283821225166, 0.04827145114541054, -0.0033203570637851954, 0.05114613100886345, -0.026806650683283806, 0.05667978525161743, -0.03167564794421196, -0.036358948796987534, 0.01632600650191307, 0.06299147009849548, -0.02745070867240429, 0.028437539935112, 0.021093271672725677, 0.004719029646366835, -0.043484002351760864, 0.05564239248633385, 0.04328474774956703, 0.05567610263824463, -0.021052617579698563, 0.059825342148542404, 0.021266112104058266, 0.023630153387784958, 0.05731221288442612, -0.05564625561237335, 0.03784313052892685, -0.018514376133680344, 0.021690495312213898, 0.02037634328007698, ...]</td>
    </tr>
    <tr>
      <th>78253</th>
      <td>oV0zkMe1K8s</td>
      <td>18.14.06</td>
      <td>How Black Panther Should Have Ended</td>
      <td>How It Should Have Ended</td>
      <td>NaN</td>
      <td>2018-05-17T17:00:04.000Z</td>
      <td>Black Panther|"HISHE"|"Marvel"|"Infinity War"|"How It Should Have Ended"|"parody"|"comedy"|"entertainment"|"wakanda"|"Chadwick Boseman"|"Michael B Jordan"|"movies"|"animation"|"fortnite"|"azerrz"|"movie"|"plothole"|"review"|"childish gambino"|"donald glover"|"this is america"|"ending explained"</td>
      <td>5660813</td>
      <td>192957</td>
      <td>2846</td>
      <td>...</td>
      <td>3268</td>
      <td>444</td>
      <td>0</td>
      <td>0.173501</td>
      <td>51</td>
      <td>0</td>
      <td>[0.0033202494960278273, -0.05254947021603584, 0.024122443050146103, 0.0021337298676371574, 0.05506988614797592, 0.03431621938943863, 0.03227626904845238, 0.005726385395973921, 0.007781274616718292, 0.0073141297325491905, 0.07873424887657166, 0.0432228222489357, -0.03761392831802368, -0.055340252816677094, 0.011576772667467594, -0.04757174849510193, 0.06109345704317093, -0.007580035366117954, 0.038850992918014526, 0.05128810182213783, 0.017932653427124023, -0.06699115037918091, 0.013761677779257298, -0.010028393939137459, 0.05731658264994621, 0.021684205159544945, 0.005537403281778097, 0.0030604207422584295, -0.020773347467184067, 0.08763379603624344, -0.027028996497392654, -0.023631349205970764, -0.017168432474136353, -0.045465752482414246, 0.06277022510766983, -0.05904810130596161, 0.02889716811478138, 0.06463479995727539, -0.038429033011198044, -0.05755572021007538, 0.044589538127183914, -0.027339819818735123, 0.05558070167899132, -0.02450472116470337, -0.07311007380485535, 0.0374005101621151, 0.02180098183453083, -0.04460097476840019, 0.051880355924367905, 0.05850601568818092, 0.08387213200330734, -0.016100164502859116, 0.0053565711714327335, -0.002559565007686615, -0.07253746688365936, 0.08859201520681381, -0.03616686910390854, 0.044176675379276276, 0.026259297505021095, -0.057265009731054306, 0.07216966897249222, -0.0024303814861923456, -0.07733748853206635, -0.08455856889486313, 0.041823435574769974, 0.01587299443781376, -0.06564577668905258, 0.027564682066440582, 0.06234807148575783, 0.03805481642484665, -0.06579948216676712, -0.028011906892061234, 0.048430539667606354, -0.006130808033049107, 0.06231459602713585, 0.016420753672719002, -0.05438760295510292, 0.03798329830169678, -0.03745676577091217, 0.02420624904334545, -0.08126898109912872, 0.07961682230234146, -0.010006899014115334, -0.06726345419883728, -0.01255960762500763, 0.06673956662416458, -0.04541696608066559, -0.01582980714738369, -0.008776489645242691, 0.008207451552152634, -0.011742538772523403, 0.04323819279670715, -0.02430824749171734, 0.020440811291337013, 0.028440013527870178, 0.008549042046070099, 0.005820330698043108, -0.004594997502863407, 0.07636399567127228, 0.020591819658875465, ...]</td>
      <td>[-0.05355843901634216, -0.030057858675718307, -0.030084481462836266, -0.002305559813976288, 0.07380882650613785, -0.09758858382701874, 0.035261157900094986, 0.0509326197206974, 0.05570771545171738, 0.016808416694402695, -0.003219899721443653, -0.05726146325469017, -0.004998377524316311, -0.08047682046890259, 0.07252588868141174, -0.03053739108145237, 0.04671235382556915, 0.0712277814745903, 0.02958875149488449, -0.02503502368927002, 0.03742220252752304, -0.014950891956686974, 0.02207597903907299, 0.0008236722787842155, -0.01777324452996254, 0.02593812718987465, 0.04553934186697006, 0.049302101135253906, -0.021535724401474, 0.01540368515998125, 0.017669081687927246, -0.03366643562912941, 0.03324731066823006, 0.023252515122294426, -0.038532406091690063, -0.029040426015853882, 0.04414821043610573, 0.02349875681102276, -0.03432027995586395, -0.005388923920691013, -0.026553485542535782, -0.014794758521020412, 0.04620802402496338, -0.05621319264173508, -0.030637001618742943, 0.08137738704681396, 0.023006655275821686, 0.005039968527853489, -0.012507306411862373, 0.022005077451467514, 0.08194702863693237, -0.04844105988740921, -0.016810821369290352, -0.02393304742872715, 0.010484467260539532, 0.08509013801813126, -0.065627820789814, -0.03848394379019737, 0.05576162040233612, 0.009370442479848862, 0.006166554056107998, -0.03913893923163414, -0.017981911078095436, -0.11408179253339767, -0.0025215744972229004, 0.033256594091653824, 0.06533467769622803, 0.004383885767310858, 0.03857553377747536, -0.04163694754242897, -0.05953912436962128, -0.013174060732126236, 0.03136279433965683, 0.01992279663681984, 0.002234508516266942, -0.01200830563902855, -0.03504631668329239, 0.02040441520512104, 0.012768052518367767, -0.024378955364227295, 0.020848894491791725, 0.025808081030845642, -0.012390381656587124, -0.0759606584906578, 0.01507214643061161, 0.11206631362438202, -0.08029565215110779, 0.007585954386740923, -0.006044813897460699, -0.03231683373451233, -0.002415318042039871, 0.05113467946648598, -0.02346418984234333, -0.052371952682733536, 0.017807487398386, 0.042951710522174835, -0.011163650080561638, 0.06052504852414131, -0.004906776826828718, -0.03625071421265602, ...]</td>
      <td>Black Panther, Chadwick Boseman, HISHE, How It Should Have Ended, Infinity War, Marvel, Michael B Jordan, animation, azerrz, childish gambino, comedy, donald glover, ending explained, entertainment, fortnite, movie, movies, parody, plothole, review, this is america, wakanda</td>
      <td>[-0.024174178019165993, -0.005242541898041964, 0.04338258132338524, 0.010114754550158978, 0.048471610993146896, 0.02339363470673561, 0.05399193614721298, -0.034790996462106705, 0.02736814133822918, -0.03421317785978317, 0.05782299488782883, 0.06133517250418663, -0.0016234689392149448, -0.025547444820404053, 0.05806874856352806, -0.057697176933288574, 0.052965618669986725, 0.027249906212091446, 0.020745182409882545, 0.046778421849012375, 0.03441935405135155, -0.06515755504369736, -0.02514464780688286, 0.06383317708969116, 0.061725083738565445, 0.056747134774923325, -0.05300178751349449, 0.05852622166275978, -0.053247347474098206, 0.0651717558503151, 0.02414797432720661, -0.035804517567157745, -0.050811462104320526, 0.00017743547505233437, 0.058871276676654816, -0.03271929547190666, 0.04063018783926964, 0.020610466599464417, 0.002601288491860032, -0.015570617280900478, 0.02757376618683338, -0.017326049506664276, -0.016391003504395485, -0.05088238790631294, 0.0612548403441906, 0.020712167024612427, 0.04832911491394043, -0.058927834033966064, 0.03432878479361534, 0.06187573820352554, 0.06409905105829239, -0.04323840141296387, -0.06425007432699203, -0.010422823950648308, -0.06413054466247559, 0.06516668200492859, 0.03513690456748009, 0.0437016636133194, 0.018347112461924553, 0.02047123573720455, 0.06502290815114975, -0.054731495678424835, -0.058886174112558365, -0.018209310248494148, -0.06285654753446579, 0.009334675967693329, -0.01708068512380123, 0.036448899656534195, 0.02219582535326481, 0.021933140233159065, -0.056246258318424225, 0.060562413185834885, 0.003034517401829362, 0.02142641693353653, 0.029924508184194565, 0.004658741410821676, -0.04265344515442848, -0.01870434544980526, -0.021270278841257095, -0.007795972749590874, -0.06327986717224121, 0.05465802922844887, 0.01651286520063877, -0.0643761083483696, -0.007618427742272615, -0.005646157544106245, -0.02090124972164631, -0.06234532967209816, -0.06515678018331528, 0.040386609733104706, -0.05588661506772041, 0.034886207431554794, 0.012688100337982178, 0.0017704891506582499, 0.0651552677154541, -0.027605419978499413, -0.04214956611394882, 0.006296456791460514, 0.0651824027299881, 0.044071391224861145, ...]</td>
    </tr>
    <tr>
      <th>78254</th>
      <td>ooyjaVdt-jA</td>
      <td>18.14.06</td>
      <td>Official Call of Duty®: Black Ops 4 — Multiplayer Reveal Trailer</td>
      <td>Call of Duty</td>
      <td>NaN</td>
      <td>2018-05-17T17:09:38.000Z</td>
      <td>call of duty|"cod"|"activision"|"Black Ops 4"</td>
      <td>10306119</td>
      <td>357079</td>
      <td>212976</td>
      <td>...</td>
      <td>709</td>
      <td>110</td>
      <td>0</td>
      <td>0.042313</td>
      <td>9</td>
      <td>0</td>
      <td>[-0.050984542816877365, 0.011884355917572975, -0.047739945352077484, -0.025676783174276352, 0.021518059074878693, 0.03273758292198181, 0.06805403530597687, 0.0504961796104908, -0.00020403196685947478, -0.02951715886592865, 0.07839362323284149, -0.019051795825362206, 0.06221635639667511, -0.010131672024726868, 0.03827289119362831, 0.036428019404411316, -0.01101321168243885, -0.052768826484680176, -0.008001971989870071, -0.008186355233192444, 0.010886099189519882, 0.04441208764910698, -0.040044933557510376, 0.022863153368234634, 0.03772962465882301, 0.07820262014865875, 0.04864129424095154, -0.05376236140727997, 0.057110272347927094, -0.005990901030600071, -0.03240516409277916, -0.059688448905944824, 0.01849464140832424, -0.05516183376312256, 0.06716994941234589, 0.03356177732348442, -0.02383418194949627, 0.015424231998622417, -0.011436441913247108, 0.0814562514424324, 0.04846053943037987, -0.01741734892129898, 0.02505888231098652, -0.08082149922847748, 0.054423026740550995, -0.04663914814591408, 0.015213312581181526, -0.04597233235836029, -0.06889896839857101, 0.08466659486293793, 0.08526326715946198, -0.07342861592769623, -0.029794391244649887, -0.010063732042908669, -0.07656971365213394, 0.0681249126791954, 0.06860889494419098, 0.03170620650053024, 0.03587879613041878, -0.05424358695745468, -0.05372506007552147, 0.08419831842184067, 0.0328679084777832, -0.012122592888772488, -0.041496407240629196, 0.018183868378400803, 0.02367740496993065, -0.030736323446035385, -0.050253529101610184, 0.06478653103113174, -0.041426870971918106, 0.04264204949140549, -0.011541427113115788, 0.027780622243881226, 0.0034209026489406824, 0.037011291831731796, -0.03965089097619057, -0.035387031733989716, 0.044584859162569046, -0.024409975856542587, -0.02098267897963524, -0.03206561133265495, -0.012149767018854618, 0.06370846927165985, -0.03344472497701645, -0.017593808472156525, -0.00877628568559885, 0.04236283525824547, -0.0502164326608181, 0.0003289694432169199, -0.00758714834228158, 0.06143958866596222, 0.04220648482441902, -0.023139338940382004, -0.003712606616318226, -0.07451385259628296, -0.06215938925743103, 0.05995526164770126, 0.06870774179697037, 0.055037301033735275, ...]</td>
      <td>[0.025727160274982452, -0.004893777891993523, -0.0616651326417923, -0.008745715953409672, -0.021615145727992058, 0.051659971475601196, 0.04313551262021065, -0.04017365351319313, -0.001053496147505939, -0.020533422008156776, 0.08212070167064667, 0.0021775756031274796, -0.004829514306038618, -0.006720295175909996, 0.024486789479851723, -0.03167669475078583, -0.03337690234184265, -0.012974115088582039, -0.00832180306315422, -0.05946950986981392, 0.04754820838570595, 0.025008583441376686, -0.009717267006635666, 0.03505721688270569, 0.04301949590444565, 0.04929612949490547, 0.0034692746121436357, -0.07990334928035736, 0.020707901567220688, -0.029976736754179, -0.027515718713402748, -0.07600509375333786, 0.05311333015561104, -0.023893849924206734, -0.01463974080979824, 0.06458637863397598, 0.06759105622768402, -0.064922995865345, -0.04030143842101097, 0.06295692920684814, -0.03637959435582161, 0.01115812361240387, 0.006455018650740385, -0.04852959141135216, 0.014177083037793636, 0.02786596119403839, -0.00026830218848772347, 0.034309592097997665, 0.010827826336026192, 0.0922117754817009, 0.09712045639753342, 0.02538873255252838, 0.07896114140748978, 0.06204085424542427, -0.08098658174276352, -0.027053648605942726, 0.0566265694797039, 0.01908530853688717, 0.02951979450881481, -0.05310368165373802, -0.008615127764642239, 0.09674684703350067, 0.03850969672203064, -0.0022332393564283848, -0.01236925832927227, 0.04949261620640755, -0.07047608494758606, 0.03575834259390831, 0.023275896906852722, 0.0486702062189579, -0.0421142503619194, -0.02020229585468769, -0.053413327783346176, 0.008297717198729515, 0.008573273196816444, -0.03436623513698578, -0.010686260648071766, -0.0010685096494853497, -0.026745159178972244, -0.07943429052829742, 0.03686367720365524, -0.0008781143696978688, -0.05622435361146927, 0.02777034044265747, -0.06544476002454758, 0.020781278610229492, -0.05506495386362076, 0.05445295199751854, -0.017039364203810692, -0.014409544877707958, -0.005409567151218653, 0.022299079224467278, 0.0023140658158808947, 0.06996132433414459, 0.01865597441792488, -0.0806884840130806, -0.021006431430578232, 0.03320552408695221, 0.04738941043615341, 0.04072137922048569, ...]</td>
      <td>Black Ops 4, activision, call of duty, cod</td>
      <td>[0.024102022871375084, 0.008049331605434418, -0.04056524485349655, -0.04375968500971794, 0.01063491404056549, 0.03791700303554535, 0.056967947632074356, -0.0315619558095932, -0.03218714892864227, 0.013789162039756775, 0.06680687516927719, -0.057278286665678024, 0.024735918268561363, -0.05126243457198143, 0.01583913527429104, 0.05755797028541565, 0.012048746459186077, -0.0640806034207344, -0.007497158832848072, -0.04227304086089134, 0.05588267371058464, 0.0626595988869667, -0.021456297487020493, -0.0037236991338431835, 0.06578610092401505, 0.06316567957401276, 0.05931638181209564, -0.05891456454992294, 0.05826779454946518, -0.03036707267165184, 0.017890088260173798, -0.06905567646026611, 0.03576172515749931, -0.02956530638039112, 0.006897113285958767, 0.06578011810779572, 0.04319078102707863, -0.051533885300159454, -0.005150877870619297, 0.0692862719297409, 0.02518884465098381, -0.0020111864432692528, 0.044444989413022995, -0.002287158276885748, 0.06343206018209457, -0.011998996138572693, 0.03678629547357559, 0.04684576392173767, -0.06047052517533302, 0.07112931460142136, 0.07113348692655563, 0.03451159968972206, -0.0016479861224070191, 0.02908421866595745, -0.07090394198894501, 0.0594642274081707, 0.009843015111982822, 0.054594460874795914, 0.04906851425766945, -0.061439983546733856, -0.0011236515128985047, 0.07109672576189041, 0.06897811591625214, 0.002507971366867423, -0.056498050689697266, -0.03725737705826759, -0.02243783138692379, 0.027541110292077065, -0.06032518297433853, 0.033779535442590714, -0.01598168909549713, 0.0519341416656971, -0.04782065004110336, 0.025197677314281464, 0.03449495509266853, -0.06065436452627182, -0.04466221481561661, -0.022461120039224625, -0.00235886313021183, -0.020911885425448418, 0.03953869268298149, -0.02932238020002842, -0.03248322382569313, 0.05828157439827919, -0.03358275815844536, 0.025439728051424026, -0.005078054964542389, 0.019985416904091835, -0.05986608937382698, -0.04529300332069397, -0.036213234066963196, 0.04433257132768631, 0.04864701256155968, 0.017186518758535385, 0.028333917260169983, -0.06970886141061783, -0.014523976482450962, 0.0617513433098793, 0.06718002259731293, 0.016942624002695084, ...]</td>
    </tr>
  </tbody>
</table>
<p>78255 rows × 37 columns</p>
</div>


