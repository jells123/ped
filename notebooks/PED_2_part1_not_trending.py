# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% id="Qox3_kxb_FRa" colab_type="code" outputId="dd728987-c569-46e2-ea13-7fca3d1e348f" colab={"base_uri": "https://localhost:8080/", "height": 272}
# !pip install imageai

# %% id="wqNfUy3a-FYl" colab_type="code" outputId="26a110dd-22c9-4655-8ab5-68a36ea8dd63" colab={"base_uri": "https://localhost:8080/", "height": 34}
from google.colab import drive
drive.mount('/content/drive')

# %% id="X3vWLvUJ-bsU" colab_type="code" outputId="f0cc4830-8733-4dcf-a65c-ded41004714e" colab={"base_uri": "https://localhost:8080/", "height": 34}
# %tensorflow_version 1.x

# %% id="F2IKQnzr99nn" colab_type="code" outputId="56c6e261-8311-481c-e5e9-401a3ec92de4" colab={"base_uri": "https://localhost:8080/", "height": 578}
import os

DOWNLOAD_IMAGES = False

path_base = "/content/drive/My Drive/data/PED/"
path = f"{path_base}data/"
files = os.listdir(path)
files


# %% id="NK3G2pO099oS" colab_type="code" outputId="f8e16ade-351e-4c93-a5b9-dddf66b80c89" colab={"base_uri": "https://localhost:8080/", "height": 71}
import pandas as pd


pd.set_option("colwidth", -1)

df = pd.read_csv(os.path.join(path, "videos_not_trending.csv")).drop_duplicates().reset_index(drop=True)

# %% id="ft-kfbPw99od" colab_type="code" outputId="e2125051-1c5e-4ec8-ec3e-32e87ac43f7c" colab={"base_uri": "https://localhost:8080/", "height": 102}
df.columns

# %% id="OY95PQes99ov" colab_type="code" colab={}
df.thumbnail_link = df.thumbnail_link.apply(lambda x: x.replace("default.jpg", "hqdefault.jpg"))

# %% id="_vsnCzlm99pD" colab_type="code" outputId="0c816a2c-61d3-44a8-b926-bb1ede8d612a" colab={"base_uri": "https://localhost:8080/", "height": 221}
df.thumbnail_link

# %% id="qNlBsQf899pL" colab_type="code" outputId="a700210b-1e52-405c-9dd1-5cb532c6a701" colab={"base_uri": "https://localhost:8080/", "height": 34}
len(df['thumbnail_link'].unique()), len(df['thumbnail_link'])

# %% id="hP6gbPSF99pg" colab_type="code" outputId="b93df803-9dd5-4e09-d153-a77e203518b4" colab={"base_uri": "https://localhost:8080/", "height": 1000}
df

# %% [markdown] id="RzgAdyll99p1" colab_type="text"
# ## Downloading images

# %% id="wQS8By7i99p2" colab_type="code" colab={}
images_path = os.path.join(path, "images2")
try:
    os.mkdir(images_path)
except FileExistsError:
    pass

def url2filename(url):
    return url.replace("/", "")

# get content and write it to file
def write_to_file(filename, content):
    f = open(filename, 'wb')
    f.write(content)
    f.close()

from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

URLs = df['thumbnail_link'].unique()

import concurrent.futures
import urllib.request

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
def download_urls(urls, images_path, url2filename_func):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(load_url, url, 60): url for url in URLs}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                if not "HTTP Error 404" in str(exc):
                    print('%r generated an exception: %s' % (url, exc))
            else:
                write_to_file(os.path.join(images_path, url2filename_func(url)), data)

if DOWNLOAD_IMAGES:
     download_urls(URLs, images_path, url2filename)
# %% [markdown] id="hrGMTvDa99qN" colab_type="text"
# ## Images preview
# > There are 'missing values' in images as well - some thumbnails are default gray picture with a rectangle in the middle

# %% id="EGkOKb1N99qO" colab_type="code" outputId="ee32c6fb-9dd3-459f-c405-adc3918a335f" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

sample_images = []

fig=plt.figure(figsize=(12, 24))
columns, rows = 4, 10
for i in range(1, columns*rows +1):
    image_path = os.path.join(images_path, os.listdir(images_path)[i*2])
    sample_images.append(image_path)
    
    img = mpimg.imread(image_path)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    
plt.show()

# %% [markdown] id="odrfXNXj99qV" colab_type="text"
# ### Add boolean `has thumbnail` attribute; True or False if image is default thumbnail

# %% id="ifmfOf3X99qW" colab_type="code" colab={}
df["image_filename"] = df["thumbnail_link"].apply(lambda x : x.replace('/', '').replace(':', '_'))

# %% id="KdRisd-199qo" colab_type="code" outputId="ce4cf690-14e0-4ae3-9612-042a797598f3" colab={"base_uri": "https://localhost:8080/", "height": 1000}
is_default_dict = {}
def check_if_image_is_default(thumbnail_link, default_sample):
    return False
    global is_default_dict
    images_path = '../data/images'
    image_filename = thumbnail_link.replace('/', '').replace(':', '_')
    if image_filename in is_default_dict.keys():
        return is_default_dict[image_filename]
    else:
        if os.path.isfile(os.path.join(images_path, image_filename)):
            img = mpimg.imread(os.path.join(images_path, image_filename))
            if img.shape != default_sample.shape:
                img = cv2.resize(img, default_sample.shape[:2][::-1], interpolation = cv2.INTER_AREA)
            diff_from_default = np.sum(np.power((img-default_sample) / 255.0, 2))
            is_default = diff_from_default < 100.0
            is_default_dict[image_filename] = is_default
            return is_default

df["has_thumbnail"] = df["thumbnail_link"].apply(lambda x : True)
default_thumbnails = df[df["has_thumbnail"] == False]["image_filename"].values
df.head(3)

# %% [markdown] id="_C6vQgTy99qv" colab_type="text"
# ### Prepare `df_images` dataframe that will store only UNIQUE links to images, which are not default images

# %% id="GFwB25NO99q8" colab_type="code" outputId="96895f60-9ad3-4c09-dce0-6f3898dd10e6" colab={"base_uri": "https://localhost:8080/", "height": 34}
df_images = pd.DataFrame(
    {"image_filename": np.unique(df.loc[df["has_thumbnail"] == True, "image_filename"].values)}
)
df_images.head(5)
print(df_images.shape)

# %% [markdown] id="g3-od_JQ99rC" colab_type="text"
# ### Load detection models
# > Tested: YOLO, tinyTOLO and Retinanet, however YOLO turned out to yield the best results.
#
# > Sometimes the results could be complementary to each other, but we skipped those detections as they were too costly computationally, while still uncertain if detections are correct AND useful.

# %% id="X5uj41Az99rD" colab_type="code" outputId="ed2a4e75-98b4-4ec0-e9bf-1eacd67773f2" colab={"base_uri": "https://localhost:8080/", "height": 255}
from imageai.Detection import ObjectDetection

def get_yolo_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(f'{path_base}models/yolo.h5')
    detector.loadModel()
    return detector

def get_retinanet_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(f'{path_base}models/resnet50_coco_best_v2.0.1.h5')
    detector.loadModel()
    return detector

yolo_d = get_yolo_detector()
print("yolo loaded")
retinanet_d = get_retinanet_detector()
print("retinanet loaded")

detectors_dict = {
    "yolo" : yolo_d,
    "retinanet": retinanet_d,
}

import cv2

models_path = f"{path_base}models"
face_cascade = cv2.CascadeClassifier(os.path.join(models_path, 'haarcascade_frontalface_default.xml'))
# %% [markdown] id="X-zQpHXn99rJ" colab_type="text"
# ### Test object detection on some sample images
# > Take only TOP 5 detections every time (sorted by probability)
#
# We decreased the `minimum_percentage_probability`, as the images are small and the models have some troubles detecting objects. In some images, all the probabilities would be below 50% (default threshold), while still accurate.

# %% id="77VpNYuB99rL" colab_type="code" outputId="29fb6e2b-5814-4860-e572-e92e40411f25" colab={"base_uri": "https://localhost:8080/", "height": 758}
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# can use output_image_path to save results
for image_path in sample_images[:5]:
    img = mpimg.imread(image_path)
    # resized = resize_image(img, 200) # ? does not seem to work better ...
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    print()
    print(image_path)
    print(type(img))
    
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.show()
    
    for d_name in detectors_dict:
        print(d_name)
        detection = sorted(detectors_dict[d_name].detectObjectsFromImage(
                input_image=img,
                input_type="array",
                minimum_percentage_probability=25,
                # output_image_path="test.png"
                # output_type="array",
            ), key=lambda x : -1*x["percentage_probability"]
        )

        for eachItem in detection[:5]:
            print("\t", eachItem["name"] , " : ", eachItem["percentage_probability"])
            
    detections = face_cascade.detectMultiScale(gray, 1.03, 3)
    print(f"Haar Cascade faces -> number of detections = {len(detections)}")


# %% id="I-TZSjNL99rT" colab_type="code" outputId="c50ea200-b6f3-47ef-9ac6-165238dcd4f6" colab={"base_uri": "https://localhost:8080/", "height": 374}
detectors_dict[d_name].detectObjectsFromImage(
                input_image=img,
                input_type="array",
                minimum_percentage_probability=25,
                output_image_path="test.png"
#                 output_type="array",
            )

# %% [markdown] id="vJM72I7y99rb" colab_type="text"
# ### Perform YOLO detection on all thumbnails
# > `WARNING`: This is taking a lot of time.
# #### Result of running the cell below is written to `yolo_detections.json` file for convenience

# %% id="qO3Fd_8e99rc" colab_type="code" outputId="24a8c693-000a-4e41-c897-096bac0ac6d6" colab={"base_uri": "https://localhost:8080/", "height": 163}
[name for name in os.listdir("../data/images2") if name.startswith("https:i.ytimg.comvi__")]

# %% id="WBQzsBDA99rl" colab_type="code" outputId="aa4d4211-2936-436c-b2c1-9f197bddfa7a" colab={"base_uri": "https://localhost:8080/", "height": 68}
image_filenames = df_images["image_filename"].values
yolo_detections = []

for idx, i in enumerate(image_filenames):
# for idx, i in enumerate(image_filenames[:5]):
    image_path = os.path.join(images_path, i)
    try:
        img = mpimg.imread(image_path.replace("https_", "https:")) # .replace("-", "_"))
        detection = sorted(yolo_d.detectObjectsFromImage(
                input_image=img,
                input_type="array",
                minimum_percentage_probability=25,
                output_image_path="test.png"
            ), key=lambda x : -1*x["percentage_probability"]
        )[:5] # assumption - take top 5 most confident
        classes = [item["name"] for item in detection]
        yolo_detections.append(classes)
    except FileNotFoundError:
        print(f"File {image_path} not found!!!")
    if idx + 1 % 200 == 0:
      print(f'Step {idx + 1}/ {len(image_filenames)} "{i}": {str(classes)},')

# %% id="ZhH4tPcG99rs" colab_type="code" outputId="83a5de5c-7ec6-4d22-cd69-601a9c4d706a" colab={"base_uri": "https://localhost:8080/", "height": 102}
yolo_detections[:5]

# %% [markdown] id="-atSZeWH99ry" colab_type="text"
# ### Load `json` storing YOLO detections
# > What are the most common detections?

# %% id="ei6D2G2m99rz" colab_type="code" colab={}
import json
with open("..\data\yolo_detections_not_trending.json", "r") as handle:
    yolo_detections_dict = json.load(handle)
    
all_detections = []
for key in yolo_detections_dict:
    all_detections.extend(yolo_detections_dict[key])

from collections import Counter
Counter(all_detections).most_common(15)

# %% [markdown] id="CnFrjUUq99r5" colab_type="text"
# What are all possible classes that were found?

# %% id="6XB56eTc99sD" colab_type="code" colab={}
print(set(all_detections), "\n", "number of classes:", len(set(all_detections)))

# %% [markdown] id="I0dPq7mk99sK" colab_type="text"
# #### Which detections are actually useful?
# - Only one class is dominant: 'person'
# - Another one, **TV**, might also be useful, but we should keep in mind that TV seems hard to detect
# - For the remaining ones, we can define custom **groups** of classes, or assign "other" group to the rest
#
# Possible features:
# - if there was any detection at all
# - if there was a detection of person
# - if there was a detection of something else than person
#
# Possible groups:
# - vehicles
# - animals
# - food 

# %% id="hRmOUKYn99sL" colab_type="code" colab={}
vehicles = "bicycle,   car,   motorcycle,   airplane, bus,   train,   truck,   boat "
vehicles = list(map(lambda x : x.strip(), vehicles.split(',')))

animals = " bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra, giraffe  "
animals = list(map(lambda x : x.strip(), animals.split(',')))

food = "  banana,   apple,   sandwich,   orange, broccoli,   carrot,   hot dog,   pizza,   donut,   cake  "
food = list(map(lambda x : x.strip(), food.split(',')))

# %% id="ShKeEP8699sS" colab_type="code" colab={}
df_images["has_detection"] = df_images["image_filename"].apply(lambda x : len(yolo_detections_dict[x]) > 0)
df_images["person_detected"] = df_images["image_filename"].apply(lambda x : "person" in yolo_detections_dict[x])
df_images["object_detected"] = df_images["image_filename"].apply(lambda x : len(yolo_detections_dict[x]) > 0 and any(d != "person" for d in yolo_detections_dict[x]))

df_images["vehicle_detected"] = df_images["image_filename"].apply(lambda x : any(d in vehicles for d in yolo_detections_dict[x]))
df_images["animal_detected"] = df_images["image_filename"].apply(lambda x : any(d in animals for d in yolo_detections_dict[x]))
df_images["food_detected"] = df_images["image_filename"].apply(lambda x : any(d in food for d in yolo_detections_dict[x]))

df_images.head(4)

# %% [markdown] id="V0oBLw7t99sZ" colab_type="text"
# #### How many rows have `True` values in the newly created attributes?
# - any_detection: 81% 
# - person: 71%
# - vehicles: almost 4% ...

# %% id="HMhEoj6f99sa" colab_type="code" colab={}
for cname in df_images.columns:
    if 'detect' in cname:
        print(cname, " : ", df_images[cname].sum(), " : ", f"{df_images[cname].sum() / df_images.shape[0] * 100.0:.2f}%")

# %% [markdown] id="S3Cp4NIA99sy" colab_type="text"
# ### Perform Haar Cascade face detection on all thumbnails

# %% id="6e-fWuP799s0" colab_type="code" outputId="cbed4ea7-f12f-40a7-e5a3-ede52ca04364" colab={"base_uri": "https://localhost:8080/", "height": 561}
image_filenames = df_images["image_filename"].values
face_detections = []

for idx, i in enumerate(image_filenames):
    image_path = os.path.join(images_path, i.replace("https_i", "https:i"))
    try:
      img = mpimg.imread(image_path)
      gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      
      detections = face_cascade.detectMultiScale(gray, 1.03, 3)
      face_detections.append(len(detections))
    except FileNotFoundError:
      pass
    if (idx + 1) % 200 == 0:
        print(f"Step {idx} / {len(image_filenames)}")

# %% id="q9IhBicg99s9" colab_type="code" colab={}
face_detections = np.asarray(face_detections)
n, bins, patches = plt.hist(face_detections, density=True, facecolor='b', alpha=0.75)

plt.grid(True)
plt.show()

# %% [markdown] id="rrjomjyE99tE" colab_type="text"
# The histogram shows that usually no face is detected, however, the algorithm manages to detect a face in some cases. Therefore, we will use it as an additional feature - single number.

# %% [markdown] id="RoTvEdg599tG" colab_type="text"
# #### Add attribute: `face_count` based upon Haar Cascade algorithm

# %% id="Oxr2RXB099tH" colab_type="code" colab={}
fd_dict = dict(zip(image_filenames, face_detections))
df_images["face_count"] = df_images["image_filename"].apply(lambda x : fd_dict[x])
df_images.head(10)

# %% [markdown] id="Mq0eEoqU99tN" colab_type="text"
# ## Images metadata - median pixel values and histograms (`5 bins`)
#
# We can analyze images on lower level than objects/text detection. 
#
# Pixel values can tell about the dominant color, or level of brightness or saturation.
#
# - Basic approach is to take median pixel values for all the images. We can characterize the images overall by analyzing the distribution of this median.
#
#
# - More in-depth approach would create a histogram of pixel values for each image. We don't need to use many bins to be able to tell if an image is rather bright or dark. Values of those 5 bins can be a feature of an image.

# %% [markdown] id="Ta6TQuO999tP" colab_type="text"
# ### A) When converted to grayscale

# %% id="iJZ2cdZL99tQ" colab_type="code" colab={}
img_cache = {}
def get_gray_histogram(image_filename):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray[11:79, :] # remove top and bottom black rectangles
        img_cache[image_filename] = gray
    else:
        gray = img_cache[image_filename]
    hist, _ = np.histogram(gray.flatten(), bins=5)
    return hist / len(gray.flatten())

def get_median_gray_value(image_filename):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray[11:79, :] # remove top and bottom black rectangles
        img_cache[image_filename] = gray
    else:
        gray = img_cache[image_filename]
    return np.median(gray.flatten())

df_images.loc[:, "gray_histogram"] = df_images["image_filename"].apply(get_gray_histogram)
df_images.loc[:, "gray_median"] = df_images["image_filename"].apply(get_median_gray_value)

df_images.head(3)

# %% [markdown] id="nPyjO2Al99tf" colab_type="text"
# #### Median pixel value of an image
#
# Looking at the histogram, we can tell that most images median value is something in the middle of pixel range: 0.0 - 255.0. However, there is a tendency towards darker images - left tail of the histogram is thicker.

# %% id="kyZkTCme99th" colab_type="code" colab={}
gray_medians = df_images.loc[:, "gray_median"].values
BINS = 10
n, bins, patches = plt.hist(gray_medians, edgecolor='black', bins=10)

for i in range(len(patches)):
    patches[i].set_facecolor((i/10.0, i/10.0, i/10.0))

# %% [markdown] id="VxAENj6Y99to" colab_type="text"
# #### More details - plot distributions of each histogram bin separately
# How is the visualization below constructed:
# - for each image, a histogram is computed with 5 bins
#
# - after all histograms are computed, we can separately aggreagate values from first, second ... fifth bin from those histograms, and display those distributions on separate plots (also histograms :))
#
# The histograms somehow reflect the more general plot above. We can see the darker shades domination.
#
# There are not many bright images, while darker images exist. Looking at the lowest pixel range, 0.0-51.0 (where max=255.0) - leftmost plot, a bin with the highest value corresponds to 20%-40%; second highest - 40%-60%. Looking at the next histograms, the leftmost bin, representing 0%-20% range, is getting higher and higher value.

# %% id="qzmI8NYo99tp" colab_type="code" colab={}
gray_histograms = np.stack(df_images.loc[:, "gray_histogram"].values)

fig, axes = plt.subplots(1, 5, figsize=(24, 4))

for i in range(len(axes)):
    axes[i].set_ylim(0, 7500)
    BINS = [i/5 for i in range(5)]
    n, bins, patches = axes[i].hist(gray_histograms[:, i], edgecolor='black', color=str(i / 5), bins=BINS)
    axes[i].set_title(f"pixel values {i/5*255.0} - {(i+1)/5*255.0}")
    axes[i].set_xlim(0.0, 1.0)
    axes[i].set_xticks(BINS)

plt.show()

# %% [markdown] id="ePlQ2mVh99t0" colab_type="text"
# ### B) When converted to HSV
#
# We can perform similar analysis in HSV color scale, as it is way more interpretable than RGB.
#
# HSV representation provides three interesting channels:
# - Hue
# - Saturation
# - Value

# %% id="b571xGb599t2" colab_type="code" colab={}
img_cache = {}
def get_hsv_histogram(image_filename, channel="hue"):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv[11:79, :, :] 
        img_cache[image_filename] = hsv
    else:
        hsv = img_cache[image_filename]
    c = 0 if channel == "hue" else 1 if channel == "saturation" else 2 if channel == "value" else -1
    hist, _ = np.histogram(hsv[:, :, c].flatten(), bins=5)
    return hist / len(hsv[:, :, c].flatten())

def get_median_hsv_value(image_filename, channel="hue"):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv[11:79, :, :] 
        img_cache[image_filename] = hsv
    else:
        hsv = img_cache[image_filename]
    c = 0 if channel == "hue" else 1 if channel == "saturation" else 2 if channel == "value" else -1
    return np.median(hsv[:, :, c].flatten())

df_images.loc[:, "hue_histogram"] = df_images["image_filename"].apply(lambda x : get_hsv_histogram(x, channel="hue"))
df_images.loc[:, "hue_median"] = df_images["image_filename"].apply(lambda x : get_median_hsv_value(x, channel="hue"))

df_images.loc[:, "saturation_histogram"] = df_images["image_filename"].apply(lambda x : get_hsv_histogram(x, channel="saturation"))
df_images.loc[:, "saturation_median"] = df_images["image_filename"].apply(lambda x : get_median_hsv_value(x, channel="saturation"))

df_images.loc[:, "value_histogram"] = df_images["image_filename"].apply(lambda x : get_hsv_histogram(x, channel="value"))
df_images.loc[:, "value_median"] = df_images["image_filename"].apply(lambda x : get_median_hsv_value(x, channel="value"))

df_images.head()

# %% [markdown] id="MFuZnWHa99t9" colab_type="text"
# #### B) a) HUE
#
# We can distinguish two dominant colors looking at the distribution of median HUE value across all the images.
#
# 1. RED shades
# 2. BLUE shades
#
# Green and pink medians are rare.

# %% id="vwtnDJHr99t-" colab_type="code" colab={}
import matplotlib.colors

hue_medians = df_images.loc[:, "hue_median"].values
BINS = 10
n, bins, patches = plt.hist(hue_medians, edgecolor='black', bins=10)

for i in range(len(patches)):
    patches[i].set_facecolor(matplotlib.colors.hsv_to_rgb((i/10, 1.0, 1.0)))

# %% [markdown] id="K6bhq12H99uQ" colab_type="text"
# #### More details - plot distributions of each histogram bin separately
#
# Looking at the detailed histograms, we can observe that:
# - some images dominant colors are shades of RED, ORANGE, and some YELLOW-ORANGE (even up to 80%-100% of the pixels)
# - most images have very little YELLOW-GREEN shades
# - most images have mild addition of GREEN, BLUE and VIOLET; their distribution is very similar (usually 20% of all colors distribution)
#
# <u>On the plot below, the colors do not refer to the bins at all - the bins are colored to represent the color range</u>

# %% id="DU8-EPgl99uR" colab_type="code" colab={}
hue_histograms = np.stack(df_images.loc[:, "hue_histogram"].values)

fig, axes = plt.subplots(1, 5, figsize=(24, 4))

import matplotlib.colors
from matplotlib.pyplot import cm


for i in range(len(axes)):
    axes[i].set_ylim(0, 8000)
    n, bins, patches = axes[i].hist(
        hue_histograms[:, i], 
        edgecolor='black', 
        color=matplotlib.colors.hsv_to_rgb((i/5, 1.0, 1.0)), 
        bins=5,
    )

    for p in range(len(patches)):
        patches[p].set_facecolor(matplotlib.colors.hsv_to_rgb(((i+(p/len(patches)))/5, 1.0, 1.0)))
    
plt.show()

# %% [markdown] id="QCTjAGBE99uc" colab_type="text"
# #### B) b) SATURATION

# %% [markdown] id="dTsmTyA499ud" colab_type="text"
# Analysis of median SATURATION of all images show that most images are not highly saturated. Highest values cummulate below the middle value of saturation available. A histogram has a right tail longer, with reflects the tendency towards mild saturation.

# %% id="l9AXw-Ih99ug" colab_type="code" colab={}
saturation_medians = df_images.loc[:, "saturation_median"].values
BINS = 10
n, bins, patches = plt.hist(saturation_medians, edgecolor='black', bins=10)

for i in range(len(patches)):
    patches[i].set_facecolor(matplotlib.colors.hsv_to_rgb((1.0, i/10.0, 1.0)))

# %% [markdown] id="GGwZpqPG99uo" colab_type="text"
# #### B) c) VALUE
#
# Median of the VALUE seems to be normally distributed. Medium values are the most frequent, and the edge values are the least.

# %% id="PCz0krOH99ut" colab_type="code" colab={}
value_medians = df_images.loc[:, "value_median"].values
BINS = 15
n, bins, patches = plt.hist(value_medians, edgecolor='black', bins=BINS)

for i in range(len(patches)):
    patches[i].set_facecolor(matplotlib.colors.hsv_to_rgb((0.75, 0.75, i/BINS)))


# %% [markdown] id="8j7bmH7j99vH" colab_type="text"
# ### Edge detection 
# We can try to describe DYNAMICS of an image by applying Canny filter.
#
# We add `edges` attribute which is a ratio of edges detected to all pixels in an image. Lower values mean that the image is smooth. Higher values suggest that the image is dynamic and the frequencies change often.

# %% id="goa4D-6D99vJ" colab_type="code" colab={}
def get_edges(image_filename):
    img = mpimg.imread(os.path.join(images_path, image_filename))
    img = img[11:79, :, :]
    edges = cv2.Canny(img, 100, 200) / 255.0
    return np.sum(edges) / edges.size

df_images.loc[:, "edges"] = df_images["image_filename"].apply(get_edges)
df_images.head(3)


# %% [markdown] id="og5HF8oG99vf" colab_type="text"
# It turns out that the distribution of such ratio is pretty normal. Usually the edges ratio is around 20-25%.

# %% id="jiN7xXLu99vg" colab_type="code" colab={}
edges = df_images.loc[:, "edges"].values
BINS = 15
n, bins, patches = plt.hist(edges, edgecolor='black', bins=BINS)

# %% [markdown] id="hhs68AdB99vm" colab_type="text"
# ## Save DF to file

# %% id="bu3-e8Wo99vo" colab_type="code" colab={}
df_images.head()

# %% id="C6i9neQf99v0" colab_type="code" colab={}
df_images.to_csv("..\data\image_attributes.csv", index=False)

# %% [markdown] id="IhPph-b499wU" colab_type="text"
# ### Voila!
