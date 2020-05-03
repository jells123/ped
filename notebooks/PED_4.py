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

# +
X_all = normalized_df.values
# MUST use -1 here! It tells semisup-learn library that those are unlabeled records.
y_all = list(map(int, agg_df.fillna(-1).loc[:, "category_id"].values))

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

# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X_all, y_all)
print("self-learning log.reg. score", ssmodel.score(X, y))
# -

pd.read_csv('aggregated.csv')


# +
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

from sklearn.base import BaseEstimator
import numpy
import sklearn.metrics
from sklearn.linear_model import LogisticRegression as LR
import nlopt
import scipy.stats

class CPLELearningModel(BaseEstimator):
    """
    Contrastive Pessimistic Likelihood Estimation framework for semi-supervised 
    learning, based on (Loog, 2015). This implementation contains two 
    significant differences to (Loog, 2015):
    - the discriminative likelihood p(y|X), instead of the generative 
    likelihood p(X), is used for optimization
    - apart from `pessimism' (the assumption that the true labels of the 
    unlabeled instances are as adversarial to the likelihood as possible), the 
    optimization objective also tries to increase the likelihood on the labeled
    examples
    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then uses global optimization to 
    find (soft) label hypotheses for the unlabeled examples in a pessimistic  
    fashion (such that the model log likelihood on the unlabeled data is as  
    small as possible, but the log likelihood on the labeled data is as high 
    as possible)
    See Loog, Marco. "Contrastive Pessimistic Likelihood Estimation for 
    Semi-Supervised Classification." arXiv preprint arXiv:1503.00269 (2015).
    http://arxiv.org/pdf/1503.00269
    Attributes
    ----------
    basemodel : BaseEstimator instance
        Base classifier to be trained on the partially supervised data
    pessimistic : boolean, optional (default=True)
        Whether the label hypotheses for the unlabeled instances should be
        pessimistic (i.e. minimize log likelihood) or optimistic (i.e. 
        maximize log likelihood).
        Pessimistic label hypotheses ensure safety (i.e. the semi-supervised
        solution will not be worse than a model trained on the purely 
        supervised instances)
        
    predict_from_probabilities : boolean, optional (default=False)
        The prediction is calculated from the probabilities if this is True 
        (1 if more likely than the mean predicted probability or 0 otherwise).
        If it is false, the normal base model predictions are used.
        This only affects the predict function. Warning: only set to true if 
        predict will be called with a substantial number of data points
        
    use_sample_weighting : boolean, optional (default=True)
        Whether to use sample weights (soft labels) for the unlabeled instances.
        Setting this to False allows the use of base classifiers which do not
        support sample weights (but might slow down the optimization)
    max_iter : int, optional (default=3000)
        Maximum number of iterations
        
    verbose : int, optional (default=1)
        Enable verbose output (1 shows progress, 2 shows the detailed log 
        likelihood at every iteration).
    """
    
    def __init__(self, basemodel, pessimistic=True, predict_from_probabilities = False, use_sample_weighting = True, max_iter=3000, verbose = 1):
        self.model = basemodel
        self.pessimistic = pessimistic
        self.predict_from_probabilities = predict_from_probabilities
        self.use_sample_weighting = use_sample_weighting
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.it = 0 # iteration counter
        self.noimprovementsince = 0 # log likelihood hasn't improved since this number of iterations
        self.maxnoimprovementsince = 3 # threshold for iterations without improvements (convergence is assumed when this is reached)
        
        self.buffersize = 200
        # buffer for the last few discriminative likelihoods (used to check for convergence)
        self.lastdls = [0]*self.buffersize
        
        # best discriminative likelihood and corresponding soft labels; updated during training
        self.bestdl = numpy.infty
        self.bestlbls = []
        
        # unique id
        self.id = str(chr(numpy.random.randint(26)+97))+str(chr(numpy.random.randint(26)+97))

    def discriminative_likelihood(self, model, labeledData, labeledy = None, unlabeledData = None, unlabeledWeights = None, unlabeledlambda = 1, gradient=[], alpha = 0.01):
        unlabeledy = (unlabeledWeights[:, 0]<0.5)*1
        uweights = numpy.copy(unlabeledWeights[:, 0]) # large prob. for k=0 instances, small prob. for k=1 instances 
        uweights[unlabeledy==1] = 1-uweights[unlabeledy==1] # subtract from 1 for k=1 instances to reflect confidence
        weights = numpy.hstack((numpy.ones(len(labeledy)), uweights))
        labels = numpy.hstack((labeledy, unlabeledy))
        
        # fit model on supervised data
        if self.use_sample_weighting:
            model.fit(numpy.vstack((labeledData, unlabeledData)), labels, sample_weight=weights)
        else:
            model.fit(numpy.vstack((labeledData, unlabeledData)), labels)
        
        # probability of labeled data
        P = model.predict_proba(labeledData)
        
        try:
            # labeled discriminative log likelihood
            labeledDL = -sklearn.metrics.log_loss(labeledy, P)
        except Exception as e:
            print(e)
            P = model.predict_proba(labeledData)

        # probability of unlabeled data
        unlabeledP = model.predict_proba(unlabeledData)  
           
        try:
            # unlabeled discriminative log likelihood
            eps = 1e-15
            unlabeledP = numpy.clip(unlabeledP, eps, 1 - eps)
            unlabeledDL = numpy.average((unlabeledWeights*numpy.vstack((1-unlabeledy, unlabeledy)).T*numpy.log(unlabeledP)).sum(axis=1))
        except Exception as e:
            print(e)
            unlabeledP = model.predict_proba(unlabeledData)
        
        if self.pessimistic:
            # pessimistic: minimize the difference between unlabeled and labeled discriminative likelihood (assume worst case for unknown true labels)
            dl = unlabeledlambda * unlabeledDL - labeledDL
        else: 
            # optimistic: minimize negative total discriminative likelihood (i.e. maximize likelihood) 
            dl = - unlabeledlambda * unlabeledDL - labeledDL
        
        return dl
        
    def discriminative_likelihood_objective(self, model, labeledData, labeledy = None, unlabeledData = None, unlabeledWeights = None, unlabeledlambda = 1, gradient=[], alpha = 0.01):
        if self.it == 0:
            self.lastdls = [0]*self.buffersize
        
        dl = self.discriminative_likelihood(model, labeledData, labeledy, unlabeledData, unlabeledWeights, unlabeledlambda, gradient, alpha)
        
        self.it += 1
        self.lastdls[numpy.mod(self.it, len(self.lastdls))] = dl
        
        if numpy.mod(self.it, self.buffersize) == 0: # or True:
            improvement = numpy.mean((self.lastdls[(len(self.lastdls)/2):])) - numpy.mean((self.lastdls[:(len(self.lastdls)/2)]))
            # ttest - test for hypothesis that the likelihoods have not changed (i.e. there has been no improvement, and we are close to convergence) 
            _, prob = scipy.stats.ttest_ind(self.lastdls[(len(self.lastdls)/2):], self.lastdls[:(len(self.lastdls)/2)])
            
            # if improvement is not certain accoring to t-test...
            noimprovement = prob > 0.1 and numpy.mean(self.lastdls[(len(self.lastdls)/2):]) < numpy.mean(self.lastdls[:(len(self.lastdls)/2)])
            if noimprovement:
                self.noimprovementsince += 1
                if self.noimprovementsince >= self.maxnoimprovementsince:
                    # no improvement since a while - converged; exit
                    self.noimprovementsince = 0
                    raise Exception(" converged.") # we need to raise an exception to get NLopt to stop before exceeding the iteration budget
            else:
                self.noimprovementsince = 0
            
            if self.verbose == 2:
                print(self.id,self.it, dl, numpy.mean(self.lastdls), improvement, round(prob, 3), (prob < 0.1))
            elif self.verbose:
                sys.stdout.write(('.' if self.pessimistic else '.') if not noimprovement else 'n')
                      
        if dl < self.bestdl:
            self.bestdl = dl
            self.bestlbls = numpy.copy(unlabeledWeights[:, 0])
                        
        return dl
    
    def fit(self, X, y): # -1 for unlabeled
        unlabeledX = X[y==-1, :]
        labeledX = X[y!=-1, :]
        labeledy = y[y!=-1]
        
        M = unlabeledX.shape[0]
        
        # train on labeled data
        self.model.fit(labeledX, labeledy)

        unlabeledy = self.predict(unlabeledX)
        
        #re-train, labeling unlabeled instances pessimistically
        
        # pessimistic soft labels ('weights') q for unlabelled points, q=P(k=0|Xu)
        f = lambda softlabels, grad=[]: self.discriminative_likelihood_objective(
            self.model, labeledX, labeledy=labeledy, unlabeledData=unlabeledX, 
            unlabeledWeights=numpy.vstack((softlabels, 1-softlabels)).T, gradient=grad
        ) #- supLL
        lblinit = numpy.random.random(len(unlabeledy))

        try:
            self.it = 0
            opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, M)
            opt.set_lower_bounds(numpy.zeros(M))
            opt.set_upper_bounds(numpy.ones(M))
            opt.set_min_objective(f)
            opt.set_maxeval(self.max_iter)
            self.bestsoftlbl = opt.optimize(lblinit)
            print(" max_iter exceeded.")
        except Exception as e:
            print(e)
            self.bestsoftlbl = self.bestlbls
            
        if numpy.any(self.bestsoftlbl != self.bestlbls):
            self.bestsoftlbl = self.bestlbls
        print(self.bestsoftlbl)
        ll = f(self.bestsoftlbl)

        unlabeledy = (self.bestsoftlbl<0.5)*1
        uweights = numpy.copy(self.bestsoftlbl) # large prob. for k=0 instances, small prob. for k=1 instances 
        uweights[unlabeledy==1] = 1-uweights[unlabeledy==1] # subtract from 1 for k=1 instances to reflect confidence
        weights = numpy.hstack((numpy.ones(len(labeledy)), uweights))
        labels = numpy.hstack((labeledy, unlabeledy))
        if self.use_sample_weighting:
            self.model.fit(numpy.vstack((labeledX, unlabeledX)), labels, sample_weight=weights)
        else:
            self.model.fit(numpy.vstack((labeledX, unlabeledX)), labels)
        
        if self.verbose > 1:
            print("number of non-one soft labels: ", numpy.sum(self.bestsoftlbl != 1), ", balance:", numpy.sum(self.bestsoftlbl<0.5), " / ", len(self.bestsoftlbl))
            print("current likelihood: ", ll)
        
        if not getattr(self.model, "predict_proba", None):
            # Platt scaling
            self.plattlr = LR()
            preds = self.model.predict(labeledX)
            self.plattlr.fit( preds.reshape( -1, 1 ), labeledy )
            
        return self
        
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.
        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        if getattr(self.model, "predict_proba", None):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
            return self.plattlr.predict_proba(preds.reshape( -1, 1 ))
        
    def predict(self, X):
        """Perform classification on samples in X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        
        if self.predict_from_probabilities:
            P = self.predict_proba(X)
            return (P[:, 0]<numpy.average(P[:, 0]))
        else:
            return self.model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
# -


