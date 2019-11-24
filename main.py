import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import math

root_path = './'    # 根資料夾

# TF-IDF
max_df = 0.95        # 忽略過高的df
min_df = 5           # 忽略過低的df
smooth_idf = True    # 透過對df + 1 去 Smooth idf 權重
sublinear_tf = True  # tf = 1 + log(tf)

# Rocchio 的可調變數
alpha = 1
beta = 0.75
gamma = 0.15
rel_count = 5   # 用前五篇當作relevant document
nrel_count = 1  # 用最後一篇當作nonrelevant document
iters = 5  # 迭代五次

# SMM 的 參數
alpha_smm = 0.7  # 可調參數alpha
iteration_time = 50  # smm 迭代次數

# 將BGLM.txt轉成dictionary格式
BGLM = {}
with open("./BGLM.txt") as f:
    for line in f:
        (key, val) = line.split()
        BGLM[key] = math.exp(float(val))


# 讀取檔名清單
with open(root_path + 'doc_list.txt') as file:
    doc_list = [line.rstrip() for line in file]

with open(root_path + 'query_list.txt') as file:
    query_list = [line.rstrip() for line in file]
# 把資料處理成 [str, str, ...] 的格式
documents, queries = [], []

for doc_name in doc_list:
    with open(root_path + 'Document/' + doc_name) as file:
        doc = ' '.join([word for line in file.readlines()[3:] for word in line.split()[:-1]])
        documents.append(doc)

for query_name in query_list:
    with open(root_path + 'Query/' + query_name) as file:
        query = ' '.join([word for line in file.readlines() for word in line.split()[:-1]])
        queries.append(query)


# 開始做VSM

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# shape = (2265,4956) -> 2265行(篇文章),4956個不同的word ; 如果沒加toarray是稀疏矩陣的寫法
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# print(doc_tfidfs.shape)
# shape = (800,4956) -> 800行(個query),4956個不同的word
query_vecs = vectorizer.transform(queries).toarray()

# print(query_vecs.shape)
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
# argsort讓他由大到小排，np.flip(axis = 1) 讓他反向, ranking 變成shape(800,2265)
rankings = np.flip(cos_sim.argsort(), axis=1)

rankings = rankings[:, :rel_count]


# 開始做SMM
# 先算c(wi,dj) -> 每個word的TF值
vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)  # 將文本中的詞語轉換為詞頻矩陣
c_widj = vectorizer.fit_transform(documents).toarray()  # 計算個詞語出現的次數   -> shape = (2265,4956)
word = vectorizer.get_feature_names()  # word可以知道第幾個index對應到甚麼字


# 隨機產生Psmm(w)
psmm = np.random.rand(1, 4956)
psmm = psmm / sum(psmm)  # normalize

pbg = np.zeros((1, 4956))  # pbg = p(w|BG)
for words_num, words in enumerate(word):
    pbg[0, words_num] = BGLM[words]

tsmm = np.zeros((1, 4956))

for que_num, query_name in enumerate(query_list):
    # 開始做E-M
    for iter in range(iteration_time):
        tsmm = ((1-alpha_smm) * psmm) / ((1-alpha_smm)*psmm + alpha_smm*pbg)  # shape = (1,4956)
        psmm = np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0) / \
            np.sum(np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0))  # shape = (4956,)

    psmm = np.argsort(-psmm)

    for i in range(60):
        new_query = word[psmm[i]]
        queries[que_num] = queries[que_num] + ' ' + new_query

# -----------------------again vsm + smm--------------------------------

# 開始做VSM

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# shape = (2265,4956) -> 2265行(篇文章),4956個不同的word ; 如果沒加toarray是稀疏矩陣的寫法
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# print(doc_tfidfs.shape)
# shape = (800,4956) -> 800行(個query),4956個不同的word
query_vecs = vectorizer.transform(queries).toarray()

# print(query_vecs.shape)
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
# argsort讓他由大到小排，np.flip(axis = 1) 讓他反向, ranking 變成shape(800,2265)
rankings = np.flip(cos_sim.argsort(), axis=1)

rankings = rankings[:, :rel_count]


# 開始做SMM
# 先算c(wi,dj) -> 每個word的TF值
vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)  # 將文本中的詞語轉換為詞頻矩陣
c_widj = vectorizer.fit_transform(documents).toarray()  # 計算個詞語出現的次數   -> shape = (2265,4956)
word = vectorizer.get_feature_names()  # word可以知道第幾個index對應到甚麼字


# 隨機產生Psmm(w)
psmm = np.random.rand(1, 4956)
psmm = psmm / sum(psmm)  # normalize

pbg = np.zeros((1, 4956))  # pbg = p(w|BG)
for words_num, words in enumerate(word):
    pbg[0, words_num] = BGLM[words]

tsmm = np.zeros((1, 4956))

for que_num, query_name in enumerate(query_list):
    # 開始做E-M
    for iter in range(iteration_time):
        tsmm = ((1-alpha_smm) * psmm) / ((1-alpha_smm)*psmm + alpha_smm*pbg)  # shape = (1,4956)
        psmm = np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0) / \
            np.sum(np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0))  # shape = (4956,)

    psmm = np.argsort(-psmm)

    for i in range(60):
        new_query = word[psmm[i]]
        queries[que_num] = queries[que_num] + ' ' + new_query

# -----------------------again vsm + smm--------------------------------

# 開始做VSM

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# shape = (2265,4956) -> 2265行(篇文章),4956個不同的word ; 如果沒加toarray是稀疏矩陣的寫法
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# print(doc_tfidfs.shape)
# shape = (800,4956) -> 800行(個query),4956個不同的word
query_vecs = vectorizer.transform(queries).toarray()

# print(query_vecs.shape)
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
# argsort讓他由大到小排，np.flip(axis = 1) 讓他反向, ranking 變成shape(800,2265)
rankings = np.flip(cos_sim.argsort(), axis=1)

rankings = rankings[:, :rel_count]


# 開始做SMM
# 先算c(wi,dj) -> 每個word的TF值
vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)  # 將文本中的詞語轉換為詞頻矩陣
c_widj = vectorizer.fit_transform(documents).toarray()  # 計算個詞語出現的次數   -> shape = (2265,4956)
word = vectorizer.get_feature_names()  # word可以知道第幾個index對應到甚麼字


# 隨機產生Psmm(w)
psmm = np.random.rand(1, 4956)
psmm = psmm / sum(psmm)  # normalize

pbg = np.zeros((1, 4956))  # pbg = p(w|BG)
for words_num, words in enumerate(word):
    pbg[0, words_num] = BGLM[words]

tsmm = np.zeros((1, 4956))

for que_num, query_name in enumerate(query_list):
    # 開始做E-M
    for iter in range(iteration_time):
        tsmm = ((1-alpha_smm) * psmm) / ((1-alpha_smm)*psmm + alpha_smm*pbg)  # shape = (1,4956)
        psmm = np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0) / \
            np.sum(np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0))  # shape = (4956,)

    psmm = np.argsort(-psmm)

    for i in range(60):
        new_query = word[psmm[i]]
        queries[que_num] = queries[que_num] + ' ' + new_query

# -----------------------again vsm + smm--------------------------------

# 開始做VSM

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# shape = (2265,4956) -> 2265行(篇文章),4956個不同的word ; 如果沒加toarray是稀疏矩陣的寫法
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# print(doc_tfidfs.shape)
# shape = (800,4956) -> 800行(個query),4956個不同的word
query_vecs = vectorizer.transform(queries).toarray()

# print(query_vecs.shape)
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
# argsort讓他由大到小排，np.flip(axis = 1) 讓他反向, ranking 變成shape(800,2265)
rankings = np.flip(cos_sim.argsort(), axis=1)

rankings = rankings[:, :rel_count]


# 開始做SMM
# 先算c(wi,dj) -> 每個word的TF值
vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)  # 將文本中的詞語轉換為詞頻矩陣
c_widj = vectorizer.fit_transform(documents).toarray()  # 計算個詞語出現的次數   -> shape = (2265,4956)
word = vectorizer.get_feature_names()  # word可以知道第幾個index對應到甚麼字


# 隨機產生Psmm(w)
psmm = np.random.rand(1, 4956)
psmm = psmm / sum(psmm)  # normalize

pbg = np.zeros((1, 4956))  # pbg = p(w|BG)
for words_num, words in enumerate(word):
    pbg[0, words_num] = BGLM[words]

tsmm = np.zeros((1, 4956))

for que_num, query_name in enumerate(query_list):
    # 開始做E-M
    for iter in range(iteration_time):
        tsmm = ((1-alpha_smm) * psmm) / ((1-alpha_smm)*psmm + alpha_smm*pbg)  # shape = (1,4956)
        psmm = np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0) / \
            np.sum(np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0))  # shape = (4956,)

    psmm = np.argsort(-psmm)

    for i in range(60):
        new_query = word[psmm[i]]
        queries[que_num] = queries[que_num] + ' ' + new_query

# -----------------------again vsm + smm--------------------------------

# 開始做VSM

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# shape = (2265,4956) -> 2265行(篇文章),4956個不同的word ; 如果沒加toarray是稀疏矩陣的寫法
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# print(doc_tfidfs.shape)
# shape = (800,4956) -> 800行(個query),4956個不同的word
query_vecs = vectorizer.transform(queries).toarray()

# print(query_vecs.shape)
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
# argsort讓他由大到小排，np.flip(axis = 1) 讓他反向, ranking 變成shape(800,2265)
rankings = np.flip(cos_sim.argsort(), axis=1)

rankings = rankings[:, :rel_count]


# 開始做SMM
# 先算c(wi,dj) -> 每個word的TF值
vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)  # 將文本中的詞語轉換為詞頻矩陣
c_widj = vectorizer.fit_transform(documents).toarray()  # 計算個詞語出現的次數   -> shape = (2265,4956)
word = vectorizer.get_feature_names()  # word可以知道第幾個index對應到甚麼字


# 隨機產生Psmm(w)
psmm = np.random.rand(1, 4956)
psmm = psmm / sum(psmm)  # normalize

pbg = np.zeros((1, 4956))  # pbg = p(w|BG)
for words_num, words in enumerate(word):
    pbg[0, words_num] = BGLM[words]

tsmm = np.zeros((1, 4956))

for que_num, query_name in enumerate(query_list):
    # 開始做E-M
    for iter in range(iteration_time):
        tsmm = ((1-alpha_smm) * psmm) / ((1-alpha_smm)*psmm + alpha_smm*pbg)  # shape = (1,4956)
        psmm = np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0) / \
            np.sum(np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0))  # shape = (4956,)

    psmm = np.argsort(-psmm)

    for i in range(60):
        new_query = word[psmm[i]]
        queries[que_num] = queries[que_num] + ' ' + new_query

# -----------------------again vsm + smm--------------------------------

# 開始做VSM

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# shape = (2265,4956) -> 2265行(篇文章),4956個不同的word ; 如果沒加toarray是稀疏矩陣的寫法
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# print(doc_tfidfs.shape)
# shape = (800,4956) -> 800行(個query),4956個不同的word
query_vecs = vectorizer.transform(queries).toarray()

# print(query_vecs.shape)
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
# argsort讓他由大到小排，np.flip(axis = 1) 讓他反向, ranking 變成shape(800,2265)
rankings = np.flip(cos_sim.argsort(), axis=1)

rankings = rankings[:, :rel_count]


# 開始做SMM
# 先算c(wi,dj) -> 每個word的TF值
vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)  # 將文本中的詞語轉換為詞頻矩陣
c_widj = vectorizer.fit_transform(documents).toarray()  # 計算個詞語出現的次數   -> shape = (2265,4956)
word = vectorizer.get_feature_names()  # word可以知道第幾個index對應到甚麼字


# 隨機產生Psmm(w)
psmm = np.random.rand(1, 4956)
psmm = psmm / sum(psmm)  # normalize

pbg = np.zeros((1, 4956))  # pbg = p(w|BG)
for words_num, words in enumerate(word):
    pbg[0, words_num] = BGLM[words]

tsmm = np.zeros((1, 4956))

for que_num, query_name in enumerate(query_list):
    # 開始做E-M
    for iter in range(iteration_time):
        tsmm = ((1-alpha_smm) * psmm) / ((1-alpha_smm)*psmm + alpha_smm*pbg)  # shape = (1,4956)
        psmm = np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0) / \
            np.sum(np.sum(c_widj[rankings[que_num], :] * tsmm, axis=0))  # shape = (4956,)

    psmm = np.argsort(-psmm)

    for i in range(60):
        new_query = word[psmm[i]]
        queries[que_num] = queries[que_num] + ' ' + new_query

# -----------------------again vsm + smm--------------------------------


# 下面再做一次VSM跟ROCHIO

# 用TF-IDF初始化query向量
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
query_vecs = vectorizer.transform(queries).toarray()

# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
rankings = np.flip(cos_sim.argsort(), axis=1)  # argsort讓他由大到小排，np.flip(axis = 1) 讓他反向

# 用Rocchio算法更新query向量
for _ in range(iters):

    # Update query vectors with Rocchio algorithm
    rel_vecs = doc_tfidfs[rankings[:, :rel_count]].mean(axis=1)
    nrel_vecs = doc_tfidfs[rankings[:, -nrel_count:]].mean(axis=1)
    query_vecs = alpha * query_vecs + beta * rel_vecs - gamma * nrel_vecs

    # Rerank documents based on cosine similarity
    cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
    rankings = np.flip(cos_sim.argsort(), axis=1)  # argsort讓他由大到小排，np.flip(axis = 1) 讓他反向


# 寫入答案
rankings = rankings[:, :50]  # 只取前50
with open('submission.txt', mode='w') as file:
    file.write('Query,RetrievedDocuments\n')
    for query_name, ranking in zip(query_list, rankings):
        ranked_docs = ' '.join([doc_list[idx] for idx in ranking])
        file.write('%s,%s\n' % (query_name, ranked_docs))
