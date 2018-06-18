# -*-coding:utf-8-*-
#https://www.jianshu.com/p/fdde9fc03f94

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

#中文分詞
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

#打印top關鍵詞
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
        for i in topic.argsort()
            [:-n_top_words - 1:-1]]))
    print()


df = pd.read_csv("text/datascience.csv", encoding='gb18030')
#head = df.head()
#print(df.shape)


df["content_cutted"] = df.content.apply(chinese_word_cut)
head=df.content_cutted.head()
print("head",head)


#-向量化
n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(df.content_cutted)


#--主题抽取，設定分類，先设定为5个分类试试。
n_topics = 5
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

#暂定每个主题输出前20个关键词
n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

#data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
#pyLDAvis.show(data)
