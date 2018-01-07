
# coding: utf-8

# In[ ]:

import pandas as pd
import datetime
import re
import string
import pandas as pd
from time import time
import logging
import gensim
from gensim import corpora
import operator
from basic import InitialMethodTextCleaner
from gensim import corpora, models, similarities
import rake
import cx_Oracle
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import query_list
import connection

now = datetime.datetime.now()
date_file = str(now.date())


# In[ ]:

logger = logging.getLogger('prediction_module')
logger.setLevel(logging.DEBUG)

# create a file handler
handler = logging.FileHandler('log/prediction_module_production.log')
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.info("Start at : " + str(datetime.datetime.now()))


# In[ ]:

filename = "result/"
resource_path = "resources/"
dataset_path = "data/"
temp_filename = "temporary/"
model_path = "model/"


# ## Defining Database

# In[ ]:

conn_str = connection.connection


# ## Loading Resources

# In[ ]:

conn = cx_Oracle.connect(conn_str)
c = conn.cursor()
athg_keyword = pd.read_sql_query(query_list.query_athg_keywords, conn)
future_keyword = pd.read_sql_query(query_list.query_future_keywords, conn)
entity_dict = pd.read_sql_query(query_list.query_entity_dict, conn)
stopwords = pd.read_sql_query(query_list.query_stopwords, conn)
c.close()
conn.close()
athg_keyword.columns = ['keyword']
future_keyword.columns = ['keyword']
stopwords.columns = ['keyword']


# In[ ]:

# transforming keyword
athg_keyword.loc[:,'keyword'] = ' ' + athg_keyword.keyword +  ' '
pattern = '|'.join(athg_keyword.keyword)


# ## Data Collection from File

# In[ ]:

logger.info("Collecting data : " + str(datetime.datetime.now()))


# In[ ]:

conn = cx_Oracle.connect(conn_str)
c = conn.cursor()
dataset = pd.read_sql_query(query_list.query_document_news, conn)
entity = pd.read_sql_query(query_list.query_document_entity, conn)
c.close()
conn.close()


# In[ ]:

logger.info("Collecting dataset | " + str(len(dataset)))
logger.info("Collecting data entity | " + str(len(entity)))


# In[ ]:

dataset['TANGGAL'] = pd.to_datetime(dataset['TANGGAL'])


# ## FUNCTION

# In[ ]:

def identify_month(word):
    if any(x in word for x in jan):
        return str(1)
    elif any(x in word for x in feb):
        return str(2)
    elif any(x in word for x in mar):
        return str(3)
    elif any(x in word for x in apr):
        return str(4)
    elif any(x in word for x in may):
        return str(5)
    elif any(x in word for x in jun):
        return str(6)
    elif any(x in word for x in jul):
        return str(7)
    elif any(x in word for x in aug):
        return str(8)
    elif any(x in word for x in sep):
        return str(9)
    elif any(x in word for x in octo):
        return str(10)
    elif any(x in word for x in nov):
        return str(11)
    elif any(x in word for x in dec):
        return str(12)
    
def identify_date(sentence):
    sentence = ' '+sentence +' '
    if any(x in sentence for x in months):
        words = sentence.split(' ')
        for i in range(len(words)):
            word = ' '+words[i]+' '
            month = identify_month(word)
            if any(x in word for x in months):
                temp_plus_1 = re.sub(r'[^0-9]', '',words[i+1])
                if(i > 0):
                    if(i < len(words)-1):
                        if(temp_plus_1.isdigit()):
                            if(len(temp_plus_1) > 2):
                                if(words[i-1].isdigit()):
                                    if(check_date(temp_plus_1,month,words[i-1])):
                                        return  month+ '/' + words[i-1]+'/'+temp_plus_1
                                elif(re.search(r'[0-9]',words[i-1]) and re.search(r'-',words[i-1])):
                                    temp = words[i-1].split('-')
                                    if(check_date(temp_plus_1,month,temp[1])):
                                        return  month + '/' +temp[1]+'/'+ temp_plus_1
                                else:
                                    if(check_date(temp_plus_1,month,'1')):
                                        return  month + '/' +'1'+'/'+ temp_plus_1
                            elif(len(temp_plus_1) < 4):
                                if(i+2 <= len(words)-1):
                                    day = temp_plus_1
                                    year = words[i+2]
                                    if(year.isdigit()):
                                        if(check_date(year,month,day)):
                                            return month + '/' +day+'/'+ year
                                    else:
                                        return ''
                                else:
                                    return ''
                        else: 
                            if(re.search('tahun lalu', sentence)):
                                year = now.year - 1
                            else:
                                year = now.year
                            if(words[i-1].isdigit()):
                                if(check_date(year,month,words[i-1])):
                                    return  month + '/' + words[i-1]+'/'+ str(year)
                            elif(re.search(r'[0-9]',words[i-1]) and re.search(r'-',words[i-1])):
                                temp = words[i-1].split('-')
                                if(check_date(year,month,temp[1])):
                                    return  month + '/' + temp[1]+'/'+ str(year)
                            else:
                                if(check_date(year,month,'1')):
                                    return  month + '/' +'1'+'/'+ str(year)
                    else:
                        temp_minus_1 = re.sub(r'[^0-9]', '',words[i-1])
                        year = now.year
                        if(temp_minus_1.isdigit()):
                            if(check_date(year,month,temp_minus_1)):
                                return month + '/' +temp_minus_1+'/'+ str(year)
                        elif(re.search(r'[0-9]',words[i-1]) and re.search(r'-',words[i-1])):
                            temp = words[i-1].split('-')
                            if(check_date(year,month,temp[1])):
                                return month + '/' +temp[1]+'/'+ str(year)
                        else:
                            if(check_date(year,month,'1')):
                                return month + '/' +'1'+'/'+ str(year)
    elif any(x in sentence for x in days):
        words = sentence.split(' ')
        for i in range(len(words)):
            word = ' '+words[i]+' '
            if any(x in word for x in days):
                if(i < len(words)-1):
                    if(words[i+1].count('/') == 2):
                        if(re.search(r'[0-9]',words[i+1])):
                            array = words[i+1].split('/')
                            month = re.sub(r'[^0-9]', '',array[1])
                            day = re.sub(r'[^0-9]', '',array[0])
                            year = re.sub(r'[^0-9]', '',array[2])
                            temp = month + '/' +day+'/'+ year
                            if(check_date(year,month, day)):
                                return temp
                    elif(words[i+1].count('/') == 1):
                        year = now.year
                        array = words[i+1].split('/')
                        month = re.sub(r'[^0-9]', '',array[1])
                        day = re.sub(r'[^0-9]', '',array[0])
                        temp = month + '/' +day+'/'+ year
                        if(check_date(year,month, day)):
                            return temp
    return ''

def identify_sentence(data):
    id = []
    set_sentences = []
    list_norm = []
    type = []
    salience = []
    mood = []
    polarity = []
    intensity = []
    aspect = []
    for index, row in data.iterrows():
        sentences = row['lower_content'].split('.')
        norm = row['VALUE']
        t_norm = re.sub(r'[^0-9a-zA-Z ]', ' ',norm)
        for i in range(len(sentences)):
            sentence = ' '+sentences[i]+' '
            t_sentence = re.sub(r'[^0-9a-zA-Z ]', ' ',sentence)
            if t_sentence.find(' '+t_norm+' ')>0:
                if(i> 0):
                    if(i < len(sentences)-1):
                        sentence = sentences[i-1] + '.'+ sentence + '.'+sentences[i+1]
                    else:
                        sentence = sentences[i-1]+ sentence
                else:
                    if(i < len(sentences)-1):
                        sentence = sentence + '.'+sentences[i+1]     
                set_sentences.append(sentence)
                id.append(row['ID'])
                list_norm.append(norm)
                type.append(row['TYPE'])
                salience.append(row['SALIENCE'])
                mood.append(row['MOOD'])
                intensity.append(row['INTENSITY'])
                aspect.append(row['ASPECT'])
                polarity.append(row['POLARITY'])
                break 
    df = pd.DataFrame({'ID':id,'type':type,'norm':list_norm,'sentence':set_sentences,'SALIENCE':salience,
                      'MOOD':mood,'INTENSITY':intensity,'POLARITY':polarity,'ASPECT':aspect})  
    return df

def filter_data_by_keyword(data):
    topic = []
    id = []
    norm = []
    sentences = []
    date = []
    type = []
    keywords = []
    for index, row in topic_org.iterrows():
        topic_id = row['TOPIC_ID']
        sentence = row['SENTENCE']
        sentence = re.sub(r'[^0-9a-zA-Z ]', ' ',sentence)
        keyword = athg_topic_keyword[athg_topic_keyword.TOPIC_ID ==topic_id].KEYWORD
        if any(x in sentence for x in keyword):
            topic.append(topic_id)
            id.append(row['ID'])
            norm.append(row['NORM'])
            sentences.append(row['SENTENCE'])
            type.append(row['TYPE'])
            keywords.append(keyword)
    df = pd.DataFrame({'TOPIC_ID':topic,'ID':id,'NORM':norm,'SENTENCE':sentences,'TYPE':type,'KEYWORD':keywords})  
    return df

def check_date(year, month, day):
    correctDate = None
    try:
        newDate = datetime.datetime(int(year), int(month), int(day))
        correctDate = True
    except ValueError:
        correctDate = False
    return correctDate

def get_athg_event(sentence):
    events = []
    for index, row in athg_keyword.iterrows():
        word = row['keyword']
        if (sentence.count(word) > 0):
            events.append(word)
    if(len(events) > 0):
        event = max(events, key=len)
    else:
        event = ''
    return event


# ## ATHG Categorization

# In[ ]:

logger.info("ATHG Categorization | " + str(datetime.datetime.now()))


# In[ ]:

dataset['TANGGAL'] = pd.to_datetime(dataset['TANGGAL'])
dataset = dataset[(dataset.CONTENT.notnull()) & (dataset.TITLE.notnull())]
dataset = dataset[dataset.SCORE.notnull()]
athg_dataset = dataset[(dataset.SCORE > 0.6)]
athg_dataset = athg_dataset[(athg_dataset.N_KEYWORDS_TITLE != athg_dataset.N_UNIQE_KEYWORDS_TITLE) | 
                            (athg_dataset.N_KEYWORDS_CONTENT != athg_dataset.N_UNIQE_KEYWORDS_CONTENT)]
athg_dataset = athg_dataset.drop(['N_KEYWORDS_TITLE','N_UNIQE_KEYWORDS_TITLE','N_KEYWORDS_CONTENT','N_UNIQE_KEYWORDS_CONTENT',
                        'TITLE_SCORE','CONTENT_SCORE','KEYWORDS_INTENSITY'], axis=1)
non_athg_dataset = dataset[~dataset.ID.isin(athg_dataset.ID)]
non_athg_dataset = non_athg_dataset.drop(['N_KEYWORDS_TITLE','N_UNIQE_KEYWORDS_TITLE','N_KEYWORDS_CONTENT','N_UNIQE_KEYWORDS_CONTENT',
                        'TITLE_SCORE','CONTENT_SCORE','KEYWORDS_INTENSITY'], axis=1)
athg_dataset.to_csv(filename+date_file+"_summary_athg.csv", index = False)
non_athg_dataset.to_csv(filename+date_file+"_summary_non_athg.csv", index = False)


# In[ ]:

logger.info("Finished ATHG Categorization | " + str(datetime.datetime.now()))
logger.info("ATHG Data: "+ str(len(athg_dataset)))
logger.info("Non ATHG Data: "+ str(len(non_athg_dataset)))


# ## Data Cleansing

# In[ ]:

# cleaning athg_dataset
athg_dataset.loc[:,'clean_content'] = athg_dataset.CONTENT.str.replace('[^0-9a-zA-Z.!?]', ' ')
athg_dataset.loc[:,'clean_title'] = athg_dataset.TITLE.str.replace('[^0-9a-zA-Z.!?]', ' ')
athg_dataset.loc[:,'lower_content'] = athg_dataset.clean_content.str.lower()
athg_dataset.loc[:,'lower_title'] = athg_dataset.clean_title.str.lower()
athg_dataset.loc[:,'final_content'] = athg_dataset.lower_content.str.replace('[^0-9a-zA-Z]', ' ')
athg_dataset.loc[:,'final_title'] = athg_dataset.lower_title.str.replace('[^0-9a-zA-Z]', ' ')

entity.loc[:,'clean_content'] = entity.CONTENT.str.replace('[^0-9a-zA-Z.!?]', ' ')
entity.loc[:,'clean_title'] = entity.TITLE.str.replace('[^0-9a-zA-Z.!?]', ' ')
entity.loc[:,'lower_content'] = entity.clean_content.str.lower()
entity.loc[:,'lower_title'] = entity.clean_title.str.lower()
entity.loc[:,'final_content'] = entity.lower_content.str.replace('[^0-9a-zA-Z]', ' ')
entity.loc[:,'final_title'] = entity.lower_title.str.replace('[^0-9a-zA-Z]', ' ')


# ## Date Entity Extraction

# In[ ]:

logger.info("Entity Extraction to Sentence | "+str(datetime.datetime.now()))


# In[ ]:

jan = [ " januari " , " january " , " jan " ]
feb = [ " februari " , " february " , " feb " , " febuari " ]
mar = [ " maret " , " march " , " mar " ]
apr = [ " april " , " apr " ]
may = [ " mei " , " may " ]
jun = [ " juni " , " june " , " jun " ]
jul = [ " juli " , " july " , " jul " ]
aug = [ " agustus " , " august " , " aug " , " agsts " ]
sep = [ " september " , " sept " , " sep " ]
octo = [ " oktober " , " okt " , " october " , " oct " ]
nov = [ " november " , " nov " , " nopember " ]
dec = [ " december " , " desember " , " dec " , " des " ]
months = jan + feb + mar+ apr+may+jun+jul+aug+sep+octo+nov+dec
days = [ " senin " ,  " selasa " ,  " rabu " ,  " kamis " ,  " jumat " ,  " sabtu "  ,  " minggu " ,  " ahad " ]
year = [" 2017 ", " 2018 "," 2019 "]
date_keyword = months + days + year


# In[ ]:

id = []
set_sentences = []
identified_date = []
date = []
athg_topics = []
entity_type = []
for index, row in athg_dataset.iterrows():
    sentences = row['lower_content'].split('.')
    for i in range(len(sentences)):
        sentence = sentences[i]
        if any(x in sentence for x in date_keyword):
            athg_topic = get_athg_event(sentence)
            id.append(row['ID'])
            if(i> 0):
                if(i < len(sentences)-1):
                    sentence = sentences[i-1] + '.'+ sentence + '.'+sentences[i+1]
                else:
                    sentence = sentences[i-1]+ sentence
            else:
                if(i < len(sentences)-1):
                    sentence = sentence + '.'+sentences[i+1]
            date.append(row['TANGGAL'])      
            set_sentences.append(sentence)
            date_identify = identify_date(sentence)
            identified_date.append(date_identify)
            athg_topics.append(athg_topic)
            entity_type.append('TIMESTAMP')
date_entity = pd.DataFrame({'ID':id,'DATE':date,'SENTENCE':set_sentences,'IDENTIFIED_DATE':identified_date,'EVENT':athg_topics,'TYPE':entity_type}) 
date_entity = date_entity[date_entity.IDENTIFIED_DATE.notnull()]
date_entity['IDENTIFIED_DATE'] = pd.to_datetime(date_entity['IDENTIFIED_DATE'])
date_entity['DATE'] = pd.to_datetime(date_entity['DATE'])

logger.info("Date Entity Size | "+str(len(date_entity)))

filtered_date = date_entity[date_entity.IDENTIFIED_DATE > date_entity.DATE]
logger.info("Filtered Date Size | "+str(len(filtered_date)))


filtered_date = filtered_date[['ID','IDENTIFIED_DATE','SENTENCE','TYPE']].copy()
filtered_date.columns = ['ID','NORM','SENTENCE','TYPE']

# ## Future Event Extraction

# In[ ]:

logger.info("Future Event Extraction | "+str(datetime.datetime.now()))
# transforming keyword
future_keyword.loc[:,'keyword'] = ' ' + future_keyword.keyword +  ' '
future_pattern = '|'.join(future_keyword.keyword)


# In[ ]:

future_dataset = athg_dataset[(athg_dataset.final_title.str.count(future_pattern) >1) | 
                       (athg_dataset.final_content.str.count(future_pattern)  > 1)]
future_dataset = future_dataset[(~future_dataset.final_title.str.contains(' pahlawan ')) & 
                                (~future_dataset.final_content.str.contains(' pahlawan '))]
future_dataset = future_dataset[~future_dataset.ID.isin(filtered_date.ID)]
non_future_dataset = athg_dataset[(~athg_dataset.ID.isin(future_dataset.ID)) 
                                  & (~athg_dataset.ID.isin(filtered_date.ID))]


# In[ ]:

if len(future_dataset) == 0 and len(filtered_date) == 0:
    logger.info("Future dataset are empty | "+str(datetime.datetime.now()))
    import if_empty
    quit()


# In[ ]:

id = []
set_sentences = []
identified_event= []
athg_topics = []
identified_date = []
entity_type = []
for index, row in future_dataset.iterrows():
    sentences = row['lower_content'].split('.')
    for i in range(len(sentences)):
        sentence = sentences[i]
        if any(x in sentence for x in future_keyword.keyword):
            athg_topic = get_athg_event(sentence)
            id.append(row['ID'])
            if(i> 0):
                if(i < len(sentences)-1):
                    sentence = sentences[i-1] + '.'+ sentence + '.'+sentences[i+1]
                else:
                    sentence = sentences[i-1]+ sentence
            else:
                if(i < len(sentences)-1):
                    sentence = sentence + '.'+sentences[i+1]
            date.append(row['TANGGAL'])      
            set_sentences.append(sentence)
            athg_topics.append(athg_topic)
            identified_date.append('')
            entity_type.append('TIMESTAMP')
future_sentence = pd.DataFrame({'ID':id,'SENTENCE':set_sentences,'EVENT':athg_topics,'IDENTIFIED_DATE':identified_date,'TYPE':entity_type})
future_sentence = future_sentence[['ID','IDENTIFIED_DATE','SENTENCE','TYPE']].copy()
future_sentence.columns = ['ID','NORM','SENTENCE','TYPE']
logger.info("Future Event Size | "+str(len(future_sentence)))

# In[ ]:

entity_future = entity[(entity.ID.isin(future_dataset.ID))]


# ## Entity Sentence Extraction

# In[ ]:

# loading the resources
translation_map = pd.read_csv(resource_path+"map_translation.csv", sep = ',')
translation_map.loc[:,'translated'] = ' ' + translation_map.translated +  ' '
translation_map.loc[:,'original'] = ' ' + translation_map.original +  ' '
exclude_org = pd.read_csv(resource_path+"exclude_org.csv", sep = ',')
exclude_place = pd.read_csv(resource_path+"exclude_place.csv", sep = ',')
exclude_person = pd.read_csv(resource_path+"exclude_person.csv", sep = ',')


# In[ ]:

entity_athg = entity[(entity.ID.isin(athg_dataset.ID))]


# In[ ]:

entity_org = entity_athg[(entity_athg.TYPE == 'ORG') & (~entity_athg.NORM.isin(exclude_org.norm))]
entity_place = entity_athg[(entity_athg.TYPE == 'PLACE') & (~entity_athg.NORM.isin(exclude_place.norm))]
entity_person = entity_athg[(entity_athg.TYPE == 'PERSON') & (~entity_athg.NORM.isin(exclude_person.norm))]


# In[ ]:

df_org = identify_sentence(entity_org) 
df_person = identify_sentence(entity_person) 
df_place = identify_sentence(entity_place) 


# ## TOPIC EXTRACTION

# In[ ]:

logger.info("Topic Extraction | "+str(datetime.datetime.now()))


# In[ ]:

text_cleaner=InitialMethodTextCleaner()
#stopwords = []
stoppath = "smartstoplist_indonesia.txt"


# In[ ]:

def get_stopword(filename):
    with open(filename, 'r') as line:
        for word in line:
            word = word.rstrip()
            stopwords.append(word)
    return

def cleaning(article):
    one = text_cleaner.clean_text(article)
    three = " ".join([i for i in str(one).lower().split() if i not in stopwords.keyword])
    return three


# Get Topic Cluster

# In[ ]:

punctuation = set(string.punctuation)

start = time()
Lda = gensim.models.ldamodel.LdaModel

df = athg_dataset[['ID', 'TITLE', 'CONTENT', 'clean_content','clean_title']].copy()
text = df.applymap(cleaning)['clean_content']
text_list = [i.split() for i in text]
dictionary = corpora.Dictionary(text_list)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]

from pathlib import Path

my_file = Path(model_path+"lda.model")
if my_file.is_file():
    #this will load previous model and update it
    logger.info("load previous model | "+str(datetime.datetime.now()))
    ldamodel = models.LdaModel.load(model_path+'lda.model')
    ldamodel.update(doc_term_matrix)
else:
    #this will create new model
    logger.info("create new model | "+str(datetime.datetime.now()))
    ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary, passes=50)

ldamodel.save(model_path+'lda.model')


# In[ ]:

topic_id = []
for doc in text_list:
    bow = dictionary.doc2bow(doc)
    topics = ldamodel[bow]
    topikterpilih,score =  max(topics, key=lambda item: item[1])
    topic_id.append(topikterpilih)

df.insert(0,column='TOPIC_ID', value=topic_id)
grouped = df.groupby('TOPIC_ID')['clean_content'].apply(lambda x: " ".join(x))


# Get keyword from topic (dump)

# In[ ]:

rake_object = rake.Rake(stoppath, 3, 3, 4)
topic_keyword = []
index = 0
dfkeyword = pd.DataFrame(columns=['TOPIC_ID', 'KEYWORD', 'SCORE'])
for index_val, series_val in grouped.iteritems():
    keywords1 = rake_object.run(series_val)
    totalKeywords = len(keywords1)
    for keyword,score in keywords1[0:int(totalKeywords)]:
        dfkeyword.loc[index] = [index_val, keyword, score]
        index = index + 1


# ## Prediction Module

# In[ ]:

logger.info("Prediction Module | "+str(datetime.datetime.now()))


# In[ ]:

athg_topic_docid = df[['TOPIC_ID','ID']].copy()


# In[ ]:

topic_org = pd.merge(athg_topic_docid, df_org, on='ID', how='left')
topic_org = pd.merge(athg_topic_docid, df_org, on='ID', how='left')
topic_org.columns = ['TOPIC_ID','ID','ASPECT', 'INTENSITY', 'MOOD', 'POLARITY','SALIENCE','NORM','SENTENCE','TYPE']
topic_per = pd.merge(athg_topic_docid, df_person, on='ID', how='left')
topic_per.columns = topic_org.columns
topic_place = pd.merge(athg_topic_docid, df_place, on='ID', how='left')
topic_place.columns = topic_org.columns
topic_date = pd.merge(athg_topic_docid, filtered_date, on='ID', how='left')
topic_future = pd.merge(athg_topic_docid, future_sentence, on='ID', how='left')
topic_future = topic_date.append(topic_future)
topic_future.columns = ['TOPIC_ID','ID','NORM','SENTENCE','TYPE']
topic_future['ASPECT'] = 0
topic_future['INTENSITY'] = 0
topic_future['MOOD'] = 0
topic_future['POLARITY'] = 0
topic_future['SALIENCE'] = 50

# appending the data
topic_org = topic_org.append(topic_per)
topic_org = topic_org.append(topic_place)
topic_org = topic_org[['TOPIC_ID','ID','NORM','SENTENCE','TYPE','ASPECT', 'INTENSITY', 'MOOD', 'POLARITY','SALIENCE']]
topic_org = topic_org.append(topic_future)
topic_org = topic_org[topic_org.NORM.isnull() == False]


# In[ ]:

df_date = topic_org[topic_org.ID.isin(filtered_date.ID)]
df_future = topic_org[topic_org.ID.isin(future_dataset.ID)]
df_future = df_future[~df_future.TOPIC_ID.isin(df_date.TOPIC_ID)]


# In[ ]:

new_group = pd.DataFrame(topic_org.groupby('TOPIC_ID')['ID'].nunique().reset_index())
std= new_group.ID.std()
mean = new_group.ID.mean()
new_group['value'] = (new_group.ID - mean)/std
new_group.loc[new_group.value> 2,'value'] = 2
new_group.loc[new_group.value< -2,'value'] = -2
new_group['value'] = new_group['value']/4+0.5


# ## ENTITY CALCULATION

# In[ ]:

logger.info("Score Calculation Module | "+str(datetime.datetime.now()))


# In[ ]:

entity_dict.ENTITY = entity_dict.ENTITY.str.lower()


# In[ ]:

score_data = []
for index, row in df_date.iterrows():
    score = 0
    norm = row['NORM']
    normtype = row['TYPE']
    listentity = entity_dict['ENTITY']
    if (normtype != 'TIMESTAMP'):
        if any(x in norm for x in listentity):
            class_pejabat = entity_dict[entity_dict.ENTITY == norm].CLASS
            if any(x in 'PEJABAT' for x in class_pejabat):
                score = 3
            elif any(x in 'TOKOH' for x in class_pejabat):
                score = 2
            elif any(x in 'LISTED' for x in class_pejabat):
                score = 2
        else:
            score = 1
    else:
        score = 0
    score_data.append(score)
df_date.loc[:,'SCORE_ENTITY'] = score_data


# In[ ]:

score_data = []
for index, row in df_future.iterrows():
    score = 0
    norm = row['NORM']
    normtype = row['TYPE']
    listentity = entity_dict['ENTITY']
    if (normtype != 'TIMESTAMP'):
        if any(x in norm for x in listentity):
            class_pejabat = entity_dict[entity_dict.ENTITY == norm].CLASS
            if any(x in 'PEJABAT' for x in class_pejabat):
                score = 3
            elif any(x in 'TOKOH' for x in class_pejabat):
                score = 2
            elif any(x in 'LISTED' for x in class_pejabat):
                score = 2
        else:
            score = 1
    else:
        score = 0
    score_data.append(score)
df_future['SCORE_ENTITY'] = score_data


# In[ ]:

df_date.loc[:,'COM_SCORE'] = (df_date.SCORE_ENTITY *0.6)+(df_date.INTENSITY.astype(float) *0.4)
max_date = pd.DataFrame(df_date[df_date.TYPE == 'TIMESTAMP'].groupby(['TOPIC_ID'])[["NORM"]].max().reset_index())
new_group2 = pd.DataFrame(df_date.groupby(['TOPIC_ID'])[["COM_SCORE"]].sum().reset_index())
 
std= new_group2.COM_SCORE.std()
mean = new_group2.COM_SCORE.mean()

new_group2['NORM_SCORE'] = (new_group2.COM_SCORE - mean)/std
new_group2.loc[new_group2.NORM_SCORE > 2,'NORM_SCORE'] = 2
new_group2.loc[new_group2.NORM_SCORE < -2,'NORM_SCORE'] = -2
new_group2['NORM_SCORE'] = new_group2['NORM_SCORE']/4+0.5

new_group2_date = pd.merge(new_group2, max_date, on = 'TOPIC_ID', how='left')
new_group_3 = pd.merge(new_group2_date, new_group, on = 'TOPIC_ID', how='left')
new_group_3['FINAL_SCORE'] = (new_group_3.NORM_SCORE + new_group_3.value)/2

sentence_date = pd.DataFrame(df_date.groupby(['TOPIC_ID'])[["SENTENCE"]].sum().reset_index())
new_group_3 = pd.merge(new_group_3, sentence_date, on = 'TOPIC_ID', how='left')


# In[ ]:

df_future['COM_SCORE'] = (df_future.SCORE_ENTITY *0.6)+(df_future.INTENSITY.astype(float)*0.4)
max_date_future = pd.DataFrame(df_future[df_future.TYPE == 'TIMESTAMP'].groupby(['TOPIC_ID'])[["NORM"]].max().reset_index())
new_group2_future = pd.DataFrame(df_future.groupby(['TOPIC_ID'])[["COM_SCORE"]].sum().reset_index())
 
std_f = new_group2_future.COM_SCORE.std()
mean_f = new_group2_future.COM_SCORE.mean()

new_group2_future['NORM_SCORE'] = (new_group2_future.COM_SCORE - mean)/std
new_group2_future.loc[new_group2_future.NORM_SCORE > 2,'NORM_SCORE'] = 2
new_group2_future.loc[new_group2_future.NORM_SCORE < -2,'NORM_SCORE'] = -2
new_group2_future['NORM_SCORE'] = new_group2_future['NORM_SCORE']/4+0.5

new_group2_f_date = pd.merge(new_group2_future, max_date_future, on = 'TOPIC_ID', how='left')
new_group_3_future = pd.merge(new_group2_f_date, new_group, on = 'TOPIC_ID', how='left')
new_group_3_future['FINAL_SCORE'] = (new_group_3_future.NORM_SCORE + new_group_3_future.value)/2

sentence_date = pd.DataFrame(df_future.groupby(['TOPIC_ID'])[["SENTENCE"]].sum().reset_index())
new_group_3_future = pd.merge(new_group_3_future, sentence_date, on = 'TOPIC_ID', how='left')


# In[ ]:

rake_object = rake.Rake(stoppath, 3, 3, 3)

topic_keyword = []
index = 0

dfkeyword_date = pd.DataFrame(columns=['TOPIC_ID', 'KEYWORD', 'SCORE'])

for idx, row in new_group_3.iterrows():
    keywords1 = rake_object.run(row['SENTENCE'])
    totalKeywords = len(keywords1)
    for keyword,score in keywords1[0:int(totalKeywords)]:
        dfkeyword_date.loc[index] = [row['TOPIC_ID'], keyword, score]
        index = index + 1

dfkeyword_date.head(1)


index2 = 0
dfkeyword_future = pd.DataFrame(columns=['TOPIC_ID', 'KEYWORD', 'SCORE'])

for idx, row in new_group_3_future.iterrows():
    keywords1 = rake_object.run(row['SENTENCE'])
    totalKeywords = len(keywords1)
    for keyword,score in keywords1[0:int(totalKeywords)]:
        dfkeyword_future.loc[index2] = [row['TOPIC_ID'], keyword, score]
        index2 = index2 + 1


# Get list keyword from topic

# In[ ]:

top_keyword_date = dfkeyword_date.groupby('TOPIC_ID').head(10)
top_keyword_list_date = pd.DataFrame(top_keyword_date.groupby('TOPIC_ID')['KEYWORD'].apply(lambda x: "%s" % ', '.join(x)).reset_index())

top_keyword_future = dfkeyword_future.groupby('TOPIC_ID').head(10)
top_keyword_list_future = pd.DataFrame(top_keyword_future.groupby('TOPIC_ID')['KEYWORD'].apply(lambda x: "%s" % ', '.join(x)).reset_index())


# In[ ]:

date_sentence = df_date[(df_date.TYPE == 'TIMESTAMP') & (df_date.NORM != '-')]
date_sentence = date_sentence[['TOPIC_ID','NORM','SENTENCE']].copy()
date_sentence.columns = ['TOPIC_ID','DATE','SENTENCE']

print date_sentence

#date_sentence.DATE = date_sentence.DATE.dt.date
date_sentence['DATE'] = pd.to_datetime(date_sentence['DATE'])
date_sentence.DATE = date_sentence.DATE.dt.date
date_sentence.DATE = date_sentence.DATE.astype(str)


# In[ ]:

topic_future_sentence = df_future[(df_future.TYPE == 'TIMESTAMP')]
topic_future_sentence = topic_future_sentence[['TOPIC_ID','NORM','SENTENCE']].copy()
topic_future_sentence.columns = ['TOPIC_ID','DATE','SENTENCE']


# In[ ]:

def filter_keyword(row):
    keyword = row['KEYWORD_1'].split(',')
    sentence = row['SENTENCE'].split('.')
    # main sentence
    for i in range(len(keyword)):
        keyword[i] = ' '.join(keyword[i].split())
        if(len(sentence) == 3):
            if (sentence[1].count(' '+keyword[i]+' ') > 0):
                return keyword[i]
        else:
            if (sentence[0].count(' '+keyword[i]+' ') > 0):
                return keyword[i]
    for i in range(len(keyword)):
        keyword[i] = ' '.join(keyword[i].split())
        words = keyword[i].split(' ')
        for j in range(len(words)):
            if(len(sentence) == 3):
                if (sentence[1].count(' '+words[j]+' ') > 0):
                    return keyword[i]
            else:
                if (sentence[0].count(' '+words[j]+' ') > 0):
                    return keyword[i]
    # first sentence
    for i in range(len(keyword)):
        keyword[i] = ' '.join(keyword[i].split())
        if(len(sentence) == 3):
            if (sentence[0].count(' '+keyword[i]+' ') > 0):
                return keyword[i]
        else:
            if (sentence[1].count(' '+keyword[i]+' ') > 0):
                return keyword[i]
    for i in range(len(keyword)):
        keyword[i] = ' '.join(keyword[i].split())
        words = keyword[i].split(' ')
        for j in range(len(words)):
            if(len(sentence) == 3):
                if (sentence[0].count(' '+words[j]+' ') > 0):
                    return keyword[i]
            else:
                if (sentence[1].count(' '+words[j]+' ') > 0):
                    return keyword[i]
    # third sentence
    for i in range(len(keyword)):
        keyword[i] = ' '.join(keyword[i].split())
        if(len(sentence) == 3):
            if (sentence[2].count(' '+keyword[i]+' ') > 0):
                return keyword[i]
        else:
            if (sentence[1].count(' '+keyword[i]+' ') > 0):
                return keyword[i]
    for i in range(len(keyword)):
        keyword[i] = ' '.join(keyword[i].split())
        words = keyword[i].split(' ')
        for j in range(len(words)):
            if(len(sentence) == 3):
                if (sentence[2].count(' '+words[j]+' ') > 0):
                    return keyword[i]
            else:
                if (sentence[1].count(' '+words[j]+' ') > 0):
                    return keyword[i]
    return ''

def get_athg_keyword(row):
    sentence = re.sub(r'[.]',' ',row['SENTENCE'])
    event = get_athg_event(sentence)
    return event

def get_keyword(row):
    if (row['KEYWORD_3'] != ''):
        return row['KEYWORD_3']
    else:
        return row['KEYWORD_2']


# In[ ]:

prediction_prob = new_group_3[['TOPIC_ID','NORM','FINAL_SCORE']].copy()
prediction_prob.columns = ['TOPIC_ID','DATE','POTENTIAL_VALUE']
prediction_prob.DATE = prediction_prob.DATE.astype(str)
prediction_prob = pd.merge(prediction_prob,top_keyword_list_date, on = 'TOPIC_ID')
prediction_prob = pd.merge(prediction_prob, date_sentence, on = ['TOPIC_ID','DATE'])
prediction_prob = prediction_prob[['TOPIC_ID','DATE','KEYWORD','SENTENCE','POTENTIAL_VALUE']].copy()
prediction_prob.columns = ['TOPIC_ID','DATE','KEYWORD_1','SENTENCE','POTENTIAL_VALUE']
prediction_prob = prediction_prob.drop_duplicates(subset =['TOPIC_ID','DATE'])


# In[ ]:

prediction_prob_f = new_group_3_future[['TOPIC_ID','NORM','FINAL_SCORE']].copy()
prediction_prob_f.columns = ['TOPIC_ID','DATE','POTENTIAL_VALUE']
prediction_prob_f = pd.merge(prediction_prob_f, top_keyword_list_future, on = 'TOPIC_ID')
prediction_prob_f = pd.merge(prediction_prob_f, topic_future_sentence, on = ['TOPIC_ID','DATE'])
prediction_prob_f = prediction_prob_f[['TOPIC_ID','DATE','KEYWORD','SENTENCE','POTENTIAL_VALUE']].copy()
prediction_prob_f.columns = ['TOPIC_ID','DATE','KEYWORD_1','SENTENCE','POTENTIAL_VALUE']
prediction_prob_f = prediction_prob_f.drop_duplicates(subset =['TOPIC_ID'])
prediction_prob = prediction_prob.append(prediction_prob_f)
prediction_prob['KEYWORD_2'] = prediction_prob.apply(filter_keyword, axis=1)
prediction_prob['KEYWORD_3'] = prediction_prob.apply(get_athg_keyword, axis=1)
prediction_prob['KEYWORD'] = prediction_prob.apply(get_keyword, axis=1)
del prediction_prob['KEYWORD_1']
del prediction_prob['KEYWORD_2']
del prediction_prob['KEYWORD_3']
prediction_prob = prediction_prob[['TOPIC_ID','DATE','KEYWORD','SENTENCE','POTENTIAL_VALUE']].copy()
prediction_prob.columns = ['ID','DATE','KEYWORD','SENTENCE','POTENTIAL_VALUE']


# In[ ]:



prediction_entity = df_date[['TOPIC_ID','NORM','TYPE']].copy()
prediction_entity.columns = ['PREDICTION_PROB_ID','NORM','ENTITY']
prediction_entity = prediction_entity[prediction_entity['ENTITY'] != 'TIMESTAMP']
prediction_entity_f = df_future[['TOPIC_ID','NORM','TYPE']].copy()
prediction_entity_f.columns = ['PREDICTION_PROB_ID','NORM','ENTITY']
prediction_entity_f = prediction_entity_f[prediction_entity_f['ENTITY'] != 'TIMESTAMP']
prediction_entity = prediction_entity.append(prediction_entity_f)
prediction_entity = prediction_entity.drop_duplicates(subset =['PREDICTION_PROB_ID','NORM','ENTITY'])
prediction_entity['ID'] = prediction_entity.index
prediction_entity = prediction_entity[['ID','PREDICTION_PROB_ID','NORM','ENTITY']].copy()


# In[ ]:

prediction_entity_sources = df_date[['TOPIC_ID','ID']].copy()
prediction_entity_sources.columns = ['TOPIC_ID','SOURCE_ID']
prediction_entity_sources = prediction_entity_sources.drop_duplicates(subset =['TOPIC_ID','SOURCE_ID'])
prediction_entity_sources_f = df_future[['TOPIC_ID','ID']].copy()
prediction_entity_sources_f.columns = ['TOPIC_ID','SOURCE_ID']
prediction_entity_sources_f = prediction_entity_sources_f.drop_duplicates(subset =['TOPIC_ID','SOURCE_ID'])
prediction_entity_sources = prediction_entity_sources.append(prediction_entity_sources_f)
temp = athg_dataset[['ID','URL']].copy()
temp.columns = ['SOURCE_ID','URL']
prediction_entity_sources = pd.merge(prediction_entity_sources,temp, on = 'SOURCE_ID', how = 'left')
prediction_entity_sources = prediction_entity_sources.drop_duplicates(subset =['TOPIC_ID','SOURCE_ID','URL'])
prediction_entity_sources = prediction_entity_sources[['TOPIC_ID','SOURCE_ID','URL']].copy()
prediction_entity_sources.columns = ['PREDICTION_PROB_ID','SOURCE_ID','URL']


# In[ ]:
logger.info("Prediction Event Size | "+str(len(prediction_prob)))
prediction_prob.to_csv(filename+date_file+"_prediction_prob.csv", index = False)
prediction_entity_sources.to_csv(filename+date_file+"_prediction_entity_sources.csv", index = False)
prediction_entity.to_csv(filename+date_file+"_prediction_entity.csv", index = False)

# In[ ]:

if len(prediction_prob) == 0:
    logger.info("No data need to be inserted | "+str(datetime.datetime.now()))
    quit()

# In[ ]:

logger.info("Insert into database | "+str(datetime.datetime.now()))


# In[ ]:

conn = cx_Oracle.connect(conn_str)
c = conn.cursor()
db_pp = pd.read_sql_query(query_list.query_pp, conn)
db_pe = pd.read_sql_query(query_list.query_pe, conn)
db_ps = pd.read_sql_query(query_list.query_ps, conn)
c.close()
conn.close()


# In[ ]:

db_pp = db_pp[['ID','PREDICTION_DATE','KEYWORD','SENTENCE','POTENTIAL_VALUE']].copy()
db_pe = db_pe[['ID','PREDICTION_PROB_ID','NORM','ENTITY']].copy()
db_ps = db_ps[['PREDICTION_PROB_ID','SOURCE_ID','URL']].copy()

db_pp.columns = ['id','prediction_date','keyword','sentence','potential_value']
db_pe.columns = ['id','prediction_prob_id','norm','entity']
db_ps.columns = ['prediction_prob_id','source_id','url']


# In[ ]:

db_pp = db_pp[db_pp.id.isin(prediction_prob.ID)]
db_pe = db_pe[db_pe.prediction_prob_id.isin(prediction_entity.PREDICTION_PROB_ID)]
db_ps = db_ps[db_ps.prediction_prob_id.isin(prediction_entity_sources.PREDICTION_PROB_ID)]


# In[ ]:

# append two datafame
prediction_prob.loc[prediction_prob.POTENTIAL_VALUE.isnull(),'POTENTIAL_VALUE'] = 0
prediction_prob.columns = db_pp.columns
prediction_entity.columns = db_pe.columns
prediction_entity_sources.columns = db_ps.columns


# In[ ]:

db_pp_1 = db_pp.append(prediction_prob)
temp_1 = db_pp_1.groupby('id').aggregate({'potential_value':'mean'}).reset_index()


# In[ ]:

db_pp_notnull = db_pp[(db_pp.prediction_date.notnull()) & (db_pp.prediction_date != '')]
db_pp_null = db_pp[(db_pp.prediction_date.isnull()) | (db_pp.prediction_date == '')]
prediction_prob_1 = prediction_prob[prediction_prob.id.isin(db_pp_notnull.id)]
prediction_prob_1 = prediction_prob_1[(prediction_prob_1.prediction_date.isnull()) | (prediction_prob_1.prediction_date == '') ]
prediction_prob_null = prediction_prob[prediction_prob.id.isin(db_pp_null.id)]
prediction_prob_null = prediction_prob_null[(prediction_prob_null.prediction_date.isnull()) | (prediction_prob_null.prediction_date == '') ]
prediction_prob = prediction_prob[(~prediction_prob.id.isin(prediction_prob_1.id)) & (~prediction_prob.id.isin(prediction_prob_null.id))]


# In[ ]:
db_pp = db_pp[~db_pp.id.isin(prediction_prob_null.id)]
if len(prediction_prob) > 0:
	db_pp = db_pp.append(prediction_prob)
	db_pe = db_pe.append(prediction_entity)
	db_ps = db_ps.append(prediction_entity_sources)
	db_pe = db_pe.drop_duplicates(subset =['prediction_prob_id','norm','entity'])
	db_ps = db_ps.drop_duplicates(subset =['prediction_prob_id','source_id','url'])
	temp = db_pp.groupby('id').aggregate({'prediction_date': 'max','keyword':'last','sentence':'last'}).reset_index()
else:
	temp = db_pp[['id','prediction_date','keyword','sentence']].copy()
	
prediction_prob_null = prediction_prob_null[['id','prediction_date','keyword','sentence']].copy()
temp = temp.append(prediction_prob_null)


# In[ ]:
df_final = pd.merge(temp,temp_1, on='id', how='left')
df_final.loc[df_final.prediction_date.isnull(),'prediction_date'] = '' 

# In[ ]:

conn = cx_Oracle.connect(conn_str)
c = conn.cursor()
# delete from prediction_prob, prediction_entity, prediction_entity_sources where id is in our df_final
for index, row in df_final.iterrows():
    id = row['id']
    c.execute("DELETE FROM PREDICTION_PROB where ID = "+ str(id))
    c.execute("DELETE FROM PREDICTION_ENTITY where PREDICTION_PROB_ID = "+ str(id))
    c.execute("DELETE FROM PREDICTION_ENTITY_SOURCES where PREDICTION_PROB_ID = "+ str(id))

# insert prediction prob
list_pp =df_final.values.tolist() 
c_1 = conn.cursor()
c_1.prepare('insert into PREDICTION_PROB (ID, PREDICTION_DATE, KEYWORD, SENTENCE,POTENTIAL_VALUE) values (:1, :2, :3, :4, :5)')
c_1.executemany(None, list_pp)

# insert prediction_entity
list_pe =db_pe.values.tolist() 
c_2 = conn.cursor()
c_2.prepare('insert into PREDICTION_ENTITY (ID,PREDICTION_PROB_ID,NORM,ENTITY) values (:1, :2, :3, :4)')
c_2.executemany(None, list_pe)

# insert prediction entity sources
list_ps =db_ps.values.tolist() 
c_3 = conn.cursor()
c_3.prepare('insert into PREDICTION_ENTITY_SOURCES (PREDICTION_PROB_ID,SOURCE_ID,URL) values (:1, :2, :3)')
c_3.executemany(None, list_ps)
conn.commit()

c.close()
c_1.close()
c_2.close()
c_3.close()
conn.close()


# In[ ]:

logger.info("Finished process | "+str(datetime.datetime.now()))

