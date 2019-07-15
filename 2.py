import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfbook = pd.read_csv('books.csv')
dfrating = pd.read_csv('ratings.csv')

def all_features(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])
dfbook['criteria'] = dfbook.apply(all_features,axis='columns')
# print(dfbook.head())

# Count Vectorizer -------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(tokenizer=lambda x:x.split(' '))
matrixFeature = model.fit_transform(dfbook['criteria'])

criteria = model.get_feature_names()

# Cosine Similarity -------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
skor = cosine_similarity(matrixFeature)

andi_1 = dfbook[dfbook['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
andi_2 = dfbook[dfbook['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
andi_3 = dfbook[dfbook['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
andi_4 = dfbook[dfbook['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
andisuka = [andi_1,andi_2,andi_3,andi_4]

budi_1 = dfbook[dfbook['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].tolist()[0]-1 
budi_2 = dfbook[dfbook['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1 
budi_3 = dfbook[dfbook['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1 
budisuka = [budi_1,budi_2,budi_3]

ciko_1 = dfbook[dfbook['original_title']=='Robots and Empire']['book_id'].tolist()[0]-1 
cikosuka = [ciko_1]
# print(cikosuka)

dedi_1 = dfbook[dfbook['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1 
dedi_2 = dfbook[dfbook['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1 
dedi_3 = dfbook[dfbook['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1 
dedisuka = [dedi_1,dedi_2,dedi_3]
# print(dedisuka)

ello_1 = dfbook[dfbook['original_title']=='Doctor Sleep']['book_id'].tolist()[0]-1 
ello_2 = dfbook[dfbook['original_title']=='The Story of Doctor Dolittle']['book_id'].tolist()[0]-1 
ello_3 = dfbook[dfbook['title']=="Bridget Jones\'s Diary (Bridget Jones, #1)"]['book_id'].tolist()[0]-1 
ellosuka = [ello_1,ello_2,ello_3]
# print(ellosuka)

# Enumerating 
list_skor_andi_1 = list(enumerate(skor[andi_1]))
list_skor_andi_2 = list(enumerate(skor[andi_2]))
list_skor_andi_3 = list(enumerate(skor[andi_3]))
list_skor_andi_4 = list(enumerate(skor[andi_4]))

list_skor_budi_1 = list(enumerate(skor[budi_1]))
list_skor_budi_2 = list(enumerate(skor[budi_2]))
list_skor_budi_3 = list(enumerate(skor[budi_3]))

list_skor_ciko = list(enumerate(skor[ciko_1]))

list_skor_dedi_1 = list(enumerate(skor[dedi_1]))
list_skor_dedi_2 = list(enumerate(skor[dedi_2]))
list_skor_dedi_3 = list(enumerate(skor[dedi_3]))

list_skor_ello_1 = list(enumerate(skor[ello_1]))
list_skor_ello_2 = list(enumerate(skor[ello_2]))
list_skor_ello_3 = list(enumerate(skor[ello_3]))


list_skor_andi = []
for i in list_skor_andi_1:
    list_skor_andi.append((i[0],0.25*(list_skor_andi_1[i[0]][1]+list_skor_andi_2[i[0]][1]+list_skor_andi_3[i[0]][1]+list_skor_andi_4[i[0]][1])))
list_skor_budi = []
for i in list_skor_andi_1:
    list_skor_budi.append((i[0],(list_skor_budi_1[i[0]][1]+list_skor_budi_2[i[0]][1]+list_skor_budi_3[i[0]][1])/3))
list_skor_dedi = []
for i in list_skor_andi_1:
    list_skor_dedi.append((i[0],(list_skor_dedi_1[i[0]][1]+list_skor_dedi_2[i[0]][1]+list_skor_dedi_3[i[0]][1])/3))
list_skor_ello = []
for i in list_skor_andi_1:
    list_skor_ello.append((i[0],(list_skor_ello_1[i[0]][1]+list_skor_ello_2[i[0]][1]+list_skor_ello_3[i[0]][1])/3))

sort_andi = sorted(list_skor_andi, key=lambda j:j[1], reverse=True)
sort_budi = sorted(list_skor_budi, key = lambda j:j[1], reverse = True)
sort_ciko = sorted(list_skor_ciko, key = lambda j:j[1], reverse = True)
sort_dedi = sorted(list_skor_dedi, key = lambda j:j[1], reverse = True)
sort_ello = sorted(list_skor_ello, key = lambda j:j[1], reverse = True)

# Recommendation Top 5 -------------------------------------------------
andi_similar = []
for i in sort_andi:
    if i[1]>0:
        andi_similar.append(i)
budi_similar = []
for i in sort_budi:
    if i[1]>0:
        budi_similar.append(i)
ciko_similar = []
for i in sort_ciko:
    if i[1]>0:
        ciko_similar.append(i)
dedi_similar = []
for i in sort_dedi:
    if i[1]>0:
        dedi_similar.append(i)
ello_similar = []
for i in sort_ello:
    if i[1]>0:
        ello_similar.append(i)

# Hasil -------------------------------------------------
print('1. Buku bagus untuk Andi:')
for i in range(0,5):
    if andi_similar[i][0] not in andisuka:
        print('-',dfbook['original_title'].iloc[andi_similar[i][0]])
    else:
        i+=5
        print('-',dfbook['original_title'].iloc[andi_similar[i][0]])

print(' ')
print('2. Buku bagus untuk Budi:')
for i in range(0,5):
    if budi_similar[i][0] not in budisuka:
        print('-',dfbook['original_title'].iloc[budi_similar[i][0]])
    else:
        i+=5
        print('-',dfbook['original_title'].iloc[budi_similar[i][0]])

print(' ')
print('3. Buku bagus untuk Ciko:')
for i in range(0,5):
    if ciko_similar[i][0] not in cikosuka:
        print('-',dfbook['original_title'].iloc[ciko_similar[i][0]])
    else:
        i+=5
        print('-',dfbook['original_title'].iloc[ciko_similar[i][0]])

print(' ')
print('4. Buku bagus untuk Dedi:')
for i in range(0,5):
    if dedi_similar[i][0] not in dedisuka:
        print('-',dfbook['original_title'].iloc[dedi_similar[i][0]])
    else:
        i+=5
        print('-',dfbook['original_title'].iloc[dedi_similar[i][0]])

print(' ')
print('5. Buku bagus untuk Ello:')
for i in range(0,5):
    if ello_similar[i][0] not in ellosuka:
        if str(dfbook['original_title'].iloc[ello_similar[i][0]])=='nan':
            print('-',dfbook['title'].iloc[ello_similar[i][0]])
        else:
            print('-',dfbook['original_title'].iloc[ello_similar[i][0]])  
    else:
        i+=5
        if str(dfbook['original_title'].iloc[ello_similar[i][0]])=='nan':
            print('-',dfbook['title'].iloc[ello_similar[i][0]])
        else:
            print('-',dfbook['original_title'].iloc[ello_similar[i][0]])  