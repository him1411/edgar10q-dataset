import bs4
import requests as rq
from pprint import pprint
from datetime import datetime as dt
import logging
import json
import os
import pandas as pd
import uuid
from tqdm import tqdm
from glob import glob
import urllib.request
import re
import lxml.etree as ET
import numpy as np
import traceback as tb
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import spacy
import warnings
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_sm')
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from transformers import pipeline
question_answerer = pipeline('question-answering',device = 6)
import re
import requests
import urllib.request
from difflib import SequenceMatcher
nltk.download('stopwords')
from nltk.corpus import stopwords
import sys
import traceback as tb
import time
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
ERROR_LOG = "./error_log.txt"

logging.basicConfig(level=logging.DEBUG)
tags_regex = re.compile('(table|span)|^p$|^ix|^hr$|^font$')

# SEC site - that needs to be scraped for 10-Q/K reports
BASE_URL  = "https://www.sec.gov/Archives/edgar/data"
ACCEPTED_TYPES = ['10-K','10-Q']

# Helper files for output and progress tracking
DONE_LIST,DONE_COMP = './content/SUBMS_LIST.txt','./content/COMPANY_NAMES.txt'
done_comps = []
base_folder = './content/Base'
dest_folder = './content/Output'
start_date = '2019-01-01'
# a mapping for company/cik codes
lookup = pd.read_excel('./content/cik_lookup.xlsx',sheet_name='Sheet1',header=None,names = ['company','cik'])


## helper functions
def make_url(*args):
    return "/".join(args) 

# create connection with URL
def url_get_contents(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'MyApp/1.0')]
    urllib.request.install_opener(opener)
    req = urllib.request.Request(url)
    f = urllib.request.urlopen(req)
    return f.read()


def get_accession_numbers(cik,type,start_date):
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={type}&count=1000"
    page_html = url_get_contents(url).decode('utf-8')
    soup = bs4.BeautifulSoup(page_html,"lxml")
    tables = soup.find_all('table')
    if len(tables) < 3:
        return []
    table = tables[2]
    table = pd.read_html(str(table))[0]
    table['Filing Date'] = pd.to_datetime(table['Filing Date'])
    table = table.loc[table['Filing Date'] > start_date,:]
    acc_numbers = list(table['Description'].apply(lambda x:x.replace('\xa0',' ').split(':')[1].split(' ')[1]))
    acc_numbers = [''.join(x.split('-')) for x in acc_numbers]
    return acc_numbers


def get_meta_data(subm_details_text):
    meta_data = {}
    running_titles = ["GLOBAL"]
    running_indents = [-1]
    for row in subm_details_text.split('\n'):
        if re.findall(r'\:',row):
            seg = re.split(r'\:\t*',row)
            seg = [x for x in seg if x.strip() != '']
            if len(seg) == 2:
                lhs,rhs = seg
                titles_copy = running_titles.copy()
                deep_set(meta_data, titles_copy, (lhs.strip(),rhs.strip()))
            elif len(seg) == 1:
                heading_title = seg[0]
                heading_indent = row.rstrip().count('\t')
                index = len(running_indents) -1
                while index >= 0:
                    if running_indents[index] <= heading_indent:
                        last_index = index+1 if heading_indent != running_indents[index] else index
                        running_titles = running_titles[:last_index] + [heading_title.strip()]
                        running_indents = running_indents[:last_index] + [heading_indent]
                        break
                    index -= 1
    return meta_data

def deep_set(dictionary:dict, key_row:list, value_pair:tuple):
    if len(key_row) == 1:
        if not key_row[0] in dictionary.keys():    
            dictionary[key_row[0]] = {}
        dictionary[key_row[0]][value_pair[0]] = value_pair[1]
        return
    if key_row[0] in dictionary.keys():
        deep_set(dictionary[key_row.pop(0)],key_row,value_pair)
    else:
        new_key = key_row.pop(0)
        dictionary[new_key] = {}
        deep_set(dictionary[new_key],key_row,value_pair)

## driver functions
def get_all_submissions(cik:int,start_date,base_folder,company_name):
    if company_name + '\n' in done_comps:
        print("All Files of the company downloaded...")
        return None
    if os.path.exists(DONE_LIST):
      f = open(DONE_LIST,'r')
      done_subs = f.readlines()
      f.close()
    cik=str(cik)
    subs_10k = get_accession_numbers(cik,'10-K',start_date)
    subs_10q = get_accession_numbers(cik,'10-Q',start_date)
    subms = subs_10k + subs_10q
    logging.info(f"Number of Submissions made after {start_date} by {cik} is {len(subms)}")
    for subm in subms:
        subm_name = subm
        subm_url = make_url(BASE_URL,cik,subm_name,"index.json")
        subm_json =json.loads(url_get_contents(subm_url).decode('utf-8'))
        subm_files = subm_json['directory']['item']
        subm_txt_files = [x for x in subm_files if x['name'].endswith('.txt')]
        if len(subm_txt_files) > 1:
            logging.warning("More than one  txt files found..")
        txt_file_name = subm_txt_files[0]['name']
        txt_url = make_url(BASE_URL,cik,subm_name,txt_file_name)
        txt_file_obj = (url_get_contents(txt_url)).decode('utf-8')
        try:
            subm_meta = get_meta_data(txt_file_obj)
            type = subm_meta['GLOBAL'].get("CONFORMED SUBMISSION TYPE",'unk')
            if type == 'unk' :
              type = subm_meta['GLOBAL']['Originator-Key-Asymmetric'].get("CONFORMED SUBMISSION TYPE",'unk')
            if not os.path.exists(f"{base_folder}/{type}/{company_name}"):
                if not os.path.exists(f"{base_folder}/{type}"):
                    os.makedirs(f"{base_folder}/{type}")
                os.makedirs(f"{base_folder}/{type}/{company_name}")
        except:
            logging.exception("------------- Error in extracting meta data ---------------")
            continue
        
        f = open(f"{base_folder}/{type}/{company_name}/{txt_file_name}",'w+',encoding='utf-8')
        f.write(txt_file_obj)
        f.close()
        f = open(DONE_LIST,'a')
        f.write(subm_name+"\n")
        f.close() 
    f = open(DONE_COMP,'a')
    f.write(company_name+'\n')
    f.close()

def IE_Parser(subm_text):
    company = subm_text.split('/')[4]
    if len(subm_text) > 10000:
      return []
    soup = bs4.BeautifulSoup(open(subm_text),'lxml')
    docs = soup.find_all('document')
    md_docs = [doc for doc in docs if next(doc.find('filename').children).strip().endswith('.json')]
    if len(md_docs) == 0:
        return []
    md = md_docs[0]
    try:
      response = json.loads(md.find('text').text)
    except JSONDecodeError :
      response = json.loads(md.find('text').contents[0])
    data = list(response['instance'].items())[0][1]['tag']
    mapper = {}
    for key, value in data.items():
      try:
        mapper[key] = (value['lang']['en-US']['role'], value['xbrltype'])
      except KeyError :
        mapper[key] = (value['lang']['en-us']['role'], value['xbrltype'])
    body = soup.find('text')
    ps = body.find_all(re.compile('p|span'))
    dataset = []
    for p in ps:
        sample = {}
        txt = p.parent.text
        sample['key_value_pairs'] = []
        ix_regex = re.compile('^ix')
        tags = p.find_all(ix_regex,recursive=True)
        for tag in tags:
            if tag.name.startswith('ix:'):
                if 'name' not in tag.attrs:
                    continue
                name = tag.attrs['name'].replace(":", "_")
                if name.endswith('TextBlock'):
                    continue
                if abs(len(str(tag.text)) - len(txt)) < 4:
                    continue
                dict1_vals =list( mapper[name][0].values())
                dict1_keys = list(mapper[name][0].keys())
                label=[]
                if (len(dict1_keys) == 1) and (re.search(r'label', dict1_keys[0].lower())):
                  label = dict1_vals[0]
                elif len(dict1_keys) > 1 :
                  for key,value in zip(dict1_keys,dict1_vals):
                    if key in ['terseLabel','totalLabel','verboseLabel']:
                      label.append(value)
                sample['key_value_pairs'].append({'text':txt,'value':tag.text, 'label':label, 'name':name, 'type': mapper[name][1],\
                                                  'company':company})
        if len(sample['key_value_pairs']) != 0:
            dataset.append(sample)
    return dataset

def clean_lib2(string):
	string = string.replace(".", "")
	string = string.replace(";", "")
	string = string.replace("-", " ")
	string = string.replace("\xa0", " ")
	#removing words between parenthesis along with parenthesis if it has non-digit chracters
	string = str(re.sub(r'\(\D*?\)','', string))
	string = str(re.sub(r'[/]*', '', string))
	# removing brackets surrounding $(65)
	string = str(re.sub(r'\(','', string))
	string = str(re.sub(r'\)','', string))
	string = string.strip()
	return string

def entity_tagger(text):    
  sentence = nlp(text)
  ans_dict = dict((ent.text,ent.label_) for ent in sentence.ents)
  ans = []
  for key,value in ans_dict.items():
      temp =[]
      temp.append(key)
      label_type = value
      temp2 = re.sub(r'\([^)]*\)', '', str(label_type))
      temp2 = str(temp2).strip()
      temp.append(temp2)
      ans.append(temp)
  return(ans)
 
def sent_parse(row):
  sents = sent_tokenize(row['paragraph'])
  for sent in sents :
    if row['value'] in sent:
      return sent


def qa_model(entity,phrases,sentence,entity_type):
	timewords = [' due ', ' age ', ' date ',' life of ', ' lives of ', ' during ']
	best_score = 0
	best_key=''
	best_question=''
	best_ans=''
	for phrase in phrases:
		if any(time in phrase for time in timewords) == True:
			temp = "when is " + phrase + " ?"
		elif entity_type =="CARDINAL":
			temp = "how many "+phrase + " ?"
		else:
			temp = "what is " + phrase + " ?"		
		qa_model_ans = question_answerer({'question': temp,'context': sentence})
		ans = qa_model_ans['answer']
		confidence_score = qa_model_ans['score']
		if entity in ans and confidence_score > best_score:
			best_ans = ans
			best_score = confidence_score
			best_question = temp
			best_key = phrase	
	temp = "What is " + entity + " ?"
	qa_model_ans = question_answerer({'question': temp,'context': sentence})
	ans = qa_model_ans['answer']
	confidence_score = qa_model_ans['score']
	entities = entity_tagger(sentence)
	if confidence_score >= best_score and (any(ent[0] in ans for ent in entities)== False ):
		ans1 =[]
		ans1.extend([ans, confidence_score, temp, ans])
		return ans1
	else:
		ans1 = []
		ans1.extend([best_key, best_score, best_question,best_ans])
		return ans1

    
def sentence_entity_flair(sentence,entity, entity_type):
  labels = ['MONEY' , 'DATE' , 'CARDINAL', 'PERCENT', 'TIME']  
    #removing words between parenthesis along with parenthesis if it has non-digit chracters
  sentence = str(re.sub(r'\(\D*?\)','', str(sentence)))
  sentence = str(re.sub(r'[/]*', '', sentence))
  # removing brackets surrounding $(65)
  sentence = str(re.sub(r'\(','', sentence))
  sentence = str(re.sub(r'\)','', sentence))
  entities_received = entity_tagger(sentence)
  for item in entities_received:
    ent = item[0]
    ent_label = item[1]
    if ent_label == 'PERCENT' and entity_type == 'percentItemType' and ((entity + '%' in ent) or (entity + ' %' in ent)):
      temp =[]
      temp.extend( [ent,ent_label,sentence] )
      return temp
    elif ent_label == 'MONEY' and entity_type == 'monetaryItemType' and (entity in ent):
      if '$' not in ent:
        ent = '$' + ent
      temp =[]
      temp.extend( [ent,ent_label,sentence] )
      return temp

    elif ent_label == 'CARDINAL' and entity_type in ['sharesItemType','integerItemType','pureItemType'] and (entity in ent):
      temp =[]
      temp.extend( [ent,ent_label,sentence] )
      return temp
      
    elif ent_label in ['DATE','TIME'] and entity_type in ['durationItemType','dateItemType'] and (entity in ent) :
      temp =[]
      temp.extend( [ent,ent_label,sentence] )
      return temp
  return [entity, 'none', sentence]	

def preposition_phrase_extraction(text):
  doc = nlp(text)
  sent = []

  for token in doc:
    if token.pos_=='ADP':
      phrase = ''
      if token.head.pos_ in ([ 'NOUN','PRONOUN']):   # if its head word is a noun
        for subtoken in token.head.children:
          if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound') or ('amod' in subtoken.dep_):  
            # if word is an adjective or has a compound dependency
            phrase += subtoken.text + ' '
          else: pass
          
        phrase += token.head.text   # append noun and preposition to phrase
        
        for right_tok in token.rights:  # check the nodes to the right of the preposition
          if (right_tok.pos_ in ['NOUN','PROPN']):
            #append_preposition
            phrase += ' '+token.text
            # append if it is a noun or proper noun
            for subtoken in right_tok.children:
              if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound') or ('amod' in subtoken.dep_):  
                # if word is an adjective or has a compound dependency
                phrase +=' ' + subtoken.text 
              else:pass
            phrase += ' '+right_tok.text
        if len(phrase)>=2:
          sent.append(phrase)
  return sent

def noun_phrase_extraction(text):
	doc = nlp(text)
	pat = []# iterate over tokens	
	for token in doc:
		phrase = ''  # if the word is a subject noun or an object noun
		if (token.pos_ in ['NOUN','PROPN']) and (token.dep_ in ['dobj','pobj','nsubj','nsubjpass']):# iterate over the children nodes
			for subtoken in token.children: # if word is an adjective or has a compound dependency
				if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound'):
					phrase += subtoken.text + ' '
				else: pass
				
			if len(phrase)!=0:
				phrase += token.text
		if  len(phrase)!=0:
			pat.append(phrase)
	return pat

def phrase_extraction(text):
	phrases = []
	entities = entity_tagger(text)
	#taking phrases not present in any entity
	for phrase in preposition_phrase_extraction(text):
		flag = True
		phrase_set = nltk.word_tokenize(str(phrase))
		phrase_set = set(phrase_set)
		for entity in entities:
			entity_set = nltk.word_tokenize(entity[0])
			entity_set = set(entity_set)

			if phrase_set.issubset(entity_set):
				flag = False
				break
				   
		if flag:
			phrases.append(phrase)				

	for phrase in noun_phrase_extraction(text):
		flag = True
		phrase_set = nltk.word_tokenize(str(phrase))
		phrase_set = set(phrase_set)
		for entity in entities:
			entity_set = nltk.word_tokenize(entity[0])
			entity_set = set(entity_set)
				   
			if phrase_set.issubset(entity_set):
				flag = False
				break
			
		if flag:
			phrases.append(phrase)
	return phrases

def driver_writer_func(company_name,cik,start_date,base_folder,dest_folder):
  type_list = ['dateItemType','sharesItemType','integerItemType','pureItemType','durationItemType','monetaryItemType','percentItemType']
  os.system(f"rm -rf {base_folder}")
  os.mkdir(base_folder)
  if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)
  get_all_submissions(cik,start_date,base_folder,company_name)
  # structuring the dataset in a tabular format and further preprocessing
  files = glob(base_folder + '/**/*.txt', recursive=True)
  files = [i for i in files if company_name in i]
  df_list = []
  for txt in tqdm(files):
    df_list.append(IE_Parser(txt))
  os.system(f"rm -rf {base_folder}")
  final_list = [j for sub in df_list for j in sub]
  ls1 = []
  for i in final_list:
    ls1+=i['key_value_pairs']
  df = pd.DataFrame(ls1)
  print('original df - ',str(df.shape[0]))
  df['text'] = df['text'].str.replace('-',' ') 
  df.rename(columns={'text':'paragraph'},inplace=True)
  df['paragraph'] = df['paragraph'].str.replace('\xa0',' ')
  #replacing new line symbols with space
  df['paragraph'] = df['paragraph'].apply(lambda x : re.sub( r'(\n+)' , ' ',x )) 
  df['value'] = df['value'].str.replace('-',' ')
  df['value'] = df['value'].str.replace('\xa0',' ')
  df = df.replace(np.nan, '', regex=True)
  df['len_txt'] = df['paragraph'].apply(lambda x : len(x.split(' ')))
  df = df[(df['len_txt']>10)&(df['len_txt']<=1000)]
  print('truncated df - ',str(df.shape[0]))
  df.drop(columns = ['len_txt'],inplace=True)
  df = df[df['type'].isin(type_list)]
  print('type-filtered df - ',str(df.shape[0]))
  df['sent'] = df.apply(sent_parse, axis=1)
  df['temp'] = df[['sent','value', 'type']].apply(lambda x: sentence_entity_flair(*x), axis=1)
  df[['entity','entity_type_ext','sentence']] = pd.DataFrame(df.temp.tolist(), index= df.index)
  del df['temp']
  df.drop(df[df['entity_type_ext'] == 'none'].index, inplace=True)
  df['label'] = df['label'].apply(lambda x : [clean_lib2(i) for i in x])
  df['paragraph'] = df['paragraph'].str.replace('“','"')
  df['paragraph'] = df['paragraph'].str.replace('”','"')
  df['paragraph'] = df['paragraph'].str.replace('’',"'")
  df['paragraph'] = df['paragraph'].str.replace(' – ','-')
  df['paragraph'] = df['paragraph'].str.strip()
  df['paragraph'] = df['paragraph'].apply(lambda x : str(re.sub(r'\s+',' ',x)))
  df['phrases'] = df[['sentence']].apply(lambda x: phrase_extraction(*x), axis=1)
  df['qa_temp'] = df[['entity','phrases','sentence','entity_type_ext']].apply(lambda x : qa_model(*x),axis=1 )
  df[['key','score','question','answer']] = pd.DataFrame(df.qa_temp.tolist(), index= df.index)
  df.to_csv(dest_folder+'/'+company_name+'.csv',index=False)


# extracting data for companies using cik codes
cik_lookup = lookup.set_index('company').to_dict()['cik']
t1 = time.time()
for k,v in tqdm(cik_lookup.items()):
  print(k,v)
  try:
    driver_writer_func(k,v,start_date,base_folder,dest_folder)
  except:
    os.system(f"rm -rf {base_folder}")
    continue
t2 = time.time()
minutes = (t2-t1)/60.0
print(f'time taken is {minutes} min.')