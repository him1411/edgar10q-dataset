import lxml.etree as ET
import bs4
import re
import pprint
import numpy as np
import json
import os
import logging
import traceback as tb
import sys

logging.basicConfig(level=logging.DEBUG)
tags_regex = re.compile('(table|span)|^p$|^ix|^hr$|^font$')

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

def get_title(title_text):
    match = re.match('item\s+\d+[a-d]?',title_text.strip().lower())
    if not match:
       return None
    else:
       return match.group().upper() 

def has_parent(tag):
    parent_iter = iter(tag.parents)
    for parent in parent_iter:
        if re.match(tags_regex,parent.name):
            return True
    return False

def IE_Parser(subm_text):
    soup = bs4.BeautifulSoup(subm_text,'lxml')
    docs = soup.find_all('document')
    #xbrl_doc = docs[0]
    md_docs = [doc for doc in docs if next(doc.find('filename').children).strip().endswith('.json')]
    if len(md_docs) == 0:
        raise Exception(f'{[next(doc.find("filename").children) for doc  in docs]}')
    else:
        print('md found')
    md = md_docs[0]
    response = json.loads(md.find('text').text)
    data = list(response['instance'].items())[0][1]['tag']
    mapper = {}
    for key, value in data.items():
        mapper[key] = (value['lang']['en-US']['role'], value['xbrltype'])
    body = soup.find('text')
    ps = body.find_all(re.compile('p|span'))
    dataset = []
    for p in ps:
        sample = {}
        sample['text'] = p.parent.text
        sample['key_value_pairs'] = []
        ix_regex = re.compile('^ix')
        tags = p.find_all(ix_regex,recursive=True)
        for tag in tags:
            if tag.name.startswith('ix:'):
                if 'name' not in tag.attrs:
                    continue
                name = tag.attrs['name'].replace(":", "_")
                if name.endsiwth('TextBlock'):
                    continue
                if abs(len(str(tag.text)) - len(sample['text'])) < 4:
                    continue
                sample['key_value_pairs'].append({'value':tag.text, 'label_info':mapper[name][0], 'name':name, 'type': mapper[name][1]})
        
        if len(sample['key_value_pairs']) != 0:
            dataset.append(sample)
    return dataset



def parse(file_path,root):

    r = open(file_path,'r')
    try:
        print(file_path)
        json_obj = IE_Parser(r.read())
        file_name = os.path.basename(file_path)[:-4] +".json"
        os.makedirs(os.path.join(DUMP_FOLDER,root),exist_ok=True)
        out_filepath = os.path.join(DUMP_FOLDER,root,file_name)
        json.dump(json_obj,open(out_filepath,"w+"))
        print("saved at ",out_filepath)
        print("*"*10)
    except:
        logging.error(tb.format_exc())

    r.close()


