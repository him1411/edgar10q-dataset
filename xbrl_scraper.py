import bs4
import requests as rq
from pprint import pprint
from datetime import datetime as dt
import logging
import json
from xblr_parser import XBLR_Parser,get_meta_data
import os
import pandas as pd
import uuid
from tqdm import tqdm

tqdm.pandas(desc="Downloaded Files")

logging.basicConfig(level=logging.INFO)
ERROR_LOG = "./error_log.txt"

BASE_URL  = "https://www.sec.gov/Archives/edgar/data"
ACCEPTED_TYPES = ['10-K','10-Q']

f = open(DONE_COMP,'r')
done_comps = f.readlines()
f.close()

def make_url(*args):
    return "/".join(args) 

def get_accession_numbers(cik,type,proxy,start_date):
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={type}&count=1000"
    page = rq.get(url,proxies=proxy)
    page_html = page.text
    soup = bs4.BeautifulSoup(page_html,"lxml")
    tables = soup.find_all('table')
    print(len(tables))
    if len(tables) < 3:
        return []
    table = tables[2]
    table = pd.read_html(str(table))[0]
    table['Filing Date'] = pd.to_datetime(table['Filing Date'])
    table = table.loc[table['Filing Date'] > start_date,:]
    acc_numbers = list(table['Description'].apply(lambda x:x.replace('\xa0',' ').split(':')[1].split(' ')[1]))
    acc_numbers = [''.join(x.split('-')) for x in acc_numbers]

    return acc_numbers



def get_all_submissions(cik:int,start_date,base_folder,company_name,proxy_dict:dict):
    if company_name + '\n' in done_comps:
        print("All Files of the company downloaded...")
        return None
    f = open(DONE_LIST,'r')
    done_subs = f.readlines()
    f.close()
    subs_10k = get_accession_numbers(cik,'10-K',proxy_dict,start_date)
    subs_10q = get_accession_numbers(cik,'10-Q',proxy_dict,start_date) 
   
    subms = subs_10k + subs_10q
    logging.info(f"Number of Submissions made after {start_date} by {cik} is {len(subms)}")
    for subm in subms:
        subm_name = subm
        
        subm_url = make_url(BASE_URL,cik,subm_name,"index.json")
        subm_json = rq.get(subm_url).json()
        subm_files = subm_json['directory']['item']
        subm_txt_files = [x for x in subm_files if x['name'].endswith('.txt')]
        if len(subm_txt_files) > 1:
            logging.warning("More than one  txt files found..")
        txt_file_name = subm_txt_files[0]['name']
        txt_url = make_url(BASE_URL,cik,subm_name,txt_file_name)
        txt_file_obj = (rq.get(txt_url).content).decode('utf-8')
        try:
            subm_meta = get_meta_data(txt_file_obj)
            type = subm_meta['GLOBAL'].get("CONFORMED SUBMISSION TYPE",'unk')
            logging.info(f"File Type:{type}")
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
 



