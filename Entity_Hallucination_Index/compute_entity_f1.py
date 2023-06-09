# -*- coding: utf-8 -*-
"""Entity_F1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15mgJ77p5RmqU7LMAzicPmkjJgMro7wL-
"""

import nltk
import time 
import datetime
import csv
from numpy import savetxt

from nltk.metrics import precision, recall, f_measure
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize

import nltk

drive_prefix=""
p_location=drive_prefix+"Prediction_cache1/"

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')



def compute_Entity_F1(path):
  start_time = time.time()            
  now = datetime.datetime.now()            
  print("The code execution started at :" + now.strftime("%Y-%m-%d %H:%M:%S"))
  Filelist=[]
  modellist=[]
  file1 = open(p_location+'modellist.txt', 'r')
  Lines = file1.readlines()
  count=0
  for model in Lines:
    count += 1
    #col1.write("Model{}: {}".format(count, model.strip()))
    modellist.append(model.strip())
    filename="Dataset_"+model.strip()
    filename=filename.replace("/",'_')+".csv"
    print(filename)
    Filelist.append(filename)
  print("completed reading filename") 
  for filename in Filelist:
    input_text=[]
    ref_summ=[]
    gen_summ=[]
    entity_f1=[]
   
    with open(p_location+filename, 'r',encoding='utf-8') as csvfile:
    # Use csv.reader to parse CSV file
        csvreader = csv.reader(csvfile)
    
    # Iterate over each row in the CSV file
        for row in csvreader:
        # Append each column value to its respective list
            input_text.append(row[0])
            ref_summ.append(row[2])
            gen_summ.append(row[3])
            
        for iter in  range(1,200):
            print(str(iter) + filename )
            e1=calculate_entity_f1(ref_summ[iter],gen_summ[iter])
            print(ref_summ[iter])
            print(gen_summ[iter])
            entity_f1.append(e1)
            print(e1)
    entity_f1= [0 if x is None else x for x in entity_f1]
    savetxt(p_location+(filename.replace(".csv",'_Ef1.csv').replace('Dataset_','')),entity_f1,delimiter=",")
    print("====================================================")
  
    
        
    




def tree_to_string(tree):
    return ' '.join([word for word, tag in tree.leaves()])

def calculate_entity_f1(gold_summary, predicted_summary):
    # Tokenize both summaries
    gold_tokens = word_tokenize(gold_summary)
    predicted_tokens = word_tokenize(predicted_summary)

    # Tag the tokens using NLTK's part-of-speech tagger
    gold_pos_tags = nltk.pos_tag(gold_tokens)
    predicted_pos_tags = nltk.pos_tag(predicted_tokens)

    # Extract named entities from both summaries using NLTK's ne_chunk function
    gold_entities = set([tree_to_string(chunk) for chunk in ne_chunk(gold_pos_tags) if hasattr(chunk, 'label')])
    predicted_entities = set([tree_to_string(chunk) for chunk in ne_chunk(predicted_pos_tags) if hasattr(chunk, 'label')])
    #print(gold_entities)
    #print(predicted_entities)
    # Calculate precision, recall, and F1 score
    precision_score = precision(gold_entities, predicted_entities)
    recall_score = recall(gold_entities, predicted_entities)
    f1_score = f_measure(gold_entities, predicted_entities)

    return f1_score


compute_Entity_F1("")
