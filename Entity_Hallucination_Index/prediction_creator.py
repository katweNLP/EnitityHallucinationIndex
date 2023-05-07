# -*- coding: utf-8 -*-

#**1. Libraries** **Installation**
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install transformers
!pip install sentencepiece
!pip install transformers datasets
!pip install transformers datasets rouge_score
!pip install bert_score

"""#**2. Dataset**"""
import transformers
from transformers import pipeline 
import pandas as pd
import numpy as np
from datasets import load_dataset
import streamlit as st

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep

m_location="metrics_cache/"
p_location="prediction_cache1/"



@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

@st.cache(suppress_st_warning=True)
def load_datasetlocal():

    output = st.expander
    with st.expander("DatasetLoading"):
        dataset = load_dataset('xsum',split='validation')
        st.write(dataset)
        st.write(dataset.column_names)
        dataset.split
        N=150
        input_txt=dataset['document'][:N]
        ref=dataset['summary'][:N]
        input=[]
        for i in range(N):
          input.append(input_txt[i].replace("\n"," "))

        st.write(input[2])
        st.write(ref[2])

        df = pd.DataFrame({"Document" : input,"Summary" : ref})
        df.to_csv("Dataset.csv", index=False)
    return input,ref
@st.cache(suppress_st_warning=True)
def generate_prediction(modelname,filename,input,ref,N=150):
   
   with st.expander(modelname):
        
        """#**3. Summarization**"""


        summarizer = pipeline("summarization", model=modelname)
        summary=[]
        input_text=[]
        ref_summ=[]
        gen_summ=[]
        count=0
        for i in range(N):
          
          try:
            st.write(i)
            st.write(input[i])
            
            

            tsummary=summarizer(input[i])
            summary.append(tsummary[0])
            st.write(tsummary[0])

           #count=count+1
            s=input[i]
            s1=ref[i]
            input_text.append(s)
            ref_summ.append(s1)
            gen_summ.append(summary[i]['summary_text'])
            df = pd.DataFrame({"Document" : input_text,"Summary" : ref_summ, "pegasus_gen": gen_summ})
            df.to_csv(p_location+filename, index=False)
          except:
             st.write("exception : ") 
             st.write(i)
            
            

Filelist=[]
modellist=[]
file1 = open(p_location+'modellist.txt', 'r')
Lines = file1.readlines()
col1 = st.expander("List of Models ")

count = 0
# Strips the newline character

input,ref=load_datasetlocal()
for model in Lines:
    count += 1
    col1.write("Model{}: {}".format(count, model.strip()))
    modellist.append(model.strip())
    filename="Dataset_"+model.strip()
    filename=filename.replace("/",'_')+".csv"
    #st.write(filename)
    Filelist.append(filename)
    
        
for i in range(len(modellist)):
    generate_prediction(modellist[i],Filelist[i],input,ref,N=3)
    
        

      





