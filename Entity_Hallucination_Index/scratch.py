
import streamlit as st
import pandas as pd
import numpy as np
import nlu

from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

  
  
  
@st.cache(suppress_st_warning=True)
def loadpipeline(name="not provided"):
    st.write("parameter passed" +name)
    return PretrainedPipeline('bert_base_token_classifier_conll03', lang='en')
 
 
 
 
 

inputtext="The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport. Mr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42. Appearing at the Old Bailey earlier, all four denied the offence. The charge relates to offences which allegedly took place between 2008 and 2014. Sam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, are due to stand trial in July. They were all released on bail."



spark = sparknlp.start()



col1 = st.expander("Use Pretrained")

langlist=['en','de']
tasklist=['classify','ner','pos',]
modelllist=['token_bert','token_albert']
datasetlist=['conll03']

langoption = col1.selectbox('choose Language',langlist)
st.write("you have selected "+ langoption)
taskoption = col1.selectbox('choose Language',tasklist)
st.write("You selected NLP task :" + taskoption)


modeloption = col1.selectbox('choose Language',modelllist)
st.write("You selected NLP Model :" + modeloption)

datasetoption = col1.selectbox('choose Language',datasetlist)
st.write("Dataset chosen is :" + datasetoption)

paramname=langoption+'.'+taskoption+'.'+modeloption+'.'+datasetoption 
st.write(paramname)

#st.write(nlu.load(paramname).predict('I am very bad person'))




pipeline = loadpipeline(paramname)
# Your testing dataset
text = """
The Mona Lisa is a 16th century oil painting created by Leonardo.
It's held at the Louvre in Paris.
"""

# Annotate your testing dataset
result = pipeline.annotate(text)

st.write(result)


















