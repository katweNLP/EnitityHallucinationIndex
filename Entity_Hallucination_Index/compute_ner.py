#!pip install transformers
#!pip install flair

#from google.colab import drive
#drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import datetime
import time
import csv

import pandas as pd
import traceback
from numpy import savetxt

#from flair.data import Sentence
#from flair.models import SequenceTagger

#drive_prefix="/content/drive/MyDrive/HallucinationStudy_master/"
drive_prefix=""
p_location=drive_prefix+"Prediction_cache1/"
h_location=drive_prefix+"Hallucination_cache1/"


def compute_NER_flair(modelname,filename,N=200,initialize=0):
# load tagger
  tagger = SequenceTagger.load("flair/pos-english")
  input,ref_summ,gen_summ,reccount,modelnamefromfile = read_file(filename)

  input_NER=[]
  ref_NER=[]
  summ_NER=[]
  if(initialize!=0):
    #print("Initializing NERs for Input")
    for i in range(0,N-1):
      input_NER.append(get_Entities(text=input[i],tagger=tagger))
    data = {'Input_NER' : input_NER}
    df = pd.DataFrame(data)
    df.to_csv(h_location+ "/NER_Input.csv", index=False) 
    #print("Initializing NERs for Reference")
    for i in range(0,N-1):
      ref_NER.append(get_Entities(text=ref_summ[i],tagger=tagger))
    data = {'Ref_NER' : ref_NER}
    df = pd.DataFrame(data)
    df.to_csv(h_location+ "/NER_Reference.csv", index=False) 




  for i in range(0,N-1):
  #  input_NER.append(get_Entities(input[i]))
  #  ref_NER.append(get_Entities(ref_summ[i]))
    summ_NER.append(get_Entities(text=gen_summ[i],tagger=tagger))
    


  
  
  data = {'summ_NER' : summ_NER}
  df = pd.DataFrame(data)
  df.to_csv(h_location+ "/NER_"+ modelname.replace("/",'_') +".csv", index=False)  
  print (df)
  df = pd.DataFrame(data)
  df.to_csv(h_location+ "/NER_"+ modelname.replace("/",'_') +".csv", index=False)  

def get_Entities(text,tagger):

#  input_pos.append("The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport. Mr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42. Appearing at the Old Bailey earlier, all four denied the offence. The charge relates to offences which allegedly took place between 2008 and 2014. Sam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, are due to stand trial in July. They were all released on bail.")
#  input_pos.append("I hate berlin")
  #for text in input:
# make example sentence
  
#sentence=Sentence("I live in bangalore")
#sentence=Sentence("The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation")
# predict NER tags
    
    sentence=Sentence(text)
    tagger.predict(sentence)
  ##print(tagger)
# print sentence
 # #print(sentence)
    #print(type(sentence))
    entities=[]
    for token in sentence:
    ##print(token)
      temp=token.get_label().value
    ##print(temp)
      if 'NN' in temp:
        #print(token.text)
        entities.append(token.text)

    #print("There were "+ str(len(entities)) + "tokens")
    entities=list(set(entities))
    #print("There were "+ str(len(entities)) + "tokens")
    #print(entities)
    output_ner=[]
    output_ner.append(len(entities))
    output_ner.append(entities)
    return output_ner



def generate_NER(modelname,filename,N=200):

    tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl", truncation=True)
    model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer,aggregation_strategy="first")
    nlp1= pipeline("ner", model=model, tokenizer=tokenizer,aggregation_strategy="none")
    input,ref_summ,gen_summ,reccount,modelnamefromfile = read_file(filename)

    input_NER=[]
    ref_NER=[]
    summ_NER=[]

    #print("Outside loop :" + str(len(ref_summ)))

    for i in range(0,N-1):
      #print("This is the current i value : "+str(i))
      #print("This is the content of input:"+input[i])
      #print("This is the content of ref_summ:"+ref_summ[i])
      #print("This is the content of gen:"+gen_summ[i])


      try:
        temp=nlp(input[i])
        if(len(temp)==1):
          temp=nlp1(input[i])
        input_NER.append(process_ner_output(temp))
      except:
        input_NER.append("ERRORXXX")
      try:
        temp=nlp(ref_summ[i])
        if(len(temp)==1):
          temp=nlp1(ref_summ[i])
        ref_NER.append(process_ner_output(temp))
      except:
        #print(traceback.format_exc())
        ref_NER.append("ERRORXXX")
        
      try:
        
        temp=nlp(gen_summ[i])
        if(len(temp)==1):
          temp=nlp1(gen_summ[i])
        summ_NER.append(process_ner_output(temp))
        #print(summ_NER[i])
      except:
        summ_NER.append("ERRORXXX")
        
      
    data = {'input_NER': input_NER,'ref_NER': ref_NER,'summ_NER' : summ_NER}
    df = pd.DataFrame(data)
    df.to_csv(h_location+ "/NER_"+ modelname.replace("/",'_') +".csv", index=False)  
    print (df)
def read_file(filename):
    data=[]
    
    with open(filename, newline='',encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile,delimiter=',' ))
    reccount=len(data)
    ##print(data)
    input=[]
    ref_summ=[]
    gen_summ=[]
    for i in range(1,reccount-1):
      
        input.append(data[i][1])
        ref_summ.append(data[i][2])
        gen_summ.append(data[i][3])
        ##print(gen_summ)
      

   
    #print("Finished reading "+ filename + " file into memory.It Contains "+ str(reccount) +" records.");
    
    return input,ref_summ,gen_summ,reccount,data[0][3]
def process_ner_output(result):
  #print(len(result))
  wordlist=[]
  for i in range(0,len(result)-1):
    try:
      wordlist.append(result[i].get('word'))
    except:
      print("seems found list" + type(result[i]))
      #print(result[i])

 
  output=[]
  output.append(wordlist)
  #print("This " + str(wordlist))
  return (str(output).replace(',',';').replace('[[\'','').replace('\']]',''))

from numpy import average
def compute_entity_hallucination(path,st):

  start_time = time.time()            
  now = datetime.datetime.now()            
  #print("The code execution started at :" + now.strftime("%Y-%m-%d %H:%M:%S"))
  Filelist=[]
  modellist=[]
  file1 = open(p_location+'modellist.txt', 'r')
  Lines = file1.readlines()
  count=0
  for model in Lines:
    count += 1
    #col1.write("Model{}: {}".format(count, model.strip()))
    modellist.append(model.strip())
    filename="NER_"+model.strip()
    filename=filename.replace("/",'_')+".csv"
    #print(filename)
    Filelist.append(filename)

  input_file=h_location+ "/NER_Input.csv"
  ref_file=h_location+ "/NER_Reference.csv"
  input_entities_count,input_entities=populate_ner_file(input_file)
  ref_entities_count,ref_entities=populate_ner_file(ref_file)
#  #print(input_entities_count[0])
#  #print(ref_entities[0])
#  #print(ref_entities_count[0])
#  #print(input_entities[0])
  
  mlist_entities=['None']*len(modellist)
  mlist_entities_count=['none']*len(modellist)
  model_entities_count=[]
  model_entities=[]
  for j in range(0,len(modellist)):
    model_file=h_location+Filelist[j]
    model_entities_count,model_entities=populate_ner_file(model_file)
    ##print(model_entities_count[0])
    ##print(model_entities[0])
    mlist_entities[j]=model_entities
    mlist_entities_count[j]=model_entities_count
  #print(len(mlist_entities))
  #for j in range(0,len(mlist_entities)):
   # #print(mlist_entities[j][0])
   # #print(mlist_entities_count[j][0])
  models,records = (len(modellist),200)
  models_hallucination=[records,models]
  #print(models_hallucination)
  #print(type(models_hallucination))
  

############################Reference Hallucination#########################  
  reference_hallucination=compute_hallucination(input_entities_count,input_entities,ref_entities_count,ref_entities)
  #print("1.Reference Hallucination")
  #print(reference_hallucination)
  #print(average(reference_hallucination))

#############################Unimportant / Noise############################
  reference_unimportant=compute_hallucination(ref_entities_count,ref_entities,input_entities_count,input_entities)
  #print("2.Unimportant or noise in Input text")
  #print(reference_unimportant)
  #print(average(reference_unimportant))

############################Model hallucination#############################
  
  
  for i in range(len(modellist)):
    #print("3."+str(i)+ " " + modellist[i] + "Model Hallucination" )
    temp_hallucination=compute_hallucination(input_entities_count,input_entities,mlist_entities_count[i],mlist_entities[i],model_name=modellist[i])
    #print(temp_hallucination)
    #print(average(temp_hallucination))
   # #print(len(temp_hallucination))
########################### Model Lost Entities #####################################

  for i in range(len(modellist)):
    #print("4."+str(i)+ " ""Lost Entities of model " + modellist[i])
    temp_hallucination=compute_hallucination(mlist_entities_count[i],mlist_entities[i],input_entities_count,input_entities,model_name=modellist[i])
   # #print("Output started for " + modellist[i] + " With respect to lost focus)
   # #print(temp_hallucination)
    #print(temp_hallucination)
    #print(average(temp_hallucination))
##################################################################################    
  for i in range(len(modellist)):
     ##print("5."+str(i)+ " ""Positie Hallucination for  " + modellist[i])
    temp_hallucination=compute_hallucination(ref_entities_count,ref_entities,mlist_entities_count[i],mlist_entities[i],model_name=modellist[i])
   # #print("Output started for " + modellist[i] + " With respect to Reference")
   # #print(temp_hallucination)
    #print(temp_hallucination)
    #print(average(temp_hallucination))
    #print("Output Ended for " + modellist[i] + " With respect to Reference") 
  
#################################################################################
#######################Positive Hallucination####################################
  for i in range(len(modellist)):
    
    positive_hallucination,overfocused_entities,negative_hallucination=compute_hallucination_ref(input_entities_count,input_entities,ref_entities_count,ref_entities,mlist_entities_count[i],mlist_entities[i])

    ##print(positive_hallucination)
    #print(average(positive_hallucination))
    #print("Output Ended for " + modellist[i] + " Positive Hallucination") 


#######################Over Focused Model##########################################
   # #print(overfocused_entities)
    #print(average(overfocused_entities))
    #print("Output Ended for " + modellist[i] + " OverFocused Entities") 


#######################Negative hallucination #####################################
   # #print(negative_hallucination)
    #print(average(negative_hallucination))
    #print("Output Ended for " + modellist[i] + " Negative Hallucination") 
    
    #print(len(positive_hallucination))
    #print(len(negative_hallucination))
    #print(len(overfocused_entities))
    EHI=[]
    for j in range(len(positive_hallucination)):
       # #print(positive_hallucination[j])
       # #print(negative_hallucination[j])
       # #print(overfocused_entities[j])
        EHI.append(1+(2*positive_hallucination[j])-(2*negative_hallucination[j])-(0.25*overfocused_entities[j]))
        
    
    ##print(EHI)
    ##print(average(EHI))
    ##print(max(EHI))
    ##print(min(EHI))
    
    ##print(modellist[i]+"Positive")
    ##print(positive_hallucination)
    ##print("negative")
    ##print(negative_hallucination)
    ##print("Overfocused")
    ##print(overfocused_entities)
    
    
    streamplot(EHI,modellist[i])
    savetxt(path+(modellist[i].replace("/",'_'))+"EHI.csv",EHI,delimiter=",")
    
    
    
    

def new_compute_entity_hallucination(path,st):

  start_time = time.time()            
  now = datetime.datetime.now()            
  #print("The code execution started at :" + now.strftime("%Y-%m-%d %H:%M:%S"))
  Filelist=[]
  modellist=[]
  file1 = open(p_location+'modellist.txt', 'r')
  Lines = file1.readlines()
  count=0
  for model in Lines:
    count += 1
    #col1.write("Model{}: {}".format(count, model.strip()))
    modellist.append(model.strip())
    filename="NER_"+model.strip()
    filename=filename.replace("/",'_')+".csv"
    #print(filename)
    Filelist.append(filename)

  input_file=h_location+ "/NER_Input.csv"
  ref_file=h_location+ "/NER_Reference.csv"
  input_entities_count,input_entities=populate_ner_file(input_file)
  ref_entities_count,ref_entities=populate_ner_file(ref_file)
#  #print(input_entities_count[0])
#  #print(ref_entities[0])
#  #print(ref_entities_count[0])
#  #print(input_entities[0])
  
  mlist_entities=['None']*len(modellist)
  mlist_entities_count=['none']*len(modellist)
  model_entities_count=[]
  model_entities=[]
  for j in range(0,len(modellist)):
    model_file=h_location+Filelist[j]
    model_entities_count,model_entities=populate_ner_file(model_file)
    ##print(model_entities_count[0])
    ##print(model_entities[0])
    mlist_entities[j]=model_entities
    mlist_entities_count[j]=model_entities_count
  #print(len(mlist_entities))
  #for j in range(0,len(mlist_entities)):
   # #print(mlist_entities[j][0])
   # #print(mlist_entities_count[j][0])
  models,records = (len(modellist),200)
  models_hallucination=[records,models]
  #print(models_hallucination)
  #print(type(models_hallucination))
  

############################Reference Hallucination#########################  
  reference_hallucination=compute_hallucination(input_entities_count,input_entities,ref_entities_count,ref_entities)
  #print("1.Reference Hallucination")
  #print(reference_hallucination)
  #print(average(reference_hallucination))

#############################Unimportant / Noise############################
  reference_unimportant=compute_hallucination(ref_entities_count,ref_entities,input_entities_count,input_entities)
  #print("2.Unimportant or noise in Input text")
  #print(reference_unimportant)
  #print(average(reference_unimportant))

############################Model hallucination#############################
  
  
  for i in range(len(modellist)):
    #print("3."+str(i)+ " " + modellist[i] + "Model Hallucination" )
    temp_hallucination=compute_hallucination(input_entities_count,input_entities,mlist_entities_count[i],mlist_entities[i],model_name=modellist[i])
    #print(temp_hallucination)
    #print(average(temp_hallucination))
   # #print(len(temp_hallucination))
########################### Model Lost Entities #####################################

  for i in range(len(modellist)):
    #print("4."+str(i)+ " ""Lost Entities of model " + modellist[i])
    temp_hallucination=compute_hallucination(mlist_entities_count[i],mlist_entities[i],input_entities_count,input_entities,model_name=modellist[i])
   # #print("Output started for " + modellist[i] + " With respect to lost focus)
   # #print(temp_hallucination)
    #print(temp_hallucination)
    #print(average(temp_hallucination))
##################################################################################    
  for i in range(len(modellist)):
     ##print("5."+str(i)+ " ""Positie Hallucination for  " + modellist[i])
    temp_hallucination=compute_hallucination(ref_entities_count,ref_entities,mlist_entities_count[i],mlist_entities[i],model_name=modellist[i])
   # #print("Output started for " + modellist[i] + " With respect to Reference")
   # #print(temp_hallucination)
    #print(temp_hallucination)
    #print(average(temp_hallucination))
    #print("Output Ended for " + modellist[i] + " With respect to Reference") 
  
#################################################################################
#######################Positive Hallucination####################################
  for i in range(len(modellist)):
    
    #positive_hallucination,overfocused_entities,negative_hallucination=compute_hallucination_ref(input_entities_count,input_entities,ref_entities_count,ref_entities,mlist_entities_count[i],mlist_entities[i])
    extractivessFactor,positive_hallucination,overfocused_entities,negative_hallucination,LostFocus =new_compute_hallucination_ref(input_entities_count,input_entities,ref_entities_count,ref_entities,mlist_entities_count[i],mlist_entities[i])

    EHI=[]
    for j in range(len(positive_hallucination)):
       # #print(positive_hallucination[j])
       # #print(negative_hallucination[j])
       # #print(overfocused_entities[j])
       # EHI.append(1+(2*positive_hallucination[j])-(2*negative_hallucination[j])-(0.25*overfocused_entities[j]))
      # st.write(j)
      # st.write(len(extractivessFactor))
       #st.write(len(positive_hallucination))
       #st.write(len(overfocused_entities))
       #st.write(len(negative_hallucination))
       #st.write(len(LostFocus))
      # st.write(modellist[i])
       #st.write((positive_hallucination))
       #st.write((overfocused_entities))
       #st.write((negative_hallucination))
       #st.write((LostFocus))
       
       #EHI.append(   (2*extractivessFactor[j]) + (2*positive_hallucination[j]) - (0.5*overfocused_entities[j]) - (2*negative_hallucination[j]) - (0.5*LostFocus[j]))
       EHI.append(  2+((extractivessFactor[j]) * (positive_hallucination[j]) )- (overfocused_entities[j]) - (negative_hallucination[j]) - (LostFocus[j]))
       
    st.write(len(EHI))
    st.write((positive_hallucination))
    st.write((overfocused_entities))
    st.write((negative_hallucination))
    st.write((LostFocus))
    st.write(EHI)
    streamplot(EHI,modellist[i])
    savetxt(path+(modellist[i].replace("/",'_'))+"EHI.csv",EHI,delimiter=",")    
    
    
    
    ##print(EHI)
    print(average(EHI))
    ##print(max(EHI))
    ##print(min(EHI))
    
    ##print(modellist[i]+"Positive")
    ##print(positive_hallucination)
    ##print("negative")
    ##print(negative_hallucination)
    ##print("Overfocused")
    #print(overfocused_entities)
    #multiplot(extractivessFactor,positive_hallucination,overfocused_entities,negative_hallucination,LostFocus,EHI,modellist[i])
    
    
    
    
    


def streamplot(list,name):
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt

# Create a list of numbers
    data = list

# Plot the data using Matplotlib
    fig, ax = plt.subplots()
    ax.set_title(name)
    ax.plot(data)

# Display the plot using Streamlit
    st.pyplot(fig)

####################################
def compute_hallucination_ref(orig_entities_count, orig_entities,ref_entities_count, ref_entities,gen_entities_count, gen_entities,N=200):
  ############################3.A. Positive hallucination ###########################
  #3.A############ 2 * matching entities /  sum of total entities in gen and ref
  #3.B############ Lost Focus in generated summary
  #3.C############ Negative hallucination in generated summary 


  positive_hallucination_index=[]
  for ref_iter in range(min(len(ref_entities),N)):
    matching_entities=0
    for gen_entry in gen_entities[ref_iter]:
      if gen_entry in ref_entities[ref_iter]:
        matching_entities=matching_entities+1
    
  
  # Lost focus 
  lost_hallucination_index=[]
  negative_hallucination_index=[]

  for orig_iter in range(min(len(orig_entities),N)):
    matching_entities=0
    gained_focus_entities=0
    hallucinated_entities=0
    for gen_entry in gen_entities[orig_iter]:
      if gen_entry in ref_entities[ref_iter]:
        matching_entities=matching_entities+1  
      if gen_entry not in ref_entities[orig_iter] and gen_entry in orig_entities[orig_iter]:
        gained_focus_entities=gained_focus_entities+1
      if gen_entry not in ref_entities[orig_iter] and gen_entry not in orig_entities[orig_iter]:
        hallucinated_entities=hallucinated_entities+1
    sum_temp=int(ref_entities_count[ref_iter])+int(gen_entities_count[ref_iter])
    temp=(2*matching_entities)/sum_temp
    positive_hallucination_index.append(temp)
  #  #print("value of count"+gen_entities_count[orig_iter])
    denominator=int(gen_entities_count[orig_iter])
    if(int(gen_entities_count[orig_iter]) == 0):
        denominator=1
    negative_hallucination_index.append(hallucinated_entities/denominator)
    lost_hallucination_index.append(gained_focus_entities/(denominator))
    #lost_hallucination_index.append(gained_focus_entities/(max(0.00000001,int(gen_entities_count[orig_iter]))))
    #negative_hallucination_index.append(hallucinated_entities/(max(0.0000001,int(gen_entities_count[orig_iter]))))

  return positive_hallucination_index,lost_hallucination_index,negative_hallucination_index

        
      


       

def new_compute_hallucination_ref(orig_entities_count, orig_entities,ref_entities_count, ref_entities,gen_entities_count, gen_entities,N=200):
  ############################3.A. Positive hallucination ###########################
  #3.A############ 2 * matching entities /  sum of total entities in gen and ref
  #3.B############ Lost Focus in generated summary
  #3.C############ Negative hallucination in generated summary 

  extractiveness_index=[]
  positive_hallucination_index=[]
  lost_hallucination_index=[]
  negative_hallucination_index=[]
  overfocus_index=[]
  lostfocus_index=[]
  

  for orig_iter in range(min(len(orig_entities),N)):
    
    extractive_count=0
    positive_count=0
    overfocus_count=0
    negative_count=0
    lostfocus_count=0
    
    for gen_entry in gen_entities[orig_iter]:
      if gen_entry in ref_entities[orig_iter] and gen_entry in orig_entities[orig_iter]:
        extractive_count=extractive_count+1
        
      if gen_entry in ref_entities[orig_iter] and gen_entry not in orig_entities[orig_iter]:
        positive_count=positive_count+1
        
      if gen_entry not in ref_entities[orig_iter] and gen_entry in orig_entities[orig_iter]:
        overfocus_count=overfocus_count+1
        
      if gen_entry not in ref_entities[orig_iter] and gen_entry not in orig_entities[orig_iter]:
        negative_count=negative_count+1
    
    for ref_entry in ref_entities[orig_iter]:
      if ref_entry not in gen_entities[orig_iter] and ref_entry in orig_entities[orig_iter]:
        lostfocus_count=lostfocus_count+1
        
    #Extractivness index 
    denominator=int(orig_entities_count[orig_iter])+int(ref_entities_count[orig_iter])+int(gen_entities_count[orig_iter])
    extractiveness_index.append((3*extractive_count)/(denominator))
    
     
    #Positive Hallucination
    denominator=int(ref_entities_count[orig_iter])+int(gen_entities_count[orig_iter])
    positive_hallucination_index.append((2*positive_count)/denominator)
    
    #OverFocus 
    denominator=int(orig_entities_count[orig_iter])+ int(gen_entities_count[orig_iter])
    overfocus_index.append((2*overfocus_count)/denominator)
    
    #Negative Hallucination
    
    denominator= int(gen_entities_count[orig_iter])
    if(int(gen_entities_count[orig_iter]) == 0):
        denominator=1
    negative_hallucination_index.append((negative_count)/denominator)
    
    
    #Lost Focus 
    denominator=int(orig_entities_count[orig_iter])+int(ref_entities_count[orig_iter])
    lostfocus_index.append(1-(lostfocus_count/(denominator)))
    #lost_hallucination_index.append(gained_focus_entities/(max(0.00000001,int(gen_entities_count[orig_iter]))))
    #negative_hallucination_index.append(hallucinated_entities/(max(0.0000001,int(gen_entities_count[orig_iter]))))

  return extractiveness_index,positive_hallucination_index,overfocus_index,negative_hallucination_index,lostfocus_index

        
      


       




def compute_hallucination(orig_entities_count, orig_entities,hall_entities_count, hall_entities,N=200,model_name="Reference summary"):
  ##print("There are " + str(len(orig_entities_count)) + " entries in the original" )
  ##print("There are " + str(len(hall_entities_count)) + " entries in the Hallucinator("+model_name+")" )
#  #print(hall_entities[0])
#  #print(orig_entities[0])
  hallucination_index=[]
  for hal_iter in range (min(len(hall_entities),N)):
    hallucination=0
    extractiveness=0
    for hal_entry in (hall_entities[hal_iter]):
        
        if hal_entry in orig_entities[hal_iter]:
            ##print("Found "+ hal_entry + " in original" )
            extractiveness=extractiveness+1
        else:
            ##print("Entity " + hal_entry + " is hallucinated ")
            hallucination=hallucination+1
    ##print("Hallucinated entity count " + str(hallucination))
    ##print("Count of hallucinator summary is "+ str(len(hall_entities[hal_iter])))
    hallucination_index.append(hallucination/len(hall_entities[hal_iter]))
    ##print("Hallucination_index="+str(hallucination_index[hal_iter]))
  
  return hallucination_index
      

def populate_ner_file(filename):
  with open(filename, newline='',encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile,delimiter=',' ))
  reccount=len(data)
  entities=[]
  entities_count=[]
  for i in range(1,reccount):
        dlist=data[i][0].replace('[','').replace(']','').replace(' ','').replace('\'','').split(',')
        entities_count.append(dlist[0])
        entities.append(set(dlist[1:]))
  ##print(input_entities_count[0])
  ##print(input_entities[0])
  return entities_count,entities  

def multiplot(list1,list2,list3,list4,list5,list6,name):

    import numpy as np
    import matplotlib.pyplot as plt

        # Define the four lists to plot
    
    
    data = np.array([list1, list2, list3, list4,list5,list6])

# Create a Pandas DataFrame from the NumPy array
    df = pd.DataFrame(data.T, columns=['Extract','Positive',  'OverFocus','Negative','LostFocus', 'EHI'])

# Display the DataFrame as a table
    st.table(df)

    # Create a new figure and plot the lists
    fig, ax = plt.subplots()
    ax.plot(list1, label='Positive')
    ax.plot(list2, label='Negative')
    ax.plot(list3, label='OverFocus')
    ax.plot(list4, label='EHI')
    ax.set_title(name)

# Add a legend to the plot
    ax.legend()

# Display the plot on Streamlit
    st.pyplot(fig)
   
    


#import streamlit as st
#new_compute_entity_hallucination(p_location,st)

