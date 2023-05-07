

import streamlit as st
import pandas as pd
import numpy as np
import time 
import random
import string
from numpy import savetxt
import csv
from numpy import loadtxt
from datasets import load_metric
from statistics import mean

m_location="metrics_cache/"
p_location="prediction_cache/"





class SummarizerResults:
    modelname="not assigned"
    reccount=0
    ResultsFile="not assigned"
    input_text=[]
    ref_summ=[]
    gen_summ=[]
    rougescores=[]
    
    def __init__(self, ResultsFile,input_text,ref_summ,gen_summ, reccount=0,name=""):
        self.ResultsFile = ResultsFile
        self.input_text=input_text
        self.ref_summ=ref_summ
        self.gen_summ=gen_summ
        self.reccount = reccount
        self.modelname=name.replace("/",'_')
       
        
        
    def dump_metrics_to_file(self):
           
            rouge1=[]
            rouge2=[]
            rougeL=[]
            rougeLsum=[]
            refabs_r1=[]
            refabs_r2=[]
            refabs_rL=[]
            summabs_r1=[]
            summabs_r2=[]
            summabs_rL=[]
            
            tscore=np.transpose(self.rougescores)
            tabscore=np.transpose(self.refabs)
            stabscore=np.transpose(self.summabs)
            for i in range(len(self.rougescores[:])):
                rouge1.append(tscore[i]['rouge1'])
                rouge2.append(tscore[i]['rouge2'])
                rougeL.append(tscore[i]['rougeL'])
                rougeLsum.append(tscore[i]['rougeLsum'])
                refabs_r1.append(1-tabscore[i]['rouge1'])
                refabs_r2.append(1-tabscore[i]['rouge2'])
                refabs_rL.append(1-tabscore[i]['rougeL'])
                summabs_r1.append(1-stabscore[i]['rouge1'])
                summabs_r2.append(1-stabscore[i]['rouge2'])
                summabs_rL.append(1-stabscore[i]['rougeL'])
                
            #st.write(refabs_r2)
                
            nprouge1=np.array(rouge1)
            nprouge2=np.array(rouge2)
            nprougeL=np.array(rougeL)
            nprougeLsum=np.array(rougeLsum)
            
            
            nprefabs_r1=np.array(refabs_r1)
            nprefabs_r2=np.array(refabs_r2)
            nprefabs_rL=np.array(refabs_rL)
            
            
            npsummabs_r1=np.array(summabs_r1)
            npsummabs_r2=np.array(summabs_r2)
            npsummabs_rL=np.array(summabs_rL)
            
            npratio_r1=np.divide(summabs_r1,refabs_r1)
            npratio_r2=np.divide(summabs_r2,refabs_r2)
            npratio_rL=np.divide(summabs_rL,refabs_rL)
            
            
            st.write("Saving to files started:")
            savetxt(m_location+self.modelname+'_Rouge1.csv',nprouge1,delimiter=",")
            savetxt(m_location+self.modelname+'_Rouge2.csv',nprouge2,delimiter=",")
            savetxt(m_location+self.modelname+'_RougeL.csv',nprougeL,delimiter=",")
            savetxt(m_location+self.modelname+'_RougeLsum.csv',nprougeLsum,delimiter=",")
            savetxt(m_location+self.modelname+'_Refabs_R1.csv',nprefabs_r1,delimiter=",")
            savetxt(m_location+self.modelname+'_Refabs_R2.csv',nprefabs_r2,delimiter=",")
            savetxt(m_location+self.modelname+'_Refabs_RL.csv',nprefabs_rL,delimiter=",")
            
            savetxt(m_location+self.modelname+'_summabs_R1.csv',npsummabs_r1,delimiter=",")
            savetxt(m_location+self.modelname+'_summabs_R2.csv',npsummabs_r2,delimiter=",")
            savetxt(m_location+self.modelname+'_summabs_RL.csv',npsummabs_rL,delimiter=",")
            
            savetxt(m_location+self.modelname+'_absratio_R1.csv',npratio_r1,delimiter=",")
            savetxt(m_location+self.modelname+'_absratio_R2.csv',npratio_r2,delimiter=",")
            savetxt(m_location+self.modelname+'_absratio_RL.csv',npratio_rL,delimiter=",")
            
            
            st.write("Saving to files Completed")
               
    
    @st.cache(suppress_st_warning=True)
    def calculate_metrics(self):
        self.rougescores=compute_rouge(self.gen_summ,self.ref_summ,self.reccount)
        self.refabs=self.compute_Reference_Abstraction()
        self.summabs=self.compute_summary_Abstraction()
        self.Absratio=self.compute_Abstration_Ratio()
        #st.write(self.rougescores)
        #st.write(self.rougescores[0])
        self.dump_metrics_to_file()
        
        return self.rougescores
        
    @st.cache(suppress_st_warning=True)
    def compute_Reference_Abstraction(self):
        ref_rouge=compute_rouge(self.ref_summ,self.input_text,self.reccount)
        
        return ref_rouge
    
    @st.cache(suppress_st_warning=True)
    def compute_summary_Abstraction(self):
        summ_rouge=compute_rouge(self.gen_summ,self.input_text,self.reccount)
        return summ_rouge
    def compute_Abstration_Ratio(self):
        abs_ratio=[]
        self.rougescores
    def newchart_rouges(self,col1):
        
# load numpy array from csv file
        from numpy import loadtxt
        
        nprouge1=loadtxt(m_location+self.modelname+'_Rouge1.csv',delimiter=",")
        nprouge2=loadtxt(m_location+self.modelname+'_Rouge2.csv',delimiter=",")
        nprougeL=loadtxt(m_location+self.modelname+'_RougeL.csv',delimiter=",")
        nprefabs_r1=loadtxt(m_location+self.modelname+'_Refabs_R1.csv',delimiter=",")
        nprefabs_r2=loadtxt(m_location+self.modelname+'_Refabs_R2.csv',delimiter=",")
        nprefabs_rL=loadtxt(m_location+self.modelname+'_Refabs_RL.csv',delimiter=",")
        npsummabs_r1=loadtxt(m_location+self.modelname+'_summabs_R1.csv',delimiter=",")
        npsummabs_r2=loadtxt(m_location+self.modelname+'_summabs_R2.csv',delimiter=",")
        npsummabs_rL=loadtxt(m_location+self.modelname+'_summabs_RL.csv',delimiter=",")
        
        npratio_r1=(loadtxt(m_location+self.modelname+'_absratio_R1.csv',delimiter=","))
        npratio_r2=(loadtxt(m_location+self.modelname+'_absratio_R2.csv',delimiter=","))
        npratio_rL=(loadtxt(m_location+self.modelname+'_absratio_RL.csv',delimiter=","))
        
        avgr1=mean(nprouge1)
        avgr2=mean(nprouge2)
        avgrL=mean(nprougeL)
        
        avgrefabs_r1=mean(nprefabs_r1)
        avgrefabs_r2=mean(nprefabs_r2)
        avgrefabs_rL=mean(nprefabs_rL)
        
        avgsummabs_r1=mean(npsummabs_r1)
        avgsummabs_r2=mean(npsummabs_r2)
        avgsummabs_rL=mean(npsummabs_rL)
        
        avgratio_r1=mean(npratio_r1)
        avgratio_r2=mean(npratio_r2)
        avgratio_rL=mean(npratio_rL)
        
        self.nprouge1=nprouge1
        self.nprouge2=nprouge2
        self.nprougeL=nprougeL
        
        self.nprefabs_r1=nprefabs_r1
        self.nprefabs_r2=nprefabs_r2
        self.nprefabs_rL=nprefabs_rL
        
        self.npsummabs_r1=npsummabs_r1
        self.npsummabs_r2=npsummabs_r2
        self.npsummabs_rL=npsummabs_rL
        
        self.npratio_r1=npratio_r1
        self.npratio_r2=npratio_r2
        self.npratio_rL=npratio_rL
        
        
        
        
        
        st.write(avgr1)
        
        count=self.reccount-1
        cols=['Rouge1','Rouge2','RougeL','RA1', 'RA2' ,'RAL','SA1','SA2','SAL','AR1','AR2','ARL']
        
        saved_cols=['Rouge1','Rouge2','RougeL','RA1', 'RA2' ,'RAL','SA1','SA2','SAL','AR1','AR2','ARL']
        temp={'Rouge1': nprouge1, 'Rouge2': nprouge2 ,'RougeL' : nprougeL, 'RA1' : nprefabs_r1,'RA2' : nprefabs_r2, 'RAL' : nprefabs_rL ,'SA1' : npsummabs_r1 , 'SA2' : npsummabs_r2 , 'SAL' : npsummabs_rL , 'AR1' : npratio_r1,'AR2': npratio_r2 , 'ARL': npratio_rL }
        saved_temp=temp
        metriclist=cols
        options = col1.multiselect('choose metrics to Hide', metriclist, metriclist,key=self.modelname)
        #st.write('You selected:', options)
        
        if(len(options)!=len(cols)):
            cols=saved_cols
            temp=saved_temp
            difference_1 = set(cols).difference(set(options))
            difference_2 = set(options).difference(set(options))
           
            if(len(difference_1)>1):
                for i in list(difference_1):
                    temp.pop((i))
                    cols.remove((i))
            else:
                
                temp.pop(list(difference_1)[0])
                cols.remove(list(difference_1)[0])
        
        avgcols=['Rouge1','Rouge2','RougeL','RA1', 'RA2' ,'RAL','SA1','SA2','SAL','AR1','AR2','ARL']
        avgtemp={'Avg R1': avgr1, 'Avg R2': avgr2 ,'Abg RL' : avgrL, 'Avg RA1' : avgrefabs_r1,'Avg RA2' : avgrefabs_r2, 'Avg RAL' : avgrefabs_rL ,'Avg SA1' : avgsummabs_r1 , 'Avg SA2' : avgsummabs_r2 , 'Avg SAL' : avgsummabs_rL , 'Avg AR1' : avgratio_r1,'Avg AR2': avgratio_r2 , 'Avg ARL': avgratio_rL }
       # temp.pop("Rouge2")
        
        
       
       # cols.remove("Rouge2")
        
        dataset=pd.DataFrame(temp,( str(i+1)  +"_Article" for i in range(count) ),columns=cols)
        avgdataset=pd.DataFrame(avgtemp,( str(i+1)  +"_Article" for i in range(12) ),columns=avgcols)
       # dataset=pd.DataFrame({'Rouge1': nprouge1, 'Rouge2': nprouge2 ,'RougeL' : nprougeL, 'RA1' : nprefabs_r1,'RA2' : nprefabs_r2, 'RAL' : nprefabs_rL ,'SA1' : npsummabs_r1 , 'SA2' : npsummabs_r2 , 'SAL' : npsummabs_rL , 'AR1' : npratio_r1,'AR2': npratio_r2 , 'ARL': npratio_rL },( str(i+1)  +"_Article" for i in range(count) ),columns=['Rouge1','Rouge2','RougeL','RA1', 'RA2' ,'RAL','SA1','SA2','SAL','AR1','AR2','ARL'])
       # st.write(np.transpose(self.rougescores)[1]['rouge1'])
        
       # dataset=pd.DataFrame({'Rouge1': nprouge1, 'Rouge2': nprouge2 ,'RougeL' : nprougeL}, ( str(i+1)  +"_Article" for i in range(count)),columns=['Rouge1','Rouge2','RougeL'])
     
       # dataset=pd.DataFrame({'RA1' : nprefabs_r1,'RA2' : nprefabs_r2, 'RAL' : nprefabs_rL },( str(i+1)  +"_Article" for i in range(count)),columns=['RA1', 'RA2' ,'RAL'])
        col1.line_chart(dataset)
        col1.area_chart(dataset)
        col1.write(avgdataset)
        col1.write(avgtemp)
        #col1.line_chart(avgtemp)
        #col1.line_chart(avgdataset)
      #  col1.vega_lite_chart(dataset)
        
        col1.dataframe(dataset.style.highlight_min(axis=0))
        #st.write(dataset)
        #col1.write(dataset)
       
        
       # chart_data = pd.DataFrame(nprouge1,("Article_" + str(i+1) for i in range(131)),columns=['Rouge1','Rouge2'])
    
        #st.write(chart_data)
           
        
    def old_newchart_rouges(self,col1):
        
# load numpy array from csv file
        from numpy import loadtxt
        
        nprouge1=loadtxt(m_location+self.modelname+'_Rouge1.csv',delimiter=",")
        nprouge2=loadtxt(m_location+self.modelname+'_Rouge2.csv',delimiter=",")
        nprougeL=loadtxt(m_location+self.modelname+'_RougeL.csv',delimiter=",")
        nprefabs_r1=loadtxt(m_location+self.modelname+'_Refabs_R1.csv',delimiter=",")
        nprefabs_r2=loadtxt(m_location+self.modelname+'_Refabs_R2.csv',delimiter=",")
        nprefabs_rL=loadtxt(m_location+self.modelname+'_Refabs_RL.csv',delimiter=",")
        npsummabs_r1=loadtxt(m_location+self.modelname+'_summabs_R1.csv',delimiter=",")
        npsummabs_r2=loadtxt(m_location+self.modelname+'_summabs_R2.csv',delimiter=",")
        npsummabs_rL=loadtxt(m_location+self.modelname+'_summabs_RL.csv',delimiter=",")
        
        npratio_r1=(loadtxt(m_location+self.modelname+'_absratio_R1.csv',delimiter=","))
        npratio_r2=(loadtxt(m_location+self.modelname+'_absratio_R2.csv',delimiter=","))
        npratio_rL=(loadtxt(m_location+self.modelname+'_absratio_RL.csv',delimiter=","))
        
        
        count=self.reccount-1
        
        dataset=pd.DataFrame({'Rouge1': nprouge1, 'Rouge2': nprouge2 ,'RougeL' : nprougeL, 'RA1' : nprefabs_r1,'RA2' : nprefabs_r2, 'RAL' : nprefabs_rL ,'SA1' : npsummabs_r1 , 'SA2' : npsummabs_r2 , 'SAL' : npsummabs_rL , 'AR1' : npratio_r1,'AR2': npratio_r2 , 'ARL': npratio_rL },( "<a>" + str(i+1)  +"_Article</a>" for i in range(count) ),columns=['Rouge1','Rouge2','RougeL','RA1', 'RA2' ,'RAL','SA1','SA2','SAL','AR1','AR2','ARL'])
       # st.write(np.transpose(self.rougescores)[1]['rouge1'])
        
       # dataset=pd.DataFrame({'Rouge1': nprouge1, 'Rouge2': nprouge2 ,'RougeL' : nprougeL}, ( str(i+1)  +"_Article" for i in range(count)),columns=['Rouge1','Rouge2','RougeL'])
        col1.write(dataset)
       # dataset=pd.DataFrame({'RA1' : nprefabs_r1,'RA2' : nprefabs_r2, 'RAL' : nprefabs_rL },( str(i+1)  +"_Article" for i in range(count)),columns=['RA1', 'RA2' ,'RAL'])
        
        col1.dataframe(dataset.style.highlight_min(axis=0))
        #st.write(dataset)
        
       
        
       # chart_data = pd.DataFrame(nprouge1,("Article_" + str(i+1) for i in range(131)),columns=['Rouge1','Rouge2'])
    
        #st.write(chart_data)
        col1.line_chart(dataset)    
        
    def chart_rouges(self):
        
        st.write(type(self.rougescores))
        st.write(self.rougescores)
        st.write(type(self.rougescores[:]))
        rouge1=[]
        rouge2=[]
        rougeL=[]
        rougeLsum=[]
        tscore=np.transpose(self.rougescores)
        st.write(tscore)
        for i in range(len(self.rougescores[:])):
            rouge1.append(tscore[i]['rouge1'])
            rouge2.append(tscore[i]['rouge2'])
            rougeL.append(tscore[i]['rougeL'])
            rougeLsum.append(tscore[i]['rougeLsum'])
        nprouge1=np.array(rouge1)
     #   nprouge1=np.insert(arr=nprouge1,obj=rouge1,values=4)
        st.write(nprouge1)
     #   nprouge1=np.insert(nprouge2,rouge1,axis=1)
        nprouge2=np.array(rouge2)
        nprougeL=np.array(rougeL)
        nprougeLsum=np.array(rougeLsum)
       # nprouge1=np.append(nprouge1,nprouge2)
        st.write(np.shape(nprouge1))
        st.write(np.shape(nprouge2))
        st.write(np.shape(nprougeL))
        dataset=pd.DataFrame({'Rouge1': nprouge1, 'Rouge2': nprouge2 ,'RougeL' : nprougeL },columns=['Rouge1','Rouge2','RougeL'])
       # st.write(np.transpose(self.rougescores)[1]['rouge1'])
        st.write(dataset)
        
        st.write(type((self.rougescores[:])))
        
       # chart_data = pd.DataFrame(nprouge1,("Article_" + str(i+1) for i in range(131)),columns=['Rouge1','Rouge2'])
    
        #st.write(chart_data)
        st.line_chart(dataset)


   
    


   


def Read_file(arg1):
    
    latest_iteration = st.empty()
    data=[]
    
    with open(arg1, newline='',encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile,delimiter=',' ))
    reccount=len(data)
    #st.write(data)
    input=[]
    ref_summ=[]
    gen_summ=[]
    for i in range(1,reccount):
        
        input.append(data[i][0])
        ref_summ.append(data[i][1])
        gen_summ.append(data[i][2])
        #st.write(gen_summ)
    col1=st.expander(data[0][2])
    col1.write("Finished reading "+ arg1+ " file into memory.It Contains "+ str(reccount) +" records.");
    
    return input,ref_summ,gen_summ,reccount,data[0][2],col1

def second_compute_rouge(gen_summ,ref_summ,reccount):
    from rouge import Rouge 
    rouger = Rouge()
    result0=[]
    results0=[]
    for i in range(1):
        scores = rouger.get_scores(gen_summ[i],ref_summ[i]) 
        st.write(scores)
        result0.append(scores)
        results0.append({k1: (v1['f']) for k1, v1 in scores[0].items()})
        #st.write({k1: (v1['f']) for k1, v1 in scores[0].items()})
    st.write("printing")
    for x, y in scores[0].items():
        st.write(x, y["f"])
    
    st.write("typeofcores",type(scores))
    st.write("typeofresult0",type(result0))
    st.write("scores",scores)
    st.write("typeofscore0item",type(scores[0].items()))
    
   # st.write("results0",results0[0])
    st.write(done)
    return results0
def compute_rouge(gen_summ,ref_summ,reccount):

    from datasets import load_metric
    rouge_score = load_metric('rouge')
    result0=[]
    results0=[]
    for i in range(reccount-1):
        result0.append(rouge_score.compute(predictions=[gen_summ[i]], references=[ref_summ[i]]))
      
        results0.append({k1: max(v1.mid.fmeasure,0.001) for k1, v1 in result0[i].items()})
    #st.write("result0",result0)
    #st.write("results0",results0)
    return results0
  
    
def old_compute_rouge(gen_summ,ref_summ,reccount):

    from datasets import load_metric
    rouge_score = load_metric('rouge')
    result0=[]
    results0=[]
    for i in range(reccount-1):
        result0.append(rouge_score.compute(predictions=[gen_summ[i]], references=[ref_summ[i]]))
        results0.append({k1: round(1-v1.mid.fmeasure, 2) for k1, v1 in result0[i].items()})
    st.write(result0)
    st.write(results0)
    return results0

def Process_cachedfiles(listoffiles):   
    
    i=0
    models=session_state.computedmodels
    for model in models:
        
        st.title(model.modelname)
        st.write("Input Text")
        st.write(models.input_text)
        st.write("Reference Summary")
        st.write(models.ref_summ)
        st.write("Generated Summary")
        st.write(models.gen_summ)
        st.write("Rouge Scores")
        #st.write(models.calculate_metrics())
        models[i].newchart_rouges()
        st.write("Completed" )
        i=i+1
    
        
    

  
    
def Process_files(listoffiles):   
    
    i=0
    models=[]
    for fname in listoffiles:
        
        fname=p_location+fname
        
        input_text=[]
        ref_summ=[]
        gen_summ=[]
        input_text,ref_summ,gen_summ,reccount,name,col1=Read_file(fname)
        
       
        models.append(SummarizerResults(fname,input_text,ref_summ,gen_summ,reccount,name))
        col1.title(models[i].modelname)
        
        selectlist=[]
        for k in range(0,reccount-1):
            selectlist.append(k)

        option = col1.selectbox('Choose article to review:',selectlist,key=name+"_input")
        


        col1.subheader("Input Text")            
        col1.write(models[i].input_text[option])
        col1.subheader("Reference Summary")
        col1.write(models[i].ref_summ[option])
        col1.subheader("Generated Summary")
        col1.write(models[i].gen_summ[option])
        col1.subheader("Rouge Scores")
        col1.write(models[i].calculate_metrics())
        
        models[i].newchart_rouges(col1)
        
        i=i+1
    st.session_state.computedmodels = models
    common_modelnamelist=[]
    common_rouge1=[]
    common_rouge2=[]
    common_rougeL=[]
    for model in models:
      if(model.reccount>129):
        st.write("name",model.modelname)
        st.write("Rougescores",model.rougescores)
        st.write(model.modelname+" Rouge1",model.nprouge1)
        st.write(model.modelname+" Rouge2",model.nprouge2)
        st.write(model.modelname+" RougeL",model.nprougeL)
        st.write(model.modelname+" Reference Abstraction R1",model.nprefabs_r1)
        st.write(model.modelname+" Reference Abstraction R2",model.nprefabs_r2)
        st.write(model.modelname+" Reference Abstraction RL",model.nprefabs_rL)
        st.write(model.modelname+" Summary Abstraction R1",model.npsummabs_r1)
        st.write(model.modelname+" Summary Abstraction R2",model.npsummabs_r2)
        st.write(model.modelname+" Summary Abstraction RL",model.npsummabs_rL)
        st.write(model.modelname+" Abstration Ratio R1",model.npratio_r1)
        st.write(model.modelname+" Abstration Ratio R2",model.npratio_r2)
        st.write(model.modelname+" Abstration Ratio RL",model.npratio_rL)
        
        common_modelnamelist.append(model.modelname)
        common_rouge1.append(model.nprouge1)
        common_rouge2.append(model.nprouge2)
        common_rougeL.append(model.nprougeL)
        
    
        
        
        
    st.write(common_modelnamelist)
    st.write(type(common_modelnamelist))
    st.write(common_rouge1)
    st.write(common_rouge2)
    st.write(common_rougeL)
    
    
    
        
        
        
        


#Filelist=['Dataset_distilbart.csv','Dataset_bart.csv','Dataset_pegasus.csv','Dataset_huggingface.csv']
Filelist=[]
if 'Reload' not in st.session_state:
    st.session_state['Reload'] = 1

Reload=st.session_state['Reload']
st.write(Reload)
if  Reload==1:

    # Using readlines()
    file1 = open(p_location+'modellist.txt', 'r')
    Lines = file1.readlines()
    col1 = st.expander("List of Models ")
    
    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        col1.write("Model{}: {}".format(count, line.strip()))
        filename="Dataset_"+line.strip()
        filename=filename.replace("/",'_')+".csv"
        #st.write(filename)
        Filelist.append(filename)
            
    Process_files(Filelist)
    Reload=1
    st.session_state['Reload']=Reload
else:
    #process_cachedfiles(Filelist)
    print("else")
    
