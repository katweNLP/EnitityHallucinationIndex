
import streamlit as st
import pandas as pd
import numpy as np
import csv
from numpy import savetxt

def Process_files(listoffiles):   
    
    i=0
    models=[]
    data_fs=[]
    reccount=[]
    for fname in listoffiles:
        
        fname=p_location+fname
        st.write(fname)
        input_text=[]
        ref_summ=[]
        gen_summ=[]
        latest_iteration = st.empty()
        
        with open(fname, newline='',encoding='utf-8') as csvfile:
            data_fs.append(list(csv.reader(csvfile,delimiter=',' )))
        reccount.append(len(data_fs[i]))
        i=i+1
   # st.write(data_fs) 
   # st.write(type(data_fs))
   # st.write(len(data_fs))
    minrec=min(reccount)
    maxrec=max(reccount)
   # st.write(minrec)
    #st.write(maxrec)
   # st.write(reccount)
    
    lrepeat=0
    hrepeat=0
    for i in range(0,len(reccount)):
        if(reccount[i]==minrec):
            lrepeat=lrepeat+1
            lindex=i
            st.write(listoffiles[i] + " has "+ str(minrec) +" records  at index "+ str(i) )
        if(reccount[i]==maxrec):
            hrepeat=hrepeat+1
            hindex=i
            st.write(listoffiles[i] + " has "+ str(maxrec) +" records  at index "+ str(i) )
    if(lrepeat> 1):
        st.write(" There are "+str(lrepeat)+" models which have lowest number records ")
    if(hrepeat> 1):
        st.write(" There are "+str(hrepeat)+" models which have highest number records ")
    missingindexes=[]
    #for w in range(1,len(data_fs)-1):
    for w in range(1,minrec-1):
        a=int(data_fs[lindex][w][0])
        b=int(data_fs[lindex][w+1][0])
      #  st.write((int(data_fs[lindex][w][0]) - int(data_fs[lindex][w+1][0])))
        if((b-a) > 1):
            st.write("Index value " +  str(a+1) + " is missing from model " + listoffiles[lindex] )
           # st.write(data_fs[lindex][w][0])
            missingindexes.append(a+1)
    for i in range(0,len(data_fs)):
        if(i==lindex):
            continue
        for j in range(1,reccount[i]-1):
            currentid=int(data_fs[i][j][0])
            if currentid in missingindexes:
                #st.write("Found extra record " + str(currentid) + "in "+listoffiles[i] )
                data_fs[i].pop(j)
                j=j-1
    #st.write(data_fs) 
    #df=pd.DataFrame({"Document" : input_text,"Summary" : ref_summ, "pegasus_gen": gen_summ})
    for i in range(0,len(data_fs)):
        df=pd.DataFrame(data_fs[i][1:],columns=['Number','Document','Summary',listoffiles[i].replace("Dataset_",'').replace(".csv",'')])
      #  st.write(df)
        df.to_csv(p_location+listoffiles[i]+"new", index=False)
    
    
        
        
    
        
    
        
        
        
    

p_location="prediction_cache1/"
  
Filelist=[]  
file1 = open(p_location+'modellist.txt', 'r')
Lines = file1.readlines()

count = 0
    # Strips the newline character
for line in Lines:
    count += 1
   
    filename="Dataset_"+line.strip()
    filename=filename.replace("/",'_')+".csv"
    #st.write(filename)
    Filelist.append(filename)
Process_files(Filelist)










