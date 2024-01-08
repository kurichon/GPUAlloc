import csv
import pandas as pd
import numpy as np
#csv_header = ["Model","GPU","Exec Time","Efficiency"]
#with open('.preferred_gpu.csv','w+') as csv_file:    
#    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
#    writer.writeheader()
#    writer.writerows(preferred_GPU)


#df_pref_gpu = pd.read_csv('.preferred_gpu.csv')
#df_pref_gpu = df_pref_gpu.sort_values(by=[csv_header[3]])
#df_pref_gpu.to_csv('.preferred_gpu_sorted.csv',index=False)
#scheduler_exec_time_new.csv
def add_exec_time(filename,increment,scale):
    df_pref_gpu = pd.read_csv(filename)
    df_exec_time = pd.read_csv('exec_time_scheduler.csv')
    for index,row in df_pref_gpu.iterrows():
        model = row[0].split('_')[1]
        batch = row[0].split('_')[2]
        opt = row[0].split('_')[3]
        epoch = row[0].split('_')[4]
        dset = row[0].split('_')[5]
        gpu = row[1]
        for idx,row_e in df_exec_time.iterrows():
            if gpu == row_e["GPU"]:
                if model == row_e["Model"]:
                    if int(batch) == row_e["Batch Size"]:
                        if opt == row_e["Optimizer"]:
                            if int(epoch) == row_e["Epoch"]:
                                if dset == row_e["Dataset"]:
                                    df_pref_gpu.loc[index,"Execution Time"] = row_e[6]
                                    if gpu == "GTX1080":
                                        gtx1080_jct.append(row_e[6])
                                    if gpu == "RTX2070":
                                        rtx2070_jct.append(row_e[6])
                                    if gpu == "TitanX":
                                        titanx_jct.append(row_e[6])
                                        
                                    break
    df_pref_gpu.to_csv(filename,index=False)
    sum_JCT(increment,scale,gtx1080_jct,rtx2070_jct,titanx_jct)
def calc_JCT(filename,increment,scale):
    rtx2070_jct = []
    gtx1080_jct = []
    titanx_jct = []
    df_pref_gpu = pd.read_csv(filename)
    df_exec_time = pd.read_csv('exec_time_scheduler.csv')
    for index,row in df_pref_gpu.iterrows():
        model = row[0].split('_')[1]
        batch = row[0].split('_')[2]
        opt = row[0].split('_')[3]
        epoch = row[0].split('_')[4]
        dset = row[0].split('_')[5]
        gpu = row[1]
        for idx,row_e in df_exec_time.iterrows():
            if gpu == row_e["GPU"]:
                if model == row_e["Model"]:
                    if int(batch) == row_e["Batch Size"]:
                        if opt == row_e["Optimizer"]:
                            if int(epoch) == row_e["Epoch"]:
                                if dset == row_e["Dataset"]:
                                    df_pref_gpu.loc[index,"Execution Time"] = row_e[6]
                                    if gpu == "GTX1080":
                                        gtx1080_jct.append(row_e[6])
                                    if gpu == "RTX2070":
                                        rtx2070_jct.append(row_e[6])
                                    if gpu == "TitanX":
                                        titanx_jct.append(row_e[6])
                                        
                                    break
    df_pref_gpu.to_csv(filename,index=False)
    sum_JCT(increment,scale,gtx1080_jct,rtx2070_jct,titanx_jct)

#df_pref_gpu = df_pref_gpu.sort_values(by=[csv_header[2]])
#df_pref_gpu.to_csv('.preferred_gpu_sorted.csv',index=False)

def sum_JCT(increment,scale,gtx1080_jct,rtx2070_jct,titanx_jct):
    sum_gtx = 0
    sum_rtx = 0 
    sum_titanx = 0
    avg_gtx = 0
    avg_rtx = 0
    avg_titanx = 0
    
    if len(gtx1080_jct) > 0:
        for i in range(len(gtx1080_jct)):
            sum_gtx+= gtx1080_jct[i]
            gtx1080_jct[i] = sum_gtx

        avg_gtx = sum(gtx1080_jct) / len(gtx1080_jct)
    if len(rtx2070_jct) > 0:
        for i in range(len(rtx2070_jct)):
            sum_rtx+= rtx2070_jct[i]
            rtx2070_jct[i] = sum_rtx
        avg_rtx = sum(rtx2070_jct) / len(rtx2070_jct)
    if len(titanx_jct) > 0:
        for i in range(len(titanx_jct)):
            sum_titanx+= titanx_jct[i]
            titanx_jct[i] = sum_titanx

        avg_titanx = sum(titanx_jct) / len(titanx_jct)
    total_avg = (sum(gtx1080_jct)+sum(rtx2070_jct)+sum(titanx_jct)) / (len(gtx1080_jct)+len(rtx2070_jct)+len(titanx_jct))
    makespan = max([sum_gtx,sum_rtx,sum_titanx])
    to_print = "GTX1080 ={0}, RTX_2070 ={1}, TitanX ={2}, Total Avg ={3}, Makespan ={4}, Scale ={5}, Increment ={6}\n"\
        .format(avg_gtx,avg_rtx,avg_titanx,total_avg,makespan,scale,increment)
    print(to_print)
    with open("statistics.log",'a+') as f:
        f.write(to_print)
