
# coding: utf-8

# ## 4------提取长度为L的微环境值，for  PSA

# In[1]:


'''
get_filename(result_filepath):
            是获取软件得到的特征文件夹result_filepath下的文件名称，
            为后面根据文件名找文件做准备
            return (filename):返回该文件夹下所有文件的名称
'''
def get_filename(result_filepath):
    ## -------------- 1.获取路径下文件的名字---------------------------###
    import math
    import os  
    path = result_filepath
    files= os.listdir(path) ## files是当前路径下所有文件的名字+后缀
    filename=[]
    for i in range(len(files)):
        tem=files[i].split('.')[0]
        filename.append(tem)
    return (filename)


# In[2]:


'''
get_POS_Name_Label(original_filepath,original_sheet_name)：
        original_filepath：是与特征文件夹对应的原始样本数据路径
        original_sheet_name：打开excel中的相应的工作表  
        return(POS,ID,label)：返回原始文件的POS,ID,label，最主要的是三者的index是完全一致的
'''
def get_POS_Name_Label(original_filepath,original_sheet_name):    
    ## -------------- 2.记录突变的名称，位置，以及标签信息--------------###
    import pandas as pd
    from pandas import DataFrame
    from AA3T1 import mut_split   ##----之前写的py程序，包括3AA1，和序列突变位点替换 两个函数---##
    from PSA_pos import psa_seq_pos

    pd = pd.read_excel(original_filepath, sheet_name=original_sheet_name)

    Variation = pd['Variation'].tolist()
    Name = pd['Name'].tolist()
    Label = pd['Label'].tolist()
    Sequence =pd['Sequence']


    POS = []  ##---记录突变的位置，为提取特征做准备，特别注意，该位置为Name是严格对应关系
    ID = []  ##----记录与突变位置对应
    label =[]
    for i in range(len(Variation)):
        qian,pos,hou = mut_split(Variation[i])
        
        ## 此时的pos为原始文件的pos,需要调整为截取PSA之后的POS。
        ## 调用 psa_seq_pos函数，返回截取后的突变位置：temp_pos
        temp_pos = psa_seq_pos(Sequence[i],pos,500)
        POS.append(temp_pos)
        ID.append(Name[i])
        label.append(Label[i])
        
    return(POS,ID,label)


# In[3]:


##  对于PSSM，有20个值；对于pdo仅有1个值。下面的程序需要改成更一般性的-----


# In[4]:


'''
QH_vetor(flines,num,length)：提取并计算突变前后的vector，并返回
    flines：特征文件夹下的所有文件
    num ：突变的位置
    length :提取突变微环境的大小，这里的length是突变前（后）的微环境长度
     return(Q_vector,H_vector) ：返回计算权重后的前，后特征vector
'''

## -----------3.每个特征值，提取并计算突变前后的vector，合并成一行，并返回-----------###
##-------------提取并计算突变前后的vector，并返回------------##
def QH_vetor(flines,num,length): 
    import numpy as np
    qian_value = flines[num-length:num]
    hou_value = flines[num+1:num+length+1]
    
    ## -- 前  
    qian_join = []
    for i in range(len(qian_value)):
        tempQ =qian_value[i].strip().split('    ')## 将类似的'0.691', '0.3', '0.007'的string类型值转化为float类型
        to_floatQ = map(float,tempQ)
        qian_join.extend(to_floatQ)

    ## -- 后      
    hou_join =[]
    for i in range(len(hou_value)):
        tempH =hou_value[i].strip().split('    ')
        to_floatH = map(float,tempH)
        hou_join.extend(to_floatH)
    return(qian_join,hou_join) 


# In[5]:


'''
QHp_list3to1(Q_vector,pos_value,H_vector):将Q_vector,pos_value，H_vector转化为1*9（行）的形式
    Q_vector:突变前特征向量
    pos_value:突变点特征向量
    H_vector:突变后特征向量
    return (feature)：返回一个样本的特征。对于SS，feature是1*9（行）的形式
'''

## -----------4.将Q_vector,pos_value，H_vector转化为1*9（行）的形式----------###
def QHp_list3to1(Q_vector,pos_value,H_vector):
    ##-------将Q_vector,pos_value，H_vector保存到excel中---------------##
    ## 1----将Q_vector,pos_value，H_vector转化为1*9（行）的形式
    feature = []
    for hang in range(len(Q_vector)):
        tp = Q_vector[hang]
        feature.append(tp)

    for hang in range(len(pos_value)):
        tp = pos_value[hang]
        feature.append(tp)

    for hang in range(len(H_vector)):
        tp = H_vector[hang]
        feature.append(tp)
    return (feature)


# In[6]:


'''
get_fea_IDN_LabelN(result_filepath,fname,POS,ID,label,MicroEn_length)：
    result_filepath：使用软件所得到的特征文件夹路径
    fname：该路径下所有文件的名称
    
    POS：原始文件中的突变位置POS
    ID：原始文件中的突变名称ID
    label：原始文件中的突变标签label。上面三者的index完全对应
    
    MicroEn_length：需要提取微环境的长度。以突变点为中心，前后的长度
    return(feature,IDName,Labelname)：返回所有样本的feature,每一行是一个样本的特征。
                                        并且对应的IDName和Labelname也返回，便于后面的文件保存。
    
'''

def get_fea_IDN_LabelN(result_filepath,fname,POS,ID,label,MicroEn_length): 
    import os
    ##-------------5. 提取所有样本的特征，以及与特征对应的name和标签-----##
    feature = []  ##用于保存全部样本的特征
    IDName = [] ##用于保存全部样本的名称，与所提取的特征相对应的
    Labelname = []  ##用于保存全部样本的标签，与所提取的特征相对应的
    
    ##--传进来的参数-----##
    POS = POS
    ID = ID 
    label = label
    
    for j in range(len(fname)):
        xiabiao=ID.index(filename[j])##filename[j]在ID list中的下标，决定了突变position的值
        Position= POS[xiabiao]##突变position的值
        
        path = result_filepath
        files= os.listdir(path) ## files是当前路径下所有文件的名字+后缀
        
        f = open(path+"/"+files[j]) #打开文件 ##打开files[j] 文件
        flines=f.readlines() ## 读第j个文件

        num=int(Position)-1 ## 突变下标 = 突变位置-1
        IDname=ID[xiabiao]
        labelname=label[xiabiao]

        IDName.append(IDname)
        Labelname.append(labelname)


        pos_value = []  ## 将字符形式['0.616', '0.324', '0.145']，改写成float类型，便于后面的保存
        pos_va = flines[num].strip().split('    ')
        for p in range(len(pos_va)):
            t = float(pos_va[p])
            pos_value.append(t)

        Q_vector, H_vector = QH_vetor(flines,num,MicroEn_length)##---获取突变点前后的vector
        
        ##---将每个样本的特征，先存放在temp_feature中
        temp_feature =  QHp_list3to1(Q_vector,pos_value,H_vector)

        feature.append(temp_feature)
        f.close()
        
    return(feature,IDName,Labelname)


# In[7]:


'''
feature_name(string,length):
    string: 保存特征文件时，每列的列名称
    length: 列名称的长度
    return (feature_name_list):返回列名称list
'''
## ------ 特征名字列表---------#
def feature_name(string,length):
    feature_name_list = []
    for i in range(0,length):
        tp = ''
        temp =string + str(i)
        feature_name_list.append(temp)
    return (feature_name_list)


# In[8]:


'''
save_file(fea_res_fpath,column_name,column_na_length):
    fea_res_fpath:保存文件的位置
    column_name：列名称
    column_na_length：列名称的长度
    return('文件保存成功！')
'''
def save_file(fea_res_fpath,column_name,column_na_length):   
    ## -------------6. 获取feature，IDName，Labelname --------------###
    import pandas as pd

    feature_name_list = []
    feature_name_list = feature_name (column_name,column_na_length)

    ##-- 将feature，IDName，以及Labelname转化为dataframe，用于后面的文件保存
    fea_pd = pd.DataFrame(feature[:])
    fea_ID = pd.DataFrame(IDName[:])
    fea_Labelname = pd.DataFrame(Labelname[:])


    fea_pd.columns = [feature_name_list] ## 添加列名称列的名称
    fea_pd.insert(0,'Name',fea_ID )     #插入一列
    fea_pd.insert(1,'Label',fea_Labelname)     #插入一列

    # 保存到本地excel
    
    fea_pd.to_excel(fea_res_fpath, index=True)
    
    return('文件保存成功！')


# In[9]:


##----------------       主函数  for PMD_dataset_D_mut -- PSA -----------------##
filename = get_filename("../data/PSA/PMD_dataset_D_mut")  ## 从得到的特征文件中获取filename

## 从原始文件中获取 POS,ID,label，便于后面提取突变点微环境的特征
POS,ID,label = get_POS_Name_Label('../data/PMD_dataset_D_mut.xlsx','deleterious subset')

##从得到的特征文件夹中，获得相应的文件，并提取突变点的微环境特征，IDname, Labelname
feature,IDName,Labelname = get_fea_IDN_LabelN("../data/PSA/PMD_dataset_D_mut",filename,POS,ID,label,29)

## 保存得到的特征文件,第二个参数是特征的列名称，第三个参数是列名称的长度
save_file("../data/PSA/PMD_dataset_D_mut_PSA_59.xlsx",'psa',177)  


# In[10]:


##----------------       主函数 for PMD_dataset_N_mut -- PSA  -----------------##
filename = get_filename("../data/PSA/PMD_dataset_N_mut")  ## 从得到的特征文件中获取filename

## 从原始文件中获取 POS,ID,label，便于后面提取突变点微环境的特征
POS,ID,label = get_POS_Name_Label('../data/PMD_dataset_N_mut.xlsx','neutral subset')

##从得到的特征文件夹中，获得相应的文件，并提取突变点的微环境特征，IDname, Labelname
feature,IDName,Labelname = get_fea_IDN_LabelN("../data/PSA/PMD_dataset_N_mut",filename,POS,ID,label,29)

## 保存得到的特征文件,第二个参数是特征的列名称，第三个参数是列名称的长度
save_file("../data/PSA/PMD_dataset_N_mut_PSA_59.xlsx",'psa',177)  

