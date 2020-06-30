
# coding: utf-8

# In[3]:


"""
psa_seq_pos(Sequence,pos,PSA_length)
Sequence:待截取的序列
pos：突变点的位置
PSA_length:最终需要的PSA长度，目前为了便于运算，设置为500AA
return (P0S):返回截取500AA之后的POS
"""

##----以pos为中心，截取特定长度的AA----##
def psa_seq_pos(Sequence,pos,PSA_length):

    posQ=Sequence[:int(pos)] ## 将进来的序列，以突变点为中心，分为前后两部分
    posH=Sequence[int(pos):]

    ##-------字符型变量使用前要先定义，后使用-----##
    posQN=''
    posHN=''
    SequenceN=''
    tempQ=''
    tempH=''

    if(len(posQ)<int(0.5*PSA_length)  and len(posH)<int(0.5*PSA_length)):## pos前后都小于二分之一的PSA要求长度
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
    #     tempH='X'*(int(0.5*PSA_length)-len(posH))
    #     posHN=posH +tempH 
        P0S = len(tempQ)+1

    elif(len(posQ)>int(0.5*PSA_length)and len(posH)<int(0.5*PSA_length)):## pos前大于，后小于二分之一的PSA要求长度
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
    #     tempH='X'*(int(0.5*PSA_length)-len(posH))
    #     posHN=posH +tempH 
        P0S = len(tempQ)+1

    elif(len(posQ)<int(0.5*PSA_length)and len(posH)>int(0.5*PSA_length)):## pos后大于，前小于二分之一的PSA要求长度
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
    #     tempH=posH[:int(0.5*PSA_length)]
    #     posHN = tempH 
        P0S = len(tempQ)+1

    else:                                                               ## pos前后都等于二分之一的PSA要求长度
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
    #     tempH=posH[:int(0.5*PSA_length)]
    #     posHN = tempH     
        P0S = len(tempQ)+1

    return (P0S)


# In[6]:


# def save_PSA_fasta(booksheet,sheet_name):

#     import pandas as pd
#     from pandas import DataFrame
#     from AA3T1 import mut_split   ##----之前写的py程序，包括3AA1，和序列突变位点替换 两个函数---##
#     from cut_sequence import cut_seq  ##----之前写的py程序：以突变点为中心，截取特定长度的AA序列---##
#     from mk_filefolder import mkdir  ## ----给定路径，创建文件夹，便于后期根据类型存放文件

#     fdir='../data/'+booksheet
#     pd = pd.read_excel(fdir, sheet_name=sheet_name)
#     #     pd = pd.read_excel('../data/PMD_dataset_D_mut.xlsx', sheet_name='deleterious subset')
#     Variation = pd['Variation'].tolist()
#     Name = pd['Name'].tolist()
#     Sequence = pd['Sequence']##Sequence是Series类型，Sequence[0]是string类型


#     if(sheet_name[0]=='d'): 
#         X='D'
#     else:
#         X='N'

#      ## 创建文件夹--
#     temp_dir = mkdir("../data/PSA/")
#     fadir= temp_dir + booksheet[:3]+'_'+X+'_PSA'+'.txt'
#     print(fadir)
#     PSA=open(fadir,'w')

#     #     PSA=open('../data/PMD_PSA_fatsa','w')
#     for i in range(len(Name)):
#         qian,pos,hou = mut_split(Variation[i])
#         POS = psa_seq_pos(Sequence[i],pos,500)
#         PSA.write('>'+Name[i]+'\n')
#         PSA.write(pos+'\t'+str(POS)+'\n')

#     PSA.close()

# save_PSA_fasta('PMD_dataset_N_mut.xlsx','neutral subset')

