
# coding: utf-8

# In[ ]:


## ------ 特征名字列表---------#
def feature_name(string,length):
    feature_name_list = []
    for i in range(0,length):
        tp = ''
        temp =string + str(i)
        feature_name_list.append(temp)
    return feature_name_list

