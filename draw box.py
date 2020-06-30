
# coding: utf-8

# In[ ]:


## 8个子图的共有legend,设置不好，就没有再设置。可以后期用PS处理


# In[27]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
get_ipython().magic(u'matplotlib inline')


ax = plt.figure(figsize=(30,5),dpi=300)
plt.figure(24)
plt.rc('font',family='Times New Roman')###-----------------

# ---------------------- Accuracy   -------------------------------##
ax1 = plt.subplot(241) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax1.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('ACC(%)',fontsize=5)
ax1.set_xticks(index + bar_width*2.5 )
ax1.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax1=plt.gca()
ax1.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax1.tick_params(labelsize=5)
## 设置Y轴的长度
ax1.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- SN(%) -------------------------------##
ax2 = plt.subplot(242) 
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax2.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax2.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax2.set_ylabel('SN(%)',fontsize=5)
ax2.set_xticks(index + bar_width*2.5 )
ax2.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax2=plt.gca()
ax2.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax2.tick_params(labelsize=5)
## 设置Y轴的长度
ax2.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- SP(%) -------------------------------##
ax3=plt.subplot(243)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax3.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax3.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax3.set_ylabel('SN(%)',fontsize=5)
ax3.set_xticks(index + bar_width*2.5 )
ax3.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax3=plt.gca()
ax3.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax3.tick_params(labelsize=5)

## 设置Y轴的长度
ax3.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- Precision(%) -------------------------------##
ax4=plt.subplot(244)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(True)
ax4.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax4.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax4.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax4.set_ylabel('Precision(%)',fontsize=5)
ax4.set_xticks(index + bar_width*2.5 )
ax4.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax4=plt.gca()
ax4.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax4.tick_params(labelsize=5)
## 设置Y轴的长度
ax4.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- NPV(%) -------------------------------##
ax5=plt.subplot(245)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(True)
ax5.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax5.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax5.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax5.set_ylabel('NPV(%)',fontsize=5)
ax5.set_xticks(index + bar_width*2.5 )
ax5.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax5=plt.gca()
ax5.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax5.tick_params(labelsize=5)
## 设置Y轴的长度
ax5.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax5.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax5.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- F1(%) -------------------------------##
ax6=plt.subplot(246)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(True)
ax6.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax6.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax6.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax6.set_ylabel('F1(%)',fontsize=5)
ax6.set_xticks(index + bar_width*2.5 )
ax6.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax6=plt.gca()
ax6.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax6.tick_params(labelsize=5)

## 设置Y轴的长度
ax6.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- PredictSNP -------------------------------##
ax7=plt.subplot(247)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['bottom'].set_visible(True)
ax7.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax7.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax7.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax7.set_ylabel('PredictSNP',fontsize=5)
ax7.set_xticks(index + bar_width*2.5 )
ax7.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax7=plt.gca()
ax7.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax7.tick_params(labelsize=5)
## 设置Y轴的长度
ax7.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax7.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax7.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# ---------------------- MMP -------------------------------##
ax8=plt.subplot(248)
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax8.spines['bottom'].set_visible(True)
ax8.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax8.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax8.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax8.set_ylabel('MMP ',fontsize=5)
ax8.set_xticks(index + bar_width*2.5 )
ax8.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)

y_major_locator=MultipleLocator(0.1)
ax8=plt.gca()
ax8.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax8.tick_params(labelsize=5)

for rect in rects6:
    height = rect.get_height()
    ax8.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)         
for rect in rects7:
    height = rect.get_height()
    ax8.text(rect.get_x() + rect.get_width()/2., 1.08*height,
            '%0.2f' % (height),ha='center', va='bottom',fontsize=5)  
plt.grid('off')

# plt.legend([rects6, rects7], ['MMP', 'PredictSNP'], loc = 'upper center') 
# plt.legend([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], ['MMP', 'PRE'])
plt.rcParams['figure.figsize'] = (16, 14) # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=0.2)
# plt.legend(loc='best',frameon=False, ncol=2,fontsize=5) #去掉图例边框
# ax.legend(['MMP', 'PredictSNP'],loc='upper center',frameon=False, ncol=2,fontsize=5)
plt.savefig('../python figures/SVM_RF_DeepMM.tif', dpi=300,bbox_inches ='tight') #指定分辨率保存
plt.show()


# In[ ]:


## 如果后期用PS处理，可以用下面图的图例


# In[25]:


ax1 = plt.subplot(111) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- MMP -------------------------##
MMP=(0.7960,0.8075,0.7982)
MMPStd = (0.020, 0.020, 0.018)
##------------------- PRE -------------------------##
PredictSNP=(0.5982,0.5990,0.5996)
PredictSNPStd = (0.021, 0.021, 0.028)
index = np.arange(1,9,3)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, MMP, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=MMPStd, error_kw=error_config,
                label='MMP',hatch='//')
rects7 = ax1.bar(index + 3*bar_width, PredictSNP, bar_width,
                alpha=opacity, color='black',
                yerr=PredictSNPStd, error_kw=error_config,
                label='PredictSNP',hatch='\\\\')
plt.axhline(y=0.6,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('ACC(%)',fontsize=5)
ax1.set_xticks(index + bar_width*2.5 )
ax1.set_xticklabels(('SVM', 'RF', 'DeepMM'),fontsize=5)
plt.legend(ncol =2 )
plt.show()

