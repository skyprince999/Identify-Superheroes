########################################################################################################
# Image classification with Convolutional Neural Networks
# CrowdAnalytix - Identifying Superheroes
# 
# https://www.crowdanalytix.com/contests/identifying-superheroes-from-product-images
# 
# Date: 24th March 2018
# 
# This is run after preprocessing the training data :: CAX Identifying Superheroes _Preprocessing.py
# Uses all data/images that were web-scraped using :: Scrape_More_Data.py
########################################################################################################

# In[13]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


# This file contains all the main external libs we'll use
from fastai.imports import *


# In[15]:


from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[16]:


torch.cuda.is_available()


# In[17]:


torch.backends.cudnn.enabled


# In[18]:


PATH = f'data_CAX/'
#Capture the classes in the train directory
trainClasses = os.listdir(f'{PATH}train_xtra')
trainClasses


# In[19]:


# Cleaning directories 
import os

for classes in trainClasses:
    try:
        os.remove(f'{PATH}train_xtra/'+ classes + '/.DS_Store')
    except:
        print("No .DS_Store file - train")
        
    try:
        os.remove(f'{PATH}valid_xtra/'+ classes + '/.DS_Store')
    except:
        print("No .DS_Store file - valid")
    


# In[20]:


def get_data(sz,bs):
    tfms=tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom =1.3)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, trn_name='train_xtra', val_name = 'valid_xtra',
                                          test_name = 'CAX_Superhero_Test', num_workers = 4)
    return data if sz > 300 else data.resize(340, 'tmp')
    #return data


# Before using a different architecture don't forget to download the pre-computed weights into the weights folder
# wget http://files.fast.ai/models/weights.tgz

# In[21]:


# Uncomment the below if you need to reset your precomputed activations
shutil.rmtree(f'{PATH}tmp', ignore_errors=True)


# In[24]:


PATH = f'data_CAX/'
bs =  64
sz = 260 # From above data vizualization on image size
arch= resnext101_64  ## resnet34 - This was giving an accuracy of around 60%


# In[25]:


data = get_data(sz, bs)  


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[11]:


lrf=learn.lr_find()


# In[12]:


learn.sched.plot_lr()


# In[13]:


learn.sched.plot()

lrf = 0.1   # Seems to be a good learning rate for the dataset
            # This is strange because last week I remember 0.1 being to high. Need to investigate this further
# In[24]:


lrf = 0.1


# In[25]:


get_ipython().run_line_magic('time', 'learn.fit(0.1, 4 )')

Changed the learning rate to 1e-2
# In[28]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 3 )')


# In[29]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 3 )')

This seems to be overfitting the data since the training loss is far lower than the validation loss
# In[31]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 3, cycle_len=1, cycle_mult=3 )')


# In[32]:


# 24th March 2018 - Saving model
learn.save('cax_final_24032018_1212')
learn.load('cax_final_24032018_1212')

Creating another iteration with a change to the batch size (bs = 34) and size of the image (sz = 254)
I'll probably create an ensemble of models 
# In[ ]:


sz = 254
bs = 34
data = get_data(sz, bs)
learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf=learn.lr_find()
learn.sched.plot()


# In[ ]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 3 )')


# In[ ]:


## Adding a precompute =False
learn.precompute = False
learn.fit(0.005, 4, cycle_len=2, cycle_mult=3 )


# In[1]:


get_ipython().run_line_magic('pinfo', 'learn.fit')


# In[ ]:


learn.sched.plot_lr() # After running  a precompute = False This went for 4+ hours !!!! with around 76% accuracy


# In[ ]:


learn.save('cax_final_13032018_0323')
learn.load('cax_final_13032018_0323')

## Run the test prediction codes.


# In[ ]:


learn.sched.plot_lr()


# In[ ]:


learn.save('cax_final_11032018_2030')
learn.load('cax_final_11032018_2030')


# Creating a confusion matrix

# In[ ]:


log_preds,y = learn.TTA()

probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs, axis=1)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)


# In[ ]:


accuracy_np(preds,y)


# In[ ]:


img = data.val_ds[2][0]
plt.imshow(img)


# In[ ]:


## Calculate test predictions
log_preds,y = learn.TTA(is_test = True)


# In[ ]:


probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs, axis=1)


# In[ ]:


pred_classes = [data.classes[i].lower().replace(" ","_") for i in preds]
pred_classes[:5]


# In[ ]:


len(pred_classes)


# In[ ]:


files = os.listdir(f'{PATH}CAX_Superhero_Test')
len(files)


# In[ ]:


if '.DS_Store' in files:
    os.remove(f'{PATH}CAX_Superhero_Test/.DS_Store')
else:
    print("No such file")


# In[ ]:


filename = [i[:-4] for i in files]
filename[:5]


# In[ ]:


submission = pd.DataFrame({'filename': filename, 'Superhero': pred_classes})
submission[:5]


# In[ ]:


submissionOrder = pd.read_csv('Superhero_Submission_Format.csv')
submissionOrder = submissionOrder[['filename']]
submissionOrder[:5]


# In[ ]:


merged = submissionOrder.merge(submission, on= 'filename', how ='outer')
merged[:10]


# In[ ]:


temp = list(set(list(merged['Superhero'])))


# In[ ]:


temp


# In[ ]:


superhero = list()
for idx, row in merged.iterrows():
    if row['Superhero'] == 'catwoman':
        temp = 'cat_woman'
    elif row['Superhero'] == 'ant-man':
        temp = 'ant_man'
    elif row['Superhero'] in ['spiderman', 'superman', 'aquaman', 'batman']:
        temp = row['Superhero'][:-3] + '_man'
    elif row['Superhero'] == 'ghost_rider':
        temp = 'ghostrider'
    else:
        temp = row['Superhero']
    superhero.append(temp)

merged['Superhero'] = superhero

merged[:10]


# In[ ]:


merged.to_csv('Submission_13032018_0323.csv', index=False)


# ## Analyzing results: looking at pictures

# As well as looking at the overall metrics, it's also a good idea to look at examples of each of:
# 1. A few correct labels at random
# 2. A few incorrect labels at random
# 3. The most correct labels of each class (ie those with highest probability that are correct)
# 4. The most incorrect labels of each class (ie those with highest probability that are incorrect)
# 5. The most uncertain labels (ie those with probability closest to 0.5).

# In[ ]:


print("Length of Training dataset: " +str(len(data.trn_ds)))
print("Length of Validation dataset: " +str(len(data.val_ds)))


# In[ ]:


# This is the label for a val data
data.val_y


# In[ ]:


# from here we know that 'cats' is label 0 and 'dogs' is label 1.
data.classes


# help(learn.predict)

# In[ ]:


# this gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()
log_preds.shape


# In[ ]:


log_preds[:10]


# In[ ]:


preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,1])        # pr(dog)


# In[ ]:


preds


# In[ ]:


probs


# In[ ]:


def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)


# In[ ]:


def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)


# In[ ]:


def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


# In[ ]:


def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))


# In[ ]:


# 1. A few correct labels at random
plot_val_with_title(rand_by_correct(True), "Correctly classified")


# In[ ]:


# 2. A few incorrect labels at random
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")


# In[ ]:


def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)


# In[ ]:


for idx in range(len(data.classes)):
    plot_val_with_title(most_by_correct(idx, True), "Most correct "+ data.classes[idx])


# In[ ]:


for idx in range(len(data.classes)):
    plot_val_with_title(most_by_correct(idx, False), "Most incorrect "+ data.classes[idx])


# In[ ]:


most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")

