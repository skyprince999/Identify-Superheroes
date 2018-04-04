########################################################################################################################
# This code was used to preprocess the training files. The training files were cleaned
# This file was screated by downloading a jupyter notebook from my paperspace account
########################################################################################################################



# In[1]:


get_ipython().system('cp -r data_CAX/train/CAX_Superhero_Train data_CAX/train_xtra')


# In[2]:


import os

PATH = f'data_CAX/train_xtra/'
#Capture the classes in the train directory
trainClasses = os.listdir(f'{PATH}CAX_Superhero_Train')

try:
    os.remove(f'{PATH}CAX_Superhero_Train' + '/.DS_Store')
except:
    print("ERR: No DS_Store")
trainClasses = os.listdir(f'{PATH}CAX_Superhero_Train')


# In[3]:


trainClasses


# In[4]:


## Clean up the training examples

# Clean Ant-Man
allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Ant-Man/')
file = [file for file in allFiles if 'cax_antman_train33' in file][0]
source = PATH + 'CAX_Superhero_Train/' + 'Ant-Man/' + file 
destination = PATH + 'CAX_Superhero_Train/' + 'Batman/' + file

try:
    os.rename(source, destination)    
except:
    print("ERROR: Moving file " + source)


# In[5]:


# Clean Aquaman

delFiles = ['cax_aquaman_train262', 'cax_aquaman_train329', 'cax_aquaman_train346', 'cax_aquaman_train381', 'cax_aquaman_train378',
            'cax_aquaman_train392', 'cax_aquaman_train422', 'cax_aquaman_train433', 'cax_aquaman_train379']

allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Aquaman/')

for file in delFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Aquaman/' + filename
    
    try:
        os.remove(source)
    except:
        print("ERROR: Deleting file" + source)

filename = [file2 for file2 in allFiles if 'cax_aquaman_train358' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Aquaman/' + filename
destination = PATH + 'CAX_Superhero_Train/' + 'Batman/' + filename

try:
    os.rename(source, destination)    
except:
    print("ERROR: Moving file " + source)


# In[6]:


# Clean Avengers

allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Avengers/')

filename = [file2 for file2 in allFiles if 'cax_avengers_train563' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename
destination = PATH + 'CAX_Superhero_Train/' + 'Captain America/' + filename

try:
    os.rename(source, destination)    
except:
    print("ERROR: Moving file " + source)


# In[7]:


# Clean Black Panther ::: For some reason this does not work 

allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Black Panther/')
movFiles = ['cax_blackpanther_train1586', 'cax_blackpanther_train1651']

for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Black Panther/' + filename
    destination = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename
    try:
        os.rename(source, destination)    
    except:
        print("ERROR: Moving file " + source)    


# In[8]:


# Clean Captain America
allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Captain America/')
filename = [file2 for file2 in allFiles if 'cax_capamerica_train2122' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Captain America/' + filename

try:
    os.remove(source)    
except:
    print("ERROR: Deleting file " + source)
    
movFiles = ['cax_capamerica_train2263', 'cax_capamerica_train2247', 'cax_capamerica_train2272', 'cax_capamerica_train2268',
            'cax_capamerica_train2288']


for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2]
    if len(filename) > 0:
        source = PATH + 'CAX_Superhero_Train/' + 'Captain America/' + filename[0]
        destination = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename[0]
        try:
            os.rename(source, destination)    
        except:
            print("ERROR: Moving file " + source)    


# In[9]:


# Clean Ghost Rider

allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Ghost Rider/')
delFiles = ['cax_ghostrider_train2560', 'cax_ghostrider_train2662']

for file in delFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Ghost Rider/' + filename
    try:
        os.remove(source)    
    except:
        print("ERROR: Deleting file " + source)

movFiles = ['cax_ghostrider_train2526', 'cax_ghostrider_train2573']
for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Ghost Rider/' + filename
    destination = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename
    try:
        os.rename(source, destination)    
    except:
        print("ERROR: Moving file " + source)

filename = [file2 for file2 in allFiles if 'cax_ghostrider_train2668' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Ghost Rider/' + filename
destination = PATH + 'CAX_Superhero_Train/' + 'Spiderman/' + filename
try:
    os.rename(source, destination)    
except:
    print("ERROR: Moving file " + source)


# In[10]:


# Clean Hulk

# delete files
allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Hulk/')

filename = [file2 for file2 in allFiles if 'cax_hulk_train3031' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Hulk/' + filename
try:
    os.remove(source)    
except:
    print("ERROR: Deleting file " + source)

#move files
movFiles = ['cax_hulk_train2818', 'cax_hulk_train2851', 'cax_hulk_train2848', 'cax_hulk_train2883', 'cax_hulk_train2904',
           'cax_hulk_train2899', 'cax_hulk_train3035', 'cax_hulk_train3073', 'cax_hulk_train3111']
for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Hulk/' + filename
    destination = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename
    try:
        os.rename(source, destination)    
    except:
        print("ERROR: Moving file " + source)


# In[11]:


allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Iron Man/')

filename = [file2 for file2 in allFiles if 'cax_ironman_train3721' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Iron Man/' + filename
destination = PATH + 'CAX_Superhero_Train/' + 'Hulk/' + filename
try:
    os.rename(source, destination)    
except:
    print("ERROR: Moving file " + source)

filename = [file2 for file2 in allFiles if 'cax_ironman_train3763' in file2][0]
source = PATH + 'CAX_Superhero_Train/' + 'Iron Man/' + filename
try:
    os.remove(source)    
except:
    print("ERROR: Deleting file " + source)

movFiles = ['cax_ironman_train3342', 'cax_ironman_train3358', 'cax_ironman_train3357', 'cax_ironman_train3361', 
            'cax_ironman_train3370', 'cax_ironman_train3395', 'cax_ironman_train3478', 'cax_ironman_train3488',
            'cax_ironman_train3485', 'cax_ironman_train3502', 'cax_ironman_train3512', 'cax_ironman_train3540', 
            'cax_ironman_train3591', 'cax_ironman_train3597', 'cax_ironman_train3610', 'cax_ironman_train3633', 
            'cax_ironman_train3646', 'cax_ironman_train3703', 'cax_ironman_train3738', 'cax_ironman_train3735',
            'cax_ironman_train3738', 'cax_ironman_train3741', 'cax_ironman_train3747', 'cax_ironman_train3751',
            'cax_ironman_train3752', 'cax_ironman_train3764', 'cax_ironman_train3785', 'cax_ironman_train3793',
            'cax_ironman_train3810', 'cax_ironman_train3815']

for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Iron Man/' + filename
    destination = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename
    try:
        os.rename(source, destination)    
    except:
        print("ERROR: Moving file " + source)



# In[12]:


# Clean spider4man
allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Spiderman/')
movFiles = ['cax_spiderman_train4664', 'cax_spiderman_train4663']
for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Spiderman/' + filename
    destination = PATH + 'CAX_Superhero_Train/' + 'Avengers/' + filename

    try:
        os.rename(source, destination)    
    except:
        print("ERROR: Moving file " + source)    


# In[13]:


# Clean superman
allFiles = os.listdir(PATH + 'CAX_Superhero_Train/' + 'Superman/')
movFiles =['cax_superman_train4725', 'cax_superman_train4809']

for file in movFiles:
    filename = [file2 for file2 in allFiles if file in file2][0]
    source = PATH + 'CAX_Superhero_Train/' + 'Superman/' + filename
    destination = PATH + 'CAX_Superhero_Train/' + 'Batman/' + filename
    try:

        os.rename(source, destination)    
    except:
        print("ERROR: Moving file " + source)

filename = [file2 for file2 in allFiles if 'cax_superman_train5149' in file2][0]    
source = PATH + 'CAX_Superhero_Train/' + 'Superman/' + filename
try:
    os.remove(source)    
except:
    print("ERROR: Deleting file " + source)


# In[14]:


import numpy as np

# Print out the number of samples for each category
for classes in trainClasses:
    source = PATH + 'CAX_Superhero_Train/' + classes + "/" 
    
    try:
        os.remove(source+".DS_Store")
    except:
        print("Error: No DS_Store found in category: "+ classes)
    
    files = os.listdir(source)
    
    print("Superhero: " + classes + " : Sample Size = " + str(len(files)))
    
    # Create directory in Validation folder
    directory = PATH + 'valid/' + classes 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Select 40 random samples
    index = np.random.choice(range(len(files)), int(len(files)*0.2) ,replace=False)
    
    for idx in index:
        print(files[idx])
        source = PATH + 'CAX_Superhero_Train/' + classes + "/" + files[idx]
        destination = PATH + 'valid/' + classes + "/" + files[idx]
        try:
            os.rename(source, destination)    
        except:
            print("ERROR: Moving file " + files[idx])


# In[17]:


get_ipython().system('mv data_CAX/train_xtra/valid data_CAX/valid_xtra')


# In[18]:


get_ipython().system('rm -rf data_CAX/train_xtra/CAX_Superhero_Train')

