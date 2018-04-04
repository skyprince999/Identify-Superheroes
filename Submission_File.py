######################################################################################################
# This file takes a pre-trained model and uses it to create a file in the required submission format.
# There were some mismatches between the names of the categories (Superhero names) and the method in 
# which the images were arranged in the train folder. This file ensures that the output is in a format
# acceptable to the scoring engine
######################################################################################################

# This loads all the main external libs we'll use
from fastai.imports import *

# Loads the required transformation and learners
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

torch.cuda.is_available()
torch.backends.cudnn.enabled

PATH = f'data_CAX/'
#Capture the classes in the train directory
trainClasses = os.listdir(f'{PATH}CAX_Superhero_Train')

def get_data(sz,bs):
    tfms=tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom =1.3)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, trn_name='CAX_Superhero_Train', val_name = 'valid',
                                          test_name = 'CAX_Superhero_Test', num_workers = 4)
    #return data if sz > 300 else data.resize(340, 'tmp')
    return data


PATH = f'data_CAX/'
bs =  8
sz = 264 # From above data vizualization on image size
arch= resnext101_64  ## resnet34 - This was giving an accuracy of around 60%
data = get_data(sz, bs) ## resnext_101_64  and lr =0.005 with 4 epochs & cycle len =2, cycle_mult = 3 gives around 75%

learn = ConvLearner.pretrained(arch, data, precompute=False)
learn.load('cax_final_13032018_1830_True.h5')

## Calculate test predictions
log_preds,y = learn.TTA(is_test = True)

probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs, axis=1)

pred_classes = [data.classes[i].lower().replace(" ","_") for i in preds]
print(len(pred_classes))

files = os.listdir(f'{PATH}CAX_Superhero_Test')
len(files)

if '.DS_Store' in files:
    os.remove(f'{PATH}CAX_Superhero_Test/.DS_Store')
else:
    print("No such file")

filename = [i[:-4] for i in files]

submission = pd.DataFrame({'filename': filename, 'Superhero': pred_classes})

submissionOrder = pd.read_csv('Superhero_Submission_Format.csv')
submissionOrder = submissionOrder[['filename']]

merged = submissionOrder.merge(submission, on= 'filename', how ='outer')

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

merged.to_csv('Submission_cax_final_13032018_1830_True.h5.csv', index=False)

