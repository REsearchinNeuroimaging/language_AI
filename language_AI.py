import numpy as np
import torch, pickle, argparse
from torchvision import models
from nilearn import image
from nilearn.maskers import NiftiSpheresMasker
from nilearn.maskers import NiftiMasker

parser = argparse.ArgumentParser(description='AI model for classifying language organization as either typical or atypical from preprocessed RS-fMRI BOLD data')

parser.add_argument('func_file', help='4d nifti img: file path for preprocessed file in MNI152NLin2009cAsym space.')
parser.add_argument('tr', help='TR in seconds: The time between scanning volumes.')

args = parser.parse_args()

func_file = args.func_file
tr = float(args.tr)
requirements_dir = './requirements'
language_network_file = '{}/MNIasym_templates/language_network_MNIasym_symetric.nii.gz'.format(requirements_dir)
MLP_models_path = '{}/trained_models'.format(requirements_dir)

coords = [(44, 28, 5)]

seed_masker = NiftiSpheresMasker(
    coords,
    radius=10,
    detrend=True,
    standardize=True,
    t_r=tr,
    memory="nilearn_cache",
    memory_level=1,
    verbose=0,
)

brain_masker = NiftiMasker(
    smoothing_fwhm=6,
    detrend=True,
    standardize=True,
    t_r=tr,
    memory="nilearn_cache",
    memory_level=1,
    verbose=0,
)


def seed_corr(func_file):
    seed_time_series = seed_masker.fit_transform(func_file)
    brain_time_series = brain_masker.fit_transform(func_file)
    seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
    seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_correlations_fisher_z.T)
    return(seed_to_voxel_corr_img)

def npy_loader(np_file):
    tensor = torch.from_numpy(np_file).float()
    return tensor

def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    return model

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
        
def input_feature_prep(corr_img,language_network_file):
    language_network_img = image.load_img(language_network_file)
    language_network_data = language_network_img.get_fdata()
    corr_data = corr_img.get_fdata()
    corr_data[language_network_data == 0] = 0
    data = np.mean(corr_data[:,:,30:50],axis=2)
    new_data = np.zeros((3,97,115))
    for k in range(3):
        new_data[k,:,:] = data[:,:,0]
    return new_data
   

def AI_prediction(input_feature,MLP_models_path):
    sample = [torch.from_numpy(input_feature).float()]
    inputs = torch.stack([img for img in sample]) 
    model = load_model().eval()
    out = model(inputs).data.numpy()
    pred_proba = 0
    for k in range(10):
        clf = load_object('{}/model_iteration_{}'.format(MLP_models_path,k))
        proba = clf.predict_proba(out)[:,1]
        pred_proba += proba
    proba = pred_proba/10
    return(proba[0])

func_img = image.load_img(func_file)
func_data =  func_img.get_fdata()
if func_data.shape[:3] != (97, 115, 97):
    print('Invalid image dimensions. Please verify if the source file is in MNI152NLin2009cAsym space.')
else:
    corr_img = seed_corr(func_file)
    input_feature = input_feature_prep(corr_img,language_network_file)
    prediction = AI_prediction(input_feature,MLP_models_path)
    print('Probability of atypical language organisation = {0:.{1}f}'.format(prediction,4))
