# Classification analysis of social perceptual features based on fMRI reponses to aduiovisual perception of these features
#
# Sections:
#   1. Make dataset for full brain classification analysis
#   2. Make dataset for ROI analysis
#   3. Tune hyperparameters for neural network (full brain dataset)
#   4. Run between-subject classification (full brain dataset)
#   5. Run between-subject classification (ROI dataset)
#   6. Run between-subject classification (ROI dataset, with feature selection)
#   7. Permute null distribution of chance level prediction accuracy (full brain dataset)
#
# Severi Santavirta & Sanaz Nazari-Farsani, last modification 06.03.2023

##------------------------------------------------------------------------------------------------------------------------------------
#   1. Make dataset for full brain classification analysis

import mvpa2.suite as mvpa
import os
import tempfile
import numpy as np

os.nice(20)

directory = '/path/' # Directory of preprocessed .nii fMRI data files for each subject
mask= '/path/brainmask.nii'  # Path to brainmask
output = '/path/' # Output directory

os.chdir(output)

# Labels as text file
attr = mvpa.SampleAttributes('/path/labels.csv') # The file should indicate the target label and a time window for each fMRI time point (columns = [labels,time-windows],rows = fMRI time points)

dhandle = mvpa.OpenFMRIDataset(directory)
dhandle.get_subj_ids()

lstfn = os.listdir(directory)
lstfn.sort()

evds_list=[]
k=0
for filename in lstfn:
    if filename.endswith(".nii"):
        k=k+1
        print(k)
        bold_fname = os.path.join(directory,filename)  
        fds = mvpa.fmri_dataset(samples = bold_fname, targets = attr.targets, chunks = attr.chunks, mask = mask) # Create fmri dataset
        fds.shape
  
        events = mvpa.find_events(targets=fds.sa.targets, chunks=fds.sa.chunks)
        
        #print events
        events = [ev for ev in events if ev['targets'] in ['Feeding,','Using_an_object,','Crying,','Antisocial_behaviour,','Running,','Prosocial_behaviour,','Communication,','Body-movement,','Searching,','Walking,','Play,']] #Hard coded target labels
        TR = np.median(np.diff(fds.sa.time_coords))
        
        for ev in events:
            ev['onset'] = (ev['onset'] * TR)
            ev['duration'] = ev['duration'] * TR
        evds = mvpa.fit_event_hrf_model(fds, events, time_attr='time_coords', condition_attr=('targets', 'chunks')) # Fit HRF model for events

        tempdir = '/tempdir/'
        tempfile.mkdtemp()
        evds_list.append(evds)

mvpa.h5save('hdf5_localizer_11clusters_10chunks_lowlevelcsfwmnonsocial_regressed_zscoreinput', evds_list, compression =9)

##------------------------------------------------------------------------------------------------------------------------------------
#   2. Make dataset for ROi analysis

import mvpa2.suite as mvpa
import os
import numpy as np

os.nice(20)

directory = '/path/' # Directory of preprocessed .nii fMRI data files for each subject
mask_directory = '/path/' # Directory of .nii brainmasks for ROIs
output = '/path/' # Output directory

# Labels as text file
attr = mvpa.SampleAttributes('/path/labels.csv') # The file should indicate the target label and a time window for each fMRI time point (columns = [labels,time-windows],rows = fMRI time points)
dhandle = mvpa.OpenFMRIDataset(directory)
dhandle.get_subj_ids()

lstfn = os.listdir(directory)
lstfn.sort()

os.chdir(output)
k = 0
for mask in os.listdir(mask_directory):
    evds_list=[]
    mask_fname = os.path.join(mask_directory, mask)
    mask_name = mask.split(".")[0]
    for filename in lstfn:
        if filename.endswith(".nii"):
            k=k+1
            print(mask_name + ": sub %.0f" % k)
            bold_fname = os.path.join(directory,filename)  
            fds = mvpa.fmri_dataset(samples = bold_fname, targets = attr.targets, chunks = attr.chunks, mask = mask_fname) # Create fmri dataset
            fds.shape
            events = mvpa.find_events(targets=fds.sa.targets, chunks=fds.sa.chunks) # Events
            events = [ev for ev in events if ev['targets'] in ['Feeding,','Using_an_object,','Crying,','Antisocial_behaviour,','Running,','Prosocial_behaviour,','Communication,','Body-movement,','Searching,','Walking,','Play,']] #Hard coded target labels
            TR = np.median(np.diff(fds.sa.time_coords)) #TR
            for ev in events:
                ev['onset'] = (ev['onset'] * TR)
                ev['duration'] = ev['duration'] * TR
            evds = mvpa.fit_event_hrf_model(fds, events, time_attr='time_coords', condition_attr=('targets', 'chunks')) # Fit HRF model for events
            evds_list.append(evds)
    outfile = "dataset_" + mask_name + ".hdf5"
    mvpa.h5save(outfile, evds_list, compression =9) #Save ROI dataset
    indices = fds.fa.voxel_indices
    ind = mask_name + '_indices.txt' # Save ROI voxel indices as text if needed 
    np.savetxt(ind, indices)

##------------------------------------------------------------------------------------------------------------------------------------
#   3. Tune hyperparameters for neural network (full brain dataset)

import csv
import mvpa2.suite as mvpa
from mvpa2.base.hdf5 import h5load
import numpy as np
import os
from scipy import stats
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.neural_network import MLPClassifier
os.chdir("/path/") # Directory of the helper functions file
from h5_helperfunctions_V2 import save_dict_to_hdf5

os.nice(20)

#Load data
print("Loading data...")
filepath = '/path/' # Path of the input data file

ds_all = h5load(filepath)
print(" data loaded")

# Inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    sd.sa['subject'] = np.repeat(i, len(sd))
    # Number of subjects
    nsubjs = len(ds_all)
    # Number of categories
    ncats = len(ds_all[0].UT)
    
# Feature selection
nf = 3000
fselector = mvpa.FixedNElementTailSelector(nf, tail='upper', mode='select', sort=False)
sbfs = mvpa.SensitivityBasedFeatureSelection(mvpa.OneWayAnova(), fselector, enable_ca=['sensitivities'])    

# Test only a limited set of predefined hyperparameters
reader = csv.reader(open('/path/parameter_values_list.csv','rb'),delimiter=',') # Set of tested hyper parameters (rows: spefific combinations of hyperparameters, columns: different hyperparameters)
header = next(reader)

# Use only one classifier
name = "Neural Net"

# Run classifier with the pre-selected hyperparameter combinations
k = 0
tuning_results = dict()

for row in reader:
    k=k+1
    hidden_layers = int(row[0]) # Hard coded based on the csv file
    nodes = int(row[1]) # Hard coded based on the csv file
    alph = float(row[2]) # Hard coded based on the csv file
    iterations = int(row[3]) # Hard coded based on the csv file
    
    if hidden_layers == 1:
        clf = MLPClassifier(hidden_layer_sizes=(nodes,),alpha=alph,max_iter=iterations) #Neural net classifier
    elif hidden_layers == 2:
        clf = MLPClassifier(hidden_layer_sizes=(nodes,nodes,),alpha=alph,max_iter=iterations) #Neural net classifier
    
    wrapped_clf = SKLLearnerAdapter(clf)
    fsclf = mvpa.FeatureSelectionClassifier(wrapped_clf, sbfs)
         
    stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)

    # Run between subject classification
    ds_mni = mvpa.vstack(ds_all)
    mni_start_time = mvpa.time.time() # Timer
    cv = mvpa.CrossValidation(fsclf, mvpa.NFoldPartitioner(attr='subject'),
    errorfx=lambda p,t: np.mean(p==t), enable_ca = ['stats'])
    bsc_mni_results = cv(ds_mni)
    
    print("Done in %.1f seconds" % (mvpa.time.time() - mni_start_time,))
    print("Parameter set %d: Average classification accuracy: %.2f +/-%.3f, time: %.1f minutes" % (k,np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1),((mvpa.time.time() - mni_start_time)/60)))
    
    accuracy_bs = bsc_mni_results.S
    confmat_bs = cv.ca.stats.matrix
    
    individual_confmat = np.zeros((len(confmat_bs),len(confmat_bs),len(cv.ca.stats.matrices)))
    for I in range(len(cv.ca.stats.matrices)):
        individual_confmat[:,:,I] = cv.ca.stats.matrices[I].matrix
    
    cv.ca.stats.labels
    tmp_confmat=dict()
    
    for dd in cv.ca.stats.matrices:
        tmp_confmat[dd]=dd._ConfusionMatrix__matrix
    
    keys=['subject_{}'.format(i) for i in range(0, nsubjs)]
    all_confmatx=dict(zip(keys,list(tmp_confmat.values())))
    acc_std_bs = np.concatenate((np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)), axis=None)   
    
    results_fulllist = dict()
    results_fullconfmatx = dict()
    results_fulllist[name] = {} 
    results_fulllist[name]['labels'] = cv.ca.stats.labels
    results_fulllist[name]['acc_avg'] =  acc_std_bs
    results_fulllist[name]['acc_all'] =  accuracy_bs 
    results_fulllist[name]['confmat'] =  confmat_bs
    results_fulllist[name]['confmat_all'] =  individual_confmat 
    results_fulllist[name]['featureimportances_indx'] =  fsclf.mapper.slicearg
    results_fulllist[name]['time'] = (mvpa.time.time() - mni_start_time)/60
    results_fullconfmatx[name] = all_confmatx

    # Save results file
    os.chdir("/path/") # Output directory 
    fname = 'fullbrain_classification_featureselection_results_hiddenlayers_'+str(hidden_layers)+'_nodes_'+str(nodes)+'_alpha_'+str(alph)+'_maxiter_'+str(iterations)
    fname = fname.replace('.','')+'.h5'
    save_dict_to_hdf5(results_fulllist, fname)
    
    # Save tuning results
    tuning_results[str(k)] = {}
    tuning_results[str(k)]['hidden_layers'] = hidden_layers
    tuning_results[str(k)]['nodes'] = nodes
    tuning_results[str(k)]['alpha'] = alph
    tuning_results[str(k)]['max_iter'] = iterations
    tuning_results[str(k)]['time'] = (mvpa.time.time() - mni_start_time)/60
    tuning_results[str(k)]['accuracy'] = acc_std_bs

fname = 'fullbrain_classification_featureselection_parameter_tuning_results.h5'
save_dict_to_hdf5(tuning_results,fname)

##------------------------------------------------------------------------------------------------------------------------------------
#   4. Run between-subject classification (full brain dataset)
import csv
import mvpa2.suite as mvpa
from mvpa2.base.hdf5 import h5load
import numpy as np
import os
from scipy import stats
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.neural_network import MLPClassifier
os.chdir("/path/") # Directory of the helper functions file
from h5_helperfunctions_V2 import save_dict_to_hdf5

os.nice(20)

print("Loading data...")
filepath= '/path/' # Path of the input data file

ds_all = h5load(filepath)
print(" data loaded")

accuracy_bs=[]
accyracy_ws=[]

# Inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    sd.sa['subject'] = np.repeat(i, len(sd))
    # Number of subjects
    nsubjs = len(ds_all)
    # Number of categories
    ncats = len(ds_all[0].UT)
    
    print("%d subjects" % len(ds_all))
    print("Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
    print("Stimulus categories: %s" % ', '.join(ds_all[0].UT))


# Define the classifier
name = "Neural Net"
clf = MLPClassifier(hidden_layer_sizes=(100,100,),alpha=1,max_iter=500) #Neural net classifier, tuned hyperparameters     
    
keys=['subject_{}'.format(i) for i in range(0, nsubjs)]
results_fulllist = dict()
results_fullconfmatx = dict()

wrapped_clf = SKLLearnerAdapter(clf)
    
# Feature selection helpers
nf = 3000
fselector = mvpa.FixedNElementTailSelector(nf, tail='upper', mode='select', sort=False)
sbfs = mvpa.SensitivityBasedFeatureSelection(mvpa.OneWayAnova(), fselector, enable_ca=['sensitivities'])
fsclf = mvpa.FeatureSelectionClassifier(wrapped_clf, sbfs)
 
stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)

# Between subject crocc-validation
print("between-subject (anatomically aligned)...", cr=False, lf=False)
ds_mni = mvpa.vstack(ds_all)
mni_start_time = mvpa.time.time()
cv = mvpa.CrossValidation(fsclf, mvpa.NFoldPartitioner(attr='subject'),
errorfx=lambda p,t: np.mean(p==t), enable_ca = ['stats'])
bsc_mni_results = cv(ds_mni)

print("done in %.1f seconds" % (mvpa.time.time() - mni_start_time,))
print(  "Average classification accuracies for:",name)
print("between-subject (anatomically aligned): %.2f +/-%.3f"
% (np.mean(bsc_mni_results),
np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)))

accuracy_bs = bsc_mni_results.S
confmat_bs = cv.ca.stats.matrix

individual_confmat = np.zeros((len(confmat_bs),len(confmat_bs),len(cv.ca.stats.matrices)))
for I in range(len(cv.ca.stats.matrices)):
    individual_confmat[:,:,I] = cv.ca.stats.matrices[I].matrix

cv.ca.stats.labels
tmp_confmat=dict()

for dd in cv.ca.stats.matrices:
    tmp_confmat[dd]=dd._ConfusionMatrix__matrix
    
all_confmatx=dict(zip(keys,list(tmp_confmat.values())))
acc_std_bs = np.concatenate((np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)), axis=None)   

results_fulllist[name] = {} 
results_fulllist[name]['labels'] = cv.ca.stats.labels
results_fulllist[name]['acc_avg'] =  acc_std_bs
results_fulllist[name]['acc_all'] =  accuracy_bs 
results_fulllist[name]['confmat'] =  confmat_bs
results_fulllist[name]['confmat_all'] =  individual_confmat 
results_fulllist[name]['featureimportances_indx'] =  fsclf.mapper.slicearg
results_fullconfmatx[name] = all_confmatx

### Save results file
os.chdir("/path/") # Output directory
save_dict_to_hdf5(results_fulllist, 'fullbrain_classification_featureselection_results.h5')

### Save feature indices
f = open("fullbrain_featureselection_voxel_indices.csv","wb")
cr = csv.writer(f,delimiter=',')
cr.writerow(results_fulllist[name]['featureimportances_indx'])
f.close()

##------------------------------------------------------------------------------------------------------------------------------------
#   5. Run between-subject classification (roi dataset)

import os
os.nice(20)

import mvpa2.suite as mvpa
from mvpa2.base.hdf5 import h5load
import numpy as np
from scipy import stats
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.neural_network import MLPClassifier
os.chdir("/h5_helper_functions/") # Directory of the helper functions file
from h5_helperfunctions_V2 import save_dict_to_hdf5

# Directory of the input files (roi dataset)
datapath = '/path/'

files = os.listdir(datapath)
input_files = list()
for file in files:
    if file.endswith(".hdf5"):
        input_files.append(os.path.join(datapath,file))
        
for input_file in input_files:
    roiname_ext = input_file.split("zscoreinput_")[-1] # Hard coded naming
    roiname = roiname_ext.split(".")[0]

    print("Loading data: " + roiname)
    ds_all = h5load(input_file)
    print("Data loaded")
    
    accuracy_bs=[]
    accyracy_ws=[]
    
    # Inject the subject ID into all datasets
    for i, sd in enumerate(ds_all):
        sd.sa['subject'] = np.repeat(i, len(sd))
        # Number of subjects
        nsubjs = len(ds_all)
        # Number of categories
        ncats = len(ds_all[0].UT)
        # Number of run   
    
    ## Classifier
    name = "Neural Net"
    clf = MLPClassifier(hidden_layer_sizes=(100,100),alpha=1,max_iter=500) #Neural net classifier, tuned parameters
    keys=['subject_{}'.format(i) for i in range(0, nsubjs)]
    results_fulllist = dict()
    results_fullconfmatx = dict()
    wrapped_clf = SKLLearnerAdapter(clf)
    
    # Choose errorfcn
    stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)
    
    # Between subject classification 
    ds_mni = mvpa.vstack(ds_all)
    mni_start_time = mvpa.time.time()
    
    print("Starting cross-validation")
    cv = mvpa.CrossValidation(wrapped_clf, mvpa.NFoldPartitioner(attr='subject'),
    errorfx=lambda p,t: np.mean(p==t), enable_ca = ['stats'])
    bsc_mni_results = cv(ds_mni)
    
    print("Between-subject classification done in %.1f seconds" % (mvpa.time.time() - mni_start_time,))
    print("Average classification accuracies for:",name)
    print("between-subject (anatomically aligned): %.2f +/-%.3f"
    % (np.mean(bsc_mni_results),
    np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)))
    
    accuracy_bs = bsc_mni_results.S
    confmat_bs = cv.ca.stats.matrix
    
    individual_confmat = np.zeros((len(confmat_bs),len(confmat_bs),len(cv.ca.stats.matrices)))
    for I in range(len(cv.ca.stats.matrices)):
        individual_confmat[:,:,I] = cv.ca.stats.matrices[I].matrix
    
    cv.ca.stats.labels
    tmp_confmat=dict()
    
    for dd in cv.ca.stats.matrices:
        tmp_confmat[dd]=dd._ConfusionMatrix__matrix
        
    all_confmatx=dict(zip(keys,list(tmp_confmat.values())))
    acc_std_bs = np.concatenate((np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)), axis=None)   
    
    results_fulllist[name] = {} 
    results_fulllist[name]['labels'] = cv.ca.stats.labels
    results_fulllist[name]['acc_avg'] =  acc_std_bs
    results_fulllist[name]['acc_all'] =  accuracy_bs 
    results_fulllist[name]['confmat'] =  confmat_bs
    results_fulllist[name]['confmat_all'] =  individual_confmat 
    results_fullconfmatx[name] = all_confmatx
    
    os.chdir("/path/") # Output directory
    out = 'roi_classification_results_' + roiname + '.h5'
    save_dict_to_hdf5(results_fulllist,out)
    print("Classification completed.")

##------------------------------------------------------------------------------------------------------------------------------------
#   6. Run between-subject classification (roi dataset, feature selection)
    
import os
os.nice(20)

import mvpa2.suite as mvpa
from mvpa2.base.hdf5 import h5load
import numpy as np
from scipy import stats
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.neural_network import MLPClassifier
os.chdir("/path/") # Directory of the helper functions file 
from h5_helperfunctions_V2 import save_dict_to_hdf5

# Directory of the input files (roi dataset, own datafile for each roi)
datapath = '/path/'

files = os.listdir(datapath)
input_files = list()
for file in files:
    if file.endswith(".hdf5"):
        input_files.append(os.path.join(datapath,file))
        
for input_file in input_files:
    roiname_ext = input_file.split("zscoreinput_")[-1] #Hard coded naming
    roiname = roiname_ext.split(".")[0]

    print("Loading data: " + roiname)
    ds_all = h5load(input_file)
    print("Data loaded")
    
    accuracy_bs=[]
    accyracy_ws=[]
    
    # Inject the subject ID into all datasets
    for i, sd in enumerate(ds_all):
        sd.sa['subject'] = np.repeat(i, len(sd))
        # Number of subjects
        nsubjs = len(ds_all)
        # Number of categories
        ncats = len(ds_all[0].UT)
        # Number of run   
    
    # Feature selection
    nf = 119 # The size of the smallest roi.
    fselector = mvpa.FixedNElementTailSelector(nf, tail='upper', mode='select', sort=False)
    sbfs = mvpa.SensitivityBasedFeatureSelection(mvpa.OneWayAnova(), fselector, enable_ca=['sensitivities'])    
    
    ## Classifier
    name = "Neural Net"
    clf = MLPClassifier(hidden_layer_sizes=(100,100),alpha=1,max_iter=500) #Neural net classifier, tuned parameters
    keys=['subject_{}'.format(i) for i in range(0, nsubjs)]
    results_fulllist = dict()
    results_fullconfmatx = dict()
    wrapped_clf = SKLLearnerAdapter(clf)
    fsclf = mvpa.FeatureSelectionClassifier(wrapped_clf, sbfs)
    
    # Choose errorfcn
    stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)
    
    # Between subject classification 
    ds_mni = mvpa.vstack(ds_all)
    mni_start_time = mvpa.time.time()
    
    print("Starting cross-validation")
    cv = mvpa.CrossValidation(fsclf, mvpa.NFoldPartitioner(attr='subject'),
    errorfx=lambda p,t: np.mean(p==t), enable_ca = ['stats'])
    bsc_mni_results = cv(ds_mni)
    
    print("Between-subject classification done in %.1f seconds" % (mvpa.time.time() - mni_start_time,))
    print("Average classification accuracies for:",name)
    print("between-subject (anatomically aligned): %.2f +/-%.3f"
    % (np.mean(bsc_mni_results),
    np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)))
    
    accuracy_bs = bsc_mni_results.S
    confmat_bs = cv.ca.stats.matrix
    
    individual_confmat = np.zeros((len(confmat_bs),len(confmat_bs),len(cv.ca.stats.matrices)))
    for I in range(len(cv.ca.stats.matrices)):
        individual_confmat[:,:,I] = cv.ca.stats.matrices[I].matrix
    
    cv.ca.stats.labels
    tmp_confmat=dict()
    
    for dd in cv.ca.stats.matrices:
        tmp_confmat[dd]=dd._ConfusionMatrix__matrix
        
    all_confmatx=dict(zip(keys,list(tmp_confmat.values())))
    acc_std_bs = np.concatenate((np.mean(bsc_mni_results),np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)), axis=None)   
    
    results_fulllist[name] = {} 
    results_fulllist[name]['labels'] = cv.ca.stats.labels
    results_fulllist[name]['acc_avg'] =  acc_std_bs
    results_fulllist[name]['acc_all'] =  accuracy_bs 
    results_fulllist[name]['confmat'] =  confmat_bs
    results_fulllist[name]['confmat_all'] =  individual_confmat 
    results_fullconfmatx[name] = all_confmatx
    
    os.chdir("/path/") #Output directory
    out = 'roi_classification_featureselection_results_' + roiname + '.h5'
    save_dict_to_hdf5(results_fulllist,out)
    print("Classification completed.")

##------------------------------------------------------------------------------------------------------------------------------------
#   7. Permute null distribution of chance level prediction accuracy (full brain dataset)

import  mvpa2.suite as mvpa
from mvpa2.base.hdf5 import h5load
import os
from scipy import stats
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.neural_network import MLPClassifier
import numpy as np
os.chdir("/path/") # Directory of the helper functions file
from h5_helperfunctions_V2 import save_dict_to_hdf5

os.nice(20)

print("Loading data...")
datapath = '/path/' # Path to input data file
ds_all = h5load(datapath)
print(" Data loaded")

# Inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    sd.sa['subject'] = np.repeat(i, len(sd))
    # Number of subjects
    nsubjs = len(ds_all)
    # Number of categories
    ncats = len(ds_all[0].UT)
    # Number of run
    nruns = len(ds_all[0].UC)

# use same classifier
clf = MLPClassifier(hidden_layer_sizes=(100,100,),alpha=1,max_iter=500) #Neural net classifier, tuned hyperparameter
wrapped_clf = SKLLearnerAdapter(clf)

# Feature selection and permutator
nf = 3000
fselector = mvpa.FixedNElementTailSelector(nf, tail='upper', mode='select', sort=False)
sbfs = mvpa.SensitivityBasedFeatureSelection(mvpa.OneWayAnova(), fselector, enable_ca=['sensitivities'])
fsclf = mvpa.FeatureSelectionClassifier(wrapped_clf, sbfs)
stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)

ds_mni = mvpa.vstack(ds_all)

# Define the permuations
# Permute null distribution with 500 permutations. To save time, permute 125 null values per run and manually (open nyt Spyder instances) run 4 parallel versions of the this script with only modified index in variable "iteration"
iteration = 1 # Idx of this parallel process (in our case from 1 to 4)
count = 25 # How many permutations to run in one round
rounds = 5 # How many rounds to run (count*rounds = total number of permutatations for this iteration per parallel process)

# Change to output dir
os.chdir('/path/') # Output directory for the permuted values

# Run permutation
perm_results_fulllist=dict()
for I in range (rounds):
    print("Starting permutations: count = %.0f/%.0f" %(I+1,rounds))
    mni_start_time = mvpa.time.time()

    # How many permutations
    repeater = mvpa.Repeater(count=count)

    # Choose leave-one-out partitioning
    partitioner = mvpa.NFoldPartitioner(attr='subject')

    # Permute the training part of a dataset exactly ONCE
    permutator = mvpa.AttributePermutator('targets', limit={'partitions': 1}, count=1)

    # Information for the McNUllDist to shuffle the training set in each CV fold.
    null_cv = mvpa.CrossValidation(
                fsclf,
                mvpa.ChainNode([partitioner, permutator], space=partitioner.get_space()),
                errorfx=lambda p,t: np.mean(p==t),
                postproc = mvpa.mean_sample())
    
    # Monte Carlo distribution estimator
    distr_est = mvpa.MCNullDist(repeater, tail='left', measure=null_cv,
                           enable_ca=['dist_samples'])
    
    # CV with null distribution estimation
    cv = mvpa.CrossValidation(fsclf,
                         partitioner,
                         errorfx=lambda p,t: np.mean(p==t),
                         postproc = mvpa.mean_sample(),
                         null_dist=distr_est,
                         enable_ca=['stats'])
    # Run CV
    bsc_null_results = cv(ds_mni)

    print("Done in %.1f seconds" % (mvpa.time.time() - mni_start_time,))
    
    #Save results
    name = "Neural Net"
    perm_results_fulllist[name] = {} 
    perm_results_fulllist[name]['labels'] = cv.ca.stats.labels
    perm_results_fulllist[name]['pvalues'] = cv.ca.null_prob.samples
    perm_results_fulllist[name]['permutation_acc'] =  cv.null_dist.ca.dist_samples.samples
    perm_results_fulllist[name]['confmat_per_permutation'] =   cv.ca.stats.matrix
    perm_results_fulllist[name]['individual_confmats'] = cv.ca.stats.matrices
    save_dict_to_hdf5(perm_results_fulllist, ('permutations_' + str(iteration) + str(count) + str(I) +'.h5')) # Save permutation results after each round to prevent data loss if the process crashes
