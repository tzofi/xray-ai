'''
Command to run this:
    python process.py mimic-cxr-2.0.0-chexpert.csv edema_labels-12-03-2019/mimic-cxr-sub-img-edema-split-allCXR.csv mimic-cxr-2.0.0-split.csv mimic-cxr-2.0.0-negbio.csv output.csv
    arg1 tells us chexpert labels; arg2 tells us the images we have copied over to lambda-stack; arg3 tells us splits; arg4 tells us negbio labels, arg5 is output file
'''

import os
import sys
import csv
import random
import numpy as np

chexpert = csv.DictReader(open(sys.argv[1]))
labels = {}

CLASSES = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']

'''
1) Get CheXpert labels
'''
for i, l in enumerate(chexpert):
    ident = l['subject_id'] + "_" + l['study_id']
    labels[ident] = [l['Atelectasis'],l['Cardiomegaly'],l['Consolidation'],l['Edema'],l['Enlarged Cardiomediastinum'],l['Fracture'],l['Lung Lesion'],l['Lung Opacity'],l['No Finding'],l['Pleural Effusion'],l['Pleural Other'],l['Pneumonia'],l['Pneumothorax'],l['Support Devices']]

'''
2) Find available DICOM from our MIMIC data
'''
frontal = csv.DictReader(open(sys.argv[2]))
dicoms = {}
for i, l in enumerate(frontal):
    dicoms[l['dicom_id']] = l['dicom_id']


'''
3) Extract split for samples that are in list of available DICOMs and which have CheXpert labels
'''
split = csv.DictReader(open(sys.argv[3]))
output = {}
output['train'] = []
names = ['dicom_id', 'study_id', 'subject_id']
names.extend(CLASSES)
names.append('fold')
for i, l in enumerate(split):
    try:
        found_dicom = dicoms[l['dicom_id']]
    except:
        continue

    ident = l['subject_id'] + "_" + l['study_id']
    try:
        label = labels[ident]
    except:
        print("Cant find: {}".format(ident))
        continue
    data = [l['subject_id'], l['study_id'], l['dicom_id']]
    data.extend(label)
    output['train'].append(data)
    #if l['split'] in output:
    #else:
    #    output[l['split']] = [data]

print(len(output['train']))
#print(len(output['validate']))
#print(len(output['test']))

random.shuffle(output['train'])

''' start - added later '''
no_uncertain = {}
for i, x in enumerate(output['train']):
    if '-1.0' not in x:
        no_uncertain[i] = x
print(len(no_uncertain))
''' end '''

pleural_other = {}
consolidation = {}
for key, val in no_uncertain.items():
    if val[-4] == '1.0':
        pleural_other[key] = key
for key, val in no_uncertain.items():
    if val[-12] == '1.0' and key not in pleural_other:
        consolidation[key] = key
t = {}
for idx in list(pleural_other.keys()):
    t[idx] = output['train'][idx]
for idx in list(consolidation.keys()):
    t[idx] = output['train'][idx]
for key, val in no_uncertain.items():
    if key not in pleural_other and key not in consolidation:
        t[key] = val

print(len(t))

'''
Get the labels from negbio
'''
bioneg = csv.DictReader(open(sys.argv[4]))
other_labels = {}
for i, l in enumerate(bioneg):
    ident = l['subject_id'] + "_" + l['study_id']
    other_labels[ident] = [l['Atelectasis'],l['Cardiomegaly'],l['Consolidation'],l['Edema'],l['Enlarged Cardiomediastinum'],l['Fracture'],l['Lung Lesion'],l['Lung Opacity'],l['No Finding'],l['Pleural Effusion'],l['Pleural Other'],l['Pneumonia'],l['Pneumothorax'],l['Support Devices']]


'''
Extract the matching labels to use for test; no uncertainty allowed; 5000 test images
'''
test_idxs = {}
test_counts_positive = {}
test_counts_negative = {}
test_counts_uncertain = {}
finished = [0 for _ in range(14)]
for i, item in t.items():
    for k, (key, val) in enumerate(list(test_counts_positive.items())):
        if val >= 885:
            finished[k] = 1
    if sum(finished) == 14:
        break
    #if len(test_idxs) == 5000:
    #    break
    #if '-1.0' in item[-14:]:
    #    continue
    ident = item[0] + "_" + item[1]
    other_labels_item = other_labels[ident]
    if item[-14:] == other_labels_item[-14:]:
        item_lbls = item[-14:]
        # check this wont send any classes over limit
        bad = False
        for k, (key, val) in enumerate(list(test_counts_positive.items())):
            try:
                val1 = test_counts_uncertain[key]
            except:
                val1 = 0
            #if val + val1 == 884:
            if val == 884:
                idx = CLASSES.index(key)
                if item_lbls[idx] == '1.0': # or item_lbls[idx] == '-1.0':
                    bad = True
                    break
        if bad == True:
            continue

        test_idxs[i] = i

        for j, nm in enumerate(CLASSES):
            if item_lbls[j] == '1.0':
                if nm in test_counts_positive:
                    test_counts_positive[nm] += 1
                else:
                    test_counts_positive[nm] = 1
            elif item_lbls[j] == '0.0':
                if nm in test_counts_negative:
                    test_counts_negative[nm] += 1
                else:
                    test_counts_negative[nm] = 1
            elif item_lbls[j] == '-1.0':
                if nm in test_counts_uncertain:
                    test_counts_uncertain[nm] += 1
                else:
                    test_counts_uncertain[nm] = 1

print("Test counts positive")
print(test_counts_positive)
#print([(key, float(val)/5000) for key, val in list(test_counts_positive.items())])
print("Test counts negative")
print(test_counts_negative)
#print([(key, float(val)/5000) for key, val in list(test_counts_negative.items())])
print("Test counts uncertain")
print(test_counts_uncertain)


counts_positive = {}
counts_negative = {}
counts_uncertain = {}
counts_unknown = {}
counts_positive_test = {}
counts_negative_test = {}
counts_uncertain_test = {}
counts_unknown_test = {}

train = []
test = []
for i, item in enumerate(output['train']):
    if i in test_idxs:
        test.append(item) 
        
        for j, lbl in enumerate(item[-14:]):
            pathology = CLASSES[j] 
            if lbl == '-1.0':
                if pathology in counts_uncertain_test:
                    counts_uncertain_test[pathology] += 1
                else:
                    counts_uncertain_test[pathology] = 1
            elif lbl == '0.0':
                if pathology in counts_negative_test:
                    counts_negative_test[pathology] += 1
                else:
                    counts_negative_test[pathology] = 1
            elif lbl == '1.0':
                if pathology in counts_positive_test:
                    counts_positive_test[pathology] += 1
                else:
                    counts_positive_test[pathology] = 1
            else:
                if pathology in counts_unknown_test:
                    counts_unknown_test[pathology] += 1
                else:
                    counts_unknown_test[pathology] = 1
    else:
        train.append(item)

        for j, lbl in enumerate(item[-14:]):
            pathology = CLASSES[j] 
            if lbl == '-1.0':
                if pathology in counts_uncertain:
                    counts_uncertain[pathology] += 1
                else:
                    counts_uncertain[pathology] = 1
            elif lbl == '0.0':
                if pathology in counts_negative:
                    counts_negative[pathology] += 1
                else:
                    counts_negative[pathology] = 1
            elif lbl == '1.0':
                if pathology in counts_positive:
                    counts_positive[pathology] += 1
                else:
                    counts_positive[pathology] = 1
            else:
                if pathology in counts_unknown:
                    counts_unknown[pathology] += 1
                else:
                    counts_unknown[pathology] = 1

print("\n--------------------------------------------\n")
print('TRAIN ({}):'.format(len(train)))
print('POSITIVE:')
print(counts_positive)
print('NEGATIVE:')
print(counts_negative)
print('UNCERTAIN:')
print(counts_uncertain)
print('UNKNOWN:')
print(counts_unknown)
print("\n--------------------------------------------\n")
print('TEST ({}):'.format(len(test)))
print('POSITIVE:')
print(counts_positive_test)
print('NEGATIVE:')
print(counts_negative_test)
print('UNCERTAIN:')
print(counts_uncertain_test)
print('UNKNOWN:')
print(counts_unknown_test)

#extra_test = output['train'][:(5000-len(output['validate']))]
#output['train'] = output['train'][(5000-len(output['validate'])):]
#output['validate'].extend(extra_test)

'''
4) Save all info in new CSV
Training methods:   train, val, test
                    train = 1,2,3,4,5,6,val,test
'''
length = len(train)
interval = int(length/12)

fold = 0
writer = csv.writer(open(sys.argv[5], 'w'))
#writer = open(sys.argv[4], 'w')
#writer.writerow(",".join(names))
writer.writerow(names)
for i, l in enumerate(train):
    if i % interval == 0:
        fold += 1
    l.append(str(fold))
    writer.writerow(l)

fold = 'TEST'
for i, l in enumerate(test):
    l.append(fold)
    writer.writerow(l)


'''
counts = {}
for i, l in enumerate(labels):
    if i == 0:
        names = list(l.keys())
        writer = csv.DictWriter(open(sys.argv[2], 'w'), fieldnames = names)
    if i < 7391 - 1:
        writer.writerow(l)
    else:
        l['fold'] = 6
        writer.writerow(l)
    if l['split'] in counts:
        counts[l['split']] += 1
    else:
        counts[l['split']] = 1

print(counts)
'''
