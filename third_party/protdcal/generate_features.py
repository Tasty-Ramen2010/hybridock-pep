#!/usr/bin/env python

__author__ = "Nashira H. Ridgeway"
__version__ = "1.0.0"
__license__ = "GPL"
__maintainer__ = "Nashira H. Ridgeway"

"""\
This script generates features for a given .csv, .tsv, or .xlsx file of peptide sequences. 
The resulting features contain ProtDCal values for the given sequence, as well as amino acid composition (AAC)

Usage: generate_features.py seqfile "colname" output
"""

import pandas as pd
import numpy as np
from propythia.protein.sequence import ReadSequence
from propythia.protein.descriptors import ProteinDescritors
import itertools
from collections import Counter
import argparse
import sys

# PARSE USER INFORMATION PASSED AT RUNTIME
parser = argparse.ArgumentParser(description="Feature generation for peptide-based sequences. Output file contains features for the sequence as well as the sequence itself (in the first column).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--peptide", action="store_true", help="name of column with peptide sequences in it (in quotations, i.e. 'peptide')")
parser.add_argument("-i", "--input", action="store_true", help="location of file containing peptide sequences to generate features for")
parser.add_argument("-o", "--output", action="store_true", help="location and name of output file containing feature data")
args = parser.parse_known_args()

# LINK PARAMETERS TO ARGUMENTS
read_in = args[1][1]
read_out = args[1][2]
seq_col = args[1][0]


# GENERATE PROTDCAL FEATURES FOR SEQUENCE
def protdcal_features (sequence, protdcal):
    # METHOD FOR GENERATING PROTDCAL MEAN VALUES FOR EACH SEQUENCE
    # FIRST: GENERATE PROTDCAL VALUES
    slist = list(sequence)   # split sequence up into list
    # Go through sequence to get protdcal value
    t1 = []
    values = []
    for i in slist:
        t1.append(protdcal.loc[i].tolist())
    t2 = list(map(lambda *x: sum(x), *t1))   # add up values
    for t in t2:
        t = t/len(slist)   # get average (mean) of summed values
        values.append(t)
    headers =  protdcal.columns.tolist()   # include headers
    
    return values, headers


# TRAINING FEATURE GENERATION: GENERATE OTHER FEATURES + FORMAT NICELY
# Import protdcal file
protdcal = pd.read_csv('./protdcal_aa_table.csv', index_col=0)

# Import user file
file_in = str(read_in)
if '.xlsx' in file_in:
    # User input an excel file
    user_feat = pd.read_excel(read_in)
elif '.tsv' in file_in:
    # User input a tab-separated file
    user_feat = pd.read_csv(read_in, sep='\t')
elif '.csv' in file_in:
    # User input a comma-separated file
    user_feat = pd.read_csv(read_in)
else:
    # User input a file in an improper format
    print("\nFile input in improper format. Please be sure file is a .xlsx, .tsv, or .csv and try again.")
    sys.exit()

try:
    user_feat = user_feat.drop(columns=['Unnamed: 0'])
except:
    user_feat = user_feat

# Define seq_col input by user as a variable, ensure it's in string format for use with dataframe
seq_col = str(seq_col)
for c in user_feat.columns:
    if seq_col in c:
        site = c
    else:
        site = None
if site == None:
    print("Error with user input column name. Please double check name of peptide sequence column in sequences file and try again.")

# Check for invalid information in sequences that we can't handle - remove + output as error file
alphabet = "ARNDCQEGHILKMFPSTWYVX" # 20 essential amino acids + X
invalid_input = user_feat[user_feat[site].str.strip(alphabet) != ""]
user_feat = user_feat.drop(invalid_input.index).reset_index(drop=True)
invalid_read_out_file = read_out+'invalid_sequences.csv'
invalid_input.to_csv(invalid_read_out_file)
print("\n WARNING! There were " + str(len(invalid_input)) + " sequences with invalid amino acids input (lower case, or characters that don't represent the 20 essential AAs). Reading out to " + invalid_read_out_file)
print("\n Carrying on...\n\n")

# Pull sequences and make X -> A for feature generation (A is least offensive AA)
user_feat[site] = user_feat[site].str.replace('X', 'A')

# Check for repeats - remove + output as repeat error file
duplicates = user_feat[user_feat.duplicated(subset = site, keep="first")]
user_feat = user_feat.drop(duplicates.index).reset_index(drop=True)
duplicates.to_csv(read_out+'duplicated_sequences.csv')
print("\n WARNING! There were " + str(len(duplicates)) + " duplicated sequences within the input. Keeping the first instance and deleting the others, reading out the deleted sequences to" + str(read_out+'duplicated_sequences.csv'))
print("\n Carrying on...\n\n")

sequences = user_feat[site]

# NOW GET INTO FEATURE GENERATION!

# Create df for protdcal results to go into
v, h = protdcal_features(sequences[0], protdcal)
features = pd.DataFrame(columns=h)
features.loc[len(features)] = v

i = 1

# Go through rest of sequences to generate protdcal features set
while i < len(sequences):
    ts = sequences[i]
    value, header = protdcal_features(ts, protdcal)
    features.loc[len(features)] = value
    i+=1
    if i % 500 == 0:
        print('ProtDCal Feature Generation:', i, 'of', len(sequences), 'completed')

features[site] = user_feat[site]

# Now let's generate the ProPythia Features
print('\nNow generating Amino Acid Composition Features...')

user_feat['Seqs_cleaned_for_X'] = user_feat[site].str.replace('X', 'A')

# first create the ProPythia descriptor object
pp_descriptor = ProteinDescritors(dataset=user_feat, col=site)

# Generate aac
aac = pp_descriptor.get_paac(lamda=0)

final_feats = user_feat[site].reset_index(drop=False)

print('\nFinished generating Amino Acid Composition\nAdding everything together...')

# Merge all propythia content together via site
final_feats = pd.merge(final_feats, aac, on=site)
all_feats = pd.merge(final_feats, features, on=site)
all_feats = all_feats.drop(columns='index')

# Merge with initial input to keep indexes if there were any
user_feat = user_feat.drop(columns=['Seqs_cleaned_for_X'])
all_feats = pd.merge(user_feat, all_feats, on=site)

print ("\nFeature Generation has finished! \nThe resulting dataset has", len(all_feats.columns), "features for", len(all_feats), "peptide sequences. \nNow saving to output location...")

file_out = str(read_out)
if '.xlsx' in file_out:
    # User output to an excel file
    all_feats.to_excel(file_out, index=False)
elif '.tsv' in file_out:
    # User output to a tab-separated file
    all_feats.to_csv(file_out, sep='\t', index=False)
elif '.csv' in file_out:
    # User output to a comma-separated file
    all_feats.to_csv(file_out, index=False)
else:
    # User output automatically to excel file
    file_out = file_out+'features.xlsx'
    all_feats.to_excel(file_out, index=False)
    
print ("\nSaved file to", file_out)
