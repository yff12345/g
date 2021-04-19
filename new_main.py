#!/usr/bin/env python
from DEAPDatasetSequence import DEAPDatasetSequence

# COMMON LOGIC
ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphSequenceProcessedData'

dataset = DEAPDatasetSequence(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir= PROCESSED_DIR)

print(dataset[:100)