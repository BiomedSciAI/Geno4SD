from __future__ import division
__author__ = "Kahn Rhrissorrakrai"
__copyright__ = "Copyright 2018, IBM Research"
__version__ = "0.0.1"
__maintainer__ = "Kahn Rhrissorrakrai"
__email__ = "krhriss@us.ibm.com"
__status__ = "Development"

import gc
import itertools
import multiprocessing as mp
import os
import pickle
import random
import re
from itertools import groupby
import networkx as nx
from collections import Counter

from matplotlib import colors
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
import seaborn as sns
from pyvis.network import Network
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import binarize
from sklearn.cluster import AgglomerativeClustering

class ParametersFile:
    def __init__(self, paramsFile):
        # Read from file
        params = pd.read_csv(paramsFile)
        paramsDict = dict(zip(params['param'], params['value']))

        self.saveIntermediate = paramsDict['saveIntermediate']
        #self.saveIntermediate = False
        self.patientMulti = paramsDict['patientMulti']
        self.simPatient = paramsDict['simPatient']

        if paramsDict['simPatientIndex'] != 'None':
            self.simPatientIndex =  [int(x) for x in paramsDict['simPatientIndex'].split(',')]
            self.simPatientIndex = range(self.simPatientIndex[0],self.simPatientIndex[1])
        else:
            self.simPatientIndex = 'None'

        # Keep the release to a single mode
        paramsDict['mode'] =  'calconly'

        self.dir_output = paramsDict['outputDir']
        self.dir_runOutput = ''

        if 'nSamplesCI' in paramsDict.keys():
            self.nSamplesCI = int(paramsDict['nSamplesCI'])
        else:
            self.nSamplesCI = 10

        if 'sampThresCI' in paramsDict.keys():
            self.sampThresCI = float(paramsDict['sampThresCI'])
        else:
            self.sampThresCI = 0.5

#        self.workingDate = paramsDict['workingDate']
        # self.file_participantInfo = paramsDict['participantFile']
        self.file_samplesInfo = paramsDict['sampleFile']
        self.file_parameters = paramsFile
        self.file_mafFile = paramsDict['mafFile']

        self.distinctColors = ['#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                               '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',
                               '#ffd8b1', '#000075', '#808080', '#ffffff', '#4363d8', '#000000']

        self.startIndex = int(paramsDict['startIndex'])
        self.endIndex = int(paramsDict['endIndex'])

        if paramsDict['dropNullTissue'] == 'True':
            self.dropNullTissue = True
        else:
            self.dropNullTissue = False
        self.ccf_lesion_thd = float(paramsDict['ccfLesionThreshold'])
        self.num_hyp = 1000
        self.runIdx = 0
        self.nthreads = int(paramsDict['nthreads'])
        self.mode = paramsDict['mode']

        # Keep release to recurrence Limit to 1.0
        self.recurrenceLimit = 1.0

        self.subSampleSize = int(paramsDict['subSampleSize'])

        dRange = [float(x) for x in paramsDict['discreteRange'].split(',')]
        self.discreteRange = np.arange(dRange[0], dRange[1], dRange[2])
        self.alphasGridSearch = pd.DataFrame(list(itertools.product(*[self.discreteRange] * self.subSampleSize)))
        self.num_hyp = self.alphasGridSearch.shape[0]

        self.simBloodAlphas = [float(x) for x in paramsDict['simBloodAlphas'].split(',')]
        self.addUniqSimAlterations = int(paramsDict['addUniqSimAlterations'])
        if paramsDict['ccfRandom'] == 'True':
            self.ccfRandom = True
        else:
            self.ccfRandom = False

        self.weightsRandomDiscrete = False
        self.weightsRandomUniform = False  # If True then the samples will get an alpha that is random from a random (0.001, 1.0) uniform distribution, else it will be drawn from bins parameter by the following:
        self.forceMixHypotheses = True
        self.useMediumBin = False

        # Keep release to lowThreshold_alphaBin to 0.3
        self.lowThreshold_alphaBin = 0.3
        # Keep release to highThreshold_alphaBin to 0.6
        self.highThreshold_alphaBin = 0.6
        # Keep release to recurrence Limit to 1.0

        self.lowRange_shedding = [0.01, self.lowThreshold_alphaBin]
        self.highRange_shedding = [self.highThreshold_alphaBin, 1.0]
        self.sampleDistanceMetric = 'euclidean'
        self.runMultipleBlood = False
        self.edgeThres = float(paramsDict['edgeThres'])

    # Write the parameters
    def writeParametersOut(self):
        existsParametersFile = os.path.isfile(self.file_parameters)
        f = open(self.file_parameters, "w")
        f.write(str(vars(self)) + "\n")
        f.close()

    # Write the parameters though if there is already a parameters file in the output directory then read in the parameters
    #   this ensures the run is non-destructive
    def writeParametersOut_nonDestructive(self):
        existsParametersFile = os.path.isfile(self.file_parameters)
        if existsParametersFile:
            self.readParameterFile()
        else:
            f = open(self.file_parameters, "w")
            f.write(str(vars(self)) + "\n")
            f.close()

    def getNumHyp(self):
        return self.num_hyp

    def setRunIdx(self, runIdx):
        self.runIdx = runIdx
        self.dir_runOutput = os.path.join(self.dir_output, "Run" + str(runIdx))
        self.file_parameters = os.path.join(self.dir_runOutput, "../parameters.txt")

    def getParametersAsDict(self):
        paramsDict = {}
        paramsDict['num_hyp'] = self.num_hyp
        paramsDict['runIdx'] = self.runIdx
        paramsDict['nthreads'] = self.nthreads
        paramsDict['specificPatients'] = self.specificPatients
        paramsDict['mode'] = self.mode
        paramsDict['recurrenceLimit'] = self.recurrenceLimit
        paramsDict['weightsRandomDiscrete'] = self.weightsRandomDiscrete
        paramsDict['weightsRandomUniform'] = self.weightsRandomUniform
        paramsDict['subSampleSize'] = self.subSampleSize
        paramsDict['forceMixHypotheses'] = self.forceMixHypotheses
        paramsDict['useMediumBin'] = self.useMediumBin
        paramsDict['lowThreshold_alphaBin'] = self.lowThreshold_alphaBin
        paramsDict['highThreshold_alphaBin'] = self.highThreshold_alphaBin
        paramsDict['lowRange_shedding'] = self.lowRange_shedding
        paramsDict['highRange_shedding'] = self.highRange_shedding
        paramsDict['sampleDistanceMetric'] = self.sampleDistanceMetric
        return paramsDict


class clinicalData:
    def __init__(self, params):
        # Get the metadata of all samples
        # self.participantInfo = pd.read_csv(params.file_participantInfo)
        self.samplesInfo = pd.read_csv(params.file_samplesInfo)
        self.samplesInfo.columns = [ re.sub( 'ulp_tumor_fraction', 'tumor_fraction', x ) for x in self.samplesInfo.columns.to_list() ]

        # Mapping from IBM ID to Broad ID and tissue Maps
        self.sampleIDMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['sampleID']))
        self.sampleIDMapFromOriginal = dict(zip(self.samplesInfo['sampleID'], self.samplesInfo['primarySampleID']))
        self.patientIDMap = dict(
            zip(self.samplesInfo['primaryParticipantID'], self.samplesInfo['participantID']))
        self.daysDxIDMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['days_from_dx']))

        self.tissueMap = dict(zip(self.samplesInfo['primarySampleID'],
                                  [re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(x))) for x
                                   in self.samplesInfo['tissueSiteSimple']]))
        self.tissueOriginalMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tissueSite']))

        # Assign a specific color to each tissue
        self.tissues = [re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(x))) for x in
                        self.samplesInfo['tissueSiteSimple'].unique()]
        self.tissueColor = dict(zip(self.tissues, params.distinctColors[0:len(self.tissues)]))

        # Tumor Purity
        self.tumorFractions = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tumor_fraction']))
        #        self.tumorFractions = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tumorFraction']))

        # Get the cohortPatients
        self.cohortPatients = list(set(self.samplesInfo['primaryParticipantID']))
        self.cohortPatients.sort()

        # Get Lesion Response Data
        self.lesionResponseMap = {}

    def updateMappings(self, params):
        # Mapping from IBM ID to Broad ID and tissue Maps
        self.sampleIDMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['sampleID']))
        self.sampleIDMapFromOriginal = dict(zip(self.samplesInfo['sampleID'], self.samplesInfo['primarySampleID']))
        self.patientIDMap = dict(
            zip(self.samplesInfo['primaryParticipantID'], self.samplesInfo['participantID']))
        self.daysDxIDMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['days_from_dx']))

        self.tissueMap = dict(zip(self.samplesInfo['primarySampleID'],
                                  [re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(x))) for x
                                   in self.samplesInfo['tissueSiteSimple']]))
        self.tissueOriginalMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tissueSite']))

        # Assign a specific color to each tissue
        self.tissues = [re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(x))) for x in
                        self.samplesInfo['tissueSiteSimple'].unique()]
        if len(self.tissues) <= len(params.distinctColors):
            self.tissueColor = dict(zip(self.tissues, params.distinctColors[0:len(self.tissues)]))
        else:
            import random
            randColors = []
            for x in range(0, len(self.tissues)):
                random_number = random.randint(0, 16777215)
                hex_number = str(hex(random_number))
                hex_number = '#' + hex_number[2:]
                randColors.append(hex_number)
            self.tissueColor = dict(zip(self.tissues, randColors))

        # Tumor Purity
        self.tumorFractions = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tumor_fraction']))
        # self.tumorFractions = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tumorFraction']))

        # Get the cohortPatients
        self.cohortPatients = list(set(self.samplesInfo['primaryParticipantID']))
        self.cohortPatients.sort()

        # Get Lesion Response Data
        self.lesionResponseMap = {}

    def filterCohortPatients(self, specificPatients):
        self.cohortPatients = [x for x in self.cohortPatients if x in specificPatients]


class simClinicalData:
    def __init__(self, params, simMafData):
        self.samplesInfo = simMafData[['SimSampType', 'participantID', 'sample', 'primarySampleID']].drop_duplicates()
        self.samplesInfo.columns = ['tissueSiteSimple', 'primaryParticipantID', 'sample', 'primarySampleID']
        self.samplesInfo['specimenType'] = self.samplesInfo['tissueSiteSimple']
        self.samplesInfo['days_from_dx'] = [1] * len(self.samplesInfo)
        self.samplesInfo['tumorFraction'] = [1] * len(self.samplesInfo)

        # Mapping from IBM ID to Broad ID and tissue Maps
        self.sampleIDMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['sample']))
        self.patientIDMap = dict(
            zip(self.samplesInfo['primaryParticipantID'], self.samplesInfo['primaryParticipantID']))
        self.daysDxIDMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['days_from_dx']))

        self.tissueMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tissueSiteSimple']))
        self.tissueOriginalMap = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tissueSiteSimple']))

        # Assign a specific color to each tissue
        self.tissues = self.samplesInfo['tissueSiteSimple'].unique()
        self.tissueColor = dict(zip(self.tissues, params.distinctColors[0:len(self.tissues)]))

        # Tumor Purity
        self.tumorFractions = dict(zip(self.samplesInfo['primarySampleID'], self.samplesInfo['tumorFraction']))

        # Get the cohortPatients
        self.cohortPatients = list(set(self.samplesInfo['primaryParticipantID']))
        self.cohortPatients.sort()

        # Get Lesion Response Data
        self.lesionResponseMap = {}

    def filterCohortPatients(self, specificPatients):
        self.cohortPatients = [x for x in self.cohortPatients if x in specificPatients]

class hypothesisBlood:
    def __init__(self, params, patID):
#        self.numHypothesis = params.num_hyp
        self.patID = patID
        self.file_saveClass = os.path.join(params.dir_runOutput, "hypothesisBloodClass-" + self.patID + ".pkl")
        self.distance = pd.DataFrame()
        self.hypotheses = dict()
        self.sbl = dict()



#### Functions ####

def _readMafData(mafFile):
    mafData =  pd.read_csv(mafFile,
                       low_memory=False,
                       usecols=['primaryParticipantID', 'primarySampleID', 'Hugo_Symbol',
                                'Start_position', 'Reference_Allele', 'Tumor_Seq_Allele2',
                                'Chromosome', 'ccf_hat', 'File'])#, 't_ref_count_post_forcecall','t_alt_count_post_forcecall'])

    mafData['key'] = [ str(row.Start_position) + '_' + 
                       str(row.Chromosome) + '_' + 
                       str(row.Reference_Allele) + '_' + 
                       str(row.Tumor_Seq_Allele2) for idx,row in mafData.iterrows() ]
    return mafData

# Get the tissue data for a given patient
def _getTissueData(mafData, patID, clinData, dropNullTissue=True):
    patData = mafData[mafData['primaryParticipantID'] == patID].copy()
    patData.loc[:, 'ids'] = patData['Hugo_Symbol'] + '-' + patData['Start_position'].astype('str')
    patSamples = list(set(patData['primarySampleID']))
    tissueData = []
    # Also be sure to remove force call mutations by checking against the 'sample' column the mutation is from
    for samp in patSamples:
        sampData = patData[patData['primarySampleID'] == samp].copy()
        #        sampData.loc[:, 'sample'] = [re.sub("_-", "-", re.sub('_v[0-9]*_Exome.*', '', re.sub('^.*/', '', x))) for x in
        #                                     sampData['sample']]
        #        sampData = sampData[sampData['sample'] == clinData.sampleIDMap[samp]]
        tissueData.append(sampData)
    tissueData = pd.concat(tissueData)

    # pivoting data so we have a table (row lesion, column mutations, entry ccf values)
    if len(tissueData) > 0:
        # Drop Blood Samples
        sampInfo = clinData.samplesInfo[clinData.samplesInfo['primaryParticipantID'] == patID].copy()
        sampIDs =list( sampInfo[(sampInfo['specimenType'] != "blood") & (sampInfo['specimenType'].isnull() == False)][
            'primarySampleID'])
        tissueData = tissueData[tissueData.primarySampleID.isin(sampIDs)]


        # Filter out mutations with Confidence Interval width > 0.75
        if 'ccfCI' in tissueData.columns:
            tissueData = tissueData[ tissueData.ccfCI <= 0.75 ]

        tissueData = tissueData.pivot_table(index='primarySampleID', columns='ids', values='ccf_hat',
                                            aggfunc='mean').fillna(0)

        # add tissue information
        tissueData.loc[:, 'tissue'] = [None] * len(tissueData)
        for samp in tissueData.index:
            if samp in clinData.samplesInfo['primarySampleID'].to_list():
                tissueData.loc[samp, 'tissue'] = re.sub(' \(.*', '', str(
                    clinData.samplesInfo[clinData.samplesInfo['primarySampleID'].str.contains(samp)][
                        'tissueSite'].values[0]))

        tissueData = tissueData.fillna(0)


        # Drop Samples without Tissue
        if dropNullTissue:
            tissueData = tissueData.drop(tissueData.index[tissueData['tissue'].isnull()], axis=0)
            tissueData = tissueData.drop(tissueData.index[(tissueData['tissue'] == 0)], axis=0)
            tissueData = tissueData.drop(tissueData.index[(tissueData['tissue'] == 'nan')], axis=0)

    return tissueData


# Get all the Blood samples sorted in descedning order by time
def _getBloodSampleInfo(patID, samplesInfo, mafData):
    sampInfo = samplesInfo[samplesInfo['primaryParticipantID'] == patID].copy()
    sampInfo = sampInfo.loc[sampInfo['specimenType'] == "blood"]
    sampInfo = sampInfo.loc[sampInfo['primarySampleID'].isin(mafData['primarySampleID'])]
    if len(sampInfo) > 0:
        sampInfo = sampInfo.sort_values('days_from_dx', ascending=False)
        return sampInfo
    else:
        return pd.DataFrame()


# Get the Blood Sample
def _getBloodSample(patID, bloodSampleID, mafData):
    patData = mafData[mafData['primarySampleID'] == bloodSampleID].copy()
    patData.loc[:, 'ids'] = patData['Hugo_Symbol'] + '-' + patData['Start_position'].astype('str')
    patData.loc[:, 'cleanFileName'] = [re.sub('_v[0-9]*_Exome.*', '', re.sub('^.*/', '', x)) for x in patData['File']]
    return patData


# Get the Latest Blood Sample
def _getLatestBloodSample(patID, samplesInfo, mafData):
    sampInfo = _getBloodSampleInfo(patID, samplesInfo, mafData)
    if len(sampInfo) > 0:
        latestBloodSampleID = sampInfo.loc[:, 'primarySampleID'].iloc[0]
        patData = mafData[mafData['primarySampleID'] == latestBloodSampleID].copy()
        patData.loc[:, 'ids'] = patData['Hugo_Symbol'] + '-' + patData['Start_position'].astype('str')
        patData.loc[:, 'cleanFileName'] = [re.sub('_v[0-9]*_Exome.*', '', re.sub('^.*/', '', x)) for x in
                                           patData['File']]
        return patData
    else:
        return pd.DataFrame()


# remove specific sample with low purity
def _dropLowPuritySamples(tissueData, tumorFractions, thd=0.05):
    lowPuritySamples = [x for x in tissueData.index if tumorFractions[x] < thd]
    tissueData = tissueData.drop(lowPuritySamples, axis=0)
    tissueData = tissueData.loc[:, (tissueData != 0).any(axis=0)]
    return tissueData


# get mutations present only in the latest blood sample
def _getMutationsUniqueToBlood(latestBlood, tissueData):
    uniqueBlood = list(set(latestBlood['ids'].tolist()).difference(set(tissueData.columns)))
    uniqueBlood = latestBlood[latestBlood['ids'].isin(uniqueBlood)][['ids', 'ccf_hat']]
    uniqueBlood.loc[:, 'sample'] = ['latestBlood'] * len(uniqueBlood)
    return (uniqueBlood)


# Generate the hypothesis blood
def generateHypothesisBlood(input):
    ###Generate Hypothesis Blood via Dirichlet and Hypothesis
    ###Each lesion has a random weight

    idx = input[0]
    tissueData = input[1]
    highRange = input[2]
    lowRange = input[3]
    lowThd = input[4]
    highThd = input[5]
    uniqueBlood = input[6]
    forceMixHypotheses = input[7]
    useMediumBin = input[8]
    weightsRandomUniform = input[9]
    weightsRandomDiscrete = input[10]

    hypothesis_dictionary_local = {}  # dictionary that contains the hypothesis
    HB_dictionary_local = {}  # dictionary that contains the hypothesis blood

    alphas = []  # alpha vector for the Dirichlet
    TF = tissueData.copy()
    A = TF.copy()
    TF = TF.drop('tissue', axis=1)

    forceHigh = ""
    forceLow = ""

    # SET THE SEED
    np.random.seed(idx)

    # Either drawn from a random uniform distribution for the lesion weight or into bins
    if weightsRandomUniform:
        for index in tissueData.index:
            wt = np.random.uniform(0.001, 1.0)
            hypothesis_dictionary_local[(idx, index)] = ('Random', wt)
            alphas.append(wt)
        np.random.seed(idx)
        pp = pd.DataFrame(np.random.dirichlet(alphas, len(TF.columns)))
    elif weightsRandomDiscrete:
        discreteRange = np.arange(0.001, 1.1, 0.1)
        for index in tissueData.index:
            wt = random.choice(discreteRange)
            hypothesis_dictionary_local[(idx, index)] = ('Random', wt)
            alphas.append(wt)
        np.random.seed(idx)
        pp = pd.DataFrame(np.random.dirichlet(alphas, len(TF.columns)))
    else:
        if forceMixHypotheses:
            [forceHigh, forceLow] = random.sample(tissueData.index, 2)

        for index in tissueData.index:
            if index == forceHigh:
                wt = random.uniform(highRange[0], highRange[1])  # high_range
            elif index == forceLow:
                wt = random.uniform(lowRange[0], lowRange[1])  # low_range
            else:
                if useMediumBin:
                    bins = ['low', 'high', 'medium']
                else:
                    bins = ['low', 'high']

                selectBin = random.choice(bins)
                if selectBin == 'high':  # Force only Low or High weights
                    wt = random.uniform(highRange[0], highRange[1])  # high_range
                elif selectBin == 'low':
                    wt = random.uniform(lowRange[0], lowRange[1])  # low_range
                else:
                    wt = random.uniform(lowRange[1], highRange[0])  # low_range

            if wt <= lowThd:
                hypothesis_dictionary_local[(idx, index)] = ('Low', wt)
            elif wt >= highThd:
                hypothesis_dictionary_local[(idx, index)] = ('High', wt)
            else:
                hypothesis_dictionary_local[(idx, index)] = ('Medium', wt)

            alphas.append(wt)
            # sumw = sumw + lab[(i,T.loc[index,'tissue'])][1]
        np.random.seed(idx)
        pp = pd.DataFrame(np.random.dirichlet(alphas, len(TF.columns)))
    ik = 0
    for index, row in TF.iterrows():
        cf = []
        for j in range(0, len(row)):
            cf.append((row[j] * pp.loc[j, ik]))
        ik = ik + 1
        TF.loc[index, :] = cf
    hypothesisBlood = TF.sum(axis=0).reset_index()
    hypothesisBlood.columns = ['ids', 'ccf_hat']
    hypothesisBlood = hypothesisBlood.append(uniqueBlood[['ids', 'ccf_hat']])
    HB_dictionary_local[idx] = hypothesisBlood

    res = {}
    res['hypothesis'] = hypothesis_dictionary_local
    res['HB'] = HB_dictionary_local
    return res


# Parallel driver of the hypothesis blood functoin
def generateHypothesisBloodParallel(nthreads, num_hyp, tissueData, highRange, lowRange, Low, High, uniqueBlood,
                                   forceMixHypotheses, useMediumBin, weightsRandomUniform, weightsRandomDiscrete):
    pool = mp.Pool(nthreads)
    results = pool.map(generateHypothesisBlood,
                       [(idx, tissueData, highRange, lowRange, Low, High, uniqueBlood, forceMixHypotheses, useMediumBin,
                         weightsRandomUniform, weightsRandomDiscrete) for idx in iter(range(0, num_hyp))])

    pool.close()

    res = {}
    res['hypothesis'] = {}
    res['HB'] = {}
    for resDicts in results:
        res['HB'].update(resDicts['HB'])
        res['hypothesis'].update(resDicts['hypothesis'])

    return res


# Fast version of HB generation and distance calculation.  THe HB is no longer store so it would need to be recalculated if needed or need to recalculate distances
def generateHypothesisBloodParallelFastFunc(tissueDat, sbIndex, y_realBlood_precision, y_realBlood_recall,
                                           weightsRandomDiscrete, ccf_lesion_thd, uniqueBlood, runIndex, alphas):
    epsilon = 0.0001

    Dist = pd.DataFrame()

    np.random.seed(sbIndex * runIndex)

    tdat = tissueDat.drop('tissue', axis=1)
    if len(alphas) == 0:
        alphas = []
        if weightsRandomDiscrete:
            discreteRange = np.arange(0.005, 1, 0.1)
            for index in tdat.index:
                wt = random.choice(discreteRange)
                alphas.append(wt)

            alphaTriesLimit = 3
            cnt = 1
            while max(alphas) <= 0.005 and cnt <= alphaTriesLimit:
                alphas = []
                for index in tdat.index:
                    wt = random.choice(discreteRange)
                    alphas.append(wt)
                cnt += 1

            if max(alphas) <= 0.005:
                alphas = random.sample(discreteRange, len(tdat))
        else:
            alphas = np.random.uniform(0.001, 1.0, len(tdat))

    np.random.seed(sbIndex * runIndex)

    wts = pd.DataFrame(np.random.dirichlet(alphas, len(tdat.columns)),
                       index=tdat.columns,
                       columns=tdat.index)
    wts = wts.transpose()
    sb = tdat * wts
    sb = sb.sum(axis=0).reset_index()
    sb.columns = ['ids', 'ccf_hat']
    sb = pd.concat( [sb, uniqueBlood[['ids', 'ccf_hat']] ] )

    sb = sb.sort_values('ids')
    for index in tissueDat.index:
        D = tissueDat.loc[index].drop('tissue')
        tt = D[D >= ccf_lesion_thd]

        distCurves = calculateBloodCurves(tt=tt.transpose(), blood=sb)
        y_hypothesisBlood_precision = distCurves['precision']
        y_hypothesisBlood_recall = distCurves['recall']

        name = str(sbIndex)
        ixD = len(Dist)
        Dist.loc[ixD, 'lesion'] = index
        Dist.loc[ixD, 'HB'] = 'HB_' + name

        yreal = y_realBlood_precision.loc[index, :]
        ysyn = y_hypothesisBlood_precision

        Dist.loc[ixD, 'yreal_precision'] = ",".join(map(str, yreal))
        Dist.loc[ixD, 'ysyn_precision'] = ",".join(map(str, ysyn))

        try:
            Dist.loc[ixD, 'Chebyshev_precision'] = distance.chebyshev(yreal, ysyn)
        except:
            Dist.loc[ixD, 'Chebyshev_precision'] = 0
        try:
            Dist.loc[ixD, 'KS_precision'] = stats.ks_2samp(yreal, ysyn)[0]
        except:
            Dist.loc[ixD, 'KS_precision'] = 0

        try:
            Dist.loc[ixD, 'Correlation_precision'] = np.corrcoef(yreal, ysyn)[1, 0]
        except:
            Dist.loc[ixD, 'Correlation_precision'] = 0

        try:
            Dist.loc[ixD, 'Euclidean_precision'] = distance.euclidean(yreal, ysyn)
        except:
            Dist.loc[ixD, 'Euclidean_precision'] = 0
        try:
            Dist.loc[ixD, 'L1_precision'] = sum(abs(np.array(yreal) - np.array(ysyn)))
        except:
            Dist.loc[ixD, 'L1_precision'] = 0

        try:
            Dist.loc[ixD, 'weightedL1_precision'] = np.mean(
                (abs(np.array(yreal) - np.array(ysyn)) + epsilon) / (np.array(ysyn) + epsilon))
        except:
            Dist.loc[ixD, 'weightedL1_precision'] = 0

        yreal = y_realBlood_recall.loc[index, :]
        ysyn = y_hypothesisBlood_recall
        Dist.loc[ixD, 'yreal_recall'] = ",".join(map(str, yreal))
        Dist.loc[ixD, 'ysyn_recall'] = ",".join(map(str, ysyn))

        try:
            Dist.loc[ixD, 'Chebyshev_recall'] = distance.chebyshev(yreal, ysyn)
        except:
            Dist.loc[ixD, 'Chebyshev_recall'] = 0
        try:
            Dist.loc[ixD, 'KS_recall'] = stats.ks_2samp(yreal, ysyn)[0]
        except:
            Dist.loc[ixD, 'KS_recall'] = 0

        try:
            Dist.loc[ixD, 'Correlation_recall'] = np.corrcoef(yreal, ysyn)[1, 0]
        except:
            Dist.loc[ixD, 'Correlation_recall'] = 0

        try:
            Dist.loc[ixD, 'Euclidean_recall'] = distance.euclidean(yreal, ysyn)
        except:
            Dist.loc[ixD, 'Euclidean_recall'] = 0

        try:
            Dist.loc[ixD, 'Chebyshev_recall'] = distance.chebyshev(yreal, ysyn)
        except:
            Dist.loc[ixD, 'Chebyshev_recall'] = 0

        try:
            Dist.loc[ixD, 'L1_recall'] = sum(abs(np.array(yreal) - np.array(ysyn)))
        except:
            Dist.loc[ixD, 'L1_recall'] = 0

        try:
            Dist.loc[ixD, 'weightedL1_recall'] = np.mean(
                (abs(np.array(yreal) - np.array(ysyn)) + epsilon) / (np.array(ysyn) + epsilon))
        except:
            Dist.loc[ixD, 'weightedL1_recall'] = 0

    alphas = pd.DataFrame(alphas, index=tissueDat.index).transpose()
    for alpha in alphas.columns:
        Dist.loc[Dist['lesion'] == alpha, 'weight'] = alphas[alpha][0]
        Dist.loc[Dist['lesion'] == alpha, 'shed'] = 'Random'
    return Dist


# Driver for fast version of HB generation and distance calculation.  THe HB is no longer store so it would need to be recalculated if needed or need to recalculate distances
def generateHypothesisBloodParallelFast(curHypothesisBlood, nthreads, tissueData, params, latestBlood, numHyp, clinData,
                                       uniqueBlood, startIdx=0):
    tissueDat = tissueData.copy()
    # Calculate Real Blood Curves

    y_realBlood_precision = []
    y_realBlood_recall = []
    for index in tissueDat.index:
        D = tissueDat.loc[index].drop('tissue')
        tt = D[D >= params.ccf_lesion_thd]

        distCurves = calculateBloodCurves(tt=tt.transpose(), blood=latestBlood)
        y_realBlood_precision.append(distCurves['precision'])
        y_realBlood_recall.append(distCurves['recall'])

    y_realBlood_precision = pd.DataFrame(y_realBlood_precision, index=tissueDat.index)
    y_realBlood_recall = pd.DataFrame(y_realBlood_recall, index=tissueDat.index)

    weightsRandomDiscrete = params.weightsRandomDiscrete
    ccf_lesion_thd = params.ccf_lesion_thd
    pool = mp.Pool(nthreads)

    results = []

    if len(params.alphasGridSearch) > 0:
        # call apply_async() without callback
        result_objects = [pool.apply_async(generateHypothesisBloodParallelFastFunc,
                                           args=(tissueDat,
                                                 idx,
                                                 y_realBlood_precision,
                                                 y_realBlood_recall,
                                                 weightsRandomDiscrete,
                                                 ccf_lesion_thd,
                                                 uniqueBlood,
                                                 params.runIdx,
                                                 list(params.alphasGridSearch.iloc[idx, :])))
                          for idx in range(startIdx, numHyp + startIdx)]
    else:
        result_objects = [pool.apply_async(generateHypothesisBloodParallelFastFunc,
                                           args=(tissueDat,
                                                 idx,
                                                 y_realBlood_precision,
                                                 y_realBlood_recall,
                                                 weightsRandomDiscrete,
                                                 ccf_lesion_thd,
                                                 uniqueBlood,
                                                 params.runIdx,
                                                 []))
                          for idx in range(startIdx, numHyp + startIdx)]

    pool.close()
    curHypothesisBlood.distance = concatManyPoolResultsDFs(dfs=result_objects, params=params)

    # curHypothesisBlood.distance = concatManyDFs( dfs = results, params = params )
    return curHypothesisBlood


def concatManyPoolResultsDFs(dfs, params):
    md, hd = 'w', True
    for df in dfs:
        df.get().to_csv(os.path.join(params.dir_runOutput, 'df_all.csv'), mode=md, header=hd, index=None)
        md, hd = 'a', False
    df_all = pd.read_csv(os.path.join(params.dir_runOutput, 'df_all.csv'), index_col=None)
    os.remove(os.path.join(params.dir_runOutput, 'df_all.csv'))
    #    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def concatManyDFs(dfs, params):
    md, hd = 'w', True
    for df in dfs:
        df.to_csv(os.path.join(params.dir_runOutput, 'df_all.csv'), mode=md, header=hd, index=None)
        md, hd = 'a', False
    df_all = pd.read_csv(os.path.join(params.dir_runOutput, 'df_all.csv'), index_col=None)
    os.remove(os.path.join(params.dir_runOutput, 'df_all.csv'))
    return df_all


# Function to calculate the distance of the hypothesis blood from the real in each lesion
def calculateBloodDist(input):
    epsilon = 0.0001
    i = input[0]
    tt = input[1]
    index = input[2]
    y_realBlood_precision = input[3]
    y_realBlood_recall = input[4]
    hypothesisBlood = input[5]
    Dist = pd.DataFrame()

    hypothesisBlood = hypothesisBlood.sort_values('ccf_hat')
    y_hypothesisBlood_precision = []
    y_hypothesisBlood_recall = []
    for ccf_thd in np.arange(0.05, 1.0, 0.05):
        traw_ccf = tt[tt > ccf_thd]
        bloodPassThd = set(hypothesisBlood[hypothesisBlood['ccf_hat'] > ccf_thd]['ids'].tolist())
        lesionPassThd = set(traw_ccf.index)

        # Precision
        try:
            if len(bloodPassThd) > 0:
                if float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(bloodPassThd)) > 0:
                    y_hypothesisBlood_precision.append(
                        float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(bloodPassThd)))
                else:
                    y_hypothesisBlood_precision.append(0)
            else:
                y_hypothesisBlood_precision.append(0)
        except:
            y_hypothesisBlood_precision.append(0)

        # Recall
        try:
            if len(lesionPassThd) > 0:
                if float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(lesionPassThd)) > 0:
                    y_hypothesisBlood_recall.append(
                        float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(lesionPassThd)))
                else:
                    y_hypothesisBlood_recall.append(0)
            else:
                y_hypothesisBlood_recall.append(0)
        except:
            y_hypothesisBlood_recall.append(0)

    name = str(i)

    ixD = len(Dist)
    Dist.loc[ixD, 'lesion'] = index
    Dist.loc[ixD, 'HB'] = 'HB_' + name

    yreal = y_realBlood_precision
    ysyn = y_hypothesisBlood_precision
    Dist.loc[ixD, 'yreal_precision'] = ",".join(map(str, yreal))
    Dist.loc[ixD, 'ysyn_precision'] = ",".join(map(str, ysyn))
    Dist.loc[ixD, 'KS_precision'] = stats.ks_2samp(yreal, ysyn)[0]
    Dist.loc[ixD, 'Correlation_precision'] = np.corrcoef(yreal, ysyn)[1, 0]
    Dist.loc[ixD, 'Euclidean_precision'] = distance.euclidean(yreal, ysyn)
    Dist.loc[ixD, 'Chebyshev_precision'] = distance.chebyshev(yreal, ysyn)
    Dist.loc[ixD, 'L1_precision'] = sum(abs(np.array(yreal) - np.array(ysyn)))
    Dist.loc[ixD, 'weightedL1_precision'] = np.mean(
        (abs(np.array(yreal) - np.array(ysyn)) + epsilon) / (np.array(ysyn) + epsilon))

    yreal = y_realBlood_recall
    ysyn = y_hypothesisBlood_recall
    Dist.loc[ixD, 'yreal_recall'] = ",".join(map(str, yreal))
    Dist.loc[ixD, 'ysyn_recall'] = ",".join(map(str, ysyn))
    Dist.loc[ixD, 'KS_recall'] = stats.ks_2samp(yreal, ysyn)[0]
    Dist.loc[ixD, 'Correlation_recall'] = np.corrcoef(yreal, ysyn)[1, 0]
    Dist.loc[ixD, 'Euclidean_recall'] = distance.euclidean(yreal, ysyn)
    Dist.loc[ixD, 'Chebyshev_recall'] = distance.chebyshev(yreal, ysyn)
    Dist.loc[ixD, 'L1_recall'] = sum(abs(np.array(yreal) - np.array(ysyn)))
    Dist.loc[ixD, 'weightedL1_recall'] = np.mean(
        (abs(np.array(yreal) - np.array(ysyn)) + epsilon) / (np.array(ysyn) + epsilon))
    return Dist


# Parallel driver of the hypothesis blood functoin
def calculateBloodDistParallel(nthreads, num_hyp, tissueData, highRange, lowRange, Low, High, uniqueBlood, params,
                               latestBlood, HB_dictionary):
    ##This can be made parallel!!
    TT = tissueData.copy()
    allDist = pd.DataFrame()
    pool = mp.Pool(nthreads)

    for index in TT.index:
        D = tissueData.loc[index].drop('tissue')
        tt = D[D >= params.ccf_lesion_thd]

        # compute recall and precision for real blood
        y_realBlood_precision = []
        y_realBlood_recall = []
        for ccf_thd in np.arange(0.05, 1.0, 0.05):
            traw_ccf = tt[tt > ccf_thd]
            bloodPassThd = set(latestBlood[latestBlood['ccf_hat'] > ccf_thd]['ids'].tolist())
            lesionPassThd = set(traw_ccf.index)

            # Precision
            if len(bloodPassThd) > 0:
                y_realBlood_precision.append(
                    float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(bloodPassThd)))
            else:
                y_realBlood_precision.append(0)
                # Recall
            if len(lesionPassThd) > 0:
                y_realBlood_recall.append(
                    float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(lesionPassThd)))
            else:
                y_realBlood_recall.append(0)

        results = pool.map(calculateBloodDist,
                           [(idx, tt, index, y_realBlood_precision, y_realBlood_recall, HB_dictionary[idx]) for idx in
                            iter(range(0, num_hyp))])
        allDist = pd.concat([allDist, pd.concat(results)])

    pool.close()
    return allDist


# calculate the blood lesion curves
def calculateBloodCurves(tt, blood):
    y_blood_precision = []
    y_blood_recall = []
    for ccf_thd in np.arange(0.05, 1.0, 0.05):
        traw_ccf = tt[tt > ccf_thd]
        bloodPassThd = set(blood[blood['ccf_hat'] > ccf_thd]['ids'].tolist())
        lesionPassThd = set(traw_ccf.index)

        # Precision
        try:
            if float(len(bloodPassThd)) > 0: 
                try:
                    val = float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(bloodPassThd))
                    y_blood_precision.append(val)
                except:
                    y_blood_precision.append(0)

            else:
                y_blood_precision.append(0)
        except:
            y_blood_precision.append(0)

        # Recall
        try:
            if float(len(lesionPassThd)) > 0:
                try:
                    val = float(len(bloodPassThd.intersection(lesionPassThd))) / float(len(lesionPassThd))
                    y_blood_recall.append(val)
                except:
                    y_blood_recall.append(0)

            else:
                y_blood_recall.append(0)
        except:
            y_blood_recall.append(0)

    return {'precision': y_blood_precision, 'recall': y_blood_recall}


# Evaluate the fit by calculating mean over the specified metric and sort into table
def evaluateFit(hypothesisBloodDist, metric="Chebyshev_precision"):
    # Select distance and pivot so that each row is a synthethic blood and column is a lesion.
    hypothesisBloodDistTable = hypothesisBloodDist.pivot('HB', 'lesion', metric)
    hypothesisBloodDistTable['mean'] = hypothesisBloodDistTable.mean(axis=1)  # Select best hypothesis
    hypothesisBloodDistTable = hypothesisBloodDistTable.sort_values('mean')  # Sort so that best hypothesis is a index[0]
    return (hypothesisBloodDistTable)


# Compute score and plot
# def computeScorePlot_best(tissueData, hypothesisBloodDistTable, hypothesis_dictionary, tissueColor):
#     TT = tissueData.sort_values('tissue')
#
#     score_x_l = []
#     score_y_l = []
#     for index in TT.index:
#         for i in [hypothesisBloodDistTable[index].idxmin()]:
#             score_x_l.append(index)
#             score_y_l.append(hypothesisBloodDistTable.loc[i, index])
#
#     trace3 = go.Scatter(
#         x=score_x_l,
#         y=score_y_l,
#         mode='markers',
#         name='Th. (lesion) best-fit',
#         marker=dict(
#             size=7,
#             color="orange"
#         ))
#
#     score_x = []
#     score_y = []
#     for index in TT.index:
#         for i in [hypothesisBloodDistTable.index[0]]:  # ,  XX[index].idxmax(),  XX.index[-1]]:#range(-5,7):
#             score_x.append(index)
#             score_y.append(hypothesisBloodDistTable.loc[i, index])
#
#     trace4 = go.Scatter(
#         x=score_x,
#         y=score_y,
#         mode='markers',
#         name='H lesion fit',
#         marker=dict(
#             size=7,
#             color="green"
#         ))
#     trace2 = go.Scatter(
#         x=[score_x[0], score_x[-1]],
#         y=[hypothesisBloodDistTable.loc[hypothesisBloodDistTable.index[0], 'mean'],
#            hypothesisBloodDistTable.loc[hypothesisBloodDistTable.index[0], 'mean']],
#         mode='lines',
#         name='best-fit (H)',
#         marker=dict(
#             size=7,
#             color="green"
#         ))
#     tscore = []
#     for i in range(0, len(score_y)):
#         tscore.append(score_y[i] - score_y_l[i])
#
#     fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True,
#                               subplot_titles=(str(np.mean(tscore)), 'Hypothesis (H)'))
#     fig.append_trace(trace3, 1, 1)
#     fig.append_trace(trace2, 1, 1)
#     fig.append_trace(trace4, 1, 1)
#
#     xlab = []
#     ylab = []
#     il = int(hypothesisBloodDistTable.index[0].replace('HB_', ''))
#     for index in TT.index:
#         xlab.append(index)
#         ylab.append(hypothesis_dictionary[(il, index)][1])
#
#     trace5 = go.Scatter(
#         x=xlab,
#         y=ylab,
#         mode='markers',
#         name='Weight',  # lab[index][0],
#         # legendgroup = 'group_'+str(i),
#         showlegend=False,
#         marker=dict(
#             size=7,
#             color="green",
#             # color = cT[icol]  #i+5
#             symbol='square',
#             # line = dict( width = 2, color = cT[icol]) #T.loc[index,'tissue'] #cT[i+5]
#
#         ))
#     fig.append_trace(trace5, 2, 1)
#
#     for ts in TT['tissue'].unique():
#         trace = go.Scatter(
#             x=[TT[TT['tissue'] == ts].index[0], TT[TT['tissue'] == ts].index[-1]],
#             y=[-0.1, -0.1],
#             mode='lines+text',
#             name=ts,  # lab[index][0],
#             text=[ts],
#             textposition='top center',
#             # legendgroup = 'group_'+str(i),
#             showlegend=False,
#             marker=dict(
#                 size=7,
#                 color=tissueColor[ts],
#             ))
#         fig.append_trace(trace, 2, 1)
#     offline.iplot(fig, filename='styled-scatter')


# Load a patients saved pickle
def loadPatientBloodObj(outputDir, patID):
    res = {}
    res = pickle.load(open(os.path.join(outputDir, "hypothesisBloodObj-" + patID + ".pkl"), "rb"))
    res['dist'] = pickle.load(open(os.path.join(outputDir, "hypothesisBloodDistObj-" + patID + ".pkl"), "rb"))
    return res


# Get Hypothesis values given a list of hypothesis indexes
def getHypothValues(hypoth, distance, hypotheses, clinData):
    if len(hypotheses) > 0:  # This is to maintain legacy support for the older hypotheses object
        hypothValue = getHypothValuesOld(hypoth, distance, hypotheses, clinData)
    else:
        hypoth = ['HB_' + str(x) for x in hypoth]
        hypothValue = distance[distance['HB'].isin(hypoth)][['HB', 'lesion', 'shed', 'weight']]
        hypothValue.columns = ['HB', 'primarySampleID', 'shed', 'weight']
        hypothValue.loc[:, 'sampleIDOriginal'] = [clinData.sampleIDMap[x] for x in hypothValue['primarySampleID']]
        hypothValue.loc[:, 'tissue'] = [
            re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(clinData.tissueMap[x]))) for x in
            hypothValue['primarySampleID']]
        hypothValue.loc[:, 'tissueOriginal'] = [clinData.tissueOriginalMap[x] for x in hypothValue['primarySampleID']]
        hypothValue = hypothValue.sort_values(by=['tissue', 'primarySampleID'])
    return hypothValue


# Get Hypothesis values given a list of hypothesis indexes
def getHypothValuesOld(hypoth, distance, hypotheses, clinData):
    hypothToGet = list(itertools.product(hypoth, list(set(distance['lesion']))))
    hypothValue = []
    for h in hypothToGet:
        hypothValue.append([item for sublist in [h, hypotheses[h]] for item in sublist])
    hypothValue = pd.DataFrame(hypothValue, columns=['hypoth', 'primarySampleID', 'shed', 'weight'])
    hypothValue.loc[:, 'HB'] = ['HB_' + str(x) for x in hypothValue['hypoth']]
    hypothValue.loc[:, 'sampleIDOriginal'] = [clinData.sampleIDMap[x] for x in hypothValue['primarySampleID']]
    hypothValue.loc[:, 'tissue'] = [
        re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(clinData.tissueMap[x]))) for x in
        hypothValue['primarySampleID']]
    hypothValue.loc[:, 'tissueOriginal'] = [clinData.tissueOriginalMap[x] for x in hypothValue['primarySampleID']]
    hypothValue = hypothValue.sort_values(by=['tissue', 'primarySampleID'])
    return hypothValue


def getHypothWeight(hypothValue, useWeight=False):
    if useWeight or 'Random' in hypothValue['shed'].tolist():
        return hypothValue.pivot("HB", 'primarySampleID', "weight")
    else:
        hypothValue.loc[:, 'shed'] = [x.lower() for x in hypothValue['shed']]
        return hypothValue.pivot("HB", 'primarySampleID', "shed")


# Plot the scatter and distribution for feature from the netstats
def plotNetStatsFeature(feature, params, netStats, edgeThres, runIndexes, numToGet, fileNameSuffix='', palette=None):
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x='in', y='out', hue=feature, data=netStats, palette=palette)
    plt.legend(loc='upper right', ncol=3)
    plt.savefig(
        os.path.join(params.dir_output, "CrossComparison" + feature + "ScatterDegree" + str(runIndexes[0]) + "-" + str(
            runIndexes[-1]) + "_Top" + str(numToGet) + "HB_" + re.sub("0.", "",
                                                                      str(edgeThres)) + fileNameSuffix + ".png"),
        dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 10))
    featStat = netStats.melt(id_vars=[feature, 'patID'])
    featStat = featStat[featStat['variable'].isin(['in', 'out'])]
    featStat['value'] = [float(x) for x in featStat['value']]
    tisCnt = netStats[feature].value_counts()
    ax = sns.boxplot(x=feature, y='value', hue='variable', data=featStat)
    plt.xticks(rotation=90)
    ax.set_xticklabels([x.get_text() + " (" + str(tisCnt[x.get_text()]) + ")" for x in ax.get_xticklabels()])
    # plt.legend( loc='upper right', bbox_to_anchor=(2, 1.1), ncol = 3)
    plt.tight_layout()
    plt.savefig(os.path.join(params.dir_output, "CrossComparison" + feature + "Degree" + str(runIndexes[0]) + "-" + str(
        runIndexes[-1]) + "_Top" + str(numToGet) + "HB_" + re.sub("0.", "", str(edgeThres)) + fileNameSuffix + ".png"),
                dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=netStats[['outFreqCat', feature]], hue=feature, x='outFreqCat')
    ax.set_xlabel('Shedding level', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=15)
    #    plt.xticks(rotation=90)
    plt.legend(loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(params.dir_output, "CrossComparison" + feature + "OutFreqCat" + str(runIndexes[0]) + "-" + str(
            runIndexes[-1]) + "_Top" + str(numToGet) + "HB_" + re.sub("0.", "",
                                                                      str(edgeThres)) + fileNameSuffix + ".png"),
        dpi=150)
    plt.show()
    plt.close()


def plotBoxplotWeights(hypothValue, clinData, legend=True):
    import matplotlib.patches as mpatches
    lesResp = dict()
    hasResp = False
    for x in hypothValue.sampleID.unique():
        if x in clinData.lesionResponseMap.keys():
            lesResp[clinData.sampleIDMap[x]] = clinData.lesionResponseMap[x]
            hasResp = True
        else:
            lesResp[clinData.sampleIDMap[x]] = "NA"

    tisColor = [clinData.tissueColor[clinData.tissueMap[x]].upper() for x in hypothValue.sampleID.unique()]
    ax = sns.boxplot(x='sampleIDOriginal', y='weight', data=hypothValue, hue='type')
    for xtick, color in zip(ax.get_xticklabels(), tisColor):
        xtick.set_backgroundcolor(color)
    ax.set(xlabel="Samples")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14)

    if hasResp:
        lesRespXlab = [lesResp[re.sub(".\\).*", "", re.sub("^.*,.", "", str(x)))] for x in ax.get_xticklabels()]
        for i in range(0, len(lesRespXlab)):
            ax.text(i + 0.25, -0.15, lesRespXlab[i], fontsize=14, color="red")

    if legend:
        firstLegend = plt.legend(bbox_to_anchor=(0.2, 1.1), loc="upper center", borderaxespad=0.1, ncol=6)
        ax = plt.gca().add_artist(firstLegend)

        # Create another legend for the second line.
        patches = []
        tis = hypothValue.tissue.unique()
        for t in tis:
            patch = mpatches.Patch(color=clinData.tissueColor[t], label=t)
            patches.append(patch)
        plt.legend(patches, tis, loc='upper right', bbox_to_anchor=(1.1, 1.1), ncol=6)
    else:
        ax.get_legend().remove()
    return ax


def discretizeWeights(params, x):
    if x >= params.highThreshold_alphaBin:
        x = 1
    elif x <= params.lowThreshold_alphaBin:
        x = 0
    else:
        x = 0.5
    return x


def plotHeatmapWeight(hypoth, scores, curHypothesisBlood, clinData):
    scoresDict = dict(
        zip([re.sub('hypothesisBlood', 'HB', x) for x in list(scores.index)], [round(x, 2) for x in list(scores)]))
    hypothValues = getHypothValues(hypoth=hypoth[0:10], distance=curHypothesisBlood.distance,
                                   hypotheses=curHypothesisBlood.hypotheses, clinData=clinData)
    weights = getHypothWeight(hypothValue=hypothValues)
    if 'Random' not in hypothValues['shed'].tolist():
        weights[weights == "high"] = 1
        weights[weights == "medium"] = 0.5
        weights[weights == "low"] = 0
    temp = pd.DataFrame(list(zip(weights.columns, [clinData.tissueMap[x] for x in weights.columns])),
                        columns=['primarySampleID', 'tissue'])
    temp = temp.sort_values(by=['tissue', 'primarySampleID'])
    weights = weights[temp['primarySampleID']]
    weights = weights.loc[['HB_' + str(x) for x in hypoth[0:10]], :]
    weights.index = [re.sub('hypothesisBlood', 'HB', x) + " (" + str(scoresDict[x]) + ")" for x in weights.index]
    g = sns.clustermap(weights.fillna(-1), row_cluster=False, col_cluster=False,
                       col_colors=[clinData.tissueColor[clinData.tissueMap[x]] for x in weights.columns],
                       linewidths=0, vmin=0, vmax=1,
                       figsize=(30, 10),
                       xticklabels=[clinData.sampleIDMap[x] for x in weights.columns])
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    return g


def plotHeatmapandBox(scores, hypothBest, hypothValueAll, curHypothesisBlood, clinData, fileName, title):
    g = plotHeatmapWeight(hypothBest, scores, curHypothesisBlood, clinData)
    g.gs.update(left=0.05, right=0.30, bottom=0.3)

    gs2 = matplotlib.gridspec.GridSpec(1, 1, left=0.4, bottom=0.35, top=0.8)
    ax2 = g.fig.add_subplot(gs2[0])
    ax = plotBoxplotWeights(hypothValue=hypothValueAll, clinData=clinData, legend=True)

    plt.suptitle(title)
    plt.savefig(fileName, dpi=300, pad_inches=0)
    # plt.show()
    plt.close()


def getHypothRandomMeanWeights(numToGet, distance, hypotheses, clinData, numHyp, numRandom=1000):
    hypothRandomMeanWeight = pd.DataFrame()
    for x in range(0, numRandom):
        hypothRandom = random.sample(range(0, numHyp), numToGet)
        hypothValueRandom = getHypothValues(hypoth=hypothRandom, distance=distance, hypotheses=hypotheses,
                                            clinData=clinData)
        hypothRandomMeanWeight = hypothRandomMeanWeight.append(
            pd.DataFrame(hypothValueRandom.pivot("hypoth", 'primarySampleID', "weight").mean(axis=0),
                         columns=['weight']))

    hypothRandomMeanWeight['primarySampleID'] = hypothRandomMeanWeight.index
    hypothRandomMeanWeight.loc[:, 'sampleIDOriginal'] = [clinData.sampleIDMap[x] for x in
                                                         hypothRandomMeanWeight['primarySampleID']]
    hypothRandomMeanWeight.loc[:, 'tissue'] = [
        re.sub("^chest.*", "chest", re.sub("^abdominal.*", "abdominal_cavity", str(clinData.tissueMap[x]))) for x in
        hypothRandomMeanWeight['primarySampleID']]
    hypothRandomMeanWeight = hypothRandomMeanWeight.sort_values(by=['tissue', 'primarySampleID'])

    return hypothRandomMeanWeight


def plotHypothWeights(scores, patID, scoreName, outputDir, curHypothesisBlood, clinData, numHyp, numToGet=10):
    # Weights of the best hypotheses
    numToGet = max(numToGet, 10)
    scores = scores.sort_values(axis=0, ascending=True)
    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
    hypothValueBest = getHypothValues(hypoth=hypothBest, distance=curHypothesisBlood.distance,
                                      hypotheses=curHypothesisBlood.hypotheses, clinData=clinData)

    scores = scores.sort_values(axis=0, ascending=False)
    hypothWorst = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
    hypothValueWorst = getHypothValues(hypoth=hypothWorst, distance=curHypothesisBlood.distance,
                                       hypotheses=curHypothesisBlood.hypotheses, clinData=clinData)

    hypothRandom = random.sample(range(0, numHyp), numToGet)
    hypothValueRandom = getHypothValues(hypoth=hypothRandom, distance=curHypothesisBlood.distance,
                                        hypotheses=curHypothesisBlood.hypotheses, clinData=clinData)

    hypothValueBest.loc[:, 'type'] = ['top'] * len(hypothValueBest)
    hypothValueWorst.loc[:, 'type'] = ['bottom'] * len(hypothValueWorst)
    hypothValueRandom.loc[:, 'type'] = ['random'] * len(hypothValueRandom)
    hypothValueAll = hypothValueBest
    hypothValueAll = hypothValueAll.append(hypothValueWorst, sort=True)
    hypothValueAll = hypothValueAll.append(hypothValueRandom, sort=True)

    plotHeatmapandBox(scores=scores, hypothBest=hypothBest, hypothValueAll=hypothValueAll,
                      curHypothesisBlood=curHypothesisBlood, clinData=clinData,
                      title="Weights of Top 10 (left - binary) and Top, Bottom, Random " + str(
                          numToGet) + " Hypotheses for " + scoreName + "\nDistance of Hypothesis Blood (" + str(
                          numHyp) + " hypotheses) for " + clinData.patientIDMap[patID],
                      fileName=os.path.join(outputDir, "HypothWeightByTopBottom" + str(
                          numToGet) + "_" + scoreName + "_" + patID + ".png"))

    hypothValueAll = hypothValueBest
    hypothValueAll = hypothValueAll.append(hypothValueWorst, sort=True)
    plotHeatmapandBox(scores=scores, hypothBest=hypothBest, hypothValueAll=hypothValueAll,
                      curHypothesisBlood=curHypothesisBlood, clinData=clinData,
                      title="Weights of Top 10 (left - binary) and Top, Bottom, Random " + str(
                          numToGet) + " Hypotheses for " + scoreName + "\nDistance of Hypothesis Blood (" + str(
                          numHyp) + " hypotheses) for " + clinData.patientIDMap[patID],
                      fileName=os.path.join(outputDir, "HypothWeightByTopBottomNoRandom" + str(
                          numToGet) + "_" + scoreName + "_" + patID + ".png"))


# Plot the Top and Bottom Hypotheses
def getBestHypothAll(patAvailable, metric, numToGet, hypothesis_dictionary, hypothesisBloodDist_ALL, clinData,
                     outputDir):
    bestDict = {}
    worstDict = {}
    for metrics in ['precision', 'recall', 'sumPrecisionRecall']:
        bestDict[metrics] = pd.DataFrame()
        worstDict[metrics] = pd.DataFrame()

    bestHypoth = pd.DataFrame()
    worstHypoth = pd.DataFrame()
    for patID in patAvailable:
        hypothesisBloodDist = hypothesisBloodDist_ALL[hypothesisBloodDist_ALL['participantID'] == patID]
        hypothesisBloodDistTable_prec = evaluateFit(hypothesisBloodDist, metric=metric + "_precision")
        hypothesisBloodDistTable_recall = evaluateFit(hypothesisBloodDist, metric=metric + "_recall")
        hypothesisBloodDistTable_prec = hypothesisBloodDistTable_prec.sort_index(axis=0)
        hypothesisBloodDistTable_recall = hypothesisBloodDistTable_recall.sort_index(axis=0)

        prec = hypothesisBloodDistTable_prec['mean']
        rec = hypothesisBloodDistTable_recall['mean']
        comb = prec + rec

        scoreName = metric + "_precision"
        scores = prec.sort_values(axis=0, ascending=True)
        hypoth = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
        bestDict['precision'] = bestDict['precision'].append(
            getHypothValues(hypoth=hypoth, distance=hypothesisBloodDist, hypotheses=hypothesis_dictionary,
                            clinData=clinData))

        scores = prec.sort_values(axis=0, ascending=False)
        hypoth = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
        worstDict['precision'] = worstDict['precision'].append(
            getHypothValues(hypoth=hypoth, distance=hypothesisBloodDist, hypotheses=hypothesis_dictionary,
                            clinData=clinData))

        scoreName = metric + "recall"
        scores = rec.sort_values(axis=0, ascending=True)
        hypoth = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
        bestDict['recall'] = bestDict['recall'].append(
            getHypothValues(hypoth=hypoth, distance=hypothesisBloodDist, hypotheses=hypothesis_dictionary,
                            clinData=clinData))

        scores = rec.sort_values(axis=0, ascending=False)
        hypoth = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
        worstDict['recall'] = worstDict['recall'].append(
            getHypothValues(hypoth=hypoth, distance=hypothesisBloodDist, hypotheses=hypothesis_dictionary,
                            clinData=clinData))

        scoreName = metric + "_sumPrecisionRecall"
        scores = comb.sort_values(axis=0, ascending=True)
        hypoth = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
        bestDict['sumPrecisionRecall'] = bestDict['sumPrecisionRecall'].append(
            getHypothValues(hypoth=hypoth, distance=hypothesisBloodDist, hypotheses=hypothesis_dictionary,
                            clinData=clinData))

        scores = comb.sort_values(axis=0, ascending=False)
        hypoth = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
        worstDict['sumPrecisionRecall'] = worstDict['sumPrecisionRecall'].append(
            getHypothValues(hypoth=hypoth, distance=hypothesisBloodDist, hypotheses=hypothesis_dictionary,
                            clinData=clinData))

    for metrics in ['precision', 'recall', 'sumPrecisionRecall']:
        scoreName = metric + "_" + metrics
        bestHypoth = bestDict[metrics]
        worstHypoth = worstDict[metrics]
        bestHypoth.loc[:, 'type'] = ['top']
        worstHypoth.loc[:, 'type'] = ['bottom']
        allHypoth = bestHypoth
        allHypoth = allHypoth.append(worstHypoth)

        # Box plot of top and bottom Weights for all patients
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        ax = sns.boxplot(x='sampleIDOriginal', y='weight', data=bestHypoth, hue='tissue',
                         palette=clinData.tissueColor)
        ax.set(xlabel="Samples")
        ax.set_title("Top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.legend(bbox_to_anchor=(.5, 1.1), loc="upper center", borderaxespad=0.1, ncol=10)
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        ax = sns.boxplot(x='sampleIDOriginal', y='weight', data=worstHypoth, hue='tissue',
                         palette=clinData.tissueColor)
        ax.set(xlabel="Samples")
        ax.set_title("Bottom")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14)
        ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "Boxplot_TopBottomHypothWeights" + str(
            numToGet) + "_ALLPatients_" + scoreName + "_byPatient.png"), dpi=300)
        plt.close()

        # Boxplot of top weights by tissues
        bestHypoth.loc[:, 'type'] = ['top']
        allHypoth = bestHypoth
        tissue = list(allHypoth['tissue'])
        tissue.sort()
        cnts = dict([[key, len(list(group)) / numToGet] for key, group in groupby(tissue)])
        allHypoth.loc[:, 'tissueCnt'] = [x + " (" + str(cnts[x]) + ")" for x in allHypoth['tissue']]
        plt.figure(figsize=(20, 10))
        ax = sns.boxplot(x='tissueCnt', y='weight', data=allHypoth, hue='type')
        ax.set(xlabel="Tissue")
        ax.set_title("Top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14, backgroundcolor="gray")
        tisColor = [clinData.tissueColor[re.sub("^.*\'", "", re.sub(" .*", "", str(x)))] for x in ax.get_xticklabels()]
        for xtick, color in zip(ax.get_xticklabels(), tisColor):
            xtick.set_backgroundcolor(color)
        plt.legend(bbox_to_anchor=(.5, 1.1), loc="upper center", borderaxespad=0.1, ncol=10)
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "Boxplot_TopHypothWeights" + str(
            numToGet) + "_ALLPatients_" + scoreName + "_byTissue.png"), dpi=300)
        plt.close()

        # Boxplot of top and bottom weights by tissues
        bestHypoth.loc[:, 'type'] = ['top']
        worstHypoth.loc[:, 'type'] = ['bottom']
        allHypoth = bestHypoth
        allHypoth = allHypoth.append(worstHypoth)
        tissue = list(allHypoth['tissue'])
        tissue.sort()
        cnts = dict([[key, len(list(group)) / numToGet] for key, group in groupby(tissue)])
        allHypoth.loc[:, 'tissueCnt'] = [x + " (" + str(cnts[x]) + ")" for x in allHypoth['tissue']]
        plt.figure(figsize=(20, 10))
        ax = sns.boxplot(x='tissueCnt', y='weight', data=allHypoth, hue='type')
        ax.set(xlabel="Tissue")
        ax.set_title("Top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14, backgroundcolor="gray")
        tisColor = [clinData.tissueColor[re.sub("^.*\'", "", re.sub(" .*", "", str(x)))] for x in ax.get_xticklabels()]
        for xtick, color in zip(ax.get_xticklabels(), tisColor):
            xtick.set_backgroundcolor(color)
        plt.legend(bbox_to_anchor=(.5, 1.1), loc="upper center", borderaxespad=0.1, ncol=10)
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "Boxplot_TopBottomHypothWeights" + str(
            numToGet) + "_ALLPatients_" + scoreName + "_byTissue.png"), dpi=300)
        plt.close()

        # Boxplot of top  weights by Response
        bestHypoth.loc[:, 'type'] = ['top']
        allHypoth = bestHypoth
        allHypoth.loc[:, "Response"] = [
            clinData.lesionResponseMap[x] if x in clinData.lesionResponseMap.keys() else 'NA' for x in
            allHypoth['primarySampleID']]
        Response = list(allHypoth['Response'])
        Response.sort()
        cnts = dict([[key, len(list(group)) / numToGet] for key, group in groupby(Response)])
        allHypoth.loc[:, 'ResponseCnt'] = [x + " (" + str(cnts[x]) + ")" for x in allHypoth['Response']]
        plt.figure(figsize=(20, 10))
        ax = sns.boxplot(x='ResponseCnt', y='weight', data=allHypoth, hue='type')
        ax.set(xlabel="Response")
        ax.set_title("Top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "Boxplot_TopHypothWeights" + str(
            numToGet) + "_ALLPatients_" + scoreName + "_byResponse.png"), dpi=300)
        plt.close()

        # Boxplot of top and bottom weights by Response
        bestHypoth.loc[:, 'type'] = ['top']
        worstHypoth.loc[:, 'type'] = ['bottom']
        allHypoth = bestHypoth
        allHypoth = allHypoth.append(worstHypoth)
        allHypoth.loc[:, "Response"] = [
            clinData.lesionResponseMap[x] if x in clinData.lesionResponseMap.keys() else 'NA' for x in
            allHypoth['primarySampleID']]
        Response = list(allHypoth['Response'])
        Response.sort()
        cnts = dict([[key, len(list(group)) / numToGet] for key, group in groupby(Response)])
        allHypoth.loc[:, 'ResponseCnt'] = [x + " (" + str(cnts[x]) + ")" for x in allHypoth['Response']]
        plt.figure(figsize=(20, 10))
        ax = sns.boxplot(x='ResponseCnt', y='weight', data=allHypoth, hue='type')
        ax.set(xlabel="Response")
        ax.set_title("Top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "Boxplot_TopBottomHypothWeights" + str(
            numToGet) + "_ALLPatients_" + scoreName + "_byResponse.png"), dpi=300)
        plt.close()

        # Boxplot of top and bottom weights by original tissues
        tissue = list(bestHypoth['tissueOriginal'])
        tissue.sort()
        cnts = dict([[key, len(list(group)) / numToGet] for key, group in groupby(tissue)])
        bestHypoth.loc[:, 'tissueOriginalCnt'] = [x + " (" + str(cnts[x]) + ")" for x in bestHypoth['tissueOriginal']]
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        ax = sns.boxplot(x='tissueOriginalCnt', y='weight', data=bestHypoth, hue='tissueOriginal', width=1)
        ax.set(xlabel="Tissue")
        ax.set_title("Top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.legend(bbox_to_anchor=(.5, 1.1), loc="upper center", borderaxespad=0.1, ncol=10)
        plt.tight_layout()

        tissue = list(worstHypoth['tissueOriginal'])
        tissue.sort()
        cnts = dict([[key, len(list(group)) / numToGet] for key, group in groupby(tissue)])
        worstHypoth.loc[:, 'tissueOriginalCnt'] = [x + " (" + str(cnts[x]) + ")" for x in worstHypoth['tissueOriginal']]

        plt.subplot(2, 1, 2)
        ax = sns.boxplot(x='tissueOriginalCnt', y='weight', data=worstHypoth, hue='tissueOriginal', width=1)
        ax.set(xlabel="Tissue")
        ax.set_title("Bottom")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "Boxplot_TopBottomHypothWeights" + str(
            numToGet) + "_ALLPatients_" + scoreName + "_byTissueOriginal.png"), dpi=300)
        plt.close()


# Get the genes that under the recurrence limit
def getGenesUnderRecurrencenLimit(tissueData, recurrenceLimit, minCCF=0.05):
    tisDat = tissueData.copy()
    tisDat = tisDat.drop('tissue', axis=1)
    return [x for x in tisDat.columns if
            sum(tisDat[x] >= minCCF) / float(tisDat.shape[0]) <= recurrenceLimit]


# Plot Mutations per lesion
def plotMutationsPerLesion(params, hypothesisBloodDist_ALL, mafData, clinData):
    plt.figure(figsize=(20, 20))
    idx = 1
    patAvailable = list(set(hypothesisBloodDist_ALL['participantID']))
    for patID in patAvailable:
        patData = mafData[mafData['participantID'] == patID].copy()
        patData.loc[:, 'ids'] = patData['Hugo_Symbol'] + '-' + patData['Start_position'].astype('str')
        patSamples = list(set(patData['primarySampleID']))
        tissueData = pd.DataFrame()
        for samp in patSamples:
            sampData = patData[patData['primarySampleID'] == samp].copy()
            sampData.loc[:, 'sample'] = [re.sub("_-", "-", re.sub('_v[0-9]*_Exome.*', '', re.sub('^.*/', '', x))) for x
                                         in sampData['sample']]
            sampData = sampData[sampData['sample'] == clinData.sampleIDMap[samp]]
            tissueData = tissueData.append(sampData)

        tissueData.loc[:, 'sampleIDOriginal'] = [clinData.sampleIDMap[x] for x in tissueData['primarySampleID']]
        tissueData.loc[:, 'tissue'] = [clinData.tissueMap[x] for x in tissueData['primarySampleID']]
        tissueData = tissueData.sort_values(by=['tissue', 'sampleIDOriginal'])

        plt.subplot(3, 3, idx)
        ax = sns.countplot(x='sampleIDOriginal', data=tissueData, hue='tissue',
                           palette=clinData.tissueColor)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title(clinData.patientIDMap[patID])
        plt.legend(loc="upper center", borderaxespad=0.1, ncol=6)
        plt.tight_layout()
        idx = idx + 1

    plt.savefig(os.path.join(params.dir_output, "MutationsPerLesion.png"), dpi=150)
    plt.close()


# Function to drive adapative
def runAdapativeGridSearch(curHypothesisBlood, params, mafData, clinData, verbose, samplesToUse, dropNullTissue = True):
    patID = curHypothesisBlood.patID

    hypothesis_dictionary = dict()
    hypothesisBloodDist = pd.DataFrame()

    tissueData = pd.DataFrame()
    tissueData = _getTissueData(mafData=mafData, patID=patID, clinData=clinData, dropNullTissue =dropNullTissue)

    if len(tissueData) > 0:
        # Drop The Low Purity Samples
        tissueData = _dropLowPuritySamples(tissueData=tissueData, tumorFractions=clinData.tumorFractions)

    if len(tissueData) > 0:
        # Get genes under the recurrence rate
        columnsToGet = getGenesUnderRecurrencenLimit(tissueData, recurrenceLimit=params.recurrenceLimit)
        columnsToGet.append('tissue')
        tissueData = tissueData.loc[:, columnsToGet]

        # Subsample the data
        tissueData = tissueData.loc[samplesToUse, :]

        if len(tissueData) > 0:
            # Get the Latest Blood Sample
            latestBlood = pd.DataFrame()
            bloodSampleIDs = _getBloodSampleInfo(patID, samplesInfo=clinData.samplesInfo, mafData=mafData)
            if len(bloodSampleIDs) > 0:
                latestBloodSampleID = bloodSampleIDs.loc[:, 'primarySampleID'].iloc[0]
                latestBlood = _getBloodSample(patID, latestBloodSampleID, mafData)
                latestBlood = latestBlood.loc[
                              [x for x in latestBlood.index if latestBlood.loc[x, 'ids'] in columnsToGet], :]

            if len(latestBlood) > 0:
                # Get Mutations Unique to blood
                uniqueBlood = _getMutationsUniqueToBlood(latestBlood=latestBlood, tissueData=tissueData)

                # Generate the hypothesis blood
                curHypothesisBlood = generateHypothesisBloodParallelFast(curHypothesisBlood=curHypothesisBlood,
                                                                       nthreads=params.nthreads, tissueData=tissueData,
                                                                       params=params, latestBlood=latestBlood,
                                                                       numHyp=params.num_hyp, clinData=clinData,
                                                                       uniqueBlood=uniqueBlood)

    return curHypothesisBlood


# Function to run per patient analysis
def runAnalysisPatient(curHypothesisBlood, params, mafData, clinData, verbose, dropNullTissue=True, tumorPurity=0.05):
    patID = curHypothesisBlood.patID

    hypothesis_dictionary = dict()
    hypothesisBloodDist = pd.DataFrame()

    existsPatObj = os.path.isfile(curHypothesisBlood.file_saveClass)

    if existsPatObj and params.mode != 'full' and params.mode != 'adapativeGrid':
        curHypothesisBlood = pickle.load(open(curHypothesisBlood.file_saveClass, 'r'))
        hypothesis_dictionary = curHypothesisBlood.hypotheses
        hypothesisBloodDist = curHypothesisBlood.distance

    if len(hypothesisBloodDist) == 0 or params.mode in ['full', 'distance', 'adapativeGrid']:
        tissueData = pd.DataFrame()
        tissueData = _getTissueData(mafData=mafData, patID=patID, clinData=clinData, dropNullTissue=dropNullTissue)

        if len(tissueData) > 0:
            # Drop The Low Purity Samples
            tissueData = _dropLowPuritySamples(tissueData=tissueData, tumorFractions=clinData.tumorFractions,
                                              thd=tumorPurity)
        if len(tissueData) > 0:
            # Get genes under the recurrence rate
            columnsToGet = getGenesUnderRecurrencenLimit(tissueData, recurrenceLimit=params.recurrenceLimit)
            columnsToGet.append('tissue')
            tissueData = tissueData.loc[:, columnsToGet]

            # Subsample the data
            if params.subSampleSize > 0:
                np.random.seed(params.runIdx)
                if params.subSampleSize > len(tissueData):
                    tissueData = tissueData.loc[np.random.choice(tissueData.index,
                                                                 len(tissueData),
                                                                 replace=False), :]
                else:
                    tissueData = tissueData.loc[np.random.choice(tissueData.index,
                                                                 params.subSampleSize,
                                                                 replace=False),
                                 :]

            if len(tissueData) > 0:
                # Get the Latest Blood Sample
                latestBlood = pd.DataFrame()
                bloodSampleIDs = _getBloodSampleInfo(patID, samplesInfo=clinData.samplesInfo, mafData=mafData)

                if len(bloodSampleIDs) > 0:
                    latestBloodSampleID = bloodSampleIDs.loc[:, 'primarySampleID'].iloc[0]
                    latestBlood = _getBloodSample(patID, latestBloodSampleID, mafData)
                    latestBlood = latestBlood.loc[
                                  [x for x in latestBlood.index if latestBlood.loc[x, 'ids'] in columnsToGet], :]

                if len(latestBlood) > 0:
                    # Get Mutations Unique to blood
                    uniqueBlood = _getMutationsUniqueToBlood(latestBlood=latestBlood, tissueData=tissueData)

                    if len(hypothesisBloodDist) == 0 or params.mode in ['full', 'distance']:
                        if verbose:
                            print
                            "$ Generating Hypothesis Blood and calculating Distances" + "\n\n"
                        # Generate the hypothesis blood

                        # Readjust the alpha grid search if the number of tissue is less than the subsample size
                        params.alphasGridSearch = pd.DataFrame(
                            list(itertools.product(*[params.discreteRange] * len(tissueData))))
                        params.num_hyp = params.alphasGridSearch.shape[0]

                        curHypothesisBlood = generateHypothesisBloodParallelFast(curHypothesisBlood=curHypothesisBlood,
                                                                               nthreads=params.nthreads,
                                                                               tissueData=tissueData,
                                                                               params=params, latestBlood=latestBlood,
                                                                               numHyp=params.num_hyp, clinData=clinData,
                                                                               uniqueBlood=uniqueBlood)

    return curHypothesisBlood


# Function to run per patient analysis
def runAnalysisPatientSlow(curHypothesisBlood, params, mafData, clinData, verbose, dropNullTissue=True):
    patID = curHypothesisBlood.patID

    HB_dictionary = dict()
    hypothesis_dictionary = dict()
    hypothesisBloodDist = pd.DataFrame()

    existsPatObj = os.path.isfile(curHypothesisBlood.file_saveClass)

    if existsPatObj and params.mode != 'full':
        curHypothesisBlood = pickle.load(open(curHypothesisBlood.file_saveClass, 'r'))
        HB_dictionary = curHypothesisBlood.sbl
        hypothesis_dictionary = curHypothesisBlood.hypotheses
        hypothesisBloodDist = curHypothesisBlood.distance

    if len(hypothesisBloodDist) == 0 or params.mode in ['full', 'distance']:
        tissueData = pd.DataFrame()
        tissueData = _getTissueData(mafData=mafData, patID=patID, clinData=clinData, dropNullTissue=dropNullTissue)

        if len(tissueData) > 0:
            # Drop The Low Purity Samples
            tissueData = _dropLowPuritySamples(tissueData=tissueData, tumorFractions=clinData.tumorFractions)

            if len(tissueData) > 0:
                # Get genes under the recurrence rate
                columnsToGet = getGenesUnderRecurrencenLimit(tissueData, recurrenceLimit=params.recurrenceLimit)
                columnsToGet.append('tissue')
                tissueData = tissueData.loc[:, columnsToGet]

                if len(tissueData) > 0:
                    # Get the Latest Blood Sample
                    latestBlood = pd.DataFrame()
                    latestBlood = _getLatestBloodSample(patID=patID, mafData=mafData, samplesInfo=clinData.samplesInfo)
                    latestBlood = latestBlood.loc[
                                  [x for x in latestBlood.index if latestBlood.loc[x, 'ids'] in columnsToGet], :]

                    if len(latestBlood) > 0:
                        # Get Mutations Unique to blood
                        uniqueBlood = _getMutationsUniqueToBlood(latestBlood=latestBlood, tissueData=tissueData)

                        if len(hypothesisBloodDist) == 0 or params.mode == "full":
                            if verbose:
                                print
                                "$ Generating Hypothesis Blood" + "\n\n"
                            # Generate the hypothesis blood

                            res = {}
                            res = generateHypothesisBloodParallel(nthreads=params.nthreads,
                                                                 num_hyp=params.num_hyp,
                                                                 tissueData=tissueData,
                                                                 highRange=params.highRange_shedding,
                                                                 lowRange=params.lowRange_shedding,
                                                                 Low=params.lowThreshold_alphaBin,
                                                                 High=params.highThreshold_alphaBin,
                                                                 uniqueBlood=uniqueBlood,
                                                                 forceMixHypotheses=params.forceMixHypotheses,
                                                                 useMediumBin=params.useMediumBin,
                                                                 weightsRandomUniform=params.weightsRandomUniform,
                                                                 weightsRandomDiscrete=params.weightsRandomDiscrete)
                            HB_dictionary = res['HB']
                            hypothesis_dictionary = res['hypothesis']

                        # Calculate the Distances
                        if len(hypothesisBloodDist) == 0 or params.mode == "full" or params.mode == "distance":
                            if verbose:
                                print
                                "$ Calculating Hypothesis Blood Distances" + "\n\n"
                            hypothesisBloodDist = calculateBloodDistParallel(nthreads=params.nthreads,
                                                                            num_hyp=params.num_hyp,
                                                                            tissueData=tissueData,
                                                                            highRange=params.highRange_shedding,
                                                                            lowRange=params.lowRange_shedding,
                                                                            Low=params.lowThreshold_alphaBin,
                                                                            High=params.highThreshold_alphaBin,
                                                                            uniqueBlood=uniqueBlood,
                                                                            params=params,
                                                                            latestBlood=latestBlood,
                                                                            HB_dictionary=HB_dictionary)
                            hypothesisBloodDist.loc[:, 'participantID'] = [re.sub("_.*", "", x) for x in
                                                                          hypothesisBloodDist['lesion']]
                            hypothesisBloodDist.loc[:, 'tissue'] = [clinData.tissueMap[x] for x in
                                                                   hypothesisBloodDist['lesion']]
                            hypothesisBloodDist.loc[:, 'tissueOriginal'] = [clinData.tissueOriginalMap[x] for x in
                                                                           hypothesisBloodDist['lesion']]

    curHypothesisBlood.distance = hypothesisBloodDist
    curHypothesisBlood.hypotheses = hypothesis_dictionary
    curHypothesisBlood.sbl = HB_dictionary
    return curHypothesisBlood


# Make the Per patient Plots
def plotPerPatient(curHypothesisBlood, params, clinData, verbose):
    patID = curHypothesisBlood.patID
    # Make Plots
    if verbose:
        print
        "$ Making Plots" + "\n\n"

    plotScoreDistributions(distance=curHypothesisBlood.distance, clinData=clinData, patID=patID,
                           outputDir=params.dir_runOutput)

    plotRealPlotPerformance(clinData, curHypothesisBlood.distance, patID, params.dir_runOutput, params.ccf_lesion_thd)

    for metric in ["Chebyshev"]:  # , "L1" ]:
        hypothesisBloodDistTable_prec = evaluateFit(curHypothesisBlood.distance, metric=metric + "_precision")
        hypothesisBloodDistTable_recall = evaluateFit(curHypothesisBlood.distance, metric=metric + "_recall")
        hypothesisBloodDistTable_prec = hypothesisBloodDistTable_prec.sort_index(axis=0)
        hypothesisBloodDistTable_recall = hypothesisBloodDistTable_recall.sort_index(axis=0)

        # Recall and Precision of bypotheses
        prec = hypothesisBloodDistTable_prec['mean']
        rec = hypothesisBloodDistTable_recall['mean']
        ax = sns.jointplot(x=prec, y=rec)
        ax.set_axis_labels(re.sub("Chebyshev", "L-infinity", metric) + ' precision',
                           re.sub("Chebyshev", "L-infinity", metric) + ' recall')
        ax.fig.suptitle(y=1.05, t="Mean " + re.sub("Chebyshev", "L-infinity",
                                                   metric) + " Distance of Recall and Precision\nof Hypothesis Blood (" + str(
            params.num_hyp) + " hypotheses) for " + clinData.patientIDMap[patID])
        ax.savefig(os.path.join(params.dir_runOutput,
                                "JointPlot_" + re.sub("Chebyshev", "L-infinity", metric) + "_" + patID + ".png"),
                   dpi=300)
        plt.close()

        # Combining the recall and precision to find best and worst hypotheses
        comb = prec + rec
        comb = comb.sort_values(axis=0)
        ax = sns.distplot(comb)
        ax.set(xlabel=re.sub("Chebyshev", "L-infinity", metric) + ' Precision + Recall',
               title="Sum of Mean " + re.sub("Chebyshev", "L-infinity",
                                             metric) + " Distance of Recall Precision\nof Hypothesis Blood (" + str(
                   params.num_hyp) + " hypotheses) for " + clinData.patientIDMap[patID])
        plt.savefig(os.path.join(params.dir_runOutput, "DistPlot_Combined_" + re.sub("Chebyshev", "L-infinity",
                                                                                     metric) + "_" + patID + ".png"),
                    dpi=300)
        plt.close()

        # Plot the weights as line plots
        plotWeightsLines(scores=prec, clinData=clinData, distance=curHypothesisBlood.distance,
                         hypotheses=curHypothesisBlood.hypotheses, numToGet=20, patID=patID,
                         outputDir=params.dir_runOutput,
                         numHyp=params.num_hyp, scoreName=re.sub("Chebyshev", "L-infinity", metric) + "_" + "precision")

        # Plot the Top and Bottom Hypotheses weight

        for numToGet in [5, 20, int(0.005 * params.num_hyp)]:  # , int( 0.01*params.num_hyp ) ]:
            #            hypothValueRandom = getHypothRandomMeanWeights( numToGet = numToGet, distance = curHypothesisBlood.distance, hypotheses = curHypothesisBlood.hypotheses, clinData = clinData, numHyp = params.num_hyp )
            plotHypothWeights(scores=prec, patID=patID,
                              scoreName=re.sub("Chebyshev", "L-infinity", metric) + "_" + "precision",
                              numToGet=numToGet, outputDir=params.dir_runOutput, curHypothesisBlood=curHypothesisBlood,
                              clinData=clinData, numHyp=params.num_hyp)
    #            plotHypothWeights( scores = rec, patID = patID, scoreName = re.sub( "Chebyshev", "L-infinity", metric ) + "_" + "recall",  numToGet = numToGet, outputDir = params.dir_runOutput, curHypothesisBlood = curHypothesisBlood, clinData = clinData, numHyp = params.num_hyp )
    #            plotHypothWeights( scores = comb, patID = patID, scoreName = re.sub( "Chebyshev", "L-infinity", metric ) + "_" + "sumPrecisionRecall",  numToGet = numToGet, outputDir = params.dir_runOutput, curHypothesisBlood = curHypothesisBlood, clinData = clinData, numHyp = params.num_hyp )

    # Plot the heatmap of all weights
    if False:
        plotWeightsHeatmap(clinData=clinData,
                           distance=curHypothesisBlood.distance,
                           hypotheses=curHypothesisBlood.hypotheses,
                           numHyp=params.num_hyp,
                           patID=patID,
                           outputDir=params.dir_runOutput)


# Plot the score distributions
def plotScoreDistributions(distance, clinData, patID, outputDir):
    plt.figure(figsize=(10, 10))
    idx = 1
    for metric in ["Chebyshev"]:  # , "L1" ]:
        hypothesisBloodDistTable_prec = evaluateFit(distance, metric=metric + "_precision")
        hypothesisBloodDistTable_recall = evaluateFit(distance, metric=metric + "_recall")

        score = hypothesisBloodDistTable_prec['mean']
        score = score.tolist()
        score.sort()
        zscores = stats.zscore(score)
        plt.subplot(2, 2, idx)
        ax = sns.distplot(score)
        pos = 20
        y = ax.get_ylim()[1] * .5
        plt.arrow(x=score[pos], y=y, dx=0, dy=y * -.85, color='red')
        plt.text(x=score[pos], y=y, s=str(pos) + ":" + str(round(zscores[pos], 2)), color='red')

        pos = len(score) - 10
        y = ax.get_ylim()[1] * .5
        plt.arrow(x=score[pos], y=y, dx=0, dy=y * -.85, color='red')
        plt.text(x=score[pos], y=y, s=str(pos) + ":" + str(round(zscores[pos], 2)), color='red')
        ax.set(xlabel="score",
               ylabel="Frequency",
               title=clinData.patientIDMap[patID] + ": " + re.sub("Chebyshev", "L-infinity", metric) + "_precision")
        plt.tight_layout()
        idx = idx + 1

        score = hypothesisBloodDistTable_recall['mean']
        score = score.tolist()
        score.sort()
        zscores = stats.zscore(score)
        plt.subplot(2, 2, idx)
        ax = sns.distplot(score)
        pos = 20
        y = ax.get_ylim()[1] * .5
        plt.arrow(x=score[pos], y=y, dx=0, dy=y * -.85, color='red')
        plt.text(x=score[pos], y=y, s=str(pos) + ":" + str(round(zscores[pos], 2)), color='red')

        pos = len(score) - 10
        y = ax.get_ylim()[1] * .5
        plt.arrow(x=score[pos], y=y, dx=0, dy=y * -.85, color='red')
        plt.text(x=score[pos], y=y, s=str(pos) + ":" + str(round(zscores[pos], 2)), color='red')
        ax.set(xlabel="score",
               ylabel="Frequency",
               title=clinData.patientIDMap[patID] + ": " + re.sub("Chebyshev", "L-infinity", metric) + "_recall")
        plt.tight_layout()
        idx = idx + 1
    plt.savefig(os.path.join(outputDir, "DistPlot_Scores_" + patID + ".png"), dpi=300)
    # plt.show()
    plt.close()


# Def plot weights as line plot
def plotWeightsLines(scores, clinData, distance, hypotheses, numToGet, patID, outputDir, numHyp, scoreName):
    scores = scores.sort_values(axis=0, ascending=True)
    numToGet = max(numToGet, 10)
    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
    hypothValueBest = getHypothValues(hypoth=hypothBest, distance=distance, hypotheses=hypotheses, clinData=clinData)
    hypothValueBest = hypothValueBest.pivot("HB", 'primarySampleID', "weight")
    for x in hypothValueBest.index:
        plt.plot(range(0, hypothValueBest.shape[1]), hypothValueBest.loc[x, :])
    plt.xticks(range(0, hypothValueBest.shape[1]), [clinData.sampleIDMap[x] for x in hypothValueBest.columns],
               rotation=90)
    plt.xlabel("Lesion")
    plt.ylabel("Weight")
    plt.title(
        "Weights of Top " + str(numToGet) + " HB (" + str(numHyp) + ") " + scoreName + " : " + clinData.patientIDMap[
            patID])
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, "Top" + str(numToGet) + "HBWeights_" + scoreName + "_" + patID + ".png"),
                dpi=300)
    plt.close()


# Def plot weights as line plot
def plotRealPlotPerformance(clinData, hypothesisBloodDist, patID, outputDir, ccf_lesion_thd):
    hypothesisBloodDist_real = hypothesisBloodDist[['lesion', 'yreal_precision', 'yreal_recall']]
    hypothesisBloodDist_real = hypothesisBloodDist_real.drop_duplicates()
    hypothesisBloodDist_real.loc[:, 'primarySampleID'] = [clinData.sampleIDMap[x] for x in
                                                         hypothesisBloodDist_real['lesion']]

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    for y in hypothesisBloodDist_real['yreal_precision']:
        y = [float(i) for i in y.split(",")]
        plt.plot(np.arange(0.05, 1.0, 0.05), y)
    plt.vlines(ccf_lesion_thd, 0.6, 1)
    plt.legend(hypothesisBloodDist_real['primarySampleID'], ncol=5, loc="upper center", bbox_to_anchor=(1, 1.2))
    plt.xlabel("CCF Threshold")
    plt.ylabel("Precision")

    plt.subplot(1, 2, 2)
    for y in hypothesisBloodDist_real['yreal_recall']:
        y = [float(i) for i in y.split(",")]
        plt.plot(np.arange(0.05, 1.0, 0.05), y)
    plt.xlabel("CCF Threshold")
    plt.ylabel("Recall")
    plt.vlines(ccf_lesion_thd, 0.6, 1)
    plt.suptitle("Precision and Recall of real blood " + clinData.patientIDMap[patID], y=1.25)
    # plt.tight_layout()
    plt.savefig(os.path.join(outputDir, "RealBloodRecallPrecision_" + patID + ".png"), dpi=150)
    plt.show()
    plt.close()


def plotWeightsHeatmap(clinData, distance, hypotheses, numHyp, patID, outputDir):
    hypothWeights = getHypothValues(hypoth=range(0, numHyp),
                                    distance=distance,
                                    hypotheses=hypotheses,
                                    clinData=clinData)

    hypothWeights = hypothWeights.pivot("HB", 'primarySampleID', "weight")
    g = sns.clustermap(hypothWeights, method="ward", metric="euclidean",
                       col_colors=[clinData.tissueColor[clinData.tissueMap[x]] for x in hypothWeights.columns],
                       xticklabels=[clinData.sampleIDMap[x] for x in hypothWeights.columns],
                       yticklabels=False)
    g.fig.suptitle("All Hypotheses (n=" + str(numHyp) + "): " + clinData.patientIDMap[patID] + "\nWard - Euclidean")
    plt.savefig(os.path.join(outputDir, "HypothWeightCluster_" + patID + ".png"), dpi=300)
    plt.show()
    plt.close()


# Calculate the distance between samples based on mutation similarity
def calcSampleSimilarity(patID, mafData, clinData, params, outputDir, threshold, plot=True, dropNullTissue=True):
    tissueData = _getTissueData(mafData=mafData, patID=patID, clinData=clinData, dropNullTissue=dropNullTissue)
    tissueData = _dropLowPuritySamples(tissueData=tissueData, tumorFractions=clinData.tumorFractions)
    tissueData_distance = pd.DataFrame()
    if len(tissueData) > 0:
        tissueData_distance = getSampleDistance(tissueData=tissueData, threshold=threshold, metric='jaccard')
        if plot:
            plotDistanceHeatmap(clinData=clinData, distance=tissueData_distance, patID=patID, outputDir=outputDir,
                                metric='jaccard', threshold=threshold)
    return tissueData_distance


# Get the Distances between samples
def getSampleDistance(tissueData, threshold, metric):
    tissueData = tissueData.sort_values(by=['tissue', 'primarySampleID'])
    tissueData = tissueData.drop('tissue', axis=1)
    tissueData = pd.DataFrame(binarize(tissueData, threshold=threshold), index=tissueData.index,
                              columns=tissueData.columns)
    tissueData_distance = pd.DataFrame(
        pairwise_distances(X=tissueData.values, Y=tissueData.values, metric=metric.lower()), index=tissueData.index,
        columns=tissueData.index)
    return tissueData_distance


# Select a random set of samples based on their tissue similarity given the parameterized set number of samples
#   When clustering samples then cut the tree and grab a random representative sample
def selectRandomSampleBySimilarity(tissueData, params, threshold, metric='jaccard'):
    tissueData_distance = getSampleDistance(tissueData=tissueData, threshold=threshold, metric=metric)
    zLinkage = hierarchy.ward(tissueData_distance)
    if params.subSampleSize < 1:
        cutMembership = hierarchy.cut_tree(zLinkage, min(len(tissueData_distance),
                                                         max(int(np.floor(len(tissueData) * params.subSampleSize)), 5)))
    else:
        cutMembership = hierarchy.cut_tree(zLinkage, min(len(tissueData_distance), max(params.subSampleSize, 5)))
    clusts = {}
    for c in range(0, len(cutMembership)):
        if str(cutMembership[c][0]) in clusts.keys():
            clusts[str(cutMembership[c][0])].append(c)
        else:
            clusts[str(cutMembership[c][0])] = [c]
    selectSamples = []
    for clust in clusts.values():
        selectSamples.append(tissueData_distance.index[np.random.choice(clust, 1)][0])
    return (selectSamples)


def plotDistanceHeatmap(clinData, distance, patID, outputDir, metric, threshold):
    linkage = hc.linkage(sp.distance.squareform(distance), method='ward')
    g = sns.clustermap(distance, row_linkage=linkage, col_linkage=linkage,
                       col_colors=[clinData.tissueColor[clinData.tissueMap[x]] for x in distance.columns],
                       cmap=sns.color_palette("Blues"),
                       vmin=0, vmax=1,
                       xticklabels=[clinData.sampleIDMap[x] for x in distance.columns],
                       yticklabels=[clinData.sampleIDMap[x] for x in distance.index],
                       figsize=(10, 10))
    g.fig.suptitle(
        "Distance " + metric + ", ccfThreshold=" + str(threshold) + "matrix : " + clinData.patientIDMap[patID])
    g.fig.subplots_adjust(right=.8, top=.95, bottom=.2)
    plt.savefig(
        os.path.join(outputDir, "DistanceMatrix_ccfThreshold" + str(threshold) + "_" + metric + "_" + patID + ".png"),
        dpi=300)
    plt.show()
    plt.close()


# Plot Mutations Per lesion
def plotMutationsPerSample(cnt, patID, clinData, outputDir, threshold):
    cnt = pd.DataFrame({'id': cnt.keys(),
                        'Load': cnt.values()})
    cnt.loc[:, 'tissue'] = [clinData.tissueMap[x] for x in cnt['id']]
    cnt.loc[:, 'Sample ID'] = [clinData.sampleIDMap[x] for x in cnt['id']]
    cnt = cnt.sort_values(by=['tissue', 'Sample ID'])

    ax = sns.barplot(x='Sample ID', y='Load',
                     hue='tissue', data=cnt,
                     palette=clinData.tissueColor)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.legend(loc="upper center", borderaxespad=0.1, ncol=6, bbox_to_anchor=(1, 1.4))
    ax.set_title("Mutational Load of " + clinData.patientIDMap[patID] + " CCF Threshold = " + str(threshold))
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, "MutationsPerSample_" + patID + ".png"), dpi=300)
    plt.show()
    plt.close()


# Run all patients in a run
def runSinglePatientMultiBlood(patID, params, runIndexes, clinData, mafData, verbose, dropNullTissue=True):
    latestBlood = pd.DataFrame()
    bloodSampleIDs = _getBloodSampleInfo(patID, samplesInfo=clinData.samplesInfo, mafData=mafData)
    patientsToAnalyze = [patID]

    for bloodSampleID in bloodSampleIDs['primarySampleID']:
        for runIdx in runIndexes:
            params.setRunIdx(runIdx)
            # Create Output Directory and write parameters to file
            if (not os.path.isdir(params.dir_runOutput)):
                os.mkdir(params.dir_runOutput)
            params.dir_runOutput = os.path.join(params.dir_runOutput, bloodSampleID)
            if (not os.path.isdir(params.dir_runOutput)):
                os.mkdir(params.dir_runOutput)

            res = {}
            hypothesis_dictionary_ALL = {}
            hypothesisBloodDist_ALL = pd.DataFrame()

            existsDistAll = os.path.isfile(os.path.join(params.dir_runOutput, "hypothesisBloodDist-ALLPatients.pkl"))
            existsDistAll = False
            if params.mode in ["plotonly", "plotnewonly", "calconly"] and existsDistAll:
                hypothesisBloodDist_ALL = pickle.load(
                    open(os.path.join(params.dir_runOutput, "hypothesisBloodDist-ALLPatients.pkl"), "rb"))
                hypothesis_dictionary_ALL = pickle.load(
                    open(os.path.join(params.dir_runOutput, "hypothesisBloodHypoth-ALLPatients.pkl"), "rb"))

            # Main Loop to do all the work
            for p in range(0, len(patientsToAnalyze)):
                patID = patientsToAnalyze[p]
                if verbose:
                    print
                    "$ Working on patient " + patID + "\n\n"

                curHypothesisBlood = hypothesisBlood(params, patID)

                if params.mode == "calconly" or len(curHypothesisBlood.distance) == 0:
                    patID = curHypothesisBlood.patID
                    hypothesisBloodDist = pd.DataFrame()

                    if (len(hypothesisBloodDist) == 0) or (params.mode in ['full', 'distance', 'adapativeGrid']):
                        tissueData = pd.DataFrame()
                        tissueData = _getTissueData(mafData=mafData, patID=patID, clinData=clinData,
                                                   dropNullTissue=dropNullTissue)
                        tissueData = tissueData.loc[
                                     [x for x in tissueData.index if 'WGS' not in clinData.sampleIDMap[x]], :]
                        tissueData = tissueData.drop(tissueData.index[(tissueData['tissue'] == 0)], axis=0)

                        if len(tissueData) > 0:
                            # Get genes under the recurrence rate
                            columnsToGet = getGenesUnderRecurrencenLimit(tissueData,
                                                                         recurrenceLimit=params.recurrenceLimit)
                            columnsToGet.append('tissue')
                            tissueData = tissueData.loc[:, columnsToGet]

                            # Get the Latest Blood Sample
                            latestBlood = _getBloodSample(patID, bloodSampleID, mafData)
                            uniqueBlood = _getMutationsUniqueToBlood(latestBlood=latestBlood, tissueData=tissueData)
                            if len(hypothesisBloodDist) == 0 or params.mode in ['full', 'distance']:
                                if verbose:
                                    print
                                    "$ Generating Hypothesis Blood and calculating Distances" + "\n\n"
                                # Generate the hypothesis blood
                                curHypothesisBlood = generateHypothesisBloodParallelFast(
                                    curHypothesisBlood=curHypothesisBlood,
                                    nthreads=params.nthreads,
                                    tissueData=tissueData.loc[np.random.choice(tissueData.index, min(len(tissueData),
                                                                                                     params.subSampleSize),
                                                                               replace=False), :],
                                    params=params, latestBlood=latestBlood, numHyp=params.num_hyp, clinData=clinData,
                                    uniqueBlood=uniqueBlood)

                if len(curHypothesisBlood.distance) > 0:
                    existPatientObj = os.path.isfile(curHypothesisBlood.file_saveClass)
                    existPatientObj = False
                    if params.mode not in ["plotonly", "plotnewonly",
                                           "calcOnly"] or existPatientObj == False or existsDistAll == False:
                        pickle.dump(curHypothesisBlood, open( curHypothesisBlood.file_saveClass,"wb"))
                        #curHypothesisBlood.writeHypothesisBlood()
                        hypothesis_dictionary_ALL.update(curHypothesisBlood.hypotheses)
                        hypothesisBloodDist_ALL = pd.concat( [hypothesisBloodDist_ALL, curHypothesisBlood.distance])

                    # Make the patient plots
                    existsPlot = os.path.isfile(
                        os.path.join(params.dir_runOutput, "JointPlot_L-infinity_" + patID + ".png"))

                    if params.mode != "plotnewonly" or not existsPlot:
                        if params.mode != "calconly":
                            plotPerPatient(curHypothesisBlood, params, clinData, verbose)

            # Save compiled objects
            hypothesisBloodDist_ALL.loc[:, 'participantID'] = [re.sub("_.*", "", x) for x in
                                                              hypothesisBloodDist_ALL['lesion']]
            if not existsDistAll or params.mode in ['full', 'distance', 'calconly']:
                if verbose:
                    print
                    "$ Saving the compiled objects"
                pickle.dump(hypothesisBloodDist_ALL,
                            open(os.path.join(params.dir_runOutput, "hypothesisBloodDist-ALLPatients.pkl"), "wb"))
                # hypothesisBloodDist_ALL.to_csv( os.path.join( params.dir_runOutput, "hypothesisBloodDist-ALLPatients.csv" ) )
                pickle.dump(hypothesis_dictionary_ALL,
                            open(os.path.join(params.dir_runOutput, "hypothesisBloodHypoth-ALLPatients.pkl"), "wb"))

            res['hypotheses'] = hypothesis_dictionary_ALL
            res['distance'] = hypothesisBloodDist_ALL
    return bloodSampleIDs

    # Run all patients in a run


def runAllPatients(patientsToAnalyze, params, clinData, mafData, verbose, useSlow=False, dropNullTissue=True,
                   tumorPurity=0.05):
    if verbose:
        print
        "$ Patients to be analyzed: " + "\n".join(patientsToAnalyze) + "\n\n"

        # Objects to hold all hypotheses and distances
        res = {}
        hypothesis_dictionary_ALL = {}
        hypothesisBloodDist_ALL = pd.DataFrame()

        existsDistAll = os.path.isfile(os.path.join(params.dir_runOutput, "hypothesisBloodDist-ALLPatients.pkl"))
        if params.mode in ["plotonly", "plotnewonly", "calconly"] and existsDistAll:
            hypothesisBloodDist_ALL = pickle.load(
                open(os.path.join(params.dir_runOutput, "hypothesisBloodDist-ALLPatients.pkl"), "rb"))
            hypothesis_dictionary_ALL = pickle.load(
                open(os.path.join(params.dir_runOutput, "hypothesisBloodHypoth-ALLPatients.pkl"), "rb"))

    # Main Loop to do all the work
    for p in range(0, len(patientsToAnalyze)):
        patID = patientsToAnalyze[p]
        if verbose:
            print
            "$ Working on patient " + patID + "\n\n"

        curHypothesisBlood = hypothesisBlood(params, patID)
        if params.mode != "full" and existsDistAll:
            curHypothesisBlood.distance = hypothesisBloodDist_ALL[hypothesisBloodDist_ALL['participantID'] == patID]
        #            curHypothesisBlood.hypotheses = hypothesis_dictionary_ALL[ hypothesis_dictionary_ALL['participantID'] == patID ]
        if params.mode == "full" or len(curHypothesisBlood.distance) == 0:
            if useSlow:
                curHypothesisBlood = runAnalysisPatientSlow(curHypothesisBlood, params, mafData, clinData, verbose,
                                                           dropNullTissue=dropNullTissue)
            else:
                curHypothesisBlood = runAnalysisPatient(curHypothesisBlood, params, mafData, clinData, verbose,
                                                       dropNullTissue=dropNullTissue, tumorPurity=tumorPurity)

        if len(curHypothesisBlood.distance) > 0:
            existPatientObj = os.path.isfile(curHypothesisBlood.file_saveClass)
            if params.mode not in ["plotonly", "plotnewonly",
                                   "calconly"] or existPatientObj == False or existsDistAll == False:
                pickle.dump(curHypothesisBlood, open(curHypothesisBlood.file_saveClass, "wb"))
                # curHypothesisBlood.writeHypothesisBlood()
                hypothesis_dictionary_ALL.update(curHypothesisBlood.hypotheses)
                hypothesisBloodDist_ALL = pd.concat( [hypothesisBloodDist_ALL,curHypothesisBlood.distance ])

            # Make the patient plots
            existsPlot = os.path.isfile(os.path.join(params.dir_runOutput, "JointPlot_L-infinity_" + patID + ".png"))

            if params.mode != "plotnewonly" or not existsPlot:
                if params.mode != "calconly":
                    plotPerPatient(curHypothesisBlood, params, clinData, verbose)

    # Save compiled objects
    hypothesisBloodDist_ALL.loc[:, 'participantID'] = [re.sub("_.*", "", x) for x in hypothesisBloodDist_ALL['lesion']]
    if not existsDistAll or params.mode in ['full', 'distance', 'calconly']:
        if verbose:
            print
            "$ Saving the compiled objects"
        pickle.dump(hypothesisBloodDist_ALL,
                    open(os.path.join(params.dir_runOutput, "hypothesisBloodDist-ALLPatients.pkl"), "wb"))
        # hypothesisBloodDist_ALL.to_csv( os.path.join( params.dir_runOutput, "hypothesisBloodDist-ALLPatients.csv" ) )
        pickle.dump(hypothesis_dictionary_ALL,
                    open(os.path.join(params.dir_runOutput, "hypothesisBloodHypoth-ALLPatients.pkl"), "wb"))

    res['hypotheses'] = hypothesis_dictionary_ALL
    res['distance'] = hypothesisBloodDist_ALL
    return res


# Compare for a patient across runs the lesion ordering
def compareCrossRunSampleOrdering(patID, runIndexes, metric, numToGet, clinData, params, medianWeights, sigCnt,
                                  tie=False):
    if not os.path.isdir(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering")):
        os.mkdir(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering"))

    if medianWeights is None:
        medianWeights = pd.DataFrame()
        curRunIndexes = []
    else:
        curRunIndexes = medianWeights.index.to_list()

    allWeights = {}
    for runIdx in runIndexes:
        if runIdx not in curRunIndexes and os.path.isdir(os.path.join(params.dir_output, "Run" + str(runIdx))):
            params.setRunIdx(runIdx)
            if os.path.isfile(os.path.join(params.dir_runOutput, "hypothesisBloodClass-" + patID + ".pkl")):
                curRunIndexes.append(runIdx)
                f = open(os.path.join(params.dir_runOutput, "hypothesisBloodClass-" + patID + ".pkl"), "rb")
                gc.disable()
                curHypothesisBlood = pickle.load(f)
                gc.enable()
                hypothesisBloodDist = curHypothesisBlood.distance
                hypothesisBloodDistTable_prec = evaluateFit(hypothesisBloodDist, metric=metric + "_precision")
                hypothesisBloodDistTable_prec = hypothesisBloodDistTable_prec.sort_index(axis=0)
                prec = hypothesisBloodDistTable_prec['mean']
                scores = prec
                scores = scores.sort_values(axis=0, ascending=True)
                if tie == True:
                    scores = scores[scores == max(scores)]
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index]
                else:
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]

                hypothValueBest = getHypothValues(hypoth=hypothBest, distance=hypothesisBloodDist,
                                                  hypotheses={}, clinData=clinData)
                weights = getHypothWeight(hypothValue=hypothValueBest, useWeight=True)
                allWeights[runIdx] = weights
                medianWeights = medianWeights.append(pd.DataFrame(weights.median(axis=0)).transpose(), sort=True)

    medianWeights.index = curRunIndexes

    if sigCnt is None:
        sigCnt = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
    for runIdx in curRunIndexes:
        if runIdx in allWeights.keys():
            weights = allWeights[runIdx]
            for i in range(0, weights.shape[1] - 1):
                for j in range(i + 1, weights.shape[1]):
                    sig = stats.ttest_ind(a=weights.iloc[:, i], b=weights.iloc[:, j], equal_var=False)
                    if sig[1] <= 0.05:
                        if sig[0] > 0:
                            sigCnt.loc[weights.columns[i], weights.columns[j]] = sigCnt.loc[weights.columns[i],
                                                                                            weights.columns[j]] + 1
                        elif sig[0] < 0:
                            sigCnt.loc[weights.columns[j], weights.columns[i]] = sigCnt.loc[weights.columns[j],
                                                                                            weights.columns[i]] + 1

    return (medianWeights, sigCnt)


def getCrossRunSaveFile(params, numToGet, runIndexes):
    return os.path.join(params.dir_output,
                        "CrossComparisonWithinHB_Top" + str(numToGet) + "HB_SubSample" + str(
                            runIndexes[0]) + "-" + str(runIndexes[-1]) + ".pkl")


def compareCrossRunDriver(params, patientsToAnalyze, runIndexes, numToGet, clinData,
                          metric="Chebyshev", crossRunComparison=dict(), tie=True, savePickle = True):
    crossRunSaveFile = getCrossRunSaveFile(params, numToGet, runIndexes)
    if not os.path.isdir(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering")):
        os.mkdir(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering"))

    for patID in patientsToAnalyze:
        if patID in crossRunComparison.keys():
            medianWeights = crossRunComparison[patID][0]
            sigCnt = crossRunComparison[patID][1]
            numTopHB = crossRunComparison[patID][2]
        else:
            medianWeights = None
            sigCnt = None
            numTopHB = None

        crossRunComparison[patID] = compareCrossRunSampleOrderingWithinHB(patID,
                                                                          runIndexes,
                                                                          metric,
                                                                          numToGet,
                                                                          clinData,
                                                                          params,
                                                                          medianWeights,
                                                                          sigCnt,
                                                                          numTopHB,
                                                                          tie=tie)
    if savePickle:
        pickle.dump(crossRunComparison, open(crossRunSaveFile, "wb"))
    return crossRunComparison


def compareCrossRunSampleOrderingWithinHB(patID, runIndexes, metric, numToGet, clinData, params, medianWeights, sigCnt,
                                          numTopHB, tie=False):
    if medianWeights is None:
        medianWeights = pd.DataFrame()
        curRunIndexes = []
    else:
        curRunIndexes = medianWeights.index.to_list()

    allWeights = {}
    for runIdx in runIndexes:
        if runIdx not in curRunIndexes and os.path.isdir(os.path.join(params.dir_output, "Run" + str(runIdx))):
            params.setRunIdx(runIdx)
            if os.path.isfile(os.path.join(params.dir_runOutput, "hypothesisBloodClass-" + patID + ".pkl")):
                curRunIndexes.append(runIdx)
                f = open(os.path.join(params.dir_runOutput, "hypothesisBloodClass-" + patID + ".pkl"), "rb")
                gc.disable()
                curHypothesisBlood = pickle.load(f)
                gc.enable()
                hypothesisBloodDist = curHypothesisBlood.distance
                hypothesisBloodDistTable_prec = evaluateFit(hypothesisBloodDist, metric=metric + "_precision")
                hypothesisBloodDistTable_prec = hypothesisBloodDistTable_prec.sort_index(axis=0)
                prec = hypothesisBloodDistTable_prec['mean']
                scores = prec
                scores = scores.sort_values(axis=0, ascending=True)
                if tie == True:
                    scores = scores[scores == min(scores)]
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index]
                else:
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
                hypothValueBest = getHypothValues(hypoth=hypothBest, distance=hypothesisBloodDist,
                                                  hypotheses={}, clinData=clinData)
                weights = getHypothWeight(hypothValue=hypothValueBest, useWeight=True)
                allWeights[runIdx] = weights
                medianWeights = pd.concat( [medianWeights, pd.DataFrame(weights.median(axis=0)).transpose() ], sort=True)

    medianWeights.index = curRunIndexes

    if sigCnt is None:
        sigCnt = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
        numTopHB = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
    for runIdx in curRunIndexes:
        if runIdx in allWeights.keys():
            weights = allWeights[runIdx]
            for i in range(0, weights.shape[1] - 1):
                for j in range(i + 1, weights.shape[1]):
                    a = weights.iloc[:, i]
                    b = weights.iloc[:, j]
                    sigCnt.loc[weights.columns[i], weights.columns[j]] = sigCnt.loc[weights.columns[i], weights.columns[
                        j]] + sum(a > b)
                    sigCnt.loc[weights.columns[j], weights.columns[i]] = sigCnt.loc[weights.columns[j], weights.columns[
                        i]] + sum(a < b)
                    numTopHB.loc[weights.columns[i], weights.columns[j]] = numTopHB.loc[
                                                                               weights.columns[i], weights.columns[
                                                                                   j]] + len(weights)
                    numTopHB.loc[weights.columns[j], weights.columns[i]] = numTopHB.loc[
                                                                               weights.columns[j], weights.columns[
                                                                                   i]] + len(weights)

    return (medianWeights, sigCnt, numTopHB, allWeights)


def compareCrossRunSampleOrderingWithinHB_timecourse(bloodSampleID, patID, runIndexes, metric, numToGet, clinData,
                                                     params, medianWeights, sigCnt,
                                                     numTopHB, tie=False):
    if medianWeights is None:
        medianWeights = pd.DataFrame()
        curRunIndexes = []
    else:
        curRunIndexes = medianWeights.index.to_list()

    allWeights = {}
    for runIdx in runIndexes:
        if runIdx not in curRunIndexes and os.path.isdir(os.path.join(params.dir_output, "Run" + str(runIdx))):
            params.setRunIdx(runIdx)
            if os.path.isfile(
                    os.path.join(params.dir_runOutput, bloodSampleID, "hypothesisBloodClass-" + patID + ".pkl")):
                curRunIndexes.append(runIdx)
                f = open(os.path.join(params.dir_runOutput, bloodSampleID, "hypothesisBloodClass-" + patID + ".pkl"),
                         "rb")
                gc.disable()
                curHypothesisBlood = pickle.load(f)
                gc.enable()
                hypothesisBloodDist = curHypothesisBlood.distance
                hypothesisBloodDistTable_prec = evaluateFit(hypothesisBloodDist, metric=metric + "_precision")
                hypothesisBloodDistTable_prec = hypothesisBloodDistTable_prec.sort_index(axis=0)
                prec = hypothesisBloodDistTable_prec['mean']
                scores = prec
                scores = scores.sort_values(axis=0, ascending=True)
                if tie == True:
                    scores = scores[scores == min(scores)]
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index]
                else:
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
                hypothValueBest = getHypothValues(hypoth=hypothBest, distance=hypothesisBloodDist,
                                                  hypotheses={}, clinData=clinData)
                weights = getHypothWeight(hypothValue=hypothValueBest, useWeight=True)
                allWeights[runIdx] = weights
                medianWeights = pd.concat( [medianWeights, pd.DataFrame(weights.median(axis=0)).transpose() ], sort=True)

    medianWeights.index = curRunIndexes

    if sigCnt is None:
        sigCnt = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
        numTopHB = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
    for runIdx in curRunIndexes:
        if runIdx in allWeights.keys():
            weights = allWeights[runIdx]
            for i in range(0, weights.shape[1] - 1):
                for j in range(i + 1, weights.shape[1]):
                    a = weights.iloc[:, i]
                    b = weights.iloc[:, j]
                    sigCnt.loc[weights.columns[i], weights.columns[j]] = sigCnt.loc[weights.columns[i], weights.columns[
                        j]] + sum(a > b)
                    sigCnt.loc[weights.columns[j], weights.columns[i]] = sigCnt.loc[weights.columns[j], weights.columns[
                        i]] + sum(a < b)
                    numTopHB.loc[weights.columns[i], weights.columns[j]] = numTopHB.loc[
                                                                               weights.columns[i], weights.columns[
                                                                                   j]] + len(weights)
                    numTopHB.loc[weights.columns[j], weights.columns[i]] = numTopHB.loc[
                                                                               weights.columns[j], weights.columns[
                                                                                   i]] + len(weights)

    return (medianWeights, sigCnt, numTopHB, allWeights)

def compareCrossRunSampleOrderingWithinHB_leaveOuts(leaveOut, patID, runIndexes, metric, numToGet, clinData,
                                                     params, medianWeights, sigCnt,
                                                     numTopHB, tie=False):
    if medianWeights is None:
        medianWeights = pd.DataFrame()
        curRunIndexes = []
    else:
        curRunIndexes = medianWeights.index.to_list()

    allScores = {}
    allWeights = {}
    for runIdx in runIndexes:
        if runIdx not in curRunIndexes and os.path.isdir(os.path.join(params.dir_output, "Run" + str(runIdx))):
            params.setRunIdx(runIdx)
            if os.path.isfile(
                    os.path.join(params.dir_runOutput, leaveOut, "hypothesisBloodClass-" + patID + ".pkl")):
                curRunIndexes.append(runIdx)
                f = open(os.path.join(params.dir_runOutput, leaveOut, "hypothesisBloodClass-" + patID + ".pkl"),
                         "rb")
                gc.disable()
                curHypothesisBlood = pickle.load(f)
                gc.enable()
                hypothesisBloodDist = curHypothesisBlood.distance
                hypothesisBloodDistTable_prec = evaluateFit(hypothesisBloodDist, metric=metric + "_precision")
                hypothesisBloodDistTable_prec = hypothesisBloodDistTable_prec.sort_index(axis=0)
                prec = hypothesisBloodDistTable_prec['mean']
                scores = prec
                scores = scores.sort_values(axis=0, ascending=True)
                if tie == True:
                    scores = scores[scores == min(scores)]
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index]
                else:
                    hypothBest = [int(re.sub("^.*_", "", x)) for x in scores.index[0:numToGet]]
                hypothValueBest = getHypothValues(hypoth=hypothBest, distance=hypothesisBloodDist,
                                                  hypotheses={}, clinData=clinData)
                weights = getHypothWeight(hypothValue=hypothValueBest, useWeight=True)
                allScores[runIdx] = scores
                allWeights[runIdx] = weights
                medianWeights = medianWeights.append(pd.DataFrame(weights.median(axis=0)).transpose(), sort=True)

    medianWeights.index = curRunIndexes

    if sigCnt is None:
        sigCnt = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
        numTopHB = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
    for runIdx in curRunIndexes:
        if runIdx in allWeights.keys():
            weights = allWeights[runIdx]
            for i in range(0, weights.shape[1] - 1):
                for j in range(i + 1, weights.shape[1]):
                    a = weights.iloc[:, i]
                    b = weights.iloc[:, j]
                    sigCnt.loc[weights.columns[i], weights.columns[j]] = sigCnt.loc[weights.columns[i], weights.columns[
                        j]] + sum(a > b)
                    sigCnt.loc[weights.columns[j], weights.columns[i]] = sigCnt.loc[weights.columns[j], weights.columns[
                        i]] + sum(a < b)
                    numTopHB.loc[weights.columns[i], weights.columns[j]] = numTopHB.loc[
                                                                               weights.columns[i], weights.columns[
                                                                                   j]] + len(weights)
                    numTopHB.loc[weights.columns[j], weights.columns[i]] = numTopHB.loc[
                                                                               weights.columns[j], weights.columns[
                                                                                   i]] + len(weights)

    return (medianWeights, sigCnt, numTopHB, allWeights, allScores)


def getDiscreteGridRangeAroundWeights(weights, stepNum=8):
    discreteRange = []
    for c in weights.columns:
        if len(weights) == 1:
            low = max(0.005, weights[c][0] - 0.1)
            high = min(1.2, weights[c][0] + 0.1)
        else:
            low = max(0.005, min(weights[c]) - 0.1)
            high = min(1.2, max(weights[c]) + 0.1)
        discreteRange.append(np.arange(low, high, (high - low) / stepNum))
    return pd.DataFrame(list(itertools.product(*discreteRange)))


# Plot all the networks for all patients in set
def plotAllCrossRunSampleOrdering(crossRunComparison, clinData, params, runIndexes, numToGet, edgeThresholds=[0.67]):
    for edgeThres in edgeThresholds:
        for patID in crossRunComparison.keys():
            # numToGet = len( crossRunComparison[patID][2] )
            G = plotCrossRunSampleOrdering(crossRunComparison[patID][0],
                                           crossRunComparison[patID][1],
                                           crossRunComparison[patID][2],
                                           clinData,
                                           params,
                                           patID,
                                           # numToGet = numToGet,
                                           layout="hierarchcial",
                                           edgeThres=edgeThres,
                                           filename="ConsensusDirectedNetwork" + str(runIndexes[0]) + "-" +
                                                    str(runIndexes[-1]) + "_lesionOrdering_withinHB_TopScore" + str(
                                               numToGet) + "_HB")


# Plot all the networks for all patients in set
def plotAllCrossRunSampleOrderingNetworkX(crossRunComparison, clinData, params, runIndexes, numToGet, edgeThresholds=[0.67]):
    for edgeThres in edgeThresholds:
        for patID in crossRunComparison.keys():
            # numToGet = len( crossRunComparison[patID][2] )
            G = plotCrossRunSampleOrderingNetworkX(crossRunComparison[patID][0],
                                           crossRunComparison[patID][1],
                                           crossRunComparison[patID][2],
                                           clinData,
                                           params,
                                           patID,
                                           # numToGet = numToGet,
                                           layout="hierarchcial",
                                           edgeThres=edgeThres,
                                           filename="ConsensusDirectedNetwork" + str(runIndexes[0]) + "-" +
                                                    str(runIndexes[-1]) + "_lesionOrdering_withinHB_TopScore" + str(
                                               numToGet) + "_HB")

# Plot the Network for a patient's cross run
def plotCrossRunSampleOrdering(medianWeights, sigCnt, numTopHB, clinData, params, patID, edgeThres=0.2, layout="",
                               filename="ConsensusDirectedNetwork_lesionOrdering", plot=True):
    G = Network(notebook=False, directed=True, width='1000px', height='1000px', layout=layout)
    if len(medianWeights) > 0:
        nodeWeights = medianWeights.median(axis=0, skipna=True)

        numRuns = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
        for i in range(0, len(medianWeights)):
            vals = medianWeights.iloc[i,]
            samp = vals[vals > 0].index.tolist()
            numRuns.loc[samp, samp] += 1

        for i in range(0, sigCnt.shape[0]):
            if clinData.tissueMap[sigCnt.index[i]] not in clinData.tissueColor.keys():
                clinData.tissueColor[clinData.tissueMap[sigCnt.index[i]]] = 'gray'
            G.add_node(i, label=clinData.sampleIDMap[sigCnt.index[i]],
                       size=(nodeWeights[sigCnt.index[i]] + 1) * 10,
                       title=str(round(nodeWeights[sigCnt.index[i]], 3)),
                       group=clinData.tissueMap[sigCnt.index[i]], physics=False,
                       color=clinData.tissueColor[clinData.tissueMap[sigCnt.index[i]]])

        for i in range(0, sigCnt.shape[0]):
            for j in range(0, sigCnt.shape[1]):
                s = sigCnt.iloc[i, j]
                n = numTopHB.iloc[i, j]  # * numToGet
                if n == 0:
                    sigFreq = 0
                else:
                    sigFreq = s / n
                if sigFreq >= edgeThres:
                    G.add_edge(i, j, title=str(round(sigFreq, 3)), physics=False,
                               weight=sigFreq, width=sigFreq * 10)
        G.show_buttons(filter_="layout")
        G.options.layout.hierarchical.sortMethod = 'directed'
        # G.repulsion( node_distance = 200, spring_strength=0.1, central_gravity = 0.1 )
        # G.inherit_edge_colors_from(True)
        G.toggle_drag_nodes(True)
        G.toggle_physics(False)
        if (plot):
            G.write_html(os.path.join(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering"),
                                      filename + "_edgeThresh" + re.sub("\\.", "",
                                                                        str(edgeThres)) + "_" + patID + ".html"),
                         notebook=False)
    return G


# Plot the Network for a patient's cross run
def plotCrossRunSampleOrderingNetworkX(medianWeights, sigCnt, numTopHB, clinData, params, patID, edgeThres=0.2, layout="",
                               filename="ConsensusDirectedNetwork_lesionOrdering", plot=True):
    G = nx.DiGraph()
    graphDF = []
    if len(medianWeights) > 0:
        nodeWeights = medianWeights.median(axis=0, skipna=True)

        numRuns = pd.DataFrame(0, index=medianWeights.columns, columns=medianWeights.columns)
        for i in range(0, len(medianWeights)):
            vals = medianWeights.iloc[i,]
            samp = vals[vals > 0].index.tolist()
            numRuns.loc[samp, samp] += 1

        color_map = []
        for i in range(0, sigCnt.shape[0]):
            if clinData.tissueMap[sigCnt.index[i]] not in clinData.tissueColor.keys():
                clinData.tissueColor[clinData.tissueMap[sigCnt.index[i]]] = 'gray'
            G.add_node(clinData.sampleIDMap[sigCnt.index[i]])
            color_map.append(clinData.tissueColor[clinData.tissueMap[sigCnt.index[i]]])

        for i in range(0, sigCnt.shape[0]):
            for j in range(0, sigCnt.shape[1]):
                s = sigCnt.iloc[i, j]
                n = numTopHB.iloc[i, j]  # * numToGet
                if n == 0:
                    sigFreq = 0
                else:
                    sigFreq = s / n
                if sigFreq >= edgeThres:
                    G.add_edge(clinData.sampleIDMap[sigCnt.index[i]], clinData.sampleIDMap[sigCnt.index[j]])
                    graphDF.append([clinData.sampleIDMap[sigCnt.index[i]], clinData.sampleIDMap[sigCnt.index[j]], sigFreq ])

        pos = nx.spring_layout(G)
        nx.draw_networkx(G, with_labels=True, pos=pos, node_color=color_map)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        y_max = max(y_values)
        y_min = min(y_values)
        x_margin = (x_max - x_min) * 0.75
        y_margin = (y_max - y_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)

        graphDF = pd.DataFrame(graphDF, columns=['Source', 'Target', 'EdgeWeight'])
        graphDF.to_csv(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                                    filename + "_edgeThresh" + re.sub("\\.", "",
                                                                      str(edgeThres)) + "_" + patID + ".csv"),
                       index = False)

        plt.savefig(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                                 filename + "_edgeThresh" + re.sub("\\.", "",
                                                                   str(edgeThres) + "_" + patID + ".png")))
        plt.show()
        plt.close()
    return G


def calcErrorBySampling(crossRunComparison, clinData, params, patID, layout='hierarchical', numToGet=1, edgeThres=0.5,
                        sampThres=0.6, nSamples=10, replace=False):
    medianWeights = crossRunComparison[0]
    allWeights = crossRunComparison[3]

    allNetStats = pd.DataFrame()

    for n in range(nSamples):
        medianWeights_samp = medianWeights.sample(n=int(np.floor(len(medianWeights) * sampThres)), replace=replace)

        sigCnt = pd.DataFrame(0, index=medianWeights_samp.columns, columns=medianWeights_samp.columns)
        numTopSB = pd.DataFrame(0, index=medianWeights_samp.columns, columns=medianWeights_samp.columns)
        curRunIndexes = medianWeights_samp.index.to_list()

        for runIdx in curRunIndexes:
            weights = allWeights[runIdx]
            for i in range(0, weights.shape[1] - 1):
                for j in range(i + 1, weights.shape[1]):
                    a = weights.iloc[:, i]
                    b = weights.iloc[:, j]
                    sigCnt.loc[weights.columns[i], weights.columns[j]] = sigCnt.loc[weights.columns[i], weights.columns[
                        j]] + sum(a > b)
                    sigCnt.loc[weights.columns[j], weights.columns[i]] = sigCnt.loc[weights.columns[j], weights.columns[
                        i]] + sum(a < b)
                    numTopSB.loc[weights.columns[i], weights.columns[j]] = numTopSB.loc[
                                                                               weights.columns[i], weights.columns[
                                                                                   j]] + len(weights)
                    numTopSB.loc[weights.columns[j], weights.columns[i]] = numTopSB.loc[
                                                                               weights.columns[j], weights.columns[
                                                                                   i]] + len(weights)
        netStats = pd.DataFrame()
        G = plotCrossRunSampleOrdering(medianWeights_samp,
                                       sigCnt,
                                       numTopSB,
                                       clinData,
                                       params,
                                       patID,
                                       layout=layout,
                                       edgeThres=edgeThres,
                                       plot=False)
        netStat = getNetStats(G, clinData)
        netStat['patID'] = [patID] * len(netStat)
        netStats = pd.concat( [netStats, netStat ] )

        # Look at the lesion shedding categories by different specific features
        highThres = 1.0
        lowThres = 0.05
        netStats['outFreq'] = netStats['out'] / netStats['total']
        outFreqCat = []
        for x in netStats['outFreq']:
            if np.isnan(x):
                outFreqCat.append('None')
            elif x >= highThres:
                outFreqCat.append('High')
            elif x <= lowThres:
                outFreqCat.append('Low')
            else:
                outFreqCat.append('Medium')
        netStats['outFreqCat'] = outFreqCat
        netStats['sampID'] = [clinData.sampleIDMapFromOriginal[x] for x in netStats.index]
        netStats['originalSampleID'] = list(netStats.index)
        netStats.index = netStats['sampID']
        netStats['rndIdx'] = [n] * len(netStats)

        allNetStats = pd.concat( [allNetStats, netStats ]) 

    return (allNetStats)


def plotLesionSheddingHeatmap(netStats, outFile, figsize=(5, 5), continuous=False, discrete=3,
                              cbar=False, sampsOfInterest = None, title = '', highThres = 0.8, lowThres = 0.15):
    disColors = ['#1134E8', '#797979', '#D12929']

    if continuous:
        toPlot = netStats.pivot_table(index='originalSampleID', columns='time', values='outFreq')
        cmap = colors.LinearSegmentedColormap.from_list(
            name='test',
            colors=disColors
        )
    else:
        netStats['outCat_discrete'] = [np.nan] * len(netStats)
        dis = []
        for x in netStats.outFreq:
            if not np.isnan(x):
                if discrete == 3:
                    if x >= highThres:
                        dis.append(10)
                    elif x < lowThres:
                        dis.append(1)
                    else:
                        dis.append(5)
                else:
                    if x >= 0.9:
                        dis.append(10)
                    elif x >= 0.8:
                        dis.append(9)
                    elif x >= 0.7:
                        dis.append(8)
                    elif x >= 0.6:
                        dis.append(7)
                    elif x >= 0.5:
                        dis.append(6)
                    elif x >= 0.4:
                        dis.append(5)
                    elif x >= 0.3:
                        dis.append(4)
                    elif x >= 0.2:
                        dis.append(3)
                    elif x >= 0.1:
                        dis.append(2)
                    elif x >= 0.0:
                        dis.append(1)
                    else:
                        dis.append(0)
            else:
                dis.append(np.nan)
        netStats['outCat_discrete'] = dis
        toPlot = netStats.pivot_table(index='originalSampleID', columns='time', values='outCat_discrete')
        if discrete == 3:
            cmap = colors.ListedColormap(disColors)
        else:
            cmap = colors.LinearSegmentedColormap.from_list(
                name='test',
                colors=disColors
            )

    times = list(set(netStats.time))
    times.sort()
    toPlot = toPlot.sort_values(times[len(times) - 1], ascending=False)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(data=toPlot, cmap=cmap, linewidths=0.2, cbar=cbar)
    ax.hlines(list(range(1, toPlot.shape[0] + 1)), *ax.get_xlim(), color='white', linewidth=4)

    ax.set_ylabel("")
    ax.set_xlabel("Time")
    plt.savefig(os.path.join(outFile), dpi = 200)
    plt.show()
    plt.close()


def plotLesionSheddingLineplot(netStats, outFile, figsize=(5, 5), sampsOfInterest = None, title = '', highThres = 0.8, lowThres = 0.2):
    netStats['outCat_discrete'] = [np.nan] * len(netStats)
    dis = []
    for x in netStats.outFreq:
        if not np.isnan(x):
            if x >= highThres:
                dis.append(5)
            elif x < lowThres:
                dis.append(1)
            else:
                dis.append(3)
        else:
            dis.append(np.nan)

    times = list(set(netStats.time))
    times.sort()

    netStats['outCat_discrete'] = dis

    netStats['outFreqRnk'] = [0] * len(netStats)
    for time in times:
        netStats.loc[ netStats.time == time, 'outFreqRnk' ] = stats.rankdata(netStats.loc[ netStats.time == time, 'outFreq' ])

    toPlot = netStats.pivot_table(index='originalSampleID', columns='time', values='outCat_discrete')
    toPlotreal = netStats.pivot_table(index='originalSampleID', columns='time', values='outCat_discrete')

    model = AgglomerativeClustering(n_clusters=None, distance_threshold=1)
    # fit model and predict clusters
    yhat = list(model.fit_predict(toPlot))
    clusters = list(set(yhat))
    clusterAssign = dict(zip(toPlot.index, yhat))

    toPlot_clust = []
    clusterAssignByClust = dict()
    for c in clusters:
        clusterAssignByClust[c] = list(toPlot.index[yhat == c])
        toPlot_clust.append(toPlotreal.loc[clusterAssignByClust[c], :].mean().to_list())
    toPlot_clust = pd.DataFrame(toPlot_clust, index=clusters, columns=toPlot.columns)
    timeCol = toPlot_clust.columns.to_list()
    toPlot_clust['clust'] = toPlot_clust.index
    toPlot = toPlot_clust.melt(id_vars=['clust'], value_vars=timeCol)
    toPlot['Cluster Size'] = [len(clusterAssignByClust[x]) for x in toPlot.clust]
    toPlot['Sample'] = [','.join(clusterAssignByClust[c]) for c in toPlot.clust]

    times = list(set(netStats.time))
    times.sort()

    # Lineplot style
    if sampsOfInterest is not None:
        clustSamp = dict(zip(set(toPlot.clust),
                             ['gray'] * len(set(toPlot.clust))))
        for s in sampsOfInterest:
            if s in clusterAssign.keys():
                clustSamp[clusterAssign[s]] = 'blue'

    sns.reset_orig()

    # sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    if sampsOfInterest is None:
        ax = sns.lineplot(data=toPlot, x='time', y='value', size='Cluster Size',
                          hue='Sample', legend=True, sizes=(3, 15.5), alpha=0.7)
        ouf = os.path.join(re.sub('.png', '_lineplot_rank.png', outFile))
    else:
        ax = sns.lineplot(data=toPlot, x='time', y='value', size='Cluster Size',
                          hue='Sample', legend=True,
                          palette=clustSamp)
        ouf = os.path.join(re.sub('.png', '_lineplot_rank_sampInterest.png', outFile))

    plt.title(title)
    plt.xlabel('Days from Dx')
    plt.ylabel('Relative shedding rank')
    times = list(set(netStats['time']))
    times.sort()
    ax.set_xticks(times)
    ax.set_xticklabels(times)
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.1), ncol=1, )
    plt.tight_layout()
    plt.savefig(ouf, dpi=200)
    plt.show()
    plt.close()


def getPrivateMutation(mafData, clinData, samps, bloodSampleID, ccfThres=0.55, altCnt = 0):
    sampMaf = mafData[mafData.primarySampleID.isin(samps)]
    if altCnt == 0:
        sampMaf = sampMaf[sampMaf.ccf_hat >= ccfThres]
    else:
        sampMaf = sampMaf[sampMaf.t_alt_count_post_forcecall >= altCnt]
    #sampMaf = sampMaf[sampMaf.ccfCI <= 0.75]
    bmaf = mafData[mafData.primarySampleID == bloodSampleID]
   # bmaf = bmaf[ bmaf.ccf_hat >= 0.1 ]

    privateMutations = []
    for soi in samps:
        priv = set(sampMaf[(sampMaf.primarySampleID == soi) ].key).difference(
            set(sampMaf[ (sampMaf.primarySampleID != soi) ].key))
 #       priv = set(sampMaf[(sampMaf.primarySampleID == soi) & (sampMaf.ccf_hat > ccfThres)].key).difference(
 #           set(sampMaf[ (sampMaf.primarySampleID != soi) & (sampMaf.ccf_hat > 0.05) ].key))
        privInBlood = set(bmaf.key).intersection(priv)
        priv_gene = []
        for p in priv:
            smaf = sampMaf[ (sampMaf.primarySampleID == soi) & (sampMaf.key == p) ][['Hugo_Symbol', 'Reference_Allele', 'Start_position', 'Tumor_Seq_Allele2', 'ccf_mean_ibm', 'ccf_hat', 'ccf_CI95_high', 'ccf_CI95_low', 't_alt_count_post_forcecall', 't_ref_count_post_forcecall' ]]
            smaf = [str(x) for x in smaf.iloc[0,:].to_list() ]
            priv_gene.append( smaf )
        privInBlood_gene = []
        for p in privInBlood:
            smaf = sampMaf[ (sampMaf.primarySampleID == soi) & (sampMaf.key == p) ][['ccf_mean_ibm', 'ccf_hat', 'ccf_CI95_high', 'ccf_CI95_low', 't_alt_count_post_forcecall', 't_ref_count_post_forcecall' ]]
            smaf = [str(x) for x in smaf.iloc[0,:].to_list() ]

            pmaf = bmaf[ bmaf.key == p ][['Hugo_Symbol', 'Reference_Allele', 'Start_position', 'Tumor_Seq_Allele2', 'ccf_mean_ibm', 'ccf_hat', 'ccf_CI95_high', 'ccf_CI95_low',
            't_alt_count_post_forcecall', 't_ref_count_post_forcecall', 'key' ]]
            pmaf = [str(x) for x in pmaf.iloc[0,:].to_list() ]

            pmaf = pmaf + smaf
            privInBlood_gene.append( pmaf )
        privateMutations.append([soi, clinData.sampleIDMap[soi], len(priv), privInBlood_gene, priv_gene])

    privateMutations = pd.DataFrame(privateMutations,
                                    columns=['primarySampleID', 'sampleID', 'Private', 'PrivateInBlood', 'PrivateLes'])
    #privateMutations = dict(zip(privateMutations.primarySampleID, privateMutations.PrivateInBlood))

    return privateMutations


def getPrivateMutationCnt(mafData, clinData, samps, bloodSampleID, ccfThres=0.55):
    sampMaf = mafData[mafData.primarySampleID.isin(samps)]
    sampMaf = sampMaf[sampMaf.ccf_hat >= ccfThres]
    #sampMaf = sampMaf[sampMaf.ccfCI <= 0.75]
    bmaf = mafData[mafData.primarySampleID == bloodSampleID]

    privateMutations = []
    for soi in samps:
        priv = set(sampMaf[sampMaf.primarySampleID == soi].key).difference(
            set(sampMaf[sampMaf.primarySampleID != soi].key))
        privInBlood = set(bmaf.key).intersection(set(sampMaf[sampMaf.primarySampleID == soi].key).difference(
            set(sampMaf[sampMaf.primarySampleID != soi].key)))
        privateMutations.append([soi, clinData.sampleIDMap[soi], len(priv), len(privInBlood)])

    privateMutations = pd.DataFrame(privateMutations,
                                    columns=['primarySampleID', 'sampleID', 'Private', 'PrivateInBlood'])
    privateMutations = dict(zip(privateMutations.primarySampleID, privateMutations.PrivateInBlood))

    return privateMutations


def calcRandomSamplingOfRuns(params,
                             patID,
                             runIndexes,
                             clinData,
                             mafData,
                             crossRunComparison,
                             edgeThres = 0.55,
                             nSamples = 10,
                             sampThres = 0.5
):
    # Calculate the random sampling of runs
    allNetStatsSampling = dict()

    bloodSampleIDs = _getBloodSampleInfo( patID, samplesInfo = clinData.samplesInfo, mafData = mafData )

    sampleDays = dict(zip( bloodSampleIDs['primarySampleID'], bloodSampleIDs['days_from_dx']))

    allNetStatsSampling = []
    for bloodSampleID in bloodSampleIDs['primarySampleID']:
        allNetStatsSampling.append( calcErrorBySampling( crossRunComparison[bloodSampleID], patID = bloodSampleID, 
                                                         clinData = clinData, params = params, sampThres = sampThres, nSamples = nSamples ) )
    allNetStatsSampling = pd.concat(allNetStatsSampling)

    allNetStatsSampling.index = allNetStatsSampling['originalSampleID']
    allNetStatsSampling['originalSampleIDs'] = allNetStatsSampling.index

    keyToUse = 'sampID'
    if params.patientMulti != 'None':
        keyToUse = 'patID'
    allNetStatsSampling['time'] = [ int( clinData.daysDxIDMap[x] ) if x in clinData.daysDxIDMap.keys() else np.nan for x in allNetStatsSampling[keyToUse]]
    outFile = os.path.join( params.dir_output, patID + '_edgeThres' + str(edgeThres) + '_run' + str(runIndexes[0]) + '-' + str(runIndexes[len(runIndexes)-1]) + '_randSampling' + str(sampThres) + 
                        'nSample' + str( nSamples) + '.csv' )
    allNetStatsSampling.to_csv( outFile )

    return(allNetStatsSampling)


def plotLesionSheddingCI(netStats, netStats_sampling, outFile,mafData, clinData, figsize=(5, 5),
                         sampsOfInterest = None, title = '', highThres = 0.8, lowThres = 0.2,
                         private = False):

    netStatsOrig = netStats.copy()
    netStats = netStats_sampling.copy()
    times = list(set(netStats.time))
    times.sort()
    sampIDs = list(set(netStats.sampID))

    #Lineplot style
    sampIDs = set(netStatsOrig.sampID)
    if sampsOfInterest is not None:
        sampsColor = dict( zip( sampIDs,
                            ['gray']*len(set(sampIDs))) )
        for s in sampsOfInterest:
            sampsColor[s] = 'blue'
    else:
        sampsColor = dict(zip(sampIDs, list(sns.color_palette('husl', n_colors=len(sampIDs) ) ) ) )

    def rankData(val):
        y = pd.Series(list(stats.rankdata(val, method='max')))
        rnkCnt = Counter(y)
        rnkCnt = pd.DataFrame([rnkCnt.keys(), rnkCnt.values()], index=['val', 'cnt']).transpose()
        rnkCnt = rnkCnt[rnkCnt.cnt > 1]
        if len(rnkCnt) > 0:
            for idx, row in rnkCnt.iterrows():
                x = list(range(row.val - row.cnt + 1, row.val+1))
#                x = [ row.val - (0.1*x) for x in range(row.cnt) ]
                random.shuffle(x)
                y[y == row.val] = x
        return list(y)

    def rankDataJitter(val):
        y = pd.Series(list(stats.rankdata(val, method='max')))
        rnkCnt = Counter(y)
        rnkCnt = pd.DataFrame([rnkCnt.keys(), rnkCnt.values()], index=['val', 'cnt']).transpose()
        rnkCnt = rnkCnt[rnkCnt.cnt > 1]
        if len(rnkCnt) > 0:
            for idx, row in rnkCnt.iterrows():
                x = [ row.val - (0.25*x) for x in range(row.cnt) ]
  #              x = list(range(row.val, row.val + row.cnt))
                random.shuffle(x)
                y[y == row.val] = x
        return list(y)

    netStats['outFreqRnk'] = [0] * len(netStats)
    netStats['outFreqRnkJitter'] = [0] * len(netStats)
    randRuns = list(set(netStats.rndIdx))
    for time in times:
        for rndIdx in randRuns:
            netStats.loc[ (netStats.time == time) & (netStats.rndIdx == rndIdx), 'outFreqRnk' ] = \
                rankData(netStats.loc[ (netStats.time == time) & (netStats.rndIdx == rndIdx)]['outFreq'])
            netStats.loc[ (netStats.time == time) & (netStats.rndIdx == rndIdx), 'outFreqRnkJitter' ] = \
                rankDataJitter(netStats.loc[ (netStats.time == time) & (netStats.rndIdx == rndIdx)]['outFreq'])

    netStatsOrig['rank'] = [0] * len(netStatsOrig)
    for time in times:
        netStatsOrig.loc[ netStatsOrig.time == time, 'rank'] = rankDataJitter( netStatsOrig.loc[ netStatsOrig.time == time, 'outFreq'] )
    rnkCIUpper = []
    rnkCILower = []

    for idx,samp in netStatsOrig.iterrows():
        dist = netStats.loc[(netStats.time == samp.time) &
                            (netStats.sampID == samp.sampID),'outFreqRnkJitter']-samp['rank']
        ci = sns.utils.ci(dist)

        rnkCIUpper.append( ci[1] )
        rnkCILower.append( ci[0] )
    netStatsOrig['rnkCIUpper'] = rnkCIUpper
    netStatsOrig['rnkCILower'] = rnkCILower

    netStats['privateInBlood'] = [0] * len(netStats)
    for time in times:
        df = netStats[(netStats.time == time)]
        bloodSampleID = list(set(df.patID))[0]
        samps = list(set(df.sampID))
        priv = getPrivateMutationCnt(mafData, clinData, samps, bloodSampleID, ccfThres=0.55)
        for samp in samps:
            netStats.loc[ (netStats.time == time) & (netStats.sampID == samp), 'privateInBlood'] = priv[samp]
    privInBlood = dict(zip(netStats.sampID, netStats.privateInBlood))

    dis = []
    for x in netStats.outFreq:
        if not np.isnan(x):
            if x >= highThres:
                dis.append(5)
            elif x < lowThres:
                dis.append(1)
            else:
                dis.append(3)
        else:
            dis.append(np.nan)
    netStats['outCat_discrete'] = dis

    dis = []
    totalLesions = len(set(netStats.sampID))
    for x in netStats.outFreqRnk:
        if not np.isnan(x):
            if x/totalLesions >= 0.7:
                dis.append(5)
            elif x/totalLesions < 0.3:
                dis.append(1)
            else:
                dis.append(3)
        else:
            dis.append(np.nan)
    netStats['outCat_discreteRnk'] = dis

    #absolute lineplot with CI by distance from subsample
    sns.reset_orig()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    netStatsOrig = netStatsOrig.sort_values( ['time', 'sampID'] )
    netStatsOrig = netStatsOrig.reset_index(drop=True)
    if sampsOfInterest is None:
        ax = sns.lineplot(data=netStatsOrig, x='time', y='rank', palette = sampsColor,
                          # style = 'privateInBlood', size = netStats['privateInBlood'],
                          #                          dashes = False, markers=True,
                          hue='sampID', legend=False)
        ouf = os.path.join(re.sub('.svg', '_lineplot_AbsRank.svg', outFile))
    else:
        ax = sns.lineplot(data=netStatsOrig, x='time', y='rank',
                          hue='sampID', legend=False,
                          palette=sampsColor)
        ouf = os.path.join(re.sub('.svg', '_lineplot_AbsRank_sampInterest.svg', outFile))

    times = list(set(netStatsOrig.time))
    times.sort()
    netStatsOrig['Private in Blood'] = [privInBlood[x] for x in netStatsOrig.sampID ]
    netStatsOrig['markerType'] = [ '.' if x == 0 else 'o' for x in netStatsOrig['Private in Blood'] ]
    if private:
        ax = sns.scatterplot(data=netStatsOrig, x='time', y='rank', size='Private in Blood', sizes=(10, 200),
                             markers=netStatsOrig['markerType'], color='gray', legend=True)
#        ax = sns.scatterplot(data=netStatsOrig, x='time', y='rank', size='Private in Blood', color='gray', legend=True)

    sampIDs = list(set(netStatsOrig.sampID))
    for samp in sampIDs:
        df = netStatsOrig[netStatsOrig.sampID == samp]
        plt.fill_between(df.time, df.rnkCILower + df['rank'], df.rnkCIUpper + df['rank'], alpha=0.15,
                         color=sampsColor[df.sampID.to_list()[0]])

    times = list(set(netStats.time))
    times.sort()
    lastTime = times[len(times) - 1]
    txtLast = netStatsOrig[ netStatsOrig.time == lastTime ][['sampID', 'time', 'rank']]
    txtLast.columns = ['id', 'x', 'y']
    txtLast = txtLast.sort_values('y', ascending=False)
    lastpoint = max(txtLast['y']) + 1
    dif = 0.4
    for idx, row in txtLast.iterrows():
        if lastpoint - row.y <= dif:
            row.y = row.y - (dif - (lastpoint - row.y))
        lastpoint = row.y
        text = ax.annotate(row.id,
                           xy=(row.x + 2, row.y),
                           xytext=(0, 0),
                           color='gray',
                           xycoords=(ax.get_xaxis_transform(),
                                     ax.get_yaxis_transform()),
                           textcoords="offset points")
        text_width = (text.get_window_extent(
            fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
        if np.isfinite(text_width):
            ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

    plt.title(title)
    plt.ylim( 0, np.max(netStatsOrig['rank']))
    plt.ylabel('Relative shedding rank')
    times = list(set(netStats['time']))
    times.sort()
    ax.set_xticks(times)
    ax.set_xticklabels(times)
    if private:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 0.6), ncol=1, title='Private in Blood')
    plt.tight_layout()
    plt.savefig(ouf, dpi=200)
    plt.show()
    plt.close()

    # Lineplot with rank
    sns.reset_orig()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    netStats = netStats.reset_index(drop=True)
    if sampsOfInterest is None:
        ax = sns.lineplot(data=netStats, x='time', y='outFreqRnk', #style = 'privateInBlood', size = netStats['privateInBlood'],
#                          dashes = False, markers=True,
                          hue='sampID', ci=95, legend = False, palette = sampsColor)
        ouf = os.path.join(re.sub('.svg', '_lineplot_rank.svg', outFile))
    else:
        ax = sns.lineplot(data=netStats, x='time', y='outFreqRnk',
                          hue='sampID', ci=95, legend = False,
                          palette=sampsColor)
        ouf = os.path.join(re.sub('.svg', '_lineplot_rank_sampInterest.svg', outFile))

    privInBloodTime = []
    times = list(set(netStats.time))
    times.sort()
    sampIDs = list(set( netStats.sampID ))
    for time in times:
        for s in sampIDs:
            mn = np.mean( netStats[ (netStats.time == time) & (netStats.sampID == s)]['outFreqRnk'] )
            markerType = 'o'
            p = 0
            if (s not in privInBlood.keys()) | (privInBlood[s] == 0):
                markerType = '.'
            else:
                p = privInBlood[s]
                #p = 10^p
            privInBloodTime.append( [p, time, mn, markerType ])
    privInBloodTime = pd.DataFrame(privInBloodTime, columns = ['Private in Blood', 'x','y', 'markerType' ])
    if private:
        ax = sns.scatterplot( data = privInBloodTime, x = 'x', y = 'y', size = 'Private in Blood',sizes=(10, 200),
                              markers = privInBloodTime['markerType'], color = 'gray', legend = True)

    txtLast = []
    times = list(set(netStats.time))
    times.sort()
    lastTime = times[len(times)-1]
    sampIDs = list(set( netStats.sampID ))
    for s in sampIDs:
        mn = np.mean( netStats[ (netStats.time == lastTime) & (netStats.sampID == s)]['outFreqRnk'] )
        txtLast.append([ s, lastTime, mn ])
    txtLast = pd.DataFrame(txtLast, columns = ['id', 'x','y' ])
    txtLast = txtLast.sort_values( 'y', ascending = False )
    lastpoint = max(txtLast['y'])+1
    dif = 0.4
    for idx,row in txtLast.iterrows():
        if lastpoint-row.y <= dif:
            row.y = row.y - ( dif - (lastpoint-row.y) )
        lastpoint = row.y
        text = ax.annotate(row.id,
                           xy=(row.x+2, row.y),
                           xytext=(0,0),
                           color= 'gray',
                           xycoords=(ax.get_xaxis_transform(),
                                     ax.get_yaxis_transform()),
                           textcoords="offset points")
        text_width = (text.get_window_extent(
            fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
        if np.isfinite(text_width):
            ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

    plt.title( title )
    plt.xlabel( 'Days from Dx')
    plt.ylabel( 'Relative shedding rank')
    times = list(set(netStats['time']))
    times.sort()
    ax.set_xticks(times)
    ax.set_xticklabels(times)
    if private:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 0.6), ncol=1, title = 'Private in Blood')
    plt.tight_layout()
    plt.savefig(ouf, dpi=200)
    plt.show()
    plt.close()

    if True:
        sns.reset_orig()
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
        if sampsOfInterest is None:
            ax = sns.lineplot(data=netStats, x='time', y='outCat_discreteRnk', palette = sampsColor,
                              hue='sampID', ci=95, legend = False)
            ouf = os.path.join(re.sub('.svg', '_lineplot_rankDiscrete.svg', outFile))
        else:
            ax = sns.lineplot(data=netStats, x='time', y='outFreqRnk',
                              hue='sampID', ci=95, legend = False,
                              palette=sampsColor)
            ouf = os.path.join(re.sub('.svg', '_lineplot_rankDiscrete_sampInterest.svg', outFile))


        txtLast = []
        times = list(set(netStats.time))
        times.sort()
        lastTime = times[len(times)-1]
        sampIDs = list(set( netStats.sampID ))
        for s in sampIDs:
            mn = np.mean( netStats[ (netStats.time == lastTime) & (netStats.sampID == s)]['outCat_discreteRnk'] )
            txtLast.append([ s, lastTime, mn ])
        txtLast = pd.DataFrame(txtLast, columns = ['id', 'x','y' ])
        txtLast = txtLast.sort_values( 'y', ascending = False )
        lastpoint = max(txtLast['y'])+1
        dif = .09
        for idx,row in txtLast.iterrows():
            if lastpoint-row.y <= dif:
                row.y = row.y - ( dif - (lastpoint-row.y) )
            lastpoint = row.y
            text = ax.annotate(row.id,
                               xy=(row.x+2, row.y),
                               xytext=(0,0),
                               color= 'gray',
                               xycoords=(ax.get_xaxis_transform(),
                                         ax.get_yaxis_transform()),
                               textcoords="offset points")
            text_width = (text.get_window_extent(
                fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
                ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

        plt.title( title )
        plt.xlabel( 'Days from Dx')
        plt.ylabel( 'Discrete Shedding Level By Rank')
        times = list(set(netStats['time']))
        times.sort()
        ax.set_xticks(times)
        ax.set_xticklabels(times)
        ax.set_yticks([1,3,5])
        ax.set_yticklabels(['low', 'intermediate', 'high'])
        plt.tight_layout()
        plt.savefig(ouf, dpi=200)
        plt.show()
        plt.close()


        sns.reset_orig()
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
        if sampsOfInterest is None:
            ax = sns.lineplot(data=netStats, x='time', y='outCat_discrete',
                              hue='sampID', ci=95, legend = False)
            ouf = os.path.join(re.sub('.svg', '_lineplot_discrete.svg', outFile))
        else:
            ax = sns.lineplot(data=netStats, x='time', y='outCat_discrete',
                              hue='sampID', ci=95, legend = False,
                              palette=samps)
            ouf = os.path.join(re.sub('.svg', '_lineplot_discrete_sampInterest.svg', outFile))


        txtLast = []
        times = list(set(netStats.time))
        times.sort()
        lastTime = times[len(times)-1]
        sampIDs = list(set( netStats.sampID ))
        for s in sampIDs:
            mn = np.mean( netStats[ (netStats.time == lastTime) & (netStats.sampID == s)]['outCat_discrete'] )
            txtLast.append([ s, lastTime, mn ])
        txtLast = pd.DataFrame(txtLast, columns = ['id', 'x','y' ])
        txtLast = txtLast.sort_values( 'y', ascending = False )
        lastpoint = max(txtLast['y'])+1
        dif = .09
        for idx,row in txtLast.iterrows():
            if lastpoint-row.y <= dif:
                row.y = row.y - ( dif - (lastpoint-row.y) )
            lastpoint = row.y
            text = ax.annotate(row.id,
                               xy=(row.x+2, row.y),
                               xytext=(0,0),
                               color= 'gray',
                               xycoords=(ax.get_xaxis_transform(),
                                         ax.get_yaxis_transform()),
                               textcoords="offset points")
            text_width = (text.get_window_extent(
                fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
                ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

        plt.title( title )
        plt.xlabel( 'Days from Dx')
        plt.ylabel( 'Discrete Shedding Level')
        times = list(set(netStats['time']))
        times.sort()
        ax.set_xticks(times)
        ax.set_xticklabels(times)
        ax.set_yticks([1,3,5])
        ax.set_yticklabels(['low', 'intermediate', 'high'])

        plt.tight_layout()
        plt.savefig(ouf, dpi=200)
        plt.show()
        plt.close()

        # Rank
        fig, ax = plt.subplots(figsize=figsize)
        sns.set_style("whitegrid")

        if sampsOfInterest is None:
            sns.lineplot(ax = ax, data=netStats, x='time', y='outFreq', palette = sampsColor,
                                 hue='sampID', ci = 'sd', legend = False)
            ouf = os.path.join(re.sub('.svg', '_lineplot.svg', outFile))
        else:
            sns.lineplot(ax = ax, data=netStats, x='time', y='outFreq',
                              hue='sampID', ci='sd', legend = False,
                              palette=sampsColor)
            ouf = os.path.join(re.sub('.svg', '_lineplot_sampInterest.svg', outFile))

        txtLast = []
        times = list(set(netStats.time))
        times.sort()
        lastTime = times[len(times)-1]
        sampIDs = list(set( netStats.sampID ))
        for s in sampIDs:
            mn = np.mean( netStats[ (netStats.time == lastTime) & (netStats.sampID == s)]['outFreq'] )
            txtLast.append([ s, lastTime, mn ])
        txtLast = pd.DataFrame(txtLast, columns = ['id', 'x','y' ])
        txtLast = txtLast.sort_values( 'y', ascending = False )
        lastpoint = 1
        dif = 0.025
        for idx,row in txtLast.iterrows():
            if lastpoint-row.y <= dif:
                row.y = row.y - ( dif - (lastpoint-row.y) )
            lastpoint = row.y
            text = ax.annotate(row.id,
                               xy=(row.x+2, row.y),
                               xytext=(0,0),
                               color= 'gray',
                               xycoords=(ax.get_xaxis_transform(),
                                         ax.get_yaxis_transform()),
                               textcoords="offset points")
            text_width = (text.get_window_extent(
                fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
                ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

        plt.title( title )
        plt.ylabel( 'Relative shedding level')
        plt.xlabel( 'Days from Dx')
        ax.set_xticks(times)
        ax.set_xticklabels(times)
        #plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), ncol=1)
        plt.tight_layout()
        plt.savefig(ouf, dpi=200)
        plt.show()
        plt.close()
        sns.reset_orig()

    res = dict()
    res['totalNetStats'] = netStatsOrig
    res['samplingNetStats'] = netStats
    return res


# Plot the Simplified Network for a patient's cross run
def plotSimpleCrossRunSampleOrdering(params, netStats, patID, runIndexes, highThres=0.9, lowThres=0.1,
                                     edgeThres=0.67, layout="hierarchical",
                                     filename="ConsensusSimpleDirectedNetwork_lesionOrderingTEST", plot=True):
    patStats = netStats[netStats['patID'].str.match(patID)]
    outFreq = patStats['out'] / patStats['total']

    highNodes = outFreq.index[outFreq >= highThres].to_list()
    lowNodes = outFreq.index[outFreq <= lowThres].to_list()
    medNodes = outFreq.index[(outFreq < highThres) & (outFreq > lowThres)].to_list()
    orphanNodes = outFreq.index[np.isnan(outFreq)]

    if len(highNodes) == 0:
        highNodes = ['None']
    if len(medNodes) == 0:
        medNodes = ['None']
    if len(lowNodes) == 0:
        lowNodes = ['None']
    if len(orphanNodes) == 0:
        orphanNodes = ['None']
    G = Network(notebook=False, directed=True, width='1000px', height='1000px', layout=layout)
    G.add_node(0,
               label='\n'.join(highNodes),
               title='high',
               size=(len(highNodes) + 1) * 20,
               physics=False,
               shape='box',
               color='red')
    G.add_node(1,
               label='\n'.join(medNodes),
               title='medium',
               size=(len(medNodes) + 1) * 10,
               physics=False,
               shape='box',
               color='orange')
    G.add_node(2,
               label='\n'.join(lowNodes),
               title='low',
               size=(len(lowNodes) + 1) * 10,
               physics=False,
               shape='box',
               color='green')
    G.add_node(3,
               label='\n'.join(orphanNodes),
               title='low',
               size=(len(orphanNodes) + 1) * 10,
               physics=False,
               shape='box',
               color='gray')

    G.add_edge(0, 1,
               physics=False)
    G.add_edge(1, 2,
               physics=False)

    G.show_buttons(filter_="layout")
    G.options.layout.hierarchical.sortMethod = 'directed'
    # G.inherit_edge_colors_from(True)
    G.toggle_drag_nodes(True)
    G.toggle_physics(False)
    G.write_html(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                              filename + str(runIndexes[0]) + "-" +
                              str(runIndexes[-1]) + "_edgeThresh" + re.sub("\\.", "",
                                                                           str(edgeThres)) + "_" + patID + ".html"),
                 notebook=False)
    patStats.to_csv( os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                              'NetStats_run' + str(runIndexes[0]) + "-" +
                              str(runIndexes[-1]) + "_edgeThresh" + re.sub("\\.", "",
                                                                           str(edgeThres)) + "_" + patID + ".csv"),
                     index = False)

# Plot the Simplified Network for a patient's cross run
def plotSimpleCrossRunSampleOrderingNetworkX(params, netStats, patID, runIndexes, highThres=0.9, lowThres=0.1,
                                     edgeThres=0.67, layout="hierarchical",
                                     filename="ConsensusSimpleDirectedNetwork_lesionOrderingTEST", plot=True):
    patStats = netStats[netStats['patID'].str.match(patID)]
    outFreq = patStats['out'] / patStats['total']


    highNodes = outFreq.index[outFreq >= highThres].to_list()
    lowNodes = outFreq.index[outFreq <= lowThres].to_list()
    medNodes = outFreq.index[(outFreq < highThres) & (outFreq > lowThres)].to_list()
    orphanNodes = outFreq.index[np.isnan(outFreq)]

    highNode = ','.join(highNodes)
    medNode = ','.join(medNodes)
    lowNode = ','.join(lowNodes)
    orphanNode = ','.join(orphanNodes)

    graphDF = []
    graphDF.append([highNode, medNode, 1, len(highNodes), len(medNodes)])
    graphDF.append([medNode, lowNode, 1, len(medNodes), len(lowNodes)])
    graphDF.append([orphanNode, None, None, len(orphanNodes), None])
    graphDF = pd.DataFrame(graphDF, columns=['Source', 'Target', 'EdgeWeight', 'SizeSourceNode', 'SizeTargetNode'])
    graphDF.to_csv(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                                filename + str(runIndexes[0]) + "-" +
                              str(runIndexes[-1]) + "_edgeThresh" + re.sub("\\.", "",
                                                                           str(edgeThres)) + "_" + patID + ".csv"),
                   index = False)

    if len(highNodes) == 0:
        highNodes = ['None']
    if len(medNodes) == 0:
        medNodes = ['None']
    if len(lowNodes) == 0:
        lowNodes = ['None']
    if len(orphanNodes) == 0:
        orphanNodes = ['None']

    highNode = '\n'.join(highNodes)
    medNode = '\n'.join(medNodes)
    lowNode = '\n'.join(lowNodes)
    orphanNode = '\n'.join(orphanNodes)

    G = nx.DiGraph()
    G.add_nodes_from([
        (highNode, {"color": "red"}),
        (medNode, {"color": "orange"}),
        (lowNode, {"color": "green"}),
        (orphanNode, {"color": "gray"})
    ])

    color_map = ['red', 'orange', 'green', 'gray']
    size_map = [
        (len(highNodes) ** 2) * 1000,
        (len(medNodes) ** 2) * 1000,
        (len(lowNodes) ** 2) * 1000,
        (len(orphanNodes) ** 2) * 1000
    ]

    G.add_weighted_edges_from([
        (medNode, lowNode, 1),
        (highNode, medNode, 1)
    ])
    pos = nx.spectral_layout(G)
    nx.draw_networkx(G, with_labels=True, pos=pos, node_color=color_map, node_size=size_map)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    y_max = max(y_values)
    y_min = min(y_values)
    x_margin = (x_max - x_min) * 0.75
    y_margin = (y_max - y_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.savefig(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                             filename + str(runIndexes[0]) + "-" +
                             str(runIndexes[-1]) + "_edgeThresh" + re.sub("\\.", "",
                                                                          str(edgeThres)) + "_" + patID + ".png"),
                bbox_inches="tight")
    plt.show()
    plt.close()

    patStats = patStats.drop('Response',axis=1)
    patStats = patStats.sort_values('outFreq', ascending=False)
    patStats.columns = ['edgeIn','edgeOut','totalEdge', 'tissue', 'patientID','outEdgeFrequency','outEdgeFrequencyCategory',
                        'sampleID','originalSampleID', 'time']
    patStats = patStats[ ['patientID', 'sampleID', 'originalSampleID', 'outEdgeFrequencyCategory', 'outEdgeFrequency',
                         'edgeIn','edgeOut','totalEdge', 'tissue', 'time' ]]
    patStats.to_csv( os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering",
                              'NetStats_run' + str(runIndexes[0]) + "-" +
                              str(runIndexes[-1]) + "_edgeThresh" + re.sub("\\.", "",
                                                                           str(edgeThres)) + "_" + patID + ".csv"),
                     index = False)

# Get the overall netstats from a cross comparison run
def getNetStatsFromCrossComparision(crossRunComparison, clinData, params, layout='hierarchical', edgeThres=0.67,
                                    highThres = 1.0,lowThres = 0.05 ):
    netStats = pd.DataFrame()
    for patID in crossRunComparison.keys():
        numToGet = len(crossRunComparison[patID][2])
        G = plotCrossRunSampleOrdering(crossRunComparison[patID][0],
                                       crossRunComparison[patID][1],
                                       crossRunComparison[patID][2],
                                       clinData,
                                       params,
                                       patID,
                                       layout=layout,
                                       edgeThres=edgeThres,
                                       plot=False)
        netStat = getNetStats(G, clinData)
        netStat['patID'] = [patID] * len(netStat)
        netStats = pd.concat( [netStats, netStat] )

    # Look at the lesion shedding categories by different specific features

    netStats['outFreq'] = netStats['out'] / netStats['total']
    outFreqCat = []
    for x in netStats['outFreq']:
        if np.isnan(x):
            outFreqCat.append('None')
        elif x >= highThres:
            outFreqCat.append('High')
        elif x <= lowThres:
            outFreqCat.append('Low')
        else:
            outFreqCat.append('Medium')
    netStats['outFreqCat'] = outFreqCat
    netStats['sampID'] = [clinData.sampleIDMapFromOriginal[x] for x in netStats.index]
    netStats['originalSampleID'] = list(netStats.index)
    netStats.index = netStats['sampID']

    netStats['Response'] = ['None'] * len(netStats)
    for x in netStats.index:
        if x in clinData.lesionResponseMap.keys():
            netStats.loc[x, 'Response'] = clinData.lesionResponseMap[x]

    keyToUse = 'sampID'
    if params.patientMulti != 'None':
        keyToUse = 'patID'
    netStats['time'] = [ clinData.daysDxIDMap[x] if x in clinData.daysDxIDMap.keys() else np.nan() for x in netStats[keyToUse] ]

    return netStats


# Get the Network stats for a specific network
def getNetStats(G, clinData):
    nodeMap = {}
    for n in G.nodes:
        nodeMap[n.get('id')] = n.get('label')

    adjMat = pd.DataFrame(0, index=nodeMap.values(), columns=nodeMap.values())
    for n in range(0, len(G.get_adj_list())):
        for t in list(G.get_adj_list()[n]):
            adjMat.loc[nodeMap[n], nodeMap[t]] += 1
    netStats = pd.DataFrame(0, index=nodeMap.values(), columns=['in', 'out', 'total'])
    inDegree = adjMat.sum(axis=0)
    outDegree = adjMat.sum(axis=1)
    totalDegree = inDegree + outDegree
    netStats.loc[inDegree.keys(), 'in'] = inDegree.values
    netStats.loc[outDegree.keys(), 'out'] = outDegree.values
    netStats.loc[totalDegree.keys(), 'total'] = totalDegree.values
    sampleIDMap = dict(zip(clinData.sampleIDMap.values(), clinData.sampleIDMap.keys()))
    netStats['tissue'] = [clinData.tissueMap[sampleIDMap[x]] for x in list(netStats.index)]
    return netStats


# Compile all the net stats at a given threshold
def compileNetStatsFromSimLesions(crossRunComparison, edgeThres, lesionWeights, clinData, params):
    netStats = pd.DataFrame()
    for patID in crossRunComparison.keys():
        numToGet = len(crossRunComparison[patID][2])
        G = plotCrossRunSampleOrdering(crossRunComparison[patID][0],
                                       crossRunComparison[patID][1],
                                       crossRunComparison[patID][2],
                                       clinData,
                                       params,
                                       patID,
                                       layout="hierarchical",
                                       edgeThres=edgeThres,
                                       plot=False)
        netStat = getNetStats(G, clinData)
        netStat['patID'] = [patID] * len(netStat)
        netStats = pd.concat( [netStats, netStat] )
    alphaMap = dict(zip(lesionWeights.sampleID, lesionWeights.alpha))
    alphas = []
    for x in netStats.index:
        if x in alphaMap.keys():
            alphas.append(alphaMap[x])
        else:
            alphas.append("")
    netStats['alpha'] = alphas
    netStats['outFreq'] = netStats['out'] / netStats['total']
    return netStats


# Get a Dataframe with the lesions and their weights
def getLesionWeightDF(crossRunComparison, clinData):
    w = pd.DataFrame()
    for patID in crossRunComparison.keys():
        dat = clinData.samplesInfo[(clinData.samplesInfo.primaryParticipantID == patID) &
                                   (clinData.samplesInfo['primarySampleID'].str.contains('Lesion'))]
        if len(dat) > 0:
            lesionAlphas = dat['lesionAlpha'][0]
            alphas = [(x.split(":")[0], clinData.sampleIDMap[x.split(":")[0]], x.split(":")[1]) for x in
                      lesionAlphas.split(";")]
            w = w.append(pd.DataFrame(alphas))
    w.columns = ["primarySampleID", 'sampleID', "alpha"]
    w['primaryParticipantID'] = [re.sub("_.*", "", x) for x in w['primarySampleID']]
    return w


def getNetworkDegree(adj):
    degreeOut = {}
    degreeIn = {}
    for k in adj.keys():
        degreeOut[k] = len(adj[k])
        for v in adj[k]:
            if v in degreeIn.keys():
                degreeIn[v] += 1
            else:
                degreeIn[v] = 0

    for k in adj.keys():
        if k not in degreeIn.keys():
            degreeIn[k] = 0
        if k not in degreeOut.keys():
            degreeOut[k] = 0

    return (degreeIn, degreeOut, degreeIn + degreeOut)


def removeBlacklistMutations(mafData, patID, mutBlacklist):
    toremove = pd.DataFrame()
    for m in range(0, len(mutBlacklist)):
        toremove = pd.concat( [toremove, mafData[(mafData['primaryParticipantID'] == patID) &
                                           (mafData['Chromosome'] == mutBlacklist.iloc[m, 0]) &
                                           (mafData['Start_position'] == mutBlacklist.iloc[m, 1]) &
                                           (mafData['Reference_Allele'] == mutBlacklist.iloc[m, 2]) &
                                           (mafData['Tumor_Seq_Allele2'] == mutBlacklist.iloc[m, 3])] ] )

    newMaf = pd.concat([mafData, toremove, toremove]).drop_duplicates(keep=False)
    return newMaf


def simulateLesion(simMafData, patIdx=0, lesionIdx=0, numberEntities=100, ccfRange=(0.6, 1.0), ccfRandom=False,
                   ccfFix=0.8, entityRandom=False):
    curSimMaf = pd.DataFrame(columns=simMafData.columns)
    if entityRandom:
        curSimMaf['Hugo_Symbol'] = ["E" + str(int(random.uniform(1, numberEntities * 10))) for x in
                                    range(0, numberEntities)]
        curSimMaf['Start_position'] = [int(random.uniform(1, numberEntities * 10)) for x in range(0, numberEntities)]
    else:
        curSimMaf['Hugo_Symbol'] = ["E" + str(x) for x in range(0, numberEntities)]
        curSimMaf['Start_position'] = range(0, numberEntities)
    # curSimMaf[ 'Chromosome' ] = [ int( random.uniform( 1, 23 ) ) for x in range( 0, numberEntities ) ]
    curSimMaf['participantID'] = ["SimPatient" + str(patIdx)] * numberEntities
    curSimMaf['sample'] = ["SimLesion" + str(lesionIdx)] * numberEntities
    curSimMaf['sample'] = curSimMaf['participantID'] + "_" + curSimMaf['sample']
    curSimMaf['primarySampleID'] = curSimMaf['sample']
    if ccfRandom:
        curSimMaf['ccf_hat'] = [round(random.uniform(ccfRange[0], ccfRange[1]), 3) for x in range(0, numberEntities)]
    else:
        curSimMaf['ccf_hat'] = [ccfFix] * numberEntities

    curSimMaf['SimSampType'] = ['lesion'] * numberEntities
    curSimMaf['File'] = curSimMaf['primarySampleID']

    return curSimMaf


def simulateBlood(simMafData, patID, alphas, addUniqBloodMutations=0,
                  ccfRange=(0.6, 1.0), ccfRandom=False, ccfFix=0.8):
    patData = simMafData[simMafData['participantID'] == patID]
    patData = patData[patData['SimSampType'] != "blood"]
    numLesion = len(set(patData.sampleID))

    patData.loc[:, 'ids'] = patData['Hugo_Symbol'] + '-' + patData['Start_position'].astype('str')
    patSamples = list(set(patData['primarySampleID']))
    tissueData = pd.DataFrame()
    for samp in patSamples:
        sampData = patData[patData['primarySampleID'] == samp].copy()
        sampData.loc[:, 'sample'] = [re.sub("_-", "-", re.sub('_v[0-9]*_Exome.*', '', re.sub('^.*/', '', x))) for x in
                                     sampData['sample']]
        tissueData = pd.concat( [tissueData, sampData ])

    # pivoting data so we have a table (row lesion, column mutations, entry ccf values)
    tissueData = tissueData.pivot_table(index='primarySampleID', columns='ids', values='ccf_hat',
                                        aggfunc='mean').fillna(0)

    # add tissue information
    tissueData.loc[:, 'tissue'] = [None] * len(tissueData)

    TF = tissueData.copy()
    TF = TF.drop('tissue', axis=1)

    wts = pd.DataFrame(np.random.dirichlet(alphas, len(TF.columns)),
                       index=TF.columns,
                       columns=TF.index)

    wts = wts.transpose()
    sb = TF * wts
    sb = sb.sum(axis=0).reset_index()
    sb.columns = ['ids', 'ccf_hat']

    if addUniqBloodMutations > 0:
        uniqMut = pd.DataFrame()
        uniqMut['ids'] = ["B" + str(x) + "-" + str(int(random.uniform(1, addUniqBloodMutations * 10))) for x in
                          range(0, addUniqBloodMutations)]
        if ccfRandom:
            uniqMut['ccf_hat'] = [round(random.uniform(ccfRange[0], ccfRange[1]), 3) for x in
                                  range(0, addUniqBloodMutations)]
        else:
            uniqMut['ccf_hat'] = [ccfFix] * addUniqBloodMutations
        sb = pd.concat( [sb,  uniqMut] )

    patSimBlood = pd.DataFrame(columns=simMafData.columns)
    patSimBlood['Hugo_Symbol'] = [re.sub("-.*", "", x) for x in sb['ids']]
    patSimBlood['Start_position'] = [re.sub("^.*-", "", x) for x in sb['ids']]
    patSimBlood['participantID'] = [patID] * len(sb)
    patSimBlood['sample'] = patSimBlood['participantID'] + "_" + "SimLesion" + str(numLesion + 1)
    patSimBlood['primarySampleID'] = patSimBlood['sample']
    patSimBlood['ccf_hat'] = [round(x, 3) for x in sb['ccf_hat']]
    patSimBlood['SimSampType'] = ['blood'] * len(patSimBlood)
    patSimBlood['File'] = patSimBlood['primarySampleID']

    return patSimBlood


def simulateBloodFromRealLesions(simMafData, patID, alphas, addUniqBloodMutations=0,
                                 ccfRange=(0.05, 1.0), ccfRandom=False, ccfFix=0.8, simBloodIndex=None):
    patData = simMafData[simMafData['primaryParticipantID'] == patID]
    numLesion = len(set(patData.primarySampleID))

    patData.loc[:, 'ids'] = patData['Hugo_Symbol'] + '-' + patData['Start_position'].astype('str')
    patSamples = list(set(patData['primarySampleID']))
    tissueData = []
    for samp in patSamples:
        sampData = patData[patData['primarySampleID'] == samp].copy()
        sampData.loc[:, 'sample'] = [re.sub("_-", "-", re.sub('_v[0-9]*_Exome.*', '', re.sub('^.*/', '', x))) for x in
                                     sampData['primarySampleID']]
        tissueData.append(sampData)
    tissueData = pd.concat(tissueData)

    # pivoting data so we have a table (row lesion, column mutations, entry ccf values)
    tissueData = tissueData.pivot_table(index='primarySampleID', columns='ids', values='ccf_hat',
                                        aggfunc='mean').fillna(0)

    # add tissue information
    tissueData.loc[:, 'tissue'] = [None] * len(tissueData)

    TF = tissueData.copy()
    TF = TF.drop('tissue', axis=1)

    wts = pd.DataFrame(np.random.dirichlet(alphas, len(TF.columns)),
                       index=TF.columns,
                       columns=TF.index)

    wts = wts.transpose()
    sb = TF * wts
    sb = sb.sum(axis=0).reset_index()
    sb.columns = ['ids', 'ccf_hat']

    if addUniqBloodMutations > 0:
        uniqMut = pd.DataFrame()
        uniqMut['ids'] = ["B" + str(x) + "-" + str(int(random.uniform(1, addUniqBloodMutations * 10))) for x in
                          range(0, addUniqBloodMutations)]
        if ccfRandom:
            uniqMut['ccf_hat'] = [round(random.uniform(ccfRange[0], ccfRange[1]), 3) for x in
                                  range(0, addUniqBloodMutations)]
        else:
            uniqMut['ccf_hat'] = [ccfFix] * addUniqBloodMutations
        sb = pd.concat( [sb, uniqMut ]) 

    patSimBlood = pd.DataFrame(columns=simMafData.columns)
    patSimBlood['Hugo_Symbol'] = [re.sub("-.*", "", x) for x in sb['ids']]
    patSimBlood['Start_position'] = [re.sub("^.*-", "", x) for x in sb['ids']]
    patSimBlood['primaryParticipantID'] = [patID] * len(sb)
    if simBloodIndex is None:
        patSimBlood['sample'] = patSimBlood['primaryParticipantID'] + "_" + "SimLesion" + str(numLesion + 1)
    else:
        patSimBlood['sample'] = patSimBlood['primaryParticipantID'] + "-" + str(
            simBloodIndex) + "_" + "SimLesion" + str(
            numLesion + 1)
    patSimBlood['primarySampleID'] = patSimBlood['sample']
    patSimBlood['ccf_hat'] = [round(x, 3) for x in sb['ccf_hat']]
    patSimBlood['File'] = patSimBlood['primarySampleID']

    lesionWeights = TF.index + ":" + [str(x) for x in alphas]

    return (lesionWeights, patSimBlood)


def simulateBloodFromRealLesionsDriver(params, clinData, mafData, patientsToProcess,
                                       ccfRandom=True, addUniqBlood=250, specificAlphas=[1.0]):
    # Remove Blood Samples from the MAF Data to
    clinData.samplesInfo = clinData.samplesInfo.loc[
        (clinData.samplesInfo['specimenType'] != "blood") & (clinData.samplesInfo['specimenType'] != "")]
    simMafData = mafData.copy()
    simMafData = simMafData[simMafData['primarySampleID'].isin(clinData.samplesInfo['primarySampleID'])]
    simMafData = simMafData[simMafData['primaryParticipantID'].isin(patientsToProcess)]
    simClinData = clinData
    simClinData.samplesInfo = simClinData.samplesInfo[
        simClinData.samplesInfo['primarySampleID'].isin(simMafData['primarySampleID'])]

    # Simulate Blood with real lesions but only changing the alphas to make the simulated blood
    alphas = []
    for patID in patientsToProcess:
        numLesions = len(set(simMafData[simMafData.primaryParticipantID == patID]['primarySampleID']))
        if numLesions > 0:
            alphas = [0.05] * numLesions
            alphas[0:len(specificAlphas)] = specificAlphas
            random.shuffle(alphas)
            simBlood = simulateBloodFromRealLesions(simMafData=simMafData, patID=patID,
                                                    alphas=alphas, addUniqBloodMutations=addUniqBlood,
                                                    ccfRandom=ccfRandom)
            simMafData = pd.concat( [simMafData,simBlood[1] ] )
            simBloodClin = pd.DataFrame(columns=simClinData.samplesInfo.columns)
            simBloodClin['primarySampleID'] = [simBlood[1]['sample'][0]] * 1
            simBloodClin['primaryParticipantID'] = re.sub("_.*", "", simBlood[1]['sample'][0])
            simBloodClin['primarySampleID'] = simBlood[1]['sample'][0]
            simBloodClin['participantID'] = re.sub("_.*", "", simBlood[1]['sample'][0])
            simBloodClin['days_from_dx'] = 1
            simBloodClin['specimenType'] = 'blood'
            simBloodClin['tissueSite'] = 'blood'
            simBloodClin['tissueSiteSimple'] = 'blood'
            simBloodClin['lesionAlpha'] = ";".join(simBlood[0])

            simClinData.samplesInfo = pd.concat( [simClinData.samplesInfo, simBloodClin ] )

    simClinData.updateMappings(params)

    return (simMafData, simClinData, alphas)

