#from __future__ import division
__author__ = "Kahn Rhrissorrakrai"
__copyright__ = "Copyright 2018, IBM Research"
__version__ = "0.0.1"
__maintainer__ = "Kahn Rhrissorrakrai"
__email__ = "krhriss@us.ibm.com"
__status__ = "Development"

# Libraries
import argparse, os, pickle, sys
import pandas as pd
import shutil
import warnings
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')

from geno4sd.liquid_biopsy.LSM import _LSMFunctions as lsm

def compute_shedding(paramFile):
    """
    Function to compute and plot the relative shedding of lesions from cfDNA

    :param paramFile: <string> path to the parameters file
    """
    
    verbose = True
    dropNullTissue = True

    if len(sys.argv) < 2:
        print ('You failed to provide the parameter filed. Please use -h to see example usage.')
        sys.exit(1)  # abort because of error

    # Read Parameters
    params = lsm.ParametersFile(paramFile)

    # Create Output Directory and write parameters to file
    if not os.path.isdir(params.dir_output):
        os.mkdir(params.dir_output)

    shutil.copy( paramFile, params.dir_output )

    # Load Clinical Data
    clinData = lsm.clinicalData(params)

    # Read MAF Data or Simulate Data
    if params.simPatient == 'None':
        runRange = range(0,1)
    else:
        runRange = params.simPatientIndex

    dirOutputOriginal = params.dir_output

    for overallRunIndex in runRange:
        mafData = lsm._readMafData(params.file_mafFile)
        # Generate simulated data if needed
        if params.simPatient != 'None':
            patRand = overallRunIndex
            params.dir_output = dirOutputOriginal
            params.dir_output = os.path.join( params.dir_output, "SimRun" + str(patRand))
            # Create Output Directory and write parameters to file
            if not os.path.isdir(params.dir_output):
                os.mkdir(params.dir_output)

            mafFile = os.path.join( params.dir_output, "SimMafData_" + params.simPatient + "_SimRun" + str(patRand) + ".csv" )
            clinFile = os.path.join( params.dir_output, "SimClinData_" + params.simPatient + "_SimRun" + str(patRand) + ".pkl" )
            if os.path.exists(mafFile):
                mafData = pd.read_csv( mafFile )
                clinData = pickle.load( open( clinFile, 'rb', ) )
            else:
                (mafData, clinData, alphas) = lsm.simulateBloodFromRealLesionsDriver(params=params,
                                                                               clinData=clinData,
                                                                               mafData=mafData,
                                                                               patientsToProcess=[params.simPatient ],
                                                                               ccfRandom=params.ccfRandom,
                                                                               addUniqBlood=params.addUniqSimAlterations,
                                                                               specificAlphas=params.simBloodAlphas)
                mafData.to_csv( os.path.join( params.dir_output, "SimMafData_" + params.simPatient + "_SimRun" + str(patRand) + ".csv" ) )
                pickle.dump( clinData, open(os.path.join( params.dir_output, "SimClinData_" + params.simPatient + "_SimRun" + str(patRand) + ".pkl" ), "wb")  )

        # Filter the cohort of patients to run by what is present in the file
        clinData.patientIDOrigMap = dict(zip(clinData.patientIDMap.values(), clinData.patientIDMap.keys()))

        clinData.cohortPatients = [c for c in clinData.cohortPatients if c in list(mafData['primaryParticipantID'])]
        clinData.cohortPatients.sort()


        # Run the the range
        runIndexes = range(params.startIndex, params.endIndex)

        if params.patientMulti == 'None':
            # Final set of patients to analyze
            patientsToAnalyze = clinData.cohortPatients

            for runIdx in runIndexes:
                params.setRunIdx(runIdx)

                # Create Output Directory
                if not os.path.isdir(params.dir_runOutput):
                    os.mkdir(params.dir_runOutput)

                res = lsm.runAllPatients(patientsToAnalyze, params, clinData, mafData, verbose=True, useSlow=False, dropNullTissue=params.dropNullTissue)
        else:
            bloodSampleIDs = lsm.runSinglePatientMultiBlood(params.patientMulti, params, runIndexes, clinData,
                                                        mafData, verbose=True, dropNullTissue=params.dropNullTissue)


        # Cross Run Comparison
        numToGet = 1
        if params.patientMulti == 'None':
            crossRunComparison = lsm.compareCrossRunDriver(params=params,
                                                       patientsToAnalyze=patientsToAnalyze,
                                                       clinData=clinData,
                                                       runIndexes=runIndexes,
                                                       numToGet=numToGet,
                                                           savePickle = params.saveIntermediate)
        else:
            if not os.path.isdir(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering")):
                os.mkdir(os.path.join(params.dir_output, "ConsensusDirectedNetwork_lesionOrdering"))

            patientsToAnalyze = [bloodSampleIDs['primaryParticipantID'].to_list()[0]]
            metric = "Chebyshev"
            crossRunComparison = dict()
            tie = True
            for bloodSampleID in bloodSampleIDs['primarySampleID']:
                for patID in patientsToAnalyze:
                    if patID in crossRunComparison.keys():
                        medianWeights = crossRunComparison[patID][0]
                        sigCnt = crossRunComparison[patID][1]
                        numTopHB = crossRunComparison[patID][2]
                    else:
                        medianWeights = None
                        sigCnt = None
                        numTopHB = None

                    crossRunComparison[bloodSampleID] = lsm.compareCrossRunSampleOrderingWithinHB_timecourse(bloodSampleID,
                                                                                                         patID,
                                                                                                         runIndexes,
                                                                                                         metric,
                                                                                                         numToGet,
                                                                                                         clinData,
                                                                                                         params,
                                                                                                         medianWeights,
                                                                                                         sigCnt,
                                                                                                         numTopHB,
                                                                                                         tie=tie)
                    if params.saveIntermediate:
                        crossRunSaveFile = lsm.getCrossRunSaveFile(params, numToGet, runIndexes)
                        pickle.dump(crossRunComparison, open(crossRunSaveFile, "wb"))

        # Make Network Plots
        # Plot simplified and complex lesion networks
        netStats = lsm.getNetStatsFromCrossComparision(crossRunComparison=crossRunComparison,
                                                   clinData=clinData, params=params,
                                                   edgeThres=params.edgeThres)
        netStats.index = netStats['originalSampleID']
        patIDs = list(set(list(netStats['patID'])))

        for patID in patIDs:
            lsm.plotSimpleCrossRunSampleOrderingNetworkX(params=params, netStats=netStats, patID=patID,
                                             runIndexes=runIndexes, edgeThres=params.edgeThres,
                                             highThres=1.0,
                                             lowThres=0.05)
                                             
            if params.patientMulti != 'None':
                # Plot the CI lineplot
                allNetStatsSampling = lsm.calcRandomSamplingOfRuns(params=params,
                                    patID=patientsToAnalyze[0],
                                    runIndexes=runIndexes,
                                    clinData=clinData,
                                    mafData=mafData,
                                    crossRunComparison=crossRunComparison,
                                    edgeThres = params.edgeThres,
                                    nSamples = params.nSamplesCI,
                                    sampThres = params.sampThresCI)
                
                outFile = os.path.join( params.dir_output, patID + '_continuous_randSampling' + str(params.sampThresCI) + 'nSample' + str( params.nSamplesCI) + '.svg' )
                lsm.plotLesionSheddingCI(netStats, allNetStatsSampling, outFile, mafData, clinData, figsize = (13.5,7),
                                            title = 'Sampling level = ' + str(params.sampThresCI),highThres = 0.8, lowThres = 0.2, private = False)


        lsm.plotAllCrossRunSampleOrderingNetworkX(crossRunComparison, clinData, params, runIndexes,
                                      numToGet, edgeThresholds=[params.edgeThres])


        # Delete Intermediate runs
        if not params.saveIntermediate:
            if params.simPatient != 'None':
                simMafFile = os.path.join(params.dir_output, "SimMafData_" + params.simPatient + "_SimRun" + str(patRand) + ".csv")
                simClinFile = os.path.join(params.dir_output, "SimClinData_" + params.simPatient + "_SimRun" + str(patRand) + ".pkl")
                os.remove(simMafFile)
                os.remove(simClinFile)
            runIndexes = range(params.startIndex, params.endIndex)
            for runIndx in runIndexes:
                if os.path.isdir(os.path.join(params.dir_output, "Run" + str(runIndx))):
                    shutil.rmtree(os.path.join(params.dir_output, "Run" + str(runIndx)))

def _main(args):
    args = vars(parser.parse_args())
    compute_shedding(args['params'])


if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='LSM')
    parser.add_argument('--params', help='File of parameters')
    args = vars(parser.parse_args())
    _main(args)
