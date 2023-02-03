
# Lesion Shedding Model

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#download">Download</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#options">Options</a></li>
        <li><a href="#input">Input</a></li>
        <li><a href="#output">Output</a></li>
      </ul>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- Abstract -->
## Abstract

Sampling cfDNA using liquid biopsies offers clinically important benefits for monitoring of cancer progression. 
A single cfDNA sample represents a mixture of shed tumor DNA from all known and unknown lesions within the patient. 
Although shedding levels have been suggested to hold the key to identifying targetable lesions and uncovering 
treatment resistance mechanisms, the amount of DNA shed by any one specific lesion is still not well characterized. 
We design the LSM (Lesion Shedding Model) that can order lesions from the strongest to the poorest shedding for a 
given patient. Our framework intrinsically models for missing/hidden lesions and operates on blood and lesion cfDNA 
assays to estimate the potential relative shedding levels of lesions into the blood. By characterizing the 
lesion-specific cfDNA shedding levels, we can better understand the mechanisms of shedding as well as more accurately 
contextualize and interpret cfDNA assays to improve their clinical impact.

<!-- Citation -->
## Citation
Kahn Rhrissorrakrai, Filippo Utro, Chaya Levovitz, and Laxmi Parida. Lesion Shedding Model: unraveling site-specific 
contributions to ctDNA.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Requirements

The following software is required for LSM:

- matplotlib (>3.3.2)
- numpy (>1.19.2)
- pandas (>1.2.0)
- plotly (>4.14.2)
- scipy (>1.5.2)
- seaborn (>0.11.1)
- scikit-learn (>0.22.1)
- networkx

### Download

[Git Large File Storage (LFS)](https://git-lfs.github.com/) is required.
To download please run:

`git lfs clone https://github.com/ComputationalGenomics/LSM.git`

<!-- USAGE EXAMPLES -->
## Usage

LSM can be run as a package by importing LSM and calling the main function 
```
# Import the LSM
from geno4sd import LSM
LSM.runLSM( parameterFile )
```

From the commandline, to run the binary LSM from is simply

```
./LSM --params <parameter_file>
```

Similary to run the python source is 

```
python LSM.py --params <parameter_file>
```

This will run it by default in single time point mode where it develops a lesion shedding model each patient in the 
MAF file using the most recent cfDNA sample.

LSM was tested on Ubuntu 18.04.5 LTS

### Options
```
usage: LSM [-h] --params PARAMETER_FILE 
                
arguments:
  -h, --help                    Show this help message and exit
  --params PARAMETER_FILE       File with parameters for run.  
```

## Input

There are three necessary inputs for the LSM.

#### PARAMETER FILE

This is a CSV file with all relevant adjustable parameters and paths for the LSM. An example of the file is provided and
the available parameters are as follows:

- outputDir: [PATH TO OUTPUT FOLDER] path to directory that all output should be created
- sampleFile: [PATH TO SAMPLE FILE] path to sample info CSV files 
- mafFile: [PATH TO MAF FILE] data file in a modified MAF format. By default the LSM runs in single-time point mode and analyzes all 
  patients in the MAF file.
- patientMulti: [NORMALIZED PATIENT ID] (primaryParticipantID) used to run LSM in longitudinal rather than single-time 
  point mode. This flag automatically engages this mode
- simPatient: [NORMALIZED PATIENT ID] (primaryParticipantID) on which to perform a simulation. This flag automatically 
  engages this mode.
- simPatientIndex: [ARRAY] If running in simulation mode, then providing this ARRAY provides the start and end of the range
  for the number of total simulations to run.
  simulations for the same patient.
- nthreads: [INTEGER] number of threads
- subSampleSize: [INTEGER] number of lesions to subsample k
- discreteRange: [ARRAY] range and increment size from which to draw alpha's. Expects 3 elements - start, end, increment. 
  E.g. "0.05,1.05, 0.25"
- startIndex: [INTEGER] Index to begin subsamplings. Each run will create a "Run[Index]" directory that contains results 
  from that subsample.
- endIndex: [INTEGER] Index to end subsamplings. This will determine the total number of subsamplings performed.
- ccfLesionThreshold: [FLOAT] The ct CCF threshold below which alterations within lesions are filtered when constructing HB.
- dropNullTissue: [True or False] Boolean to indciate whether to ignore tissues without tumor fraction data.
- edgeThres: [FLOAT] Threshold above which to connect a source and target lesion in the consensus graph.
- simBloodAlphas: [ARRAY] Discrete alpha values to assign to lesions in simulated blood construction. Lesions are 
  randomly assigned each of the specified elements and the remainder are assigned 0.05, e.g. "1.0,0.6,0.3"
- addUniqSimAlterations: [INTEGER] Number of random mutation to spike into simulated blood samples.
- ccfRandom: [True or False] Boolean to indicate whether spiked in mutations are assigned CCFs randomly drawn from a 
  uniform distribution.


#### Mutation MAF FILE

The LSM expects a modified, compiled MAF file of all patient data to be analyzed 
with the following minimal fields containing the following:
- 'primaryParticipantID': normalized patient ID
- 'primarySampleID': normalized sample ID
- 'Hugo_Symbol': Gene Hugo Symbol
- 'Start_position': Start position of alteration
- 'Reference_Allele': Reference Allele
- 'Tumor_Seq_Allele2': Tumor Allele
- 'Chromosome': Chromosome
- 'ccf_hat': CCF or VAF value
- 'File': Original MAF filename from which the mutation is compiled from. 


#### SAMPLE INFO FILE

The sample info file contains relevant clincial information in a CSV format with the folloowing minimal fields:
- 'primaryParticipantID': normalized patient ID
- 'primarySampleID': normalizedsample ID
- 'participantID': sample patient ID
- 'sampleID': original sample ID
- 'tumor_fraction': the tumor fraction of the sample
- 'days_from_dx': date of sample as days from diagnosis
- 'tissueSite': tissue from which the sample was taken. If missing leave blank. If the sample is cfDNA, then enter 
  'blood'
-  'tissueSiteSimple': additional tissue location column if a simplified tissue location desired. This may be the same
   value as the 'tissueSite' for simplicity.


## Output

The LSM produces an output directory as specified in the parameters file with the final consensus results. The parameters file
is copied into the output directory. *ConsensusDirectedNetwork_lesionOrdering* contains the consensus shedding results.
PNG graph images for the simplified and detailed networks are produced. In addition, CSV files describing the graph topologies
as well as network files to ease import into other graphing packages are provided. 

If the LSM is run in the longitudinal mode, then all output files are keyed by the specific cfDNA sample ID being analyzed.

If the LSM is run in simulation mode then all simulated runs for a given *simRunIndex* are output into folder *SimRun[XXX]*.
This folder also contains a pickle object of the simulated clinical data and CSV of simulated MAF data for the respective
simulation. *ConsensusDirectedNetwork_lesionOrdering* is created within each.

<!-- CONTACT -->
## Contact

For assistance with running LSM, interpreting the results, or other related questions, 
please feel free to contact: Laxmi Parida <parida@us.ibm.com> or Kahn Rhrissorrakrai <krhriss@us.ibm.com>

<!-- LICENSE -->
## License

See [LICENSE](https://github.com/ComputationalGenomics/LSM/blob/main/license) for license information.