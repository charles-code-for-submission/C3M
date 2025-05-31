# C3M 

In submission.

## Requirements

+ Python 3.10
+ Torch 2.5.1
+ scipy 1.12.0
+ scikit-learn 1.1.1
+ pillow 11.0.0
+ pandas 2.2.3
+ numpy 1.26.4

## Dataset

We evaluate the proposed model C3M using real-world healthcare data and leverage the national [All of Us Research Platform](https://www.researchallofus.org/) as the source cohort, and three target cohorts respectively from the [OHSU EHR data warehouse](https://research-data-catalog.ohsu.edu/records/ksqgw-95972), the  Mount Sinai hospital (INSIGHT MS) and the  Columbia University (INSIGHT Columbia) within the [INSIGHT Clinical Research Network](https://insightcrn.org/) to simulate our setting. 

In this work, we focus on the Alzheimer's Disease and Related Dementias (ADRD) prediction prior to disease onset, considering its pervasive influence in the elderly population and practical data availability. 

The data preprocessing is provided in the [Preprocess] directory.

## Run

We employ the pretrained EHR foundation model [Motor](https://shahlab.stanford.edu/doku.php?id=motor) and follow their [tutorial](https://github.com/som-shahlab/motor_tutorial) to obtain EHR representations. 

To train C3M on the source cohort, run 
```bash
python main.py
```

To fine-tune C3M on local cohorts, run 
```bash
python local_evaluation.py
```

   
## Update

The code will be further organized and refactored upon acceptance.
