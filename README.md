# Heart Failure Prediction : Microbiome
## FINRISK DREAM Challenge 2022

### Instructions

To run the model locally on synthetic data, activate the singularity container in Terminal using:

    singularity shell ./singularity/<container name>.sif

The container name is in the form `<Team_Name_Submission_Number>.sif`. For example `TristanF_Submission_1.sif`.

This will allow you to have an environment (`Singularity>`) to run the model without installing any additional package.
In order to run the model, run the following command : 

    Singularity> python3 src/model.py <Argument for input path>

For example, on my computer, when my cwd is the DreamHF folder, I run : 

    Singularity> python3 src/model.py ./

This command also can be used to run the singularity without accessing the singularity shell:

    singularity exec container.sif python3 src/model.py <Argument for input path>

The predictions generated are saved to `/output/scores.csv`.

Have to provide an argument to specify the input directory
The output of predictions score of test dataset are placed in a folder named `<Team_Name_Submission_Number>/output/scores.csv`.


The `baselineModels_example.R` provided in the challenge can be launched using :  

    singularity exec ./singularity/TristanF_Submission_1.sif Rscript src/baselineModels.R

Or in the singularity shell: 

    Singularity> Rscript src/baselineModels.R

### Content
The Singularity container is in the folder `singularity`, the models are in `src`.

 