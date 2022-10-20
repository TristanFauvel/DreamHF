# Heart Failure Prediction : Microbiome
## FINRISK DREAM Challenge 2022

Author : Tristan Fauvel

### Challenge description

See <https://www.synapse.org/#!Synapse:syn27130803/wiki/616705>
### Instructions to run the model

To run the model locally on synthetic data, activate the singularity container in Terminal using:

    singularity shell ./singularity/container.sif

This will allow you to have an environment (`Singularity>`) to run the model without installing any additional package.
In order to run the model, run the following command :

    Singularity> python3 src/model.py <Argument for input path>

For example, on my computer, when my cwd is the DreamHF folder, I run :

    Singularity> python3 src/model.py ./

This command can also be used to run the singularity without accessing the singularity shell:

    singularity exec container.sif python3 src/model.py <Argument for input path>


The prediction scores on test dataset are saved in `./output/scores.csv`.


The `baselineModels_example.R` provided in the challenge can be launched using :  

    singularity exec ./singularity/TristanF_Submission_1.sif Rscript src/baselineModels.R

Or in the singularity shell:

    Singularity> Rscript src/baselineModels.R
In that case, the predictions generated are saved to `/output/scores.csv`.

### Instruction to build the singularity container from the recipe.txt
In the `singularity/` folder, run :

    sudo singularity build --force container.sif recipe.txt
### Content

- `singularity/`: singularity container
- `src/`: models
- `test/` : test data
- `train/` : training data
- `R_baseline/` : the code and singularity container provided by the challenge organizes, containing the baseline.
- `output/` : where submissions and predictions are stored
- `utils/` : utility scripts (to make submissions, etc)

### License

This software is distributed under the GNU GENERAL PUBLIC LICENSE. Please refer to the file LICENCE.txt included for details.
