# Heart Failure Prediction : Microbiome
## FINRISK DREAM Challenge 2022

Author : Tristan Fauvel

### Challenge description

See <https://www.synapse.org/#!Synapse:syn27130803/wiki/616705>


### Instruction to build the singularity container from the recipe.txt

To create an executable singularity container (which contains the code), run :

```bash
./utils/build_container.sh
```

This will create a container container.sif in /singularity.
To test the executable singularity container, run :

```bash
./utils/test_container.sh
```

This will output the results in singularity/test_singularity/output.

### Instructions to run the model

The singularity container container.sif is executable, you can reproduce the results by running:

```bash
singularity run ./singularity/container.sif <Argument for input path>
```

Note: `<Argument for input path> ` is optional, default is ./


Note that container.sif should be in the same folder as /training and /test.
The prediction scores on test dataset are saved in `./output/scores.csv`.
### Instructions to run a custom model in the same environment

To run a model locally on synthetic data in the container environement, activate the singularity container in Terminal using:

```bash
singularity shell ./singularity/container.sif
```

This will allow you to have an environment (`Singularity>`) to run the model without installing any additional package.
In order to run the model, run the following command :

```bash
Singularity> python3 src/model.py <Argument for input path>
```

For example, on my computer, when my cwd is the DreamHF folder, I run :

```bash
Singularity> python3 src/model.py ./
```

This command can also be used to run the singularity without accessing the singularity shell:

```bash
singularity exec container.sif python3 src/model.py <Argument for input path>
```
### Baseline models

The `baselineModels_example.R` provided by the challenge organizers can be launched using the singularity container provided by the organizing team (singularity.sif) :  

```bash
singularity exec ./singularity/TristanF_Submission_1.sif Rscript src/baselineModels.R
```

Or in the singularity shell:

```bash
Singularity> Rscript src/baselineModels.R
```

In that case, the predictions generated are saved to `/output/scores.csv`.

### Submissions:

- Submission 1: Gradient boosted tree (sksurv)
- Submission 2: Gradient boosted tree with PCA and cross-validation
- Submission 3: CoxPH with alpha diversity, and selection of clinical covariates with RFECV + crossvalidation (there was an issue in this submission as the risk score was not computed using 1-predict_survival_function : may cause calibration issues)
- Submission 4: CoxPH with alpha diversity + crossvalidation and correct risk scores
- Submission 5: sksurv gbt (without Shannon) with RFECV (based on regularized CoxPH) with PCA and cross-validation

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
