# Heart Failure Prediction : Microbiome
## FINRISK DREAM Challenge 2022

Tristan Fauvel
No Affiliations

### Challenge description

See <https://www.synapse.org/#!Synapse:syn27130803/wiki/616705>

### Source code

The source code to reproduce the results is available at <https://github.com/TristanFauvel/DreamHF>

* Will you be able to make your submission public as part of the challenge archive? Yes

##Summary Sentence

A Cox Proportional Hazards model with clinical covariates selection and gut microbiote alpha diversity to predict heart failure.

##Background/Introduction

In an exploratory analysis, I noticed that microbiote richness was strongly correlated with the risk of heart failure. Since the baseline Cox PH model with all covariates was outperforming all other submissions, I decided to build upon this baseline by adding microbiome richness as a new feature. Given the huge difference in the predictive performance of the models trained and evaluated on the simulated data vs the real data, it was clear that we could not rely too much on an exploratory analysis performed on the simulated data. Therefore, a second aspect of my strategy was to automate feature selection and model tuning as much as possible. First, I used recursive feature elimination on a Cox PH model to select the best features. Then I added the microbiome richness as a feature and fitted a Cox PH model with L2 penalty. I then optimized the L2 penalty coefficient using a randomized search with repeated 10-fold cross-validation.


##Methods 
Imputation :
- Missing categorical data were imputed by replacing the missing values with the variable with the highest frequency.
- Missing continuous data were imputed by replacing the missing values with the mean.

Additional preprocessing:
For the scoring phase, data from the test and training set were merged into a single training set.

Variables selection:
Clinical covariates were selected using recursive feature elimination with repeated 10-fold cross-validation.

Model : 
I used a Cox Proportional Hazards model (implementation from the scikit-survival Python package), with L2 regularization.

Cross-validation: 
Hyperparameters of the Cox Proportional Hazards model were selected using a randomized search with repeated 10-fold cross-validation.

##Conclusion/Discussion

During the submission phase, I tested much more complex preprocessing approaches based on the recent literature. I was also expecting ensemble models such as gradient-boosted trees to perform well on this task, as these kinds of models performed well on the related Dream Preterm Birth prediction challenge. Unfortunately, it was not the case, and the best model relies on a simple preprocessing and Cox PH model and performs barely better than the best baseline. Despite its simplicity, this model ranked 2nd in the submission phase (and it seems likely that the extraordinary performance of the model ranking 1st is due to data leakage). From this, I draw the following conclusions:
- The absence of direct access to the data made it very difficult to get insights that would allow us to beat the baseline. The difference in performance between the simulated and the real test set was about 0.13, orders of magnitude bigger than the difference in performance between the top-performing models. 
- Maybe as a consequence, almost no submission reached the performance of the Cox PH model based on clinical covariates. Therefore one possible approach was to build on this baseline and add features related to the microbiota composition that was strongly correlated with the risk of heart failure, and perform automated feature selection as well as extensive hyperparameters optimization.

It is interesting to note that despite the fact that many covariates were strongly correlated with the occurrence of heart failure, most of them did not contribute to improving the predictions on the simulated test set. This may be due to some underlying causal mechanism.

##References
S. Pölsterl, “scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn,” Journal of Machine Learning Research, vol. 21, no. 212, pp. 1–6, 2020.

FINRISK - Heart Failure and Microbiome DREAM Challenge (syn27130803)

##Authors Statement
T.F. designed the analysis, wrote the code, and wrote the write-up.
