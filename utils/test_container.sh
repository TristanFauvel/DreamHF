#! /bin/bash
echo "Run container.sif, check outputs in ./singularity/my_submission/test_singularity/output"
cd ./singularity/test_singularity #working directory
singularity run ./my_submission/container.sif ./
read -p "Proceed to the second test? (y/n) " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    singularity run ./my_submission/container.sif  
fi



 
 
