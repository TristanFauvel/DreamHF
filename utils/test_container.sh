#! /bin/bash
echo "Run container.sif, check outputs in ./singularity/test_singularity/output"
singularity run ./singularity/test_singularity/container.sif ./singularity/test_singularity
read -p "Proceed to the second test? (y/n) " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    singularity run ./singularity/test_singularity/container.sif  
fi



 
 
