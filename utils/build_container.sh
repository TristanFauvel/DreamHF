#! /bin/bash
echo "Building the singularity container ./singularity/container.sif from ./singularity/recipe.txt"
sudo singularity build --force ./singularity/container.sif ./singularity/recipe.txt
echo "Copying ./singularity/container.sif to ./singularity/test_singularity/my_submission"
cp ./singularity/container.sif ./singularity/test_singularity/my_submission/container.sif
