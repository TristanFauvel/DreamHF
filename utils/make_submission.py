#%%
import synapseclient
from synapseclient import File
import os
from dotenv import load_dotenv
from zipfile import ZipFile
from os.path import basename
import shutil
import pathlib

load_dotenv("./utils/.env")

submission_name = os.environ.get("submission_name")

# Save everything in a submission_name folder in order to keep track of submissions
outdir = "./output/" + submission_name + "/"
p = pathlib.Path(outdir)
p.mkdir(parents=True, exist_ok=True)
shutil.copyfile("./src/model.py", outdir + "model.py")
shutil.copyfile("./singularity/container.sif", outdir + submission_name + ".sif")


outdir = "./output/" + submission_name + "/output/"
p = pathlib.Path(outdir)
p.mkdir(parents=True, exist_ok=True)
shutil.copyfile("./output/score.csv", outdir + "score.csv")

file_to_upload = "./output/" + submission_name + '/' + submission_name + '.zip' 
try:
    os.remove(file_to_upload)
except OSError:
    pass

to_zip = "./output/" + submission_name +'/'
zip_file =  "./output/" + submission_name
shutil.make_archive(zip_file, "zip",to_zip)

shutil.move(
    zip_file + '.zip', file_to_upload
)  # In two steps in order to avoid putting a zip in a zip

username = os.environ.get("username")
print(username)
password = os.environ.get("password")
project_id = os.environ.get("project_id")

syn = synapseclient.login(username, password)

#%%
# Add a local file to an existing project on Synapse
file = File(path=file_to_upload, parent=project_id)
#file = syn.store(file) ####################################################################################################""""

# syn.delete(file) # API documentation : https://help.synapse.org/docs/API-Clients-and-Documentation.1985446128.html

 