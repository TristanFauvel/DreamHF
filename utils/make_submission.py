# %%
import os
import pathlib
import shutil
from os.path import basename
from zipfile import ZipFile

import synapseclient
from dotenv import load_dotenv
from synapseclient import File

load_dotenv()
ROOT = os.environ.get("root_folder")

submission_type = ".sif"
load_dotenv()

submission_name = os.environ.get("submission_name")
print(submission_name)
#%%
# Save everything in a submission_name folder in order to keep track of submissions
outdir = ROOT + "/output/" + submission_name + "/"
p = pathlib.Path(outdir)
p.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ROOT + "/src/model.py", outdir + "model.py")
shutil.copyfile(ROOT + "/singularity/container.sif",
                outdir + submission_name + ".sif")


outdir = ROOT + "/output/" + submission_name + "/output/"
p = pathlib.Path(outdir)
p.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ROOT + "/output/scores.csv", outdir + "scores.csv")

if submission_type == ".zip":
    file_to_upload = (
        ROOT + "/output/" + submission_name + "/" + submission_name + ".zip"
    )
    try:
        os.remove(file_to_upload)
    except OSError:
        pass

    to_zip = ROOT + "/output/" + submission_name + "/"
    zip_file = ROOT + "/output/" + submission_name
    shutil.make_archive(zip_file, "zip", to_zip)

    shutil.move(
        zip_file + ".zip", file_to_upload
    )  # In two steps in order to avoid putting a zip in a zip
elif submission_type == ".sif":
    file_to_upload = (
        ROOT + "/output/" + submission_name + "/" + submission_name + ".sif"
    )

username = os.environ.get("username")
print(username)
password = os.environ.get("password")
project_id = os.environ.get("project_id")

syn = synapseclient.login(username, password)


# Add a local file to an existing project on Synapse
file = File(path=file_to_upload, parent=project_id)

file = syn.store(
    file
)  # """"

# syn.delete(file) # API documentation : https://help.synapse.org/docs/API-Clients-and-Documentation.1985446128.html


# %%
