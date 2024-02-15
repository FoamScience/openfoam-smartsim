#!/usr/bin/python3

# This script sets up a smartsim parameter variation with ensembles, running the simpleFoam solver 
# on the pitzDaily case for different mesh resolutions.
# The experiment involves the use of the fieldToSmartRedis function objects, 
# which writes a set of OpenFOAM fields to the SmartRedis database. The SmartRedis client 
# then reads these fields

# To run this script, you need to `mkdir input && cp -r pitzDaily input/`

from smartsim import Experiment
from smartredis import Client 
import os
import shutil
import jinja2 as jj

env = jj.Environment()

def get_field_name(client, fn_name, field_name, processor=0, timestep=None, ensemble_member=None):
    """
    Get the name of the field from the database. This function uses
    a metadata dataset posted by the function object itself to determine
    how things are named through Jinja2 templates

    Args:
        fn_name (str): The name of the function object
        field_name (str): The name of the OpenFOAM field
        processor (int): The MPI rank
        timestep (int): The target timestep index
    """
    if (ensemble_member != None):
        client.set_data_source(ensemble_member.name)
        client.use_dataset_ensemble_prefix(True)
        client.use_list_ensemble_prefix(True)
        client.use_model_ensemble_prefix(True)
        client.use_tensor_ensemble_prefix(True)
    ds_name =  f"{fn_name}_metadata"
    ds_found = client.poll_dataset(ds_name, 10, 1000)
    if not ds_found:
        raise ValueError(f"Could not find dataset {ds_name} under prefix '{os.environ['SSKEYIN']}'")
    meta = client.get_dataset(ds_name)
    ds_naming = env.from_string(str(meta.get_meta_strings("dataset")[0]))
    ds_name = ds_naming.render(time_index=timestep, mpi_rank=processor)
    f_naming = env.from_string(str(meta.get_meta_strings("field")[0]))
    f_name = f_naming.render(name=field_name, patch="internal")
    return f"{{{ds_name}}}.{f_name}"

casename = "pitzDaily"

# Create an experiment
exp = Experiment(name=casename, launcher="local")

# blockMesh runner
rs = exp.create_run_settings(exe="./input/pitzDaily/Allrun.pre")

# Study parameters
params = {
    "n_cells": [20, 25, 30, 50]
}

# Create and start the DB
db = exp.create_database(port=8000, interface="lo")
exp.start(db)

# Create an ensemble
ensemble = exp.create_ensemble("mesh-study", 
                               params=params, 
                               run_settings=rs, 
                               perm_strategy="random",
                               n_models=4,
                               db=db)

# Copy the case dir. and process tagged files
ensemble.attach_generator_files(
        to_copy=f"./input/{casename}",
        to_configure=[f"./input/{casename}/system/blockMeshDict"])

# Generate parameter variation schemes
exp.generate(ensemble, overwrite=True)

# Fix paths to retain correct structure of OpenFOAM cases
for ent in ensemble.entities:
    for f in ent.files.tagged:
        bn = os.path.basename(f)
        dn = os.path.dirname(f)
        src = "./{}/{}/{}/{}".format(exp.name, ensemble.name, ent.name, bn)
        dst = "./{}/{}/{}/{}".format(exp.name, ensemble.name, ent.name,
                                     os.path.relpath(f, f"./input/{casename}"))
        shutil.move(src, dst)

# Start experiments which will run blockMesh on all possible cases
exp.start(ensemble)

# Set the data sources
os.environ['SSKEYIN'] = ",".join([f"{e.name}" for e in ensemble])
print(os.environ['SSKEYIN'])

# Instantiate the DB client
# Note: SSKEYIN before the client instantiation
client = Client(address=db.get_address()[0], cluster=False)
fn_name = "pUPhiTest"

# Loop through ensemble members looking for the U field
for e in ensemble:
    U_name = get_field_name(client, fn_name, "U", 0, 1, e)
    t_exists = client.tensor_exists(U_name)
    print(f"--- {U_name} is found for member {e.name}? {t_exists}")

# Stop the experiment
exp.stop(db)
