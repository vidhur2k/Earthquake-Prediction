# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import azureml.train.automl
from azureml.train.automl.constants import AUTOML_SETTINGS_PATH, AUTOML_FIT_PARAMS_PATH
from azureml.train.automl._azureautomlclient import AzureAutoMLClient
from azureml.train.automl._azureautomlsettings import AzureAutoMLSettings
from azureml.core import Run
import pickle as pkl

if __name__ == '__main__':
    run = Run.get_context()

    with open(AUTOML_SETTINGS_PATH, 'rb+') as f:
        automl_setting = pkl.load(f)
    with open(AUTOML_FIT_PARAMS_PATH, 'rb+') as f:
        fit_params = pkl.load(f)

    experiment = run.experiment
    ws = experiment.workspace
    if "show_output" in automl_setting:
        del automl_setting["show_output"]
    if "show_output" in fit_params:
        del fit_params["show_output"]
    fit_params["_script_run"] = run

    settings = AzureAutoMLSettings(experiment, **automl_setting)
    automl_estimator = AzureAutoMLClient(experiment, settings)

    local_run = automl_estimator.fit(**fit_params)
