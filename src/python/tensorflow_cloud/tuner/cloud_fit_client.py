# Lint as: python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to enable remote execution of in memory model & Dataset.

To train models in AI Platform Training there is a need to capture a model that
is provided in memory along with other objects such as training and testing
dataset, where the code behind the model may not be readily available as a
Python module, but only available as an in-memory object of the calling process.
"""

import datetime
import os
import pickle
from typing import Text, Dict, Optional, Sequence, Any, Generator
from absl import logging
import google.auth
from googleapiclient import discovery
import tensorflow as tf

from tensorflow_cloud.tuner import cloud_fit_utils
from tensorflow_cloud.utils import google_api_client

MULTI_WORKER_MIRRORED_STRATEGY_NAME = cloud_fit_utils.MULTI_WORKER_MIRRORED_STRATEGY_NAME
MIRRORED_STRATEGY_NAME = cloud_fit_utils.MIRRORED_STRATEGY_NAME
SUPPORTED_DISTRIBUTION_STRATEGIES = cloud_fit_utils.SUPPORTED_DISTRIBUTION_STRATEGIES

# Constants for default cluster spec
DEFAULT_INSTANCE_TYPE = "n1-standard-4"
DEFAULT_NUM_WORKERS = 1  # 1 master + 1 workers
DEFAULT_DISTRIBUTION_STRATEGY = MULTI_WORKER_MIRRORED_STRATEGY_NAME


def cloud_fit(
    model: tf.keras.Model,
    remote_dir: Text,
    region: Text = None,
    project_id: Text = None,
    image_uri: Text = None,
    distribution_strategy: Text = DEFAULT_DISTRIBUTION_STRATEGY,
    job_spec: Dict[str, Any] = None,
    job_id: Text = None,
    **fit_kwargs
) -> Text:
    """Executes in-memory Model and Dataset remotely on AI Platform.

    Args:
        model: A compiled Keras Model.
        remote_dir: Google Cloud Storage path for temporary assets and
            AI Platform training output. Will overwrite value in job_spec.
        region: Target region for running the AI Platform Training job.
        project_id: Project id where the training should be deployed to.
        image_uri: based image used to use for AI Platform Training
        distribution_strategy: Specifies the distribution strategy for remote
            execution when a jobspec is provided. Accepted values are strategy
            names as specified by 'tf.distribute.<strategy>.__name__'.
        job_spec: AI Platform Training job_spec, will take precedence over all
            other provided values except for remote_dir. If none is provided a
            default cluster spec and distribution strategy will be used.
        job_id: A name to use for the AI Platform Training job (mixed-case
            letters, numbers, and underscores only, starting with a letter).
        **fit_kwargs: Args to pass to model.fit() including training and eval
            data. Only keyword arguments are supported. Callback functions will
            be serialized as is, they must be available in run time environment.

    Returns:
        AI Platform job ID

    Raises:
        RuntimeError: If executing in graph mode, eager execution is required
            for cloud_fit.
        NotImplementedError: Tensorflow v1.x is not supported.
    """
    logging.set_verbosity(logging.INFO)

    if distribution_strategy not in SUPPORTED_DISTRIBUTION_STRATEGIES:
        raise ValueError(
            "{} is not supported. Supported Strategies are {}".format(
                distribution_strategy,
                list(SUPPORTED_DISTRIBUTION_STRATEGIES.keys()),
            )
        )

    if cloud_fit_utils.is_tf_v1():
        raise NotImplementedError("Tensorflow v1.x is not supported.")

    # Can only export Datasets which were created executing eagerly
    # Raise an error if eager execution is not enabled.
    if not tf.executing_eagerly():
        raise RuntimeError("Eager execution is required for cloud_fit.")

    if job_spec:
        job_spec["trainingInput"]["args"] = [
            "--remote_dir",
            remote_dir,
            "--distribution_strategy",
            distribution_strategy,
        ]

    else:
        job_spec = _default_job_spec(
            region=region,
            image_uri=image_uri,
            entry_point_args=[
                "--remote_dir",
                remote_dir,
                "--distribution_strategy",
                distribution_strategy,
            ],
        )

    _serialize_assets(remote_dir, model, **fit_kwargs)

    # Setting AI Platform Training to use chief in TF_CONFIG environment
    # variable.
    # https://cloud.google.com/ai-platform/training/docs/distributed-training-details#chief-versus-master  # pylint: disable=line-too-long
    job_spec["trainingInput"]["useChiefInTfConfig"] = "True"

    # If job_id is provided overwrite the job_id value.
    if job_id:
        job_spec["job_id"] = job_id

    _submit_job(job_spec, project_id)
    return job_spec["job_id"]


def _serialize_assets(remote_dir: Text,
                      model: tf.keras.Model,
                      **fit_kwargs) -> None:
    """Serialize Model and Dataset and store them in the local tmp folder.

    Args:
        remote_dir: A Google Cloud Storage path for assets and outputs
        model: A compiled Keras Model.
        **fit_kwargs: Args to pass to model.fit()

    Raises:
        NotImplementedError for callback functions and Generator input types.
    """
    to_export = tf.Module()
    # If x is instance of dataset or generators it needs to be serialized
    # differently.
    if "x" in fit_kwargs:
        if isinstance(fit_kwargs["x"], tf.data.Dataset):
            to_export.x = fit_kwargs.pop("x")
            x_fn = lambda: to_export.x
            to_export.x_fn = tf.function(x_fn, input_signature=())
        elif isinstance(fit_kwargs["x"], Generator):
            raise NotImplementedError("Generators are not currently supported!")
        logging.info("x was serialized successfully.")

    if "validation_data" in fit_kwargs and isinstance(
        fit_kwargs["validation_data"], tf.data.Dataset
    ):
        to_export.validation_data = fit_kwargs.pop("validation_data")
        validation_data_fn = lambda: to_export.validation_data
        to_export.validation_data_fn = tf.function(
            validation_data_fn, input_signature=()
        )
        logging.info("validation_data was serialized successfully.")

    callbacks = []
    if "callbacks" in fit_kwargs:
        callbacks = fit_kwargs.pop("callbacks")

    # The remote component does not save the model after training. To ensure the
    # model is saved after training completes we add a ModelCheckpoint callback,
    # if one is not provided by the user
    has_model_checkpoint = False
    for callback in callbacks:
        if issubclass(tf.keras.callbacks.ModelCheckpoint, callback.__class__):
            has_model_checkpoint = True

    if not has_model_checkpoint:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(remote_dir, "checkpoint"),
            save_freq="epoch"))

    # Add all serializable callbacks to assets.
    to_export.callbacks = pickle.dumps(callbacks)
    callbacks_fn = lambda: to_export.callbacks
    to_export.callbacks_fn = tf.function(callbacks_fn, input_signature=())
    logging.info("callbacks were serialized successfully.")

    # All remaining items can be directly serialized as a dict.
    to_export.fit_kwargs = fit_kwargs
    fit_kwargs_fn = lambda: to_export.fit_kwargs
    to_export.fit_kwargs_fn = tf.function(fit_kwargs_fn, input_signature=())

    tf.saved_model.save(
        to_export, os.path.join(remote_dir, "training_assets"), signatures={}
    )

    # Saving the model
    model.save(os.path.join(remote_dir, "model"))


def _default_job_spec(
    region: Text,
    image_uri: Text,
    entry_point_args: Sequence[Text] = None,
) -> Dict[str, Any]:
    """Creates a basic job_spec for cloud AI Training.

    Args:
        region: Target region for running the AI Platform Training job.
        image_uri: based image used to use for AI Platform Training
        entry_point_args: Args to pass in to the training job.

    Returns:
        a dictionary corresponding to AI Platform Training job spec
    """
    training_inputs = {}

    if entry_point_args is not None:
        training_inputs["args"] = entry_point_args
    training_inputs["region"] = region
    training_inputs["scaleTier"] = "CUSTOM"
    training_inputs["masterType"] = DEFAULT_INSTANCE_TYPE
    training_inputs["workerType"] = DEFAULT_INSTANCE_TYPE
    training_inputs["masterConfig"] = {"imageUri": image_uri}
    training_inputs["workerCount"] = DEFAULT_NUM_WORKERS
    job_spec = {"trainingInput": training_inputs}
    job_spec["job_id"] = "cloud_fit_{}".format(
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    return job_spec


def _submit_job(
    job_spec: Dict[Text, Text],
    project_id: Optional[Text] = None,
) -> None:
    """Submits a training job to cloud AI Training .

    Args:
        job_spec: AI Platform Training job_spec.
        project_id: Project id where the training should be deployed to.

    Raises:
        RuntimeError: If fails to submit the job.
        ValueError: if project id is not provided and can not be retrieved from
        the environment.
    """
    if project_id is None:
        _, project_id = google.auth.default()

    if project_id is None:
        raise ValueError(
            "Could not retrieve the Project ID, it must be provided or "
            "pre-configured in the environment."
        )
    project_id = "projects/{}".format(project_id)

    # Submit job to AIP Training
    logging.info(
        "Submitting job=%s, project=%s to AI Platform.",
        job_spec["job_id"], project_id)

    # Configure AI Platform Training job
    # Disabling cache discovery to suppress noisy warning. More details at:
    # https://github.com/googleapis/google-api-python-client/issues/299
    api_client = discovery.build(
        "ml",
        "v1",
        cache_discovery=False,
        requestBuilder=google_api_client.TFCloudHttpRequest,
    )

    try:
        request = api_client.projects().jobs().create(body=job_spec,
                                                      parent=project_id)
        request.execute()
    except Exception as e:
        raise RuntimeError(
            "Submitting job to AI Platform Training failed with error: "
            "{}".format(e)
        )

    logging.info("Job submitted successfully to AI Platform Training.")
    logging.info("Use gcloud to get the job status or stream logs as follows:")
    logging.info(
        "gcloud ai-platform jobs describe %s/jobs/%s",
        project_id, job_spec["job_id"]
    )
    logging.info(
        "gcloud ai-platform jobs stream-logs %s/jobs/%s",
        project_id, job_spec["job_id"]
    )
