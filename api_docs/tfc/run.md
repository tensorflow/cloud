description: Runs your Tensorflow code in Google Cloud Platform.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.run" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.run

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/core/run.py#L104-L337">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Runs your Tensorflow code in Google Cloud Platform.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.run(
    entry_point=None, requirements_txt=None, docker_config=&#x27;auto&#x27;,
    distribution_strategy=&#x27;auto&#x27;, chief_config=&#x27;auto&#x27;,
    worker_config=&#x27;auto&#x27;, worker_count=0, entry_point_args=None,
    stream_logs=(False), job_labels=None, service_account=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`entry_point`
</td>
<td>
Optional string. File path to the python file or iPython
notebook that contains the TensorFlow code.
Note this path must be in the current working directory tree.
Example - 'train.py', 'training/mnist.py', 'mnist.ipynb'
If `entry_point` is not provided, then
- If you are in an iPython notebook environment, then the
current notebook is taken as the `entry_point`.
- Otherwise, the current python script is taken as the
`entry_point`.
</td>
</tr><tr>
<td>
`requirements_txt`
</td>
<td>
Optional string. File path to requirements.txt file
containing additional pip dependencies if any. ie. a file with a
list of pip dependency package names.
Note this path must be in the current working directory tree.
Example - 'requirements.txt', 'deps/reqs.txt'
</td>
</tr><tr>
<td>
`docker_config`
</td>
<td>
Optional `DockerConfig`. Represents Docker related
configuration for the `run` API.
- image: Optional Docker image URI for the Docker image being built.
- parent_image: Optional parent Docker image to use.
- cache_from: Optional Docker image URI to be used as a cache when
building the new Docker image.
- image_build_bucket: Optional GCS bucket name to be used for
building a Docker image via
[Google Cloud Build](https://cloud.google.com/cloud-build/).
Defaults to 'auto'. 'auto' maps to a default <a href="../tfc/DockerConfig.md"><code>tfc.DockerConfig</code></a>
instance.
</td>
</tr><tr>
<td>
`distribution_strategy`
</td>
<td>
'auto' or None. Defaults to 'auto'.
'auto' means we will take care of creating a Tensorflow
distribution strategy instance based on the machine configurations
you have provided using the `chief_config`, `worker_config` and
`worker_count` params.
- If the number of workers > 0, we will use
`tf.distribute.experimental.MultiWorkerMirroredStrategy` or
`tf.distribute.experimental.TPUStrategy` based on the
accelerator type.
- If number of GPUs > 0, we will use
`tf.distribute.MirroredStrategy`
- Otherwise, we will use `tf.distribute.OneDeviceStrategy`
If you have created a distribution strategy instance in your script
already, please set `distribution_strategy` as None here.
For example, if you are using `tf.keras` custom training loops,
you will need to create a strategy in the script for distributing
the dataset.
</td>
</tr><tr>
<td>
`chief_config`
</td>
<td>
Optional `MachineConfig` that represents the
configuration for the chief worker in a distribution cluster.
Defaults to 'auto'. 'auto' maps to a standard gpu config such as
`COMMON_MACHINE_CONFIGS.T4_1X` (8 cpu cores, 30GB memory,
1 Nvidia Tesla T4).
For TPU strategy, `chief_config` refers to the config of the host
that controls the TPU workers.
</td>
</tr><tr>
<td>
`worker_config`
</td>
<td>
Optional `MachineConfig` that represents the
configuration for the general workers in a distribution cluster.
Defaults to 'auto'. 'auto' maps to a standard gpu config such as
`COMMON_MACHINE_CONFIGS.T4_1X` (8 cpu cores, 30GB memory,
1 Nvidia Tesla T4).
For TPU strategy, `worker_config` should be a TPU config with
8 TPU cores (eg. `COMMON_MACHINE_CONFIGS.TPU`).
</td>
</tr><tr>
<td>
`worker_count`
</td>
<td>
Optional integer that represents the number of general
workers in a distribution cluster. Defaults to 0. This count does
not include the chief worker.
For TPU strategy, `worker_count` should be set to 1.
</td>
</tr><tr>
<td>
`entry_point_args`
</td>
<td>
Optional list of strings. Defaults to None.
Command line arguments to pass to the `entry_point` program.
</td>
</tr><tr>
<td>
`stream_logs`
</td>
<td>
Boolean flag which when enabled streams logs back from
the cloud job.
</td>
</tr><tr>
<td>
`job_labels`
</td>
<td>
Dict of str: str. Labels to organize jobs. You can specify
up to 64 key-value pairs in lowercase letters and numbers, where
the first character must be lowercase letter. For more details see
[resource-labels](
https://cloud.google.com/ai-platform/training/docs/resource-labels)
</td>
</tr><tr>
<td>
`service_account`
</td>
<td>
The email address of a user-managed service account
to be used for training instead of the service account that AI
Platform Training uses by default. see [custom-service-account](
https://cloud.google.com/ai-platform/training/docs/custom-service-account)
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dictionary with two keys.'job_id' - the training job id and
'docker_image'- Docker image generated for the training job.
</td>
</tr>

</table>

