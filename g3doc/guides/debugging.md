# Debugging TensorFlow Cloud Workflows

Here are some tips for fixing unexpected issues.

#### Operation disallowed within distribution strategy scope

**Error like**: Creating a generator within a strategy scope is disallowed,
because there is ambiguity on how to replicate a generator (e.g. should it be
copied so that each replica gets the same random numbers, or 'split' so that
each replica gets different random numbers).

**Solution**: Passing `distribution_strategy='auto'` to `run` API wraps all of
your script in a TF distribution strategy based on the cluster configuration
provided. You will see the above error or something similar to it, if for some
reason an operation is not allowed inside distribution strategy scope. To fix
the error, please pass `None` to the `distribution_strategy` param and create a
strategy instance as part of your training code as shown in
[this](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/testdata/save_and_load.py)
example.

#### Docker image build timeout

**Error like**: requests.exceptions.ConnectionError: ('Connection aborted.',
timeout('The write operation timed out'))

**Solution**: The directory being used as an entry point likely has too much
data for the image to successfully build, and there may be extraneous data
included in the build. Reformat your directory structure such that the folder
which contains the entry point only includes files necessary for the current
project.

#### Version not supported for TPU training

**Error like**: There was an error submitting the job.Field: tpu_tf_version
Error: The specified runtime version '2.3' is not supported for TPU training.
Please specify a different runtime version.

**Solution**: Please use TF version 2.1. See TPU Strategy in
[Cluster and distribution strategy configuration section](#cluster-and-distribution-strategy-configuration).

#### TF nightly build.

**Warning like**: Docker parent image '2.4.0.dev20200720' does not exist. Using
the latest TF nightly build.

**Solution**: If you do not provide `docker_config.parent_image` param, then by
default we use pre-built TF docker images as parent image. If you do not have TF
installed on the environment where `run` is called, then TF docker image for the
`latest` stable release will be used. Otherwise, the version of the docker image
will match the locally installed TF version. However, pre-built TF docker images
aren't available for TF nightlies except for the latest. So, if your local TF is
an older nightly version, we upgrade to the latest nightly automatically and
raise this warning.

#### Mixing distribution strategy objects.

**Error like**: RuntimeError: Mixing different tf.distribute.Strategy objects.

**Solution**: Please provide `distribution_strategy=None` when you already have
a distribution strategy defined in your model code. Specifying
`distribution_strategy'='auto'`, will wrap your code in a TensorFlow
distribution strategy. This will cause the above error, if there is a strategy
object already used in your code.
