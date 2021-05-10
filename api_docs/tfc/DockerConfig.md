description: Represents Docker-related configuration for the run API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.DockerConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tfc.DockerConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/core/docker_config.py#L17-L163">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents Docker-related configuration for the `run` API.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.DockerConfig(
    image=None, parent_image=None, cache_from=None, image_build_bucket=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`run` API uses Docker for containerizing your code and its dependencies.

A new Docker image is built every time `run` is called. For this image, you
can configure:

1. The URI (name+tag) of the image using the `image` parameter.
2. The parent Docker image using the `parent_image` parameter. A parent
image is the image that your image is based on. It refers to the
contents of the FROM directive in the Dockerfile.
3. The Docker image to be used as a cache when building the new image,
using the `cache_from` parameter.
4. An option to use
[Google Cloud Build](https://cloud.google.com/cloud-build/) for building
the Docker image instead of using a local Docker daemon process. This
can be done by providing a GCS bucket for storing your code as a tarball
using the `image_build_bucket` parameter.

#### Usage examples:



**1. All defaults.**

```python
tfc.DockerConfig()
```

Use this configuration when you have a local Docker process and are
experimenting with <a href="../tfc/run.md"><code>tfc.run()</code></a>.

A new Docker image will be built using the `local` Docker daemon with a
newly generated `image` URI, and TensorFlow Docker image will be used as the
`parent_image`. There will be no Docker cache used here.

**2. With `parent_image`.**

```python
tfc.DockerConfig(parent_image="tensorflow/tensorflow:latest-gpu")
```

Use this configuration when you want to use your own custom docker image,
which contains TensorFlow, as the parent Docker image. This can be combined
with the other parameters as required.

A new Docker image will be built using the `local` Docker daemon with a
newly generated `image` URI, with the given `parent_image` as the Docker
parent image. There will be no cache used.

**3. With `cache_from`.**

```python
tfc.DockerConfig(cache_from="gcr.io/test-project/tf_cloud_train:01")
```

Using `cache_from` will speed up the docker build process. Use this when you
want to build a new image in the <a href="../tfc/run.md"><code>tfc.run</code></a> call but want to start from an
existing image. This can be combined with the other parameters as required.

A new Docker image will be built using the `local` Docker daemon with a
newly generated `image` URI, TensorFlow Docker image will be used as the
`parent_image`. `cache_from` will be used as the Docker cache image.

**4. With `image_build_bucket`.**

```python
tfc.DockerConfig(image_build_bucket="test-gcs-bucket")
```

Use this configuration if you do not have a local docker installation.

A new Docker image will be built using the
[Google Cloud Build](https://cloud.google.com/cloud-build) Docker with
a newly generated `image` URI, TensorFlow Docker image will be used as the
`parent_image`. There will be no cache used.

**5. With `image`.**

```python
tfc.DockerConfig(
    parent_image="tensorflow/tensorflow:latest-gpu",
    image="gcr.io/test-project/tf_cloud_train:01",
    image_build_bucket="test-gcs-bucket")
```

Use this configuration when you want to provide a custom image URI for the
Docker image created. Note that if you have not provided `cache_from`
parameter explicitly, then this `image` will also be used as the cache
image. This is useful when you have called <a href="../tfc/run.md"><code>tfc.run</code></a> once and received
an image URI, you can iteratively cache from and rebuild the same image
using this parameter. This can be combined with the other parameters as
required.

6. All custom values.

```python
tfc.DockerConfig(
    parent_image="tensorflow/tensorflow:latest-gpu",
    image="gcr.io/test-project/tf_cloud_train:02",
    image_build_bucket="test-gcs-bucket",
    cache_from="gcr.io/test-project/tf_cloud_train:01")
```

Here, we are combining all the above configs.

A new Docker image will be built using the `Google Cloud Build` Docker with
the given `image` URI, with the given `parent_image` as the Docker parent
image. `cache_from` will be used as the Docker cache image.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`image`
</td>
<td>
Optional Docker image URI for the Docker image being built.
If this is not specified, a new URI is generated.
</td>
</tr><tr>
<td>
`parent_image`
</td>
<td>
Optional parent Docker image to use.
Example value - 'gcr.io/my_gcp_project/deep_learning:v2'
If a parent Docker image is not provided here, we will use a
[TensorFlow Docker image](https://www.tensorflow.org/install/docker)
as the parent image. The version of TensorFlow and Python in that
case will match your local environment.
If both `parent_image` and a local TF installation are not
available, the latest stable TF Docker image will be used.
Example - 'tensorflow/tensorflow:latest-gpu'.
</td>
</tr><tr>
<td>
`cache_from`
</td>
<td>
Optional Docker image URI to be used as a cache when building
the new Docker image. This is especially useful if you are iteratively
improving your model architecture/training code.
If this parameter is not provided, then we will use `image` URI as
cache.
</td>
</tr><tr>
<td>
`image_build_bucket`
</td>
<td>
Optional GCS bucket name to be used for building a
Docker image via
[Google Cloud Build](https://cloud.google.com/cloud-build/).
If it is not specified, then your local Docker daemon will be used for
Docker build.
If a GCS bucket name is provided, then we will upload your code as
a tarfile to this bucket, which is then used by Google Cloud Build
for remote Docker containization.
Note - This parameter is required when using `run` API from within
an iPython notebook.
</td>
</tr>
</table>



