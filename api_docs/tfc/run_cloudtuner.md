description: A wrapper for tfc.run that allows for running concurrent CloudTuner jobs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.run_cloudtuner" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.run_cloudtuner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/core/run.py#L33-L101">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A wrapper for tfc.run that allows for running concurrent CloudTuner jobs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.run_cloudtuner(
    num_jobs=1, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This method takes the same parameters as tfc.run() and it allows duplicating
a job multiple times to enable running parallel tuning jobs using
CloudTuner. All jobs are identical except they will have a unique
KERASTUNER_TUNER_ID environment variable set in the cluster to enable tuning
job concurrency. This feature is only supported in Notebooks and Colab.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_jobs`
</td>
<td>
Number of concurrent jobs to be submitted to AI Platform
training. Note that these are clones of the same job that are executed
independently. Setting this value to 1 is identical to just calling
<a href="../tfc/run.md"><code>tfc.run()</code></a>.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments for <a href="../tfc/run.md"><code>tfc.run()</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dictionary with two keys.'job_ids' - a list of training job ids
and 'docker_image'- Docker image generated for the training job.
</td>
</tr>

</table>

