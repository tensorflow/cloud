description: Represents the configuration or type of machine to be used.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.MachineConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="validate"/>
</div>

# tfc.MachineConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/core/machine_config.py#L58-L93">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents the configuration or type of machine to be used.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.MachineConfig(
    cpu_cores=8, memory=30, accelerator_type=&#x27;auto&#x27;, accelerator_count=1
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cpu_cores`
</td>
<td>
Number of virtual CPU cores. Defaults to 8.
</td>
</tr><tr>
<td>
`memory`
</td>
<td>
Amount of memory in GB. Defaults to 30GB.
</td>
</tr><tr>
<td>
`accelerator_type`
</td>
<td>
Type of the accelerator to be used
('K80', 'P100', 'V100', 'P4', 'T4', 'TPU_V2', 'TPU_V3') or 'CPU'
for no accelerator. Defaults to 'auto', which maps to a standard
gpu config such as 'P100'.
</td>
</tr><tr>
<td>
`accelerator_count`
</td>
<td>
Number of accelerators. Defaults to 1.
</td>
</tr>
</table>



## Methods

<h3 id="validate"><code>validate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/core/machine_config.py#L87-L93">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate()
</code></pre>

Checks that the machine configuration created is valid for GCP.




