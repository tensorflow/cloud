description: Core module in tensorflow_cloud.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="COMMON_MACHINE_CONFIGS"/>
<meta itemprop="property" content="__version__"/>
</div>

# Module: tfc

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Core module in tensorflow_cloud.



## Classes

[`class AcceleratorType`](./tfc/AcceleratorType.md): Types of accelerators.

[`class CloudOracle`](./tfc/CloudOracle.md): KerasTuner Oracle interface for Vizier Service backend.

[`class CloudTuner`](./tfc/CloudTuner.md): KerasTuner interface implementation backed by Vizier Service.

[`class DockerConfig`](./tfc/DockerConfig.md): Represents Docker-related configuration for the `run` API.

[`class MachineConfig`](./tfc/MachineConfig.md): Represents the configuration or type of machine to be used.

## Functions

[`remote(...)`](./tfc/remote.md): True when code is run in a remote cloud environment by TF Cloud.

[`run(...)`](./tfc/run.md): Runs your Tensorflow code in Google Cloud Platform.

[`run_cloudtuner(...)`](./tfc/run_cloudtuner.md): A wrapper for tfc.run that allows for running concurrent CloudTuner jobs.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
COMMON_MACHINE_CONFIGS<a id="COMMON_MACHINE_CONFIGS"></a>
</td>
<td>
```
{
 'CPU': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81ff0bb50>,
 'K80_1X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81ffeb610>,
 'K80_4X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81ffeb6d0>,
 'K80_8X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f2b0>,
 'P100_1X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f310>,
 'P100_4X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f370>,
 'P4_1X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f3d0>,
 'P4_4X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f430>,
 'T4_1X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f550>,
 'T4_4X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f5b0>,
 'TPU': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f610>,
 'V100_1X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f490>,
 'V100_4X': <tensorflow_cloud.core.machine_config.MachineConfig object at 0x7fa81fa5f4f0>
}
```
</td>
</tr><tr>
<td>
__version__<a id="__version__"></a>
</td>
<td>
`'0.1.15.dev'`
</td>
</tr>
</table>

