description: KerasTuner Oracle interface for Vizier Service backend.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.CloudOracle" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_trial"/>
<meta itemprop="property" content="end_trial"/>
<meta itemprop="property" content="get_best_trials"/>
<meta itemprop="property" content="get_space"/>
<meta itemprop="property" content="get_state"/>
<meta itemprop="property" content="get_trial"/>
<meta itemprop="property" content="reload"/>
<meta itemprop="property" content="remaining_trials"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="set_state"/>
<meta itemprop="property" content="update_space"/>
<meta itemprop="property" content="update_trial"/>
</div>

# tfc.CloudOracle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L56-L402">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



KerasTuner Oracle interface for Vizier Service backend.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.CloudOracle(
    project_id: Text,
    region: Text,
    objective: Union[Text, oracle_module.Objective] = None,
    hyperparameters: hp_module.HyperParameters = None,
    study_config: Optional[Dict[Text, Any]] = None,
    max_trials: int = None,
    study_id: Optional[Text] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is an implementation of kerasTuner Oracel that uses Google Cloud's
Vizier Service.
Each Oracle class implements a particular hyperparameter tuning algorithm.
An Oracle is passed as an argument to a Tuner. The Oracle tells the Tuner
which hyperparameters should be tried next.

To learn more about KerasTuner Oracles please refer to:
https://keras-team.github.io/keras-tuner/documentation/oracles/

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`project_id`
</td>
<td>
A GCP project id.
</td>
</tr><tr>
<td>
`region`
</td>
<td>
A GCP region. e.g. 'us-central1'.
</td>
</tr><tr>
<td>
`objective`
</td>
<td>
If a string, the direction of the optimization (min or
max) will be inferred.
</td>
</tr><tr>
<td>
`hyperparameters`
</td>
<td>
Mandatory and must include definitions for all
hyperparameters used during the search. Can be used to override
(or register in advance) hyperparameters in the search space.
</td>
</tr><tr>
<td>
`study_config`
</td>
<td>
Study configuration for Vizier service.
</td>
</tr><tr>
<td>
`max_trials`
</td>
<td>
Total number of trials (model configurations) to test at
most. If None, it continues the search until it reaches the
Vizier trial limit for each study. Users may stop the search
externally (e.g. by killing the job). Note that the Oracle may
interrupt the search before `max_trials` models have been
tested.
</td>
</tr><tr>
<td>
`study_id`
</td>
<td>
An identifier of the study. If not supplied,
system-determined unique ID is given.
The full study name will be
`projects/{project_id}/locations/{region}/studies/{study_id}`,
and the full trial name will be
`{study name}/trials/{trial_id}`.
</td>
</tr>
</table>



## Methods

<h3 id="create_trial"><code>create_trial</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L178-L249">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_trial(
    tuner_id: Text
) -> trial_module.Trial
</code></pre>

Create a new `Trial` to be run by the `Tuner`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tuner_id`
</td>
<td>
An ID that identifies the `Tuner` requesting a `Trial`.
`Tuners` that should run the same trial (for instance, when
running a multi-worker model) should have the same ID. If
multiple suggestTrialsRequests have the same tuner_id, the
service will return the identical suggested trial if the trial
is PENDING, and provide a new trial if the last suggested trial
was completed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Trial` object containing a set of hyperparameter values to run
in a `Tuner`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`SuggestionInactiveError`
</td>
<td>
Indicates that a suggestion was requested
from an inactive study.
</td>
</tr>
</table>



<h3 id="end_trial"><code>end_trial</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L307-L347">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>end_trial(
    trial_id: Text,
    status: Text = &#x27;COMPLETED&#x27;
)
</code></pre>

Record the measured objective for a set of parameter values.


<h3 id="get_best_trials"><code>get_best_trials</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L349-L389">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_best_trials(
    num_trials: int = 1
) -> List[trial_module.Trial]
</code></pre>

Returns the trials with the best objective values found so far.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`num_trials`
</td>
<td>
positive int, number of trials to return.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of KerasTuner Trials.
</td>
</tr>

</table>



<h3 id="get_space"><code>get_space</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_space()
</code></pre>

Returns the `HyperParameters` search space.


<h3 id="get_state"><code>get_state</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_state()
</code></pre>

Returns the current state of this object.

This method is called during `save`.

<h3 id="get_trial"><code>get_trial</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_trial(
    trial_id
)
</code></pre>

Returns the `Trial` specified by `trial_id`.


<h3 id="reload"><code>reload</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L391-L393">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reload()
</code></pre>

Reloads this object using `set_state`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`fname`
</td>
<td>
The file name to restore from.
</td>
</tr>
</table>



<h3 id="remaining_trials"><code>remaining_trials</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remaining_trials()
</code></pre>




<h3 id="save"><code>save</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L395-L397">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save()
</code></pre>

Saves this object using `get_state`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`fname`
</td>
<td>
The file name to save to.
</td>
</tr>
</table>



<h3 id="set_state"><code>set_state</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_state(
    state
)
</code></pre>

Sets the current state of this object.

This method is called during `reload`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`state`
</td>
<td>
Dict. The state to restore for this object.
</td>
</tr>
</table>



<h3 id="update_space"><code>update_space</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_space(
    hyperparameters
)
</code></pre>

Add new hyperparameters to the tracking space.

Already recorded parameters get ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`hyperparameters`
</td>
<td>
An updated HyperParameters object.
</td>
</tr>
</table>



<h3 id="update_trial"><code>update_trial</code></h3>

<a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L251-L305">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_trial(
    trial_id: Text,
    metrics: Mapping[Text, Union[int, float]],
    step: int = 0
)
</code></pre>

Used by a worker to report the status of a trial.




