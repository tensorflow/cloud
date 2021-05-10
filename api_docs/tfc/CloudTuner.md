description: KerasTuner interface implementation backed by Vizier Service.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.CloudTuner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_best_hyperparameters"/>
<meta itemprop="property" content="get_best_models"/>
<meta itemprop="property" content="get_state"/>
<meta itemprop="property" content="get_trial_dir"/>
<meta itemprop="property" content="load_model"/>
<meta itemprop="property" content="on_batch_begin"/>
<meta itemprop="property" content="on_batch_end"/>
<meta itemprop="property" content="on_epoch_begin"/>
<meta itemprop="property" content="on_epoch_end"/>
<meta itemprop="property" content="on_search_begin"/>
<meta itemprop="property" content="on_search_end"/>
<meta itemprop="property" content="on_trial_begin"/>
<meta itemprop="property" content="on_trial_end"/>
<meta itemprop="property" content="reload"/>
<meta itemprop="property" content="results_summary"/>
<meta itemprop="property" content="run_trial"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="save_model"/>
<meta itemprop="property" content="search"/>
<meta itemprop="property" content="search_space_summary"/>
<meta itemprop="property" content="set_state"/>
</div>

# tfc.CloudTuner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/tuner/tuner.py#L405-L470">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



KerasTuner interface implementation backed by Vizier Service.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.CloudTuner(
    hypermodel: Union[hypermodel_module.HyperModel, Callable[[hp_module.HyperParameters],
        tf.keras.Model]],
    project_id: Text,
    region: Text,
    objective: Union[Text, oracle_module.Objective] = None,
    hyperparameters: hp_module.HyperParameters = None,
    study_config: Optional[Dict[Text, Any]] = None,
    max_trials: int = None,
    study_id: Optional[Text] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

CloudTuner is a implmentation of KerasTuner that usese Google Cloud Vizier
Service as it's Oracle. To learn more about KerasTuner and Oracles please
refer to:
  https://keras-team.github.io/keras-tuner/
  https://keras-team.github.io/keras-tuner/documentation/oracles/

  Example:
    ```
    >>> tuner = CloudTuner(
          build_model,
          project_id="MY_PROJECT_ID",
          region='us-central1',
          objective='accuracy',
          hyperparameters=HPS,
          max_trials=5,
          directory='tmp/MY_JOB')
    ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`hypermodel`
</td>
<td>
Instance of HyperModel class (or callable that takes
hyperparameters and returns a Model instance).
</td>
</tr><tr>
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
Name of model metric to minimize or maximize, e.g.
"val_accuracy".
</td>
</tr><tr>
<td>
`hyperparameters`
</td>
<td>
Can be used to override (or register in advance)
hyperparameters in the search space.
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
most. Note that the oracle may interrupt the search before
`max_trials` models have been tested if the search space has
been exhausted.
</td>
</tr><tr>
<td>
`study_id`
</td>
<td>
An identifier of the study. The full study name will be
projects/{project_id}/locations/{region}/studies/{study_id}.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments relevant to all `Tuner` subclasses.
Please see the docstring for `Tuner`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`project_dir`
</td>
<td>

</td>
</tr><tr>
<td>
`remaining_trials`
</td>
<td>
Returns the number of trials remaining.

Will return `None` if `max_trials` is not set.
</td>
</tr>
</table>



## Methods

<h3 id="get_best_hyperparameters"><code>get_best_hyperparameters</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_best_hyperparameters(
    num_trials=1
)
</code></pre>

Returns the best hyperparameters, as determined by the objective.

This method can be used to reinstantiate the (untrained) best model
found during the search process.

#### Example:



```python
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`num_trials`
</td>
<td>
(int, optional). Number of `HyperParameters` objects to
return. `HyperParameters` will be returned in sorted order based on
trial performance.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of `HyperParameter` objects.
</td>
</tr>

</table>



<h3 id="get_best_models"><code>get_best_models</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_best_models(
    num_models=1
)
</code></pre>

Returns the best model(s), as determined by the tuner's objective.

The models are loaded with the weights corresponding to
their best checkpoint (at the end of the best epoch of best trial).

This method is only a convenience shortcut. For best performance, It is
recommended to retrain your Model on the full dataset using the best
hyperparameters found during `search`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
num_models (int, optional): Number of best models to return.
Models will be returned in sorted order. Defaults to 1.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of trained model instances.
</td>
</tr>

</table>



<h3 id="get_state"><code>get_state</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_state()
</code></pre>

Returns the current state of this object.

This method is called during `save`.

<h3 id="get_trial_dir"><code>get_trial_dir</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_trial_dir(
    trial_id
)
</code></pre>




<h3 id="load_model"><code>load_model</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_model(
    trial
)
</code></pre>

Loads a Model from a given trial.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance. For models that report intermediate
results to the `Oracle`, generally `load_model` should load the
best reported `step` by relying of `trial.best_step`
</td>
</tr>
</table>



<h3 id="on_batch_begin"><code>on_batch_begin</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_batch_begin(
    trial, model, batch, logs
)
</code></pre>

A hook called at the start of every batch.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
A Keras `Model`.
</td>
</tr><tr>
<td>
`batch`
</td>
<td>
The current batch number within the
curent epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Additional metrics.
</td>
</tr>
</table>



<h3 id="on_batch_end"><code>on_batch_end</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_batch_end(
    trial, model, batch, logs=None
)
</code></pre>

A hook called at the end of every batch.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
A Keras `Model`.
</td>
</tr><tr>
<td>
`batch`
</td>
<td>
The current batch number within the
curent epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Additional metrics.
</td>
</tr>
</table>



<h3 id="on_epoch_begin"><code>on_epoch_begin</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_begin(
    trial, model, epoch, logs=None
)
</code></pre>

A hook called at the start of every epoch.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
A Keras `Model`.
</td>
</tr><tr>
<td>
`epoch`
</td>
<td>
The current epoch number.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Additional metrics.
</td>
</tr>
</table>



<h3 id="on_epoch_end"><code>on_epoch_end</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_end(
    trial, model, epoch, logs=None
)
</code></pre>

A hook called at the end of every epoch.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
A Keras `Model`.
</td>
</tr><tr>
<td>
`epoch`
</td>
<td>
The current epoch number.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Metrics for this epoch. This should include
the value of the objective for this epoch.
</td>
</tr>
</table>



<h3 id="on_search_begin"><code>on_search_begin</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_search_begin()
</code></pre>

A hook called at the beginning of `search`.


<h3 id="on_search_end"><code>on_search_end</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_search_end()
</code></pre>

A hook called at the end of `search`.


<h3 id="on_trial_begin"><code>on_trial_begin</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_trial_begin(
    trial
)
</code></pre>

A hook called before starting each trial.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance.
</td>
</tr>
</table>



<h3 id="on_trial_end"><code>on_trial_end</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_trial_end(
    trial
)
</code></pre>

A hook called after each trial is run.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance.
</td>
</tr>
</table>



<h3 id="reload"><code>reload</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reload()
</code></pre>

Reloads this object from its project directory.


<h3 id="results_summary"><code>results_summary</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>results_summary(
    num_trials=10
)
</code></pre>

Display tuning results summary.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
num_trials (int, optional): Number of trials to display.
Defaults to 10.
</td>
</tr>

</table>



<h3 id="run_trial"><code>run_trial</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_trial(
    trial, *fit_args, **fit_kwargs
)
</code></pre>

Evaluates a set of hyperparameter values.

This method is called during `search` to evaluate a set of
hyperparameters.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial`
</td>
<td>
A `Trial` instance that contains the information
needed to run this trial. `Hyperparameters` can be accessed
via `trial.hyperparameters`.
</td>
</tr><tr>
<td>
`*fit_args`
</td>
<td>
Positional arguments passed by `search`.
</td>
</tr><tr>
<td>
`*fit_kwargs`
</td>
<td>
Keyword arguments passed by `search`.
</td>
</tr>
</table>



<h3 id="save"><code>save</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save()
</code></pre>

Saves this object to its project directory.


<h3 id="save_model"><code>save_model</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_model(
    trial_id, model, step=0
)
</code></pre>

Saves a Model for a given trial.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`trial_id`
</td>
<td>
The ID of the `Trial` that corresponds to this Model.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
The trained model.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
For models that report intermediate results to the `Oracle`,
the step that this saved file should correspond to. For example,
for Keras models this is the number of epochs trained.
</td>
</tr>
</table>



<h3 id="search"><code>search</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>search(
    *fit_args, **fit_kwargs
)
</code></pre>

Performs a search for best hyperparameter configuations.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`*fit_args`
</td>
<td>
Positional arguments that should be passed to
`run_trial`, for example the training and validation data.
</td>
</tr><tr>
<td>
`*fit_kwargs`
</td>
<td>
Keyword arguments that should be passed to
`run_trial`, for example the training and validation data.
</td>
</tr>
</table>



<h3 id="search_space_summary"><code>search_space_summary</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>search_space_summary(
    extended=(False)
)
</code></pre>

Print search space summary.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`extended`
</td>
<td>
Bool, optional. Display extended summary.
Defaults to False.
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





