dataset = tfds.load('wikipedia/20220620.en', split='train', shuffle_files=True)

def ul2_objective(dataset: tf.data.Dataset,
    sequence_length: seqio.preprocessors.SequenceLengthType, 
    output_features: seqio.preprocessors.OutputFeaturesType, 
    use_prefix_lm_task: bool = False,
    rates: Optional[Sequence[float]] = None,
    mean_noise_span_lengths: Sequence[float] = (3.0,),
    noise_densities: Sequence[float] = (0.15,), 
    shard_ds: bool = True, 
    optional_task_prefixes: Optional[Sequence[str]] = None, 
    input_feature_key: str = "inputs", 
    merge_examples_to_reduce_padding: bool = True, 
    reserved_for_packing: bool = None, 
    seed: int = 7) -> tf.data.Dataset:
    
    """
    UL2-like pre-training objectives. This preprocessor amounts to calling the ‘span_corruption‘ function several times with different values of ‘noise_density‘ and ‘mean_noise_span_length‘. 
    We either shard or copy the dataset, then apply each function to each shard. Add S-denoising (prefixLM) using use_prefix_lm_task. 
    
    Args: 
    dataset: A tf.data.Dataset with dictionaries containing the key ‘input_feature_key‘. 
    sequence_length: dict mapping of feature key to int length for that feature. 
    output_features: mapping of keys to features. 
    use_prefix_lm_task: <bool> If True, include PrefixLM in the task mix. 
    rates: <Optional<List<float>> List of rates per task. If None, tasks are sampled uniformly. 
    mean_noise_span_lengths: List of mean number of tokens per masked span per example. 
    noise_densities: List of what fraction of the tokens to mask. 
    shard_ds: <bool> If True, shard dataset per objective. 
    optional_task_prefixes: <Optional<list<str>> Strings to prepend for each orruption scheme. 
    NOTE: If including prefixLM task, it must be the last prefix. 
    input_feature_key: which feature to use from the dataset as the input text tokens. 
    merge_examples_to_reduce_padding: if True, combines multiple input examples to reduce padding. reserved_for_packing: if specified, reduces the desired inputs length by the specified amount to enable multiple examples to be packed together downstream. 
    seed: tf.int64 for controlling the random choice of spans. Returns: a dataset 
    """

    if optional_task_prefixes: # Ensure each task has a prefix. 
        num_tasks = len(noise_densities) + int(use_prefix_lm_task) 
        valid_number_of_prefixes = num_tasks == len(optional_task_prefixes) 
        if not valid_number_of_prefixes: 
            raise ValueError("Number of task prefixes must match number of tasks.") 
    inputs_length = sequence_length[input_feature_key] 
    input_lengths, targets_lengths = [], [] 
    sequence_lengths = {x: y for x, y in sequence_length.items()} 
    if reserved_for_packing: 
        inputs_length -= reserved_for_packing 
        for x, y in sequence_length.items(): 
            sequence_lengths[x] = y - reserved_for_packing 
    hyperparams = list(zip(mean_noise_span_lengths, noise_densities)) 
    for mean_noise_span_length, noise_density in hyperparams: 
        input_length, targets_length = t5.data.preprocessors.random_spans_helper(
            extra_tokens_per_span_inputs=1, 
            extra_tokens_per_span_targets=1, 
            inputs_length=inputs_length, 
            mean_noise_span_length=mean_noise_span_length, 
            noise_density=noise_density) 
        input_lengths.append(input_length) 
        targets_lengths.append(targets_length)

        if sequence_length["targets"] < targets_length: 
            upper_bound = max(targets_lengths) 
            raise ValueError(f"Targets length {sequence_length['targets']} is too small for the given noise_density and mean_noise_span_length. Please increase the targets length to at least {upper_bound}.")
            #raise ValueError("f’Expected max targets length for span corruption ({upper_bound}) is ’ f’greater than configured targets length ’ f"({sequence_length[’targets’]})")

    ds = dataset 
    ds = t5.data.preprocessors.select_random_chunk(
        ds, 
        output_features=output_features, 
        feature_key="targets", 
        max_length=65536) 
    if merge_examples_to_reduce_padding: 
        ds = t5.data.preprocessors.reduce_concat_tokens(
            ds, 
            feature_key="targets", 
            batch_size=128) 
    num_shards = len(input_lengths) + int(use_prefix_lm_task) 
    if shard_ds: 
        ds_shards = [ds.shard(num_shards, i) for i in range(num_shards)] 
    else: 
         ds_shards = [ds for _ in range(num_shards)] 
    processed_ds = [] 
    hyperparams = zip(input_lengths, hyperparams, range(num_shards)) 
    for input_length, (noise_span_length, noise_density), i in hyperparams: 
        ds = ds_shards[i] 
        ds = t5.data.preprocessors.split_tokens(
            ds,
            feature_key="targets", 
            min_tokens_per_segment=None, 
            max_tokens_per_segment=input_length) 
        ds = t5.data.preprocessors.denoise(
            ds, 
            output_features, 
            inputs_fn=t5.data.preprocessors.noise_span_to_unique_sentinel, 
            targets_fn=t5.data.preprocessors.nonnoise_span_to_unique_sentinel, 
            noise_density=noise_density, 
            noise_mask_fn=functools.partial(
                t5.data.preprocessors.random_spans_noise_mask, 
                mean_noise_span_length=noise_span_length), 
                input_feature_key=input_feature_key) 
        if optional_task_prefixes: 
            ds = prepend_prompt(
                ds, 
                output_features, 
                prompt_mode=optional_task_prefixes[i], 
                mode=optional_task_prefixes[i]) 
        processed_ds.append(ds) 
    if use_prefix_lm_task: 
        ds = ds_shards[-1] 
        ds = t5.data.preprocessors.prefix_lm(ds, sequence_lengths, output_features) 
        if optional_task_prefixes: 
            ds = prepend_prompt(
                ds, 
                output_features, 
                prompt_mode=optional_task_prefixes[-1], 
                mode=optional_task_prefixes[-1]) 
        processed_ds.append(ds) 
    ds = tf.data.experimental.sample_from_datasets(processed_ds, rates, seed) 
    return ds

sequence_length = {
    "inputs": 512,
    "targets": 512,
}

output_features = {
    "inputs":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False),
    "targets":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=False)
}

ul2_data = ul2_objective(
    dataset,
    sequence_length, 
    output_features, 
    use_prefix_lm_task=False,
    rates=None,
    mean_noise_span_lengths=(3.0,),
    noise_densities=(0.15,), 
    shard_ds=True, 
    optional_task_prefixes=None, 
    input_feature_key="text", 
    merge_examples_to_reduce_padding=True, 
    reserved_for_packing=None, 
    seed=7)


def prepend_prompt(dataset: tf.data.Dataset,
                   output_features: seqio.preprocessors.OutputFeaturesType,
                   sequence_length: Optional[
                       seqio.preprocessors.SequenceLengthType] = None,
                   prompt_mode: str = "",
                   key: str = "inputs",
                   mode: str = "") -> tf.data.Dataset:
    """Prepends a prompt at the beginning of an input sequence."""
    del sequence_length
    if prompt_mode and mode:
        # output_features may not have inputs key
        out_keys = list(output_features.keys())
        prompt_tokens = output_features[out_keys[0]
                                        ].vocabulary.encode_tf(prompt_mode)

        def add_to_inputs(x):
            x[key] = tf.concat([prompt_tokens, x[key]], axis=0)
            return x

        dataset = dataset.map(add_to_inputs)
    return dataset