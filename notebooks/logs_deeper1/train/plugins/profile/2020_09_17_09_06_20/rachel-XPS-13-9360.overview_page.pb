�	�K��τ9@�K��τ9@!�K��τ9@	cn\B��?cn\B��?!cn\B��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�K��τ9@:�6��?A�L1Aw7@Y��^(`;�?*	X9��6X@2F
Iterator::ModelU�M�Mӧ?!�\NҲH@)�4~�$�?1�j�uf?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]��Nw�?!������>@)�J�8��?1Z�d�9@:Preprocessing2U
Iterator::Model::ParallelMapV2����?!�N���0@)����?1�N���0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate������?!����,@)s* ���?1u��%�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceKi��u?!y����@)Ki��u?1y����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipC� �é?!,��-M�I@)�[z4u?1w:�XTa@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor䞮�Xls?!B�$n�@)䞮�Xls?1B�$n�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapٲ|]��?!8xp�/@)eo)狽W?11,�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9cn\B��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:�6��?:�6��?!:�6��?      ��!       "      ��!       *      ��!       2	�L1Aw7@�L1Aw7@!�L1Aw7@:      ��!       B      ��!       J	��^(`;�?��^(`;�?!��^(`;�?R      ��!       Z	��^(`;�?��^(`;�?!��^(`;�?JCPU_ONLYYcn\B��?b Y      Y@q�yP���?"�
both�Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 