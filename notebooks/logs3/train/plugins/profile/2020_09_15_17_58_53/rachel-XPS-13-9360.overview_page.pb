�	 Q���=@ Q���=@! Q���=@	��^i���?��^i���?!��^i���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ Q���=@Qk�w���?Ag{��<@Y�%���?*	    ���@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�v�
�@!u�F��X@)e�I)�@1�����X@:Preprocessing2F
Iterator::Modelۤ���w�?!IB���U�?)4�?O�?1��'=
p�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����a�?!�J�h�s�?);R}�%�?1
e �&�?:Preprocessing2U
Iterator::Model::ParallelMapV2�.���ǅ?!�5�b���?)�.���ǅ?1�5�b���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;���.�@!{=TlT�X@)U.T����?1b�j.K�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��Dׅ|?!���?�u�?)��Dׅ|?1���?�u�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR~R���x?!'�0O�?)R~R���x?1'�0O�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2��n�@!^�M0G�X@)�{�i��c?1�B�~Q��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��^i���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Qk�w���?Qk�w���?!Qk�w���?      ��!       "      ��!       *      ��!       2	g{��<@g{��<@!g{��<@:      ��!       B      ��!       J	�%���?�%���?!�%���?R      ��!       Z	�%���?�%���?!�%���?JCPU_ONLYY��^i���?b Y      Y@q�x*2�9'@"�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb�11.6132% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 