	�~m���@�~m���@!�~m���@	�����?�����?!�����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�~m���@�ю~��?AI�5C@Y�Wya�?*�O��n�V@)      =2F
Iterator::Modelx����գ?!x��0&VE@):�}�kϜ?1]
�b�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�<c_��?!�����9B@)�|�R��?1KX�]
�>@:Preprocessing2U
Iterator::Model::ParallelMapV2m��p��?!%5؇�]'@)m��p��?1%5؇�]'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenateg��j+��?!��O�:�)@)�<Fy��?1�HE)-"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�u��ť�?!�m)�٩L@)�0e��v?1���4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��'�H0u?!4�����@)��'�H0u?14�����@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�a��Al?!�@Ed@)�a��Al?1�@Ed@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��.ޏ�?!3l���-@)TUh ��\?1d4��h��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ю~��?�ю~��?!�ю~��?      ��!       "      ��!       *      ��!       2	I�5C@I�5C@!I�5C@:      ��!       B      ��!       J	�Wya�?�Wya�?!�Wya�?R      ��!       Z	�Wya�?�Wya�?!�Wya�?JCPU_ONLYY�����?b 