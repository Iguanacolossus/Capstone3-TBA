	U�G��@U�G��@!U�G��@	��)��?��)��?!��)��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$U�G��@RG�����?A}�.PR@@Y�eS��?*	A`��"C[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ0c
֨?!�:M��=F@)�(B�v��?1����C@:Preprocessing2F
Iterator::Model����=��?!|)�Q>@)�������?1Yӫ�pT3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�j����?!�X��1@)��?�0�?1�5/{�,@:Preprocessing2U
Iterator::Model::ParallelMapV2�c�~��?!Gj�b>�%@)�c�~��?1Gj�b>�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip
��t�?!ή�kQ@)�K��$wx?1 �8���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9CqǛ�v?!z�cҕ@)9CqǛ�v?1z�cҕ@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�΢w*�n?!��9|c�@)�΢w*�n?1��9|c�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(�hr1�?!PCn&�3@)D��<��_?1�����I�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��)��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	RG�����?RG�����?!RG�����?      ��!       "      ��!       *      ��!       2	}�.PR@@}�.PR@@!}�.PR@@:      ��!       B      ��!       J	�eS��?�eS��?!�eS��?R      ��!       Z	�eS��?�eS��?!�eS��?JCPU_ONLYY��)��?b 