	�nf���@�nf���@!�nf���@	�;:=�q�?�;:=�q�?!�;:=�q�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�nf���@"�����?A&W��MA@Y
0,�-�?*	X9��v�P@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�2���?!�.g��B@)=G仔��?1���l >@:Preprocessing2F
Iterator::Model�����?!��և�B@)~5��?1������9@:Preprocessing2U
Iterator::Model::ParallelMapV2�
�<�?!��9�~'@)�
�<�?1��9�~'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate8��d�`�?!��@��}-@)��sE)!x?1��r�v!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��9� u?!��B��d@)��9� u?1��B��d@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�B����?!�Df)x*O@)/�r�]�t?1���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�H�+�p?!D4���@)�H�+�p?1D4���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^f�(�?!���N1@)Z.��S\?1���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�;:=�q�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	"�����?"�����?!"�����?      ��!       "      ��!       *      ��!       2	&W��MA@&W��MA@!&W��MA@:      ��!       B      ��!       J	
0,�-�?
0,�-�?!
0,�-�?R      ��!       Z	
0,�-�?
0,�-�?!
0,�-�?JCPU_ONLYY�;:=�q�?b 