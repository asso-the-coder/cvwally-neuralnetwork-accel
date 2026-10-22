| config | git_commit | git_dirty |
| --- | --- | --- |
| B0 | a867093e6a57d70b751f70fb73218418dcd1ae7b | False |

| scope | metric | value |
| --- | --- | --- |
| inference / validation | instret_invoke | 23764202246 |
| inference / validation | total_macs | 385187552 |
| inference / validation | mac_per_instr | 0.016208730594557824 |
| inference / validation | instr_per_mac | 61.69514596878769 |
| inference / validation | histogram_total | 23789945005 |
| inference / validation | instret_end | 23789880049 |
| inference / validation | post_counter_tail | 64951 |
| inference / validation | csr_serialization_retries | 5 |
| inference / validation | whole_program_instret | 23789945000 |
| inference / validation | text_size | 5682584 |
| inference / validation | linked_text_size | 92168 |
| inference / validation | data_size | 248 |
| inference / validation | bss_size | 71565920 |
| inference / validation | profiled_instret_invoke | 23764204486 |
| inference / validation | per_op_instret_sum | 23764201251 |
| inference / validation | profiled_minus_op_sum | 3235 |
| inference / validation | profiled_minus_headline | 2240 |
| inference / validation | op_sum_minus_headline | -995 |
| inference / validation | profiler_overhead_fraction | 9.425942334660268e-08 |
| inference / validation | spike_histogram_wall_seconds | 557.4451118099969 |
| inference / validation | spike_profiled_wall_seconds | 105.08836975599115 |
| whole program | dyn_instr | 23789945005 |
| whole program | dyn_loads | 4178143819 |
| whole program | dyn_stores | 99359206 |
| whole program | dyn_memory_instructions | 4277503025 |
| whole program | mem_instr_share | 0.17980298080138415 |
| whole program | loads_stores_per_mac | 11.104987694410228 |
| whole program | mul_share | 0.11192171291023965 |
| whole program | branch_share | 0.315379320440762 |
| whole program | custom_count | 0 |
| whole program | compressed_count | 6797514675 |
| whole program | compressed_fraction | 0.2857305753994533 |
| kernel functions | dyn_instr | 23763803918 |
| kernel functions | dyn_loads | 4171443152 |
| kernel functions | dyn_stores | 98087562 |
| kernel functions | dyn_memory_instructions | 4269530714 |
| kernel functions | mem_instr_share | 0.17966528964523332 |
| kernel functions | loads_stores_per_mac | 11.084290475721293 |
| kernel functions | mul_share | 0.11203481703463196 |
| kernel functions | branch_share | 0.3154016624974241 |
| kernel functions | custom_count | 0 |
| kernel functions | compressed_count | 6783693509 |
| kernel functions | compressed_fraction | 0.28546328409407806 |

| type | conv_kind | count | macs | instret | share | instr_per_mac |
| --- | --- | --- | --- | --- | --- | --- |
| CONV_2D | 1x1 | 32 | 338537472 | 20599031513 | 0.8668092182566148 | 60.847123927835085 |
| DEPTHWISE_CONV_2D | - | 16 | 34532064 | 2350807939 | 0.0989222231438427 | 68.07609122350752 |
| CONV_2D | kxk | 1 | 10838016 | 774932391 | 0.03260922920674787 | 71.50131453948767 |
| ADD | - | 9 | 0 | 25136222 | 0.0010577346283486277 |  |
| FULLY_CONNECTED | - | 1 | 1280000 | 11583035 | 0.00048741522178130615 | 9.04924609375 |
| AVERAGE_POOL_2D | - | 1 | 0 | 1345329 | 5.6611573124299705e-05 |  |
| QUANTIZE | - | 2 | 0 | 909527 | 3.8272983239805976e-05 |  |
| SOFTMAX | - | 1 | 0 | 447426 | 1.8827728917397095e-05 |  |
| RESHAPE | - | 1 | 0 | 7869 | 3.311282733926901e-07 |  |

| scope | category | count | share |
| --- | --- | --- | --- |
| whole program | load | 4178143819 | 0.17562645975523977 |
| whole program | store | 99359206 | 0.004176521046144386 |
| whole program | mul | 2662611395 | 0.11192171291023965 |
| whole program | div | 4520627 | 0.00019002259143726004 |
| whole program | branch | 7502856689 | 0.315379320440762 |
| whole program | jump | 28774251 | 0.001209513136493272 |
| whole program | alu | 9313679008 | 0.391496449699338 |
| whole program | csr/system | 10 | 4.2034565434675327e-10 |
| whole program | custom | 0 | 0.0 |
| whole program | other | 0 | 0.0 |
| kernel functions | load | 4171443152 | 0.17553768607055042 |
| kernel functions | store | 98087562 | 0.004127603574682887 |
| kernel functions | mul | 2662373424 | 0.11203481703463196 |
| kernel functions | div | 4432547 | 0.00018652514619692462 |
| kernel functions | branch | 7495143263 | 0.3154016624974241 |
| kernel functions | jump | 27567833 | 0.0011600766062170131 |
| kernel functions | alu | 9304756137 | 0.3915516290702967 |
| kernel functions | csr/system | 0 | 0.0 |
| kernel functions | custom | 0 | 0.0 |
| kernel functions | other | 0 | 0.0 |

| name | dyn_instr | share | size_bytes |
| --- | --- | --- | --- |
| tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 21198383903 | 0.8910648552800217 | 1070 |
| tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 2259689543 | 0.09498506795728509 | 996 |
| tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 266702197 | 0.01121071095136817 | 196 |
| tflite::micro::EvalAddQuantized(tflite::micro::TfLiteContext*, tflite::micro::TfLiteNode*, tflite::micro::TfLiteAddParams*, tflite::micro::OpDataAdd const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor*) | 25125926 | 0.00105615738055381 | 4224 |
| void tflite::micro::reference_integer_ops::FullyConnected<signed char, signed char, signed char, int>(tflite::micro::FullyConnectedParams const&, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 11543106 | 0.0004852094444763934 | 380 |
| htif_putchar | 10566150 | 0.00044414352356759475 | 32 |
| __muldf3 | 3560316 | 0.00014965633587012153 | 1188 |
| _start | 2097471 | 8.816628199683389e-05 | 68 |
| __divdf3 | 1865996 | 7.843633096284242e-05 | 1088 |
| tflite::micro::internal::InitializeTfLiteTensorFromFlatbuffer(tflite::micro::IPersistentBufferAllocator*, tflite::micro::INonPersistentBufferAllocator*, bool, tflite::Tensor const&, flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>, unsigned int> const*, tflite::micro::TfLiteTensor*) | 1846024 | 7.759681662198109e-05 | 806 |

| rank | type | conv_kind | instret | share |
| --- | --- | --- | --- | --- |
| 1 | CONV_2D | 1x1 | 20599031513 | 0.8668092182566148 |
| 2 | DEPTHWISE_CONV_2D | - | 2350807939 | 0.0989222231438427 |
| 3 | CONV_2D | kxk | 774932391 | 0.03260922920674787 |

| rank | name | dyn_instr | share |
| --- | --- | --- | --- |
| 1 | tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 21198383903 | 0.8910648552800217 |
| 2 | tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 2259689543 | 0.09498506795728509 |
| 3 | tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 266702197 | 0.01121071095136817 |
