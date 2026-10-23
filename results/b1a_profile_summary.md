| config | git_commit | git_dirty |
| --- | --- | --- |
| B1A | 1a9cea939c95795a4fde42707d1bb8643135254d | True |

| scope | metric | value |
| --- | --- | --- |
| inference / validation | instret_invoke | 4873990558 |
| inference / validation | total_macs | 385187552 |
| inference / validation | mac_per_instr | 0.07902919536185117 |
| inference / validation | instr_per_mac | 12.65355158206151 |
| inference / validation | histogram_total | 4899450005 |
| inference / validation | instret_end | 4899390048 |
| inference / validation | post_counter_tail | 59952 |
| inference / validation | csr_serialization_retries | 5 |
| inference / validation | whole_program_instret | 4899450000 |
| inference / validation | text_size | 5672928 |
| inference / validation | linked_text_size | 84512 |
| inference / validation | data_size | 248 |
| inference / validation | bss_size | 71565920 |
| inference / validation | profiled_instret_invoke | 4873992734 |
| inference / validation | per_op_instret_sum | 4873989564 |
| inference / validation | profiled_minus_op_sum | 3170 |
| inference / validation | profiled_minus_headline | 2176 |
| inference / validation | op_sum_minus_headline | -994 |
| inference / validation | profiler_overhead_fraction | 4.4645141883346257e-07 |
| inference / validation | spike_histogram_wall_seconds | 125.29076014601742 |
| inference / validation | spike_profiled_wall_seconds | 26.496308089990634 |
| whole program | dyn_instr | 4899450005 |
| whole program | dyn_loads | 1187998994 |
| whole program | dyn_stores | 65242903 |
| whole program | dyn_memory_instructions | 1253241897 |
| whole program | mem_instr_share | 0.2557923635757153 |
| whole program | loads_stores_per_mac | 3.2535887790060256 |
| whole program | mul_share | 0.12500804322423126 |
| whole program | branch_share | 0.12599632068293756 |
| whole program | custom_count | 0 |
| whole program | compressed_count | 1571058987 |
| whole program | compressed_fraction | 0.32066027521389107 |
| kernel functions | dyn_instr | 4873593834 |
| kernel functions | dyn_loads | 1181310542 |
| kernel functions | dyn_stores | 63972926 |
| kernel functions | dyn_memory_instructions | 1245283468 |
| kernel functions | mem_instr_share | 0.25551646493649927 |
| kernel functions | loads_stores_per_mac | 3.2329275999033324 |
| kernel functions | mul_share | 0.12562242727919537 |
| kernel functions | branch_share | 0.1250977600444822 |
| kernel functions | custom_count | 0 |
| kernel functions | compressed_count | 1557367014 |
| kernel functions | compressed_fraction | 0.3195520732842424 |

| type | conv_kind | count | macs | instret | share | instr_per_mac |
| --- | --- | --- | --- | --- | --- | --- |
| CONV_2D | 1x1 | 32 | 338537472 | 3293275045 | 0.6756832077378309 | 9.72794835840211 |
| DEPTHWISE_CONV_2D | - | 16 | 34532064 | 1297328502 | 0.26617366352438226 | 37.568808571650976 |
| CONV_2D | kxk | 1 | 10838016 | 246970111 | 0.0506710051652695 | 22.78739125315925 |
| ADD | - | 9 | 0 | 23034616 | 0.00472602592107188 |  |
| FULLY_CONNECTED | - | 1 | 1280000 | 11580007 | 0.0023758769518099983 | 9.04688046875 |
| QUANTIZE | - | 2 | 0 | 909513 | 0.00018660532537429097 |  |
| SOFTMAX | - | 1 | 0 | 447400 | 9.179332518881838e-05 |  |
| AVERAGE_POOL_2D | - | 1 | 0 | 436506 | 8.955819670288412e-05 |  |
| RESHAPE | - | 1 | 0 | 7864 | 1.6134615764078405e-06 |  |

| scope | category | count | share |
| --- | --- | --- | --- |
| whole program | load | 1187998994 | 0.24247599073112697 |
| whole program | store | 65242903 | 0.013316372844588299 |
| whole program | mul | 612470658 | 0.12500804322423126 |
| whole program | div | 4520590 | 0.0009226729521449622 |
| whole program | branch | 617312674 | 0.12599632068293756 |
| whole program | jump | 19923325 | 0.004066441126997478 |
| whole program | alu | 2391980851 | 0.48821415639692806 |
| whole program | csr/system | 10 | 2.041045421382966e-09 |
| whole program | custom | 0 | 0.0 |
| whole program | other | 0 | 0.0 |
| kernel functions | load | 1181310542 | 0.2423900272030753 |
| kernel functions | store | 63972926 | 0.01312643773342397 |
| kernel functions | mul | 612232687 | 0.12562242727919537 |
| kernel functions | div | 4432514 | 0.0009094959799639306 |
| kernel functions | branch | 609675672 | 0.1250977600444822 |
| kernel functions | jump | 18716469 | 0.003840383429047157 |
| kernel functions | alu | 2383253024 | 0.48901346833081205 |
| kernel functions | csr/system | 0 | 0.0 |
| kernel functions | custom | 0 | 0.0 |
| kernel functions | other | 0 | 0.0 |

| name | dyn_instr | share | size_bytes |
| --- | --- | --- | --- |
| tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 3377959682 | 0.689456914256236 | 768 |
| tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 1213130138 | 0.24760537137065858 | 632 |
| tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.05030893442089527 | 140 |
| tflite::micro::EvalAddQuantized(tflite::micro::TfLiteContext*, tflite::micro::TfLiteNode*, tflite::micro::TfLiteAddParams*, tflite::micro::OpDataAdd const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor*) | 23025049 | 0.004699517083856844 | 3980 |
| void tflite::micro::reference_integer_ops::FullyConnected<signed char, signed char, signed char, int>(tflite::micro::FullyConnectedParams const&, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 11543100 | 0.0023559991403565715 | 360 |
| htif_putchar | 10553786 | 0.0021540756593555646 | 32 |
| __muldf3 | 3560316 | 0.0007266766670476516 | 1188 |
| _start | 2097471 | 0.0004281033581033551 | 68 |
| __divdf3 | 1865996 | 0.0003808582592118929 | 1088 |
| tflite::micro::internal::InitializeTfLiteTensorFromFlatbuffer(tflite::micro::IPersistentBufferAllocator*, tflite::micro::INonPersistentBufferAllocator*, bool, tflite::Tensor const&, flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>, unsigned int> const*, tflite::micro::TfLiteTensor*) | 1592972 | 0.0003251328206991266 | 776 |

| rank | type | conv_kind | instret | share |
| --- | --- | --- | --- | --- |
| 1 | CONV_2D | 1x1 | 3293275045 | 0.6756832077378309 |
| 2 | DEPTHWISE_CONV_2D | - | 1297328502 | 0.26617366352438226 |
| 3 | CONV_2D | kxk | 246970111 | 0.0506710051652695 |

| rank | name | dyn_instr | share |
| --- | --- | --- | --- |
| 1 | tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 3377959682 | 0.689456914256236 |
| 2 | tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 1213130138 | 0.24760537137065858 |
| 3 | tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.05030893442089527 |
