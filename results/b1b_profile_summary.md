| config | git_commit | git_dirty |
| --- | --- | --- |
| B1B | b6e05911dd69a9b7f9e374877ce7c1ae861c92f4 | True |

| scope | metric | value |
| --- | --- | --- |
| inference / validation | instret_invoke | 4065900906 |
| inference / validation | total_macs | 385187552 |
| inference / validation | mac_per_instr | 0.09473608946828573 |
| inference / validation | instr_per_mac | 10.555639414847965 |
| inference / validation | histogram_total | 4091475007 |
| inference / validation | instret_end | 4091415048 |
| inference / validation | post_counter_tail | 59952 |
| inference / validation | csr_serialization_retries | 7 |
| inference / validation | whole_program_instret | 4091475000 |
| inference / validation | text_size | 5674060 |
| inference / validation | linked_text_size | 85628 |
| inference / validation | data_size | 248 |
| inference / validation | bss_size | 71565920 |
| inference / validation | profiled_instret_invoke | 4065903082 |
| inference / validation | per_op_instret_sum | 4065899912 |
| inference / validation | profiled_minus_op_sum | 3170 |
| inference / validation | profiled_minus_headline | 2176 |
| inference / validation | op_sum_minus_headline | -994 |
| inference / validation | profiler_overhead_fraction | 5.35182743088722e-07 |
| inference / validation | spike_histogram_wall_seconds | 104.83777337499487 |
| inference / validation | spike_profiled_wall_seconds | 22.9317638160137 |
| whole program | dyn_instr | 4091475007 |
| whole program | dyn_loads | 979096048 |
| whole program | dyn_stores | 39918765 |
| whole program | dyn_memory_instructions | 1019014813 |
| whole program | mem_instr_share | 0.24905805638714487 |
| whole program | loads_stores_per_mac | 2.64550297045944 |
| whole program | mul_share | 0.11057588014737199 |
| whole program | branch_share | 0.12358354924200078 |
| whole program | custom_count | 0 |
| whole program | compressed_count | 1316482307 |
| whole program | compressed_fraction | 0.3217622751569212 |
| kernel functions | dyn_instr | 4065504902 |
| kernel functions | dyn_loads | 972350928 |
| kernel functions | dyn_stores | 38648686 |
| kernel functions | dyn_memory_instructions | 1010999614 |
| kernel functions | mem_instr_share | 0.24867750460776594 |
| kernel functions | loads_stores_per_mac | 2.624694408608511 |
| kernel functions | mul_share | 0.11122369543265158 |
| kernel functions | branch_share | 0.12248057006524303 |
| kernel functions | custom_count | 0 |
| kernel functions | compressed_count | 1302698503 |
| kernel functions | compressed_fraction | 0.3204272370595705 |

| type | conv_kind | count | macs | instret | share | instr_per_mac |
| --- | --- | --- | --- | --- | --- | --- |
| CONV_2D | 1x1 | 32 | 338537472 | 3293275045 | 0.809973818505298 | 9.72794835840211 |
| DEPTHWISE_CONV_2D | - | 16 | 34532064 | 489238850 | 0.12032722869511822 | 14.1676689235836 |
| CONV_2D | kxk | 1 | 10838016 | 246970111 | 0.060741760445139896 | 22.78739125315925 |
| ADD | - | 9 | 0 | 23034616 | 0.0056653135934242125 |  |
| FULLY_CONNECTED | - | 1 | 1280000 | 11580007 | 0.002848077479088322 | 9.04688046875 |
| QUANTIZE | - | 2 | 0 | 909513 | 0.00022369273975724344 |  |
| SOFTMAX | - | 1 | 0 | 447400 | 0.00011003705473961419 |  |
| AVERAGE_POOL_2D | - | 1 | 0 | 436506 | 0.00010735769918679041 |  |
| RESHAPE | - | 1 | 0 | 7864 | 1.9341336577387705e-06 |  |

| scope | category | count | share |
| --- | --- | --- | --- |
| whole program | load | 979096048 | 0.2393014857294471 |
| whole program | store | 39918765 | 0.009756570657697776 |
| whole program | mul | 452418450 | 0.11057588014737199 |
| whole program | div | 4520670 | 0.0011048998203009187 |
| whole program | branch | 505639003 | 0.12358354924200078 |
| whole program | jump | 22252915 | 0.005438849061017862 |
| whole program | alu | 2087629142 | 0.5102387619204147 |
| whole program | csr/system | 14 | 3.421748874439599e-09 |
| whole program | custom | 0 | 0.0 |
| whole program | other | 0 | 0.0 |
| kernel functions | load | 972350928 | 0.2391710135490571 |
| kernel functions | store | 38648686 | 0.009506491058708851 |
| kernel functions | mul | 452180479 | 0.11122369543265158 |
| kernel functions | div | 4432578 | 0.0010902896704956427 |
| kernel functions | branch | 497945358 | 0.12248057006524303 |
| kernel functions | jump | 21045932 | 0.0051767080614382194 |
| kernel functions | alu | 2078900941 | 0.5113512321624056 |
| kernel functions | csr/system | 0 | 0.0 |
| kernel functions | custom | 0 | 0.0 |
| kernel functions | other | 0 | 0.0 |

| name | dyn_instr | share | size_bytes |
| --- | --- | --- | --- |
| tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 3377959682 | 0.8256092671275603 | 768 |
| cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 405041478 | 0.09899644438913226 | 788 |
| tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.06024382614541045 | 140 |
| tflite::micro::EvalAddQuantized(tflite::micro::TfLiteContext*, tflite::micro::TfLiteNode*, tflite::micro::TfLiteAddParams*, tflite::micro::OpDataAdd const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor*) | 23025049 | 0.005627566821404758 | 3980 |
| void tflite::micro::reference_integer_ops::FullyConnected<signed char, signed char, signed char, int>(tflite::micro::FullyConnectedParams const&, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 11543100 | 0.0028212563880388383 | 360 |
| htif_putchar | 10666942 | 0.0026071140558723204 | 32 |
| __muldf3 | 3560316 | 0.000870179090403521 | 1188 |
| _start | 2097471 | 0.0005126442166728357 | 68 |
| __divdf3 | 1865996 | 0.00045606926519348525 | 1088 |
| tflite::micro::internal::InitializeTfLiteTensorFromFlatbuffer(tflite::micro::IPersistentBufferAllocator*, tflite::micro::INonPersistentBufferAllocator*, bool, tflite::Tensor const&, flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>, unsigned int> const*, tflite::micro::TfLiteTensor*) | 1592972 | 0.00038933929628669976 | 776 |

| rank | type | conv_kind | instret | share |
| --- | --- | --- | --- | --- |
| 1 | CONV_2D | 1x1 | 3293275045 | 0.809973818505298 |
| 2 | DEPTHWISE_CONV_2D | - | 489238850 | 0.12032722869511822 |
| 3 | CONV_2D | kxk | 246970111 | 0.060741760445139896 |

| rank | name | dyn_instr | share |
| --- | --- | --- | --- |
| 1 | tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 3377959682 | 0.8256092671275603 |
| 2 | cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 405041478 | 0.09899644438913226 |
| 3 | tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.06024382614541045 |
