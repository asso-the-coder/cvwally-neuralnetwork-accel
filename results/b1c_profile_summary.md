| config | git_commit | git_dirty |
| --- | --- | --- |
| B1C | e0eb12ac75f6db2f47869fd7f549818322a42e41 | False |

| scope | metric | value |
| --- | --- | --- |
| inference / validation | instret_invoke | 3393991551 |
| inference / validation | total_macs | 385187552 |
| inference / validation | mac_per_instr | 0.1134910167606366 |
| inference / validation | instr_per_mac | 8.811270077076633 |
| inference / validation | histogram_total | 3437450007 |
| inference / validation | instret_end | 3437390048 |
| inference / validation | post_counter_tail | 59952 |
| inference / validation | csr_serialization_retries | 7 |
| inference / validation | whole_program_instret | 3437450000 |
| inference / validation | text_size | 5676308 |
| inference / validation | linked_text_size | 87460 |
| inference / validation | data_size | 248 |
| inference / validation | bss_size | 71565920 |
| inference / validation | profiled_instret_invoke | 3393993727 |
| inference / validation | per_op_instret_sum | 3393990557 |
| inference / validation | profiled_minus_op_sum | 3170 |
| inference / validation | profiled_minus_headline | 2176 |
| inference / validation | op_sum_minus_headline | -994 |
| inference / validation | profiler_overhead_fraction | 6.411330043997508e-07 |
| inference / validation | spike_histogram_wall_seconds | 87.18654977099504 |
| inference / validation | spike_profiled_wall_seconds | 18.9451419209945 |
| whole program | dyn_instr | 3437450007 |
| whole program | dyn_loads | 865268927 |
| whole program | dyn_stores | 11456615 |
| whole program | dyn_memory_instructions | 876725542 |
| whole program | mem_instr_share | 0.2550511397153826 |
| whole program | loads_stores_per_mac | 2.2761004021230677 |
| whole program | mul_share | 0.11989525088677166 |
| whole program | branch_share | 0.13901342013030185 |
| whole program | custom_count | 0 |
| whole program | compressed_count | 1478248265 |
| whole program | compressed_fraction | 0.43004211319137886 |
| kernel functions | dyn_instr | 3393597866 |
| kernel functions | dyn_loads | 854081421 |
| kernel functions | dyn_stores | 10172824 |
| kernel functions | dyn_memory_instructions | 864254245 |
| kernel functions | mem_instr_share | 0.2546719673709271 |
| kernel functions | loads_stores_per_mac | 2.2437231953954733 |
| kernel functions | mul_share | 0.12137058080057149 |
| kernel functions | branch_share | 0.13723028107302565 |
| kernel functions | custom_count | 0 |
| kernel functions | compressed_count | 1455501200 |
| kernel functions | compressed_fraction | 0.4288961914381402 |

| type | conv_kind | count | macs | instret | share | instr_per_mac |
| --- | --- | --- | --- | --- | --- | --- |
| CONV_2D | 1x1 | 32 | 338537472 | 2623924771 | 0.773108314881691 | 7.750766127892631 |
| DEPTHWISE_CONV_2D | - | 16 | 34532064 | 489238850 | 0.14414842493903054 | 14.1676689235836 |
| CONV_2D | kxk | 1 | 10838016 | 246970156 | 0.07276682747976099 | 22.787395405210695 |
| ADD | - | 9 | 0 | 23034616 | 0.006786876421354093 |  |
| FULLY_CONNECTED | - | 1 | 1280000 | 9020881 | 0.0026578956019384535 | 7.04756328125 |
| QUANTIZE | - | 2 | 0 | 909513 | 0.00026797721892194885 |  |
| SOFTMAX | - | 1 | 0 | 447400 | 0.0001318211039816692 |  |
| AVERAGE_POOL_2D | - | 1 | 0 | 436506 | 0.00012861131608096223 |  |
| RESHAPE | - | 1 | 0 | 7864 | 2.317034335520444e-06 |  |

| scope | category | count | share |
| --- | --- | --- | --- |
| whole program | load | 865268927 | 0.2517182577893416 |
| whole program | store | 11456615 | 0.0033328819260410557 |
| whole program | mul | 412133931 | 0.11989525088677166 |
| whole program | div | 490880 | 0.0001428035313969295 |
| whole program | branch | 477851682 | 0.13901342013030185 |
| whole program | jump | 18223877 | 0.005301568593838171 |
| whole program | alu | 1652024081 | 0.48059581306952226 |
| whole program | csr/system | 14 | 4.072786504964579e-09 |
| whole program | custom | 0 | 0.0 |
| whole program | other | 0 | 0.0 |
| kernel functions | load | 854081421 | 0.25167431579237093 |
| kernel functions | store | 10172824 | 0.0029976515785562438 |
| kernel functions | mul | 411882944 | 0.12137058080057149 |
| kernel functions | div | 402755 | 0.00011868082663392386 |
| kernel functions | branch | 465704389 | 0.13723028107302565 |
| kernel functions | jump | 17016075 | 0.005014169525058276 |
| kernel functions | alu | 1634337458 | 0.4815943204037835 |
| kernel functions | csr/system | 0 | 0.0 |
| kernel functions | custom | 0 | 0.0 |
| kernel functions | other | 0 | 0.0 |

| name | dyn_instr | share | size_bytes |
| --- | --- | --- | --- |
| cvw::Conv1x1(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 2476373066 | 0.7204099146044686 | 270 |
| cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 405041478 | 0.11783196182495055 | 788 |
| tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.07170609274260203 | 140 |
| tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 232239894 | 0.0675616790141146 | 768 |
| tflite::micro::EvalAddQuantized(tflite::micro::TfLiteContext*, tflite::micro::TfLiteNode*, tflite::micro::TfLiteAddParams*, tflite::micro::OpDataAdd const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor*) | 23025049 | 0.006698293488810585 | 3980 |
| cvw::FoldBias(int, int, int, signed char const*, int const*, int*) | 17874052 | 0.0051997998410453685 | 116 |
| htif_putchar | 10668282 | 0.0031035453543397525 | 32 |
| cvw::FullyConnected(tflite::micro::FullyConnectedParams const&, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 8984076 | 0.002613587392312583 | 282 |
| __muldf3 | 3560316 | 0.0010357433541578194 | 1188 |
| _start | 2097471 | 0.0006101822559538973 | 68 |

| rank | type | conv_kind | instret | share |
| --- | --- | --- | --- | --- |
| 1 | CONV_2D | 1x1 | 2623924771 | 0.773108314881691 |
| 2 | DEPTHWISE_CONV_2D | - | 489238850 | 0.14414842493903054 |
| 3 | CONV_2D | kxk | 246970156 | 0.07276682747976099 |

| rank | name | dyn_instr | share |
| --- | --- | --- | --- |
| 1 | cvw::Conv1x1(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 2476373066 | 0.7204099146044686 |
| 2 | cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 405041478 | 0.11783196182495055 |
| 3 | tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.07170609274260203 |
