| config | git_commit | git_dirty |
| --- | --- | --- |
| X1 | e0eb12ac75f6db2f47869fd7f549818322a42e41 | False |

| scope | metric | value |
| --- | --- | --- |
| inference / validation | instret_invoke | 1333373407 |
| inference / validation | total_macs | 385187552 |
| inference / validation | mac_per_instr | 0.28888198157982314 |
| inference / validation | instr_per_mac | 3.461621228611251 |
| inference / validation | histogram_total | 1376995007 |
| inference / validation | instret_end | 1376935048 |
| inference / validation | post_counter_tail | 59952 |
| inference / validation | csr_serialization_retries | 7 |
| inference / validation | whole_program_instret | 1376995000 |
| inference / validation | text_size | 5676716 |
| inference / validation | linked_text_size | 87820 |
| inference / validation | data_size | 248 |
| inference / validation | bss_size | 71565936 |
| inference / validation | profiled_instret_invoke | 1333375583 |
| inference / validation | per_op_instret_sum | 1333372413 |
| inference / validation | profiled_minus_op_sum | 3170 |
| inference / validation | profiled_minus_headline | 2176 |
| inference / validation | op_sum_minus_headline | -994 |
| inference / validation | profiler_overhead_fraction | 1.6319509513061707e-06 |
| inference / validation | spike_histogram_wall_seconds | 41.46291501200176 |
| inference / validation | spike_profiled_wall_seconds | 11.304488768000738 |
| whole program | dyn_instr | 1376995007 |
| whole program | dyn_loads | 274787166 |
| whole program | dyn_stores | 11544194 |
| whole program | dyn_memory_instructions | 286331360 |
| whole program | mem_instr_share | 0.20793928703039952 |
| whole program | loads_stores_per_mac | 0.7433556939036285 |
| whole program | mul_share | 0.05251758984773138 |
| whole program | branch_share | 0.13700391362421258 |
| whole program | custom_count | 42477184 |
| whole program | compressed_count | 679811391 |
| whole program | compressed_fraction | 0.4936919796688849 |
| kernel functions | dyn_instr | 1332979722 |
| kernel functions | dyn_loads | 263518863 |
| kernel functions | dyn_stores | 10260114 |
| kernel functions | dyn_memory_instructions | 273778977 |
| kernel functions | mem_instr_share | 0.2053887035799934 |
| kernel functions | loads_stores_per_mac | 0.7107679767387706 |
| kernel functions | mul_share | 0.054063442084380034 |
| kernel functions | branch_share | 0.1323543419965094 |
| kernel functions | custom_count | 42477184 |
| kernel functions | compressed_count | 656903451 |
| kernel functions | compressed_fraction | 0.4928082851961044 |

| type | conv_kind | count | macs | instret | share | instr_per_mac |
| --- | --- | --- | --- | --- | --- | --- |
| CONV_2D | 1x1 | 32 | 338537472 | 571142607 | 0.42834338222616153 | 1.6870882966834466 |
| DEPTHWISE_CONV_2D | - | 16 | 34532064 | 489238850 | 0.3669175108931029 | 14.1676689235836 |
| CONV_2D | kxk | 1 | 10838016 | 246970156 | 0.18522174783217102 | 22.787395405210695 |
| ADD | - | 9 | 0 | 23034616 | 0.017275414589619044 |  |
| FULLY_CONNECTED | - | 1 | 1280000 | 1184901 | 0.0008886475912016157 | 0.92570390625 |
| QUANTIZE | - | 2 | 0 | 909513 | 0.0006821131357105405 |  |
| SOFTMAX | - | 1 | 0 | 447400 | 0.00033553936768017145 |  |
| AVERAGE_POOL_2D | - | 1 | 0 | 436506 | 0.000327369126572644 |  |
| RESHAPE | - | 1 | 0 | 7864 | 5.8978131145213865e-06 |  |

| scope | category | count | share |
| --- | --- | --- | --- |
| whole program | load | 274787166 | 0.19955567347964973 |
| whole program | store | 11544194 | 0.00838361355074979 |
| whole program | mul | 72316459 | 0.05251758984773138 |
| whole program | div | 490886 | 0.0003564907624970059 |
| whole program | branch | 188653705 | 0.13700391362421258 |
| whole program | jump | 22254903 | 0.016161934420144197 |
| whole program | alu | 764470496 | 0.5551730341168913 |
| whole program | csr/system | 14 | 1.0167066640641785e-08 |
| whole program | custom | 42477184 | 0.030847740031057353 |
| whole program | other | 0 | 0.0 |
| kernel functions | load | 263518863 | 0.19769157673652893 |
| kernel functions | store | 10260114 | 0.007697126843464465 |
| kernel functions | mul | 72065472 | 0.054063442084380034 |
| kernel functions | div | 402755 | 0.0003021463817886946 |
| kernel functions | branch | 176425654 | 0.1323543419965094 |
| kernel functions | jump | 21046835 | 0.015789313710205113 |
| kernel functions | alu | 746782845 | 0.5602357130230973 |
| kernel functions | csr/system | 0 | 0.0 |
| kernel functions | custom | 42477184 | 0.0318663392240261 |
| kernel functions | other | 0 | 0.0 |

| name | dyn_instr | share | size_bytes |
| --- | --- | --- | --- |
| cvw::Conv1x1(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 423590902 | 0.30761978064311163 | 430 |
| cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 405041478 | 0.29414883564643163 | 788 |
| tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.17900290687110676 | 140 |
| tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 232239894 | 0.16865703420811315 | 768 |
| tflite::micro::EvalAddQuantized(tflite::micro::TfLiteContext*, tflite::micro::TfLiteNode*, tflite::micro::TfLiteAddParams*, tflite::micro::OpDataAdd const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor*) | 23025049 | 0.016721229113360176 | 3980 |
| cvw::FoldBias(int, int, int, signed char const*, int const*, int*) | 17874052 | 0.012980476987306897 | 116 |
| htif_putchar | 10829236 | 0.007864397434231219 | 32 |
| __muldf3 | 3560316 | 0.002585569288124514 | 1188 |
| _start | 2097475 | 0.0015232262930057233 | 68 |
| __divdf3 | 1865996 | 0.0013551218345122147 | 1088 |

| rank | type | conv_kind | instret | share |
| --- | --- | --- | --- | --- |
| 1 | CONV_2D | 1x1 | 571142607 | 0.42834338222616153 |
| 2 | DEPTHWISE_CONV_2D | - | 489238850 | 0.3669175108931029 |
| 3 | CONV_2D | kxk | 246970156 | 0.18522174783217102 |

| rank | name | dyn_instr | share |
| --- | --- | --- | --- |
| 1 | cvw::Conv1x1(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 423590902 | 0.30761978064311163 |
| 2 | cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*) | 405041478 | 0.29414883564643163 |
| 3 | tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 246486109 | 0.17900290687110676 |
