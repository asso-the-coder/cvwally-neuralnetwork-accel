| config | git_commit | git_dirty |
| --- | --- | --- |
| X2 | 0c0f12252b380447d7d359c46cd8d0e5d7153152 | False |

| scope | metric | value |
| --- | --- | --- |
| inference / validation | instret_invoke | 1035349510 |
| inference / validation | total_macs | 385187552 |
| inference / validation | mac_per_instr | 0.37203625276260577 |
| inference / validation | instr_per_mac | 2.6879100963262696 |
| inference / validation | histogram_total | 1079655007 |
| inference / validation | instret_end | 1079595049 |
| inference / validation | post_counter_tail | 59951 |
| inference / validation | csr_serialization_retries | 7 |
| inference / validation | whole_program_instret | 1079655000 |
| inference / validation | text_size | 5677656 |
| inference / validation | linked_text_size | 88696 |
| inference / validation | data_size | 248 |
| inference / validation | bss_size | 71565952 |
| inference / validation | profiled_instret_invoke | 1035351686 |
| inference / validation | per_op_instret_sum | 1035348516 |
| inference / validation | profiled_minus_op_sum | 3170 |
| inference / validation | profiled_minus_headline | 2176 |
| inference / validation | op_sum_minus_headline | -994 |
| inference / validation | profiler_overhead_fraction | 2.101705732202452e-06 |
| inference / validation | spike_histogram_wall_seconds | 36.72300237399759 |
| inference / validation | spike_profiled_wall_seconds | 10.071565771002497 |
| inference / validation | instret_setup | 29542302 |
| inference / validation | rqnt_layers | 49 |
| inference / validation | rqnt_scalar_layers | 0 |
| inference / validation | rqnt_table_bytes | 167816 |
| inference / validation | qmac8_count | 42477184 |
| inference / validation | rqnt_count | 6337288 |
| whole program | dyn_instr | 1079655007 |
| whole program | dyn_loads | 240491549 |
| whole program | dyn_stores | 11458205 |
| whole program | dyn_memory_instructions | 251949754 |
| whole program | mem_instr_share | 0.23336135373473055 |
| whole program | loads_stores_per_mac | 0.6540963037144045 |
| whole program | mul_share | 0.06111134628397088 |
| whole program | branch_share | 0.1456014976828612 |
| whole program | custom_count | 48814472 |
| whole program | compressed_count | 517818625 |
| whole program | compressed_fraction | 0.47961489702052573 |
| kernel functions | dyn_instr | 1034955825 |
| kernel functions | dyn_loads | 228989517 |
| kernel functions | dyn_stores | 10151668 |
| kernel functions | dyn_memory_instructions | 239141185 |
| kernel functions | mem_instr_share | 0.2310641471098537 |
| kernel functions | loads_stores_per_mac | 0.6208434923670638 |
| kernel functions | mul_share | 0.06350820239115036 |
| kernel functions | branch_share | 0.13985066753936093 |
| kernel functions | custom_count | 48814472 |
| kernel functions | compressed_count | 494379741 |
| kernel functions | compressed_fraction | 0.4776819735277107 |

| type | conv_kind | count | macs | instret | share | instr_per_mac |
| --- | --- | --- | --- | --- | --- | --- |
| CONV_2D | 1x1 | 32 | 338537472 | 381956746 | 0.3689149794845652 | 1.1282554446439537 |
| DEPTHWISE_CONV_2D | - | 16 | 34532064 | 380447206 | 0.3674569821485759 | 11.01721594168249 |
| CONV_2D | kxk | 1 | 10838016 | 246970156 | 0.23853745479871657 | 22.787395405210695 |
| ADD | - | 9 | 0 | 23034616 | 0.022248107876264182 |  |
| FULLY_CONNECTED | - | 1 | 1280000 | 1138509 | 0.0010996350471003144 | 0.88946015625 |
| QUANTIZE | - | 2 | 0 | 909513 | 0.0008784580276425995 |  |
| SOFTMAX | - | 1 | 0 | 447400 | 0.0004321236986907268 |  |
| AVERAGE_POOL_2D | - | 1 | 0 | 436506 | 0.0004216016701401305 |  |
| RESHAPE | - | 1 | 0 | 7864 | 7.595486737827169e-06 |  |

| scope | category | count | share |
| --- | --- | --- | --- |
| whole program | load | 240491549 | 0.2227485145169155 |
| whole program | store | 11458205 | 0.010612839217815067 |
| whole program | mul | 65979171 | 0.06111134628397088 |
| whole program | div | 490953 | 0.00045473136957350303 |
| whole program | branch | 157199386 | 0.1456014976828612 |
| whole program | jump | 7275215 | 0.006738462705985487 |
| whole program | alu | 547946042 | 0.5075195673130426 |
| whole program | csr/system | 14 | 1.2967105148617163e-08 |
| whole program | custom | 48814472 | 0.0452130279427306 |
| whole program | other | 0 | 0.0 |
| kernel functions | load | 228989517 | 0.22125535357994627 |
| kernel functions | store | 10151668 | 0.009808793529907423 |
| kernel functions | mul | 65728184 | 0.06350820239115036 |
| kernel functions | div | 402755 | 0.0003891518751537052 |
| kernel functions | branch | 144739263 | 0.13985066753936093 |
| kernel functions | jump | 6065731 | 0.005860859810127645 |
| kernel functions | alu | 530064235 | 0.5121612171224796 |
| kernel functions | csr/system | 0 | 0.0 |
| kernel functions | custom | 48814472 | 0.047165754151874066 |
| kernel functions | other | 0 | 0.0 |

| name | dyn_instr | share | size_bytes |
| --- | --- | --- | --- |
| cvw::Conv1x1(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*, unsigned long const*) | 381937962 | 0.3537592652501819 | 496 |
| cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*, unsigned long const*) | 380436902 | 0.3523689507605831 | 830 |
| tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 232239894 | 0.21510565180012173 | 768 |
| tflite::micro::EvalAddQuantized(tflite::micro::TfLiteContext*, tflite::micro::TfLiteNode*, tflite::micro::TfLiteAddParams*, tflite::micro::OpDataAdd const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor const*, tflite::micro::TfLiteEvalTensor*) | 23025049 | 0.021326302245361604 | 3980 |
| cvw::FoldBias(int, int, int, signed char const*, int const*, int*) | 17874052 | 0.01655533655113221 | 116 |
| tflite::micro::MultiplyByQuantizedMultiplier(int, int, int) | 14729522 | 0.01364280432591927 | 140 |
| htif_putchar | 11123266 | 0.01030261141557416 | 32 |
| __muldf3 | 3560316 | 0.003297642281021719 | 1188 |
| _start | 2097483 | 0.0019427344720312125 | 68 |
| __divdf3 | 1865996 | 0.0017283261670642167 | 1088 |

| rank | type | conv_kind | instret | share |
| --- | --- | --- | --- | --- |
| 1 | CONV_2D | 1x1 | 381956746 | 0.3689149794845652 |
| 2 | DEPTHWISE_CONV_2D | - | 380447206 | 0.3674569821485759 |
| 3 | CONV_2D | kxk | 246970156 | 0.23853745479871657 |

| rank | name | dyn_instr | share |
| --- | --- | --- | --- |
| 1 | cvw::Conv1x1(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*, unsigned long const*) | 381937962 | 0.3537592652501819 |
| 2 | cvw::Depthwise(tflite::micro::DepthwiseParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, int const*, tflite::micro::RuntimeShape const&, signed char*, unsigned long const*) | 380436902 | 0.3523689507605831 |
| 3 | tflite::micro::reference_integer_ops::ConvPerChannel(tflite::micro::ConvParams const&, int const*, int const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, signed char const*, tflite::micro::RuntimeShape const&, int const*, tflite::micro::RuntimeShape const&, signed char*) [clone .isra.0] | 232239894 | 0.21510565180012173 |
