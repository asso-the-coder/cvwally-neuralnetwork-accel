#ifndef CVW_KERNELS_H_
#define CVW_KERNELS_H_
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"
namespace cvw {
tflite::TFLMRegistration RegisterDepthwise();
tflite::TFLMRegistration RegisterConv1x1();
tflite::TFLMRegistration RegisterFC();
void Depthwise(const tflite::DepthwiseParams&, const int32_t*, const int32_t*,
               const tflite::RuntimeShape&, const int8_t*, const tflite::RuntimeShape&,
               const int8_t*, const int32_t*, const tflite::RuntimeShape&, int8_t*);
bool FoldBias(int channels, int depth, int32_t offset, const int8_t* weights,
              const int32_t* bias, int32_t* folded);
void Conv1x1(const tflite::ConvParams&, const int32_t*, const int32_t*,
             const tflite::RuntimeShape&, const int8_t*, const tflite::RuntimeShape&,
             const int8_t*, const int32_t*, const tflite::RuntimeShape&, int8_t*);
void FullyConnected(const tflite::FullyConnectedParams&, const tflite::RuntimeShape&,
                    const int8_t*, const tflite::RuntimeShape&, const int8_t*,
                    const int32_t*, const tflite::RuntimeShape&, int8_t*);
}
#endif
