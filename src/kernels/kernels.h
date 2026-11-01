#ifndef CVW_KERNELS_H_
#define CVW_KERNELS_H_
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"
namespace cvw {
#ifdef CVW_RQNT
extern unsigned rqnt_layers, rqnt_scalar_layers, rqnt_table_bytes;
bool RqntFieldsFit(int channels, const int32_t* mult, const int32_t* shift,
                   int zp, int lo, int hi);
uint64_t* PrepareRqnt(TfLiteContext* ctx, int channels, const int32_t* mult,
                      const int32_t* shift, int zp, int lo, int hi);
#endif
tflite::TFLMRegistration RegisterDepthwise();
tflite::TFLMRegistration RegisterConv1x1();
tflite::TFLMRegistration RegisterFC();
void Depthwise(const tflite::DepthwiseParams&, const int32_t*, const int32_t*,
               const tflite::RuntimeShape&, const int8_t*, const tflite::RuntimeShape&,
               const int8_t*, const int32_t*, const tflite::RuntimeShape&, int8_t*
#ifdef CVW_RQNT
               , const uint64_t* rqnt = nullptr
#endif
               );
bool FoldBias(int channels, int depth, int32_t offset, const int8_t* weights,
              const int32_t* bias, int32_t* folded);
void Conv1x1(const tflite::ConvParams&, const int32_t*, const int32_t*,
             const tflite::RuntimeShape&, const int8_t*, const tflite::RuntimeShape&,
             const int8_t*, const int32_t*, const tflite::RuntimeShape&, int8_t*
#ifdef CVW_RQNT
               , const uint64_t* rqnt = nullptr
#endif
               );
void FullyConnected(const tflite::FullyConnectedParams&, const tflite::RuntimeShape&,
                    const int8_t*, const tflite::RuntimeShape&, const int8_t*,
                    const int32_t*, const tflite::RuntimeShape&, int8_t*
#ifdef CVW_RQNT
               , const uint64_t* rqnt = nullptr
#endif
               );
}
#endif
