#include "src/kernels/kernels.h"
#include <algorithm>
#include <climits>
#include <new>
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifdef CVW_XQMAC
#include "src/kernels/xqmac.h"
#endif
namespace cvw {
using namespace tflite;
#ifdef CVW_XQMAC
unsigned qmac8_layers = 0, scalar_layers = 0;
#endif
bool FoldBias(int channels, int depth, int32_t offset, const int8_t* weights,
              const int32_t* bias, int32_t* folded) {
  for (int k = 0; k < channels; ++k) {
    int64_t sum = 0;
    for (int c = 0; c < depth; ++c) sum += *weights++;
    const int64_t value = bias[k] + static_cast<int64_t>(offset) * sum;
    if (value < INT32_MIN || value > INT32_MAX) return false;
    folded[k] = static_cast<int32_t>(value);
  }
  return true;
}
void Conv1x1(const ConvParams& p, const int32_t* mult, const int32_t* shift,
             const RuntimeShape& in, const int8_t* input, const RuntimeShape& f,
             const int8_t* weights, const int32_t* bias,
             const RuntimeShape& out, int8_t* output) {
  const int depth = in.Dims(3), channels = f.Dims(0);
  const int pixels = out.Dims(0) * out.Dims(1) * out.Dims(2);
#ifdef CVW_XQMAC
  const bool packed = ((reinterpret_cast<uintptr_t>(input) |
                        reinterpret_cast<uintptr_t>(weights) | static_cast<uintptr_t>(depth)) & 7) == 0;
  if (packed) ++qmac8_layers; else ++scalar_layers;
#endif
  for (int pixel = 0; pixel < pixels; ++pixel) {
    const int8_t* wt = weights;
    for (int k = 0; k < channels; ++k) {
      const int8_t* x = input;
      int32_t acc = 0;
#ifdef CVW_XQMAC
      int c = 0;
      if (packed) {
        for (; c + 8 <= depth; c += 8) {
          acc += cvw_qmac8(cvw_load8(wt), cvw_load8(x));
          wt += 8;
          x += 8;
        }
      }
      for (; c < depth; ++c) acc += static_cast<int32_t>(*wt++) * *x++;
#else
      for (int c = 0; c < depth; ++c) acc += static_cast<int32_t>(*wt++) * *x++;
#endif
      acc += bias[k];
      acc = tflite::MultiplyByQuantizedMultiplier(acc, mult[k], shift[k]);
      acc += p.output_offset;
      acc = std::max(p.quantized_activation_min, std::min(p.quantized_activation_max, acc));
      *output++ = static_cast<int8_t>(acc);
    }
    input += depth;
  }
}
namespace {
// The first member preserves the upstream Prepare/Eval user_data layout.
struct Data { OpDataConv reference; int32_t* folded; };
void* Init(TfLiteContext* ctx, const char*, size_t) {
  void* memory = ctx->AllocatePersistentBuffer(ctx, sizeof(Data));
  return memory ? new (memory) Data{} : nullptr;
}
TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(ctx, ConvPrepare(ctx, node));
  auto& d = *static_cast<Data*>(node->user_data);
  d.folded = nullptr;
  const auto* input = micro::GetEvalInput(ctx, node, 0);
  const auto* filter = micro::GetEvalInput(ctx, node, 1);
  const auto* bias = node->inputs->size > 2 ? micro::GetEvalInput(ctx, node, 2) : nullptr;
  const auto* output = micro::GetEvalOutput(ctx, node, 0);
  const auto& p = *static_cast<TfLiteConvParams*>(node->builtin_data);
  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      !bias || bias->type != kTfLiteInt32 || filter->dims->size != 4 ||
      filter->dims->data[1] != 1 || filter->dims->data[2] != 1 ||
      p.stride_height != 1 || p.stride_width != 1 ||
      p.dilation_height_factor != 1 || p.dilation_width_factor != 1 ||
      d.reference.padding.height != 0 || d.reference.padding.width != 0 ||
      input->dims->data[1] != output->dims->data[1] ||
      input->dims->data[2] != output->dims->data[2] ||
      input->dims->data[3] != filter->dims->data[3] || d.reference.filter_zero_point != 0) return kTfLiteOk;
  const int channels = filter->dims->data[0], depth = filter->dims->data[3];
  auto* folded = static_cast<int32_t*>(ctx->AllocatePersistentBuffer(ctx, channels * sizeof(int32_t)));
  if (folded && FoldBias(channels, depth, -d.reference.input_zero_point,
                        micro::GetTensorData<int8_t>(filter), micro::GetTensorData<int32_t>(bias), folded)) {
    d.folded = folded;
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
  const auto& d = *static_cast<Data*>(node->user_data);
  if (!d.folded) return micro::Register_CONV_2D().invoke(ctx, node);
  const auto* input = micro::GetEvalInput(ctx, node, 0);
  const auto* filter = micro::GetEvalInput(ctx, node, 1);
  auto* output = micro::GetEvalOutput(ctx, node, 0);
  const auto& p = *static_cast<TfLiteConvParams*>(node->builtin_data);
  Conv1x1(ConvParamsQuantized(p, d.reference), d.reference.per_channel_output_multiplier,
          d.reference.per_channel_output_shift, micro::GetTensorShape(input),
          micro::GetTensorData<int8_t>(input), micro::GetTensorShape(filter),
          micro::GetTensorData<int8_t>(filter), d.folded,
          micro::GetTensorShape(output), micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}
}
TFLMRegistration RegisterConv1x1() { return micro::RegisterOp(Init, Prepare, Eval); }
}
