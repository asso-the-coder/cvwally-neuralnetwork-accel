#include "src/kernels/kernels.h"
#include <algorithm>
#include <new>
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
namespace cvw {
using namespace tflite;
void FullyConnected(const FullyConnectedParams& p, const RuntimeShape&,
                    const int8_t* input, const RuntimeShape& f, const int8_t* weights,
                    const int32_t* bias, const RuntimeShape& out, int8_t* output) {
  const int depth = f.Dims(f.DimensionsCount() - 1);
  const int channels = out.Dims(out.DimensionsCount() - 1);
  const int batches = out.FlatSize() / channels;
  for (int b = 0; b < batches; ++b) {
    const int8_t* wt = weights;
    for (int k = 0; k < channels; ++k) {
      const int8_t* x = input;
      int32_t acc = 0;
      for (int c = 0; c < depth; ++c) acc += static_cast<int32_t>(*wt++) * *x++;
      acc += bias[k];
      acc = tflite::MultiplyByQuantizedMultiplier(acc, p.output_multiplier, p.output_shift);
      acc += p.output_offset;
      acc = std::max(p.quantized_activation_min, std::min(p.quantized_activation_max, acc));
      *output++ = static_cast<int8_t>(acc);
    }
    input += depth;
  }
}
namespace {
struct Data { OpDataFullyConnected reference; int32_t* folded; };
void* Init(TfLiteContext* ctx, const char*, size_t) {
  void* memory = ctx->AllocatePersistentBuffer(ctx, sizeof(Data));
  return memory ? new (memory) Data{} : nullptr;
}
TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
  // This pinned TFLM revision exposes FC Prepare through its registration.
  TF_LITE_ENSURE_OK(ctx, micro::Register_FULLY_CONNECTED().prepare(ctx, node));
  auto& d = *static_cast<Data*>(node->user_data);
  d.folded = nullptr;
  const auto* input = micro::GetEvalInput(ctx, node, 0);
  const auto* filter = micro::GetEvalInput(ctx, node, 1);
  const auto* bias = node->inputs->size > 2 ? micro::GetEvalInput(ctx, node, 2) : nullptr;
  const auto* output = micro::GetEvalOutput(ctx, node, 0);
  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      !bias || bias->type != kTfLiteInt32 || d.reference.filter_zero_point != 0 ||
      d.reference.is_per_channel || filter->dims->size != 2) return kTfLiteOk;
  const int channels = filter->dims->data[0], depth = filter->dims->data[1];
  auto* folded = static_cast<int32_t*>(ctx->AllocatePersistentBuffer(ctx, channels * sizeof(int32_t)));
  if (folded && FoldBias(channels, depth, -d.reference.input_zero_point,
                        micro::GetTensorData<int8_t>(filter), micro::GetTensorData<int32_t>(bias), folded)) {
    d.folded = folded;
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
  const auto& d = *static_cast<Data*>(node->user_data);
  if (!d.folded) return micro::Register_FULLY_CONNECTED().invoke(ctx, node);
  const auto* input = micro::GetEvalInput(ctx, node, 0);
  const auto* filter = micro::GetEvalInput(ctx, node, 1);
  auto* output = micro::GetEvalOutput(ctx, node, 0);
  FullyConnected(micro::FullyConnectedParamsQuantized(d.reference), micro::GetTensorShape(input),
                 micro::GetTensorData<int8_t>(input), micro::GetTensorShape(filter),
                 micro::GetTensorData<int8_t>(filter), d.folded,
                 micro::GetTensorShape(output), micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}
}
TFLMRegistration RegisterFC() { return micro::RegisterOp(Init, Prepare, Eval); }
}
