#include "src/kernels/kernels.h"
#include <algorithm>
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
namespace cvw {
using namespace tflite;
void Depthwise(const DepthwiseParams& p, const int32_t* mult, const int32_t* shift,
               const RuntimeShape& in, const int8_t* input, const RuntimeShape& f,
               const int8_t* weights, const int32_t* bias,
               const RuntimeShape& out, int8_t* output) {
  const int h = in.Dims(1), w = in.Dims(2), c = in.Dims(3);
  const int kh = f.Dims(1), kw = f.Dims(2);
  const int oh = out.Dims(1), ow = out.Dims(2);
  const int py = p.padding_values.height, px = p.padding_values.width;
  const int ybegin = (py + p.stride_height - 1) / p.stride_height;
  const int xbegin = (px + p.stride_width - 1) / p.stride_width;
  const int yend = h + py < kh ? 0 : (h + py - kh) / p.stride_height + 1;
  const int xend = w + px < kw ? 0 : (w + px - kw) / p.stride_width + 1;
  for (int b = 0; b < in.Dims(0); ++b) {
    for (int oy = 0; oy < oh; ++oy) {
      const int iy = oy * p.stride_height - py;
      for (int ox = 0; ox < ow; ++ox) {
        const int ix = ox * p.stride_width - px;
        int fy0 = 0, fy1 = kh, fx0 = 0, fx1 = kw;
        if (!(oy >= ybegin && oy < yend && ox >= xbegin && ox < xend)) {
          // Padding taps contribute zero; clip once instead of testing each tap.
          fy0 = std::max(0, -iy); fy1 = std::min(kh, h - iy);
          fx0 = std::max(0, -ix); fx1 = std::min(kw, w - ix);
        }
        for (int ch = 0; ch < c; ++ch) {
          int32_t acc = 0;
          // Do not form an out-of-array pointer for an empty border window.
          if (fy0 < fy1 && fx0 < fx1) {
            const int8_t* row = input + ((b * h + iy + fy0) * w + ix + fx0) * c;
            const int8_t* wr = weights + (fy0 * kw + fx0) * c;
            for (int fy = fy0; fy < fy1; ++fy) {
              const int8_t* x = row;
              const int8_t* wt = wr;
              for (int fx = fx0; fx < fx1; ++fx) {
                acc += static_cast<int32_t>(wt[ch]) * (static_cast<int32_t>(x[ch]) + p.input_offset);
                x += c; wt += c;
              }
              if (fy + 1 < fy1) { row += w * c; wr += kw * c; }
            }
          }
          acc += bias[ch];
          acc = tflite::MultiplyByQuantizedMultiplier(acc, mult[ch], shift[ch]);
          acc += p.output_offset;
          acc = std::max(p.quantized_activation_min, std::min(p.quantized_activation_max, acc));
          *output++ = static_cast<int8_t>(acc);
        }
      }
    }
  }
}
namespace {
TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
  const auto* input = micro::GetEvalInput(ctx, node, 0);
  const auto* filter = micro::GetEvalInput(ctx, node, 1);
  const auto* bias = node->inputs->size > 2 ? micro::GetEvalInput(ctx, node, 2) : nullptr;
  auto* output = micro::GetEvalOutput(ctx, node, 0);
  const auto& p = *static_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 ||
      output->type != kTfLiteInt8 || !bias || bias->type != kTfLiteInt32 ||
      p.depth_multiplier != 1 || p.dilation_height_factor != 1 || p.dilation_width_factor != 1) {
    return micro::Register_DEPTHWISE_CONV_2D().invoke(ctx, node);
  }
  const auto& d = *static_cast<OpDataConv*>(node->user_data);
  Depthwise(DepthwiseConvParamsQuantized(p, d), d.per_channel_output_multiplier,
            d.per_channel_output_shift, micro::GetTensorShape(input),
            micro::GetTensorData<int8_t>(input), micro::GetTensorShape(filter),
            micro::GetTensorData<int8_t>(filter), micro::GetTensorData<int32_t>(bias),
            micro::GetTensorShape(output), micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}
}
TFLMRegistration RegisterDepthwise() {
  auto reg = micro::Register_DEPTHWISE_CONV_2D();
  reg.invoke = Eval;
  return reg;
}
}
