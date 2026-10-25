#include <climits>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include "src/kernels/kernels.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h"

namespace {
using namespace tflite;
enum Kind { DW, Conv, FC };
struct Case { int op; Kind kind; int h, w, c, oc, kh, kw, sh, sw, ph, pw, oh, ow; };
// Graph-order shapes and leading padding from the pinned EfficientNet-Lite0 model.
const Case layers[] = {
  {2, DW, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 112, 112},
  {3, Conv, 112, 112, 32, 16, 1, 1, 1, 1, 0, 0, 112, 112},
  {4, Conv, 112, 112, 16, 96, 1, 1, 1, 1, 0, 0, 112, 112},
  {5, DW, 112, 112, 96, 96, 3, 3, 2, 2, 0, 0, 56, 56},
  {6, Conv, 56, 56, 96, 24, 1, 1, 1, 1, 0, 0, 56, 56},
  {7, Conv, 56, 56, 24, 144, 1, 1, 1, 1, 0, 0, 56, 56},
  {8, DW, 56, 56, 144, 144, 3, 3, 1, 1, 1, 1, 56, 56},
  {9, Conv, 56, 56, 144, 24, 1, 1, 1, 1, 0, 0, 56, 56},
  {11, Conv, 56, 56, 24, 144, 1, 1, 1, 1, 0, 0, 56, 56},
  {12, DW, 56, 56, 144, 144, 5, 5, 2, 2, 1, 1, 28, 28},
  {13, Conv, 28, 28, 144, 40, 1, 1, 1, 1, 0, 0, 28, 28},
  {14, Conv, 28, 28, 40, 240, 1, 1, 1, 1, 0, 0, 28, 28},
  {15, DW, 28, 28, 240, 240, 5, 5, 1, 1, 2, 2, 28, 28},
  {16, Conv, 28, 28, 240, 40, 1, 1, 1, 1, 0, 0, 28, 28},
  {18, Conv, 28, 28, 40, 240, 1, 1, 1, 1, 0, 0, 28, 28},
  {19, DW, 28, 28, 240, 240, 3, 3, 2, 2, 0, 0, 14, 14},
  {20, Conv, 14, 14, 240, 80, 1, 1, 1, 1, 0, 0, 14, 14},
  {21, Conv, 14, 14, 80, 480, 1, 1, 1, 1, 0, 0, 14, 14},
  {22, DW, 14, 14, 480, 480, 3, 3, 1, 1, 1, 1, 14, 14},
  {23, Conv, 14, 14, 480, 80, 1, 1, 1, 1, 0, 0, 14, 14},
  {25, Conv, 14, 14, 80, 480, 1, 1, 1, 1, 0, 0, 14, 14},
  {26, DW, 14, 14, 480, 480, 3, 3, 1, 1, 1, 1, 14, 14},
  {27, Conv, 14, 14, 480, 80, 1, 1, 1, 1, 0, 0, 14, 14},
  {29, Conv, 14, 14, 80, 480, 1, 1, 1, 1, 0, 0, 14, 14},
  {30, DW, 14, 14, 480, 480, 5, 5, 1, 1, 2, 2, 14, 14},
  {31, Conv, 14, 14, 480, 112, 1, 1, 1, 1, 0, 0, 14, 14},
  {32, Conv, 14, 14, 112, 672, 1, 1, 1, 1, 0, 0, 14, 14},
  {33, DW, 14, 14, 672, 672, 5, 5, 1, 1, 2, 2, 14, 14},
  {34, Conv, 14, 14, 672, 112, 1, 1, 1, 1, 0, 0, 14, 14},
  {36, Conv, 14, 14, 112, 672, 1, 1, 1, 1, 0, 0, 14, 14},
  {37, DW, 14, 14, 672, 672, 5, 5, 1, 1, 2, 2, 14, 14},
  {38, Conv, 14, 14, 672, 112, 1, 1, 1, 1, 0, 0, 14, 14},
  {40, Conv, 14, 14, 112, 672, 1, 1, 1, 1, 0, 0, 14, 14},
  {41, DW, 14, 14, 672, 672, 5, 5, 2, 2, 1, 1, 7, 7},
  {42, Conv, 7, 7, 672, 192, 1, 1, 1, 1, 0, 0, 7, 7},
  {43, Conv, 7, 7, 192, 1152, 1, 1, 1, 1, 0, 0, 7, 7},
  {44, DW, 7, 7, 1152, 1152, 5, 5, 1, 1, 2, 2, 7, 7},
  {45, Conv, 7, 7, 1152, 192, 1, 1, 1, 1, 0, 0, 7, 7},
  {47, Conv, 7, 7, 192, 1152, 1, 1, 1, 1, 0, 0, 7, 7},
  {48, DW, 7, 7, 1152, 1152, 5, 5, 1, 1, 2, 2, 7, 7},
  {49, Conv, 7, 7, 1152, 192, 1, 1, 1, 1, 0, 0, 7, 7},
  {51, Conv, 7, 7, 192, 1152, 1, 1, 1, 1, 0, 0, 7, 7},
  {52, DW, 7, 7, 1152, 1152, 5, 5, 1, 1, 2, 2, 7, 7},
  {53, Conv, 7, 7, 1152, 192, 1, 1, 1, 1, 0, 0, 7, 7},
  {55, Conv, 7, 7, 192, 1152, 1, 1, 1, 1, 0, 0, 7, 7},
  {56, DW, 7, 7, 1152, 1152, 3, 3, 1, 1, 1, 1, 7, 7},
  {57, Conv, 7, 7, 1152, 320, 1, 1, 1, 1, 0, 0, 7, 7},
  {58, Conv, 7, 7, 320, 1280, 1, 1, 1, 1, 0, 0, 7, 7},
  {61, FC, 1, 1, 1280, 1000, 1, 1, 1, 1, 0, 0, 1, 1},
};
alignas(16) int8_t input[1204224], weights[1280000], expected[1204224], actual[1204224];
int32_t bias[1280], folded[1280], multiplier[1280], shift[1280];
uint32_t state = 0x005b1b1c;
uint32_t Random() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }

RuntimeShape Shape(std::initializer_list<int32_t> dims) {
  return RuntimeShape(static_cast<int>(dims.size()), dims.begin());
}

bool Run(const Case& s, int zero_point, int fill) {
  const RuntimeShape in = Shape({1, s.h, s.w, s.c});
  const RuntimeShape filter = s.kind == DW ? Shape({1, s.kh, s.kw, s.c}) :
      s.kind == Conv ? Shape({s.oc, 1, 1, s.c}) : Shape({s.oc, s.c});
  const RuntimeShape out = s.kind == FC ? Shape({1, s.oc}) : Shape({1, s.oh, s.ow, s.oc});
  const RuntimeShape bs = Shape({s.oc});
  if (in.FlatSize() > static_cast<int>(sizeof(input)) || filter.FlatSize() > static_cast<int>(sizeof(weights)) ||
      out.FlatSize() > static_cast<int>(sizeof(actual)) || s.oc > 1280) return false;
  for (int i = 0; i < in.FlatSize(); ++i) input[i] = fill == 0 ? static_cast<int8_t>(static_cast<int>(Random() & 255) - 128) : fill == 1 ? -128 : 127;
  for (int i = 0; i < filter.FlatSize(); ++i) weights[i] = fill == 0 ? static_cast<int8_t>(static_cast<int>(Random() & 255) - 128) : fill == 1 ? -128 : 127;
  // Scale the right shift with sqrt(taps) so random sums land mostly inside int8;
  // saturated outputs would hide accumulation errors.
  const int taps = s.kind == DW ? s.kh * s.kw : s.c;
  int log2_taps = 0;
  while ((1 << log2_taps) < taps) ++log2_taps;
  for (int i = 0; i < s.oc; ++i) {
    bias[i] = static_cast<int32_t>(Random() % 20001) - 10000;
    multiplier[i] = 0x40000000 + (Random() & 0x3fffffff);
    shift[i] = -8 - log2_taps / 2 - static_cast<int>(Random() % 2);
  }
  std::memset(actual, 0x5a, out.FlatSize());
  if (s.kind == DW) {
    DepthwiseParams p{};
    p.input_offset = -zero_point; p.output_offset = -7;
    p.stride_height = s.sh; p.stride_width = s.sw;
    p.padding_values.height = s.ph; p.padding_values.width = s.pw;
    p.depth_multiplier = 1; p.dilation_height_factor = p.dilation_width_factor = 1;
    p.quantized_activation_min = -128; p.quantized_activation_max = 127;
    micro::reference_integer_ops::DepthwiseConvPerChannel(p, multiplier, shift, in, input,
        filter, weights, bs, bias, out, expected);
    cvw::Depthwise(p, multiplier, shift, in, input, filter, weights, bias, out, actual);
  } else {
    if (!cvw::FoldBias(s.oc, s.c, -zero_point, weights, bias, folded)) return false;
    if (s.kind == Conv) {
      ConvParams p{};
      p.input_offset = -zero_point; p.output_offset = -7;
      p.stride_height = p.stride_width = 1;
      p.dilation_height_factor = p.dilation_width_factor = 1;
      p.quantized_activation_min = -128; p.quantized_activation_max = 127;
      micro::reference_integer_ops::ConvPerChannel(p, multiplier, shift, in, input,
          filter, weights, bs, bias, out, expected);
      cvw::Conv1x1(p, multiplier, shift, in, input, filter, weights, folded, out, actual);
    } else {
      FullyConnectedParams p{};
      p.input_offset = -zero_point; p.output_offset = -7;
      p.output_multiplier = multiplier[0]; p.output_shift = shift[0];
      p.quantized_activation_min = -128; p.quantized_activation_max = 127;
      micro::reference_integer_ops::FullyConnected(p, in, input, filter, weights, bs, bias, out, expected);
      cvw::FullyConnected(p, in, input, filter, weights, folded, out, actual);
    }
  }
  bool ok = true;
  int saturated = 0;
  for (int i = 0; i < out.FlatSize(); ++i) {
    if (expected[i] == -128 || expected[i] == 127) ++saturated;
    if (actual[i] != expected[i]) {
      std::printf("mismatch index=%d expected=%d actual=%d\n", i, expected[i], actual[i]);
      ok = false; break;
    }
  }
  std::printf("%s: op=%d kind=%d input=%dx%dx%d output=%dx%dx%d kernel=%dx%d stride=%dx%d pad=%dx%d zp=%d fill=%d saturated=%d%%\n",
      ok ? "PASS" : "FAIL", s.op, s.kind, s.h, s.w, s.c, s.oh, s.ow, s.oc,
      s.kh, s.kw, s.sh, s.sw, s.ph, s.pw, zero_point, fill,
      static_cast<int>(100LL * saturated / out.FlatSize()));
  return ok;
}
}
int main() {
  bool ok = true;
  for (const auto& s : layers) ok = Run(s, -128, 0) && ok;
  const Case directed[] = {
    {-1, DW, 7, 7, 3, 3, 5, 5, 1, 1, 2, 2, 7, 7},
    {-2, DW, 7, 9, 5, 5, 3, 3, 2, 2, 1, 1, 4, 5},
    {-3, DW, 8, 6, 5, 5, 3, 3, 2, 2, 0, 0, 4, 3},
    {-4, Conv, 3, 5, 17, 9, 1, 1, 1, 1, 0, 0, 3, 5},
    {-5, FC, 1, 1, 31, 13, 1, 1, 1, 1, 0, 0, 1, 1},
  };
  for (const auto& s : directed) {
    for (int zp : {-128, 17}) {
      for (int fill = 0; fill < 3; ++fill) ok = Run(s, zp, fill) && ok;
    }
  }
  const int8_t w[] = {127, -128};
  const int32_t b[] = {INT32_MAX, INT32_MIN};
  int32_t f[2];
  const bool bounds = !cvw::FoldBias(1, 1, 128, w, b, f) &&
      !cvw::FoldBias(1, 1, 128, w + 1, b + 1, f) &&
      cvw::FoldBias(2, 1, 0, w, b, f) && f[0] == INT32_MAX && f[1] == INT32_MIN;
  std::printf("%s: folded bias int32 limits\n", bounds ? "PASS" : "FAIL");
  ok = ok && bounds;
  std::printf("%s: kernel tests\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
