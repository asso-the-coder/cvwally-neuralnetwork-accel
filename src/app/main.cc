#include <cstdio>
#include <cstring>
#include "inputs.h"
#include "model_data.h"
#include "src/runtime/instret.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#if defined(CVW_DW_B1) || defined(CVW_OFFSET_FOLD)
#include "src/kernels/kernels.h"
#endif

#ifdef PROFILE_OPS
#include "src/app/op_profiler.h"
#endif

#ifndef INPUT_INDEX
#define INPUT_INDEX 0
#endif

namespace {
// Static storage accommodates the activation peak plus persistent metadata.
alignas(16) unsigned char arena[4 * 1024 * 1024];
}

int main() {
    const auto* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return 1;
    tflite::MicroMutableOpResolver<8> resolver;
    if (resolver.AddAdd() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
#ifdef CVW_OFFSET_FOLD
        resolver.AddConv2D(cvw::RegisterConv1x1()) != kTfLiteOk ||
#else
        resolver.AddConv2D() != kTfLiteOk ||
#endif
#ifdef CVW_DW_B1
        resolver.AddDepthwiseConv2D(cvw::RegisterDepthwise()) != kTfLiteOk ||
#else
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
#endif
#ifdef CVW_OFFSET_FOLD
        resolver.AddFullyConnected(cvw::RegisterFC()) != kTfLiteOk ||
#else
        resolver.AddFullyConnected() != kTfLiteOk ||
#endif
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) return 2;
#ifdef PROFILE_OPS
    static OpProfiler profiler;
    tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena), nullptr, &profiler);
#else
    tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
#endif
#ifdef CVW_DW_B1
    const uint64_t setup_start = read_instret();
#endif
    if (interpreter.AllocateTensors() != kTfLiteOk) return 3;
#ifdef CVW_DW_B1
    const uint64_t setup_end = read_instret();
    std::printf("instret_setup=%lu\n", static_cast<unsigned long>(setup_end - setup_start));
#endif
    std::printf("arena_size=%u arena_used=%u\n", static_cast<unsigned>(sizeof(arena)),
                static_cast<unsigned>(interpreter.arena_used_bytes()));
    const unsigned char* const inputs[] = {input_0, input_1, input_2};
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if (input->type != kTfLiteUInt8 || input->bytes != input_0_len ||
        output->type != kTfLiteUInt8 || output->bytes != 1000) return 4;
    std::memcpy(input->data.uint8, inputs[INPUT_INDEX], input->bytes);
    std::printf("input=%d\n", INPUT_INDEX);
    const uint64_t start = read_instret();
    const TfLiteStatus status = interpreter.Invoke();
    const uint64_t end = read_instret();
    if (status != kTfLiteOk) return 5;
    // Nano printf does not support long long; RV64 unsigned long is 64 bits.
    std::printf("instret_invoke=%lu\n", static_cast<unsigned long>(end - start));
#ifdef PROFILE_OPS
    if (!profiler.Print()) return 6;
#endif
    int top1 = 0;
    std::printf("output_hex=");
    for (int i = 0; i < 1000; ++i) {
        std::printf("%02x", static_cast<unsigned>(output->data.uint8[i]));
        if (output->data.uint8[i] > output->data.uint8[top1]) top1 = i;
    }
    std::printf("\ntop1=%d\n", top1);
    std::printf("instret_end=");
    std::fflush(stdout);
    const uint64_t instret_end = read_instret();
    std::printf("%lu\n", static_cast<unsigned long>(instret_end));
    return 0;
}
