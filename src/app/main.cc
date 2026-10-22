#include <cstdio>
#include <cstring>
#include "inputs.h"
#include "model_data.h"
#include "src/runtime/instret.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

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
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) return 2;
    tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
    if (interpreter.AllocateTensors() != kTfLiteOk) return 3;
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
    int top1 = 0;
    std::printf("output_hex=");
    for (int i = 0; i < 1000; ++i) {
        std::printf("%02x", static_cast<unsigned>(output->data.uint8[i]));
        if (output->data.uint8[i] > output->data.uint8[top1]) top1 = i;
    }
    std::printf("\ntop1=%d\n", top1);
    return 0;
}
