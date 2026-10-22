#include <cstdlib>

extern "C" {
void* __dso_handle = nullptr;
void __cxa_pure_virtual() { std::abort(); }
}
