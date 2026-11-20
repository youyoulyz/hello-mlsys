#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void add_kernel(float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] += b[i];
  }
}

static void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

int main() {
  constexpr int n = 1 << 20;
  float* a = nullptr;
  float* b = nullptr;
  check(hipMalloc(&a, n * sizeof(float)), "hipMalloc(a)");
  check(hipMalloc(&b, n * sizeof(float)), "hipMalloc(b)");
  check(hipMemset(a, 0, n * sizeof(float)), "hipMemset(a)");
  check(hipMemset(b, 0, n * sizeof(float)), "hipMemset(b)");

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  for (int i = 0; i < 20; ++i) {
    add_kernel<<<blocks, threads>>>(a, b, n);
  }
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");

  float value = 0.0f;
  check(hipMemcpy(&value, a, sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy");
  hipDeviceProp_t prop{};
  check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::printf("device=%s value=%.1f\n", prop.name, value);
  check(hipFree(a), "hipFree(a)");
  check(hipFree(b), "hipFree(b)");
  return 0;
}
