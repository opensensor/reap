// SPDX-License-Identifier: Apache-2.0
// Reuse Qwen's INT4 schedules with symmetric offsets and Muse shard geometry.
#include "kernel_operator.h"
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"
using namespace AscendC;
template<uint32_t N>
__aicore__ inline void run(GM_ADDR lo, GM_ADDR hi, GM_ADDR xs, GM_ADDR sums,
                          GM_ADDR q, GM_ADDR scale, GM_ADDR ws, GM_ADDR routes,
                          GM_ADDR output, __gm__ const QwenW4A8Int4KernelTilingData* td) {
  if (td->routed) {
    native_int4::Schedule<16,N> op;
    op.Init(lo,hi,xs,sums,q,scale,scale,ws,routes,nullptr,output,td);
    op.Process();
  } else {
    native_int4::Schedule<32,N> op;
    op.Init(lo,hi,xs,sums,q,scale,scale,ws,routes,nullptr,output,td);
    op.Process();
  }
}
extern "C" __global__ __aicore__ void muse_a8_native(
    GM_ADDR lo, GM_ADDR hi, GM_ADDR xs, GM_ADDR sums, GM_ADDR q, GM_ADDR scale,
    GM_ADDR ws, GM_ADDR routes, GM_ADDR output, GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  auto td=reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(config);
  if(td->nDim % 64 == 0) run<64>(lo,hi,xs,sums,q,scale,ws,routes,output,td);
  else run<32>(lo,hi,xs,sums,q,scale,ws,routes,output,td);
}
