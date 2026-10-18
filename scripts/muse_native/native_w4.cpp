// SPDX-License-Identifier: Apache-2.0
// Muse uses the existing lossless NZ W4 packing and FP16 Cube products.
#include "compat_310p.h"
#include "qwen_w4_group_matmul_v310_tiling_data.h"
#include "qwen_w4_group_matmul_v310.h"

extern "C" __global__ __aicore__ void muse_w4_native(
    GM_ADDR x, GM_ADDR codes, GM_ADDR scales, GM_ADDR offsets,
    GM_ADDR routes, GM_ADDR output, GM_ADDR scratch, GM_ADDR geometry) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  auto meta = reinterpret_cast<__gm__ int64_t*>(geometry);
  const int64_t rows = meta[0], n = meta[1], k = meta[2], experts = meta[3], mode = meta[4];
  const uint32_t tiles = n / NsQwenW4::QW4_TILE_N;
  const uint32_t core = AscendC::GetBlockIdx(), cores = AscendC::GetBlockNum();
  NsQwenW4::QwenW4GroupMatmulV310Cube op;
  AscendC::GlobalTensor<int32_t> ids;
  ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(routes));
  AscendC::GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(routes));
  const int64_t groups = mode == 0 ? rows : experts;
  for (int64_t group = 0; group < groups; ++group) {
    const int64_t expert = mode == 0 ? ids.GetValue(group) : group;
    const int64_t start = mode == 0 ? group : (group == 0 ? 0 : ends.GetValue(group - 1));
    const int64_t end = mode == 0 ? group + 1 : ends.GetValue(group);
    if (start == end) continue;
    const int64_t metadata = expert * n * (k / 128);
    for (uint32_t tile = core; tile < tiles; tile += cores) {
      op.InitGeometry(x + start * k * sizeof(half), codes + expert * n * k / 2,
                      scales + metadata * sizeof(half), offsets + metadata,
                      output + start * n * sizeof(half), scratch,
                      end - start, n, k, true);
      op.SetTileRange(tile, tiles);
      op.SetWorkspaceTile(core);
      op.Process();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}
