# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
generated_kernel_0 = async_compile.metal('generated_kernel_0', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_0(
        device bfloat* out_ptr1,
        constant long* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = in_ptr0[x0];
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp2 = static_cast<float>(tmp1);
            auto tmp3 = tmp2 * tmp2;
            tmp_acc_1 += tmp3;
        }
        auto tmp4 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp5 = static_cast<float>(in_ptr2[r0_1]);
            auto tmp6 = in_ptr0[x0];
            if ((tmp6 < 0 || tmp6 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp6, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp7 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp6]);
            auto tmp8 = static_cast<float>(tmp7);
            auto tmp9 = 2048.0;
            auto tmp10 = tmp4 / tmp9;
            auto tmp11 = 1e-06;
            auto tmp12 = tmp10 + tmp11;
            auto tmp13 = metal::precise::rsqrt(tmp12);
            auto tmp14 = tmp8 * tmp13;
            auto tmp15 = static_cast<bfloat>(tmp14);
            auto tmp16 = tmp5 * tmp15;
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp16);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_1 = async_compile.metal('generated_kernel_1', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_1(
        device bfloat* out_ptr2,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant float* in_ptr2,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[4];
        int x3 = c10::metal::floor_divide(xindex, 16);
        auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 128*x0]);
        auto tmp1 = static_cast<float>(tmp0);
        auto tmp2 = tmp1 * tmp1;
        auto tmp3 = c10::metal::threadgroup_sum(tmp_acc_0, tmp2, r0_index * 1, 128);
        auto tmp16 = in_ptr2[(r0_1) % (64)];
        auto tmp23 = static_cast<float>(in_ptr1[r0_1]);
        auto tmp4 = r0_1;
        auto tmp5 = static_cast<long>(tmp4);
        auto tmp6 = 0;
        auto tmp7 = tmp5 >= tmp6;
        auto tmp8 = 64;
        auto tmp9 = tmp5 < tmp8;
        float tmp10;
        if (tmp9) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[64 + (r0_1)]);
            auto tmp_scoped1 = static_cast<float>(in_ptr0[64 + 128*x0 + (r0_1)]);
            auto tmp_scoped2 = static_cast<float>(tmp_scoped1);
            auto tmp_scoped3 = 128.0;
            auto tmp_scoped4 = tmp3 / tmp_scoped3;
            auto tmp_scoped5 = 1e-06;
            auto tmp_scoped6 = tmp_scoped4 + tmp_scoped5;
            auto tmp_scoped7 = metal::precise::rsqrt(tmp_scoped6);
            auto tmp_scoped8 = tmp_scoped2 * tmp_scoped7;
            auto tmp_scoped9 = static_cast<bfloat>(tmp_scoped8);
            auto tmp_scoped10 = tmp_scoped0 * tmp_scoped9;
            auto tmp_scoped11 = static_cast<decltype(tmp_scoped10)>(-tmp_scoped10);
            tmp10 = static_cast<decltype(tmp10)>(tmp_scoped11);
        } else tmp10 = static_cast<decltype(tmp10)>(0.0);
        auto tmp11 = tmp5 >= tmp8;
        auto tmp12 = 128;
        auto tmp13 = tmp5 < tmp12;
        float tmp14;
        if (tmp11) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[(-64) + r0_1]);
            auto tmp_scoped1 = static_cast<float>(in_ptr0[128*x0 + ((-64) + r0_1)]);
            auto tmp_scoped2 = static_cast<float>(tmp_scoped1);
            auto tmp_scoped3 = 128.0;
            auto tmp_scoped4 = tmp3 / tmp_scoped3;
            auto tmp_scoped5 = 1e-06;
            auto tmp_scoped6 = tmp_scoped4 + tmp_scoped5;
            auto tmp_scoped7 = metal::precise::rsqrt(tmp_scoped6);
            auto tmp_scoped8 = tmp_scoped2 * tmp_scoped7;
            auto tmp_scoped9 = static_cast<bfloat>(tmp_scoped8);
            auto tmp_scoped10 = tmp_scoped0 * tmp_scoped9;
            tmp14 = static_cast<decltype(tmp14)>(tmp_scoped10);
        } else tmp14 = static_cast<decltype(tmp14)>(0.0);
        auto tmp15 = tmp9 ? tmp10 : static_cast<decltype(tmp10)>(tmp14);
        auto tmp17 = x3;
        auto tmp18 = static_cast<float>(tmp17);
        auto tmp19 = tmp18 * tmp16;
        auto tmp20 = metal::precise::sin(tmp19);
        auto tmp21 = static_cast<bfloat>(tmp20);
        auto tmp22 = tmp15 * tmp21;
        auto tmp24 = 128.0;
        auto tmp25 = tmp3 / tmp24;
        auto tmp26 = 1e-06;
        auto tmp27 = tmp25 + tmp26;
        auto tmp28 = metal::precise::rsqrt(tmp27);
        auto tmp29 = tmp1 * tmp28;
        auto tmp30 = static_cast<bfloat>(tmp29);
        auto tmp31 = tmp23 * tmp30;
        auto tmp32 = metal::precise::cos(tmp19);
        auto tmp33 = static_cast<bfloat>(tmp32);
        auto tmp34 = tmp31 * tmp33;
        auto tmp35 = tmp34 + tmp22;
        out_ptr2[r0_1 + 128*x0] = static_cast<bfloat>(tmp35);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_2 = async_compile.metal('generated_kernel_2', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_2(
        device bfloat* out_ptr2,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant float* in_ptr2,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[4];
        int x3 = c10::metal::floor_divide(xindex, 8);
        auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 128*x0]);
        auto tmp1 = static_cast<float>(tmp0);
        auto tmp2 = tmp1 * tmp1;
        auto tmp3 = c10::metal::threadgroup_sum(tmp_acc_0, tmp2, r0_index * 1, 128);
        auto tmp16 = in_ptr2[(r0_1) % (64)];
        auto tmp23 = static_cast<float>(in_ptr1[r0_1]);
        auto tmp4 = r0_1;
        auto tmp5 = static_cast<long>(tmp4);
        auto tmp6 = 0;
        auto tmp7 = tmp5 >= tmp6;
        auto tmp8 = 64;
        auto tmp9 = tmp5 < tmp8;
        float tmp10;
        if (tmp9) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[64 + (r0_1)]);
            auto tmp_scoped1 = static_cast<float>(in_ptr0[64 + 128*x0 + (r0_1)]);
            auto tmp_scoped2 = static_cast<float>(tmp_scoped1);
            auto tmp_scoped3 = 128.0;
            auto tmp_scoped4 = tmp3 / tmp_scoped3;
            auto tmp_scoped5 = 1e-06;
            auto tmp_scoped6 = tmp_scoped4 + tmp_scoped5;
            auto tmp_scoped7 = metal::precise::rsqrt(tmp_scoped6);
            auto tmp_scoped8 = tmp_scoped2 * tmp_scoped7;
            auto tmp_scoped9 = static_cast<bfloat>(tmp_scoped8);
            auto tmp_scoped10 = tmp_scoped0 * tmp_scoped9;
            auto tmp_scoped11 = static_cast<decltype(tmp_scoped10)>(-tmp_scoped10);
            tmp10 = static_cast<decltype(tmp10)>(tmp_scoped11);
        } else tmp10 = static_cast<decltype(tmp10)>(0.0);
        auto tmp11 = tmp5 >= tmp8;
        auto tmp12 = 128;
        auto tmp13 = tmp5 < tmp12;
        float tmp14;
        if (tmp11) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[(-64) + r0_1]);
            auto tmp_scoped1 = static_cast<float>(in_ptr0[128*x0 + ((-64) + r0_1)]);
            auto tmp_scoped2 = static_cast<float>(tmp_scoped1);
            auto tmp_scoped3 = 128.0;
            auto tmp_scoped4 = tmp3 / tmp_scoped3;
            auto tmp_scoped5 = 1e-06;
            auto tmp_scoped6 = tmp_scoped4 + tmp_scoped5;
            auto tmp_scoped7 = metal::precise::rsqrt(tmp_scoped6);
            auto tmp_scoped8 = tmp_scoped2 * tmp_scoped7;
            auto tmp_scoped9 = static_cast<bfloat>(tmp_scoped8);
            auto tmp_scoped10 = tmp_scoped0 * tmp_scoped9;
            tmp14 = static_cast<decltype(tmp14)>(tmp_scoped10);
        } else tmp14 = static_cast<decltype(tmp14)>(0.0);
        auto tmp15 = tmp9 ? tmp10 : static_cast<decltype(tmp10)>(tmp14);
        auto tmp17 = x3;
        auto tmp18 = static_cast<float>(tmp17);
        auto tmp19 = tmp18 * tmp16;
        auto tmp20 = metal::precise::sin(tmp19);
        auto tmp21 = static_cast<bfloat>(tmp20);
        auto tmp22 = tmp15 * tmp21;
        auto tmp24 = 128.0;
        auto tmp25 = tmp3 / tmp24;
        auto tmp26 = 1e-06;
        auto tmp27 = tmp25 + tmp26;
        auto tmp28 = metal::precise::rsqrt(tmp27);
        auto tmp29 = tmp1 * tmp28;
        auto tmp30 = static_cast<bfloat>(tmp29);
        auto tmp31 = tmp23 * tmp30;
        auto tmp32 = metal::precise::cos(tmp19);
        auto tmp33 = static_cast<bfloat>(tmp32);
        auto tmp34 = tmp31 * tmp33;
        auto tmp35 = tmp34 + tmp22;
        out_ptr2[r0_1 + 128*x0] = static_cast<bfloat>(tmp35);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_3 = async_compile.metal('generated_kernel_3', '''

    kernel void generated_kernel_3(
        device long* out_ptr0,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        auto tmp0 = -1;
        out_ptr0[0] = static_cast<long>(tmp0);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_4 = async_compile.metal('generated_kernel_4', '''

    kernel void generated_kernel_4(
        device long* out_ptr0,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp0 = x0;
        auto tmp1 = static_cast<long>(tmp0);
        out_ptr0[x0] = static_cast<long>(tmp1);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_5 = async_compile.metal('generated_kernel_5', '''

    kernel void generated_kernel_5(
        device long* out_ptr0,
        constant long* in_ptr0,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp0 = in_ptr0[1 + x0];
        auto tmp1 = in_ptr0[x0];
        auto tmp2 = tmp0 - tmp1;
        auto tmp3 = 1;
        auto tmp4 = tmp2 != tmp3;
        auto tmp5 = static_cast<long>(tmp4);
        out_ptr0[x0] = static_cast<long>(tmp5);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_6 = async_compile.metal('generated_kernel_6', '''

    kernel void generated_kernel_6(
        device bfloat* out_ptr0,
        device bfloat* out_ptr1,
        constant long* in_ptr0,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (71);
        int x1 = c10::metal::floor_divide(xindex, 71);
        int x2 = xindex;
        auto tmp7 = in_ptr0[x1];
        auto tmp8 = in_ptr0[x0];
        auto tmp0 = x0;
        auto tmp1 = static_cast<long>(tmp0);
        auto tmp2 = x1;
        auto tmp3 = static_cast<long>(tmp2);
        auto tmp4 = tmp1 <= tmp3;
        auto tmp5 = true;
        auto tmp6 = tmp5 & tmp4;
        auto tmp9 = tmp7 == tmp8;
        auto tmp10 = tmp6 & tmp9;
        auto tmp11 = 0.0;
        auto tmp12 = -HUGE_VALF;
        auto tmp13 = tmp10 ? tmp11 : static_cast<decltype(tmp11)>(tmp12);
        out_ptr0[x2] = static_cast<bfloat>(tmp13);
        out_ptr1[x2] = static_cast<bfloat>(tmp13);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_7 = async_compile.metal('generated_kernel_7', '''

    kernel void generated_kernel_7(
        device bfloat* out_ptr0,
        constant bfloat* in_ptr0,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (128);
        int x1 = ((xindex) / (128)) % (16);
        int x2 = c10::metal::floor_divide(xindex, 2048);
        int x3 = xindex;
        auto tmp0 = static_cast<float>(in_ptr0[x0 + 128*x2 + 9088*x1]);
        out_ptr0[x3] = static_cast<bfloat>(tmp0);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_8 = async_compile.metal('generated_kernel_8', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_8(
        device bfloat* out_ptr1,
        constant long* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant bfloat* in_ptr3,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = in_ptr0[x0];
            auto tmp2 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(tmp3);
            auto tmp5 = tmp4 * tmp4;
            tmp_acc_1 += tmp5;
        }
        auto tmp6 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp7 = static_cast<float>(in_ptr3[r0_1]);
            auto tmp8 = in_ptr0[x0];
            auto tmp10 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            if ((tmp8 < 0 || tmp8 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp8, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp9 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp8]);
            auto tmp11 = tmp9 + tmp10;
            auto tmp12 = static_cast<float>(tmp11);
            auto tmp13 = 2048.0;
            auto tmp14 = tmp6 / tmp13;
            auto tmp15 = 1e-06;
            auto tmp16 = tmp14 + tmp15;
            auto tmp17 = metal::precise::rsqrt(tmp16);
            auto tmp18 = tmp12 * tmp17;
            auto tmp19 = static_cast<bfloat>(tmp18);
            auto tmp20 = tmp7 * tmp19;
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp20);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_9 = async_compile.metal('generated_kernel_9', '''

    kernel void generated_kernel_9(
        device bfloat* out_ptr0,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp0 = static_cast<float>(in_ptr0[x0]);
        auto tmp8 = static_cast<float>(in_ptr1[x0]);
        auto tmp1 = static_cast<float>(tmp0);
        auto tmp2 = static_cast<decltype(tmp1)>(-tmp1);
        auto tmp3 = metal::precise::exp(tmp2);
        auto tmp4 = 1.0;
        auto tmp5 = tmp3 + tmp4;
        auto tmp6 = tmp1 / tmp5;
        auto tmp7 = static_cast<bfloat>(tmp6);
        auto tmp9 = tmp7 * tmp8;
        out_ptr0[x0] = static_cast<bfloat>(tmp9);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_10 = async_compile.metal('generated_kernel_10', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_10(
        device bfloat* out_ptr1,
        constant long* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant bfloat* in_ptr3,
        constant bfloat* in_ptr4,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = in_ptr0[x0];
            auto tmp2 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp4 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp3 = tmp1 + tmp2;
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = static_cast<float>(tmp5);
            auto tmp7 = tmp6 * tmp6;
            tmp_acc_1 += tmp7;
        }
        auto tmp8 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp9 = static_cast<float>(in_ptr4[r0_1]);
            auto tmp10 = in_ptr0[x0];
            auto tmp12 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp14 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            if ((tmp10 < 0 || tmp10 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp10, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp11 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp10]);
            auto tmp13 = tmp11 + tmp12;
            auto tmp15 = tmp13 + tmp14;
            auto tmp16 = static_cast<float>(tmp15);
            auto tmp17 = 2048.0;
            auto tmp18 = tmp8 / tmp17;
            auto tmp19 = 1e-06;
            auto tmp20 = tmp18 + tmp19;
            auto tmp21 = metal::precise::rsqrt(tmp20);
            auto tmp22 = tmp16 * tmp21;
            auto tmp23 = static_cast<bfloat>(tmp22);
            auto tmp24 = tmp9 * tmp23;
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp24);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_11 = async_compile.metal('generated_kernel_11', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_11(
        device bfloat* out_ptr0,
        device bfloat* out_ptr2,
        constant long* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant bfloat* in_ptr3,
        constant bfloat* in_ptr4,
        constant bfloat* in_ptr5,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = in_ptr0[x0];
            auto tmp2 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp4 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp6 = static_cast<float>(in_ptr4[r0_1 + 2048*x0]);
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp3 = tmp1 + tmp2;
            auto tmp5 = tmp3 + tmp4;
            auto tmp7 = tmp5 + tmp6;
            out_ptr0[r0_1 + 2048*x0] = static_cast<bfloat>(tmp7);
            auto tmp8 = static_cast<float>(tmp7);
            auto tmp9 = tmp8 * tmp8;
            tmp_acc_1 += tmp9;
        }
        auto tmp10 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp11 = static_cast<float>(in_ptr5[r0_1]);
            auto tmp12 = static_cast<float>(out_ptr0[r0_1 + 2048*x0]);
            auto tmp13 = static_cast<float>(tmp12);
            auto tmp14 = 2048.0;
            auto tmp15 = tmp10 / tmp14;
            auto tmp16 = 1e-06;
            auto tmp17 = tmp15 + tmp16;
            auto tmp18 = metal::precise::rsqrt(tmp17);
            auto tmp19 = tmp13 * tmp18;
            auto tmp20 = static_cast<bfloat>(tmp19);
            auto tmp21 = tmp11 * tmp20;
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp21);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_12 = async_compile.metal('generated_kernel_12', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_12(
        device bfloat* out_ptr1,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = static_cast<float>(tmp2);
            auto tmp4 = tmp3 * tmp3;
            tmp_acc_1 += tmp4;
        }
        auto tmp5 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp6 = static_cast<float>(in_ptr2[r0_1]);
            auto tmp7 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp8 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp9 = tmp7 + tmp8;
            auto tmp10 = static_cast<float>(tmp9);
            auto tmp11 = 2048.0;
            auto tmp12 = tmp5 / tmp11;
            auto tmp13 = 1e-06;
            auto tmp14 = tmp12 + tmp13;
            auto tmp15 = metal::precise::rsqrt(tmp14);
            auto tmp16 = tmp10 * tmp15;
            auto tmp17 = static_cast<bfloat>(tmp16);
            auto tmp18 = tmp6 * tmp17;
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp18);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_13 = async_compile.metal('generated_kernel_13', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_13(
        device bfloat* out_ptr1,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant bfloat* in_ptr3,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp3 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = static_cast<float>(tmp4);
            auto tmp6 = tmp5 * tmp5;
            tmp_acc_1 += tmp6;
        }
        auto tmp7 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp8 = static_cast<float>(in_ptr3[r0_1]);
            auto tmp9 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp10 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp12 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp11 = tmp9 + tmp10;
            auto tmp13 = tmp11 + tmp12;
            auto tmp14 = static_cast<float>(tmp13);
            auto tmp15 = 2048.0;
            auto tmp16 = tmp7 / tmp15;
            auto tmp17 = 1e-06;
            auto tmp18 = tmp16 + tmp17;
            auto tmp19 = metal::precise::rsqrt(tmp18);
            auto tmp20 = tmp14 * tmp19;
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = tmp8 * tmp21;
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp22);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_14 = async_compile.metal('generated_kernel_14', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_14(
        device bfloat* out_ptr1,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant bfloat* in_ptr3,
        constant bfloat* in_ptr4,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp3 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp5 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = static_cast<float>(tmp6);
            auto tmp8 = tmp7 * tmp7;
            tmp_acc_1 += tmp8;
        }
        auto tmp9 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp10 = static_cast<float>(in_ptr4[r0_1]);
            auto tmp11 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp12 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp14 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp16 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp13 = tmp11 + tmp12;
            auto tmp15 = tmp13 + tmp14;
            auto tmp17 = tmp15 + tmp16;
            auto tmp18 = static_cast<float>(tmp17);
            auto tmp19 = 2048.0;
            auto tmp20 = tmp9 / tmp19;
            auto tmp21 = 1e-06;
            auto tmp22 = tmp20 + tmp21;
            auto tmp23 = metal::precise::rsqrt(tmp22);
            auto tmp24 = tmp18 * tmp23;
            auto tmp25 = static_cast<bfloat>(tmp24);
            auto tmp26 = tmp10 * tmp25;
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp26);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_15 = async_compile.metal('generated_kernel_15', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_15(
        device bfloat* out_ptr0,
        device bfloat* out_ptr2,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant bfloat* in_ptr3,
        constant bfloat* in_ptr4,
        constant bfloat* in_ptr5,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp3 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp5 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp7 = static_cast<float>(in_ptr4[r0_1 + 2048*x0]);
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp8 = tmp6 + tmp7;
            out_ptr0[r0_1 + 2048*x0] = static_cast<bfloat>(tmp8);
            auto tmp9 = static_cast<float>(tmp8);
            auto tmp10 = tmp9 * tmp9;
            tmp_acc_1 += tmp10;
        }
        auto tmp11 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp12 = static_cast<float>(in_ptr5[r0_1]);
            auto tmp13 = static_cast<float>(out_ptr0[r0_1 + 2048*x0]);
            auto tmp14 = static_cast<float>(tmp13);
            auto tmp15 = 2048.0;
            auto tmp16 = tmp11 / tmp15;
            auto tmp17 = 1e-06;
            auto tmp18 = tmp16 + tmp17;
            auto tmp19 = metal::precise::rsqrt(tmp18);
            auto tmp20 = tmp14 * tmp19;
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = tmp12 * tmp21;
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp22);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_16 = async_compile.metal('generated_kernel_16', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_16(
        device bfloat* out_ptr1,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        device c10::metal::ErrorMessages* error_buf,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[4];
        auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 128*x0]);
        auto tmp1 = static_cast<float>(tmp0);
        auto tmp2 = tmp1 * tmp1;
        auto tmp3 = c10::metal::threadgroup_sum(tmp_acc_0, tmp2, r0_index * 1, 128);
        auto tmp4 = static_cast<float>(in_ptr1[r0_1]);
        auto tmp5 = 128.0;
        auto tmp6 = tmp3 / tmp5;
        auto tmp7 = 1e-06;
        auto tmp8 = tmp6 + tmp7;
        auto tmp9 = metal::precise::rsqrt(tmp8);
        auto tmp10 = tmp1 * tmp9;
        auto tmp11 = static_cast<bfloat>(tmp10);
        auto tmp12 = tmp4 * tmp11;
        out_ptr1[r0_1 + 128*x0] = static_cast<bfloat>(tmp12);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_17 = async_compile.metal('generated_kernel_17', '''

    kernel void generated_kernel_17(
        device bfloat* out_ptr0,
        constant bfloat* in_ptr0,
        constant float* in_ptr1,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x3 = xindex;
        int x2 = c10::metal::floor_divide(xindex, 2048);
        int x0 = (xindex) % (128);
        int x4 = c10::metal::floor_divide(xindex, 128);
        auto tmp0 = static_cast<float>(in_ptr0[x3]);
        auto tmp1 = in_ptr1[(x3) % (64)];
        auto tmp2 = x2;
        auto tmp3 = static_cast<float>(tmp2);
        auto tmp4 = tmp3 * tmp1;
        auto tmp5 = metal::precise::cos(tmp4);
        auto tmp6 = static_cast<bfloat>(tmp5);
        auto tmp7 = tmp0 * tmp6;
        auto tmp8 = x0;
        auto tmp9 = static_cast<long>(tmp8);
        auto tmp10 = 0;
        auto tmp11 = tmp9 >= tmp10;
        auto tmp12 = 64;
        auto tmp13 = tmp9 < tmp12;
        float tmp14;
        if (tmp13) {
            auto tmp_scoped0 = static_cast<float>(in_ptr0[64 + 128*x4 + (x0)]);
            auto tmp_scoped1 = static_cast<decltype(tmp_scoped0)>(-tmp_scoped0);
            tmp14 = static_cast<decltype(tmp14)>(tmp_scoped1);
        } else tmp14 = static_cast<decltype(tmp14)>(0.0);
        auto tmp15 = tmp9 >= tmp12;
        auto tmp16 = 128;
        auto tmp17 = tmp9 < tmp16;
        float tmp18;
        if (tmp15) {
            auto tmp_scoped0 = static_cast<float>(in_ptr0[128*x4 + ((-64) + x0)]);
            tmp18 = static_cast<decltype(tmp18)>(tmp_scoped0);
        } else tmp18 = static_cast<decltype(tmp18)>(0.0);
        auto tmp19 = tmp13 ? tmp14 : static_cast<decltype(tmp14)>(tmp18);
        auto tmp20 = metal::precise::sin(tmp4);
        auto tmp21 = static_cast<bfloat>(tmp20);
        auto tmp22 = tmp19 * tmp21;
        auto tmp23 = tmp7 + tmp22;
        out_ptr0[x3] = static_cast<bfloat>(tmp23);
    }

''', ['error', 'reduction_utils', 'utils'])


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1 = args
        args.clear()
        assert_size_stride_grouped((arg0_1, arg1_1, arg3_1), ((1, 71), (151936, 2048), (2048, )), ((71, 1), (2048, 1), (1, )), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf1 = empty_strided((1, 71, 2048), (145408, 2048, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_0(buf1, arg0_1, arg1_1, arg3_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg3_1
        assert_size_stride(arg4_1, (2048, 2048), (2048, 1), 'input')
        buf2 = empty_strided((71, 2048), (2048, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_3, rsqrt, hidden_states_1, to_6, hidden_states_2, linear], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf1, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg4_1, (2048, 2048), (1, 2048), 0), out=buf2)
        del arg4_1
        assert_size_stride_grouped((arg5_1, arg2_1), ((128, ), (64, )), ((1, ), (1, )), 'input')
        buf14 = empty_strided((1, 16, 71, 128), (145408, 128, 2048, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_1(buf14, buf2, arg5_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg5_1
        assert_size_stride(arg6_1, (1024, 2048), (2048, 1), 'input')
        buf5 = empty_strided((71, 1024), (1024, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf1, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg6_1, (2048, 1024), (1, 2048), 0), out=buf5)
        del arg6_1
        assert_size_stride(arg7_1, (128, ), (1, ), 'input')
        buf15 = empty_strided((1, 8, 71, 128), (72704, 128, 1024, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_2(buf15, buf5, arg7_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg7_1
        assert_size_stride(arg8_1, (1024, 2048), (2048, 1), 'input')
        buf8 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf1, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg8_1, (2048, 1024), (1, 2048), 0), out=buf8)
        del arg8_1
        buf11 = empty_strided((1, 72), (72, 1), device='mps', dtype=torch.int64)
        buf9 = reinterpret_tensor(buf11, (1, 1), (72, 1), 0)  # alias
        generated_kernel_3(buf9, threads=[1], error_buf_idx=1)
        buf10 = reinterpret_tensor(buf11, (1, 71), (72, 1), 1)  # alias
        generated_kernel_4(buf10, threads=[71], error_buf_idx=1)
        buf12 = empty_strided((1, 71), (71, 1), device='mps', dtype=torch.int64)
        generated_kernel_5(buf12, buf11, threads=[71], error_buf_idx=2)
        del buf10
        del buf11
        del buf9
        # Topologically Sorted Source Nodes: [position_diff, ne, packed_sequence_mask], Original ATen: [aten.slice, aten.sub, aten.ne, aten.cumsum]
        buf13 = torch.ops.aten.cumsum.default(buf12, dim=-1, dtype=torch.int64)
        assert_tensor_metadata(buf13, (1, 71), (71, 1), torch.int64, 'torch.ops.aten.cumsum.default')
        assert_alignment(buf13, 16, 'torch.ops.aten.cumsum.default')
        del buf12
        buf16 = empty_strided((1, 1, 71, 71), (5056, 5056, 71, 1), device='mps', dtype=torch.bfloat16)
        buf39 = empty_strided((1, 1, 71, 71), (5056, 5056, 71, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_6(buf16, buf39, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_8, mul_6, query_states, position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, cos_3, mul_9, q_embed, linear_1, view_1, hidden_states_5, pow_3, variance_2, add_5, rsqrt_2, hidden_states_6, to_10, mul_8, key_states, mul_11, k_embed, linear_2, view_2, value_states, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, attn_output], Original ATen: [aten.linear, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf17 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf14, buf15, reinterpret_tensor(buf8, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf16, scale=0.08838834764831845, enable_gqa=True)
        buf18 = buf17[0]
        assert_tensor_metadata(buf18, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf17
        buf20 = reinterpret_tensor(buf14, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf14  # reuse
        generated_kernel_7(buf20, buf18, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg9_1, (2048, 2048), (2048, 1), 'input')
        buf21 = reinterpret_tensor(buf18, (71, 2048), (2048, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, attn_output_1, reshape, attn_output_3], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf20, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg9_1, (2048, 2048), (1, 2048), 0), out=buf21)
        del arg9_1
        assert_size_stride(arg10_1, (2048, ), (1, ), 'input')
        buf23 = reinterpret_tensor(buf20, (1, 71, 2048), (145408, 2048, 1), 0); del buf20  # reuse
        generated_kernel_8(buf23, arg0_1, arg1_1, buf21, arg10_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg10_1
        assert_size_stride(arg11_1, (6144, 2048), (2048, 1), 'input')
        buf24 = empty_strided((71, 6144), (6144, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf23, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 6144), (1, 2048), 0), out=buf24)
        del arg11_1
        assert_size_stride(arg12_1, (6144, 2048), (2048, 1), 'input')
        buf25 = empty_strided((71, 6144), (6144, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf23, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg12_1, (2048, 6144), (1, 2048), 0), out=buf25)
        del arg12_1
        buf26 = empty_strided((1, 71, 6144), (436224, 6144, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_9(buf26, buf24, buf25, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg13_1, (2048, 6144), (6144, 1), 'input')
        buf27 = reinterpret_tensor(buf23, (71, 2048), (2048, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_15, down_proj], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf26, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg13_1, (6144, 2048), (1, 6144), 0), out=buf27)
        del arg13_1
        assert_size_stride(arg14_1, (2048, ), (1, ), 'input')
        buf29 = buf1; del buf1  # reuse
        generated_kernel_10(buf29, arg0_1, arg1_1, buf21, buf27, arg14_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg14_1
        assert_size_stride(arg15_1, (2048, 2048), (2048, 1), 'input')
        buf30 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf29, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg15_1, (2048, 2048), (1, 2048), 0), out=buf30)
        del arg15_1
        assert_size_stride(arg16_1, (128, ), (1, ), 'input')
        buf37 = empty_strided((1, 16, 71, 128), (145408, 128, 2048, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_1(buf37, buf30, arg16_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg16_1
        assert_size_stride(arg17_1, (1024, 2048), (2048, 1), 'input')
        buf33 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf29, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 1024), (1, 2048), 0), out=buf33)
        del arg17_1
        assert_size_stride(arg18_1, (128, ), (1, ), 'input')
        buf38 = buf15; del buf15  # reuse
        generated_kernel_2(buf38, buf33, arg18_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg18_1
        assert_size_stride(arg19_1, (1024, 2048), (2048, 1), 'input')
        buf36 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf29, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg19_1, (2048, 1024), (1, 2048), 0), out=buf36)
        del arg19_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_7, view_3, hidden_states_15, pow_6, variance_5, add_12, rsqrt_5, hidden_states_16, to_16, mul_19, query_states_1, cos_4, mul_22, q_embed_1, linear_8, view_4, hidden_states_17, pow_7, variance_6, add_13, rsqrt_6, hidden_states_18, to_18, mul_21, key_states_1, mul_24, k_embed_1, linear_9, view_5, value_states_1, attn_output_4], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf40 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf37, buf38, reinterpret_tensor(buf36, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf39, scale=0.08838834764831845, enable_gqa=True)
        buf41 = buf40[0]
        assert_tensor_metadata(buf41, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf41, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf40
        buf43 = reinterpret_tensor(buf37, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf37  # reuse
        generated_kernel_7(buf43, buf41, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg20_1, (2048, 2048), (2048, 1), 'input')
        buf44 = reinterpret_tensor(buf41, (71, 2048), (2048, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [transpose_7, attn_output_5, reshape_1, attn_output_7], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf43, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg20_1, (2048, 2048), (1, 2048), 0), out=buf44)
        del arg20_1
        assert_size_stride(arg21_1, (2048, ), (1, ), 'input')
        buf45 = reinterpret_tensor(buf43, (1, 71, 2048), (145408, 2048, 1), 0); del buf43  # reuse
        buf47 = buf29; del buf29  # reuse
        generated_kernel_11(buf45, buf47, arg0_1, arg1_1, buf21, buf27, buf44, arg21_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg0_1
        del arg21_1
        assert_size_stride(arg22_1, (6144, 2048), (2048, 1), 'input')
        buf48 = reinterpret_tensor(buf26, (71, 6144), (6144, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, pow_8, variance_7, add_17, rsqrt_7, hidden_states_21, to_20, hidden_states_22, linear_11], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf47, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg22_1, (2048, 6144), (1, 2048), 0), out=buf48)
        del arg22_1
        assert_size_stride(arg23_1, (6144, 2048), (2048, 1), 'input')
        buf49 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf47, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg23_1, (2048, 6144), (1, 2048), 0), out=buf49)
        del arg23_1
        buf50 = reinterpret_tensor(buf24, (1, 71, 6144), (436224, 6144, 1), 0); del buf24  # reuse
        generated_kernel_9(buf50, buf48, buf49, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg24_1, (2048, 6144), (6144, 1), 'input')
        buf51 = reinterpret_tensor(buf47, (71, 2048), (2048, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_28, down_proj_1], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf50, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg24_1, (6144, 2048), (1, 6144), 0), out=buf51)
        del arg24_1
        assert_size_stride(arg25_1, (2048, ), (1, ), 'input')
        buf53 = reinterpret_tensor(buf44, (1, 71, 2048), (145408, 2048, 1), 0); del buf44  # reuse
        generated_kernel_12(buf53, buf45, buf51, arg25_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg25_1
        assert_size_stride(arg26_1, (2048, 2048), (2048, 1), 'input')
        buf54 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_23, hidden_states_24, pow_9, variance_8, add_19, rsqrt_8, hidden_states_25, to_22, hidden_states_26, linear_14], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf53, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg26_1, (2048, 2048), (1, 2048), 0), out=buf54)
        del arg26_1
        assert_size_stride(arg27_1, (128, ), (1, ), 'input')
        buf61 = reinterpret_tensor(buf21, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf21  # reuse
        generated_kernel_1(buf61, buf54, arg27_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg27_1
        assert_size_stride(arg28_1, (1024, 2048), (2048, 1), 'input')
        buf57 = reinterpret_tensor(buf38, (71, 1024), (1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf53, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg28_1, (2048, 1024), (1, 2048), 0), out=buf57)
        del arg28_1
        assert_size_stride(arg29_1, (128, ), (1, ), 'input')
        buf62 = reinterpret_tensor(buf36, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf36  # reuse
        generated_kernel_2(buf62, buf57, arg29_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg29_1
        assert_size_stride(arg30_1, (1024, 2048), (2048, 1), 'input')
        buf60 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf53, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg30_1, (2048, 1024), (1, 2048), 0), out=buf60)
        del arg30_1
        buf63 = buf39; del buf39  # reuse
        buf86 = buf16; del buf16  # reuse
        generated_kernel_6(buf63, buf86, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_14, view_6, hidden_states_27, pow_10, variance_9, add_20, rsqrt_9, hidden_states_28, to_24, mul_32, query_states_2, cos_5, mul_35, q_embed_2, linear_15, view_7, hidden_states_29, pow_11, variance_10, add_21, rsqrt_10, hidden_states_30, to_26, mul_34, key_states_2, mul_37, k_embed_2, linear_16, view_8, value_states_2, attn_output_8], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf64 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf61, buf62, reinterpret_tensor(buf60, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf63, scale=0.08838834764831845, enable_gqa=True)
        buf65 = buf64[0]
        assert_tensor_metadata(buf65, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf65, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf64
        buf67 = reinterpret_tensor(buf61, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf61  # reuse
        generated_kernel_7(buf67, buf65, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg31_1, (2048, 2048), (2048, 1), 'input')
        buf68 = reinterpret_tensor(buf65, (71, 2048), (2048, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [transpose_11, attn_output_9, reshape_2, attn_output_11], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf67, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg31_1, (2048, 2048), (1, 2048), 0), out=buf68)
        del arg31_1
        assert_size_stride(arg32_1, (2048, ), (1, ), 'input')
        buf70 = reinterpret_tensor(buf67, (1, 71, 2048), (145408, 2048, 1), 0); del buf67  # reuse
        generated_kernel_13(buf70, buf45, buf51, buf68, arg32_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg32_1
        assert_size_stride(arg33_1, (6144, 2048), (2048, 1), 'input')
        buf71 = reinterpret_tensor(buf50, (71, 6144), (6144, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf70, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 6144), (1, 2048), 0), out=buf71)
        del arg33_1
        assert_size_stride(arg34_1, (6144, 2048), (2048, 1), 'input')
        buf72 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf70, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg34_1, (2048, 6144), (1, 2048), 0), out=buf72)
        del arg34_1
        buf73 = reinterpret_tensor(buf48, (1, 71, 6144), (436224, 6144, 1), 0); del buf48  # reuse
        generated_kernel_9(buf73, buf71, buf72, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg35_1, (2048, 6144), (6144, 1), 'input')
        buf74 = reinterpret_tensor(buf70, (71, 2048), (2048, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_41, down_proj_2], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf73, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg35_1, (6144, 2048), (1, 6144), 0), out=buf74)
        del arg35_1
        assert_size_stride(arg36_1, (2048, ), (1, ), 'input')
        buf76 = buf53; del buf53  # reuse
        generated_kernel_14(buf76, buf45, buf51, buf68, buf74, arg36_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg36_1
        assert_size_stride(arg37_1, (2048, 2048), (2048, 1), 'input')
        buf77 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf76, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg37_1, (2048, 2048), (1, 2048), 0), out=buf77)
        del arg37_1
        assert_size_stride(arg38_1, (128, ), (1, ), 'input')
        buf84 = reinterpret_tensor(buf30, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf30  # reuse
        generated_kernel_1(buf84, buf77, arg38_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg38_1
        del buf77
        assert_size_stride(arg39_1, (1024, 2048), (2048, 1), 'input')
        buf80 = reinterpret_tensor(buf62, (71, 1024), (1024, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf76, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg39_1, (2048, 1024), (1, 2048), 0), out=buf80)
        del arg39_1
        assert_size_stride(arg40_1, (128, ), (1, ), 'input')
        buf85 = reinterpret_tensor(buf60, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf60  # reuse
        generated_kernel_2(buf85, buf80, arg40_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg40_1
        assert_size_stride(arg41_1, (1024, 2048), (2048, 1), 'input')
        buf83 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf76, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg41_1, (2048, 1024), (1, 2048), 0), out=buf83)
        del arg41_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_21, view_9, hidden_states_39, pow_14, variance_13, add_28, rsqrt_13, hidden_states_40, to_32, mul_45, query_states_3, cos_6, mul_48, q_embed_3, linear_22, view_10, hidden_states_41, pow_15, variance_14, add_29, rsqrt_14, hidden_states_42, to_34, mul_47, key_states_3, mul_50, k_embed_3, linear_23, view_11, value_states_3, attn_output_12], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf87 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf84, buf85, reinterpret_tensor(buf83, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf86, scale=0.08838834764831845, enable_gqa=True)
        buf88 = buf87[0]
        assert_tensor_metadata(buf88, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf88, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf87
        buf90 = reinterpret_tensor(buf84, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf84  # reuse
        generated_kernel_7(buf90, buf88, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg42_1, (2048, 2048), (2048, 1), 'input')
        buf91 = reinterpret_tensor(buf88, (71, 2048), (2048, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [transpose_15, attn_output_13, reshape_3, attn_output_15], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf90, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg42_1, (2048, 2048), (1, 2048), 0), out=buf91)
        del arg42_1
        assert_size_stride(arg43_1, (2048, ), (1, ), 'input')
        buf92 = reinterpret_tensor(buf90, (1, 71, 2048), (145408, 2048, 1), 0); del buf90  # reuse
        buf94 = buf76; del buf76  # reuse
        generated_kernel_15(buf92, buf94, buf45, buf51, buf68, buf74, buf91, arg43_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg43_1
        del buf45
        assert_size_stride(arg44_1, (6144, 2048), (2048, 1), 'input')
        buf95 = reinterpret_tensor(buf73, (71, 6144), (6144, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44, pow_16, variance_15, add_33, rsqrt_15, hidden_states_45, to_36, hidden_states_46, linear_25], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf94, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 6144), (1, 2048), 0), out=buf95)
        del arg44_1
        assert_size_stride(arg45_1, (6144, 2048), (2048, 1), 'input')
        buf96 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf94, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 6144), (1, 2048), 0), out=buf96)
        del arg45_1
        buf97 = reinterpret_tensor(buf71, (1, 71, 6144), (436224, 6144, 1), 0); del buf71  # reuse
        generated_kernel_9(buf97, buf95, buf96, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg46_1, (2048, 6144), (6144, 1), 'input')
        buf98 = reinterpret_tensor(buf94, (71, 2048), (2048, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_54, down_proj_3], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf97, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg46_1, (6144, 2048), (1, 6144), 0), out=buf98)
        del arg46_1
        assert_size_stride(arg47_1, (2048, ), (1, ), 'input')
        buf100 = reinterpret_tensor(buf91, (1, 71, 2048), (145408, 2048, 1), 0); del buf91  # reuse
        generated_kernel_12(buf100, buf92, buf98, arg47_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg47_1
        assert_size_stride(arg48_1, (2048, 2048), (2048, 1), 'input')
        buf101 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_47, hidden_states_48, pow_17, variance_16, add_35, rsqrt_16, hidden_states_49, to_38, hidden_states_50, linear_28], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf100, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg48_1, (2048, 2048), (1, 2048), 0), out=buf101)
        del arg48_1
        assert_size_stride(arg49_1, (128, ), (1, ), 'input')
        buf108 = reinterpret_tensor(buf68, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf68  # reuse
        generated_kernel_1(buf108, buf101, arg49_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg49_1
        assert_size_stride(arg50_1, (1024, 2048), (2048, 1), 'input')
        buf104 = reinterpret_tensor(buf85, (71, 1024), (1024, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf100, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg50_1, (2048, 1024), (1, 2048), 0), out=buf104)
        del arg50_1
        assert_size_stride(arg51_1, (128, ), (1, ), 'input')
        buf109 = reinterpret_tensor(buf83, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf83  # reuse
        generated_kernel_2(buf109, buf104, arg51_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg51_1
        assert_size_stride(arg52_1, (1024, 2048), (2048, 1), 'input')
        buf107 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf100, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg52_1, (2048, 1024), (1, 2048), 0), out=buf107)
        del arg52_1
        buf110 = buf86; del buf86  # reuse
        buf133 = buf63; del buf63  # reuse
        generated_kernel_6(buf110, buf133, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_28, view_12, hidden_states_51, pow_18, variance_17, add_36, rsqrt_17, hidden_states_52, to_40, mul_58, query_states_4, cos_7, mul_61, q_embed_4, linear_29, view_13, hidden_states_53, pow_19, variance_18, add_37, rsqrt_18, hidden_states_54, to_42, mul_60, key_states_4, mul_63, k_embed_4, linear_30, view_14, value_states_4, attn_output_16], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf111 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf108, buf109, reinterpret_tensor(buf107, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf110, scale=0.08838834764831845, enable_gqa=True)
        buf112 = buf111[0]
        assert_tensor_metadata(buf112, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf112, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf111
        buf114 = reinterpret_tensor(buf108, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf108  # reuse
        generated_kernel_7(buf114, buf112, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg53_1, (2048, 2048), (2048, 1), 'input')
        buf115 = reinterpret_tensor(buf112, (71, 2048), (2048, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [transpose_19, attn_output_17, reshape_4, attn_output_19], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf114, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg53_1, (2048, 2048), (1, 2048), 0), out=buf115)
        del arg53_1
        assert_size_stride(arg54_1, (2048, ), (1, ), 'input')
        buf117 = reinterpret_tensor(buf114, (1, 71, 2048), (145408, 2048, 1), 0); del buf114  # reuse
        generated_kernel_13(buf117, buf92, buf98, buf115, arg54_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg54_1
        assert_size_stride(arg55_1, (6144, 2048), (2048, 1), 'input')
        buf118 = reinterpret_tensor(buf97, (71, 6144), (6144, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf117, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg55_1, (2048, 6144), (1, 2048), 0), out=buf118)
        del arg55_1
        assert_size_stride(arg56_1, (6144, 2048), (2048, 1), 'input')
        buf119 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf117, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg56_1, (2048, 6144), (1, 2048), 0), out=buf119)
        del arg56_1
        buf120 = reinterpret_tensor(buf95, (1, 71, 6144), (436224, 6144, 1), 0); del buf95  # reuse
        generated_kernel_9(buf120, buf118, buf119, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg57_1, (2048, 6144), (6144, 1), 'input')
        buf121 = reinterpret_tensor(buf117, (71, 2048), (2048, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_67, down_proj_4], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf120, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg57_1, (6144, 2048), (1, 6144), 0), out=buf121)
        del arg57_1
        assert_size_stride(arg58_1, (2048, ), (1, ), 'input')
        buf123 = buf100; del buf100  # reuse
        generated_kernel_14(buf123, buf92, buf98, buf115, buf121, arg58_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg58_1
        assert_size_stride(arg59_1, (2048, 2048), (2048, 1), 'input')
        buf124 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf123, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg59_1, (2048, 2048), (1, 2048), 0), out=buf124)
        del arg59_1
        assert_size_stride(arg60_1, (128, ), (1, ), 'input')
        buf131 = reinterpret_tensor(buf51, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf51  # reuse
        generated_kernel_1(buf131, buf124, arg60_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg60_1
        del buf124
        assert_size_stride(arg61_1, (1024, 2048), (2048, 1), 'input')
        buf127 = reinterpret_tensor(buf109, (71, 1024), (1024, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf123, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg61_1, (2048, 1024), (1, 2048), 0), out=buf127)
        del arg61_1
        assert_size_stride(arg62_1, (128, ), (1, ), 'input')
        buf132 = reinterpret_tensor(buf107, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf107  # reuse
        generated_kernel_2(buf132, buf127, arg62_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg62_1
        assert_size_stride(arg63_1, (1024, 2048), (2048, 1), 'input')
        buf130 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf123, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 1024), (1, 2048), 0), out=buf130)
        del arg63_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_35, view_15, hidden_states_63, pow_22, variance_21, add_44, rsqrt_21, hidden_states_64, to_48, mul_71, query_states_5, cos_8, mul_74, q_embed_5, linear_36, view_16, hidden_states_65, pow_23, variance_22, add_45, rsqrt_22, hidden_states_66, to_50, mul_73, key_states_5, mul_76, k_embed_5, linear_37, view_17, value_states_5, attn_output_20], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf134 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf131, buf132, reinterpret_tensor(buf130, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf133, scale=0.08838834764831845, enable_gqa=True)
        buf135 = buf134[0]
        assert_tensor_metadata(buf135, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf135, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf134
        buf137 = reinterpret_tensor(buf131, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf131  # reuse
        generated_kernel_7(buf137, buf135, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg64_1, (2048, 2048), (2048, 1), 'input')
        buf138 = reinterpret_tensor(buf135, (71, 2048), (2048, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [transpose_23, attn_output_21, reshape_5, attn_output_23], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf137, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg64_1, (2048, 2048), (1, 2048), 0), out=buf138)
        del arg64_1
        assert_size_stride(arg65_1, (2048, ), (1, ), 'input')
        buf139 = reinterpret_tensor(buf137, (1, 71, 2048), (145408, 2048, 1), 0); del buf137  # reuse
        buf141 = buf123; del buf123  # reuse
        generated_kernel_15(buf139, buf141, buf92, buf98, buf115, buf121, buf138, arg65_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg65_1
        del buf115
        assert_size_stride(arg66_1, (6144, 2048), (2048, 1), 'input')
        buf142 = reinterpret_tensor(buf120, (71, 6144), (6144, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68, pow_24, variance_23, add_49, rsqrt_23, hidden_states_69, to_52, hidden_states_70, linear_39], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf141, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg66_1, (2048, 6144), (1, 2048), 0), out=buf142)
        del arg66_1
        assert_size_stride(arg67_1, (6144, 2048), (2048, 1), 'input')
        buf143 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf141, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg67_1, (2048, 6144), (1, 2048), 0), out=buf143)
        del arg67_1
        buf144 = reinterpret_tensor(buf118, (1, 71, 6144), (436224, 6144, 1), 0); del buf118  # reuse
        generated_kernel_9(buf144, buf142, buf143, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg68_1, (2048, 6144), (6144, 1), 'input')
        buf145 = reinterpret_tensor(buf141, (71, 2048), (2048, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_80, down_proj_5], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf144, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg68_1, (6144, 2048), (1, 6144), 0), out=buf145)
        del arg68_1
        assert_size_stride(arg69_1, (2048, ), (1, ), 'input')
        buf147 = reinterpret_tensor(buf98, (1, 71, 2048), (145408, 2048, 1), 0); del buf98  # reuse
        generated_kernel_12(buf147, buf139, buf145, arg69_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg69_1
        assert_size_stride(arg70_1, (2048, 2048), (2048, 1), 'input')
        buf148 = reinterpret_tensor(buf92, (71, 2048), (2048, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_71, hidden_states_72, pow_25, variance_24, add_51, rsqrt_24, hidden_states_73, to_54, hidden_states_74, linear_42], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf147, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg70_1, (2048, 2048), (1, 2048), 0), out=buf148)
        del arg70_1
        assert_size_stride(arg71_1, (128, ), (1, ), 'input')
        buf155 = reinterpret_tensor(buf138, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf138  # reuse
        generated_kernel_1(buf155, buf148, arg71_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg71_1
        assert_size_stride(arg72_1, (1024, 2048), (2048, 1), 'input')
        buf151 = reinterpret_tensor(buf132, (71, 1024), (1024, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf147, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg72_1, (2048, 1024), (1, 2048), 0), out=buf151)
        del arg72_1
        assert_size_stride(arg73_1, (128, ), (1, ), 'input')
        buf156 = reinterpret_tensor(buf130, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf130  # reuse
        generated_kernel_2(buf156, buf151, arg73_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg73_1
        assert_size_stride(arg74_1, (1024, 2048), (2048, 1), 'input')
        buf154 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf147, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg74_1, (2048, 1024), (1, 2048), 0), out=buf154)
        del arg74_1
        buf157 = buf133; del buf133  # reuse
        buf180 = buf110; del buf110  # reuse
        generated_kernel_6(buf157, buf180, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_42, view_18, hidden_states_75, pow_26, variance_25, add_52, rsqrt_25, hidden_states_76, to_56, mul_84, query_states_6, cos_9, mul_87, q_embed_6, linear_43, view_19, hidden_states_77, pow_27, variance_26, add_53, rsqrt_26, hidden_states_78, to_58, mul_86, key_states_6, mul_89, k_embed_6, linear_44, view_20, value_states_6, attn_output_24], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf158 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf155, buf156, reinterpret_tensor(buf154, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf157, scale=0.08838834764831845, enable_gqa=True)
        buf159 = buf158[0]
        assert_tensor_metadata(buf159, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf159, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf158
        buf161 = reinterpret_tensor(buf155, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf155  # reuse
        generated_kernel_7(buf161, buf159, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg75_1, (2048, 2048), (2048, 1), 'input')
        buf162 = reinterpret_tensor(buf159, (71, 2048), (2048, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [transpose_27, attn_output_25, reshape_6, attn_output_27], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf161, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg75_1, (2048, 2048), (1, 2048), 0), out=buf162)
        del arg75_1
        assert_size_stride(arg76_1, (2048, ), (1, ), 'input')
        buf164 = reinterpret_tensor(buf161, (1, 71, 2048), (145408, 2048, 1), 0); del buf161  # reuse
        generated_kernel_13(buf164, buf139, buf145, buf162, arg76_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg76_1
        assert_size_stride(arg77_1, (6144, 2048), (2048, 1), 'input')
        buf165 = reinterpret_tensor(buf144, (71, 6144), (6144, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf164, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg77_1, (2048, 6144), (1, 2048), 0), out=buf165)
        del arg77_1
        assert_size_stride(arg78_1, (6144, 2048), (2048, 1), 'input')
        buf166 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf164, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg78_1, (2048, 6144), (1, 2048), 0), out=buf166)
        del arg78_1
        buf167 = reinterpret_tensor(buf142, (1, 71, 6144), (436224, 6144, 1), 0); del buf142  # reuse
        generated_kernel_9(buf167, buf165, buf166, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg79_1, (2048, 6144), (6144, 1), 'input')
        buf168 = reinterpret_tensor(buf164, (71, 2048), (2048, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_93, down_proj_6], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf167, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg79_1, (6144, 2048), (1, 6144), 0), out=buf168)
        del arg79_1
        assert_size_stride(arg80_1, (2048, ), (1, ), 'input')
        buf170 = buf147; del buf147  # reuse
        generated_kernel_14(buf170, buf139, buf145, buf162, buf168, arg80_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg80_1
        assert_size_stride(arg81_1, (2048, 2048), (2048, 1), 'input')
        buf171 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf170, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg81_1, (2048, 2048), (1, 2048), 0), out=buf171)
        del arg81_1
        assert_size_stride(arg82_1, (128, ), (1, ), 'input')
        buf178 = reinterpret_tensor(buf121, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf121  # reuse
        generated_kernel_1(buf178, buf171, arg82_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg82_1
        del buf171
        assert_size_stride(arg83_1, (1024, 2048), (2048, 1), 'input')
        buf174 = reinterpret_tensor(buf156, (71, 1024), (1024, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf170, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg83_1, (2048, 1024), (1, 2048), 0), out=buf174)
        del arg83_1
        assert_size_stride(arg84_1, (128, ), (1, ), 'input')
        buf179 = reinterpret_tensor(buf154, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf154  # reuse
        generated_kernel_2(buf179, buf174, arg84_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg84_1
        assert_size_stride(arg85_1, (1024, 2048), (2048, 1), 'input')
        buf177 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf170, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg85_1, (2048, 1024), (1, 2048), 0), out=buf177)
        del arg85_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_49, view_21, hidden_states_87, pow_30, variance_29, add_60, rsqrt_29, hidden_states_88, to_64, mul_97, query_states_7, cos_10, mul_100, q_embed_7, linear_50, view_22, hidden_states_89, pow_31, variance_30, add_61, rsqrt_30, hidden_states_90, to_66, mul_99, key_states_7, mul_102, k_embed_7, linear_51, view_23, value_states_7, attn_output_28], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf181 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf178, buf179, reinterpret_tensor(buf177, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf180, scale=0.08838834764831845, enable_gqa=True)
        buf182 = buf181[0]
        assert_tensor_metadata(buf182, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf182, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf181
        buf184 = reinterpret_tensor(buf178, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf178  # reuse
        generated_kernel_7(buf184, buf182, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg86_1, (2048, 2048), (2048, 1), 'input')
        buf185 = reinterpret_tensor(buf182, (71, 2048), (2048, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [transpose_31, attn_output_29, reshape_7, attn_output_31], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf184, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg86_1, (2048, 2048), (1, 2048), 0), out=buf185)
        del arg86_1
        assert_size_stride(arg87_1, (2048, ), (1, ), 'input')
        buf186 = reinterpret_tensor(buf184, (1, 71, 2048), (145408, 2048, 1), 0); del buf184  # reuse
        buf188 = buf170; del buf170  # reuse
        generated_kernel_15(buf186, buf188, buf139, buf145, buf162, buf168, buf185, arg87_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg87_1
        del buf139
        assert_size_stride(arg88_1, (6144, 2048), (2048, 1), 'input')
        buf189 = reinterpret_tensor(buf167, (71, 6144), (6144, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92, pow_32, variance_31, add_65, rsqrt_31, hidden_states_93, to_68, hidden_states_94, linear_53], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf188, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg88_1, (2048, 6144), (1, 2048), 0), out=buf189)
        del arg88_1
        assert_size_stride(arg89_1, (6144, 2048), (2048, 1), 'input')
        buf190 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf188, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg89_1, (2048, 6144), (1, 2048), 0), out=buf190)
        del arg89_1
        buf191 = reinterpret_tensor(buf165, (1, 71, 6144), (436224, 6144, 1), 0); del buf165  # reuse
        generated_kernel_9(buf191, buf189, buf190, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg90_1, (2048, 6144), (6144, 1), 'input')
        buf192 = reinterpret_tensor(buf188, (71, 2048), (2048, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_106, down_proj_7], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf191, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg90_1, (6144, 2048), (1, 6144), 0), out=buf192)
        del arg90_1
        assert_size_stride(arg91_1, (2048, ), (1, ), 'input')
        buf194 = reinterpret_tensor(buf185, (1, 71, 2048), (145408, 2048, 1), 0); del buf185  # reuse
        generated_kernel_12(buf194, buf186, buf192, arg91_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg91_1
        assert_size_stride(arg92_1, (2048, 2048), (2048, 1), 'input')
        buf195 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_95, hidden_states_96, pow_33, variance_32, add_67, rsqrt_32, hidden_states_97, to_70, hidden_states_98, linear_56], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf194, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg92_1, (2048, 2048), (1, 2048), 0), out=buf195)
        del arg92_1
        assert_size_stride(arg93_1, (128, ), (1, ), 'input')
        buf202 = reinterpret_tensor(buf162, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf162  # reuse
        generated_kernel_1(buf202, buf195, arg93_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg93_1
        assert_size_stride(arg94_1, (1024, 2048), (2048, 1), 'input')
        buf198 = reinterpret_tensor(buf179, (71, 1024), (1024, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf194, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg94_1, (2048, 1024), (1, 2048), 0), out=buf198)
        del arg94_1
        assert_size_stride(arg95_1, (128, ), (1, ), 'input')
        buf203 = reinterpret_tensor(buf177, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf177  # reuse
        generated_kernel_2(buf203, buf198, arg95_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg95_1
        assert_size_stride(arg96_1, (1024, 2048), (2048, 1), 'input')
        buf201 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf194, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 1024), (1, 2048), 0), out=buf201)
        del arg96_1
        buf204 = buf180; del buf180  # reuse
        buf227 = buf157; del buf157  # reuse
        generated_kernel_6(buf204, buf227, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_56, view_24, hidden_states_99, pow_34, variance_33, add_68, rsqrt_33, hidden_states_100, to_72, mul_110, query_states_8, cos_11, mul_113, q_embed_8, linear_57, view_25, hidden_states_101, pow_35, variance_34, add_69, rsqrt_34, hidden_states_102, to_74, mul_112, key_states_8, mul_115, k_embed_8, linear_58, view_26, value_states_8, attn_output_32], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf205 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf202, buf203, reinterpret_tensor(buf201, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf204, scale=0.08838834764831845, enable_gqa=True)
        buf206 = buf205[0]
        assert_tensor_metadata(buf206, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf206, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf205
        buf208 = reinterpret_tensor(buf202, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf202  # reuse
        generated_kernel_7(buf208, buf206, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg97_1, (2048, 2048), (2048, 1), 'input')
        buf209 = reinterpret_tensor(buf206, (71, 2048), (2048, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [transpose_35, attn_output_33, reshape_8, attn_output_35], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf208, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg97_1, (2048, 2048), (1, 2048), 0), out=buf209)
        del arg97_1
        assert_size_stride(arg98_1, (2048, ), (1, ), 'input')
        buf211 = reinterpret_tensor(buf208, (1, 71, 2048), (145408, 2048, 1), 0); del buf208  # reuse
        generated_kernel_13(buf211, buf186, buf192, buf209, arg98_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg98_1
        assert_size_stride(arg99_1, (6144, 2048), (2048, 1), 'input')
        buf212 = reinterpret_tensor(buf191, (71, 6144), (6144, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf211, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg99_1, (2048, 6144), (1, 2048), 0), out=buf212)
        del arg99_1
        assert_size_stride(arg100_1, (6144, 2048), (2048, 1), 'input')
        buf213 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf211, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 6144), (1, 2048), 0), out=buf213)
        del arg100_1
        buf214 = reinterpret_tensor(buf189, (1, 71, 6144), (436224, 6144, 1), 0); del buf189  # reuse
        generated_kernel_9(buf214, buf212, buf213, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg101_1, (2048, 6144), (6144, 1), 'input')
        buf215 = reinterpret_tensor(buf211, (71, 2048), (2048, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_119, down_proj_8], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf214, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg101_1, (6144, 2048), (1, 6144), 0), out=buf215)
        del arg101_1
        assert_size_stride(arg102_1, (2048, ), (1, ), 'input')
        buf217 = buf194; del buf194  # reuse
        generated_kernel_14(buf217, buf186, buf192, buf209, buf215, arg102_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg102_1
        assert_size_stride(arg103_1, (2048, 2048), (2048, 1), 'input')
        buf218 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf217, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg103_1, (2048, 2048), (1, 2048), 0), out=buf218)
        del arg103_1
        assert_size_stride(arg104_1, (128, ), (1, ), 'input')
        buf225 = reinterpret_tensor(buf145, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf145  # reuse
        generated_kernel_1(buf225, buf218, arg104_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg104_1
        del buf218
        assert_size_stride(arg105_1, (1024, 2048), (2048, 1), 'input')
        buf221 = reinterpret_tensor(buf203, (71, 1024), (1024, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf217, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg105_1, (2048, 1024), (1, 2048), 0), out=buf221)
        del arg105_1
        assert_size_stride(arg106_1, (128, ), (1, ), 'input')
        buf226 = reinterpret_tensor(buf201, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf201  # reuse
        generated_kernel_2(buf226, buf221, arg106_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg106_1
        assert_size_stride(arg107_1, (1024, 2048), (2048, 1), 'input')
        buf224 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf217, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg107_1, (2048, 1024), (1, 2048), 0), out=buf224)
        del arg107_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_63, view_27, hidden_states_111, pow_38, variance_37, add_76, rsqrt_37, hidden_states_112, to_80, mul_123, query_states_9, cos_12, mul_126, q_embed_9, linear_64, view_28, hidden_states_113, pow_39, variance_38, add_77, rsqrt_38, hidden_states_114, to_82, mul_125, key_states_9, mul_128, k_embed_9, linear_65, view_29, value_states_9, attn_output_36], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf228 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf225, buf226, reinterpret_tensor(buf224, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf227, scale=0.08838834764831845, enable_gqa=True)
        buf229 = buf228[0]
        assert_tensor_metadata(buf229, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf229, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf228
        buf231 = reinterpret_tensor(buf225, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf225  # reuse
        generated_kernel_7(buf231, buf229, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg108_1, (2048, 2048), (2048, 1), 'input')
        buf232 = reinterpret_tensor(buf229, (71, 2048), (2048, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [transpose_39, attn_output_37, reshape_9, attn_output_39], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf231, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg108_1, (2048, 2048), (1, 2048), 0), out=buf232)
        del arg108_1
        assert_size_stride(arg109_1, (2048, ), (1, ), 'input')
        buf233 = reinterpret_tensor(buf231, (1, 71, 2048), (145408, 2048, 1), 0); del buf231  # reuse
        buf235 = buf217; del buf217  # reuse
        generated_kernel_15(buf233, buf235, buf186, buf192, buf209, buf215, buf232, arg109_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg109_1
        del buf186
        assert_size_stride(arg110_1, (6144, 2048), (2048, 1), 'input')
        buf236 = reinterpret_tensor(buf214, (71, 6144), (6144, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_116, pow_40, variance_39, add_81, rsqrt_39, hidden_states_117, to_84, hidden_states_118, linear_67], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf235, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 6144), (1, 2048), 0), out=buf236)
        del arg110_1
        assert_size_stride(arg111_1, (6144, 2048), (2048, 1), 'input')
        buf237 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf235, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg111_1, (2048, 6144), (1, 2048), 0), out=buf237)
        del arg111_1
        buf238 = reinterpret_tensor(buf212, (1, 71, 6144), (436224, 6144, 1), 0); del buf212  # reuse
        generated_kernel_9(buf238, buf236, buf237, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg112_1, (2048, 6144), (6144, 1), 'input')
        buf239 = reinterpret_tensor(buf235, (71, 2048), (2048, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_132, down_proj_9], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf238, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg112_1, (6144, 2048), (1, 6144), 0), out=buf239)
        del arg112_1
        assert_size_stride(arg113_1, (2048, ), (1, ), 'input')
        buf241 = reinterpret_tensor(buf232, (1, 71, 2048), (145408, 2048, 1), 0); del buf232  # reuse
        generated_kernel_12(buf241, buf233, buf239, arg113_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg113_1
        assert_size_stride(arg114_1, (2048, 2048), (2048, 1), 'input')
        buf242 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_119, hidden_states_120, pow_41, variance_40, add_83, rsqrt_40, hidden_states_121, to_86, hidden_states_122, linear_70], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf241, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg114_1, (2048, 2048), (1, 2048), 0), out=buf242)
        del arg114_1
        assert_size_stride(arg115_1, (128, ), (1, ), 'input')
        buf249 = reinterpret_tensor(buf209, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf209  # reuse
        generated_kernel_1(buf249, buf242, arg115_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg115_1
        assert_size_stride(arg116_1, (1024, 2048), (2048, 1), 'input')
        buf245 = reinterpret_tensor(buf226, (71, 1024), (1024, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf241, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg116_1, (2048, 1024), (1, 2048), 0), out=buf245)
        del arg116_1
        assert_size_stride(arg117_1, (128, ), (1, ), 'input')
        buf250 = reinterpret_tensor(buf224, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf224  # reuse
        generated_kernel_2(buf250, buf245, arg117_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg117_1
        assert_size_stride(arg118_1, (1024, 2048), (2048, 1), 'input')
        buf248 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf241, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg118_1, (2048, 1024), (1, 2048), 0), out=buf248)
        del arg118_1
        buf251 = buf227; del buf227  # reuse
        buf274 = buf204; del buf204  # reuse
        generated_kernel_6(buf251, buf274, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_70, view_30, hidden_states_123, pow_42, variance_41, add_84, rsqrt_41, hidden_states_124, to_88, mul_136, query_states_10, cos_13, mul_139, q_embed_10, linear_71, view_31, hidden_states_125, pow_43, variance_42, add_85, rsqrt_42, hidden_states_126, to_90, mul_138, key_states_10, mul_141, k_embed_10, linear_72, view_32, value_states_10, attn_output_40], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf252 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf249, buf250, reinterpret_tensor(buf248, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf251, scale=0.08838834764831845, enable_gqa=True)
        buf253 = buf252[0]
        assert_tensor_metadata(buf253, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf253, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf252
        buf255 = reinterpret_tensor(buf249, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf249  # reuse
        generated_kernel_7(buf255, buf253, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg119_1, (2048, 2048), (2048, 1), 'input')
        buf256 = reinterpret_tensor(buf253, (71, 2048), (2048, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [transpose_43, attn_output_41, reshape_10, attn_output_43], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf255, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg119_1, (2048, 2048), (1, 2048), 0), out=buf256)
        del arg119_1
        assert_size_stride(arg120_1, (2048, ), (1, ), 'input')
        buf258 = reinterpret_tensor(buf255, (1, 71, 2048), (145408, 2048, 1), 0); del buf255  # reuse
        generated_kernel_13(buf258, buf233, buf239, buf256, arg120_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg120_1
        assert_size_stride(arg121_1, (6144, 2048), (2048, 1), 'input')
        buf259 = reinterpret_tensor(buf238, (71, 6144), (6144, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf258, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg121_1, (2048, 6144), (1, 2048), 0), out=buf259)
        del arg121_1
        assert_size_stride(arg122_1, (6144, 2048), (2048, 1), 'input')
        buf260 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf258, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg122_1, (2048, 6144), (1, 2048), 0), out=buf260)
        del arg122_1
        buf261 = reinterpret_tensor(buf236, (1, 71, 6144), (436224, 6144, 1), 0); del buf236  # reuse
        generated_kernel_9(buf261, buf259, buf260, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg123_1, (2048, 6144), (6144, 1), 'input')
        buf262 = reinterpret_tensor(buf258, (71, 2048), (2048, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_145, down_proj_10], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf261, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg123_1, (6144, 2048), (1, 6144), 0), out=buf262)
        del arg123_1
        assert_size_stride(arg124_1, (2048, ), (1, ), 'input')
        buf264 = buf241; del buf241  # reuse
        generated_kernel_14(buf264, buf233, buf239, buf256, buf262, arg124_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg124_1
        assert_size_stride(arg125_1, (2048, 2048), (2048, 1), 'input')
        buf265 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf264, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg125_1, (2048, 2048), (1, 2048), 0), out=buf265)
        del arg125_1
        assert_size_stride(arg126_1, (128, ), (1, ), 'input')
        buf272 = reinterpret_tensor(buf192, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf192  # reuse
        generated_kernel_1(buf272, buf265, arg126_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg126_1
        del buf265
        assert_size_stride(arg127_1, (1024, 2048), (2048, 1), 'input')
        buf268 = reinterpret_tensor(buf250, (71, 1024), (1024, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf264, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg127_1, (2048, 1024), (1, 2048), 0), out=buf268)
        del arg127_1
        assert_size_stride(arg128_1, (128, ), (1, ), 'input')
        buf273 = reinterpret_tensor(buf248, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf248  # reuse
        generated_kernel_2(buf273, buf268, arg128_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg128_1
        assert_size_stride(arg129_1, (1024, 2048), (2048, 1), 'input')
        buf271 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf264, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg129_1, (2048, 1024), (1, 2048), 0), out=buf271)
        del arg129_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_77, view_33, hidden_states_135, pow_46, variance_45, add_92, rsqrt_45, hidden_states_136, to_96, mul_149, query_states_11, cos_14, mul_152, q_embed_11, linear_78, view_34, hidden_states_137, pow_47, variance_46, add_93, rsqrt_46, hidden_states_138, to_98, mul_151, key_states_11, mul_154, k_embed_11, linear_79, view_35, value_states_11, attn_output_44], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf275 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf272, buf273, reinterpret_tensor(buf271, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf274, scale=0.08838834764831845, enable_gqa=True)
        buf276 = buf275[0]
        assert_tensor_metadata(buf276, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf276, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf275
        buf278 = reinterpret_tensor(buf272, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf272  # reuse
        generated_kernel_7(buf278, buf276, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg130_1, (2048, 2048), (2048, 1), 'input')
        buf279 = reinterpret_tensor(buf276, (71, 2048), (2048, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [transpose_47, attn_output_45, reshape_11, attn_output_47], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf278, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 2048), (1, 2048), 0), out=buf279)
        del arg130_1
        assert_size_stride(arg131_1, (2048, ), (1, ), 'input')
        buf280 = reinterpret_tensor(buf278, (1, 71, 2048), (145408, 2048, 1), 0); del buf278  # reuse
        buf282 = buf264; del buf264  # reuse
        generated_kernel_15(buf280, buf282, buf233, buf239, buf256, buf262, buf279, arg131_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg131_1
        del buf233
        assert_size_stride(arg132_1, (6144, 2048), (2048, 1), 'input')
        buf283 = reinterpret_tensor(buf261, (71, 6144), (6144, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140, pow_48, variance_47, add_97, rsqrt_47, hidden_states_141, to_100, hidden_states_142, linear_81], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf282, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg132_1, (2048, 6144), (1, 2048), 0), out=buf283)
        del arg132_1
        assert_size_stride(arg133_1, (6144, 2048), (2048, 1), 'input')
        buf284 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf282, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg133_1, (2048, 6144), (1, 2048), 0), out=buf284)
        del arg133_1
        buf285 = reinterpret_tensor(buf259, (1, 71, 6144), (436224, 6144, 1), 0); del buf259  # reuse
        generated_kernel_9(buf285, buf283, buf284, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg134_1, (2048, 6144), (6144, 1), 'input')
        buf286 = reinterpret_tensor(buf282, (71, 2048), (2048, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_158, down_proj_11], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf285, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg134_1, (6144, 2048), (1, 6144), 0), out=buf286)
        del arg134_1
        assert_size_stride(arg135_1, (2048, ), (1, ), 'input')
        buf288 = reinterpret_tensor(buf279, (1, 71, 2048), (145408, 2048, 1), 0); del buf279  # reuse
        generated_kernel_12(buf288, buf280, buf286, arg135_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg135_1
        assert_size_stride(arg136_1, (2048, 2048), (2048, 1), 'input')
        buf289 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_143, hidden_states_144, pow_49, variance_48, add_99, rsqrt_48, hidden_states_145, to_102, hidden_states_146, linear_84], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf288, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg136_1, (2048, 2048), (1, 2048), 0), out=buf289)
        del arg136_1
        assert_size_stride(arg137_1, (128, ), (1, ), 'input')
        buf296 = reinterpret_tensor(buf256, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf256  # reuse
        generated_kernel_1(buf296, buf289, arg137_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg137_1
        assert_size_stride(arg138_1, (1024, 2048), (2048, 1), 'input')
        buf292 = reinterpret_tensor(buf273, (71, 1024), (1024, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf288, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg138_1, (2048, 1024), (1, 2048), 0), out=buf292)
        del arg138_1
        assert_size_stride(arg139_1, (128, ), (1, ), 'input')
        buf297 = reinterpret_tensor(buf271, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf271  # reuse
        generated_kernel_2(buf297, buf292, arg139_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg139_1
        assert_size_stride(arg140_1, (1024, 2048), (2048, 1), 'input')
        buf295 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf288, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg140_1, (2048, 1024), (1, 2048), 0), out=buf295)
        del arg140_1
        buf298 = buf274; del buf274  # reuse
        buf321 = buf251; del buf251  # reuse
        generated_kernel_6(buf298, buf321, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_84, view_36, hidden_states_147, pow_50, variance_49, add_100, rsqrt_49, hidden_states_148, to_104, mul_162, query_states_12, cos_15, mul_165, q_embed_12, linear_85, view_37, hidden_states_149, pow_51, variance_50, add_101, rsqrt_50, hidden_states_150, to_106, mul_164, key_states_12, mul_167, k_embed_12, linear_86, view_38, value_states_12, attn_output_48], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf299 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf296, buf297, reinterpret_tensor(buf295, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf298, scale=0.08838834764831845, enable_gqa=True)
        buf300 = buf299[0]
        assert_tensor_metadata(buf300, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf300, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf299
        buf302 = reinterpret_tensor(buf296, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf296  # reuse
        generated_kernel_7(buf302, buf300, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg141_1, (2048, 2048), (2048, 1), 'input')
        buf303 = reinterpret_tensor(buf300, (71, 2048), (2048, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [transpose_51, attn_output_49, reshape_12, attn_output_51], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf302, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg141_1, (2048, 2048), (1, 2048), 0), out=buf303)
        del arg141_1
        assert_size_stride(arg142_1, (2048, ), (1, ), 'input')
        buf305 = reinterpret_tensor(buf302, (1, 71, 2048), (145408, 2048, 1), 0); del buf302  # reuse
        generated_kernel_13(buf305, buf280, buf286, buf303, arg142_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg142_1
        assert_size_stride(arg143_1, (6144, 2048), (2048, 1), 'input')
        buf306 = reinterpret_tensor(buf285, (71, 6144), (6144, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf305, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg143_1, (2048, 6144), (1, 2048), 0), out=buf306)
        del arg143_1
        assert_size_stride(arg144_1, (6144, 2048), (2048, 1), 'input')
        buf307 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf305, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg144_1, (2048, 6144), (1, 2048), 0), out=buf307)
        del arg144_1
        buf308 = reinterpret_tensor(buf283, (1, 71, 6144), (436224, 6144, 1), 0); del buf283  # reuse
        generated_kernel_9(buf308, buf306, buf307, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg145_1, (2048, 6144), (6144, 1), 'input')
        buf309 = reinterpret_tensor(buf305, (71, 2048), (2048, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_171, down_proj_12], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf308, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg145_1, (6144, 2048), (1, 6144), 0), out=buf309)
        del arg145_1
        assert_size_stride(arg146_1, (2048, ), (1, ), 'input')
        buf311 = buf288; del buf288  # reuse
        generated_kernel_14(buf311, buf280, buf286, buf303, buf309, arg146_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg146_1
        assert_size_stride(arg147_1, (2048, 2048), (2048, 1), 'input')
        buf312 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf311, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg147_1, (2048, 2048), (1, 2048), 0), out=buf312)
        del arg147_1
        assert_size_stride(arg148_1, (128, ), (1, ), 'input')
        buf319 = reinterpret_tensor(buf239, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf239  # reuse
        generated_kernel_1(buf319, buf312, arg148_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg148_1
        del buf312
        assert_size_stride(arg149_1, (1024, 2048), (2048, 1), 'input')
        buf315 = reinterpret_tensor(buf297, (71, 1024), (1024, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf311, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg149_1, (2048, 1024), (1, 2048), 0), out=buf315)
        del arg149_1
        assert_size_stride(arg150_1, (128, ), (1, ), 'input')
        buf320 = reinterpret_tensor(buf295, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf295  # reuse
        generated_kernel_2(buf320, buf315, arg150_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg150_1
        assert_size_stride(arg151_1, (1024, 2048), (2048, 1), 'input')
        buf318 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf311, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg151_1, (2048, 1024), (1, 2048), 0), out=buf318)
        del arg151_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_91, view_39, hidden_states_159, pow_54, variance_53, add_108, rsqrt_53, hidden_states_160, to_112, mul_175, query_states_13, cos_16, mul_178, q_embed_13, linear_92, view_40, hidden_states_161, pow_55, variance_54, add_109, rsqrt_54, hidden_states_162, to_114, mul_177, key_states_13, mul_180, k_embed_13, linear_93, view_41, value_states_13, attn_output_52], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf322 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf319, buf320, reinterpret_tensor(buf318, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf321, scale=0.08838834764831845, enable_gqa=True)
        buf323 = buf322[0]
        assert_tensor_metadata(buf323, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf323, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf322
        buf325 = reinterpret_tensor(buf319, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf319  # reuse
        generated_kernel_7(buf325, buf323, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg152_1, (2048, 2048), (2048, 1), 'input')
        buf326 = reinterpret_tensor(buf323, (71, 2048), (2048, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [transpose_55, attn_output_53, reshape_13, attn_output_55], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf325, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg152_1, (2048, 2048), (1, 2048), 0), out=buf326)
        del arg152_1
        assert_size_stride(arg153_1, (2048, ), (1, ), 'input')
        buf327 = reinterpret_tensor(buf325, (1, 71, 2048), (145408, 2048, 1), 0); del buf325  # reuse
        buf329 = buf311; del buf311  # reuse
        generated_kernel_15(buf327, buf329, buf280, buf286, buf303, buf309, buf326, arg153_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg153_1
        assert_size_stride(arg154_1, (6144, 2048), (2048, 1), 'input')
        buf330 = reinterpret_tensor(buf308, (71, 6144), (6144, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_164, pow_56, variance_55, add_113, rsqrt_55, hidden_states_165, to_116, hidden_states_166, linear_95], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf329, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg154_1, (2048, 6144), (1, 2048), 0), out=buf330)
        del arg154_1
        assert_size_stride(arg155_1, (6144, 2048), (2048, 1), 'input')
        buf331 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf329, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg155_1, (2048, 6144), (1, 2048), 0), out=buf331)
        del arg155_1
        buf332 = reinterpret_tensor(buf306, (1, 71, 6144), (436224, 6144, 1), 0); del buf306  # reuse
        generated_kernel_9(buf332, buf330, buf331, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg156_1, (2048, 6144), (6144, 1), 'input')
        buf333 = reinterpret_tensor(buf329, (71, 2048), (2048, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_184, down_proj_13], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf332, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg156_1, (6144, 2048), (1, 6144), 0), out=buf333)
        del arg156_1
        assert_size_stride(arg157_1, (2048, ), (1, ), 'input')
        buf335 = reinterpret_tensor(buf326, (1, 71, 2048), (145408, 2048, 1), 0); del buf326  # reuse
        generated_kernel_12(buf335, buf327, buf333, arg157_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg157_1
        assert_size_stride(arg158_1, (2048, 2048), (2048, 1), 'input')
        buf336 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_167, hidden_states_168, pow_57, variance_56, add_115, rsqrt_56, hidden_states_169, to_118, hidden_states_170, linear_98], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf335, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg158_1, (2048, 2048), (1, 2048), 0), out=buf336)
        del arg158_1
        assert_size_stride(arg159_1, (128, ), (1, ), 'input')
        buf338 = reinterpret_tensor(buf303, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf303  # reuse
        generated_kernel_16(buf338, buf336, arg159_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=3)
        del arg159_1
        assert_size_stride(arg160_1, (1024, 2048), (2048, 1), 'input')
        buf339 = reinterpret_tensor(buf320, (71, 1024), (1024, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf335, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg160_1, (2048, 1024), (1, 2048), 0), out=buf339)
        del arg160_1
        assert_size_stride(arg161_1, (128, ), (1, ), 'input')
        buf344 = reinterpret_tensor(buf318, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf318  # reuse
        generated_kernel_2(buf344, buf339, arg161_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg161_1
        assert_size_stride(arg162_1, (1024, 2048), (2048, 1), 'input')
        buf342 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf335, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg162_1, (2048, 1024), (1, 2048), 0), out=buf342)
        del arg162_1
        buf343 = reinterpret_tensor(buf335, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf335  # reuse
        generated_kernel_17(buf343, buf338, arg2_1, threads=[145408], error_buf_idx=3)
        buf345 = buf321; del buf321  # reuse
        buf368 = buf298; del buf298  # reuse
        generated_kernel_6(buf345, buf368, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, query_states_14, cos_17, mul_191, x2_28, neg_28, x1_28, cat_29, sin_17, mul_192, q_embed_14, linear_99, view_43, hidden_states_171, pow_59, variance_57, add_117, rsqrt_58, hidden_states_172, to_122, mul_190, key_states_14, mul_193, k_embed_14, linear_100, view_44, value_states_14, attn_output_56], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.transpose, aten.slice, aten.neg, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf346 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf343, buf344, reinterpret_tensor(buf342, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf345, scale=0.08838834764831845, enable_gqa=True)
        buf347 = buf346[0]
        assert_tensor_metadata(buf347, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf347, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf346
        buf349 = reinterpret_tensor(buf343, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf343  # reuse
        generated_kernel_7(buf349, buf347, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg163_1, (2048, 2048), (2048, 1), 'input')
        buf350 = reinterpret_tensor(buf347, (71, 2048), (2048, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [transpose_59, attn_output_57, reshape_14, attn_output_59], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf349, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg163_1, (2048, 2048), (1, 2048), 0), out=buf350)
        del arg163_1
        assert_size_stride(arg164_1, (2048, ), (1, ), 'input')
        buf352 = reinterpret_tensor(buf349, (1, 71, 2048), (145408, 2048, 1), 0); del buf349  # reuse
        generated_kernel_13(buf352, buf327, buf333, buf350, arg164_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg164_1
        assert_size_stride(arg165_1, (6144, 2048), (2048, 1), 'input')
        buf353 = reinterpret_tensor(buf332, (71, 6144), (6144, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf352, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg165_1, (2048, 6144), (1, 2048), 0), out=buf353)
        del arg165_1
        assert_size_stride(arg166_1, (6144, 2048), (2048, 1), 'input')
        buf354 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf352, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg166_1, (2048, 6144), (1, 2048), 0), out=buf354)
        del arg166_1
        buf355 = reinterpret_tensor(buf330, (1, 71, 6144), (436224, 6144, 1), 0); del buf330  # reuse
        generated_kernel_9(buf355, buf353, buf354, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg167_1, (2048, 6144), (6144, 1), 'input')
        buf356 = reinterpret_tensor(buf352, (71, 2048), (2048, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_197, down_proj_14], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf355, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg167_1, (6144, 2048), (1, 6144), 0), out=buf356)
        del arg167_1
        assert_size_stride(arg168_1, (2048, ), (1, ), 'input')
        buf358 = reinterpret_tensor(buf336, (1, 71, 2048), (145408, 2048, 1), 0); del buf336  # reuse
        generated_kernel_14(buf358, buf327, buf333, buf350, buf356, arg168_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg168_1
        assert_size_stride(arg169_1, (2048, 2048), (2048, 1), 'input')
        buf359 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf358, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg169_1, (2048, 2048), (1, 2048), 0), out=buf359)
        del arg169_1
        assert_size_stride(arg170_1, (128, ), (1, ), 'input')
        buf366 = reinterpret_tensor(buf280, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf280  # reuse
        generated_kernel_1(buf366, buf359, arg170_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg170_1
        del buf359
        assert_size_stride(arg171_1, (1024, 2048), (2048, 1), 'input')
        buf362 = reinterpret_tensor(buf344, (71, 1024), (1024, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf358, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg171_1, (2048, 1024), (1, 2048), 0), out=buf362)
        del arg171_1
        assert_size_stride(arg172_1, (128, ), (1, ), 'input')
        buf367 = reinterpret_tensor(buf342, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf342  # reuse
        generated_kernel_2(buf367, buf362, arg172_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg172_1
        assert_size_stride(arg173_1, (1024, 2048), (2048, 1), 'input')
        buf365 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf358, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg173_1, (2048, 1024), (1, 2048), 0), out=buf365)
        del arg173_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_105, view_45, hidden_states_181, pow_62, variance_60, add_124, rsqrt_61, hidden_states_182, to_128, mul_201, query_states_15, cos_18, mul_204, q_embed_15, linear_106, view_46, hidden_states_183, pow_63, variance_61, add_125, rsqrt_62, hidden_states_184, to_130, mul_203, key_states_15, mul_206, k_embed_15, linear_107, view_47, value_states_15, attn_output_60], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf369 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf366, buf367, reinterpret_tensor(buf365, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf368, scale=0.08838834764831845, enable_gqa=True)
        buf370 = buf369[0]
        assert_tensor_metadata(buf370, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf370, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf369
        buf372 = reinterpret_tensor(buf366, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf366  # reuse
        generated_kernel_7(buf372, buf370, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg174_1, (2048, 2048), (2048, 1), 'input')
        buf373 = reinterpret_tensor(buf370, (71, 2048), (2048, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [transpose_63, attn_output_61, reshape_15, attn_output_63], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf372, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg174_1, (2048, 2048), (1, 2048), 0), out=buf373)
        del arg174_1
        assert_size_stride(arg175_1, (2048, ), (1, ), 'input')
        buf374 = reinterpret_tensor(buf372, (1, 71, 2048), (145408, 2048, 1), 0); del buf372  # reuse
        buf376 = buf358; del buf358  # reuse
        generated_kernel_15(buf374, buf376, buf327, buf333, buf350, buf356, buf373, arg175_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg175_1
        del buf327
        assert_size_stride(arg176_1, (6144, 2048), (2048, 1), 'input')
        buf377 = reinterpret_tensor(buf355, (71, 6144), (6144, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_186, pow_64, variance_62, add_129, rsqrt_63, hidden_states_187, to_132, hidden_states_188, linear_109], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf376, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 6144), (1, 2048), 0), out=buf377)
        del arg176_1
        assert_size_stride(arg177_1, (6144, 2048), (2048, 1), 'input')
        buf378 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf376, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg177_1, (2048, 6144), (1, 2048), 0), out=buf378)
        del arg177_1
        buf379 = reinterpret_tensor(buf353, (1, 71, 6144), (436224, 6144, 1), 0); del buf353  # reuse
        generated_kernel_9(buf379, buf377, buf378, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg178_1, (2048, 6144), (6144, 1), 'input')
        buf380 = reinterpret_tensor(buf376, (71, 2048), (2048, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_210, down_proj_15], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf379, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg178_1, (6144, 2048), (1, 6144), 0), out=buf380)
        del arg178_1
        assert_size_stride(arg179_1, (2048, ), (1, ), 'input')
        buf382 = reinterpret_tensor(buf373, (1, 71, 2048), (145408, 2048, 1), 0); del buf373  # reuse
        generated_kernel_12(buf382, buf374, buf380, arg179_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg179_1
        assert_size_stride(arg180_1, (2048, 2048), (2048, 1), 'input')
        buf383 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_189, hidden_states_190, pow_65, variance_63, add_131, rsqrt_64, hidden_states_191, to_134, hidden_states_192, linear_112], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf382, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg180_1, (2048, 2048), (1, 2048), 0), out=buf383)
        del arg180_1
        assert_size_stride(arg181_1, (128, ), (1, ), 'input')
        buf390 = reinterpret_tensor(buf350, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf350  # reuse
        generated_kernel_1(buf390, buf383, arg181_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg181_1
        assert_size_stride(arg182_1, (1024, 2048), (2048, 1), 'input')
        buf386 = reinterpret_tensor(buf367, (71, 1024), (1024, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf382, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg182_1, (2048, 1024), (1, 2048), 0), out=buf386)
        del arg182_1
        assert_size_stride(arg183_1, (128, ), (1, ), 'input')
        buf391 = reinterpret_tensor(buf365, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf365  # reuse
        generated_kernel_2(buf391, buf386, arg183_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg183_1
        assert_size_stride(arg184_1, (1024, 2048), (2048, 1), 'input')
        buf389 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf382, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg184_1, (2048, 1024), (1, 2048), 0), out=buf389)
        del arg184_1
        buf392 = buf368; del buf368  # reuse
        buf415 = buf345; del buf345  # reuse
        generated_kernel_6(buf392, buf415, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_112, view_48, hidden_states_193, pow_66, variance_64, add_132, rsqrt_65, hidden_states_194, to_136, mul_214, query_states_16, cos_19, mul_217, q_embed_16, linear_113, view_49, hidden_states_195, pow_67, variance_65, add_133, rsqrt_66, hidden_states_196, to_138, mul_216, key_states_16, mul_219, k_embed_16, linear_114, view_50, value_states_16, attn_output_64], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf393 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf390, buf391, reinterpret_tensor(buf389, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf392, scale=0.08838834764831845, enable_gqa=True)
        buf394 = buf393[0]
        assert_tensor_metadata(buf394, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf394, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf393
        buf396 = reinterpret_tensor(buf390, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf390  # reuse
        generated_kernel_7(buf396, buf394, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg185_1, (2048, 2048), (2048, 1), 'input')
        buf397 = reinterpret_tensor(buf394, (71, 2048), (2048, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [transpose_67, attn_output_65, reshape_16, attn_output_67], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf396, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg185_1, (2048, 2048), (1, 2048), 0), out=buf397)
        del arg185_1
        assert_size_stride(arg186_1, (2048, ), (1, ), 'input')
        buf399 = reinterpret_tensor(buf396, (1, 71, 2048), (145408, 2048, 1), 0); del buf396  # reuse
        generated_kernel_13(buf399, buf374, buf380, buf397, arg186_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg186_1
        assert_size_stride(arg187_1, (6144, 2048), (2048, 1), 'input')
        buf400 = reinterpret_tensor(buf379, (71, 6144), (6144, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf399, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg187_1, (2048, 6144), (1, 2048), 0), out=buf400)
        del arg187_1
        assert_size_stride(arg188_1, (6144, 2048), (2048, 1), 'input')
        buf401 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf399, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 6144), (1, 2048), 0), out=buf401)
        del arg188_1
        buf402 = reinterpret_tensor(buf377, (1, 71, 6144), (436224, 6144, 1), 0); del buf377  # reuse
        generated_kernel_9(buf402, buf400, buf401, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg189_1, (2048, 6144), (6144, 1), 'input')
        buf403 = reinterpret_tensor(buf399, (71, 2048), (2048, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_223, down_proj_16], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf402, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg189_1, (6144, 2048), (1, 6144), 0), out=buf403)
        del arg189_1
        assert_size_stride(arg190_1, (2048, ), (1, ), 'input')
        buf405 = buf382; del buf382  # reuse
        generated_kernel_14(buf405, buf374, buf380, buf397, buf403, arg190_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg190_1
        assert_size_stride(arg191_1, (2048, 2048), (2048, 1), 'input')
        buf406 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf405, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg191_1, (2048, 2048), (1, 2048), 0), out=buf406)
        del arg191_1
        assert_size_stride(arg192_1, (128, ), (1, ), 'input')
        buf413 = reinterpret_tensor(buf333, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf333  # reuse
        generated_kernel_1(buf413, buf406, arg192_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg192_1
        del buf406
        assert_size_stride(arg193_1, (1024, 2048), (2048, 1), 'input')
        buf409 = reinterpret_tensor(buf391, (71, 1024), (1024, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf405, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg193_1, (2048, 1024), (1, 2048), 0), out=buf409)
        del arg193_1
        assert_size_stride(arg194_1, (128, ), (1, ), 'input')
        buf414 = reinterpret_tensor(buf389, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf389  # reuse
        generated_kernel_2(buf414, buf409, arg194_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg194_1
        assert_size_stride(arg195_1, (1024, 2048), (2048, 1), 'input')
        buf412 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf405, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg195_1, (2048, 1024), (1, 2048), 0), out=buf412)
        del arg195_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_119, view_51, hidden_states_205, pow_70, variance_68, add_140, rsqrt_69, hidden_states_206, to_144, mul_227, query_states_17, cos_20, mul_230, q_embed_17, linear_120, view_52, hidden_states_207, pow_71, variance_69, add_141, rsqrt_70, hidden_states_208, to_146, mul_229, key_states_17, mul_232, k_embed_17, linear_121, view_53, value_states_17, attn_output_68], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf416 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf413, buf414, reinterpret_tensor(buf412, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf415, scale=0.08838834764831845, enable_gqa=True)
        buf417 = buf416[0]
        assert_tensor_metadata(buf417, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf417, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf416
        buf419 = reinterpret_tensor(buf413, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf413  # reuse
        generated_kernel_7(buf419, buf417, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg196_1, (2048, 2048), (2048, 1), 'input')
        buf420 = reinterpret_tensor(buf417, (71, 2048), (2048, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [transpose_71, attn_output_69, reshape_17, attn_output_71], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf419, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg196_1, (2048, 2048), (1, 2048), 0), out=buf420)
        del arg196_1
        assert_size_stride(arg197_1, (2048, ), (1, ), 'input')
        buf421 = reinterpret_tensor(buf419, (1, 71, 2048), (145408, 2048, 1), 0); del buf419  # reuse
        buf423 = buf405; del buf405  # reuse
        generated_kernel_15(buf421, buf423, buf374, buf380, buf397, buf403, buf420, arg197_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg197_1
        del buf374
        assert_size_stride(arg198_1, (6144, 2048), (2048, 1), 'input')
        buf424 = reinterpret_tensor(buf402, (71, 6144), (6144, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_210, pow_72, variance_70, add_145, rsqrt_71, hidden_states_211, to_148, hidden_states_212, linear_123], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf423, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg198_1, (2048, 6144), (1, 2048), 0), out=buf424)
        del arg198_1
        assert_size_stride(arg199_1, (6144, 2048), (2048, 1), 'input')
        buf425 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf423, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg199_1, (2048, 6144), (1, 2048), 0), out=buf425)
        del arg199_1
        buf426 = reinterpret_tensor(buf400, (1, 71, 6144), (436224, 6144, 1), 0); del buf400  # reuse
        generated_kernel_9(buf426, buf424, buf425, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg200_1, (2048, 6144), (6144, 1), 'input')
        buf427 = reinterpret_tensor(buf423, (71, 2048), (2048, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_236, down_proj_17], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf426, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg200_1, (6144, 2048), (1, 6144), 0), out=buf427)
        del arg200_1
        assert_size_stride(arg201_1, (2048, ), (1, ), 'input')
        buf429 = reinterpret_tensor(buf420, (1, 71, 2048), (145408, 2048, 1), 0); del buf420  # reuse
        generated_kernel_12(buf429, buf421, buf427, arg201_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg201_1
        assert_size_stride(arg202_1, (2048, 2048), (2048, 1), 'input')
        buf430 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_213, hidden_states_214, pow_73, variance_71, add_147, rsqrt_72, hidden_states_215, to_150, hidden_states_216, linear_126], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf429, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg202_1, (2048, 2048), (1, 2048), 0), out=buf430)
        del arg202_1
        assert_size_stride(arg203_1, (128, ), (1, ), 'input')
        buf437 = reinterpret_tensor(buf397, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf397  # reuse
        generated_kernel_1(buf437, buf430, arg203_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg203_1
        assert_size_stride(arg204_1, (1024, 2048), (2048, 1), 'input')
        buf433 = reinterpret_tensor(buf414, (71, 1024), (1024, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf429, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg204_1, (2048, 1024), (1, 2048), 0), out=buf433)
        del arg204_1
        assert_size_stride(arg205_1, (128, ), (1, ), 'input')
        buf438 = reinterpret_tensor(buf412, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf412  # reuse
        generated_kernel_2(buf438, buf433, arg205_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg205_1
        assert_size_stride(arg206_1, (1024, 2048), (2048, 1), 'input')
        buf436 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf429, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg206_1, (2048, 1024), (1, 2048), 0), out=buf436)
        del arg206_1
        buf439 = buf415; del buf415  # reuse
        buf462 = buf392; del buf392  # reuse
        generated_kernel_6(buf439, buf462, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_126, view_54, hidden_states_217, pow_74, variance_72, add_148, rsqrt_73, hidden_states_218, to_152, mul_240, query_states_18, cos_21, mul_243, q_embed_18, linear_127, view_55, hidden_states_219, pow_75, variance_73, add_149, rsqrt_74, hidden_states_220, to_154, mul_242, key_states_18, mul_245, k_embed_18, linear_128, view_56, value_states_18, attn_output_72], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf440 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf437, buf438, reinterpret_tensor(buf436, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf439, scale=0.08838834764831845, enable_gqa=True)
        buf441 = buf440[0]
        assert_tensor_metadata(buf441, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf441, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf440
        buf443 = reinterpret_tensor(buf437, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf437  # reuse
        generated_kernel_7(buf443, buf441, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg207_1, (2048, 2048), (2048, 1), 'input')
        buf444 = reinterpret_tensor(buf441, (71, 2048), (2048, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [transpose_75, attn_output_73, reshape_18, attn_output_75], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf443, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg207_1, (2048, 2048), (1, 2048), 0), out=buf444)
        del arg207_1
        assert_size_stride(arg208_1, (2048, ), (1, ), 'input')
        buf446 = reinterpret_tensor(buf443, (1, 71, 2048), (145408, 2048, 1), 0); del buf443  # reuse
        generated_kernel_13(buf446, buf421, buf427, buf444, arg208_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg208_1
        assert_size_stride(arg209_1, (6144, 2048), (2048, 1), 'input')
        buf447 = reinterpret_tensor(buf426, (71, 6144), (6144, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf446, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg209_1, (2048, 6144), (1, 2048), 0), out=buf447)
        del arg209_1
        assert_size_stride(arg210_1, (6144, 2048), (2048, 1), 'input')
        buf448 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf446, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg210_1, (2048, 6144), (1, 2048), 0), out=buf448)
        del arg210_1
        buf449 = reinterpret_tensor(buf424, (1, 71, 6144), (436224, 6144, 1), 0); del buf424  # reuse
        generated_kernel_9(buf449, buf447, buf448, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg211_1, (2048, 6144), (6144, 1), 'input')
        buf450 = reinterpret_tensor(buf446, (71, 2048), (2048, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_249, down_proj_18], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf449, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg211_1, (6144, 2048), (1, 6144), 0), out=buf450)
        del arg211_1
        assert_size_stride(arg212_1, (2048, ), (1, ), 'input')
        buf452 = buf429; del buf429  # reuse
        generated_kernel_14(buf452, buf421, buf427, buf444, buf450, arg212_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg212_1
        assert_size_stride(arg213_1, (2048, 2048), (2048, 1), 'input')
        buf453 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf452, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg213_1, (2048, 2048), (1, 2048), 0), out=buf453)
        del arg213_1
        assert_size_stride(arg214_1, (128, ), (1, ), 'input')
        buf460 = reinterpret_tensor(buf380, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf380  # reuse
        generated_kernel_1(buf460, buf453, arg214_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg214_1
        del buf453
        assert_size_stride(arg215_1, (1024, 2048), (2048, 1), 'input')
        buf456 = reinterpret_tensor(buf438, (71, 1024), (1024, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf452, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg215_1, (2048, 1024), (1, 2048), 0), out=buf456)
        del arg215_1
        assert_size_stride(arg216_1, (128, ), (1, ), 'input')
        buf461 = reinterpret_tensor(buf436, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf436  # reuse
        generated_kernel_2(buf461, buf456, arg216_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg216_1
        assert_size_stride(arg217_1, (1024, 2048), (2048, 1), 'input')
        buf459 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf452, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg217_1, (2048, 1024), (1, 2048), 0), out=buf459)
        del arg217_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_133, view_57, hidden_states_229, pow_78, variance_76, add_156, rsqrt_77, hidden_states_230, to_160, mul_253, query_states_19, cos_22, mul_256, q_embed_19, linear_134, view_58, hidden_states_231, pow_79, variance_77, add_157, rsqrt_78, hidden_states_232, to_162, mul_255, key_states_19, mul_258, k_embed_19, linear_135, view_59, value_states_19, attn_output_76], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf463 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf460, buf461, reinterpret_tensor(buf459, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf462, scale=0.08838834764831845, enable_gqa=True)
        buf464 = buf463[0]
        assert_tensor_metadata(buf464, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf464, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf463
        buf466 = reinterpret_tensor(buf460, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf460  # reuse
        generated_kernel_7(buf466, buf464, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg218_1, (2048, 2048), (2048, 1), 'input')
        buf467 = reinterpret_tensor(buf464, (71, 2048), (2048, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [transpose_79, attn_output_77, reshape_19, attn_output_79], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf466, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg218_1, (2048, 2048), (1, 2048), 0), out=buf467)
        del arg218_1
        assert_size_stride(arg219_1, (2048, ), (1, ), 'input')
        buf468 = reinterpret_tensor(buf466, (1, 71, 2048), (145408, 2048, 1), 0); del buf466  # reuse
        buf470 = buf452; del buf452  # reuse
        generated_kernel_15(buf468, buf470, buf421, buf427, buf444, buf450, buf467, arg219_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg219_1
        del buf421
        assert_size_stride(arg220_1, (6144, 2048), (2048, 1), 'input')
        buf471 = reinterpret_tensor(buf449, (71, 6144), (6144, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_234, pow_80, variance_78, add_161, rsqrt_79, hidden_states_235, to_164, hidden_states_236, linear_137], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf470, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg220_1, (2048, 6144), (1, 2048), 0), out=buf471)
        del arg220_1
        assert_size_stride(arg221_1, (6144, 2048), (2048, 1), 'input')
        buf472 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf470, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg221_1, (2048, 6144), (1, 2048), 0), out=buf472)
        del arg221_1
        buf473 = reinterpret_tensor(buf447, (1, 71, 6144), (436224, 6144, 1), 0); del buf447  # reuse
        generated_kernel_9(buf473, buf471, buf472, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg222_1, (2048, 6144), (6144, 1), 'input')
        buf474 = reinterpret_tensor(buf470, (71, 2048), (2048, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_262, down_proj_19], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf473, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg222_1, (6144, 2048), (1, 6144), 0), out=buf474)
        del arg222_1
        assert_size_stride(arg223_1, (2048, ), (1, ), 'input')
        buf476 = reinterpret_tensor(buf467, (1, 71, 2048), (145408, 2048, 1), 0); del buf467  # reuse
        generated_kernel_12(buf476, buf468, buf474, arg223_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg223_1
        assert_size_stride(arg224_1, (2048, 2048), (2048, 1), 'input')
        buf477 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_237, hidden_states_238, pow_81, variance_79, add_163, rsqrt_80, hidden_states_239, to_166, hidden_states_240, linear_140], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf476, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg224_1, (2048, 2048), (1, 2048), 0), out=buf477)
        del arg224_1
        assert_size_stride(arg225_1, (128, ), (1, ), 'input')
        buf484 = reinterpret_tensor(buf444, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf444  # reuse
        generated_kernel_1(buf484, buf477, arg225_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg225_1
        assert_size_stride(arg226_1, (1024, 2048), (2048, 1), 'input')
        buf480 = reinterpret_tensor(buf461, (71, 1024), (1024, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf476, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 1024), (1, 2048), 0), out=buf480)
        del arg226_1
        assert_size_stride(arg227_1, (128, ), (1, ), 'input')
        buf485 = reinterpret_tensor(buf459, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf459  # reuse
        generated_kernel_2(buf485, buf480, arg227_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg227_1
        assert_size_stride(arg228_1, (1024, 2048), (2048, 1), 'input')
        buf483 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf476, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg228_1, (2048, 1024), (1, 2048), 0), out=buf483)
        del arg228_1
        buf486 = buf462; del buf462  # reuse
        buf509 = buf439; del buf439  # reuse
        generated_kernel_6(buf486, buf509, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_140, view_60, hidden_states_241, pow_82, variance_80, add_164, rsqrt_81, hidden_states_242, to_168, mul_266, query_states_20, cos_23, mul_269, q_embed_20, linear_141, view_61, hidden_states_243, pow_83, variance_81, add_165, rsqrt_82, hidden_states_244, to_170, mul_268, key_states_20, mul_271, k_embed_20, linear_142, view_62, value_states_20, attn_output_80], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf487 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf484, buf485, reinterpret_tensor(buf483, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf486, scale=0.08838834764831845, enable_gqa=True)
        buf488 = buf487[0]
        assert_tensor_metadata(buf488, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf488, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf487
        buf490 = reinterpret_tensor(buf484, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf484  # reuse
        generated_kernel_7(buf490, buf488, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg229_1, (2048, 2048), (2048, 1), 'input')
        buf491 = reinterpret_tensor(buf488, (71, 2048), (2048, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [transpose_83, attn_output_81, reshape_20, attn_output_83], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf490, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg229_1, (2048, 2048), (1, 2048), 0), out=buf491)
        del arg229_1
        assert_size_stride(arg230_1, (2048, ), (1, ), 'input')
        buf493 = reinterpret_tensor(buf490, (1, 71, 2048), (145408, 2048, 1), 0); del buf490  # reuse
        generated_kernel_13(buf493, buf468, buf474, buf491, arg230_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg230_1
        assert_size_stride(arg231_1, (6144, 2048), (2048, 1), 'input')
        buf494 = reinterpret_tensor(buf473, (71, 6144), (6144, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf493, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg231_1, (2048, 6144), (1, 2048), 0), out=buf494)
        del arg231_1
        assert_size_stride(arg232_1, (6144, 2048), (2048, 1), 'input')
        buf495 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf493, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg232_1, (2048, 6144), (1, 2048), 0), out=buf495)
        del arg232_1
        buf496 = reinterpret_tensor(buf471, (1, 71, 6144), (436224, 6144, 1), 0); del buf471  # reuse
        generated_kernel_9(buf496, buf494, buf495, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg233_1, (2048, 6144), (6144, 1), 'input')
        buf497 = reinterpret_tensor(buf493, (71, 2048), (2048, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_275, down_proj_20], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf496, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg233_1, (6144, 2048), (1, 6144), 0), out=buf497)
        del arg233_1
        assert_size_stride(arg234_1, (2048, ), (1, ), 'input')
        buf499 = buf476; del buf476  # reuse
        generated_kernel_14(buf499, buf468, buf474, buf491, buf497, arg234_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg234_1
        assert_size_stride(arg235_1, (2048, 2048), (2048, 1), 'input')
        buf500 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf499, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg235_1, (2048, 2048), (1, 2048), 0), out=buf500)
        del arg235_1
        assert_size_stride(arg236_1, (128, ), (1, ), 'input')
        buf507 = reinterpret_tensor(buf427, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf427  # reuse
        generated_kernel_1(buf507, buf500, arg236_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg236_1
        del buf500
        assert_size_stride(arg237_1, (1024, 2048), (2048, 1), 'input')
        buf503 = reinterpret_tensor(buf485, (71, 1024), (1024, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf499, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg237_1, (2048, 1024), (1, 2048), 0), out=buf503)
        del arg237_1
        assert_size_stride(arg238_1, (128, ), (1, ), 'input')
        buf508 = reinterpret_tensor(buf483, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf483  # reuse
        generated_kernel_2(buf508, buf503, arg238_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg238_1
        assert_size_stride(arg239_1, (1024, 2048), (2048, 1), 'input')
        buf506 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf499, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg239_1, (2048, 1024), (1, 2048), 0), out=buf506)
        del arg239_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_147, view_63, hidden_states_253, pow_86, variance_84, add_172, rsqrt_85, hidden_states_254, to_176, mul_279, query_states_21, cos_24, mul_282, q_embed_21, linear_148, view_64, hidden_states_255, pow_87, variance_85, add_173, rsqrt_86, hidden_states_256, to_178, mul_281, key_states_21, mul_284, k_embed_21, linear_149, view_65, value_states_21, attn_output_84], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf510 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf507, buf508, reinterpret_tensor(buf506, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf509, scale=0.08838834764831845, enable_gqa=True)
        buf511 = buf510[0]
        assert_tensor_metadata(buf511, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf511, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf510
        buf513 = reinterpret_tensor(buf507, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf507  # reuse
        generated_kernel_7(buf513, buf511, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg240_1, (2048, 2048), (2048, 1), 'input')
        buf514 = reinterpret_tensor(buf511, (71, 2048), (2048, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [transpose_87, attn_output_85, reshape_21, attn_output_87], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf513, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg240_1, (2048, 2048), (1, 2048), 0), out=buf514)
        del arg240_1
        assert_size_stride(arg241_1, (2048, ), (1, ), 'input')
        buf515 = reinterpret_tensor(buf513, (1, 71, 2048), (145408, 2048, 1), 0); del buf513  # reuse
        buf517 = buf499; del buf499  # reuse
        generated_kernel_15(buf515, buf517, buf468, buf474, buf491, buf497, buf514, arg241_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg241_1
        del buf468
        assert_size_stride(arg242_1, (6144, 2048), (2048, 1), 'input')
        buf518 = reinterpret_tensor(buf496, (71, 6144), (6144, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_258, pow_88, variance_86, add_177, rsqrt_87, hidden_states_259, to_180, hidden_states_260, linear_151], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf517, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg242_1, (2048, 6144), (1, 2048), 0), out=buf518)
        del arg242_1
        assert_size_stride(arg243_1, (6144, 2048), (2048, 1), 'input')
        buf519 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf517, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg243_1, (2048, 6144), (1, 2048), 0), out=buf519)
        del arg243_1
        buf520 = reinterpret_tensor(buf494, (1, 71, 6144), (436224, 6144, 1), 0); del buf494  # reuse
        generated_kernel_9(buf520, buf518, buf519, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg244_1, (2048, 6144), (6144, 1), 'input')
        buf521 = reinterpret_tensor(buf517, (71, 2048), (2048, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_288, down_proj_21], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf520, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg244_1, (6144, 2048), (1, 6144), 0), out=buf521)
        del arg244_1
        assert_size_stride(arg245_1, (2048, ), (1, ), 'input')
        buf523 = reinterpret_tensor(buf514, (1, 71, 2048), (145408, 2048, 1), 0); del buf514  # reuse
        generated_kernel_12(buf523, buf515, buf521, arg245_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg245_1
        assert_size_stride(arg246_1, (2048, 2048), (2048, 1), 'input')
        buf524 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_261, hidden_states_262, pow_89, variance_87, add_179, rsqrt_88, hidden_states_263, to_182, hidden_states_264, linear_154], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf523, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg246_1, (2048, 2048), (1, 2048), 0), out=buf524)
        del arg246_1
        assert_size_stride(arg247_1, (128, ), (1, ), 'input')
        buf531 = reinterpret_tensor(buf491, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf491  # reuse
        generated_kernel_1(buf531, buf524, arg247_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg247_1
        assert_size_stride(arg248_1, (1024, 2048), (2048, 1), 'input')
        buf527 = reinterpret_tensor(buf508, (71, 1024), (1024, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf523, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg248_1, (2048, 1024), (1, 2048), 0), out=buf527)
        del arg248_1
        assert_size_stride(arg249_1, (128, ), (1, ), 'input')
        buf532 = reinterpret_tensor(buf506, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf506  # reuse
        generated_kernel_2(buf532, buf527, arg249_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg249_1
        assert_size_stride(arg250_1, (1024, 2048), (2048, 1), 'input')
        buf530 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf523, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg250_1, (2048, 1024), (1, 2048), 0), out=buf530)
        del arg250_1
        buf533 = buf509; del buf509  # reuse
        buf556 = buf486; del buf486  # reuse
        generated_kernel_6(buf533, buf556, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_154, view_66, hidden_states_265, pow_90, variance_88, add_180, rsqrt_89, hidden_states_266, to_184, mul_292, query_states_22, cos_25, mul_295, q_embed_22, linear_155, view_67, hidden_states_267, pow_91, variance_89, add_181, rsqrt_90, hidden_states_268, to_186, mul_294, key_states_22, mul_297, k_embed_22, linear_156, view_68, value_states_22, attn_output_88], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf534 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf531, buf532, reinterpret_tensor(buf530, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf533, scale=0.08838834764831845, enable_gqa=True)
        buf535 = buf534[0]
        assert_tensor_metadata(buf535, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf535, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf534
        buf537 = reinterpret_tensor(buf531, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf531  # reuse
        generated_kernel_7(buf537, buf535, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg251_1, (2048, 2048), (2048, 1), 'input')
        buf538 = reinterpret_tensor(buf535, (71, 2048), (2048, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [transpose_91, attn_output_89, reshape_22, attn_output_91], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf537, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg251_1, (2048, 2048), (1, 2048), 0), out=buf538)
        del arg251_1
        assert_size_stride(arg252_1, (2048, ), (1, ), 'input')
        buf540 = reinterpret_tensor(buf537, (1, 71, 2048), (145408, 2048, 1), 0); del buf537  # reuse
        generated_kernel_13(buf540, buf515, buf521, buf538, arg252_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg252_1
        assert_size_stride(arg253_1, (6144, 2048), (2048, 1), 'input')
        buf541 = reinterpret_tensor(buf520, (71, 6144), (6144, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf540, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 6144), (1, 2048), 0), out=buf541)
        del arg253_1
        assert_size_stride(arg254_1, (6144, 2048), (2048, 1), 'input')
        buf542 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [linear_159], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf540, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg254_1, (2048, 6144), (1, 2048), 0), out=buf542)
        del arg254_1
        buf543 = reinterpret_tensor(buf518, (1, 71, 6144), (436224, 6144, 1), 0); del buf518  # reuse
        generated_kernel_9(buf543, buf541, buf542, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg255_1, (2048, 6144), (6144, 1), 'input')
        buf544 = reinterpret_tensor(buf540, (71, 2048), (2048, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_301, down_proj_22], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf543, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg255_1, (6144, 2048), (1, 6144), 0), out=buf544)
        del arg255_1
        assert_size_stride(arg256_1, (2048, ), (1, ), 'input')
        buf546 = buf523; del buf523  # reuse
        generated_kernel_14(buf546, buf515, buf521, buf538, buf544, arg256_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg256_1
        assert_size_stride(arg257_1, (2048, 2048), (2048, 1), 'input')
        buf547 = buf524; del buf524  # reuse
        # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf546, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg257_1, (2048, 2048), (1, 2048), 0), out=buf547)
        del arg257_1
        assert_size_stride(arg258_1, (128, ), (1, ), 'input')
        buf554 = reinterpret_tensor(buf474, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf474  # reuse
        generated_kernel_1(buf554, buf547, arg258_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg258_1
        del buf547
        assert_size_stride(arg259_1, (1024, 2048), (2048, 1), 'input')
        buf550 = reinterpret_tensor(buf532, (71, 1024), (1024, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf546, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg259_1, (2048, 1024), (1, 2048), 0), out=buf550)
        del arg259_1
        assert_size_stride(arg260_1, (128, ), (1, ), 'input')
        buf555 = reinterpret_tensor(buf530, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf530  # reuse
        generated_kernel_2(buf555, buf550, arg260_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg260_1
        assert_size_stride(arg261_1, (1024, 2048), (2048, 1), 'input')
        buf553 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf546, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg261_1, (2048, 1024), (1, 2048), 0), out=buf553)
        del arg261_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_161, view_69, hidden_states_277, pow_94, variance_92, add_188, rsqrt_93, hidden_states_278, to_192, mul_305, query_states_23, cos_26, mul_308, q_embed_23, linear_162, view_70, hidden_states_279, pow_95, variance_93, add_189, rsqrt_94, hidden_states_280, to_194, mul_307, key_states_23, mul_310, k_embed_23, linear_163, view_71, value_states_23, attn_output_92], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf557 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf554, buf555, reinterpret_tensor(buf553, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf556, scale=0.08838834764831845, enable_gqa=True)
        buf558 = buf557[0]
        assert_tensor_metadata(buf558, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf558, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf557
        buf560 = reinterpret_tensor(buf554, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf554  # reuse
        generated_kernel_7(buf560, buf558, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg262_1, (2048, 2048), (2048, 1), 'input')
        buf561 = reinterpret_tensor(buf558, (71, 2048), (2048, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [transpose_95, attn_output_93, reshape_23, attn_output_95], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf560, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg262_1, (2048, 2048), (1, 2048), 0), out=buf561)
        del arg262_1
        assert_size_stride(arg263_1, (2048, ), (1, ), 'input')
        buf562 = reinterpret_tensor(buf560, (1, 71, 2048), (145408, 2048, 1), 0); del buf560  # reuse
        buf564 = buf546; del buf546  # reuse
        generated_kernel_15(buf562, buf564, buf515, buf521, buf538, buf544, buf561, arg263_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg263_1
        del buf515
        assert_size_stride(arg264_1, (6144, 2048), (2048, 1), 'input')
        buf565 = reinterpret_tensor(buf543, (71, 6144), (6144, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_282, pow_96, variance_94, add_193, rsqrt_95, hidden_states_283, to_196, hidden_states_284, linear_165], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf564, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg264_1, (2048, 6144), (1, 2048), 0), out=buf565)
        del arg264_1
        assert_size_stride(arg265_1, (6144, 2048), (2048, 1), 'input')
        buf566 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf564, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg265_1, (2048, 6144), (1, 2048), 0), out=buf566)
        del arg265_1
        buf567 = reinterpret_tensor(buf541, (1, 71, 6144), (436224, 6144, 1), 0); del buf541  # reuse
        generated_kernel_9(buf567, buf565, buf566, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg266_1, (2048, 6144), (6144, 1), 'input')
        buf568 = reinterpret_tensor(buf564, (71, 2048), (2048, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_314, down_proj_23], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf567, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg266_1, (6144, 2048), (1, 6144), 0), out=buf568)
        del arg266_1
        assert_size_stride(arg267_1, (2048, ), (1, ), 'input')
        buf570 = reinterpret_tensor(buf561, (1, 71, 2048), (145408, 2048, 1), 0); del buf561  # reuse
        generated_kernel_12(buf570, buf562, buf568, arg267_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg267_1
        assert_size_stride(arg268_1, (2048, 2048), (2048, 1), 'input')
        buf571 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_285, hidden_states_286, pow_97, variance_95, add_195, rsqrt_96, hidden_states_287, to_198, hidden_states_288, linear_168], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf570, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg268_1, (2048, 2048), (1, 2048), 0), out=buf571)
        del arg268_1
        assert_size_stride(arg269_1, (128, ), (1, ), 'input')
        buf578 = reinterpret_tensor(buf538, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf538  # reuse
        generated_kernel_1(buf578, buf571, arg269_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg269_1
        assert_size_stride(arg270_1, (1024, 2048), (2048, 1), 'input')
        buf574 = reinterpret_tensor(buf555, (71, 1024), (1024, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf570, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg270_1, (2048, 1024), (1, 2048), 0), out=buf574)
        del arg270_1
        assert_size_stride(arg271_1, (128, ), (1, ), 'input')
        buf579 = reinterpret_tensor(buf553, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf553  # reuse
        generated_kernel_2(buf579, buf574, arg271_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg271_1
        assert_size_stride(arg272_1, (1024, 2048), (2048, 1), 'input')
        buf577 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf570, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg272_1, (2048, 1024), (1, 2048), 0), out=buf577)
        del arg272_1
        buf580 = buf556; del buf556  # reuse
        buf603 = buf533; del buf533  # reuse
        generated_kernel_6(buf580, buf603, buf13, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_168, view_72, hidden_states_289, pow_98, variance_96, add_196, rsqrt_97, hidden_states_290, to_200, mul_318, query_states_24, cos_27, mul_321, q_embed_24, linear_169, view_73, hidden_states_291, pow_99, variance_97, add_197, rsqrt_98, hidden_states_292, to_202, mul_320, key_states_24, mul_323, k_embed_24, linear_170, view_74, value_states_24, attn_output_96], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf581 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf578, buf579, reinterpret_tensor(buf577, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf580, scale=0.08838834764831845, enable_gqa=True)
        buf582 = buf581[0]
        assert_tensor_metadata(buf582, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf582, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf581
        buf584 = reinterpret_tensor(buf578, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf578  # reuse
        generated_kernel_7(buf584, buf582, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg273_1, (2048, 2048), (2048, 1), 'input')
        buf585 = reinterpret_tensor(buf582, (71, 2048), (2048, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [transpose_99, attn_output_97, reshape_24, attn_output_99], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf584, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg273_1, (2048, 2048), (1, 2048), 0), out=buf585)
        del arg273_1
        assert_size_stride(arg274_1, (2048, ), (1, ), 'input')
        buf587 = reinterpret_tensor(buf584, (1, 71, 2048), (145408, 2048, 1), 0); del buf584  # reuse
        generated_kernel_13(buf587, buf562, buf568, buf585, arg274_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg274_1
        assert_size_stride(arg275_1, (6144, 2048), (2048, 1), 'input')
        buf588 = reinterpret_tensor(buf567, (71, 6144), (6144, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [linear_172], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf587, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg275_1, (2048, 6144), (1, 2048), 0), out=buf588)
        del arg275_1
        assert_size_stride(arg276_1, (6144, 2048), (2048, 1), 'input')
        buf589 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [linear_173], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf587, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg276_1, (2048, 6144), (1, 2048), 0), out=buf589)
        del arg276_1
        buf590 = reinterpret_tensor(buf565, (1, 71, 6144), (436224, 6144, 1), 0); del buf565  # reuse
        generated_kernel_9(buf590, buf588, buf589, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg277_1, (2048, 6144), (6144, 1), 'input')
        buf591 = reinterpret_tensor(buf587, (71, 2048), (2048, 1), 0); del buf587  # reuse
        # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_327, down_proj_24], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf590, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg277_1, (6144, 2048), (1, 6144), 0), out=buf591)
        del arg277_1
        assert_size_stride(arg278_1, (2048, ), (1, ), 'input')
        buf593 = buf570; del buf570  # reuse
        generated_kernel_14(buf593, buf562, buf568, buf585, buf591, arg278_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg278_1
        assert_size_stride(arg279_1, (2048, 2048), (2048, 1), 'input')
        buf594 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [linear_175], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf593, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg279_1, (2048, 2048), (1, 2048), 0), out=buf594)
        del arg279_1
        assert_size_stride(arg280_1, (128, ), (1, ), 'input')
        buf601 = reinterpret_tensor(buf521, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf521  # reuse
        generated_kernel_1(buf601, buf594, arg280_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg280_1
        del buf594
        assert_size_stride(arg281_1, (1024, 2048), (2048, 1), 'input')
        buf597 = reinterpret_tensor(buf579, (71, 1024), (1024, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf593, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg281_1, (2048, 1024), (1, 2048), 0), out=buf597)
        del arg281_1
        assert_size_stride(arg282_1, (128, ), (1, ), 'input')
        buf602 = reinterpret_tensor(buf577, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf577  # reuse
        generated_kernel_2(buf602, buf597, arg282_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg282_1
        assert_size_stride(arg283_1, (1024, 2048), (2048, 1), 'input')
        buf600 = buf597; del buf597  # reuse
        # Topologically Sorted Source Nodes: [linear_177], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf593, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg283_1, (2048, 1024), (1, 2048), 0), out=buf600)
        del arg283_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_175, view_75, hidden_states_301, pow_102, variance_100, add_204, rsqrt_101, hidden_states_302, to_208, mul_331, query_states_25, cos_28, mul_334, q_embed_25, linear_176, view_76, hidden_states_303, pow_103, variance_101, add_205, rsqrt_102, hidden_states_304, to_210, mul_333, key_states_25, mul_336, k_embed_25, linear_177, view_77, value_states_25, attn_output_100], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf604 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf601, buf602, reinterpret_tensor(buf600, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf603, scale=0.08838834764831845, enable_gqa=True)
        buf605 = buf604[0]
        assert_tensor_metadata(buf605, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf605, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf604
        buf607 = reinterpret_tensor(buf601, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf601  # reuse
        generated_kernel_7(buf607, buf605, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg284_1, (2048, 2048), (2048, 1), 'input')
        buf608 = reinterpret_tensor(buf605, (71, 2048), (2048, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [transpose_103, attn_output_101, reshape_25, attn_output_103], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf607, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 2048), (1, 2048), 0), out=buf608)
        del arg284_1
        assert_size_stride(arg285_1, (2048, ), (1, ), 'input')
        buf609 = reinterpret_tensor(buf607, (1, 71, 2048), (145408, 2048, 1), 0); del buf607  # reuse
        buf611 = buf593; del buf593  # reuse
        generated_kernel_15(buf609, buf611, buf562, buf568, buf585, buf591, buf608, arg285_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg285_1
        del buf562
        assert_size_stride(arg286_1, (6144, 2048), (2048, 1), 'input')
        buf612 = reinterpret_tensor(buf590, (71, 6144), (6144, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_306, pow_104, variance_102, add_209, rsqrt_103, hidden_states_307, to_212, hidden_states_308, linear_179], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf611, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg286_1, (2048, 6144), (1, 2048), 0), out=buf612)
        del arg286_1
        assert_size_stride(arg287_1, (6144, 2048), (2048, 1), 'input')
        buf613 = buf589; del buf589  # reuse
        # Topologically Sorted Source Nodes: [linear_180], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf611, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg287_1, (2048, 6144), (1, 2048), 0), out=buf613)
        del arg287_1
        buf614 = reinterpret_tensor(buf588, (1, 71, 6144), (436224, 6144, 1), 0); del buf588  # reuse
        generated_kernel_9(buf614, buf612, buf613, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg288_1, (2048, 6144), (6144, 1), 'input')
        buf615 = reinterpret_tensor(buf611, (71, 2048), (2048, 1), 0); del buf611  # reuse
        # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_340, down_proj_25], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf614, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg288_1, (6144, 2048), (1, 6144), 0), out=buf615)
        del arg288_1
        assert_size_stride(arg289_1, (2048, ), (1, ), 'input')
        buf617 = reinterpret_tensor(buf608, (1, 71, 2048), (145408, 2048, 1), 0); del buf608  # reuse
        generated_kernel_12(buf617, buf609, buf615, arg289_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg289_1
        assert_size_stride(arg290_1, (2048, 2048), (2048, 1), 'input')
        buf618 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_309, hidden_states_310, pow_105, variance_103, add_211, rsqrt_104, hidden_states_311, to_214, hidden_states_312, linear_182], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf617, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg290_1, (2048, 2048), (1, 2048), 0), out=buf618)
        del arg290_1
        assert_size_stride(arg291_1, (128, ), (1, ), 'input')
        buf625 = reinterpret_tensor(buf585, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf585  # reuse
        generated_kernel_1(buf625, buf618, arg291_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg291_1
        assert_size_stride(arg292_1, (1024, 2048), (2048, 1), 'input')
        buf621 = reinterpret_tensor(buf602, (71, 1024), (1024, 1), 0); del buf602  # reuse
        # Topologically Sorted Source Nodes: [linear_183], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf617, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg292_1, (2048, 1024), (1, 2048), 0), out=buf621)
        del arg292_1
        assert_size_stride(arg293_1, (128, ), (1, ), 'input')
        buf626 = reinterpret_tensor(buf600, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf600  # reuse
        generated_kernel_2(buf626, buf621, arg293_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg293_1
        assert_size_stride(arg294_1, (1024, 2048), (2048, 1), 'input')
        buf624 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [linear_184], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf617, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg294_1, (2048, 1024), (1, 2048), 0), out=buf624)
        del arg294_1
        buf627 = buf603; del buf603  # reuse
        buf650 = buf580; del buf580  # reuse
        generated_kernel_6(buf627, buf650, buf13, threads=[5041], error_buf_idx=3)
        del buf13
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_182, view_78, hidden_states_313, pow_106, variance_104, add_212, rsqrt_105, hidden_states_314, to_216, mul_344, query_states_26, cos_29, mul_347, q_embed_26, linear_183, view_79, hidden_states_315, pow_107, variance_105, add_213, rsqrt_106, hidden_states_316, to_218, mul_346, key_states_26, mul_349, k_embed_26, linear_184, view_80, value_states_26, attn_output_104], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf628 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf625, buf626, reinterpret_tensor(buf624, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf627, scale=0.08838834764831845, enable_gqa=True)
        del buf627
        buf629 = buf628[0]
        assert_tensor_metadata(buf629, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf629, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf628
        buf631 = reinterpret_tensor(buf625, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf625  # reuse
        generated_kernel_7(buf631, buf629, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg295_1, (2048, 2048), (2048, 1), 'input')
        buf632 = reinterpret_tensor(buf629, (71, 2048), (2048, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [transpose_107, attn_output_105, reshape_26, attn_output_107], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf631, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg295_1, (2048, 2048), (1, 2048), 0), out=buf632)
        del arg295_1
        assert_size_stride(arg296_1, (2048, ), (1, ), 'input')
        buf634 = reinterpret_tensor(buf631, (1, 71, 2048), (145408, 2048, 1), 0); del buf631  # reuse
        generated_kernel_13(buf634, buf609, buf615, buf632, arg296_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg296_1
        assert_size_stride(arg297_1, (6144, 2048), (2048, 1), 'input')
        buf635 = reinterpret_tensor(buf614, (71, 6144), (6144, 1), 0); del buf614  # reuse
        # Topologically Sorted Source Nodes: [linear_186], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf634, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg297_1, (2048, 6144), (1, 2048), 0), out=buf635)
        del arg297_1
        assert_size_stride(arg298_1, (6144, 2048), (2048, 1), 'input')
        buf636 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf634, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg298_1, (2048, 6144), (1, 2048), 0), out=buf636)
        del arg298_1
        buf637 = reinterpret_tensor(buf612, (1, 71, 6144), (436224, 6144, 1), 0); del buf612  # reuse
        generated_kernel_9(buf637, buf635, buf636, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg299_1, (2048, 6144), (6144, 1), 'input')
        buf638 = reinterpret_tensor(buf634, (71, 2048), (2048, 1), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_353, down_proj_26], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf637, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg299_1, (6144, 2048), (1, 6144), 0), out=buf638)
        del arg299_1
        assert_size_stride(arg300_1, (2048, ), (1, ), 'input')
        buf640 = buf617; del buf617  # reuse
        generated_kernel_14(buf640, buf609, buf615, buf632, buf638, arg300_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg300_1
        assert_size_stride(arg301_1, (2048, 2048), (2048, 1), 'input')
        buf641 = buf618; del buf618  # reuse
        # Topologically Sorted Source Nodes: [linear_189], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf640, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg301_1, (2048, 2048), (1, 2048), 0), out=buf641)
        del arg301_1
        assert_size_stride(arg302_1, (128, ), (1, ), 'input')
        buf648 = reinterpret_tensor(buf568, (1, 16, 71, 128), (145408, 128, 2048, 1), 0); del buf568  # reuse
        generated_kernel_1(buf648, buf641, arg302_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg302_1
        del buf641
        assert_size_stride(arg303_1, (1024, 2048), (2048, 1), 'input')
        buf644 = reinterpret_tensor(buf626, (71, 1024), (1024, 1), 0); del buf626  # reuse
        # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf640, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg303_1, (2048, 1024), (1, 2048), 0), out=buf644)
        del arg303_1
        assert_size_stride(arg304_1, (128, ), (1, ), 'input')
        buf649 = reinterpret_tensor(buf624, (1, 8, 71, 128), (72704, 128, 1024, 1), 0); del buf624  # reuse
        generated_kernel_2(buf649, buf644, arg304_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg2_1
        del arg304_1
        assert_size_stride(arg305_1, (1024, 2048), (2048, 1), 'input')
        buf647 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf640, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg305_1, (2048, 1024), (1, 2048), 0), out=buf647)
        del arg305_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, cos_1, cos_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, linear_189, view_81, hidden_states_325, pow_110, variance_108, add_220, rsqrt_109, hidden_states_326, to_224, mul_357, query_states_27, cos_30, mul_360, q_embed_27, linear_190, view_82, hidden_states_327, pow_111, variance_109, add_221, rsqrt_110, hidden_states_328, to_226, mul_359, key_states_27, mul_362, k_embed_27, linear_191, view_83, value_states_27, attn_output_108], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf651 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf648, buf649, reinterpret_tensor(buf647, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf650, scale=0.08838834764831845, enable_gqa=True)
        del buf647
        del buf649
        del buf650
        buf652 = buf651[0]
        assert_tensor_metadata(buf652, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf652, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf651
        buf654 = reinterpret_tensor(buf648, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf648  # reuse
        generated_kernel_7(buf654, buf652, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg306_1, (2048, 2048), (2048, 1), 'input')
        buf655 = reinterpret_tensor(buf652, (71, 2048), (2048, 1), 0); del buf652  # reuse
        # Topologically Sorted Source Nodes: [transpose_111, attn_output_109, reshape_27, attn_output_111], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf654, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg306_1, (2048, 2048), (1, 2048), 0), out=buf655)
        del arg306_1
        assert_size_stride(arg307_1, (2048, ), (1, ), 'input')
        buf656 = reinterpret_tensor(buf654, (1, 71, 2048), (145408, 2048, 1), 0); del buf654  # reuse
        buf658 = buf640; del buf640  # reuse
        generated_kernel_15(buf656, buf658, buf609, buf615, buf632, buf638, buf655, arg307_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg307_1
        del buf609
        del buf615
        del buf632
        del buf638
        assert_size_stride(arg308_1, (6144, 2048), (2048, 1), 'input')
        buf659 = reinterpret_tensor(buf637, (71, 6144), (6144, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_330, pow_112, variance_110, add_225, rsqrt_111, hidden_states_331, to_228, hidden_states_332, linear_193], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf658, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg308_1, (2048, 6144), (1, 2048), 0), out=buf659)
        del arg308_1
        assert_size_stride(arg309_1, (6144, 2048), (2048, 1), 'input')
        buf660 = buf636; del buf636  # reuse
        # Topologically Sorted Source Nodes: [linear_194], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf658, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg309_1, (2048, 6144), (1, 2048), 0), out=buf660)
        del arg309_1
        buf661 = reinterpret_tensor(buf635, (1, 71, 6144), (436224, 6144, 1), 0); del buf635  # reuse
        generated_kernel_9(buf661, buf659, buf660, threads=[436224], error_buf_idx=3)
        del buf659
        del buf660
        assert_size_stride(arg310_1, (2048, 6144), (6144, 1), 'input')
        buf662 = reinterpret_tensor(buf658, (71, 2048), (2048, 1), 0); del buf658  # reuse
        # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_366, down_proj_27], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf661, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg310_1, (6144, 2048), (1, 6144), 0), out=buf662)
        del arg310_1
        del buf661
        assert_size_stride(arg311_1, (2048, ), (1, ), 'input')
        buf664 = reinterpret_tensor(buf655, (1, 71, 2048), (145408, 2048, 1), 0); del buf655  # reuse
        generated_kernel_12(buf664, buf656, buf662, arg311_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg311_1
        del buf656
        del buf662
        buf665 = empty_strided((71, 151936), (151936, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [down_proj_27, hidden_states_333, hidden_states_334, pow_113, variance_111, add_227, rsqrt_112, hidden_states_335, to_230, hidden_states_336, hid, linear_196], Original ATen: [aten.linear, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.select]
        extern_kernels.mm(reinterpret_tensor(buf664, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg1_1, (2048, 151936), (1, 2048), 0), out=buf665)
        del arg1_1
        return (reinterpret_tensor(buf664, (71, 2048), (2048, 1), 0), buf665, buf338, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 71), (71, 1), device='mps:0', dtype=torch.int64)
    arg1_1 = rand_strided((151936, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg198_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg199_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg200_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg201_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg202_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg203_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg204_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg205_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg206_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg207_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg208_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg209_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg210_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg211_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg212_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg213_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg214_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg215_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg216_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg217_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg218_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg219_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg220_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg221_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg222_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg223_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg224_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg225_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg226_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg227_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg228_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg229_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg230_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg231_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg232_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg233_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg234_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg235_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg236_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg237_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg238_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg239_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg240_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg241_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg242_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg243_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg244_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg245_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg246_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg247_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg248_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg249_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg250_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg251_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg252_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg253_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg254_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg255_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg256_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg257_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg258_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg259_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg260_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg261_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg262_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg263_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg264_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg265_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg266_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg267_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg268_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg269_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg270_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg271_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg272_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg273_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg274_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg275_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg276_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg277_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg278_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg279_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg280_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg281_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg282_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg283_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg284_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg285_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg286_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg287_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg288_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg289_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg290_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg291_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg292_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg293_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg294_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg295_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg296_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg297_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg298_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg299_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg300_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg301_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg302_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg303_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg304_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg305_1 = rand_strided((1024, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg306_1 = rand_strided((2048, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg307_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    arg308_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg309_1 = rand_strided((6144, 2048), (2048, 1), device='mps:0', dtype=torch.bfloat16)
    arg310_1 = rand_strided((2048, 6144), (6144, 1), device='mps:0', dtype=torch.bfloat16)
    arg311_1 = rand_strided((2048, ), (1, ), device='mps:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='mps')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
