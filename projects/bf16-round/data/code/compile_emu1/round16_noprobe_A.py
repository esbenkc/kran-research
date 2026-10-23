# AOT ID: ['8_inference']
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
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp3 = static_cast<bfloat>(tmp2);
            auto tmp4 = static_cast<float>(tmp3);
            auto tmp5 = tmp4 * tmp4;
            tmp_acc_1 += tmp5;
        }
        auto tmp6 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp7 = static_cast<float>(in_ptr2[r0_1]);
            auto tmp10 = in_ptr0[x0];
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            if ((tmp10 < 0 || tmp10 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp10, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp11 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp10]);
            auto tmp12 = static_cast<bfloat>(tmp11);
            auto tmp13 = static_cast<bfloat>(tmp12);
            auto tmp14 = static_cast<float>(tmp13);
            auto tmp15 = 2048.0;
            auto tmp16 = tmp6 / tmp15;
            auto tmp17 = 1e-06;
            auto tmp18 = tmp16 + tmp17;
            auto tmp19 = metal::precise::rsqrt(tmp18);
            auto tmp20 = tmp14 * tmp19;
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = static_cast<bfloat>(tmp21);
            auto tmp23 = static_cast<bfloat>(tmp22);
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp25 = static_cast<bfloat>(tmp24);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp27 = tmp9 * tmp26;
            auto tmp28 = static_cast<bfloat>(tmp27);
            auto tmp29 = static_cast<bfloat>(tmp28);
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp29);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_1 = async_compile.metal('generated_kernel_1', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_1(
        device bfloat* out_ptr3,
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
        int x2 = (xindex) % (16);
        auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 128*x0]);
        auto tmp1 = static_cast<bfloat>(tmp0);
        auto tmp2 = static_cast<bfloat>(tmp1);
        auto tmp3 = static_cast<float>(tmp2);
        auto tmp4 = tmp3 * tmp3;
        auto tmp5 = c10::metal::threadgroup_sum(tmp_acc_0, tmp4, r0_index * 1, 128);
        auto tmp18 = static_cast<float>(in_ptr1[r0_1]);
        auto tmp38 = in_ptr2[(r0_1) % (64)];
        auto tmp6 = r0_1;
        auto tmp7 = static_cast<long>(tmp6);
        auto tmp8 = 0;
        auto tmp9 = tmp7 >= tmp8;
        auto tmp10 = 64;
        auto tmp11 = tmp7 < tmp10;
        bfloat tmp12;
        if (tmp11) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[64 + (r0_1)]);
            auto tmp_scoped1 = static_cast<bfloat>(tmp_scoped0);
            auto tmp_scoped2 = static_cast<bfloat>(tmp_scoped1);
            auto tmp_scoped3 = static_cast<float>(in_ptr0[64 + 128*x0 + (r0_1)]);
            auto tmp_scoped4 = static_cast<bfloat>(tmp_scoped3);
            auto tmp_scoped5 = static_cast<bfloat>(tmp_scoped4);
            auto tmp_scoped6 = static_cast<float>(tmp_scoped5);
            auto tmp_scoped7 = 128.0;
            auto tmp_scoped8 = tmp5 / tmp_scoped7;
            auto tmp_scoped9 = 1e-06;
            auto tmp_scoped10 = tmp_scoped8 + tmp_scoped9;
            auto tmp_scoped11 = metal::precise::rsqrt(tmp_scoped10);
            auto tmp_scoped12 = tmp_scoped6 * tmp_scoped11;
            auto tmp_scoped13 = static_cast<bfloat>(tmp_scoped12);
            auto tmp_scoped14 = static_cast<bfloat>(tmp_scoped13);
            auto tmp_scoped15 = static_cast<bfloat>(tmp_scoped14);
            auto tmp_scoped16 = static_cast<bfloat>(tmp_scoped15);
            auto tmp_scoped17 = static_cast<bfloat>(tmp_scoped16);
            auto tmp_scoped18 = static_cast<bfloat>(tmp_scoped17);
            auto tmp_scoped19 = tmp_scoped2 * tmp_scoped18;
            auto tmp_scoped20 = static_cast<bfloat>(tmp_scoped19);
            auto tmp_scoped21 = static_cast<bfloat>(tmp_scoped20);
            auto tmp_scoped22 = static_cast<bfloat>(tmp_scoped21);
            auto tmp_scoped23 = static_cast<bfloat>(tmp_scoped22);
            auto tmp_scoped24 = static_cast<decltype(tmp_scoped23)>(-tmp_scoped23);
            auto tmp_scoped25 = static_cast<bfloat>(tmp_scoped24);
            auto tmp_scoped26 = static_cast<bfloat>(tmp_scoped25);
            tmp12 = static_cast<decltype(tmp12)>(tmp_scoped26);
        } else tmp12 = static_cast<decltype(tmp12)>(0.0);
        auto tmp13 = tmp7 >= tmp10;
        auto tmp14 = 128;
        auto tmp15 = tmp7 < tmp14;
        bfloat tmp16;
        if (tmp13) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[(-64) + r0_1]);
            auto tmp_scoped1 = static_cast<bfloat>(tmp_scoped0);
            auto tmp_scoped2 = static_cast<bfloat>(tmp_scoped1);
            auto tmp_scoped3 = static_cast<float>(in_ptr0[128*x0 + ((-64) + r0_1)]);
            auto tmp_scoped4 = static_cast<bfloat>(tmp_scoped3);
            auto tmp_scoped5 = static_cast<bfloat>(tmp_scoped4);
            auto tmp_scoped6 = static_cast<float>(tmp_scoped5);
            auto tmp_scoped7 = 128.0;
            auto tmp_scoped8 = tmp5 / tmp_scoped7;
            auto tmp_scoped9 = 1e-06;
            auto tmp_scoped10 = tmp_scoped8 + tmp_scoped9;
            auto tmp_scoped11 = metal::precise::rsqrt(tmp_scoped10);
            auto tmp_scoped12 = tmp_scoped6 * tmp_scoped11;
            auto tmp_scoped13 = static_cast<bfloat>(tmp_scoped12);
            auto tmp_scoped14 = static_cast<bfloat>(tmp_scoped13);
            auto tmp_scoped15 = static_cast<bfloat>(tmp_scoped14);
            auto tmp_scoped16 = static_cast<bfloat>(tmp_scoped15);
            auto tmp_scoped17 = static_cast<bfloat>(tmp_scoped16);
            auto tmp_scoped18 = static_cast<bfloat>(tmp_scoped17);
            auto tmp_scoped19 = tmp_scoped2 * tmp_scoped18;
            auto tmp_scoped20 = static_cast<bfloat>(tmp_scoped19);
            auto tmp_scoped21 = static_cast<bfloat>(tmp_scoped20);
            tmp16 = static_cast<decltype(tmp16)>(tmp_scoped21);
        } else tmp16 = static_cast<decltype(tmp16)>(0.0);
        auto tmp17 = tmp11 ? tmp12 : static_cast<decltype(tmp12)>(tmp16);
        auto tmp19 = static_cast<bfloat>(tmp18);
        auto tmp20 = static_cast<bfloat>(tmp19);
        auto tmp21 = 128.0;
        auto tmp22 = tmp5 / tmp21;
        auto tmp23 = 1e-06;
        auto tmp24 = tmp22 + tmp23;
        auto tmp25 = metal::precise::rsqrt(tmp24);
        auto tmp26 = tmp3 * tmp25;
        auto tmp27 = static_cast<bfloat>(tmp26);
        auto tmp28 = static_cast<bfloat>(tmp27);
        auto tmp29 = static_cast<bfloat>(tmp28);
        auto tmp30 = static_cast<bfloat>(tmp29);
        auto tmp31 = static_cast<bfloat>(tmp30);
        auto tmp32 = static_cast<bfloat>(tmp31);
        auto tmp33 = tmp20 * tmp32;
        auto tmp34 = static_cast<bfloat>(tmp33);
        auto tmp35 = static_cast<bfloat>(tmp34);
        auto tmp36 = static_cast<bfloat>(tmp35);
        auto tmp37 = static_cast<bfloat>(tmp36);
        auto tmp39 = x3;
        auto tmp40 = static_cast<float>(tmp39);
        auto tmp41 = tmp40 * tmp38;
        auto tmp42 = metal::precise::cos(tmp41);
        auto tmp43 = static_cast<bfloat>(tmp42);
        auto tmp44 = static_cast<bfloat>(tmp43);
        auto tmp45 = static_cast<bfloat>(tmp44);
        auto tmp46 = static_cast<bfloat>(tmp45);
        auto tmp47 = static_cast<bfloat>(tmp46);
        auto tmp48 = static_cast<bfloat>(tmp47);
        auto tmp49 = tmp37 * tmp48;
        auto tmp50 = static_cast<bfloat>(tmp49);
        auto tmp51 = static_cast<bfloat>(tmp50);
        auto tmp52 = static_cast<bfloat>(tmp51);
        auto tmp53 = static_cast<bfloat>(tmp52);
        auto tmp54 = static_cast<bfloat>(tmp17);
        auto tmp55 = static_cast<bfloat>(tmp54);
        auto tmp56 = metal::precise::sin(tmp41);
        auto tmp57 = static_cast<bfloat>(tmp56);
        auto tmp58 = static_cast<bfloat>(tmp57);
        auto tmp59 = static_cast<bfloat>(tmp58);
        auto tmp60 = static_cast<bfloat>(tmp59);
        auto tmp61 = static_cast<bfloat>(tmp60);
        auto tmp62 = static_cast<bfloat>(tmp61);
        auto tmp63 = tmp55 * tmp62;
        auto tmp64 = static_cast<bfloat>(tmp63);
        auto tmp65 = static_cast<bfloat>(tmp64);
        auto tmp66 = static_cast<bfloat>(tmp65);
        auto tmp67 = static_cast<bfloat>(tmp66);
        auto tmp68 = tmp53 + tmp67;
        auto tmp69 = static_cast<bfloat>(tmp68);
        auto tmp70 = static_cast<bfloat>(tmp69);
        out_ptr3[r0_1 + 128*x3 + 9088*x2] = static_cast<bfloat>(tmp70);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_2 = async_compile.metal('generated_kernel_2', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_2(
        device bfloat* out_ptr3,
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
        int x2 = (xindex) % (8);
        auto tmp0 = static_cast<float>(in_ptr0[r0_1 + 128*x0]);
        auto tmp1 = static_cast<bfloat>(tmp0);
        auto tmp2 = static_cast<bfloat>(tmp1);
        auto tmp3 = static_cast<float>(tmp2);
        auto tmp4 = tmp3 * tmp3;
        auto tmp5 = c10::metal::threadgroup_sum(tmp_acc_0, tmp4, r0_index * 1, 128);
        auto tmp18 = static_cast<float>(in_ptr1[r0_1]);
        auto tmp38 = in_ptr2[(r0_1) % (64)];
        auto tmp6 = r0_1;
        auto tmp7 = static_cast<long>(tmp6);
        auto tmp8 = 0;
        auto tmp9 = tmp7 >= tmp8;
        auto tmp10 = 64;
        auto tmp11 = tmp7 < tmp10;
        bfloat tmp12;
        if (tmp11) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[64 + (r0_1)]);
            auto tmp_scoped1 = static_cast<bfloat>(tmp_scoped0);
            auto tmp_scoped2 = static_cast<bfloat>(tmp_scoped1);
            auto tmp_scoped3 = static_cast<float>(in_ptr0[64 + 128*x0 + (r0_1)]);
            auto tmp_scoped4 = static_cast<bfloat>(tmp_scoped3);
            auto tmp_scoped5 = static_cast<bfloat>(tmp_scoped4);
            auto tmp_scoped6 = static_cast<float>(tmp_scoped5);
            auto tmp_scoped7 = 128.0;
            auto tmp_scoped8 = tmp5 / tmp_scoped7;
            auto tmp_scoped9 = 1e-06;
            auto tmp_scoped10 = tmp_scoped8 + tmp_scoped9;
            auto tmp_scoped11 = metal::precise::rsqrt(tmp_scoped10);
            auto tmp_scoped12 = tmp_scoped6 * tmp_scoped11;
            auto tmp_scoped13 = static_cast<bfloat>(tmp_scoped12);
            auto tmp_scoped14 = static_cast<bfloat>(tmp_scoped13);
            auto tmp_scoped15 = static_cast<bfloat>(tmp_scoped14);
            auto tmp_scoped16 = static_cast<bfloat>(tmp_scoped15);
            auto tmp_scoped17 = static_cast<bfloat>(tmp_scoped16);
            auto tmp_scoped18 = static_cast<bfloat>(tmp_scoped17);
            auto tmp_scoped19 = tmp_scoped2 * tmp_scoped18;
            auto tmp_scoped20 = static_cast<bfloat>(tmp_scoped19);
            auto tmp_scoped21 = static_cast<bfloat>(tmp_scoped20);
            auto tmp_scoped22 = static_cast<bfloat>(tmp_scoped21);
            auto tmp_scoped23 = static_cast<bfloat>(tmp_scoped22);
            auto tmp_scoped24 = static_cast<decltype(tmp_scoped23)>(-tmp_scoped23);
            auto tmp_scoped25 = static_cast<bfloat>(tmp_scoped24);
            auto tmp_scoped26 = static_cast<bfloat>(tmp_scoped25);
            tmp12 = static_cast<decltype(tmp12)>(tmp_scoped26);
        } else tmp12 = static_cast<decltype(tmp12)>(0.0);
        auto tmp13 = tmp7 >= tmp10;
        auto tmp14 = 128;
        auto tmp15 = tmp7 < tmp14;
        bfloat tmp16;
        if (tmp13) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[(-64) + r0_1]);
            auto tmp_scoped1 = static_cast<bfloat>(tmp_scoped0);
            auto tmp_scoped2 = static_cast<bfloat>(tmp_scoped1);
            auto tmp_scoped3 = static_cast<float>(in_ptr0[128*x0 + ((-64) + r0_1)]);
            auto tmp_scoped4 = static_cast<bfloat>(tmp_scoped3);
            auto tmp_scoped5 = static_cast<bfloat>(tmp_scoped4);
            auto tmp_scoped6 = static_cast<float>(tmp_scoped5);
            auto tmp_scoped7 = 128.0;
            auto tmp_scoped8 = tmp5 / tmp_scoped7;
            auto tmp_scoped9 = 1e-06;
            auto tmp_scoped10 = tmp_scoped8 + tmp_scoped9;
            auto tmp_scoped11 = metal::precise::rsqrt(tmp_scoped10);
            auto tmp_scoped12 = tmp_scoped6 * tmp_scoped11;
            auto tmp_scoped13 = static_cast<bfloat>(tmp_scoped12);
            auto tmp_scoped14 = static_cast<bfloat>(tmp_scoped13);
            auto tmp_scoped15 = static_cast<bfloat>(tmp_scoped14);
            auto tmp_scoped16 = static_cast<bfloat>(tmp_scoped15);
            auto tmp_scoped17 = static_cast<bfloat>(tmp_scoped16);
            auto tmp_scoped18 = static_cast<bfloat>(tmp_scoped17);
            auto tmp_scoped19 = tmp_scoped2 * tmp_scoped18;
            auto tmp_scoped20 = static_cast<bfloat>(tmp_scoped19);
            auto tmp_scoped21 = static_cast<bfloat>(tmp_scoped20);
            tmp16 = static_cast<decltype(tmp16)>(tmp_scoped21);
        } else tmp16 = static_cast<decltype(tmp16)>(0.0);
        auto tmp17 = tmp11 ? tmp12 : static_cast<decltype(tmp12)>(tmp16);
        auto tmp19 = static_cast<bfloat>(tmp18);
        auto tmp20 = static_cast<bfloat>(tmp19);
        auto tmp21 = 128.0;
        auto tmp22 = tmp5 / tmp21;
        auto tmp23 = 1e-06;
        auto tmp24 = tmp22 + tmp23;
        auto tmp25 = metal::precise::rsqrt(tmp24);
        auto tmp26 = tmp3 * tmp25;
        auto tmp27 = static_cast<bfloat>(tmp26);
        auto tmp28 = static_cast<bfloat>(tmp27);
        auto tmp29 = static_cast<bfloat>(tmp28);
        auto tmp30 = static_cast<bfloat>(tmp29);
        auto tmp31 = static_cast<bfloat>(tmp30);
        auto tmp32 = static_cast<bfloat>(tmp31);
        auto tmp33 = tmp20 * tmp32;
        auto tmp34 = static_cast<bfloat>(tmp33);
        auto tmp35 = static_cast<bfloat>(tmp34);
        auto tmp36 = static_cast<bfloat>(tmp35);
        auto tmp37 = static_cast<bfloat>(tmp36);
        auto tmp39 = x3;
        auto tmp40 = static_cast<float>(tmp39);
        auto tmp41 = tmp40 * tmp38;
        auto tmp42 = metal::precise::cos(tmp41);
        auto tmp43 = static_cast<bfloat>(tmp42);
        auto tmp44 = static_cast<bfloat>(tmp43);
        auto tmp45 = static_cast<bfloat>(tmp44);
        auto tmp46 = static_cast<bfloat>(tmp45);
        auto tmp47 = static_cast<bfloat>(tmp46);
        auto tmp48 = static_cast<bfloat>(tmp47);
        auto tmp49 = tmp37 * tmp48;
        auto tmp50 = static_cast<bfloat>(tmp49);
        auto tmp51 = static_cast<bfloat>(tmp50);
        auto tmp52 = static_cast<bfloat>(tmp51);
        auto tmp53 = static_cast<bfloat>(tmp52);
        auto tmp54 = static_cast<bfloat>(tmp17);
        auto tmp55 = static_cast<bfloat>(tmp54);
        auto tmp56 = metal::precise::sin(tmp41);
        auto tmp57 = static_cast<bfloat>(tmp56);
        auto tmp58 = static_cast<bfloat>(tmp57);
        auto tmp59 = static_cast<bfloat>(tmp58);
        auto tmp60 = static_cast<bfloat>(tmp59);
        auto tmp61 = static_cast<bfloat>(tmp60);
        auto tmp62 = static_cast<bfloat>(tmp61);
        auto tmp63 = tmp55 * tmp62;
        auto tmp64 = static_cast<bfloat>(tmp63);
        auto tmp65 = static_cast<bfloat>(tmp64);
        auto tmp66 = static_cast<bfloat>(tmp65);
        auto tmp67 = static_cast<bfloat>(tmp66);
        auto tmp68 = tmp53 + tmp67;
        auto tmp69 = static_cast<bfloat>(tmp68);
        auto tmp70 = static_cast<bfloat>(tmp69);
        out_ptr3[r0_1 + 128*x3 + 9088*x2] = static_cast<bfloat>(tmp70);
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
        device bfloat* out_ptr2,
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
        auto tmp14 = static_cast<bfloat>(tmp13);
        auto tmp15 = static_cast<bfloat>(tmp14);
        out_ptr0[x2] = static_cast<bfloat>(tmp15);
        out_ptr1[x2] = static_cast<bfloat>(tmp15);
        out_ptr2[x2] = static_cast<bfloat>(tmp15);
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
            auto tmp4 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp3 = static_cast<bfloat>(tmp2);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = static_cast<bfloat>(tmp5);
            auto tmp7 = tmp3 + tmp6;
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp11 = static_cast<bfloat>(tmp10);
            auto tmp12 = static_cast<float>(tmp11);
            auto tmp13 = tmp12 * tmp12;
            tmp_acc_1 += tmp13;
        }
        auto tmp14 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp15 = static_cast<float>(in_ptr3[r0_1]);
            auto tmp18 = in_ptr0[x0];
            auto tmp22 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp17 = static_cast<bfloat>(tmp16);
            if ((tmp18 < 0 || tmp18 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp18, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp19 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp18]);
            auto tmp20 = static_cast<bfloat>(tmp19);
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp23 = static_cast<bfloat>(tmp22);
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp25 = tmp21 + tmp24;
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp27 = static_cast<bfloat>(tmp26);
            auto tmp28 = static_cast<bfloat>(tmp27);
            auto tmp29 = static_cast<bfloat>(tmp28);
            auto tmp30 = static_cast<float>(tmp29);
            auto tmp31 = 2048.0;
            auto tmp32 = tmp14 / tmp31;
            auto tmp33 = 1e-06;
            auto tmp34 = tmp32 + tmp33;
            auto tmp35 = metal::precise::rsqrt(tmp34);
            auto tmp36 = tmp30 * tmp35;
            auto tmp37 = static_cast<bfloat>(tmp36);
            auto tmp38 = static_cast<bfloat>(tmp37);
            auto tmp39 = static_cast<bfloat>(tmp38);
            auto tmp40 = static_cast<bfloat>(tmp39);
            auto tmp41 = static_cast<bfloat>(tmp40);
            auto tmp42 = static_cast<bfloat>(tmp41);
            auto tmp43 = tmp17 * tmp42;
            auto tmp44 = static_cast<bfloat>(tmp43);
            auto tmp45 = static_cast<bfloat>(tmp44);
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp45);
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
        auto tmp13 = static_cast<float>(in_ptr1[x0]);
        auto tmp1 = static_cast<float>(tmp0);
        auto tmp2 = static_cast<decltype(tmp1)>(-tmp1);
        auto tmp3 = metal::precise::exp(tmp2);
        auto tmp4 = 1.0;
        auto tmp5 = tmp3 + tmp4;
        auto tmp6 = tmp1 / tmp5;
        auto tmp7 = static_cast<bfloat>(tmp6);
        auto tmp8 = static_cast<bfloat>(tmp7);
        auto tmp9 = static_cast<bfloat>(tmp8);
        auto tmp10 = static_cast<bfloat>(tmp9);
        auto tmp11 = static_cast<bfloat>(tmp10);
        auto tmp12 = static_cast<bfloat>(tmp11);
        auto tmp14 = static_cast<bfloat>(tmp13);
        auto tmp15 = static_cast<bfloat>(tmp14);
        auto tmp16 = tmp12 * tmp15;
        auto tmp17 = static_cast<bfloat>(tmp16);
        auto tmp18 = static_cast<bfloat>(tmp17);
        out_ptr0[x0] = static_cast<bfloat>(tmp18);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_10 = async_compile.metal('generated_kernel_10', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_10(
        device bfloat* out_ptr2,
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
            auto tmp4 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp12 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp3 = static_cast<bfloat>(tmp2);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = static_cast<bfloat>(tmp5);
            auto tmp7 = tmp3 + tmp6;
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp11 = static_cast<bfloat>(tmp10);
            auto tmp13 = static_cast<bfloat>(tmp12);
            auto tmp14 = static_cast<bfloat>(tmp13);
            auto tmp15 = tmp11 + tmp14;
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp17 = static_cast<bfloat>(tmp16);
            auto tmp18 = static_cast<bfloat>(tmp17);
            auto tmp19 = static_cast<bfloat>(tmp18);
            auto tmp20 = static_cast<float>(tmp19);
            auto tmp21 = tmp20 * tmp20;
            tmp_acc_1 += tmp21;
        }
        auto tmp22 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp23 = in_ptr0[x0];
            auto tmp27 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp35 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp54 = static_cast<float>(in_ptr4[r0_1]);
            if ((tmp23 < 0 || tmp23 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp23, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp24 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp23]);
            auto tmp25 = static_cast<bfloat>(tmp24);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp28 = static_cast<bfloat>(tmp27);
            auto tmp29 = static_cast<bfloat>(tmp28);
            auto tmp30 = tmp26 + tmp29;
            auto tmp31 = static_cast<bfloat>(tmp30);
            auto tmp32 = static_cast<bfloat>(tmp31);
            auto tmp33 = static_cast<bfloat>(tmp32);
            auto tmp34 = static_cast<bfloat>(tmp33);
            auto tmp36 = static_cast<bfloat>(tmp35);
            auto tmp37 = static_cast<bfloat>(tmp36);
            auto tmp38 = tmp34 + tmp37;
            auto tmp39 = static_cast<bfloat>(tmp38);
            auto tmp40 = static_cast<bfloat>(tmp39);
            auto tmp41 = static_cast<bfloat>(tmp40);
            auto tmp42 = static_cast<bfloat>(tmp41);
            auto tmp43 = static_cast<float>(tmp42);
            auto tmp44 = 2048.0;
            auto tmp45 = tmp22 / tmp44;
            auto tmp46 = 1e-06;
            auto tmp47 = tmp45 + tmp46;
            auto tmp48 = metal::precise::rsqrt(tmp47);
            auto tmp49 = tmp43 * tmp48;
            auto tmp50 = static_cast<bfloat>(tmp49);
            auto tmp51 = static_cast<bfloat>(tmp50);
            auto tmp52 = static_cast<bfloat>(tmp51);
            auto tmp53 = static_cast<bfloat>(tmp52);
            auto tmp55 = static_cast<bfloat>(tmp54);
            auto tmp56 = static_cast<bfloat>(tmp55);
            auto tmp57 = static_cast<bfloat>(tmp53);
            auto tmp58 = static_cast<bfloat>(tmp57);
            auto tmp59 = tmp56 * tmp58;
            auto tmp60 = static_cast<bfloat>(tmp59);
            auto tmp61 = static_cast<bfloat>(tmp60);
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp61);
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
            auto tmp4 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp12 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp20 = static_cast<float>(in_ptr4[r0_1 + 2048*x0]);
            if ((tmp0 < 0 || tmp0 >= 151936)) {
                TORCH_REPORT_ERROR(error_buf, "Index ", tmp0, " out of range [0, ", 151936, ")");
                return;
            }
            auto tmp1 = static_cast<float>(in_ptr1[r0_1 + 2048*tmp0]);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp3 = static_cast<bfloat>(tmp2);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = static_cast<bfloat>(tmp5);
            auto tmp7 = tmp3 + tmp6;
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp11 = static_cast<bfloat>(tmp10);
            auto tmp13 = static_cast<bfloat>(tmp12);
            auto tmp14 = static_cast<bfloat>(tmp13);
            auto tmp15 = tmp11 + tmp14;
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp17 = static_cast<bfloat>(tmp16);
            auto tmp18 = static_cast<bfloat>(tmp17);
            auto tmp19 = static_cast<bfloat>(tmp18);
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = static_cast<bfloat>(tmp21);
            auto tmp23 = tmp19 + tmp22;
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp25 = static_cast<bfloat>(tmp24);
            out_ptr0[r0_1 + 2048*x0] = static_cast<bfloat>(tmp25);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp27 = static_cast<bfloat>(tmp26);
            auto tmp28 = static_cast<float>(tmp27);
            auto tmp29 = tmp28 * tmp28;
            tmp_acc_1 += tmp29;
        }
        auto tmp30 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp31 = static_cast<float>(in_ptr5[r0_1]);
            auto tmp34 = static_cast<float>(out_ptr0[r0_1 + 2048*x0]);
            auto tmp32 = static_cast<bfloat>(tmp31);
            auto tmp33 = static_cast<bfloat>(tmp32);
            auto tmp35 = static_cast<bfloat>(tmp34);
            auto tmp36 = static_cast<bfloat>(tmp35);
            auto tmp37 = static_cast<float>(tmp36);
            auto tmp38 = 2048.0;
            auto tmp39 = tmp30 / tmp38;
            auto tmp40 = 1e-06;
            auto tmp41 = tmp39 + tmp40;
            auto tmp42 = metal::precise::rsqrt(tmp41);
            auto tmp43 = tmp37 * tmp42;
            auto tmp44 = static_cast<bfloat>(tmp43);
            auto tmp45 = static_cast<bfloat>(tmp44);
            auto tmp46 = static_cast<bfloat>(tmp45);
            auto tmp47 = static_cast<bfloat>(tmp46);
            auto tmp48 = static_cast<bfloat>(tmp47);
            auto tmp49 = static_cast<bfloat>(tmp48);
            auto tmp50 = tmp33 * tmp49;
            auto tmp51 = static_cast<bfloat>(tmp50);
            auto tmp52 = static_cast<bfloat>(tmp51);
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp52);
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
            auto tmp3 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<bfloat>(tmp0);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp4 = static_cast<bfloat>(tmp3);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = tmp2 + tmp5;
            auto tmp7 = static_cast<bfloat>(tmp6);
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp11 = static_cast<float>(tmp10);
            auto tmp12 = tmp11 * tmp11;
            tmp_acc_1 += tmp12;
        }
        auto tmp13 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp14 = static_cast<float>(in_ptr2[r0_1]);
            auto tmp17 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp20 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp15 = static_cast<bfloat>(tmp14);
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp18 = static_cast<bfloat>(tmp17);
            auto tmp19 = static_cast<bfloat>(tmp18);
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = static_cast<bfloat>(tmp21);
            auto tmp23 = tmp19 + tmp22;
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp25 = static_cast<bfloat>(tmp24);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp27 = static_cast<bfloat>(tmp26);
            auto tmp28 = static_cast<float>(tmp27);
            auto tmp29 = 2048.0;
            auto tmp30 = tmp13 / tmp29;
            auto tmp31 = 1e-06;
            auto tmp32 = tmp30 + tmp31;
            auto tmp33 = metal::precise::rsqrt(tmp32);
            auto tmp34 = tmp28 * tmp33;
            auto tmp35 = static_cast<bfloat>(tmp34);
            auto tmp36 = static_cast<bfloat>(tmp35);
            auto tmp37 = static_cast<bfloat>(tmp36);
            auto tmp38 = static_cast<bfloat>(tmp37);
            auto tmp39 = static_cast<bfloat>(tmp38);
            auto tmp40 = static_cast<bfloat>(tmp39);
            auto tmp41 = tmp16 * tmp40;
            auto tmp42 = static_cast<bfloat>(tmp41);
            auto tmp43 = static_cast<bfloat>(tmp42);
            out_ptr1[r0_1 + 2048*x0] = static_cast<bfloat>(tmp43);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_13 = async_compile.metal('generated_kernel_13', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_13(
        device bfloat* out_ptr2,
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
            auto tmp3 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp11 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<bfloat>(tmp0);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp4 = static_cast<bfloat>(tmp3);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = tmp2 + tmp5;
            auto tmp7 = static_cast<bfloat>(tmp6);
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp12 = static_cast<bfloat>(tmp11);
            auto tmp13 = static_cast<bfloat>(tmp12);
            auto tmp14 = tmp10 + tmp13;
            auto tmp15 = static_cast<bfloat>(tmp14);
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp17 = static_cast<bfloat>(tmp16);
            auto tmp18 = static_cast<bfloat>(tmp17);
            auto tmp19 = static_cast<float>(tmp18);
            auto tmp20 = tmp19 * tmp19;
            tmp_acc_1 += tmp20;
        }
        auto tmp21 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp22 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp25 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp33 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp52 = static_cast<float>(in_ptr3[r0_1]);
            auto tmp23 = static_cast<bfloat>(tmp22);
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp27 = static_cast<bfloat>(tmp26);
            auto tmp28 = tmp24 + tmp27;
            auto tmp29 = static_cast<bfloat>(tmp28);
            auto tmp30 = static_cast<bfloat>(tmp29);
            auto tmp31 = static_cast<bfloat>(tmp30);
            auto tmp32 = static_cast<bfloat>(tmp31);
            auto tmp34 = static_cast<bfloat>(tmp33);
            auto tmp35 = static_cast<bfloat>(tmp34);
            auto tmp36 = tmp32 + tmp35;
            auto tmp37 = static_cast<bfloat>(tmp36);
            auto tmp38 = static_cast<bfloat>(tmp37);
            auto tmp39 = static_cast<bfloat>(tmp38);
            auto tmp40 = static_cast<bfloat>(tmp39);
            auto tmp41 = static_cast<float>(tmp40);
            auto tmp42 = 2048.0;
            auto tmp43 = tmp21 / tmp42;
            auto tmp44 = 1e-06;
            auto tmp45 = tmp43 + tmp44;
            auto tmp46 = metal::precise::rsqrt(tmp45);
            auto tmp47 = tmp41 * tmp46;
            auto tmp48 = static_cast<bfloat>(tmp47);
            auto tmp49 = static_cast<bfloat>(tmp48);
            auto tmp50 = static_cast<bfloat>(tmp49);
            auto tmp51 = static_cast<bfloat>(tmp50);
            auto tmp53 = static_cast<bfloat>(tmp52);
            auto tmp54 = static_cast<bfloat>(tmp53);
            auto tmp55 = static_cast<bfloat>(tmp51);
            auto tmp56 = static_cast<bfloat>(tmp55);
            auto tmp57 = tmp54 * tmp56;
            auto tmp58 = static_cast<bfloat>(tmp57);
            auto tmp59 = static_cast<bfloat>(tmp58);
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp59);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_14 = async_compile.metal('generated_kernel_14', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_14(
        device bfloat* out_ptr2,
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
            auto tmp3 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp11 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp19 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<bfloat>(tmp0);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp4 = static_cast<bfloat>(tmp3);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = tmp2 + tmp5;
            auto tmp7 = static_cast<bfloat>(tmp6);
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp12 = static_cast<bfloat>(tmp11);
            auto tmp13 = static_cast<bfloat>(tmp12);
            auto tmp14 = tmp10 + tmp13;
            auto tmp15 = static_cast<bfloat>(tmp14);
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp17 = static_cast<bfloat>(tmp16);
            auto tmp18 = static_cast<bfloat>(tmp17);
            auto tmp20 = static_cast<bfloat>(tmp19);
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = tmp18 + tmp21;
            auto tmp23 = static_cast<bfloat>(tmp22);
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp25 = static_cast<bfloat>(tmp24);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp27 = static_cast<float>(tmp26);
            auto tmp28 = tmp27 * tmp27;
            tmp_acc_1 += tmp28;
        }
        auto tmp29 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp30 = static_cast<float>(in_ptr0[r0_1 + 2048*x0]);
            auto tmp33 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp41 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp49 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp64 = static_cast<float>(in_ptr4[r0_1]);
            auto tmp31 = static_cast<bfloat>(tmp30);
            auto tmp32 = static_cast<bfloat>(tmp31);
            auto tmp34 = static_cast<bfloat>(tmp33);
            auto tmp35 = static_cast<bfloat>(tmp34);
            auto tmp36 = tmp32 + tmp35;
            auto tmp37 = static_cast<bfloat>(tmp36);
            auto tmp38 = static_cast<bfloat>(tmp37);
            auto tmp39 = static_cast<bfloat>(tmp38);
            auto tmp40 = static_cast<bfloat>(tmp39);
            auto tmp42 = static_cast<bfloat>(tmp41);
            auto tmp43 = static_cast<bfloat>(tmp42);
            auto tmp44 = tmp40 + tmp43;
            auto tmp45 = static_cast<bfloat>(tmp44);
            auto tmp46 = static_cast<bfloat>(tmp45);
            auto tmp47 = static_cast<bfloat>(tmp46);
            auto tmp48 = static_cast<bfloat>(tmp47);
            auto tmp50 = static_cast<bfloat>(tmp49);
            auto tmp51 = static_cast<bfloat>(tmp50);
            auto tmp52 = tmp48 + tmp51;
            auto tmp53 = static_cast<bfloat>(tmp52);
            auto tmp54 = static_cast<bfloat>(tmp53);
            auto tmp55 = static_cast<bfloat>(tmp54);
            auto tmp56 = static_cast<bfloat>(tmp55);
            auto tmp57 = static_cast<float>(tmp56);
            auto tmp58 = 2048.0;
            auto tmp59 = tmp29 / tmp58;
            auto tmp60 = 1e-06;
            auto tmp61 = tmp59 + tmp60;
            auto tmp62 = metal::precise::rsqrt(tmp61);
            auto tmp63 = tmp57 * tmp62;
            auto tmp65 = static_cast<bfloat>(tmp64);
            auto tmp66 = static_cast<bfloat>(tmp65);
            auto tmp67 = static_cast<bfloat>(tmp63);
            auto tmp68 = static_cast<bfloat>(tmp67);
            auto tmp69 = static_cast<bfloat>(tmp68);
            auto tmp70 = static_cast<bfloat>(tmp69);
            auto tmp71 = static_cast<bfloat>(tmp70);
            auto tmp72 = static_cast<bfloat>(tmp71);
            auto tmp73 = tmp66 * tmp72;
            auto tmp74 = static_cast<bfloat>(tmp73);
            auto tmp75 = static_cast<bfloat>(tmp74);
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp75);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_15 = async_compile.metal('generated_kernel_15', '''

    kernel void generated_kernel_15(
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
        auto tmp14 = static_cast<bfloat>(tmp13);
        auto tmp15 = static_cast<bfloat>(tmp14);
        out_ptr0[x2] = static_cast<bfloat>(tmp15);
        out_ptr1[x2] = static_cast<bfloat>(tmp15);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_16 = async_compile.metal('generated_kernel_16', '''

    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel_16(
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
            auto tmp3 = static_cast<float>(in_ptr1[r0_1 + 2048*x0]);
            auto tmp11 = static_cast<float>(in_ptr2[r0_1 + 2048*x0]);
            auto tmp19 = static_cast<float>(in_ptr3[r0_1 + 2048*x0]);
            auto tmp27 = static_cast<float>(in_ptr4[r0_1 + 2048*x0]);
            auto tmp1 = static_cast<bfloat>(tmp0);
            auto tmp2 = static_cast<bfloat>(tmp1);
            auto tmp4 = static_cast<bfloat>(tmp3);
            auto tmp5 = static_cast<bfloat>(tmp4);
            auto tmp6 = tmp2 + tmp5;
            auto tmp7 = static_cast<bfloat>(tmp6);
            auto tmp8 = static_cast<bfloat>(tmp7);
            auto tmp9 = static_cast<bfloat>(tmp8);
            auto tmp10 = static_cast<bfloat>(tmp9);
            auto tmp12 = static_cast<bfloat>(tmp11);
            auto tmp13 = static_cast<bfloat>(tmp12);
            auto tmp14 = tmp10 + tmp13;
            auto tmp15 = static_cast<bfloat>(tmp14);
            auto tmp16 = static_cast<bfloat>(tmp15);
            auto tmp17 = static_cast<bfloat>(tmp16);
            auto tmp18 = static_cast<bfloat>(tmp17);
            auto tmp20 = static_cast<bfloat>(tmp19);
            auto tmp21 = static_cast<bfloat>(tmp20);
            auto tmp22 = tmp18 + tmp21;
            auto tmp23 = static_cast<bfloat>(tmp22);
            auto tmp24 = static_cast<bfloat>(tmp23);
            auto tmp25 = static_cast<bfloat>(tmp24);
            auto tmp26 = static_cast<bfloat>(tmp25);
            auto tmp28 = static_cast<bfloat>(tmp27);
            auto tmp29 = static_cast<bfloat>(tmp28);
            auto tmp30 = tmp26 + tmp29;
            auto tmp31 = static_cast<bfloat>(tmp30);
            auto tmp32 = static_cast<bfloat>(tmp31);
            out_ptr0[r0_1 + 2048*x0] = static_cast<bfloat>(tmp32);
            auto tmp33 = static_cast<bfloat>(tmp32);
            auto tmp34 = static_cast<bfloat>(tmp33);
            auto tmp35 = static_cast<float>(tmp34);
            auto tmp36 = tmp35 * tmp35;
            tmp_acc_1 += tmp36;
        }
        auto tmp37 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0__cnt = 0; r0__cnt < 2; ++r0__cnt) {
            int r0__linear_idx = 2 * r0_index + r0__cnt;
            int r0_1 = r0__linear_idx;
            auto tmp38 = static_cast<float>(in_ptr5[r0_1]);
            auto tmp41 = static_cast<float>(out_ptr0[r0_1 + 2048*x0]);
            auto tmp39 = static_cast<bfloat>(tmp38);
            auto tmp40 = static_cast<bfloat>(tmp39);
            auto tmp42 = static_cast<bfloat>(tmp41);
            auto tmp43 = static_cast<bfloat>(tmp42);
            auto tmp44 = static_cast<float>(tmp43);
            auto tmp45 = 2048.0;
            auto tmp46 = tmp37 / tmp45;
            auto tmp47 = 1e-06;
            auto tmp48 = tmp46 + tmp47;
            auto tmp49 = metal::precise::rsqrt(tmp48);
            auto tmp50 = tmp44 * tmp49;
            auto tmp51 = static_cast<bfloat>(tmp50);
            auto tmp52 = static_cast<bfloat>(tmp51);
            auto tmp53 = static_cast<bfloat>(tmp52);
            auto tmp54 = static_cast<bfloat>(tmp53);
            auto tmp55 = static_cast<bfloat>(tmp54);
            auto tmp56 = static_cast<bfloat>(tmp55);
            auto tmp57 = tmp40 * tmp56;
            auto tmp58 = static_cast<bfloat>(tmp57);
            auto tmp59 = static_cast<bfloat>(tmp58);
            out_ptr2[r0_1 + 2048*x0] = static_cast<bfloat>(tmp59);
        }
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_17 = async_compile.metal('generated_kernel_17', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_17(
        device float* out_ptr0,
        constant bfloat* in_ptr0,
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
        auto tmp1 = static_cast<bfloat>(tmp0);
        auto tmp2 = static_cast<bfloat>(tmp1);
        auto tmp3 = static_cast<float>(tmp2);
        auto tmp4 = tmp3 * tmp3;
        auto tmp5 = c10::metal::threadgroup_sum(tmp_acc_0, tmp4, r0_index * 1, 128);
        if (r0_index == 0) out_ptr0[x0] = static_cast<float>(tmp5);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_18 = async_compile.metal('generated_kernel_18', '''

    kernel void generated_kernel_18(
        device bfloat* out_ptr0,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant float* in_ptr2,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (64);
        int x1 = c10::metal::floor_divide(xindex, 64);
        int x2 = xindex;
        auto tmp0 = static_cast<float>(in_ptr0[64 + x0]);
        auto tmp4 = static_cast<float>(in_ptr1[64 + x0 + 128*x1]);
        auto tmp8 = in_ptr2[x1];
        auto tmp1 = static_cast<bfloat>(tmp0);
        auto tmp2 = static_cast<bfloat>(tmp1);
        auto tmp3 = static_cast<float>(tmp2);
        auto tmp5 = static_cast<bfloat>(tmp4);
        auto tmp6 = static_cast<bfloat>(tmp5);
        auto tmp7 = static_cast<float>(tmp6);
        auto tmp9 = 128.0;
        auto tmp10 = tmp8 / tmp9;
        auto tmp11 = 1e-06;
        auto tmp12 = tmp10 + tmp11;
        auto tmp13 = metal::precise::rsqrt(tmp12);
        auto tmp14 = tmp7 * tmp13;
        auto tmp15 = static_cast<bfloat>(tmp14);
        auto tmp16 = static_cast<bfloat>(tmp15);
        auto tmp17 = static_cast<bfloat>(tmp16);
        auto tmp18 = static_cast<bfloat>(tmp17);
        auto tmp19 = static_cast<bfloat>(tmp18);
        auto tmp20 = static_cast<bfloat>(tmp19);
        auto tmp21 = static_cast<float>(tmp20);
        auto tmp22 = tmp3 * tmp21;
        auto tmp23 = static_cast<bfloat>(tmp22);
        auto tmp24 = static_cast<bfloat>(tmp23);
        auto tmp25 = static_cast<bfloat>(tmp24);
        auto tmp26 = static_cast<bfloat>(tmp25);
        auto tmp27 = static_cast<bfloat>(tmp26);
        auto tmp28 = static_cast<bfloat>(tmp27);
        auto tmp29 = static_cast<decltype(tmp28)>(-tmp28);
        auto tmp30 = static_cast<bfloat>(tmp29);
        auto tmp31 = static_cast<bfloat>(tmp30);
        out_ptr0[x2] = static_cast<bfloat>(tmp31);
    }

''', ['error', 'reduction_utils', 'utils'])
generated_kernel_19 = async_compile.metal('generated_kernel_19', '''

    kernel void generated_kernel_19(
        device bfloat* out_ptr2,
        constant bfloat* in_ptr0,
        constant bfloat* in_ptr1,
        constant bfloat* in_ptr2,
        constant float* in_ptr3,
        constant float* in_ptr4,
        device c10::metal::ErrorMessages* error_buf,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (128);
        int x1 = c10::metal::floor_divide(xindex, 128);
        int x4 = xindex;
        int x3 = c10::metal::floor_divide(xindex, 2048);
        int x2 = ((xindex) / (128)) % (16);
        auto tmp12 = static_cast<float>(in_ptr1[x0]);
        auto tmp16 = static_cast<float>(in_ptr2[x4]);
        auto tmp20 = in_ptr3[x1];
        auto tmp41 = in_ptr4[(x4) % (64)];
        auto tmp0 = x0;
        auto tmp1 = static_cast<long>(tmp0);
        auto tmp2 = 0;
        auto tmp3 = tmp1 >= tmp2;
        auto tmp4 = 64;
        auto tmp5 = tmp1 < tmp4;
        float tmp6;
        if (tmp5) {
            auto tmp_scoped0 = static_cast<float>(in_ptr0[64*x1 + (x0)]);
            tmp6 = static_cast<decltype(tmp6)>(tmp_scoped0);
        } else tmp6 = static_cast<decltype(tmp6)>(0.0);
        auto tmp7 = tmp1 >= tmp4;
        auto tmp8 = 128;
        auto tmp9 = tmp1 < tmp8;
        bfloat tmp10;
        if (tmp7) {
            auto tmp_scoped0 = static_cast<float>(in_ptr1[(-64) + x0]);
            auto tmp_scoped1 = static_cast<bfloat>(tmp_scoped0);
            auto tmp_scoped2 = static_cast<bfloat>(tmp_scoped1);
            auto tmp_scoped3 = static_cast<float>(tmp_scoped2);
            auto tmp_scoped4 = static_cast<float>(in_ptr2[128*x1 + ((-64) + x0)]);
            auto tmp_scoped5 = static_cast<bfloat>(tmp_scoped4);
            auto tmp_scoped6 = static_cast<bfloat>(tmp_scoped5);
            auto tmp_scoped7 = static_cast<float>(tmp_scoped6);
            auto tmp_scoped8 = in_ptr3[x1];
            auto tmp_scoped9 = 128.0;
            auto tmp_scoped10 = tmp_scoped8 / tmp_scoped9;
            auto tmp_scoped11 = 1e-06;
            auto tmp_scoped12 = tmp_scoped10 + tmp_scoped11;
            auto tmp_scoped13 = metal::precise::rsqrt(tmp_scoped12);
            auto tmp_scoped14 = tmp_scoped7 * tmp_scoped13;
            auto tmp_scoped15 = static_cast<bfloat>(tmp_scoped14);
            auto tmp_scoped16 = static_cast<bfloat>(tmp_scoped15);
            auto tmp_scoped17 = static_cast<bfloat>(tmp_scoped16);
            auto tmp_scoped18 = static_cast<bfloat>(tmp_scoped17);
            auto tmp_scoped19 = static_cast<bfloat>(tmp_scoped18);
            auto tmp_scoped20 = static_cast<bfloat>(tmp_scoped19);
            auto tmp_scoped21 = static_cast<float>(tmp_scoped20);
            auto tmp_scoped22 = tmp_scoped3 * tmp_scoped21;
            auto tmp_scoped23 = static_cast<bfloat>(tmp_scoped22);
            auto tmp_scoped24 = static_cast<bfloat>(tmp_scoped23);
            auto tmp_scoped25 = static_cast<bfloat>(tmp_scoped24);
            auto tmp_scoped26 = static_cast<bfloat>(tmp_scoped25);
            tmp10 = static_cast<decltype(tmp10)>(tmp_scoped26);
        } else tmp10 = static_cast<decltype(tmp10)>(0.0);
        auto tmp11 = tmp5 ? tmp6 : static_cast<decltype(tmp6)>(tmp10);
        auto tmp13 = static_cast<bfloat>(tmp12);
        auto tmp14 = static_cast<bfloat>(tmp13);
        auto tmp15 = static_cast<float>(tmp14);
        auto tmp17 = static_cast<bfloat>(tmp16);
        auto tmp18 = static_cast<bfloat>(tmp17);
        auto tmp19 = static_cast<float>(tmp18);
        auto tmp21 = 128.0;
        auto tmp22 = tmp20 / tmp21;
        auto tmp23 = 1e-06;
        auto tmp24 = tmp22 + tmp23;
        auto tmp25 = metal::precise::rsqrt(tmp24);
        auto tmp26 = tmp19 * tmp25;
        auto tmp27 = static_cast<bfloat>(tmp26);
        auto tmp28 = static_cast<bfloat>(tmp27);
        auto tmp29 = static_cast<bfloat>(tmp28);
        auto tmp30 = static_cast<bfloat>(tmp29);
        auto tmp31 = static_cast<bfloat>(tmp30);
        auto tmp32 = static_cast<bfloat>(tmp31);
        auto tmp33 = static_cast<float>(tmp32);
        auto tmp34 = tmp15 * tmp33;
        auto tmp35 = static_cast<bfloat>(tmp34);
        auto tmp36 = static_cast<bfloat>(tmp35);
        auto tmp37 = static_cast<bfloat>(tmp36);
        auto tmp38 = static_cast<bfloat>(tmp37);
        auto tmp39 = static_cast<bfloat>(tmp38);
        auto tmp40 = static_cast<bfloat>(tmp39);
        auto tmp42 = x3;
        auto tmp43 = static_cast<float>(tmp42);
        auto tmp44 = tmp43 * tmp41;
        auto tmp45 = metal::precise::cos(tmp44);
        auto tmp46 = static_cast<bfloat>(tmp45);
        auto tmp47 = static_cast<bfloat>(tmp46);
        auto tmp48 = static_cast<bfloat>(tmp47);
        auto tmp49 = static_cast<bfloat>(tmp48);
        auto tmp50 = static_cast<bfloat>(tmp49);
        auto tmp51 = static_cast<bfloat>(tmp50);
        auto tmp52 = tmp40 * tmp51;
        auto tmp53 = static_cast<bfloat>(tmp52);
        auto tmp54 = static_cast<bfloat>(tmp53);
        auto tmp55 = static_cast<bfloat>(tmp54);
        auto tmp56 = static_cast<bfloat>(tmp55);
        auto tmp57 = static_cast<bfloat>(tmp11);
        auto tmp58 = static_cast<bfloat>(tmp57);
        auto tmp59 = metal::precise::sin(tmp44);
        auto tmp60 = static_cast<bfloat>(tmp59);
        auto tmp61 = static_cast<bfloat>(tmp60);
        auto tmp62 = static_cast<bfloat>(tmp61);
        auto tmp63 = static_cast<bfloat>(tmp62);
        auto tmp64 = static_cast<bfloat>(tmp63);
        auto tmp65 = static_cast<bfloat>(tmp64);
        auto tmp66 = tmp58 * tmp65;
        auto tmp67 = static_cast<bfloat>(tmp66);
        auto tmp68 = static_cast<bfloat>(tmp67);
        auto tmp69 = static_cast<bfloat>(tmp68);
        auto tmp70 = static_cast<bfloat>(tmp69);
        auto tmp71 = tmp56 + tmp70;
        auto tmp72 = static_cast<bfloat>(tmp71);
        auto tmp73 = static_cast<bfloat>(tmp72);
        out_ptr2[x0 + 128*x3 + 9088*x2] = static_cast<bfloat>(tmp73);
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
        buf16 = empty_strided((1, 16, 71, 128), (145408, 9088, 128, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_1(buf16, buf2, arg5_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg5_1
        assert_size_stride(arg6_1, (1024, 2048), (2048, 1), 'input')
        buf6 = empty_strided((71, 1024), (1024, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf1, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg6_1, (2048, 1024), (1, 2048), 0), out=buf6)
        del arg6_1
        assert_size_stride(arg7_1, (128, ), (1, ), 'input')
        buf17 = empty_strided((1, 8, 71, 128), (72704, 9088, 128, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_2(buf17, buf6, arg7_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg7_1
        assert_size_stride(arg8_1, (1024, 2048), (2048, 1), 'input')
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf1, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg8_1, (2048, 1024), (1, 2048), 0), out=buf10)
        del arg8_1
        buf13 = empty_strided((1, 72), (72, 1), device='mps', dtype=torch.int64)
        buf11 = reinterpret_tensor(buf13, (1, 1), (72, 1), 0)  # alias
        generated_kernel_3(buf11, threads=[1], error_buf_idx=1)
        buf12 = reinterpret_tensor(buf13, (1, 71), (72, 1), 1)  # alias
        generated_kernel_4(buf12, threads=[71], error_buf_idx=1)
        buf14 = empty_strided((1, 71), (71, 1), device='mps', dtype=torch.int64)
        generated_kernel_5(buf14, buf13, threads=[71], error_buf_idx=2)
        del buf11
        del buf12
        del buf13
        # Topologically Sorted Source Nodes: [position_diff, ne, packed_sequence_mask], Original ATen: [aten.slice, aten.sub, aten.ne, aten.cumsum]
        buf15 = torch.ops.aten.cumsum.default(buf14, dim=-1, dtype=torch.int64)
        assert_tensor_metadata(buf15, (1, 71), (71, 1), torch.int64, 'torch.ops.aten.cumsum.default')
        assert_alignment(buf15, 16, 'torch.ops.aten.cumsum.default')
        del buf14
        buf18 = empty_strided((1, 1, 71, 71), (5056, 5056, 71, 1), device='mps', dtype=torch.bfloat16)
        buf44 = empty_strided((1, 1, 71, 71), (5056, 5056, 71, 1), device='mps', dtype=torch.bfloat16)
        buf70 = empty_strided((1, 1, 71, 71), (5056, 5056, 71, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_6(buf18, buf44, buf70, buf15, threads=[5041], error_buf_idx=4)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, sin_3, mul_10, q_embed, mul_12, k_embed, linear_2, view_2, value_states, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, attn_output], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.linear, aten.view, aten.transpose, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf19 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf16, buf17, reinterpret_tensor(buf10, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf18, scale=0.08838834764831845, enable_gqa=True)
        buf20 = buf19[0]
        assert_tensor_metadata(buf20, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf20, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf19
        buf22 = reinterpret_tensor(buf16, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf16  # reuse
        generated_kernel_7(buf22, buf20, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg9_1, (2048, 2048), (2048, 1), 'input')
        buf23 = reinterpret_tensor(buf20, (71, 2048), (2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, attn_output_1, reshape, attn_output_3], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf22, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg9_1, (2048, 2048), (1, 2048), 0), out=buf23)
        del arg9_1
        assert_size_stride(arg10_1, (2048, ), (1, ), 'input')
        buf25 = reinterpret_tensor(buf22, (1, 71, 2048), (145408, 2048, 1), 0); del buf22  # reuse
        generated_kernel_8(buf25, arg0_1, arg1_1, buf23, arg10_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg10_1
        assert_size_stride(arg11_1, (6144, 2048), (2048, 1), 'input')
        buf26 = empty_strided((71, 6144), (6144, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf25, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 6144), (1, 2048), 0), out=buf26)
        del arg11_1
        assert_size_stride(arg12_1, (6144, 2048), (2048, 1), 'input')
        buf27 = empty_strided((71, 6144), (6144, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf25, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg12_1, (2048, 6144), (1, 2048), 0), out=buf27)
        del arg12_1
        buf28 = empty_strided((1, 71, 6144), (436224, 6144, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_9(buf28, buf26, buf27, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg13_1, (2048, 6144), (6144, 1), 'input')
        buf29 = reinterpret_tensor(buf25, (71, 2048), (2048, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_15, down_proj], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf28, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg13_1, (6144, 2048), (1, 6144), 0), out=buf29)
        del arg13_1
        assert_size_stride(arg14_1, (2048, ), (1, ), 'input')
        buf32 = buf1; del buf1  # reuse
        generated_kernel_10(buf32, arg0_1, arg1_1, buf23, buf29, arg14_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg14_1
        assert_size_stride(arg15_1, (2048, 2048), (2048, 1), 'input')
        buf33 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14, linear_7], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf32, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg15_1, (2048, 2048), (1, 2048), 0), out=buf33)
        del arg15_1
        assert_size_stride(arg16_1, (128, ), (1, ), 'input')
        buf42 = empty_strided((1, 16, 71, 128), (145408, 9088, 128, 1), device='mps', dtype=torch.bfloat16)
        generated_kernel_1(buf42, buf33, arg16_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg16_1
        assert_size_stride(arg17_1, (1024, 2048), (2048, 1), 'input')
        buf37 = reinterpret_tensor(buf17, (71, 1024), (1024, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf32, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 1024), (1, 2048), 0), out=buf37)
        del arg17_1
        assert_size_stride(arg18_1, (128, ), (1, ), 'input')
        buf43 = reinterpret_tensor(buf10, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf10  # reuse
        generated_kernel_2(buf43, buf37, arg18_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg18_1
        assert_size_stride(arg19_1, (1024, 2048), (2048, 1), 'input')
        buf41 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf32, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg19_1, (2048, 1024), (1, 2048), 0), out=buf41)
        del arg19_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_4, mul_23, q_embed_1, mul_25, k_embed_1, linear_9, view_5, value_states_1, attn_output_4], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf45 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf42, buf43, reinterpret_tensor(buf41, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf44, scale=0.08838834764831845, enable_gqa=True)
        buf46 = buf45[0]
        assert_tensor_metadata(buf46, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf46, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf45
        buf48 = reinterpret_tensor(buf42, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf42  # reuse
        generated_kernel_7(buf48, buf46, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg20_1, (2048, 2048), (2048, 1), 'input')
        buf49 = reinterpret_tensor(buf46, (71, 2048), (2048, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [transpose_7, attn_output_5, reshape_1, attn_output_7], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf48, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg20_1, (2048, 2048), (1, 2048), 0), out=buf49)
        del arg20_1
        assert_size_stride(arg21_1, (2048, ), (1, ), 'input')
        buf50 = reinterpret_tensor(buf48, (1, 71, 2048), (145408, 2048, 1), 0); del buf48  # reuse
        buf52 = buf32; del buf32  # reuse
        generated_kernel_11(buf50, buf52, arg0_1, arg1_1, buf23, buf29, buf49, arg21_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg0_1
        del arg21_1
        assert_size_stride(arg22_1, (6144, 2048), (2048, 1), 'input')
        buf53 = reinterpret_tensor(buf28, (71, 6144), (6144, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, pow_8, variance_7, add_17, rsqrt_7, hidden_states_21, to_20, hidden_states_22, linear_11], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf52, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg22_1, (2048, 6144), (1, 2048), 0), out=buf53)
        del arg22_1
        assert_size_stride(arg23_1, (6144, 2048), (2048, 1), 'input')
        buf54 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf52, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg23_1, (2048, 6144), (1, 2048), 0), out=buf54)
        del arg23_1
        buf55 = reinterpret_tensor(buf26, (1, 71, 6144), (436224, 6144, 1), 0); del buf26  # reuse
        generated_kernel_9(buf55, buf53, buf54, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg24_1, (2048, 6144), (6144, 1), 'input')
        buf56 = reinterpret_tensor(buf52, (71, 2048), (2048, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_28, down_proj_1], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf55, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg24_1, (6144, 2048), (1, 6144), 0), out=buf56)
        del arg24_1
        assert_size_stride(arg25_1, (2048, ), (1, ), 'input')
        buf58 = reinterpret_tensor(buf49, (1, 71, 2048), (145408, 2048, 1), 0); del buf49  # reuse
        generated_kernel_12(buf58, buf50, buf56, arg25_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg25_1
        assert_size_stride(arg26_1, (2048, 2048), (2048, 1), 'input')
        buf59 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf58, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg26_1, (2048, 2048), (1, 2048), 0), out=buf59)
        del arg26_1
        assert_size_stride(arg27_1, (128, ), (1, ), 'input')
        buf68 = reinterpret_tensor(buf23, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf23  # reuse
        generated_kernel_1(buf68, buf59, arg27_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg27_1
        assert_size_stride(arg28_1, (1024, 2048), (2048, 1), 'input')
        buf63 = reinterpret_tensor(buf43, (71, 1024), (1024, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf58, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg28_1, (2048, 1024), (1, 2048), 0), out=buf63)
        del arg28_1
        assert_size_stride(arg29_1, (128, ), (1, ), 'input')
        buf69 = reinterpret_tensor(buf41, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf41  # reuse
        generated_kernel_2(buf69, buf63, arg29_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg29_1
        assert_size_stride(arg30_1, (1024, 2048), (2048, 1), 'input')
        buf67 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf58, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg30_1, (2048, 1024), (1, 2048), 0), out=buf67)
        del arg30_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_5, mul_36, q_embed_2, mul_38, k_embed_2, linear_16, view_8, value_states_2, attn_output_8], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf71 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf68, buf69, reinterpret_tensor(buf67, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf70, scale=0.08838834764831845, enable_gqa=True)
        buf72 = buf71[0]
        assert_tensor_metadata(buf72, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf72, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf71
        buf74 = reinterpret_tensor(buf68, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf68  # reuse
        generated_kernel_7(buf74, buf72, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg31_1, (2048, 2048), (2048, 1), 'input')
        buf75 = reinterpret_tensor(buf72, (71, 2048), (2048, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [transpose_11, attn_output_9, reshape_2, attn_output_11], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf74, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg31_1, (2048, 2048), (1, 2048), 0), out=buf75)
        del arg31_1
        assert_size_stride(arg32_1, (2048, ), (1, ), 'input')
        buf78 = reinterpret_tensor(buf74, (1, 71, 2048), (145408, 2048, 1), 0); del buf74  # reuse
        generated_kernel_13(buf78, buf50, buf56, buf75, arg32_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg32_1
        assert_size_stride(arg33_1, (6144, 2048), (2048, 1), 'input')
        buf79 = reinterpret_tensor(buf55, (71, 6144), (6144, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34, linear_18], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf78, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 6144), (1, 2048), 0), out=buf79)
        del arg33_1
        assert_size_stride(arg34_1, (6144, 2048), (2048, 1), 'input')
        buf80 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf78, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg34_1, (2048, 6144), (1, 2048), 0), out=buf80)
        del arg34_1
        buf81 = reinterpret_tensor(buf53, (1, 71, 6144), (436224, 6144, 1), 0); del buf53  # reuse
        generated_kernel_9(buf81, buf79, buf80, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg35_1, (2048, 6144), (6144, 1), 'input')
        buf82 = reinterpret_tensor(buf78, (71, 2048), (2048, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_41, down_proj_2], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf81, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg35_1, (6144, 2048), (1, 6144), 0), out=buf82)
        del arg35_1
        assert_size_stride(arg36_1, (2048, ), (1, ), 'input')
        buf85 = buf58; del buf58  # reuse
        generated_kernel_14(buf85, buf50, buf56, buf75, buf82, arg36_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg36_1
        assert_size_stride(arg37_1, (2048, 2048), (2048, 1), 'input')
        buf86 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [to_30, hidden_states_38, linear_21], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf85, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg37_1, (2048, 2048), (1, 2048), 0), out=buf86)
        del arg37_1
        assert_size_stride(arg38_1, (128, ), (1, ), 'input')
        buf95 = reinterpret_tensor(buf33, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf33  # reuse
        generated_kernel_1(buf95, buf86, arg38_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg38_1
        del buf86
        assert_size_stride(arg39_1, (1024, 2048), (2048, 1), 'input')
        buf90 = reinterpret_tensor(buf69, (71, 1024), (1024, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf85, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg39_1, (2048, 1024), (1, 2048), 0), out=buf90)
        del arg39_1
        assert_size_stride(arg40_1, (128, ), (1, ), 'input')
        buf96 = reinterpret_tensor(buf67, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf67  # reuse
        generated_kernel_2(buf96, buf90, arg40_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg40_1
        assert_size_stride(arg41_1, (1024, 2048), (2048, 1), 'input')
        buf94 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf85, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg41_1, (2048, 1024), (1, 2048), 0), out=buf94)
        del arg41_1
        buf97 = buf70; del buf70  # reuse
        buf123 = buf44; del buf44  # reuse
        generated_kernel_15(buf97, buf123, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_6, mul_49, q_embed_3, mul_51, k_embed_3, linear_23, view_11, value_states_3, attn_output_12], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf98 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf95, buf96, reinterpret_tensor(buf94, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf97, scale=0.08838834764831845, enable_gqa=True)
        buf99 = buf98[0]
        assert_tensor_metadata(buf99, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf99, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf98
        buf101 = reinterpret_tensor(buf95, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf95  # reuse
        generated_kernel_7(buf101, buf99, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg42_1, (2048, 2048), (2048, 1), 'input')
        buf102 = reinterpret_tensor(buf99, (71, 2048), (2048, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [transpose_15, attn_output_13, reshape_3, attn_output_15], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf101, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg42_1, (2048, 2048), (1, 2048), 0), out=buf102)
        del arg42_1
        assert_size_stride(arg43_1, (2048, ), (1, ), 'input')
        buf103 = reinterpret_tensor(buf101, (1, 71, 2048), (145408, 2048, 1), 0); del buf101  # reuse
        buf105 = buf85; del buf85  # reuse
        generated_kernel_16(buf103, buf105, buf50, buf56, buf75, buf82, buf102, arg43_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg43_1
        del buf102
        assert_size_stride(arg44_1, (6144, 2048), (2048, 1), 'input')
        buf106 = reinterpret_tensor(buf81, (71, 6144), (6144, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44, pow_16, variance_15, add_33, rsqrt_15, hidden_states_45, to_36, hidden_states_46, linear_25], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf105, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 6144), (1, 2048), 0), out=buf106)
        del arg44_1
        assert_size_stride(arg45_1, (6144, 2048), (2048, 1), 'input')
        buf107 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf105, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 6144), (1, 2048), 0), out=buf107)
        del arg45_1
        buf108 = reinterpret_tensor(buf79, (1, 71, 6144), (436224, 6144, 1), 0); del buf79  # reuse
        generated_kernel_9(buf108, buf106, buf107, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg46_1, (2048, 6144), (6144, 1), 'input')
        buf109 = reinterpret_tensor(buf105, (71, 2048), (2048, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_54, down_proj_3], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf108, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg46_1, (6144, 2048), (1, 6144), 0), out=buf109)
        del arg46_1
        assert_size_stride(arg47_1, (2048, ), (1, ), 'input')
        buf111 = reinterpret_tensor(buf82, (1, 71, 2048), (145408, 2048, 1), 0); del buf82  # reuse
        generated_kernel_12(buf111, buf103, buf109, arg47_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg47_1
        assert_size_stride(arg48_1, (2048, 2048), (2048, 1), 'input')
        buf112 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf111, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg48_1, (2048, 2048), (1, 2048), 0), out=buf112)
        del arg48_1
        assert_size_stride(arg49_1, (128, ), (1, ), 'input')
        buf121 = reinterpret_tensor(buf56, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf56  # reuse
        generated_kernel_1(buf121, buf112, arg49_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg49_1
        assert_size_stride(arg50_1, (1024, 2048), (2048, 1), 'input')
        buf116 = reinterpret_tensor(buf96, (71, 1024), (1024, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf111, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg50_1, (2048, 1024), (1, 2048), 0), out=buf116)
        del arg50_1
        assert_size_stride(arg51_1, (128, ), (1, ), 'input')
        buf122 = reinterpret_tensor(buf94, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf94  # reuse
        generated_kernel_2(buf122, buf116, arg51_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg51_1
        assert_size_stride(arg52_1, (1024, 2048), (2048, 1), 'input')
        buf120 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf111, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg52_1, (2048, 1024), (1, 2048), 0), out=buf120)
        del arg52_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_7, mul_62, q_embed_4, mul_64, k_embed_4, linear_30, view_14, value_states_4, attn_output_16], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf124 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf121, buf122, reinterpret_tensor(buf120, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf123, scale=0.08838834764831845, enable_gqa=True)
        buf125 = buf124[0]
        assert_tensor_metadata(buf125, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf125, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf124
        buf127 = reinterpret_tensor(buf121, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf121  # reuse
        generated_kernel_7(buf127, buf125, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg53_1, (2048, 2048), (2048, 1), 'input')
        buf128 = reinterpret_tensor(buf125, (71, 2048), (2048, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [transpose_19, attn_output_17, reshape_4, attn_output_19], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf127, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg53_1, (2048, 2048), (1, 2048), 0), out=buf128)
        del arg53_1
        assert_size_stride(arg54_1, (2048, ), (1, ), 'input')
        buf131 = reinterpret_tensor(buf127, (1, 71, 2048), (145408, 2048, 1), 0); del buf127  # reuse
        generated_kernel_13(buf131, buf103, buf109, buf128, arg54_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg54_1
        assert_size_stride(arg55_1, (6144, 2048), (2048, 1), 'input')
        buf132 = reinterpret_tensor(buf108, (71, 6144), (6144, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, linear_32], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf131, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg55_1, (2048, 6144), (1, 2048), 0), out=buf132)
        del arg55_1
        assert_size_stride(arg56_1, (6144, 2048), (2048, 1), 'input')
        buf133 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf131, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg56_1, (2048, 6144), (1, 2048), 0), out=buf133)
        del arg56_1
        buf134 = reinterpret_tensor(buf106, (1, 71, 6144), (436224, 6144, 1), 0); del buf106  # reuse
        generated_kernel_9(buf134, buf132, buf133, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg57_1, (2048, 6144), (6144, 1), 'input')
        buf135 = reinterpret_tensor(buf131, (71, 2048), (2048, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_67, down_proj_4], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf134, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg57_1, (6144, 2048), (1, 6144), 0), out=buf135)
        del arg57_1
        assert_size_stride(arg58_1, (2048, ), (1, ), 'input')
        buf138 = buf111; del buf111  # reuse
        generated_kernel_14(buf138, buf103, buf109, buf128, buf135, arg58_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg58_1
        assert_size_stride(arg59_1, (2048, 2048), (2048, 1), 'input')
        buf139 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [to_46, hidden_states_62, linear_35], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf138, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg59_1, (2048, 2048), (1, 2048), 0), out=buf139)
        del arg59_1
        assert_size_stride(arg60_1, (128, ), (1, ), 'input')
        buf148 = reinterpret_tensor(buf50, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf50  # reuse
        generated_kernel_1(buf148, buf139, arg60_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg60_1
        del buf139
        assert_size_stride(arg61_1, (1024, 2048), (2048, 1), 'input')
        buf143 = reinterpret_tensor(buf122, (71, 1024), (1024, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf138, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg61_1, (2048, 1024), (1, 2048), 0), out=buf143)
        del arg61_1
        assert_size_stride(arg62_1, (128, ), (1, ), 'input')
        buf149 = reinterpret_tensor(buf120, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf120  # reuse
        generated_kernel_2(buf149, buf143, arg62_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg62_1
        assert_size_stride(arg63_1, (1024, 2048), (2048, 1), 'input')
        buf147 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf138, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 1024), (1, 2048), 0), out=buf147)
        del arg63_1
        buf150 = buf123; del buf123  # reuse
        buf176 = buf97; del buf97  # reuse
        generated_kernel_15(buf150, buf176, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_8, mul_75, q_embed_5, mul_77, k_embed_5, linear_37, view_17, value_states_5, attn_output_20], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf151 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf148, buf149, reinterpret_tensor(buf147, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf150, scale=0.08838834764831845, enable_gqa=True)
        buf152 = buf151[0]
        assert_tensor_metadata(buf152, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf152, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf151
        buf154 = reinterpret_tensor(buf148, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf148  # reuse
        generated_kernel_7(buf154, buf152, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg64_1, (2048, 2048), (2048, 1), 'input')
        buf155 = reinterpret_tensor(buf152, (71, 2048), (2048, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [transpose_23, attn_output_21, reshape_5, attn_output_23], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf154, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg64_1, (2048, 2048), (1, 2048), 0), out=buf155)
        del arg64_1
        assert_size_stride(arg65_1, (2048, ), (1, ), 'input')
        buf156 = reinterpret_tensor(buf154, (1, 71, 2048), (145408, 2048, 1), 0); del buf154  # reuse
        buf158 = buf138; del buf138  # reuse
        generated_kernel_16(buf156, buf158, buf103, buf109, buf128, buf135, buf155, arg65_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg65_1
        del buf103
        assert_size_stride(arg66_1, (6144, 2048), (2048, 1), 'input')
        buf159 = reinterpret_tensor(buf134, (71, 6144), (6144, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68, pow_24, variance_23, add_49, rsqrt_23, hidden_states_69, to_52, hidden_states_70, linear_39], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf158, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg66_1, (2048, 6144), (1, 2048), 0), out=buf159)
        del arg66_1
        assert_size_stride(arg67_1, (6144, 2048), (2048, 1), 'input')
        buf160 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf158, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg67_1, (2048, 6144), (1, 2048), 0), out=buf160)
        del arg67_1
        buf161 = reinterpret_tensor(buf132, (1, 71, 6144), (436224, 6144, 1), 0); del buf132  # reuse
        generated_kernel_9(buf161, buf159, buf160, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg68_1, (2048, 6144), (6144, 1), 'input')
        buf162 = reinterpret_tensor(buf158, (71, 2048), (2048, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_80, down_proj_5], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf161, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg68_1, (6144, 2048), (1, 6144), 0), out=buf162)
        del arg68_1
        assert_size_stride(arg69_1, (2048, ), (1, ), 'input')
        buf164 = reinterpret_tensor(buf155, (1, 71, 2048), (145408, 2048, 1), 0); del buf155  # reuse
        generated_kernel_12(buf164, buf156, buf162, arg69_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg69_1
        assert_size_stride(arg70_1, (2048, 2048), (2048, 1), 'input')
        buf165 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf164, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg70_1, (2048, 2048), (1, 2048), 0), out=buf165)
        del arg70_1
        assert_size_stride(arg71_1, (128, ), (1, ), 'input')
        buf174 = reinterpret_tensor(buf128, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf128  # reuse
        generated_kernel_1(buf174, buf165, arg71_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg71_1
        assert_size_stride(arg72_1, (1024, 2048), (2048, 1), 'input')
        buf169 = reinterpret_tensor(buf149, (71, 1024), (1024, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf164, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg72_1, (2048, 1024), (1, 2048), 0), out=buf169)
        del arg72_1
        assert_size_stride(arg73_1, (128, ), (1, ), 'input')
        buf175 = reinterpret_tensor(buf147, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf147  # reuse
        generated_kernel_2(buf175, buf169, arg73_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg73_1
        assert_size_stride(arg74_1, (1024, 2048), (2048, 1), 'input')
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf164, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg74_1, (2048, 1024), (1, 2048), 0), out=buf173)
        del arg74_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_9, mul_88, q_embed_6, mul_90, k_embed_6, linear_44, view_20, value_states_6, attn_output_24], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf177 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf174, buf175, reinterpret_tensor(buf173, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf176, scale=0.08838834764831845, enable_gqa=True)
        buf178 = buf177[0]
        assert_tensor_metadata(buf178, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf178, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf177
        buf180 = reinterpret_tensor(buf174, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf174  # reuse
        generated_kernel_7(buf180, buf178, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg75_1, (2048, 2048), (2048, 1), 'input')
        buf181 = reinterpret_tensor(buf178, (71, 2048), (2048, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [transpose_27, attn_output_25, reshape_6, attn_output_27], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf180, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg75_1, (2048, 2048), (1, 2048), 0), out=buf181)
        del arg75_1
        assert_size_stride(arg76_1, (2048, ), (1, ), 'input')
        buf184 = reinterpret_tensor(buf180, (1, 71, 2048), (145408, 2048, 1), 0); del buf180  # reuse
        generated_kernel_13(buf184, buf156, buf162, buf181, arg76_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg76_1
        assert_size_stride(arg77_1, (6144, 2048), (2048, 1), 'input')
        buf185 = reinterpret_tensor(buf161, (71, 6144), (6144, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, linear_46], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf184, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg77_1, (2048, 6144), (1, 2048), 0), out=buf185)
        del arg77_1
        assert_size_stride(arg78_1, (6144, 2048), (2048, 1), 'input')
        buf186 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf184, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg78_1, (2048, 6144), (1, 2048), 0), out=buf186)
        del arg78_1
        buf187 = reinterpret_tensor(buf159, (1, 71, 6144), (436224, 6144, 1), 0); del buf159  # reuse
        generated_kernel_9(buf187, buf185, buf186, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg79_1, (2048, 6144), (6144, 1), 'input')
        buf188 = reinterpret_tensor(buf184, (71, 2048), (2048, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_93, down_proj_6], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf187, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg79_1, (6144, 2048), (1, 6144), 0), out=buf188)
        del arg79_1
        assert_size_stride(arg80_1, (2048, ), (1, ), 'input')
        buf191 = buf164; del buf164  # reuse
        generated_kernel_14(buf191, buf156, buf162, buf181, buf188, arg80_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg80_1
        assert_size_stride(arg81_1, (2048, 2048), (2048, 1), 'input')
        buf192 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [to_62, hidden_states_86, linear_49], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf191, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg81_1, (2048, 2048), (1, 2048), 0), out=buf192)
        del arg81_1
        assert_size_stride(arg82_1, (128, ), (1, ), 'input')
        buf201 = reinterpret_tensor(buf109, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf109  # reuse
        generated_kernel_1(buf201, buf192, arg82_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg82_1
        del buf192
        assert_size_stride(arg83_1, (1024, 2048), (2048, 1), 'input')
        buf196 = reinterpret_tensor(buf175, (71, 1024), (1024, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf191, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg83_1, (2048, 1024), (1, 2048), 0), out=buf196)
        del arg83_1
        assert_size_stride(arg84_1, (128, ), (1, ), 'input')
        buf202 = reinterpret_tensor(buf173, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf173  # reuse
        generated_kernel_2(buf202, buf196, arg84_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg84_1
        assert_size_stride(arg85_1, (1024, 2048), (2048, 1), 'input')
        buf200 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf191, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg85_1, (2048, 1024), (1, 2048), 0), out=buf200)
        del arg85_1
        buf203 = buf176; del buf176  # reuse
        buf229 = buf150; del buf150  # reuse
        generated_kernel_15(buf203, buf229, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_10, mul_101, q_embed_7, mul_103, k_embed_7, linear_51, view_23, value_states_7, attn_output_28], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf204 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf201, buf202, reinterpret_tensor(buf200, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf203, scale=0.08838834764831845, enable_gqa=True)
        buf205 = buf204[0]
        assert_tensor_metadata(buf205, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf205, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf204
        buf207 = reinterpret_tensor(buf201, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf201  # reuse
        generated_kernel_7(buf207, buf205, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg86_1, (2048, 2048), (2048, 1), 'input')
        buf208 = reinterpret_tensor(buf205, (71, 2048), (2048, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [transpose_31, attn_output_29, reshape_7, attn_output_31], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf207, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg86_1, (2048, 2048), (1, 2048), 0), out=buf208)
        del arg86_1
        assert_size_stride(arg87_1, (2048, ), (1, ), 'input')
        buf209 = reinterpret_tensor(buf207, (1, 71, 2048), (145408, 2048, 1), 0); del buf207  # reuse
        buf211 = buf191; del buf191  # reuse
        generated_kernel_16(buf209, buf211, buf156, buf162, buf181, buf188, buf208, arg87_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg87_1
        del buf156
        assert_size_stride(arg88_1, (6144, 2048), (2048, 1), 'input')
        buf212 = reinterpret_tensor(buf187, (71, 6144), (6144, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92, pow_32, variance_31, add_65, rsqrt_31, hidden_states_93, to_68, hidden_states_94, linear_53], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf211, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg88_1, (2048, 6144), (1, 2048), 0), out=buf212)
        del arg88_1
        assert_size_stride(arg89_1, (6144, 2048), (2048, 1), 'input')
        buf213 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf211, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg89_1, (2048, 6144), (1, 2048), 0), out=buf213)
        del arg89_1
        buf214 = reinterpret_tensor(buf185, (1, 71, 6144), (436224, 6144, 1), 0); del buf185  # reuse
        generated_kernel_9(buf214, buf212, buf213, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg90_1, (2048, 6144), (6144, 1), 'input')
        buf215 = reinterpret_tensor(buf211, (71, 2048), (2048, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_106, down_proj_7], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf214, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg90_1, (6144, 2048), (1, 6144), 0), out=buf215)
        del arg90_1
        assert_size_stride(arg91_1, (2048, ), (1, ), 'input')
        buf217 = reinterpret_tensor(buf208, (1, 71, 2048), (145408, 2048, 1), 0); del buf208  # reuse
        generated_kernel_12(buf217, buf209, buf215, arg91_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg91_1
        assert_size_stride(arg92_1, (2048, 2048), (2048, 1), 'input')
        buf218 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf217, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg92_1, (2048, 2048), (1, 2048), 0), out=buf218)
        del arg92_1
        assert_size_stride(arg93_1, (128, ), (1, ), 'input')
        buf227 = reinterpret_tensor(buf181, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf181  # reuse
        generated_kernel_1(buf227, buf218, arg93_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg93_1
        assert_size_stride(arg94_1, (1024, 2048), (2048, 1), 'input')
        buf222 = reinterpret_tensor(buf202, (71, 1024), (1024, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf217, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg94_1, (2048, 1024), (1, 2048), 0), out=buf222)
        del arg94_1
        assert_size_stride(arg95_1, (128, ), (1, ), 'input')
        buf228 = reinterpret_tensor(buf200, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf200  # reuse
        generated_kernel_2(buf228, buf222, arg95_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg95_1
        assert_size_stride(arg96_1, (1024, 2048), (2048, 1), 'input')
        buf226 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf217, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 1024), (1, 2048), 0), out=buf226)
        del arg96_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_11, mul_114, q_embed_8, mul_116, k_embed_8, linear_58, view_26, value_states_8, attn_output_32], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf230 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf227, buf228, reinterpret_tensor(buf226, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf229, scale=0.08838834764831845, enable_gqa=True)
        buf231 = buf230[0]
        assert_tensor_metadata(buf231, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf231, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf230
        buf233 = reinterpret_tensor(buf227, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf227  # reuse
        generated_kernel_7(buf233, buf231, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg97_1, (2048, 2048), (2048, 1), 'input')
        buf234 = reinterpret_tensor(buf231, (71, 2048), (2048, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [transpose_35, attn_output_33, reshape_8, attn_output_35], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf233, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg97_1, (2048, 2048), (1, 2048), 0), out=buf234)
        del arg97_1
        assert_size_stride(arg98_1, (2048, ), (1, ), 'input')
        buf237 = reinterpret_tensor(buf233, (1, 71, 2048), (145408, 2048, 1), 0); del buf233  # reuse
        generated_kernel_13(buf237, buf209, buf215, buf234, arg98_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg98_1
        assert_size_stride(arg99_1, (6144, 2048), (2048, 1), 'input')
        buf238 = reinterpret_tensor(buf214, (71, 6144), (6144, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_106, linear_60], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf237, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg99_1, (2048, 6144), (1, 2048), 0), out=buf238)
        del arg99_1
        assert_size_stride(arg100_1, (6144, 2048), (2048, 1), 'input')
        buf239 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf237, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 6144), (1, 2048), 0), out=buf239)
        del arg100_1
        buf240 = reinterpret_tensor(buf212, (1, 71, 6144), (436224, 6144, 1), 0); del buf212  # reuse
        generated_kernel_9(buf240, buf238, buf239, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg101_1, (2048, 6144), (6144, 1), 'input')
        buf241 = reinterpret_tensor(buf237, (71, 2048), (2048, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_119, down_proj_8], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf240, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg101_1, (6144, 2048), (1, 6144), 0), out=buf241)
        del arg101_1
        assert_size_stride(arg102_1, (2048, ), (1, ), 'input')
        buf244 = buf217; del buf217  # reuse
        generated_kernel_14(buf244, buf209, buf215, buf234, buf241, arg102_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg102_1
        assert_size_stride(arg103_1, (2048, 2048), (2048, 1), 'input')
        buf245 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [to_78, hidden_states_110, linear_63], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf244, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg103_1, (2048, 2048), (1, 2048), 0), out=buf245)
        del arg103_1
        assert_size_stride(arg104_1, (128, ), (1, ), 'input')
        buf254 = reinterpret_tensor(buf162, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf162  # reuse
        generated_kernel_1(buf254, buf245, arg104_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg104_1
        del buf245
        assert_size_stride(arg105_1, (1024, 2048), (2048, 1), 'input')
        buf249 = reinterpret_tensor(buf228, (71, 1024), (1024, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf244, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg105_1, (2048, 1024), (1, 2048), 0), out=buf249)
        del arg105_1
        assert_size_stride(arg106_1, (128, ), (1, ), 'input')
        buf255 = reinterpret_tensor(buf226, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf226  # reuse
        generated_kernel_2(buf255, buf249, arg106_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg106_1
        assert_size_stride(arg107_1, (1024, 2048), (2048, 1), 'input')
        buf253 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf244, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg107_1, (2048, 1024), (1, 2048), 0), out=buf253)
        del arg107_1
        buf256 = buf229; del buf229  # reuse
        buf282 = buf203; del buf203  # reuse
        generated_kernel_15(buf256, buf282, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_12, mul_127, q_embed_9, mul_129, k_embed_9, linear_65, view_29, value_states_9, attn_output_36], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf257 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf254, buf255, reinterpret_tensor(buf253, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf256, scale=0.08838834764831845, enable_gqa=True)
        buf258 = buf257[0]
        assert_tensor_metadata(buf258, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf258, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf257
        buf260 = reinterpret_tensor(buf254, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf254  # reuse
        generated_kernel_7(buf260, buf258, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg108_1, (2048, 2048), (2048, 1), 'input')
        buf261 = reinterpret_tensor(buf258, (71, 2048), (2048, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [transpose_39, attn_output_37, reshape_9, attn_output_39], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf260, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg108_1, (2048, 2048), (1, 2048), 0), out=buf261)
        del arg108_1
        assert_size_stride(arg109_1, (2048, ), (1, ), 'input')
        buf262 = reinterpret_tensor(buf260, (1, 71, 2048), (145408, 2048, 1), 0); del buf260  # reuse
        buf264 = buf244; del buf244  # reuse
        generated_kernel_16(buf262, buf264, buf209, buf215, buf234, buf241, buf261, arg109_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg109_1
        del buf209
        assert_size_stride(arg110_1, (6144, 2048), (2048, 1), 'input')
        buf265 = reinterpret_tensor(buf240, (71, 6144), (6144, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_116, pow_40, variance_39, add_81, rsqrt_39, hidden_states_117, to_84, hidden_states_118, linear_67], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf264, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 6144), (1, 2048), 0), out=buf265)
        del arg110_1
        assert_size_stride(arg111_1, (6144, 2048), (2048, 1), 'input')
        buf266 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf264, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg111_1, (2048, 6144), (1, 2048), 0), out=buf266)
        del arg111_1
        buf267 = reinterpret_tensor(buf238, (1, 71, 6144), (436224, 6144, 1), 0); del buf238  # reuse
        generated_kernel_9(buf267, buf265, buf266, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg112_1, (2048, 6144), (6144, 1), 'input')
        buf268 = reinterpret_tensor(buf264, (71, 2048), (2048, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_132, down_proj_9], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf267, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg112_1, (6144, 2048), (1, 6144), 0), out=buf268)
        del arg112_1
        assert_size_stride(arg113_1, (2048, ), (1, ), 'input')
        buf270 = reinterpret_tensor(buf261, (1, 71, 2048), (145408, 2048, 1), 0); del buf261  # reuse
        generated_kernel_12(buf270, buf262, buf268, arg113_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg113_1
        assert_size_stride(arg114_1, (2048, 2048), (2048, 1), 'input')
        buf271 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf270, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg114_1, (2048, 2048), (1, 2048), 0), out=buf271)
        del arg114_1
        assert_size_stride(arg115_1, (128, ), (1, ), 'input')
        buf280 = reinterpret_tensor(buf234, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf234  # reuse
        generated_kernel_1(buf280, buf271, arg115_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg115_1
        assert_size_stride(arg116_1, (1024, 2048), (2048, 1), 'input')
        buf275 = reinterpret_tensor(buf255, (71, 1024), (1024, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf270, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg116_1, (2048, 1024), (1, 2048), 0), out=buf275)
        del arg116_1
        assert_size_stride(arg117_1, (128, ), (1, ), 'input')
        buf281 = reinterpret_tensor(buf253, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf253  # reuse
        generated_kernel_2(buf281, buf275, arg117_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg117_1
        assert_size_stride(arg118_1, (1024, 2048), (2048, 1), 'input')
        buf279 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf270, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg118_1, (2048, 1024), (1, 2048), 0), out=buf279)
        del arg118_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_13, mul_140, q_embed_10, mul_142, k_embed_10, linear_72, view_32, value_states_10, attn_output_40], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf283 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf280, buf281, reinterpret_tensor(buf279, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf282, scale=0.08838834764831845, enable_gqa=True)
        buf284 = buf283[0]
        assert_tensor_metadata(buf284, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf284, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf283
        buf286 = reinterpret_tensor(buf280, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf280  # reuse
        generated_kernel_7(buf286, buf284, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg119_1, (2048, 2048), (2048, 1), 'input')
        buf287 = reinterpret_tensor(buf284, (71, 2048), (2048, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [transpose_43, attn_output_41, reshape_10, attn_output_43], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf286, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg119_1, (2048, 2048), (1, 2048), 0), out=buf287)
        del arg119_1
        assert_size_stride(arg120_1, (2048, ), (1, ), 'input')
        buf290 = reinterpret_tensor(buf286, (1, 71, 2048), (145408, 2048, 1), 0); del buf286  # reuse
        generated_kernel_13(buf290, buf262, buf268, buf287, arg120_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg120_1
        assert_size_stride(arg121_1, (6144, 2048), (2048, 1), 'input')
        buf291 = reinterpret_tensor(buf267, (71, 6144), (6144, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130, linear_74], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf290, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg121_1, (2048, 6144), (1, 2048), 0), out=buf291)
        del arg121_1
        assert_size_stride(arg122_1, (6144, 2048), (2048, 1), 'input')
        buf292 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf290, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg122_1, (2048, 6144), (1, 2048), 0), out=buf292)
        del arg122_1
        buf293 = reinterpret_tensor(buf265, (1, 71, 6144), (436224, 6144, 1), 0); del buf265  # reuse
        generated_kernel_9(buf293, buf291, buf292, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg123_1, (2048, 6144), (6144, 1), 'input')
        buf294 = reinterpret_tensor(buf290, (71, 2048), (2048, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_145, down_proj_10], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf293, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg123_1, (6144, 2048), (1, 6144), 0), out=buf294)
        del arg123_1
        assert_size_stride(arg124_1, (2048, ), (1, ), 'input')
        buf297 = buf270; del buf270  # reuse
        generated_kernel_14(buf297, buf262, buf268, buf287, buf294, arg124_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg124_1
        assert_size_stride(arg125_1, (2048, 2048), (2048, 1), 'input')
        buf298 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [to_94, hidden_states_134, linear_77], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf297, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg125_1, (2048, 2048), (1, 2048), 0), out=buf298)
        del arg125_1
        assert_size_stride(arg126_1, (128, ), (1, ), 'input')
        buf307 = reinterpret_tensor(buf215, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf215  # reuse
        generated_kernel_1(buf307, buf298, arg126_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg126_1
        del buf298
        assert_size_stride(arg127_1, (1024, 2048), (2048, 1), 'input')
        buf302 = reinterpret_tensor(buf281, (71, 1024), (1024, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf297, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg127_1, (2048, 1024), (1, 2048), 0), out=buf302)
        del arg127_1
        assert_size_stride(arg128_1, (128, ), (1, ), 'input')
        buf308 = reinterpret_tensor(buf279, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf279  # reuse
        generated_kernel_2(buf308, buf302, arg128_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg128_1
        assert_size_stride(arg129_1, (1024, 2048), (2048, 1), 'input')
        buf306 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf297, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg129_1, (2048, 1024), (1, 2048), 0), out=buf306)
        del arg129_1
        buf309 = buf282; del buf282  # reuse
        buf335 = buf256; del buf256  # reuse
        buf362 = buf18; del buf18  # reuse
        generated_kernel_6(buf309, buf335, buf362, buf15, threads=[5041], error_buf_idx=4)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_14, mul_153, q_embed_11, mul_155, k_embed_11, linear_79, view_35, value_states_11, attn_output_44], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf310 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf307, buf308, reinterpret_tensor(buf306, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf309, scale=0.08838834764831845, enable_gqa=True)
        buf311 = buf310[0]
        assert_tensor_metadata(buf311, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf311, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf310
        buf313 = reinterpret_tensor(buf307, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf307  # reuse
        generated_kernel_7(buf313, buf311, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg130_1, (2048, 2048), (2048, 1), 'input')
        buf314 = reinterpret_tensor(buf311, (71, 2048), (2048, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [transpose_47, attn_output_45, reshape_11, attn_output_47], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf313, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 2048), (1, 2048), 0), out=buf314)
        del arg130_1
        assert_size_stride(arg131_1, (2048, ), (1, ), 'input')
        buf315 = reinterpret_tensor(buf313, (1, 71, 2048), (145408, 2048, 1), 0); del buf313  # reuse
        buf317 = buf297; del buf297  # reuse
        generated_kernel_16(buf315, buf317, buf262, buf268, buf287, buf294, buf314, arg131_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg131_1
        del buf262
        assert_size_stride(arg132_1, (6144, 2048), (2048, 1), 'input')
        buf318 = reinterpret_tensor(buf293, (71, 6144), (6144, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140, pow_48, variance_47, add_97, rsqrt_47, hidden_states_141, to_100, hidden_states_142, linear_81], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf317, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg132_1, (2048, 6144), (1, 2048), 0), out=buf318)
        del arg132_1
        assert_size_stride(arg133_1, (6144, 2048), (2048, 1), 'input')
        buf319 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf317, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg133_1, (2048, 6144), (1, 2048), 0), out=buf319)
        del arg133_1
        buf320 = reinterpret_tensor(buf291, (1, 71, 6144), (436224, 6144, 1), 0); del buf291  # reuse
        generated_kernel_9(buf320, buf318, buf319, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg134_1, (2048, 6144), (6144, 1), 'input')
        buf321 = reinterpret_tensor(buf317, (71, 2048), (2048, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_158, down_proj_11], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf320, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg134_1, (6144, 2048), (1, 6144), 0), out=buf321)
        del arg134_1
        assert_size_stride(arg135_1, (2048, ), (1, ), 'input')
        buf323 = reinterpret_tensor(buf314, (1, 71, 2048), (145408, 2048, 1), 0); del buf314  # reuse
        generated_kernel_12(buf323, buf315, buf321, arg135_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg135_1
        assert_size_stride(arg136_1, (2048, 2048), (2048, 1), 'input')
        buf324 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf323, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg136_1, (2048, 2048), (1, 2048), 0), out=buf324)
        del arg136_1
        assert_size_stride(arg137_1, (128, ), (1, ), 'input')
        buf333 = reinterpret_tensor(buf287, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf287  # reuse
        generated_kernel_1(buf333, buf324, arg137_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg137_1
        assert_size_stride(arg138_1, (1024, 2048), (2048, 1), 'input')
        buf328 = reinterpret_tensor(buf308, (71, 1024), (1024, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf323, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg138_1, (2048, 1024), (1, 2048), 0), out=buf328)
        del arg138_1
        assert_size_stride(arg139_1, (128, ), (1, ), 'input')
        buf334 = reinterpret_tensor(buf306, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf306  # reuse
        generated_kernel_2(buf334, buf328, arg139_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg139_1
        assert_size_stride(arg140_1, (1024, 2048), (2048, 1), 'input')
        buf332 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf323, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg140_1, (2048, 1024), (1, 2048), 0), out=buf332)
        del arg140_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_15, mul_166, q_embed_12, mul_168, k_embed_12, linear_86, view_38, value_states_12, attn_output_48], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf336 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf333, buf334, reinterpret_tensor(buf332, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf335, scale=0.08838834764831845, enable_gqa=True)
        buf337 = buf336[0]
        assert_tensor_metadata(buf337, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf337, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf336
        buf339 = reinterpret_tensor(buf333, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf333  # reuse
        generated_kernel_7(buf339, buf337, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg141_1, (2048, 2048), (2048, 1), 'input')
        buf340 = reinterpret_tensor(buf337, (71, 2048), (2048, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [transpose_51, attn_output_49, reshape_12, attn_output_51], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf339, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg141_1, (2048, 2048), (1, 2048), 0), out=buf340)
        del arg141_1
        assert_size_stride(arg142_1, (2048, ), (1, ), 'input')
        buf343 = reinterpret_tensor(buf339, (1, 71, 2048), (145408, 2048, 1), 0); del buf339  # reuse
        generated_kernel_13(buf343, buf315, buf321, buf340, arg142_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg142_1
        assert_size_stride(arg143_1, (6144, 2048), (2048, 1), 'input')
        buf344 = reinterpret_tensor(buf320, (71, 6144), (6144, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_154, linear_88], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf343, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg143_1, (2048, 6144), (1, 2048), 0), out=buf344)
        del arg143_1
        assert_size_stride(arg144_1, (6144, 2048), (2048, 1), 'input')
        buf345 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf343, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg144_1, (2048, 6144), (1, 2048), 0), out=buf345)
        del arg144_1
        buf346 = reinterpret_tensor(buf318, (1, 71, 6144), (436224, 6144, 1), 0); del buf318  # reuse
        generated_kernel_9(buf346, buf344, buf345, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg145_1, (2048, 6144), (6144, 1), 'input')
        buf347 = reinterpret_tensor(buf343, (71, 2048), (2048, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_171, down_proj_12], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf346, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg145_1, (6144, 2048), (1, 6144), 0), out=buf347)
        del arg145_1
        assert_size_stride(arg146_1, (2048, ), (1, ), 'input')
        buf350 = buf323; del buf323  # reuse
        generated_kernel_14(buf350, buf315, buf321, buf340, buf347, arg146_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg146_1
        assert_size_stride(arg147_1, (2048, 2048), (2048, 1), 'input')
        buf351 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [to_110, hidden_states_158, linear_91], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf350, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg147_1, (2048, 2048), (1, 2048), 0), out=buf351)
        del arg147_1
        assert_size_stride(arg148_1, (128, ), (1, ), 'input')
        buf360 = reinterpret_tensor(buf268, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf268  # reuse
        generated_kernel_1(buf360, buf351, arg148_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg148_1
        del buf351
        assert_size_stride(arg149_1, (1024, 2048), (2048, 1), 'input')
        buf355 = reinterpret_tensor(buf334, (71, 1024), (1024, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf350, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg149_1, (2048, 1024), (1, 2048), 0), out=buf355)
        del arg149_1
        assert_size_stride(arg150_1, (128, ), (1, ), 'input')
        buf361 = reinterpret_tensor(buf332, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf332  # reuse
        generated_kernel_2(buf361, buf355, arg150_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg150_1
        assert_size_stride(arg151_1, (1024, 2048), (2048, 1), 'input')
        buf359 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf350, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg151_1, (2048, 1024), (1, 2048), 0), out=buf359)
        del arg151_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_16, mul_179, q_embed_13, mul_181, k_embed_13, linear_93, view_41, value_states_13, attn_output_52], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf363 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf360, buf361, reinterpret_tensor(buf359, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf362, scale=0.08838834764831845, enable_gqa=True)
        buf364 = buf363[0]
        assert_tensor_metadata(buf364, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf364, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf363
        buf366 = reinterpret_tensor(buf360, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf360  # reuse
        generated_kernel_7(buf366, buf364, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg152_1, (2048, 2048), (2048, 1), 'input')
        buf367 = reinterpret_tensor(buf364, (71, 2048), (2048, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [transpose_55, attn_output_53, reshape_13, attn_output_55], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf366, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg152_1, (2048, 2048), (1, 2048), 0), out=buf367)
        del arg152_1
        assert_size_stride(arg153_1, (2048, ), (1, ), 'input')
        buf368 = reinterpret_tensor(buf366, (1, 71, 2048), (145408, 2048, 1), 0); del buf366  # reuse
        buf370 = buf350; del buf350  # reuse
        generated_kernel_16(buf368, buf370, buf315, buf321, buf340, buf347, buf367, arg153_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg153_1
        del buf315
        assert_size_stride(arg154_1, (6144, 2048), (2048, 1), 'input')
        buf371 = reinterpret_tensor(buf346, (71, 6144), (6144, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_164, pow_56, variance_55, add_113, rsqrt_55, hidden_states_165, to_116, hidden_states_166, linear_95], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf370, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg154_1, (2048, 6144), (1, 2048), 0), out=buf371)
        del arg154_1
        assert_size_stride(arg155_1, (6144, 2048), (2048, 1), 'input')
        buf372 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf370, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg155_1, (2048, 6144), (1, 2048), 0), out=buf372)
        del arg155_1
        buf373 = reinterpret_tensor(buf344, (1, 71, 6144), (436224, 6144, 1), 0); del buf344  # reuse
        generated_kernel_9(buf373, buf371, buf372, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg156_1, (2048, 6144), (6144, 1), 'input')
        buf374 = reinterpret_tensor(buf370, (71, 2048), (2048, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_184, down_proj_13], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf373, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg156_1, (6144, 2048), (1, 6144), 0), out=buf374)
        del arg156_1
        assert_size_stride(arg157_1, (2048, ), (1, ), 'input')
        buf376 = reinterpret_tensor(buf367, (1, 71, 2048), (145408, 2048, 1), 0); del buf367  # reuse
        generated_kernel_12(buf376, buf368, buf374, arg157_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg157_1
        assert_size_stride(arg158_1, (2048, 2048), (2048, 1), 'input')
        buf377 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf376, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg158_1, (2048, 2048), (1, 2048), 0), out=buf377)
        del arg158_1
        buf378 = empty_strided((1, 71, 16, 1), (1152, 16, 1, 1152), device='mps', dtype=torch.float32)
        generated_kernel_17(buf378, buf377, threads=[1136, 128], group_size=[1, 128], error_buf_idx=2)
        assert_size_stride(arg159_1, (128, ), (1, ), 'input')
        buf380 = reinterpret_tensor(buf361, (1, 16, 71, 64), (72704, 64, 1024, 1), 0); del buf361  # reuse
        generated_kernel_18(buf380, arg159_1, buf377, buf378, threads=[72704], error_buf_idx=4)
        buf387 = reinterpret_tensor(buf340, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf340  # reuse
        generated_kernel_19(buf387, buf380, arg159_1, buf377, buf378, arg2_1, threads=[145408], error_buf_idx=6)
        del arg159_1
        del buf378
        assert_size_stride(arg160_1, (1024, 2048), (2048, 1), 'input')
        buf382 = reinterpret_tensor(buf380, (71, 1024), (1024, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf376, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg160_1, (2048, 1024), (1, 2048), 0), out=buf382)
        del arg160_1
        assert_size_stride(arg161_1, (128, ), (1, ), 'input')
        buf388 = reinterpret_tensor(buf359, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf359  # reuse
        generated_kernel_2(buf388, buf382, arg161_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg161_1
        assert_size_stride(arg162_1, (1024, 2048), (2048, 1), 'input')
        buf386 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf376, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg162_1, (2048, 1024), (1, 2048), 0), out=buf386)
        del arg162_1
        buf389 = buf362; del buf362  # reuse
        buf416 = buf335; del buf335  # reuse
        buf442 = buf309; del buf309  # reuse
        generated_kernel_6(buf389, buf416, buf442, buf15, threads=[5041], error_buf_idx=4)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_17, mul_192, q_embed_14, mul_194, k_embed_14, linear_100, view_44, value_states_14, attn_output_56], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf390 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf387, buf388, reinterpret_tensor(buf386, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf389, scale=0.08838834764831845, enable_gqa=True)
        buf391 = buf390[0]
        assert_tensor_metadata(buf391, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf391, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf390
        buf393 = reinterpret_tensor(buf387, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf387  # reuse
        generated_kernel_7(buf393, buf391, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg163_1, (2048, 2048), (2048, 1), 'input')
        buf394 = reinterpret_tensor(buf391, (71, 2048), (2048, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [transpose_59, attn_output_57, reshape_14, attn_output_59], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf393, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg163_1, (2048, 2048), (1, 2048), 0), out=buf394)
        del arg163_1
        assert_size_stride(arg164_1, (2048, ), (1, ), 'input')
        buf397 = reinterpret_tensor(buf393, (1, 71, 2048), (145408, 2048, 1), 0); del buf393  # reuse
        generated_kernel_13(buf397, buf368, buf374, buf394, arg164_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg164_1
        assert_size_stride(arg165_1, (6144, 2048), (2048, 1), 'input')
        buf398 = reinterpret_tensor(buf373, (71, 6144), (6144, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_176, linear_102], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf397, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg165_1, (2048, 6144), (1, 2048), 0), out=buf398)
        del arg165_1
        assert_size_stride(arg166_1, (6144, 2048), (2048, 1), 'input')
        buf399 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf397, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg166_1, (2048, 6144), (1, 2048), 0), out=buf399)
        del arg166_1
        buf400 = reinterpret_tensor(buf371, (1, 71, 6144), (436224, 6144, 1), 0); del buf371  # reuse
        generated_kernel_9(buf400, buf398, buf399, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg167_1, (2048, 6144), (6144, 1), 'input')
        buf401 = reinterpret_tensor(buf397, (71, 2048), (2048, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_197, down_proj_14], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf400, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg167_1, (6144, 2048), (1, 6144), 0), out=buf401)
        del arg167_1
        assert_size_stride(arg168_1, (2048, ), (1, ), 'input')
        buf404 = buf376; del buf376  # reuse
        generated_kernel_14(buf404, buf368, buf374, buf394, buf401, arg168_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg168_1
        assert_size_stride(arg169_1, (2048, 2048), (2048, 1), 'input')
        buf405 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [to_129, hidden_states_180, linear_105], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf404, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg169_1, (2048, 2048), (1, 2048), 0), out=buf405)
        del arg169_1
        assert_size_stride(arg170_1, (128, ), (1, ), 'input')
        buf414 = reinterpret_tensor(buf321, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf321  # reuse
        generated_kernel_1(buf414, buf405, arg170_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg170_1
        del buf405
        assert_size_stride(arg171_1, (1024, 2048), (2048, 1), 'input')
        buf409 = reinterpret_tensor(buf388, (71, 1024), (1024, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf404, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg171_1, (2048, 1024), (1, 2048), 0), out=buf409)
        del arg171_1
        assert_size_stride(arg172_1, (128, ), (1, ), 'input')
        buf415 = reinterpret_tensor(buf386, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf386  # reuse
        generated_kernel_2(buf415, buf409, arg172_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg172_1
        assert_size_stride(arg173_1, (1024, 2048), (2048, 1), 'input')
        buf413 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf404, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg173_1, (2048, 1024), (1, 2048), 0), out=buf413)
        del arg173_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_18, mul_205, q_embed_15, mul_207, k_embed_15, linear_107, view_47, value_states_15, attn_output_60], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf417 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf414, buf415, reinterpret_tensor(buf413, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf416, scale=0.08838834764831845, enable_gqa=True)
        buf418 = buf417[0]
        assert_tensor_metadata(buf418, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf418, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf417
        buf420 = reinterpret_tensor(buf414, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf414  # reuse
        generated_kernel_7(buf420, buf418, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg174_1, (2048, 2048), (2048, 1), 'input')
        buf421 = reinterpret_tensor(buf418, (71, 2048), (2048, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [transpose_63, attn_output_61, reshape_15, attn_output_63], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf420, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg174_1, (2048, 2048), (1, 2048), 0), out=buf421)
        del arg174_1
        assert_size_stride(arg175_1, (2048, ), (1, ), 'input')
        buf422 = reinterpret_tensor(buf420, (1, 71, 2048), (145408, 2048, 1), 0); del buf420  # reuse
        buf424 = buf404; del buf404  # reuse
        generated_kernel_16(buf422, buf424, buf368, buf374, buf394, buf401, buf421, arg175_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg175_1
        del buf368
        assert_size_stride(arg176_1, (6144, 2048), (2048, 1), 'input')
        buf425 = reinterpret_tensor(buf400, (71, 6144), (6144, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_186, pow_64, variance_62, add_129, rsqrt_63, hidden_states_187, to_135, hidden_states_188, linear_109], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf424, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 6144), (1, 2048), 0), out=buf425)
        del arg176_1
        assert_size_stride(arg177_1, (6144, 2048), (2048, 1), 'input')
        buf426 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf424, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg177_1, (2048, 6144), (1, 2048), 0), out=buf426)
        del arg177_1
        buf427 = reinterpret_tensor(buf398, (1, 71, 6144), (436224, 6144, 1), 0); del buf398  # reuse
        generated_kernel_9(buf427, buf425, buf426, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg178_1, (2048, 6144), (6144, 1), 'input')
        buf428 = reinterpret_tensor(buf424, (71, 2048), (2048, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_210, down_proj_15], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf427, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg178_1, (6144, 2048), (1, 6144), 0), out=buf428)
        del arg178_1
        assert_size_stride(arg179_1, (2048, ), (1, ), 'input')
        buf430 = reinterpret_tensor(buf421, (1, 71, 2048), (145408, 2048, 1), 0); del buf421  # reuse
        generated_kernel_12(buf430, buf422, buf428, arg179_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg179_1
        assert_size_stride(arg180_1, (2048, 2048), (2048, 1), 'input')
        buf431 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf430, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg180_1, (2048, 2048), (1, 2048), 0), out=buf431)
        del arg180_1
        assert_size_stride(arg181_1, (128, ), (1, ), 'input')
        buf440 = reinterpret_tensor(buf394, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf394  # reuse
        generated_kernel_1(buf440, buf431, arg181_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg181_1
        assert_size_stride(arg182_1, (1024, 2048), (2048, 1), 'input')
        buf435 = reinterpret_tensor(buf415, (71, 1024), (1024, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf430, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg182_1, (2048, 1024), (1, 2048), 0), out=buf435)
        del arg182_1
        assert_size_stride(arg183_1, (128, ), (1, ), 'input')
        buf441 = reinterpret_tensor(buf413, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf413  # reuse
        generated_kernel_2(buf441, buf435, arg183_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg183_1
        assert_size_stride(arg184_1, (1024, 2048), (2048, 1), 'input')
        buf439 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf430, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg184_1, (2048, 1024), (1, 2048), 0), out=buf439)
        del arg184_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_19, mul_218, q_embed_16, mul_220, k_embed_16, linear_114, view_50, value_states_16, attn_output_64], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf443 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf440, buf441, reinterpret_tensor(buf439, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf442, scale=0.08838834764831845, enable_gqa=True)
        buf444 = buf443[0]
        assert_tensor_metadata(buf444, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf444, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf443
        buf446 = reinterpret_tensor(buf440, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf440  # reuse
        generated_kernel_7(buf446, buf444, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg185_1, (2048, 2048), (2048, 1), 'input')
        buf447 = reinterpret_tensor(buf444, (71, 2048), (2048, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [transpose_67, attn_output_65, reshape_16, attn_output_67], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf446, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg185_1, (2048, 2048), (1, 2048), 0), out=buf447)
        del arg185_1
        assert_size_stride(arg186_1, (2048, ), (1, ), 'input')
        buf450 = reinterpret_tensor(buf446, (1, 71, 2048), (145408, 2048, 1), 0); del buf446  # reuse
        generated_kernel_13(buf450, buf422, buf428, buf447, arg186_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg186_1
        assert_size_stride(arg187_1, (6144, 2048), (2048, 1), 'input')
        buf451 = reinterpret_tensor(buf427, (71, 6144), (6144, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_200, linear_116], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf450, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg187_1, (2048, 6144), (1, 2048), 0), out=buf451)
        del arg187_1
        assert_size_stride(arg188_1, (6144, 2048), (2048, 1), 'input')
        buf452 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf450, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 6144), (1, 2048), 0), out=buf452)
        del arg188_1
        buf453 = reinterpret_tensor(buf425, (1, 71, 6144), (436224, 6144, 1), 0); del buf425  # reuse
        generated_kernel_9(buf453, buf451, buf452, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg189_1, (2048, 6144), (6144, 1), 'input')
        buf454 = reinterpret_tensor(buf450, (71, 2048), (2048, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_223, down_proj_16], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf453, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg189_1, (6144, 2048), (1, 6144), 0), out=buf454)
        del arg189_1
        assert_size_stride(arg190_1, (2048, ), (1, ), 'input')
        buf457 = buf430; del buf430  # reuse
        generated_kernel_14(buf457, buf422, buf428, buf447, buf454, arg190_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg190_1
        assert_size_stride(arg191_1, (2048, 2048), (2048, 1), 'input')
        buf458 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [to_145, hidden_states_204, linear_119], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf457, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg191_1, (2048, 2048), (1, 2048), 0), out=buf458)
        del arg191_1
        assert_size_stride(arg192_1, (128, ), (1, ), 'input')
        buf467 = reinterpret_tensor(buf374, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf374  # reuse
        generated_kernel_1(buf467, buf458, arg192_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg192_1
        del buf458
        assert_size_stride(arg193_1, (1024, 2048), (2048, 1), 'input')
        buf462 = reinterpret_tensor(buf441, (71, 1024), (1024, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf457, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg193_1, (2048, 1024), (1, 2048), 0), out=buf462)
        del arg193_1
        assert_size_stride(arg194_1, (128, ), (1, ), 'input')
        buf468 = reinterpret_tensor(buf439, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf439  # reuse
        generated_kernel_2(buf468, buf462, arg194_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg194_1
        assert_size_stride(arg195_1, (1024, 2048), (2048, 1), 'input')
        buf466 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf457, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg195_1, (2048, 1024), (1, 2048), 0), out=buf466)
        del arg195_1
        buf469 = buf442; del buf442  # reuse
        buf495 = buf416; del buf416  # reuse
        generated_kernel_15(buf469, buf495, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_20, mul_231, q_embed_17, mul_233, k_embed_17, linear_121, view_53, value_states_17, attn_output_68], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf470 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf467, buf468, reinterpret_tensor(buf466, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf469, scale=0.08838834764831845, enable_gqa=True)
        buf471 = buf470[0]
        assert_tensor_metadata(buf471, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf471, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf470
        buf473 = reinterpret_tensor(buf467, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf467  # reuse
        generated_kernel_7(buf473, buf471, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg196_1, (2048, 2048), (2048, 1), 'input')
        buf474 = reinterpret_tensor(buf471, (71, 2048), (2048, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [transpose_71, attn_output_69, reshape_17, attn_output_71], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf473, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg196_1, (2048, 2048), (1, 2048), 0), out=buf474)
        del arg196_1
        assert_size_stride(arg197_1, (2048, ), (1, ), 'input')
        buf475 = reinterpret_tensor(buf473, (1, 71, 2048), (145408, 2048, 1), 0); del buf473  # reuse
        buf477 = buf457; del buf457  # reuse
        generated_kernel_16(buf475, buf477, buf422, buf428, buf447, buf454, buf474, arg197_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg197_1
        del buf422
        assert_size_stride(arg198_1, (6144, 2048), (2048, 1), 'input')
        buf478 = reinterpret_tensor(buf453, (71, 6144), (6144, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_210, pow_72, variance_70, add_145, rsqrt_71, hidden_states_211, to_151, hidden_states_212, linear_123], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf477, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg198_1, (2048, 6144), (1, 2048), 0), out=buf478)
        del arg198_1
        assert_size_stride(arg199_1, (6144, 2048), (2048, 1), 'input')
        buf479 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf477, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg199_1, (2048, 6144), (1, 2048), 0), out=buf479)
        del arg199_1
        buf480 = reinterpret_tensor(buf451, (1, 71, 6144), (436224, 6144, 1), 0); del buf451  # reuse
        generated_kernel_9(buf480, buf478, buf479, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg200_1, (2048, 6144), (6144, 1), 'input')
        buf481 = reinterpret_tensor(buf477, (71, 2048), (2048, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_236, down_proj_17], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf480, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg200_1, (6144, 2048), (1, 6144), 0), out=buf481)
        del arg200_1
        assert_size_stride(arg201_1, (2048, ), (1, ), 'input')
        buf483 = reinterpret_tensor(buf474, (1, 71, 2048), (145408, 2048, 1), 0); del buf474  # reuse
        generated_kernel_12(buf483, buf475, buf481, arg201_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg201_1
        assert_size_stride(arg202_1, (2048, 2048), (2048, 1), 'input')
        buf484 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf483, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg202_1, (2048, 2048), (1, 2048), 0), out=buf484)
        del arg202_1
        assert_size_stride(arg203_1, (128, ), (1, ), 'input')
        buf493 = reinterpret_tensor(buf447, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf447  # reuse
        generated_kernel_1(buf493, buf484, arg203_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg203_1
        assert_size_stride(arg204_1, (1024, 2048), (2048, 1), 'input')
        buf488 = reinterpret_tensor(buf468, (71, 1024), (1024, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf483, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg204_1, (2048, 1024), (1, 2048), 0), out=buf488)
        del arg204_1
        assert_size_stride(arg205_1, (128, ), (1, ), 'input')
        buf494 = reinterpret_tensor(buf466, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf466  # reuse
        generated_kernel_2(buf494, buf488, arg205_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg205_1
        assert_size_stride(arg206_1, (1024, 2048), (2048, 1), 'input')
        buf492 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf483, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg206_1, (2048, 1024), (1, 2048), 0), out=buf492)
        del arg206_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_21, mul_244, q_embed_18, mul_246, k_embed_18, linear_128, view_56, value_states_18, attn_output_72], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf496 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf493, buf494, reinterpret_tensor(buf492, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf495, scale=0.08838834764831845, enable_gqa=True)
        buf497 = buf496[0]
        assert_tensor_metadata(buf497, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf497, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf496
        buf499 = reinterpret_tensor(buf493, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf493  # reuse
        generated_kernel_7(buf499, buf497, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg207_1, (2048, 2048), (2048, 1), 'input')
        buf500 = reinterpret_tensor(buf497, (71, 2048), (2048, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [transpose_75, attn_output_73, reshape_18, attn_output_75], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf499, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg207_1, (2048, 2048), (1, 2048), 0), out=buf500)
        del arg207_1
        assert_size_stride(arg208_1, (2048, ), (1, ), 'input')
        buf503 = reinterpret_tensor(buf499, (1, 71, 2048), (145408, 2048, 1), 0); del buf499  # reuse
        generated_kernel_13(buf503, buf475, buf481, buf500, arg208_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg208_1
        assert_size_stride(arg209_1, (6144, 2048), (2048, 1), 'input')
        buf504 = reinterpret_tensor(buf480, (71, 6144), (6144, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_224, linear_130], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf503, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg209_1, (2048, 6144), (1, 2048), 0), out=buf504)
        del arg209_1
        assert_size_stride(arg210_1, (6144, 2048), (2048, 1), 'input')
        buf505 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf503, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg210_1, (2048, 6144), (1, 2048), 0), out=buf505)
        del arg210_1
        buf506 = reinterpret_tensor(buf478, (1, 71, 6144), (436224, 6144, 1), 0); del buf478  # reuse
        generated_kernel_9(buf506, buf504, buf505, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg211_1, (2048, 6144), (6144, 1), 'input')
        buf507 = reinterpret_tensor(buf503, (71, 2048), (2048, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_249, down_proj_18], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf506, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg211_1, (6144, 2048), (1, 6144), 0), out=buf507)
        del arg211_1
        assert_size_stride(arg212_1, (2048, ), (1, ), 'input')
        buf510 = buf483; del buf483  # reuse
        generated_kernel_14(buf510, buf475, buf481, buf500, buf507, arg212_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg212_1
        assert_size_stride(arg213_1, (2048, 2048), (2048, 1), 'input')
        buf511 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [to_161, hidden_states_228, linear_133], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf510, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg213_1, (2048, 2048), (1, 2048), 0), out=buf511)
        del arg213_1
        assert_size_stride(arg214_1, (128, ), (1, ), 'input')
        buf520 = reinterpret_tensor(buf428, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf428  # reuse
        generated_kernel_1(buf520, buf511, arg214_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg214_1
        del buf511
        assert_size_stride(arg215_1, (1024, 2048), (2048, 1), 'input')
        buf515 = reinterpret_tensor(buf494, (71, 1024), (1024, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf510, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg215_1, (2048, 1024), (1, 2048), 0), out=buf515)
        del arg215_1
        assert_size_stride(arg216_1, (128, ), (1, ), 'input')
        buf521 = reinterpret_tensor(buf492, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf492  # reuse
        generated_kernel_2(buf521, buf515, arg216_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg216_1
        assert_size_stride(arg217_1, (1024, 2048), (2048, 1), 'input')
        buf519 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf510, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg217_1, (2048, 1024), (1, 2048), 0), out=buf519)
        del arg217_1
        buf522 = buf495; del buf495  # reuse
        buf548 = buf469; del buf469  # reuse
        generated_kernel_15(buf522, buf548, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_22, mul_257, q_embed_19, mul_259, k_embed_19, linear_135, view_59, value_states_19, attn_output_76], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf523 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf520, buf521, reinterpret_tensor(buf519, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf522, scale=0.08838834764831845, enable_gqa=True)
        buf524 = buf523[0]
        assert_tensor_metadata(buf524, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf524, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf523
        buf526 = reinterpret_tensor(buf520, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf520  # reuse
        generated_kernel_7(buf526, buf524, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg218_1, (2048, 2048), (2048, 1), 'input')
        buf527 = reinterpret_tensor(buf524, (71, 2048), (2048, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [transpose_79, attn_output_77, reshape_19, attn_output_79], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf526, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg218_1, (2048, 2048), (1, 2048), 0), out=buf527)
        del arg218_1
        assert_size_stride(arg219_1, (2048, ), (1, ), 'input')
        buf528 = reinterpret_tensor(buf526, (1, 71, 2048), (145408, 2048, 1), 0); del buf526  # reuse
        buf530 = buf510; del buf510  # reuse
        generated_kernel_16(buf528, buf530, buf475, buf481, buf500, buf507, buf527, arg219_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg219_1
        del buf475
        assert_size_stride(arg220_1, (6144, 2048), (2048, 1), 'input')
        buf531 = reinterpret_tensor(buf506, (71, 6144), (6144, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_234, pow_80, variance_78, add_161, rsqrt_79, hidden_states_235, to_167, hidden_states_236, linear_137], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf530, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg220_1, (2048, 6144), (1, 2048), 0), out=buf531)
        del arg220_1
        assert_size_stride(arg221_1, (6144, 2048), (2048, 1), 'input')
        buf532 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf530, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg221_1, (2048, 6144), (1, 2048), 0), out=buf532)
        del arg221_1
        buf533 = reinterpret_tensor(buf504, (1, 71, 6144), (436224, 6144, 1), 0); del buf504  # reuse
        generated_kernel_9(buf533, buf531, buf532, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg222_1, (2048, 6144), (6144, 1), 'input')
        buf534 = reinterpret_tensor(buf530, (71, 2048), (2048, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_262, down_proj_19], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf533, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg222_1, (6144, 2048), (1, 6144), 0), out=buf534)
        del arg222_1
        assert_size_stride(arg223_1, (2048, ), (1, ), 'input')
        buf536 = reinterpret_tensor(buf527, (1, 71, 2048), (145408, 2048, 1), 0); del buf527  # reuse
        generated_kernel_12(buf536, buf528, buf534, arg223_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg223_1
        assert_size_stride(arg224_1, (2048, 2048), (2048, 1), 'input')
        buf537 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf536, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg224_1, (2048, 2048), (1, 2048), 0), out=buf537)
        del arg224_1
        assert_size_stride(arg225_1, (128, ), (1, ), 'input')
        buf546 = reinterpret_tensor(buf500, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf500  # reuse
        generated_kernel_1(buf546, buf537, arg225_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg225_1
        assert_size_stride(arg226_1, (1024, 2048), (2048, 1), 'input')
        buf541 = reinterpret_tensor(buf521, (71, 1024), (1024, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf536, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 1024), (1, 2048), 0), out=buf541)
        del arg226_1
        assert_size_stride(arg227_1, (128, ), (1, ), 'input')
        buf547 = reinterpret_tensor(buf519, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf519  # reuse
        generated_kernel_2(buf547, buf541, arg227_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg227_1
        assert_size_stride(arg228_1, (1024, 2048), (2048, 1), 'input')
        buf545 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf536, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg228_1, (2048, 1024), (1, 2048), 0), out=buf545)
        del arg228_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_23, mul_270, q_embed_20, mul_272, k_embed_20, linear_142, view_62, value_states_20, attn_output_80], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf549 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf546, buf547, reinterpret_tensor(buf545, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf548, scale=0.08838834764831845, enable_gqa=True)
        buf550 = buf549[0]
        assert_tensor_metadata(buf550, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf550, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf549
        buf552 = reinterpret_tensor(buf546, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf546  # reuse
        generated_kernel_7(buf552, buf550, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg229_1, (2048, 2048), (2048, 1), 'input')
        buf553 = reinterpret_tensor(buf550, (71, 2048), (2048, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [transpose_83, attn_output_81, reshape_20, attn_output_83], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf552, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg229_1, (2048, 2048), (1, 2048), 0), out=buf553)
        del arg229_1
        assert_size_stride(arg230_1, (2048, ), (1, ), 'input')
        buf556 = reinterpret_tensor(buf552, (1, 71, 2048), (145408, 2048, 1), 0); del buf552  # reuse
        generated_kernel_13(buf556, buf528, buf534, buf553, arg230_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg230_1
        assert_size_stride(arg231_1, (6144, 2048), (2048, 1), 'input')
        buf557 = reinterpret_tensor(buf533, (71, 6144), (6144, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_248, linear_144], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf556, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg231_1, (2048, 6144), (1, 2048), 0), out=buf557)
        del arg231_1
        assert_size_stride(arg232_1, (6144, 2048), (2048, 1), 'input')
        buf558 = buf532; del buf532  # reuse
        # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf556, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg232_1, (2048, 6144), (1, 2048), 0), out=buf558)
        del arg232_1
        buf559 = reinterpret_tensor(buf531, (1, 71, 6144), (436224, 6144, 1), 0); del buf531  # reuse
        generated_kernel_9(buf559, buf557, buf558, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg233_1, (2048, 6144), (6144, 1), 'input')
        buf560 = reinterpret_tensor(buf556, (71, 2048), (2048, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_275, down_proj_20], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf559, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg233_1, (6144, 2048), (1, 6144), 0), out=buf560)
        del arg233_1
        assert_size_stride(arg234_1, (2048, ), (1, ), 'input')
        buf563 = buf536; del buf536  # reuse
        generated_kernel_14(buf563, buf528, buf534, buf553, buf560, arg234_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg234_1
        assert_size_stride(arg235_1, (2048, 2048), (2048, 1), 'input')
        buf564 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [to_177, hidden_states_252, linear_147], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf563, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg235_1, (2048, 2048), (1, 2048), 0), out=buf564)
        del arg235_1
        assert_size_stride(arg236_1, (128, ), (1, ), 'input')
        buf573 = reinterpret_tensor(buf481, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf481  # reuse
        generated_kernel_1(buf573, buf564, arg236_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg236_1
        del buf564
        assert_size_stride(arg237_1, (1024, 2048), (2048, 1), 'input')
        buf568 = reinterpret_tensor(buf547, (71, 1024), (1024, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf563, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg237_1, (2048, 1024), (1, 2048), 0), out=buf568)
        del arg237_1
        assert_size_stride(arg238_1, (128, ), (1, ), 'input')
        buf574 = reinterpret_tensor(buf545, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf545  # reuse
        generated_kernel_2(buf574, buf568, arg238_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg238_1
        assert_size_stride(arg239_1, (1024, 2048), (2048, 1), 'input')
        buf572 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf563, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg239_1, (2048, 1024), (1, 2048), 0), out=buf572)
        del arg239_1
        buf575 = buf548; del buf548  # reuse
        buf601 = buf522; del buf522  # reuse
        generated_kernel_15(buf575, buf601, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_24, mul_283, q_embed_21, mul_285, k_embed_21, linear_149, view_65, value_states_21, attn_output_84], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf576 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf573, buf574, reinterpret_tensor(buf572, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf575, scale=0.08838834764831845, enable_gqa=True)
        buf577 = buf576[0]
        assert_tensor_metadata(buf577, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf577, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf576
        buf579 = reinterpret_tensor(buf573, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf573  # reuse
        generated_kernel_7(buf579, buf577, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg240_1, (2048, 2048), (2048, 1), 'input')
        buf580 = reinterpret_tensor(buf577, (71, 2048), (2048, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [transpose_87, attn_output_85, reshape_21, attn_output_87], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf579, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg240_1, (2048, 2048), (1, 2048), 0), out=buf580)
        del arg240_1
        assert_size_stride(arg241_1, (2048, ), (1, ), 'input')
        buf581 = reinterpret_tensor(buf579, (1, 71, 2048), (145408, 2048, 1), 0); del buf579  # reuse
        buf583 = buf563; del buf563  # reuse
        generated_kernel_16(buf581, buf583, buf528, buf534, buf553, buf560, buf580, arg241_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg241_1
        del buf528
        assert_size_stride(arg242_1, (6144, 2048), (2048, 1), 'input')
        buf584 = reinterpret_tensor(buf559, (71, 6144), (6144, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_258, pow_88, variance_86, add_177, rsqrt_87, hidden_states_259, to_183, hidden_states_260, linear_151], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf583, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg242_1, (2048, 6144), (1, 2048), 0), out=buf584)
        del arg242_1
        assert_size_stride(arg243_1, (6144, 2048), (2048, 1), 'input')
        buf585 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf583, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg243_1, (2048, 6144), (1, 2048), 0), out=buf585)
        del arg243_1
        buf586 = reinterpret_tensor(buf557, (1, 71, 6144), (436224, 6144, 1), 0); del buf557  # reuse
        generated_kernel_9(buf586, buf584, buf585, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg244_1, (2048, 6144), (6144, 1), 'input')
        buf587 = reinterpret_tensor(buf583, (71, 2048), (2048, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_288, down_proj_21], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf586, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg244_1, (6144, 2048), (1, 6144), 0), out=buf587)
        del arg244_1
        assert_size_stride(arg245_1, (2048, ), (1, ), 'input')
        buf589 = reinterpret_tensor(buf580, (1, 71, 2048), (145408, 2048, 1), 0); del buf580  # reuse
        generated_kernel_12(buf589, buf581, buf587, arg245_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg245_1
        assert_size_stride(arg246_1, (2048, 2048), (2048, 1), 'input')
        buf590 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf589, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg246_1, (2048, 2048), (1, 2048), 0), out=buf590)
        del arg246_1
        assert_size_stride(arg247_1, (128, ), (1, ), 'input')
        buf599 = reinterpret_tensor(buf553, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf553  # reuse
        generated_kernel_1(buf599, buf590, arg247_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg247_1
        assert_size_stride(arg248_1, (1024, 2048), (2048, 1), 'input')
        buf594 = reinterpret_tensor(buf574, (71, 1024), (1024, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf589, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg248_1, (2048, 1024), (1, 2048), 0), out=buf594)
        del arg248_1
        assert_size_stride(arg249_1, (128, ), (1, ), 'input')
        buf600 = reinterpret_tensor(buf572, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf572  # reuse
        generated_kernel_2(buf600, buf594, arg249_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg249_1
        assert_size_stride(arg250_1, (1024, 2048), (2048, 1), 'input')
        buf598 = buf594; del buf594  # reuse
        # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf589, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg250_1, (2048, 1024), (1, 2048), 0), out=buf598)
        del arg250_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_25, mul_296, q_embed_22, mul_298, k_embed_22, linear_156, view_68, value_states_22, attn_output_88], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf602 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf599, buf600, reinterpret_tensor(buf598, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf601, scale=0.08838834764831845, enable_gqa=True)
        buf603 = buf602[0]
        assert_tensor_metadata(buf603, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf603, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf602
        buf605 = reinterpret_tensor(buf599, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf599  # reuse
        generated_kernel_7(buf605, buf603, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg251_1, (2048, 2048), (2048, 1), 'input')
        buf606 = reinterpret_tensor(buf603, (71, 2048), (2048, 1), 0); del buf603  # reuse
        # Topologically Sorted Source Nodes: [transpose_91, attn_output_89, reshape_22, attn_output_91], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf605, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg251_1, (2048, 2048), (1, 2048), 0), out=buf606)
        del arg251_1
        assert_size_stride(arg252_1, (2048, ), (1, ), 'input')
        buf609 = reinterpret_tensor(buf605, (1, 71, 2048), (145408, 2048, 1), 0); del buf605  # reuse
        generated_kernel_13(buf609, buf581, buf587, buf606, arg252_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg252_1
        assert_size_stride(arg253_1, (6144, 2048), (2048, 1), 'input')
        buf610 = reinterpret_tensor(buf586, (71, 6144), (6144, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_272, linear_158], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf609, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 6144), (1, 2048), 0), out=buf610)
        del arg253_1
        assert_size_stride(arg254_1, (6144, 2048), (2048, 1), 'input')
        buf611 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [linear_159], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf609, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg254_1, (2048, 6144), (1, 2048), 0), out=buf611)
        del arg254_1
        buf612 = reinterpret_tensor(buf584, (1, 71, 6144), (436224, 6144, 1), 0); del buf584  # reuse
        generated_kernel_9(buf612, buf610, buf611, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg255_1, (2048, 6144), (6144, 1), 'input')
        buf613 = reinterpret_tensor(buf609, (71, 2048), (2048, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_301, down_proj_22], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf612, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg255_1, (6144, 2048), (1, 6144), 0), out=buf613)
        del arg255_1
        assert_size_stride(arg256_1, (2048, ), (1, ), 'input')
        buf616 = buf589; del buf589  # reuse
        generated_kernel_14(buf616, buf581, buf587, buf606, buf613, arg256_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg256_1
        assert_size_stride(arg257_1, (2048, 2048), (2048, 1), 'input')
        buf617 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [to_193, hidden_states_276, linear_161], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf616, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg257_1, (2048, 2048), (1, 2048), 0), out=buf617)
        del arg257_1
        assert_size_stride(arg258_1, (128, ), (1, ), 'input')
        buf626 = reinterpret_tensor(buf534, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf534  # reuse
        generated_kernel_1(buf626, buf617, arg258_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg258_1
        del buf617
        assert_size_stride(arg259_1, (1024, 2048), (2048, 1), 'input')
        buf621 = reinterpret_tensor(buf600, (71, 1024), (1024, 1), 0); del buf600  # reuse
        # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf616, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg259_1, (2048, 1024), (1, 2048), 0), out=buf621)
        del arg259_1
        assert_size_stride(arg260_1, (128, ), (1, ), 'input')
        buf627 = reinterpret_tensor(buf598, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf598  # reuse
        generated_kernel_2(buf627, buf621, arg260_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg260_1
        assert_size_stride(arg261_1, (1024, 2048), (2048, 1), 'input')
        buf625 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf616, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg261_1, (2048, 1024), (1, 2048), 0), out=buf625)
        del arg261_1
        buf628 = buf601; del buf601  # reuse
        buf654 = buf575; del buf575  # reuse
        generated_kernel_15(buf628, buf654, buf15, threads=[5041], error_buf_idx=3)
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_26, mul_309, q_embed_23, mul_311, k_embed_23, linear_163, view_71, value_states_23, attn_output_92], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf629 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf626, buf627, reinterpret_tensor(buf625, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf628, scale=0.08838834764831845, enable_gqa=True)
        buf630 = buf629[0]
        assert_tensor_metadata(buf630, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf630, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf629
        buf632 = reinterpret_tensor(buf626, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf626  # reuse
        generated_kernel_7(buf632, buf630, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg262_1, (2048, 2048), (2048, 1), 'input')
        buf633 = reinterpret_tensor(buf630, (71, 2048), (2048, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [transpose_95, attn_output_93, reshape_23, attn_output_95], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf632, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg262_1, (2048, 2048), (1, 2048), 0), out=buf633)
        del arg262_1
        assert_size_stride(arg263_1, (2048, ), (1, ), 'input')
        buf634 = reinterpret_tensor(buf632, (1, 71, 2048), (145408, 2048, 1), 0); del buf632  # reuse
        buf636 = buf616; del buf616  # reuse
        generated_kernel_16(buf634, buf636, buf581, buf587, buf606, buf613, buf633, arg263_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg263_1
        del buf581
        assert_size_stride(arg264_1, (6144, 2048), (2048, 1), 'input')
        buf637 = reinterpret_tensor(buf612, (71, 6144), (6144, 1), 0); del buf612  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_282, pow_96, variance_94, add_193, rsqrt_95, hidden_states_283, to_199, hidden_states_284, linear_165], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf636, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg264_1, (2048, 6144), (1, 2048), 0), out=buf637)
        del arg264_1
        assert_size_stride(arg265_1, (6144, 2048), (2048, 1), 'input')
        buf638 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf636, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg265_1, (2048, 6144), (1, 2048), 0), out=buf638)
        del arg265_1
        buf639 = reinterpret_tensor(buf610, (1, 71, 6144), (436224, 6144, 1), 0); del buf610  # reuse
        generated_kernel_9(buf639, buf637, buf638, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg266_1, (2048, 6144), (6144, 1), 'input')
        buf640 = reinterpret_tensor(buf636, (71, 2048), (2048, 1), 0); del buf636  # reuse
        # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_314, down_proj_23], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf639, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg266_1, (6144, 2048), (1, 6144), 0), out=buf640)
        del arg266_1
        assert_size_stride(arg267_1, (2048, ), (1, ), 'input')
        buf642 = reinterpret_tensor(buf633, (1, 71, 2048), (145408, 2048, 1), 0); del buf633  # reuse
        generated_kernel_12(buf642, buf634, buf640, arg267_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg267_1
        assert_size_stride(arg268_1, (2048, 2048), (2048, 1), 'input')
        buf643 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [linear_168], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf642, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg268_1, (2048, 2048), (1, 2048), 0), out=buf643)
        del arg268_1
        assert_size_stride(arg269_1, (128, ), (1, ), 'input')
        buf652 = reinterpret_tensor(buf606, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf606  # reuse
        generated_kernel_1(buf652, buf643, arg269_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg269_1
        assert_size_stride(arg270_1, (1024, 2048), (2048, 1), 'input')
        buf647 = reinterpret_tensor(buf627, (71, 1024), (1024, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf642, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg270_1, (2048, 1024), (1, 2048), 0), out=buf647)
        del arg270_1
        assert_size_stride(arg271_1, (128, ), (1, ), 'input')
        buf653 = reinterpret_tensor(buf625, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf625  # reuse
        generated_kernel_2(buf653, buf647, arg271_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg271_1
        assert_size_stride(arg272_1, (1024, 2048), (2048, 1), 'input')
        buf651 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf642, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg272_1, (2048, 1024), (1, 2048), 0), out=buf651)
        del arg272_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_27, mul_322, q_embed_24, mul_324, k_embed_24, linear_170, view_74, value_states_24, attn_output_96], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf655 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf652, buf653, reinterpret_tensor(buf651, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf654, scale=0.08838834764831845, enable_gqa=True)
        buf656 = buf655[0]
        assert_tensor_metadata(buf656, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf656, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf655
        buf658 = reinterpret_tensor(buf652, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf652  # reuse
        generated_kernel_7(buf658, buf656, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg273_1, (2048, 2048), (2048, 1), 'input')
        buf659 = reinterpret_tensor(buf656, (71, 2048), (2048, 1), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [transpose_99, attn_output_97, reshape_24, attn_output_99], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf658, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg273_1, (2048, 2048), (1, 2048), 0), out=buf659)
        del arg273_1
        assert_size_stride(arg274_1, (2048, ), (1, ), 'input')
        buf662 = reinterpret_tensor(buf658, (1, 71, 2048), (145408, 2048, 1), 0); del buf658  # reuse
        generated_kernel_13(buf662, buf634, buf640, buf659, arg274_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg274_1
        assert_size_stride(arg275_1, (6144, 2048), (2048, 1), 'input')
        buf663 = reinterpret_tensor(buf639, (71, 6144), (6144, 1), 0); del buf639  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_296, linear_172], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf662, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg275_1, (2048, 6144), (1, 2048), 0), out=buf663)
        del arg275_1
        assert_size_stride(arg276_1, (6144, 2048), (2048, 1), 'input')
        buf664 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [linear_173], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf662, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg276_1, (2048, 6144), (1, 2048), 0), out=buf664)
        del arg276_1
        buf665 = reinterpret_tensor(buf637, (1, 71, 6144), (436224, 6144, 1), 0); del buf637  # reuse
        generated_kernel_9(buf665, buf663, buf664, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg277_1, (2048, 6144), (6144, 1), 'input')
        buf666 = reinterpret_tensor(buf662, (71, 2048), (2048, 1), 0); del buf662  # reuse
        # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_327, down_proj_24], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf665, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg277_1, (6144, 2048), (1, 6144), 0), out=buf666)
        del arg277_1
        assert_size_stride(arg278_1, (2048, ), (1, ), 'input')
        buf669 = buf642; del buf642  # reuse
        generated_kernel_14(buf669, buf634, buf640, buf659, buf666, arg278_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg278_1
        assert_size_stride(arg279_1, (2048, 2048), (2048, 1), 'input')
        buf670 = buf643; del buf643  # reuse
        # Topologically Sorted Source Nodes: [to_209, hidden_states_300, linear_175], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf669, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg279_1, (2048, 2048), (1, 2048), 0), out=buf670)
        del arg279_1
        assert_size_stride(arg280_1, (128, ), (1, ), 'input')
        buf679 = reinterpret_tensor(buf587, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf587  # reuse
        generated_kernel_1(buf679, buf670, arg280_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg280_1
        del buf670
        assert_size_stride(arg281_1, (1024, 2048), (2048, 1), 'input')
        buf674 = reinterpret_tensor(buf653, (71, 1024), (1024, 1), 0); del buf653  # reuse
        # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf669, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg281_1, (2048, 1024), (1, 2048), 0), out=buf674)
        del arg281_1
        assert_size_stride(arg282_1, (128, ), (1, ), 'input')
        buf680 = reinterpret_tensor(buf651, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf651  # reuse
        generated_kernel_2(buf680, buf674, arg282_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg282_1
        assert_size_stride(arg283_1, (1024, 2048), (2048, 1), 'input')
        buf678 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [linear_177], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf669, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg283_1, (2048, 1024), (1, 2048), 0), out=buf678)
        del arg283_1
        buf681 = buf654; del buf654  # reuse
        buf707 = buf628; del buf628  # reuse
        buf734 = buf389; del buf389  # reuse
        generated_kernel_6(buf681, buf707, buf734, buf15, threads=[5041], error_buf_idx=4)
        del buf15
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_28, mul_335, q_embed_25, mul_337, k_embed_25, linear_177, view_77, value_states_25, attn_output_100], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf682 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf679, buf680, reinterpret_tensor(buf678, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf681, scale=0.08838834764831845, enable_gqa=True)
        del buf681
        buf683 = buf682[0]
        assert_tensor_metadata(buf683, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf683, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf682
        buf685 = reinterpret_tensor(buf679, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf679  # reuse
        generated_kernel_7(buf685, buf683, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg284_1, (2048, 2048), (2048, 1), 'input')
        buf686 = reinterpret_tensor(buf683, (71, 2048), (2048, 1), 0); del buf683  # reuse
        # Topologically Sorted Source Nodes: [transpose_103, attn_output_101, reshape_25, attn_output_103], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf685, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 2048), (1, 2048), 0), out=buf686)
        del arg284_1
        assert_size_stride(arg285_1, (2048, ), (1, ), 'input')
        buf687 = reinterpret_tensor(buf685, (1, 71, 2048), (145408, 2048, 1), 0); del buf685  # reuse
        buf689 = buf669; del buf669  # reuse
        generated_kernel_16(buf687, buf689, buf634, buf640, buf659, buf666, buf686, arg285_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg285_1
        del buf634
        assert_size_stride(arg286_1, (6144, 2048), (2048, 1), 'input')
        buf690 = reinterpret_tensor(buf665, (71, 6144), (6144, 1), 0); del buf665  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_306, pow_104, variance_102, add_209, rsqrt_103, hidden_states_307, to_215, hidden_states_308, linear_179], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf689, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg286_1, (2048, 6144), (1, 2048), 0), out=buf690)
        del arg286_1
        assert_size_stride(arg287_1, (6144, 2048), (2048, 1), 'input')
        buf691 = buf664; del buf664  # reuse
        # Topologically Sorted Source Nodes: [linear_180], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf689, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg287_1, (2048, 6144), (1, 2048), 0), out=buf691)
        del arg287_1
        buf692 = reinterpret_tensor(buf663, (1, 71, 6144), (436224, 6144, 1), 0); del buf663  # reuse
        generated_kernel_9(buf692, buf690, buf691, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg288_1, (2048, 6144), (6144, 1), 'input')
        buf693 = reinterpret_tensor(buf689, (71, 2048), (2048, 1), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_340, down_proj_25], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf692, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg288_1, (6144, 2048), (1, 6144), 0), out=buf693)
        del arg288_1
        assert_size_stride(arg289_1, (2048, ), (1, ), 'input')
        buf695 = reinterpret_tensor(buf686, (1, 71, 2048), (145408, 2048, 1), 0); del buf686  # reuse
        generated_kernel_12(buf695, buf687, buf693, arg289_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg289_1
        assert_size_stride(arg290_1, (2048, 2048), (2048, 1), 'input')
        buf696 = buf666; del buf666  # reuse
        # Topologically Sorted Source Nodes: [linear_182], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf695, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg290_1, (2048, 2048), (1, 2048), 0), out=buf696)
        del arg290_1
        assert_size_stride(arg291_1, (128, ), (1, ), 'input')
        buf705 = reinterpret_tensor(buf659, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf659  # reuse
        generated_kernel_1(buf705, buf696, arg291_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg291_1
        assert_size_stride(arg292_1, (1024, 2048), (2048, 1), 'input')
        buf700 = reinterpret_tensor(buf680, (71, 1024), (1024, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [linear_183], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf695, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg292_1, (2048, 1024), (1, 2048), 0), out=buf700)
        del arg292_1
        assert_size_stride(arg293_1, (128, ), (1, ), 'input')
        buf706 = reinterpret_tensor(buf678, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf678  # reuse
        generated_kernel_2(buf706, buf700, arg293_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg293_1
        assert_size_stride(arg294_1, (1024, 2048), (2048, 1), 'input')
        buf704 = buf700; del buf700  # reuse
        # Topologically Sorted Source Nodes: [linear_184], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf695, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg294_1, (2048, 1024), (1, 2048), 0), out=buf704)
        del arg294_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_29, mul_348, q_embed_26, mul_350, k_embed_26, linear_184, view_80, value_states_26, attn_output_104], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf708 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf705, buf706, reinterpret_tensor(buf704, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf707, scale=0.08838834764831845, enable_gqa=True)
        del buf707
        buf709 = buf708[0]
        assert_tensor_metadata(buf709, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf709, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf708
        buf711 = reinterpret_tensor(buf705, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf705  # reuse
        generated_kernel_7(buf711, buf709, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg295_1, (2048, 2048), (2048, 1), 'input')
        buf712 = reinterpret_tensor(buf709, (71, 2048), (2048, 1), 0); del buf709  # reuse
        # Topologically Sorted Source Nodes: [transpose_107, attn_output_105, reshape_26, attn_output_107], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf711, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg295_1, (2048, 2048), (1, 2048), 0), out=buf712)
        del arg295_1
        assert_size_stride(arg296_1, (2048, ), (1, ), 'input')
        buf715 = reinterpret_tensor(buf711, (1, 71, 2048), (145408, 2048, 1), 0); del buf711  # reuse
        generated_kernel_13(buf715, buf687, buf693, buf712, arg296_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=5)
        del arg296_1
        assert_size_stride(arg297_1, (6144, 2048), (2048, 1), 'input')
        buf716 = reinterpret_tensor(buf692, (71, 6144), (6144, 1), 0); del buf692  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_320, linear_186], Original ATen: [aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf715, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg297_1, (2048, 6144), (1, 2048), 0), out=buf716)
        del arg297_1
        assert_size_stride(arg298_1, (6144, 2048), (2048, 1), 'input')
        buf717 = buf691; del buf691  # reuse
        # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf715, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg298_1, (2048, 6144), (1, 2048), 0), out=buf717)
        del arg298_1
        buf718 = reinterpret_tensor(buf690, (1, 71, 6144), (436224, 6144, 1), 0); del buf690  # reuse
        generated_kernel_9(buf718, buf716, buf717, threads=[436224], error_buf_idx=3)
        assert_size_stride(arg299_1, (2048, 6144), (6144, 1), 'input')
        buf719 = reinterpret_tensor(buf715, (71, 2048), (2048, 1), 0); del buf715  # reuse
        # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_353, down_proj_26], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf718, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg299_1, (6144, 2048), (1, 6144), 0), out=buf719)
        del arg299_1
        assert_size_stride(arg300_1, (2048, ), (1, ), 'input')
        buf722 = buf695; del buf695  # reuse
        generated_kernel_14(buf722, buf687, buf693, buf712, buf719, arg300_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=6)
        del arg300_1
        assert_size_stride(arg301_1, (2048, 2048), (2048, 1), 'input')
        buf723 = buf696; del buf696  # reuse
        # Topologically Sorted Source Nodes: [to_225, hidden_states_324, linear_189], Original ATen: [aten._to_copy, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf722, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg301_1, (2048, 2048), (1, 2048), 0), out=buf723)
        del arg301_1
        assert_size_stride(arg302_1, (128, ), (1, ), 'input')
        buf732 = reinterpret_tensor(buf640, (1, 16, 71, 128), (145408, 9088, 128, 1), 0); del buf640  # reuse
        generated_kernel_1(buf732, buf723, arg302_1, arg2_1, threads=[1136, 128], group_size=[1, 128], error_buf_idx=4)
        del arg302_1
        del buf723
        assert_size_stride(arg303_1, (1024, 2048), (2048, 1), 'input')
        buf727 = reinterpret_tensor(buf706, (71, 1024), (1024, 1), 0); del buf706  # reuse
        # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf722, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg303_1, (2048, 1024), (1, 2048), 0), out=buf727)
        del arg303_1
        assert_size_stride(arg304_1, (128, ), (1, ), 'input')
        buf733 = reinterpret_tensor(buf704, (1, 8, 71, 128), (72704, 9088, 128, 1), 0); del buf704  # reuse
        generated_kernel_2(buf733, buf727, arg304_1, arg2_1, threads=[568, 128], group_size=[1, 128], error_buf_idx=4)
        del arg2_1
        del arg304_1
        assert_size_stride(arg305_1, (1024, 2048), (2048, 1), 'input')
        buf731 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf722, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg305_1, (2048, 1024), (1, 2048), 0), out=buf731)
        del arg305_1
        # Topologically Sorted Source Nodes: [position_ids, position_ids_1, getitem_7, float_1, freqs, emb, sin_1, sin_2, result, kv_arange, kv_indices, q_arange, q_indices, le, result_1, batch_arange, batch_indices, getitem_5, getitem_6, eq, result_2, attention_mask, sin_30, mul_361, q_embed_27, mul_363, k_embed_27, linear_191, view_83, value_states_27, attn_output_108], Original ATen: [aten.add, aten.unsqueeze, aten._to_copy, aten.mul, aten.cat, aten.new_ones, aten.le, aten.bitwise_and, aten.arange, aten.index, aten.eq, aten.expand, aten.linear, aten.view, aten.transpose, aten.scalar_tensor, aten.where, aten._scaled_dot_product_attention_math_for_mps]
        buf735 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf732, buf733, reinterpret_tensor(buf731, (1, 8, 71, 128), (72704, 128, 1024, 1), 0), buf734, scale=0.08838834764831845, enable_gqa=True)
        del buf731
        del buf733
        del buf734
        buf736 = buf735[0]
        assert_tensor_metadata(buf736, (1, 16, 71, 128), (145408, 9088, 128, 1), torch.bfloat16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf736, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf735
        buf738 = reinterpret_tensor(buf732, (1, 71, 16, 128), (145408, 2048, 128, 1), 0); del buf732  # reuse
        generated_kernel_7(buf738, buf736, threads=[145408], error_buf_idx=2)
        assert_size_stride(arg306_1, (2048, 2048), (2048, 1), 'input')
        buf739 = reinterpret_tensor(buf736, (71, 2048), (2048, 1), 0); del buf736  # reuse
        # Topologically Sorted Source Nodes: [transpose_111, attn_output_109, reshape_27, attn_output_111], Original ATen: [aten.transpose, aten.clone, aten.view, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf738, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg306_1, (2048, 2048), (1, 2048), 0), out=buf739)
        del arg306_1
        assert_size_stride(arg307_1, (2048, ), (1, ), 'input')
        buf740 = reinterpret_tensor(buf738, (1, 71, 2048), (145408, 2048, 1), 0); del buf738  # reuse
        buf742 = buf722; del buf722  # reuse
        generated_kernel_16(buf740, buf742, buf687, buf693, buf712, buf719, buf739, arg307_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=8)
        del arg307_1
        del buf687
        del buf693
        del buf712
        del buf719
        assert_size_stride(arg308_1, (6144, 2048), (2048, 1), 'input')
        buf743 = reinterpret_tensor(buf718, (71, 6144), (6144, 1), 0); del buf718  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_330, pow_112, variance_110, add_225, rsqrt_111, hidden_states_331, to_231, hidden_states_332, linear_193], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf742, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg308_1, (2048, 6144), (1, 2048), 0), out=buf743)
        del arg308_1
        assert_size_stride(arg309_1, (6144, 2048), (2048, 1), 'input')
        buf744 = buf717; del buf717  # reuse
        # Topologically Sorted Source Nodes: [linear_194], Original ATen: [aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf742, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg309_1, (2048, 6144), (1, 2048), 0), out=buf744)
        del arg309_1
        buf745 = reinterpret_tensor(buf716, (1, 71, 6144), (436224, 6144, 1), 0); del buf716  # reuse
        generated_kernel_9(buf745, buf743, buf744, threads=[436224], error_buf_idx=3)
        del buf743
        del buf744
        assert_size_stride(arg310_1, (2048, 6144), (6144, 1), 'input')
        buf746 = reinterpret_tensor(buf742, (71, 2048), (2048, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_366, down_proj_27], Original ATen: [aten.linear, aten.silu, aten.mul]
        extern_kernels.mm(reinterpret_tensor(buf745, (71, 6144), (6144, 1), 0), reinterpret_tensor(arg310_1, (6144, 2048), (1, 6144), 0), out=buf746)
        del arg310_1
        del buf745
        assert_size_stride(arg311_1, (2048, ), (1, ), 'input')
        buf748 = reinterpret_tensor(buf739, (1, 71, 2048), (145408, 2048, 1), 0); del buf739  # reuse
        generated_kernel_12(buf748, buf740, buf746, arg311_1, threads=[71, 1024], group_size=[1, 1024], error_buf_idx=4)
        del arg311_1
        del buf740
        del buf746
        buf749 = empty_strided((71, 151936), (151936, 1), device='mps', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [hid, linear_196], Original ATen: [aten.select, aten.linear]
        extern_kernels.mm(reinterpret_tensor(buf748, (71, 2048), (2048, 1), 0), reinterpret_tensor(arg1_1, (2048, 151936), (1, 2048), 0), out=buf749)
        del arg1_1
        return (reinterpret_tensor(buf748, (71, 2048), (2048, 1), 0), buf749, )

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
