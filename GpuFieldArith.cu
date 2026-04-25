// GpuFieldArith.cu - secp256k1 field arithmetic device functions
// (c) 2024, RetiredCoder (RC) — refactored for SASS register pressure reduction
//
// SASS OPTIMIZATION: MulModP and SqrModP are declared __noinline__ so the LTO
// compiler cannot expand them inline into KernelA.
//
// Why this matters:
//   MulModP:  buff[8] + tmp[5] + tmp2[2] + tmp3 = 128 bytes of local arrays
//   SqrModP:  buff[8] + mar[28] + tmp[5] + tmp2[2] + tmp3 = 352 bytes of local arrays
//
// When __forceinline__, ALL of these arrays are simultaneously live in KernelA's
// register frame alongside KernelA's own variables (x[4], y[4], loop state, etc.)
// → 124 registers → 1 block/SM → 16 warps → 33% occupancy
//
// As separate device functions with __noinline__:
//   - Each function has its own register frame
//   - KernelA only needs to keep live the pointer arguments across the call
//   - Expected KernelA registers: ~60-70 → 2 blocks/SM → 32 warps → 66% occupancy

#include "defs.h"
#include "RCGpuUtils.h"

// ---------------------------------------------------------------------------
// MulModP: res = val1 * val2  (mod secp256k1 prime P)
// Algorithm: schoolbook 256x256→512 bit multiply, then fast Barrett reduction
// using the special form of P = 2^256 - 2^32 - 977
// ---------------------------------------------------------------------------
__device__ __noinline__ void MulModP(u64 *res, u64 *val1, u64 *val2)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3;
//calc 512 bits
	mul_256_by_64(tmp, val1, val2[1]);
	mul_256_by_64(buff, val1, val2[0]);
	add_320_to_256(buff + 1, tmp);
	mul_256_by_64(tmp, val1, val2[2]);
	add_320_to_256(buff + 2, tmp);
	mul_256_by_64(tmp, val1, val2[3]);
	add_320_to_256(buff + 3, tmp);
//fast mod P
	mul_256_by_P0inv((u32*)tmp, (u32*)(buff + 4));
	add_cc_64(buff[0], buff[0], tmp[0]);
	addc_cc_64(buff[1], buff[1], tmp[1]);
	addc_cc_64(buff[2], buff[2], tmp[2]);
	addc_cc_64(buff[3], buff[3], tmp[3]);
	addc_64(tmp[4], tmp[4], 0ull);
//see mul_256_by_P0inv for details
	u32* t32 = (u32*)tmp;
	u32* a32 = (u32*)tmp2;
	u32* k = (u32*)&tmp3;
	mul_wide_32(tmp2[0], t32[8], P_INV32);
	mul_wide_32(tmp3, t32[9], P_INV32);
	add_cc_32(a32[1], a32[1], k[0]);
	addc_32(a32[2], k[1], 0); //we cannot get carry here for a32[3]
	add_cc_32(a32[1], a32[1], t32[8]);
	addc_cc_32(a32[2], a32[2], t32[9]);
	addc_32(a32[3], 0, 0);

	add_cc_64(res[0], buff[0], tmp2[0]);
	addc_cc_64(res[1], buff[1], tmp2[1]);
	addc_cc_64(res[2], buff[2], 0ull);
	addc_64(res[3], buff[3], 0ull);
}

// ---------------------------------------------------------------------------
// SqrModP: res = val * val  (mod secp256k1 prime P)
// Algorithm: optimized squaring using the symmetry of a^2 — computes only
// the upper triangle of the schoolbook product (28 half-products instead of
// 64), then doubles them and adds the diagonal terms.
// mar[28] is the most register-intensive array in the entire solver.
// ---------------------------------------------------------------------------
__device__ __noinline__ void SqrModP(u64* res, u64* val)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3, mm;
	u32* a = (u32*)val;
	u64 mar[28];
	u32* b32 = (u32*)buff;
	u32* m32 = (u32*)mar;
//calc 512 bits
	mul_wide_32(mar[0], a[1], a[0]); //ab
	mul_wide_32(mar[1], a[2], a[0]); //ac
	mul_wide_32(mar[2], a[3], a[0]); //ad
	mul_wide_32(mar[3], a[4], a[0]); //ae
	mul_wide_32(mar[4], a[5], a[0]); //af
	mul_wide_32(mar[5], a[6], a[0]); //ag
	mul_wide_32(mar[6], a[7], a[0]); //ah
	mul_wide_32(mar[7], a[2], a[1]); //bc
	mul_wide_32(mar[8], a[3], a[1]); //bd
	mul_wide_32(mar[9], a[4], a[1]); //be
	mul_wide_32(mar[10], a[5], a[1]); //bf
	mul_wide_32(mar[11], a[6], a[1]); //bg
	mul_wide_32(mar[12], a[7], a[1]); //bh
	mul_wide_32(mar[13], a[3], a[2]); //cd
	mul_wide_32(mar[14], a[4], a[2]); //ce
	mul_wide_32(mar[15], a[5], a[2]); //cf
	mul_wide_32(mar[16], a[6], a[2]); //cg
	mul_wide_32(mar[17], a[7], a[2]); //ch
	mul_wide_32(mar[18], a[4], a[3]); //de
	mul_wide_32(mar[19], a[5], a[3]); //df
	mul_wide_32(mar[20], a[6], a[3]); //dg
	mul_wide_32(mar[21], a[7], a[3]); //dh
	mul_wide_32(mar[22], a[5], a[4]); //ef
	mul_wide_32(mar[23], a[6], a[4]); //eg
	mul_wide_32(mar[24], a[7], a[4]); //eh
	mul_wide_32(mar[25], a[6], a[5]); //fg
	mul_wide_32(mar[26], a[7], a[5]); //fh
	mul_wide_32(mar[27], a[7], a[6]); //gh
//a
	mul_wide_32(buff[0], a[0], a[0]); //aa
	add_cc_32(b32[1], b32[1], m32[0]);
	addc_cc_32(b32[2], m32[1], m32[2]);
	addc_cc_32(b32[3], m32[3], m32[4]);
	addc_cc_32(b32[4], m32[5], m32[6]);
	addc_cc_32(b32[5], m32[7], m32[8]);
	addc_cc_32(b32[6], m32[9], m32[10]);
	addc_cc_32(b32[7], m32[11], m32[12]);
	addc_cc_32(b32[8], m32[13], 0);
	b32[9] = 0;
//b+
	mul_wide_32(mm, a[1], a[1]); //bb
	add_320_to_256s(b32 + 1, mar[0], mm, mar[7], mar[8], mar[9], mar[10], mar[11], mar[12]);
	mul_wide_32(mm, a[2], a[2]); //cc
	add_320_to_256s(b32 + 2, mar[1], mar[7], mm, mar[13], mar[14], mar[15], mar[16], mar[17]);
	mul_wide_32(mm, a[3], a[3]); //dd
	add_320_to_256s(b32 + 3, mar[2], mar[8], mar[13], mm, mar[18], mar[19], mar[20], mar[21]);
	mul_wide_32(mm, a[4], a[4]); //ee
	add_320_to_256s(b32 + 4, mar[3], mar[9], mar[14], mar[18], mm, mar[22], mar[23], mar[24]);
	mul_wide_32(mm, a[5], a[5]); //ff
	add_320_to_256s(b32 + 5, mar[4], mar[10], mar[15], mar[19], mar[22], mm, mar[25], mar[26]);
	mul_wide_32(mm, a[6], a[6]); //gg
	add_320_to_256s(b32 + 6, mar[5], mar[11], mar[16], mar[20], mar[23], mar[25], mm, mar[27]);
	mul_wide_32(mm, a[7], a[7]); //hh
	add_320_to_256s(b32 + 7, mar[6], mar[12], mar[17], mar[21], mar[24], mar[26], mar[27], mm);
//fast mod P
	mul_256_by_P0inv((u32*)tmp, (u32*)(buff + 4));
	add_cc_64(buff[0], buff[0], tmp[0]);
	addc_cc_64(buff[1], buff[1], tmp[1]);
	addc_cc_64(buff[2], buff[2], tmp[2]);
	addc_cc_64(buff[3], buff[3], tmp[3]);
	addc_64(tmp[4], tmp[4], 0ull);
//see mul_256_by_P0inv for details
	u32* t32 = (u32*)tmp;
	u32* a32 = (u32*)tmp2;
	u32* k = (u32*)&tmp3;
	mul_wide_32(tmp2[0], t32[8], P_INV32);
	mul_wide_32(tmp3, t32[9], P_INV32);
	add_cc_32(a32[1], a32[1], k[0]);
	addc_32(a32[2], k[1], 0); //we cannot get carry here for a32[3]
	add_cc_32(a32[1], a32[1], t32[8]);
	addc_cc_32(a32[2], a32[2], t32[9]);
	addc_32(a32[3], 0, 0);

	add_cc_64(res[0], buff[0], tmp2[0]);
	addc_cc_64(res[1], buff[1], tmp2[1]);
	addc_cc_64(res[2], buff[2], 0ull);
	addc_64(res[3], buff[3], 0ull);
}
