// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC
#pragma once


//PTX asm
//"volatile" is important
#define add_64(res, a, b)				asm volatile ("add.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define add_cc_64(res, a, b)			asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define addc_64(res, a, b)				asm volatile ("addc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define addc_cc_64(res, a, b)			asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );

#define add_32(res, a, b)				asm volatile ("add.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define add_cc_32(res, a, b)			asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define addc_32(res, a, b)				asm volatile ("addc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define addc_cc_32(res, a, b)			asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );

#define sub_64(res, a, b)				asm volatile ("sub.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define sub_cc_64(res, a, b)			asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b) );
#define subc_cc_64(res, a, b)			asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define subc_64(res, a, b)				asm volatile ("subc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));

#define sub_32(res, a, b)				asm volatile ("sub.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b) );
#define sub_cc_32(res, a, b)			asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b) );
#define subc_cc_32(res, a, b)			asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define subc_32(res, a, b)				asm volatile ("subc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));

#define mul_lo_64(res, a, b)			asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define mul_hi_64(res, a, b)			asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define mad_lo_64(res, a, b, c)			asm volatile ("mad.lo.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define mad_hi_64(res, a, b, c)			asm volatile ("mad.hi.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define mad_lo_cc_64(res, a, b, c)		asm volatile ("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define mad_hi_cc_64(res, a, b, c)		asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define madc_lo_64(res, a, b, c)		asm volatile ("madc.lo.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c));
#define madc_hi_64(res, a, b, c)		asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c));
#define madc_lo_cc_64(res, a, b, c)		asm volatile ("madc.lo.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define madc_hi_cc_64(res, a, b, c)		asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );

#define mul_lo_32(res, a, b)			asm volatile ("mul.lo.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define mul_hi_32(res, a, b)			asm volatile ("mul.hi.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define mad_lo_32(res, a, b, c)			asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define mad_hi_32(res, a, b, c)			asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define mad_lo_cc_32(res, a, b, c)		asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define mad_hi_cc_32(res, a, b, c)		asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_lo_32(res, a, b, c)		asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_hi_32(res, a, b, c)		asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_lo_cc_32(res, a, b, c)		asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_hi_cc_32(res, a, b, c)		asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));

#define mul_wide_32(res, a, b)			asm volatile ("mul.wide.u32 %0, %1, %2;" : "=l"(res) : "r"(a), "r"(b));
#define mad_wide_32(res,a,b,c)			asm volatile ("mad.wide.u32 %0, %1, %2, %3;" : "=l"(res) : "r"(a), "r"(b), "l"(c) );

#define st_cs_v4_b32(addr,val)			asm volatile("st.cs.global.v4.b32 [%0], {%1, %2, %3, %4};\n":: "l"(addr), "r"((val).x), "r"((val).y), "r"((val).z), "r"((val).w));


//P-related constants
#define P_0			0xFFFFFFFEFFFFFC2Full
#define P_123		0xFFFFFFFFFFFFFFFFull
#define P_INV32		0x000003D1

#define Add192to192(res, val) { \
  add_cc_64((res)[0], (res)[0], (val)[0]); \
  addc_cc_64((res)[1], (res)[1], (val)[1]); \
  addc_64((res)[2], (res)[2], (val)[2]); }

#define Sub192from192(res, val) { \
  sub_cc_64((res)[0], (res)[0], (val)[0]); \
  subc_cc_64((res)[1], (res)[1], (val)[1]); \
  subc_64((res)[2], (res)[2], (val)[2]); }

#define Copy_int4_x2(dst, src) {\
  ((int4*)(dst))[0] = ((int4*)(src))[0]; \
  ((int4*)(dst))[1] = ((int4*)(src))[1]; }

#define Copy_u64_x4(dst, src) {\
  ((u64*)(dst))[0] = ((u64*)(src))[0]; \
  ((u64*)(dst))[1] = ((u64*)(src))[1]; \
  ((u64*)(dst))[2] = ((u64*)(src))[2]; \
  ((u64*)(dst))[3] = ((u64*)(src))[3]; }

__device__ __forceinline__ void NegModP(u64* res)
{
	sub_cc_64(res[0], P_0, res[0]);
	subc_cc_64(res[1], P_123, res[1]);
	subc_cc_64(res[2], P_123, res[2]);
	subc_64(res[3], P_123, res[3]);
}

__device__ __forceinline__ void SubModP(u64* res, u64* val1, u64* val2)
{
	sub_cc_64(res[0], val1[0], val2[0]);
    subc_cc_64(res[1], val1[1], val2[1]);
    subc_cc_64(res[2], val1[2], val2[2]);
    subc_cc_64(res[3], val1[3], val2[3]);
    u32 carry;
    subc_32(carry, 0, 0);
    if (carry)
    {
		add_cc_64(res[0], res[0], P_0);
		addc_cc_64(res[1], res[1], P_123);
		addc_cc_64(res[2], res[2], P_123);
		addc_64(res[3], res[3], P_123);
    }
}

__device__ __forceinline__ void AddModP(u64* res, u64* val1, u64* val2)
{
	u64 tmp[4];
	u32 carry;
	add_cc_64(tmp[0], val1[0], val2[0]);
	addc_cc_64(tmp[1], val1[1], val2[1]);
	addc_cc_64(tmp[2], val1[2], val2[2]);
	addc_cc_64(tmp[3], val1[3], val2[3]);
	addc_32(carry, 0, 0);
	Copy_u64_x4(res, tmp);

	sub_cc_64(res[0], res[0], P_0);
	subc_cc_64(res[1], res[1], P_123);
	subc_cc_64(res[2], res[2], P_123);
	subc_cc_64(res[3], res[3], P_123);
	subc_cc_32(carry, carry, 0);
	subc_32(carry, 0, 0);
	if (carry)
		Copy_u64_x4(res, tmp);
}

__device__ __forceinline__ void add_320_to_256(u64* res, u64* val)
{
	add_cc_64(res[0], res[0], val[0]);
	addc_cc_64(res[1], res[1], val[1]);
	addc_cc_64(res[2], res[2], val[2]);
	addc_cc_64(res[3], res[3], val[3]);
	addc_64(res[4], val[4], 0ull);
}

//mul 256bit by 0x1000003D1
__device__ __forceinline__ void mul_256_by_P0inv(u32* res, u32* val)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	mul_wide_32(*(u64*)res, val[0], P_INV32);
	mul_wide_32(tmp64[0], val[1], P_INV32);
	mul_wide_32(tmp64[1], val[2], P_INV32);
	mul_wide_32(tmp64[2], val[3], P_INV32);
	mul_wide_32(tmp64[3], val[4], P_INV32);
	mul_wide_32(tmp64[4], val[5], P_INV32);
	mul_wide_32(tmp64[5], val[6], P_INV32);
	mul_wide_32(tmp64[6], val[7], P_INV32);

	add_cc_32(res[1], res[1], tmp[0]);
	addc_cc_32(res[2], tmp[1], tmp[2]);
	addc_cc_32(res[3], tmp[3], tmp[4]);
	addc_cc_32(res[4], tmp[5], tmp[6]);
	addc_cc_32(res[5], tmp[7], tmp[8]);
	addc_cc_32(res[6], tmp[9], tmp[10]);
	addc_cc_32(res[7], tmp[11], tmp[12]);
	addc_32(res[8], tmp[13], 0); //t[13] cannot be MAX_UINT so we wont have carry here for r[9]

	add_cc_32(res[1], res[1], val[0]);
	addc_cc_32(res[2], res[2], val[1]);
	addc_cc_32(res[3], res[3], val[2]);
	addc_cc_32(res[4], res[4], val[3]);
	addc_cc_32(res[5], res[5], val[4]);
	addc_cc_32(res[6], res[6], val[5]);
	addc_cc_32(res[7], res[7], val[6]);
	addc_cc_32(res[8], res[8], val[7]);
	addc_32(res[9], 0, 0);
}

//mul 256bit by 64bit
__device__ __forceinline__ void mul_256_by_64(u64* res, u64* val256, u64 val64)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	u32* rs = (u32*)res;
	u32* a = (u32*)val256;
	u32* b = (u32*)&val64;

	mul_wide_32(res[0], a[0], b[0]);
	mul_wide_32(tmp64[0], a[1], b[0]);
	mul_wide_32(tmp64[1], a[2], b[0]);
	mul_wide_32(tmp64[2], a[3], b[0]);
	mul_wide_32(tmp64[3], a[4], b[0]);
	mul_wide_32(tmp64[4], a[5], b[0]);
	mul_wide_32(tmp64[5], a[6], b[0]);
	mul_wide_32(tmp64[6], a[7], b[0]);

	add_cc_32(rs[1], rs[1], tmp[0]);
	addc_cc_32(rs[2], tmp[1], tmp[2]);
	addc_cc_32(rs[3], tmp[3], tmp[4]);
	addc_cc_32(rs[4], tmp[5], tmp[6]);
	addc_cc_32(rs[5], tmp[7], tmp[8]);
	addc_cc_32(rs[6], tmp[9], tmp[10]);
	addc_cc_32(rs[7], tmp[11], tmp[12]);
	addc_32(rs[8], tmp[13], 0); //we cannot get carry here for rs[9] because mul 8+1=9 words, rs[9] is 10th word

	u64 kk[7];
	u32* k = (u32*)kk;
	mul_wide_32(kk[0], a[0], b[1]);
	mul_wide_32(tmp64[0], a[1], b[1]);
	mul_wide_32(tmp64[1], a[2], b[1]);
	mul_wide_32(tmp64[2], a[3], b[1]);
	mul_wide_32(tmp64[3], a[4], b[1]);
	mul_wide_32(tmp64[4], a[5], b[1]);
	mul_wide_32(tmp64[5], a[6], b[1]);
	mul_wide_32(tmp64[6], a[7], b[1]);

	add_cc_32(k[1], k[1], tmp[0]);
	addc_cc_32(k[2], tmp[1], tmp[2]);
	addc_cc_32(k[3], tmp[3], tmp[4]);
	addc_cc_32(k[4], tmp[5], tmp[6]);
	addc_cc_32(k[5], tmp[7], tmp[8]);
	addc_cc_32(k[6], tmp[9], tmp[10]);
	addc_cc_32(k[7], tmp[11], tmp[12]);
	addc_32(k[8], tmp[13], 0); //we cannot get carry here for k[9] because mul 8+1=9 words, k[9] is 10th word

	add_cc_32(rs[1], rs[1], k[0]);
	addc_cc_32(rs[2], rs[2], k[1]);
	addc_cc_32(rs[3], rs[3], k[2]);
	addc_cc_32(rs[4], rs[4], k[3]);
	addc_cc_32(rs[5], rs[5], k[4]);
	addc_cc_32(rs[6], rs[6], k[5]);
	addc_cc_32(rs[7], rs[7], k[6]);
	addc_cc_32(rs[8], rs[8], k[7]);
	addc_32(rs[9], k[8], 0);
}

__device__ __forceinline__ void MulModP(u64 *res, u64 *val1, u64 *val2)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3;
	// Unpack 256-bit inputs into 8x u32 limbs
	u32 a0,a1,a2,a3,a4,a5,a6,a7;
	u32 b0,b1,b2,b3,b4,b5,b6,b7;
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(a0),"=r"(a1):"l"(val1[0]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(a2),"=r"(a3):"l"(val1[1]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(a4),"=r"(a5):"l"(val1[2]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(a6),"=r"(a7):"l"(val1[3]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(b0),"=r"(b1):"l"(val2[0]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(b2),"=r"(b3):"l"(val2[1]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(b4),"=r"(b5):"l"(val2[2]));
	asm volatile("mov.b64 {%0,%1},%2;":"=r"(b6),"=r"(b7):"l"(val2[3]));
	// 8-row single-tile schedule: 8 independent IMAD.WIDE per row before accumulation.
	// Gap from t0 issue to first read = 7 instructions > SM86 IMAD.WIDE latency (~6 cyc).
	// Eliminates u0..u7 (saves 16 u32 PTX regs): 44 internal regs vs 60 previously.
	// t0..t7(16) + r0..r15(16) + k0..k8(9) + lo/hi/ph(3) = 44 u32 PTX-internal regs.
	asm volatile(
		"{\n\t"
		".reg .u64 t0,t1,t2,t3,t4,t5,t6,t7;\n\t"
		".reg .u32 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;\n\t"
		".reg .u32 k0,k1,k2,k3,k4,k5,k6,k7,k8;\n\t"
		".reg .u32 lo,hi,ph;\n\t"

		// ── Row 0: a0×b[0..7] — init r0..r8 ──────────────────────────────
		"mul.wide.u32 t0,%8,%16;\n\t"  "mul.wide.u32 t1,%8,%17;\n\t"
		"mul.wide.u32 t2,%8,%18;\n\t"  "mul.wide.u32 t3,%8,%19;\n\t"
		"mul.wide.u32 t4,%8,%20;\n\t"  "mul.wide.u32 t5,%8,%21;\n\t"
		"mul.wide.u32 t6,%8,%22;\n\t"  "mul.wide.u32 t7,%8,%23;\n\t"
		"mov.b64 {r0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  r1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 r2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 r3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 r4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 r5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 r6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 r7,hi,lo;\n\t" "addc.u32 r8,ph,0;\n\t"

		// ── Row 1: a1×b[0..7] — accumulate at offset 1 into r1..r9 ───────
		"mul.wide.u32 t0,%9,%16;\n\t"  "mul.wide.u32 t1,%9,%17;\n\t"
		"mul.wide.u32 t2,%9,%18;\n\t"  "mul.wide.u32 t3,%9,%19;\n\t"
		"mul.wide.u32 t4,%9,%20;\n\t"  "mul.wide.u32 t5,%9,%21;\n\t"
		"mul.wide.u32 t6,%9,%22;\n\t"  "mul.wide.u32 t7,%9,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r1,r1,k0;\n\t" "addc.cc.u32 r2,r2,k1;\n\t" "addc.cc.u32 r3,r3,k2;\n\t"
		"addc.cc.u32 r4,r4,k3;\n\t" "addc.cc.u32 r5,r5,k4;\n\t" "addc.cc.u32 r6,r6,k5;\n\t"
		"addc.cc.u32 r7,r7,k6;\n\t" "addc.cc.u32 r8,r8,k7;\n\t" "addc.u32 r9,k8,0;\n\t"

		// ── Row 2: a2×b[0..7] — accumulate at offset 2 into r2..r10 ──────
		"mul.wide.u32 t0,%10,%16;\n\t" "mul.wide.u32 t1,%10,%17;\n\t"
		"mul.wide.u32 t2,%10,%18;\n\t" "mul.wide.u32 t3,%10,%19;\n\t"
		"mul.wide.u32 t4,%10,%20;\n\t" "mul.wide.u32 t5,%10,%21;\n\t"
		"mul.wide.u32 t6,%10,%22;\n\t" "mul.wide.u32 t7,%10,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r2,r2,k0;\n\t" "addc.cc.u32 r3,r3,k1;\n\t" "addc.cc.u32 r4,r4,k2;\n\t"
		"addc.cc.u32 r5,r5,k3;\n\t" "addc.cc.u32 r6,r6,k4;\n\t" "addc.cc.u32 r7,r7,k5;\n\t"
		"addc.cc.u32 r8,r8,k6;\n\t" "addc.cc.u32 r9,r9,k7;\n\t" "addc.u32 r10,k8,0;\n\t"

		// ── Row 3: a3×b[0..7] — accumulate at offset 3 into r3..r11 ──────
		"mul.wide.u32 t0,%11,%16;\n\t" "mul.wide.u32 t1,%11,%17;\n\t"
		"mul.wide.u32 t2,%11,%18;\n\t" "mul.wide.u32 t3,%11,%19;\n\t"
		"mul.wide.u32 t4,%11,%20;\n\t" "mul.wide.u32 t5,%11,%21;\n\t"
		"mul.wide.u32 t6,%11,%22;\n\t" "mul.wide.u32 t7,%11,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r3,r3,k0;\n\t" "addc.cc.u32 r4,r4,k1;\n\t" "addc.cc.u32 r5,r5,k2;\n\t"
		"addc.cc.u32 r6,r6,k3;\n\t" "addc.cc.u32 r7,r7,k4;\n\t" "addc.cc.u32 r8,r8,k5;\n\t"
		"addc.cc.u32 r9,r9,k6;\n\t" "addc.cc.u32 r10,r10,k7;\n\t" "addc.u32 r11,k8,0;\n\t"

		// ── Row 4: a4×b[0..7] — accumulate at offset 4 into r4..r12 ──────
		"mul.wide.u32 t0,%12,%16;\n\t" "mul.wide.u32 t1,%12,%17;\n\t"
		"mul.wide.u32 t2,%12,%18;\n\t" "mul.wide.u32 t3,%12,%19;\n\t"
		"mul.wide.u32 t4,%12,%20;\n\t" "mul.wide.u32 t5,%12,%21;\n\t"
		"mul.wide.u32 t6,%12,%22;\n\t" "mul.wide.u32 t7,%12,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r4,r4,k0;\n\t" "addc.cc.u32 r5,r5,k1;\n\t" "addc.cc.u32 r6,r6,k2;\n\t"
		"addc.cc.u32 r7,r7,k3;\n\t" "addc.cc.u32 r8,r8,k4;\n\t" "addc.cc.u32 r9,r9,k5;\n\t"
		"addc.cc.u32 r10,r10,k6;\n\t" "addc.cc.u32 r11,r11,k7;\n\t" "addc.u32 r12,k8,0;\n\t"

		// ── Row 5: a5×b[0..7] — accumulate at offset 5 into r5..r13 ──────
		"mul.wide.u32 t0,%13,%16;\n\t" "mul.wide.u32 t1,%13,%17;\n\t"
		"mul.wide.u32 t2,%13,%18;\n\t" "mul.wide.u32 t3,%13,%19;\n\t"
		"mul.wide.u32 t4,%13,%20;\n\t" "mul.wide.u32 t5,%13,%21;\n\t"
		"mul.wide.u32 t6,%13,%22;\n\t" "mul.wide.u32 t7,%13,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r5,r5,k0;\n\t" "addc.cc.u32 r6,r6,k1;\n\t" "addc.cc.u32 r7,r7,k2;\n\t"
		"addc.cc.u32 r8,r8,k3;\n\t" "addc.cc.u32 r9,r9,k4;\n\t" "addc.cc.u32 r10,r10,k5;\n\t"
		"addc.cc.u32 r11,r11,k6;\n\t" "addc.cc.u32 r12,r12,k7;\n\t" "addc.u32 r13,k8,0;\n\t"

		// ── Row 6: a6×b[0..7] — accumulate at offset 6 into r6..r14 ──────
		"mul.wide.u32 t0,%14,%16;\n\t" "mul.wide.u32 t1,%14,%17;\n\t"
		"mul.wide.u32 t2,%14,%18;\n\t" "mul.wide.u32 t3,%14,%19;\n\t"
		"mul.wide.u32 t4,%14,%20;\n\t" "mul.wide.u32 t5,%14,%21;\n\t"
		"mul.wide.u32 t6,%14,%22;\n\t" "mul.wide.u32 t7,%14,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r6,r6,k0;\n\t" "addc.cc.u32 r7,r7,k1;\n\t" "addc.cc.u32 r8,r8,k2;\n\t"
		"addc.cc.u32 r9,r9,k3;\n\t" "addc.cc.u32 r10,r10,k4;\n\t" "addc.cc.u32 r11,r11,k5;\n\t"
		"addc.cc.u32 r12,r12,k6;\n\t" "addc.cc.u32 r13,r13,k7;\n\t" "addc.u32 r14,k8,0;\n\t"

		// ── Row 7: a7×b[0..7] — accumulate at offset 7 into r7..r15 ──────
		"mul.wide.u32 t0,%15,%16;\n\t" "mul.wide.u32 t1,%15,%17;\n\t"
		"mul.wide.u32 t2,%15,%18;\n\t" "mul.wide.u32 t3,%15,%19;\n\t"
		"mul.wide.u32 t4,%15,%20;\n\t" "mul.wide.u32 t5,%15,%21;\n\t"
		"mul.wide.u32 t6,%15,%22;\n\t" "mul.wide.u32 t7,%15,%23;\n\t"
		"mov.b64 {k0,hi},t0;\n\t"
		"mov.b64 {lo,ph},t1;\n\t" "add.cc.u32  k1,hi,lo;\n\t"
		"mov.b64 {lo,hi},t2;\n\t" "addc.cc.u32 k2,ph,lo;\n\t"
		"mov.b64 {lo,ph},t3;\n\t" "addc.cc.u32 k3,hi,lo;\n\t"
		"mov.b64 {lo,hi},t4;\n\t" "addc.cc.u32 k4,ph,lo;\n\t"
		"mov.b64 {lo,ph},t5;\n\t" "addc.cc.u32 k5,hi,lo;\n\t"
		"mov.b64 {lo,hi},t6;\n\t" "addc.cc.u32 k6,ph,lo;\n\t"
		"mov.b64 {lo,ph},t7;\n\t" "addc.cc.u32 k7,hi,lo;\n\t" "addc.u32 k8,ph,0;\n\t"
		"add.cc.u32 r7,r7,k0;\n\t" "addc.cc.u32 r8,r8,k1;\n\t" "addc.cc.u32 r9,r9,k2;\n\t"
		"addc.cc.u32 r10,r10,k3;\n\t" "addc.cc.u32 r11,r11,k4;\n\t" "addc.cc.u32 r12,r12,k5;\n\t"
		"addc.cc.u32 r13,r13,k6;\n\t" "addc.cc.u32 r14,r14,k7;\n\t" "addc.u32 r15,k8,0;\n\t"

		"mov.b64 %0,{r0,r1};\n\t" "mov.b64 %1,{r2,r3};\n\t"
		"mov.b64 %2,{r4,r5};\n\t" "mov.b64 %3,{r6,r7};\n\t"
		"mov.b64 %4,{r8,r9};\n\t" "mov.b64 %5,{r10,r11};\n\t"
		"mov.b64 %6,{r12,r13};\n\t" "mov.b64 %7,{r14,r15};\n\t"
		"}\n"
		: "=l"(buff[0]),"=l"(buff[1]),"=l"(buff[2]),"=l"(buff[3]),
		  "=l"(buff[4]),"=l"(buff[5]),"=l"(buff[6]),"=l"(buff[7])
		: "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6),"r"(a7),
		  "r"(b0),"r"(b1),"r"(b2),"r"(b3),"r"(b4),"r"(b5),"r"(b6),"r"(b7)
	);
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

__device__ __forceinline__ void add_320_to_256s(u32* res, u64 _v1, u64 _v2, u64 _v3, u64 _v4, u64 _v5, u64 _v6, u64 _v7, u64 _v8)
{
	u32* v1 = (u32*)&_v1;
	u32* v2 = (u32*)&_v2;
	u32* v3 = (u32*)&_v3;
	u32* v4 = (u32*)&_v4;
	u32* v5 = (u32*)&_v5;
	u32* v6 = (u32*)&_v6;
	u32* v7 = (u32*)&_v7;
	u32* v8 = (u32*)&_v8;

	add_cc_32(res[0], res[0], v1[0]);
	addc_cc_32(res[1], res[1], v1[1]);
	addc_cc_32(res[2], res[2], v3[0]);
	addc_cc_32(res[3], res[3], v3[1]);
	addc_cc_32(res[4], res[4], v5[0]);
	addc_cc_32(res[5], res[5], v5[1]);
	addc_cc_32(res[6], res[6], v7[0]);
	addc_cc_32(res[7], res[7], v7[1]);
	addc_32(res[8], res[8], 0);

	add_cc_32(res[1], res[1], v2[0]);
	addc_cc_32(res[2], res[2], v2[1]);
	addc_cc_32(res[3], res[3], v4[0]);
	addc_cc_32(res[4], res[4], v4[1]);
	addc_cc_32(res[5], res[5], v6[0]);
	addc_cc_32(res[6], res[6], v6[1]);
	addc_cc_32(res[7], res[7], v8[0]);
	addc_cc_32(res[8], res[8], v8[1]);
	addc_32(res[9], 0, 0);
}

__device__ __forceinline__ void SqrModP(u64* res, u64* val)
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

__device__ __forceinline__ void add_288(u32* res, u32* val1, u32* val2)
{
	add_cc_32(res[0], val1[0], val2[0]);
	addc_cc_32(res[1], val1[1], val2[1]);
	addc_cc_32(res[2], val1[2], val2[2]);
	addc_cc_32(res[3], val1[3], val2[3]);
	addc_cc_32(res[4], val1[4], val2[4]);
	addc_cc_32(res[5], val1[5], val2[5]);
	addc_cc_32(res[6], val1[6], val2[6]);
	addc_cc_32(res[7], val1[7], val2[7]);
	addc_32(res[8], val1[8], val2[8]);
}

__device__ __forceinline__ void neg_288(u32* res)
{
	sub_cc_32(res[0], 0, res[0]);
	subc_cc_32(res[1], 0, res[1]);
	subc_cc_32(res[2], 0, res[2]);
	subc_cc_32(res[3], 0, res[3]);
	subc_cc_32(res[4], 0, res[4]);
	subc_cc_32(res[5], 0, res[5]);
	subc_cc_32(res[6], 0, res[6]);
	subc_cc_32(res[7], 0, res[7]);
	subc_32(res[8], 0, res[8]);
}

__device__ __forceinline__ void mul_288_by_i32(u32* res, u32* val288, int ival32)
{
	u32 val32 = abs(ival32);
	u64 tmp64[4];
	u32* tmp = (u32*)tmp64;
	u64* r32 = (u64*)res; 
	mul_wide_32(r32[0], val288[0], val32);
	mul_wide_32(r32[1], val288[2], val32);
	mul_wide_32(r32[2], val288[4], val32);
	mul_wide_32(r32[3], val288[6], val32);
	mul_wide_32(tmp64[0], val288[1], val32);
	mul_wide_32(tmp64[1], val288[3], val32);
	mul_wide_32(tmp64[2], val288[5], val32);
	mul_wide_32(tmp64[3], val288[7], val32);

	add_cc_32(res[1], res[1], tmp[0]);
	addc_cc_32(res[2], res[2], tmp[1]);
	addc_cc_32(res[3], res[3], tmp[2]);
	addc_cc_32(res[4], res[4], tmp[3]);
	addc_cc_32(res[5], res[5], tmp[4]);
	addc_cc_32(res[6], res[6], tmp[5]);
	addc_cc_32(res[7], res[7], tmp[6]);
	madc_lo_32(res[8], val288[8], val32, tmp[7]);

	if (ival32 < 0)
		neg_288(res);
}

__device__ __forceinline__ void set_288_i32(u32* res, int val)
{
	res[0] = val;
	res[1] = (val < 0) ? 0xFFFFFFFF : 0;
	res[2] = res[1];
	res[3] = res[1];
	res[4] = res[1];
	res[5] = res[1];
	res[6] = res[1];
	res[7] = res[1];
	res[8] = res[1];
}

//mul P by 32bit, get 288bit result
__device__ __forceinline__ void mul_P_by_32(u32* res, u32 val)
{
	__align__(8) u32 tmp[3];
	mul_wide_32(*(u64*)tmp, val, P_INV32);
	add_cc_32(tmp[1], tmp[1], val);
	addc_32(tmp[2], 0, 0);

	sub_cc_32(res[0], 0, tmp[0]);
	subc_cc_32(res[1], 0, tmp[1]);
	subc_cc_32(res[2], 0, tmp[2]);
	subc_cc_32(res[3], 0, 0);
	subc_cc_32(res[4], 0, 0);
	subc_cc_32(res[5], 0, 0);
	subc_cc_32(res[6], 0, 0);
	subc_cc_32(res[7], 0, 0);
	subc_32(res[8], val, 0);
}

__device__ __forceinline__ void shiftR_288_by_30(u32* res)
{
	res[0] = __funnelshift_r(res[0], res[1], 30);
	res[1] = __funnelshift_r(res[1], res[2], 30);
	res[2] = __funnelshift_r(res[2], res[3], 30);
	res[3] = __funnelshift_r(res[3], res[4], 30);
	res[4] = __funnelshift_r(res[4], res[5], 30);
	res[5] = __funnelshift_r(res[5], res[6], 30);
	res[6] = __funnelshift_r(res[6], res[7], 30);
	res[7] = __funnelshift_r(res[7], res[8], 30);
	res[8] = ((int)res[8]) >> 30;
}

__device__ __forceinline__ void add_288_P(u32* res)
{
	add_cc_32(res[0], res[0], 0xFFFFFC2F);
	addc_cc_32(res[1], res[1], 0xFFFFFFFE);
	addc_cc_32(res[2], res[2], 0xFFFFFFFF);
	addc_cc_32(res[3], res[3], 0xFFFFFFFF);
	addc_cc_32(res[4], res[4], 0xFFFFFFFF);
	addc_cc_32(res[5], res[5], 0xFFFFFFFF);
	addc_cc_32(res[6], res[6], 0xFFFFFFFF);
	addc_cc_32(res[7], res[7], 0xFFFFFFFF);
	addc_32(res[8], res[8], 0);
}

__device__ __forceinline__ void sub_288_P(u32* res)
{
	sub_cc_32(res[0], res[0], 0xFFFFFC2F);
	subc_cc_32(res[1], res[1], 0xFFFFFFFE);
	subc_cc_32(res[2], res[2], 0xFFFFFFFF);
	subc_cc_32(res[3], res[3], 0xFFFFFFFF);
	subc_cc_32(res[4], res[4], 0xFFFFFFFF);
	subc_cc_32(res[5], res[5], 0xFFFFFFFF);
	subc_cc_32(res[6], res[6], 0xFFFFFFFF);
	subc_cc_32(res[7], res[7], 0xFFFFFFFF);
	subc_32(res[8], res[8], 0);
}

#define APPLY_DIV_SHIFT()	matrix[0] <<= index; matrix[1] <<= index; kbnt -= index; _val >>= index;  
#define DO_INV_STEP()		{kbnt = -kbnt; int tmp = -_modp; _modp = _val; _val = tmp; tmp = -matrix[0]; \
							matrix[0] = matrix[2]; matrix[2] = tmp; tmp = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tmp;}

// https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494
//a bit tricky
//res must be at least 288bits
__device__ __forceinline__ void InvModP(u32* res)
{
	int matrix[4], _val, _modp, index, cnt, mx, kbnt;
	__align__(8) u32 modp[9];
	__align__(8) u32 val[9];
	__align__(8) u32 a[9];
	__align__(8) u32 tmp[4][9];

	((u64*)modp)[0] = P_0;
	((u64*)modp)[1] = P_123;
	((u64*)modp)[2] = P_123;
	((u64*)modp)[3] = P_123;
	modp[8] = 0;
	res[8] = 0;
	val[0] = res[0]; val[1] = res[1]; val[2] = res[2]; val[3] = res[3];
	val[4] = res[4]; val[5] = res[5]; val[6] = res[6]; val[7] = res[7];
	val[8] = 0;
	matrix[0] = matrix[3] = 1;
	matrix[1] = matrix[2] = 0;
	kbnt = -1;
	_val = (int)res[0];
	_modp = (int)P_0;
	index = __ffs(_val | 0x40000000) - 1;
	APPLY_DIV_SHIFT();
	cnt = 30 - index;
	while (cnt > 0)
	{
		if (kbnt < 0)
			DO_INV_STEP();
		mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
		i32 mul = (-_modp * _val) & 7;
		mul &= 0xFFFFFFFF >> mx;
		_val += _modp * mul;
		matrix[2] += matrix[0] * mul;
		matrix[3] += matrix[1] * mul;
		index = __ffs(_val | (1 << cnt)) - 1;
		APPLY_DIV_SHIFT();
		cnt -= index;
	}
	mul_288_by_i32(tmp[0], modp, matrix[0]);
	mul_288_by_i32(tmp[1], val, matrix[1]);
	mul_288_by_i32(tmp[2], modp, matrix[2]);
	mul_288_by_i32(tmp[3], val, matrix[3]);
	add_288(modp, tmp[0], tmp[1]);
	shiftR_288_by_30(modp);
	add_288(val, tmp[2], tmp[3]);
	shiftR_288_by_30(val);
	set_288_i32(tmp[1], matrix[1]);
	set_288_i32(tmp[3], matrix[3]);
	mul_P_by_32(res, (tmp[1][0] * 0xD2253531) & 0x3FFFFFFF);
	add_288(res, res, tmp[1]);
	shiftR_288_by_30(res);
	mul_P_by_32(a, (tmp[3][0] * 0xD2253531) & 0x3FFFFFFF);
	add_288(a, a, tmp[3]);
	shiftR_288_by_30(a);
	while (1)
	{
		matrix[0] = matrix[3] = 1;
		matrix[1] = matrix[2] = 0;
		_val = val[0];
		_modp = modp[0];
		index = __ffs(_val | 0x40000000) - 1;
		APPLY_DIV_SHIFT();
		cnt = 30 - index;
		while (cnt > 0)
		{
			if (kbnt < 0)
				DO_INV_STEP();
			mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
			i32 mul = (-_modp * _val) & 7;
			mul &= 0xFFFFFFFF >> mx;
			_val += _modp * mul;
			matrix[2] += matrix[0] * mul;
			matrix[3] += matrix[1] * mul;
			index = __ffs(_val | (1 << cnt)) - 1;
			APPLY_DIV_SHIFT();
			cnt -= index;
		}
		mul_288_by_i32(tmp[0], modp, matrix[0]);
		mul_288_by_i32(tmp[1], val, matrix[1]);
		mul_288_by_i32(tmp[2], modp, matrix[2]);
		mul_288_by_i32(tmp[3], val, matrix[3]);
		add_288(modp, tmp[0], tmp[1]);
		shiftR_288_by_30(modp);
		add_288(val, tmp[2], tmp[3]);
		shiftR_288_by_30(val);
		mul_288_by_i32(tmp[0], res, matrix[0]);
		mul_288_by_i32(tmp[1], a, matrix[1]);

		if ((val[0] | val[1] | val[2] | val[3] | val[4] | val[5] | val[6] | val[7]) == 0)
			break;

		mul_288_by_i32(tmp[2], res, matrix[2]);
		mul_288_by_i32(tmp[3], a, matrix[3]);
		mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531) & 0x3FFFFFFF);
		add_288(res, res, tmp[0]);
		add_288(res, res, tmp[1]);
		shiftR_288_by_30(res);
		mul_P_by_32(a, ((tmp[2][0] + tmp[3][0]) * 0xD2253531) & 0x3FFFFFFF);
		add_288(a, a, tmp[2]);
		add_288(a, a, tmp[3]);	
		shiftR_288_by_30(a);
	}
	mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531) & 0x3FFFFFFF);
	add_288(res, res, tmp[0]);
	add_288(res, res, tmp[1]);
	shiftR_288_by_30(res);
	if ((int)modp[8] < 0)
		neg_288(res);	
	while ((int)res[8] < 0)
		add_288_P(res);
	while ((int)res[8] > 0)
		sub_288_P(res);
}

// === Extensiones: utilidades Jacobianas (GPU) ================================
__device__ __forceinline__ void SquareModP(u64 *res, u64 *a) {
	MulModP(res, a, a);
}

// r = 2*p  (doblado jacobiano, a=0)
__device__ __forceinline__ void JacobianDouble(u64 *X3,u64 *Y3,u64 *Z3,const u64 *X1,const u64 *Y1,const u64 *Z1){
	u64 XX[4],YY[4],YYYY[4],S[4],M[4],T[4],twoS[4],Z3tmp[4];
	Copy_u64_x4(XX,(void*)X1); MulModP(XX,XX,XX);
	Copy_u64_x4(YY,(void*)Y1); MulModP(YY,YY,YY);
	Copy_u64_x4(YYYY,YY);      MulModP(YYYY,YYYY,YY);
	Copy_u64_x4(S,(void*)X1);  MulModP(S,S,YY); AddModP(S,S,S); // 2*X1*Y1^2
	Copy_u64_x4(M,XX); AddModP(M,M,XX); AddModP(M,M,XX);        // 3*X1^2
	Copy_u64_x4(T,M);  MulModP(T,T,M);                           // M^2
	Copy_u64_x4(twoS,S); AddModP(twoS,twoS,S);                   // 2*S
	SubModP((u64*)X3,T,twoS);                                    // X3
	u64 V[4]; Copy_u64_x4(V,S); SubModP(V,V,(u64*)X3); MulModP(V,V,M);
	u64 eightYYYY[4]; Copy_u64_x4(eightYYYY,YYYY);
	for (int i=0;i<3;i++) AddModP(eightYYYY,eightYYYY,YYYY); // *8
	SubModP((u64*)Y3,V,eightYYYY);
	Copy_u64_x4(Z3tmp,(void*)Y1); AddModP(Z3tmp,Z3tmp,(u64*)Y1); MulModP(Z3tmp,Z3tmp,(u64*)Z1);
	Copy_u64_x4((u64*)Z3,Z3tmp);
}

// r = p + q_aff  (mixta)
__device__ __forceinline__ void JacobianAddMixed(u64 *X3,u64 *Y3,u64 *Z3,const u64 *X1,const u64 *Y1,const u64 *Z1,const u64 *Qx,const u64 *Qy){
	u64 Z2[4],Z3_[4],U2[4],S2[4],H[4],R[4];
	Copy_u64_x4(Z2,(void*)Z1); MulModP(Z2,Z2,Z2);
	Copy_u64_x4(Z3_,Z2);       MulModP(Z3_,Z3_,(u64*)Z1);
	Copy_u64_x4(U2,(void*)Qx); MulModP(U2,U2,Z2);
	Copy_u64_x4(S2,(void*)Qy); MulModP(S2,S2,Z3_);
	Copy_u64_x4(H,U2); SubModP(H,H,(u64*)X1);
	Copy_u64_x4(R,S2); SubModP(R,R,(u64*)Y1);
	// if H==0 -> degenerate (omit; improbable en saltos aleatorios)
	u64 HH[4],HHH[4],V[4],X3_[4],Y3_[4],Z3tmp[4];
	Copy_u64_x4(HH,H);  MulModP(HH,HH,H);
	Copy_u64_x4(HHH,HH);MulModP(HHH,HHH,H);
	Copy_u64_x4(V,(void*)X1);  MulModP(V,V,HH);
	Copy_u64_x4(X3_,R); MulModP(X3_,X3_,R);
	SubModP(X3_,X3_,HHH);
	u64 twoV[4]; Copy_u64_x4(twoV,V); AddModP(twoV,twoV,V);
	SubModP((u64*)X3,X3_,twoV);
	Copy_u64_x4(Y3_,V); SubModP(Y3_,Y3_,(u64*)X3); MulModP(Y3_,Y3_,R);
	u64 S1HHH[4]; Copy_u64_x4(S1HHH,(void*)Y1); MulModP(S1HHH,S1HHH,HHH);
	SubModP((u64*)Y3,Y3_,S1HHH);
	Copy_u64_x4(Z3tmp,(void*)Z1); MulModP(Z3tmp,Z3tmp,H);
	Copy_u64_x4((u64*)Z3,Z3tmp);
}
// ============================================================================

