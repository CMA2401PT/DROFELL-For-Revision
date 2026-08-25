#ifndef _qdq_util_cuh
#define _qdq_util_cuh

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
};

// Max_scale premultiplied by 1/256

__forceinline__ __device__ half dq_scale(const int qs, const half max_scale)
{
    int qs_i = qs + 1;
    half qs_h = __int2half_rn(qs_i * qs_i);
    qs_h = __hmul(qs_h, max_scale);
    return qs_h;
}

__forceinline__ __device__ half dq(const int q, const int qzero, const half scale)
{
    return __hmul(__int2half_rn(q - qzero), scale);
}

__forceinline__ __device__ half dq_ns(const int q, const int qzero)
{
    //return __hsub(__int2half_rn(q), __int2half_rn(qzero));
    return __int2half_rn(q - qzero);
}

__forceinline__ __device__ int exb(const uint32_t q, const int shift, const int mask)
{
    return (int)((q >> shift) & mask);
}

__forceinline__ __device__ int exb(const uint32_t q1, const uint32_t q0, const int shift, const int mask)
{
    return (int)(__funnelshift_rc(q0, q1, shift) & mask);
}


__forceinline__ __device__ void shuffle_2bit_16
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        uint32_t qa0 = qa & 0x03;
        uint32_t qa1 = (qa & 0x0c) >> 2;
        qa >>= 4;
        qb |= (qa1 << (i * 2 + 16));
        qb |= (qa0 << (i * 2));
    }
    q[0] = qb;
}

__forceinline__ __device__ void dequant_2bit_16
(
    const uint32_t q_0,
    half2 (&dq)[8],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y4_  = __float2half_rn(1.0f /  4.0f);
    const half y16_ = __float2half_rn(1.0f / 16.0f);
    const half y64_ = __float2half_rn(1.0f / 64.0f);
    const half2 y4  = __halves2half2(y4_,  y4_);
    const half2 y16 = __halves2half2(y16_, y16_);
    const half2 y64 = __halves2half2(y64_, y64_);
    const half z1_  = __float2half_rn(-1024.0f         - 2.0f);
    const half z4_  = __float2half_rn(-1024.0f /  4.0f - 2.0f);
    const half z16_ = __float2half_rn(-1024.0f / 16.0f - 2.0f);
    const half z64_ = __float2half_rn(-1024.0f / 64.0f - 2.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z4  = __halves2half2(z4_,  z4_);
    const half2 z16 = __halves2half2(z16_, z16_);
    const half2 z64 = __halves2half2(z64_, z64_);

    uint32_t qa = q_0;
    half2_uint32 q0((qa & 0x00030003) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x000c000c) | c0); // half2(q[ 2], q[ 3]) *  4 + 1024
    half2_uint32 q2((qa & 0x00300030) | c0); // half2(q[ 4], q[ 5]) * 16 + 1024
    half2_uint32 q3((qa & 0x00c000c0) | c0); // half2(q[ 6], q[ 7]) * 64 + 1024
    qa >>= 8;
    half2_uint32 q4((qa & 0x00030003) | c0); // half2(q[ 8], q[ 8])      + 1024
    half2_uint32 q5((qa & 0x000c000c) | c0); // half2(q[10], q[11]) *  4 + 1024
    half2_uint32 q6((qa & 0x00300030) | c0); // half2(q[12], q[13]) * 16 + 1024
    half2_uint32 q7((qa & 0x00c000c0) | c0); // half2(q[14], q[15]) * 64 + 1024

    dq[0] = __hadd2(q0.as_half2, z1);
    dq[1] = __hfma2(q1.as_half2, y4,  z4);
    dq[2] = __hfma2(q2.as_half2, y16, z16);
    dq[3] = __hfma2(q3.as_half2, y64, z64);
    dq[4] = __hadd2(q4.as_half2, z1);
    dq[5] = __hfma2(q5.as_half2, y4,  z4);
    dq[6] = __hfma2(q6.as_half2, y16, z16);
    dq[7] = __hfma2(q7.as_half2, y64, z64);
}


// Permutation:
//
// v9997775 55333111  u8886664 44222000  (u, v lsb)
// vjjjhhhf ffdddbbb  uiiiggge eecccaaa
// vtttrrrp ppnnnlll  usssqqqo oommmkkk

__forceinline__ __device__ void shuffle_3bit_32
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0 * stride];
    uint32_t qb = q[1 * stride];
    uint32_t qc = q[2 * stride];

    // qa: aa999888 77766655  54443332 22111000
    // qb: lkkkjjji iihhhggg  fffeeedd dcccbbba
    // qc: vvvuuutt tsssrrrq  qqpppooo nnnmmmll

    uint32_t qd = qc >> 26;
    qc <<= 4;
    qc |= qb >> 28;
    qb <<= 2;
    qb |= qa >> 30;

    // qa: ..999888 77766655  54443332 22111000
    // qb: ..jjjiii hhhgggff  feeedddc ccbbbaaa
    // qc: ..tttsss rrrqqqpp  pooonnnm mmlllkkk
    // qd:                               vvvuuu

    uint32_t za = 0;
    uint32_t zb = 0;
    uint32_t zc = 0;

    for (int i = 0; i < 5; i++) { uint32_t t0 = qa & 0x07; uint32_t t1 = (qa & 0x38) >> 3; qa >>= 6; za |= (t0 << (i * 3)); za |= (t1 << (i * 3 + 16)); }
    for (int i = 0; i < 5; i++) { uint32_t t0 = qb & 0x07; uint32_t t1 = (qb & 0x38) >> 3; qb >>= 6; zb |= (t0 << (i * 3)); zb |= (t1 << (i * 3 + 16)); }
    for (int i = 0; i < 5; i++) { uint32_t t0 = qc & 0x07; uint32_t t1 = (qc & 0x38) >> 3; qc >>= 6; zc |= (t0 << (i * 3)); zc |= (t1 << (i * 3 + 16)); }

    // za:  9997775 55333111   8886664 44222000
    // zb:  jjjhhhf ffdddbbb   iiiggge eecccaaa
    // zc:  tttrrrp ppnnnlll   sssqqqo oommmkkk
    // qd:                               vvvuuu

    za |= ((qd & 0x01) >> 0) << 15;
    zb |= ((qd & 0x02) >> 1) << 15;
    zc |= ((qd & 0x04) >> 2) << 15;
    za |= ((qd & 0x08) >> 3) << 31;
    zb |= ((qd & 0x10) >> 4) << 31;
    zc |= ((qd & 0x20) >> 5) << 31;

    // za: v9997775 55333111  u8886664 44222000  (u, v lsb)
    // zb: vjjjhhhf ffdddbbb  uiiiggge eecccaaa
    // zc: vtttrrrp ppnnnlll  usssqqqo oommmkkk

    q[0 * stride] = za;
    q[1 * stride] = zb;
    q[2 * stride] = zc;
}

__forceinline__ __device__ void dequant_3bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    half2 (&dq)[16],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y8_  = __float2half_rn(1.0f /  8.0f);
    const half y64_ = __float2half_rn(1.0f / 64.0f);
    const half2 y8  = __halves2half2(y8_,  y8_);
    const half2 y64 = __halves2half2(y64_, y64_);
    const half z1_  = __float2half_rn(-1024.0f         - 4.0f);
    const half z8_  = __float2half_rn(-1024.0f /  8.0f - 4.0f);
    const half z64_ = __float2half_rn(-1024.0f / 64.0f - 4.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z8  = __halves2half2(z8_,  z8_);
    const half2 z64 = __halves2half2(z64_, z64_);

    uint32_t qa = q_0;
    uint32_t qb = q_1;
    uint32_t qc = q_2;

    half2_uint32 q0((qa & 0x00070007) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x00380038) | c0); // half2(q[ 2], q[ 3]) *  8 + 1024
    qa >>= 6;
    half2_uint32 q2((qa & 0x00070007) | c0); // half2(q[ 4], q[ 5])      + 1024
    half2_uint32 q3((qa & 0x00380038) | c0); // half2(q[ 6], q[ 7]) *  8 + 1024
    half2_uint32 q4((qa & 0x01c001c0) | c0); // half2(q[ 8], q[ 9]) * 64 + 1024
    qa >>= 9;
    qa &= 0x00010001;
    half2_uint32 q5((qb & 0x00070007) | c0); // half2(q[10], q[11])      + 1024
    half2_uint32 q6((qb & 0x00380038) | c0); // half2(q[12], q[13]) *  8 + 1024
    qb >>= 6;
    half2_uint32 q7((qb & 0x00070007) | c0); // half2(q[14], q[15])      + 1024
    half2_uint32 q8((qb & 0x00380038) | c0); // half2(q[16], q[17]) *  8 + 1024
    half2_uint32 q9((qb & 0x01c001c0) | c0); // half2(q[18], q[19]) * 64 + 1024
    qb >>= 8;
    qb &= 0x00020002;
    half2_uint32 q10((qc & 0x00070007) | c0); // half2(q[20], q[21])      + 1024
    half2_uint32 q11((qc & 0x00380038) | c0); // half2(q[22], q[23]) *  8 + 1024
    qc >>= 6;
    half2_uint32 q12((qc & 0x00070007) | c0); // half2(q[24], q[25])      + 1024
    half2_uint32 q13((qc & 0x00380038) | c0); // half2(q[26], q[27]) *  8 + 1024
    half2_uint32 q14((qc & 0x01c001c0) | c0); // half2(q[28], q[29]) * 64 + 1024
    qc >>= 7;
    qc &= 0x00040004;
    half2_uint32 q15((qa | qb | qc) | c0);

    dq[ 0] = __hadd2( q0.as_half2, z1);
    dq[ 1] = __hfma2( q1.as_half2, y8,  z8);
    dq[ 2] = __hadd2( q2.as_half2, z1);
    dq[ 3] = __hfma2( q3.as_half2, y8,  z8);
    dq[ 4] = __hfma2( q4.as_half2, y64, z64);
    dq[ 5] = __hadd2( q5.as_half2, z1);
    dq[ 6] = __hfma2( q6.as_half2, y8,  z8);
    dq[ 7] = __hadd2( q7.as_half2, z1);
    dq[ 8] = __hfma2( q8.as_half2, y8,  z8);
    dq[ 9] = __hfma2( q9.as_half2, y64, z64);
    dq[10] = __hadd2(q10.as_half2, z1);
    dq[11] = __hfma2(q11.as_half2, y8,  z8);
    dq[12] = __hadd2(q12.as_half2, z1);
    dq[13] = __hfma2(q13.as_half2, y8,  z8);
    dq[14] = __hfma2(q14.as_half2, y64, z64);
    dq[15] = __hadd2(q15.as_half2, z1);
}


// Permutation:
//
// 77775555 33331111  66664444 22220000

__forceinline__ __device__ void shuffle_4bit_8
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t qa0 = qa & 0x0f;
        uint32_t qa1 = (qa & 0xf0) >> 4;
        qa >>= 8;
        qb |= (qa1 << (i * 4 + 16));
        qb |= (qa0 << (i * 4));
    }
    q[0] = qb;
}

__forceinline__ __device__ void dequant_4bit_8
(
    const uint32_t q_0,
    half2 (&dq)[4],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y16_ = __float2half_rn(1.0f / 16.0f);
    const half2 y16 = __halves2half2(y16_, y16_);
    const half z1_  = __float2half_rn(-1024.0f         - 8.0f);
    const half z16_ = __float2half_rn(-1024.0f / 16.0f - 8.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z16 = __halves2half2(z16_, z16_);

    uint32_t qa = q_0;
    half2_uint32 q0((qa & 0x000f000f) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x00f000f0) | c0); // half2(q[ 2], q[ 3]) * 16 + 1024
    qa >>= 8;
    half2_uint32 q2((qa & 0x000f000f) | c0); // half2(q[ 4], q[ 5])      + 1024
    half2_uint32 q3((qa & 0x00f000f0) | c0); // half2(q[ 6], q[ 7]) * 16 + 1024

    dq[0] = __hadd2(q0.as_half2, z1);
    dq[1] = __hfma2(q1.as_half2, y16, z16);
    dq[2] = __hadd2(q2.as_half2, z1);
    dq[3] = __hfma2(q3.as_half2, y16, z16);
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale
(
    const uint32_t zero,
    const half scale,
    half2 (&z1z16)[2],
    half2 (&y1y16)[2]
)
{
    half_uint16 z1(0xe400 | zero); // half(-1024.0f - zero);
    half z16 = __hsub(__int2half_rn(-64), __int2half_rn(zero));

    half2 scale2 = __half2half2(scale);

    z1z16[0] = __hmul2(scale2, __half2half2(z1.as_half));
    z1z16[1] = __hmul2(scale2, __half2half2(z16));

    const half y1 = __float2half_rn(1.0f);
    const half y16 = __float2half_rn(1.0f / 16.0f);

    y1y16[0] = __hmul2(scale2, __half2half2(y1));
    y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero
(
    const uint32_t zero,
    half2(&z1z16)[2],
    half2(&y1y16)[2]
)
{
    half_uint16 z1(0xe400 | zero); // half(-1024.0f - zero);
    half z16 = __hsub(__int2half_rn(-64), __int2half_rn(zero));

    z1z16[0] = __half2half2(z1.as_half);
    z1z16[1] = __half2half2(z16);

    const half y1 = __float2half_rn(1.0f);
    const half y16 = __float2half_rn(1.0f / 16.0f);

    y1y16[0] = __half2half2(y1);
    y1y16[1] = __half2half2(y16);
}


__forceinline__ __device__ void dequant_4bit_8_gptq
(
    const uint32_t q_0,
    half2 (&dq)[4],
    half2 (&z1z16)[2],
    half2 (&y1y16)[2],
    int stride,
    bool scaled
)
{
    const uint32_t c0 = 0x64006400;

    uint32_t qa = q_0;
    half2_uint32 q0((qa & 0x000f000f) | c0); // half2( q[0]      + 1024, q[1]      + 1024 )
    half2_uint32 q1((qa & 0x00f000f0) | c0); // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
    qa >>= 8;
    half2_uint32 q2((qa & 0x000f000f) | c0); // half2( q[4]      + 1024, q[5]      + 1024 )
    half2_uint32 q3((qa & 0x00f000f0) | c0); // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

    if (scaled)
    {
        dq[0] = __hfma2(q0.as_half2, y1y16[0], z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
        dq[1] = __hfma2(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
        dq[2] = __hfma2(q2.as_half2, y1y16[0], z1z16[0]);
        dq[3] = __hfma2(q3.as_half2, y1y16[1], z1z16[1]);
    }
    else
    {
        dq[0] = __hadd2(q0.as_half2,           z1z16[0]);  // half2( q[0] - z, q[1] - z )
        dq[1] = __hfma2(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] - z, q[3] - z )
        dq[2] = __hadd2(q2.as_half2,           z1z16[0]);  // half2( q[4] - z, q[5] - z )
        dq[3] = __hfma2(q3.as_half2, y1y16[1], z1z16[1]);  // half2( q[6] - z, q[7] - z )
    }
}

#endif
