#include "impl.h"
#include <arm_neon.h>
#include <cstdio>

static void kernel_8x8(float *a, float *b, float *c, int lda, int ldb, int ldc, int k) {
    float32x4_t c00;
    float32x4_t c01;
    float32x4_t c10;
    float32x4_t c11;
    float32x4_t c20;
    float32x4_t c21;
    float32x4_t c30;
    float32x4_t c31;
    float32x4_t c40;
    float32x4_t c41;
    float32x4_t c50;
    float32x4_t c51;
    float32x4_t c60;
    float32x4_t c61;
    float32x4_t c70;
    float32x4_t c71;

    float32x4_t a0;
    float32x4_t a1;

    float32x4_t b0;
    float32x4_t b1;

    // load c
    float *cc = c;
    c00 = vld1q_f32(cc);
    c01 = vld1q_f32(cc + 4);
    cc += ldc;
    c10 = vld1q_f32(cc);
    c11 = vld1q_f32(cc + 4);
    cc += ldc;
    c20 = vld1q_f32(cc);
    c21 = vld1q_f32(cc + 4);
    cc += ldc;
    c30 = vld1q_f32(cc);
    c31 = vld1q_f32(cc + 4);
    cc += ldc;
    c40 = vld1q_f32(cc);
    c41 = vld1q_f32(cc + 4);
    cc += ldc;
    c50 = vld1q_f32(cc);
    c51 = vld1q_f32(cc + 4);
    cc += ldc;
    c60 = vld1q_f32(cc);
    c61 = vld1q_f32(cc + 4);
    cc += ldc;
    c70 = vld1q_f32(cc);
    c71 = vld1q_f32(cc + 4);

    for (int p = 0; p < k; p++) {
        a0 = vld1q_f32(a);
        a1 = vld1q_f32(a + 4);
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);

        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);

        a += lda;
        b += ldb;
    }

    // store c
    vst1q_f32(c, c00);
    vst1q_f32(c + 4, c01);
    c += ldc;
    vst1q_f32(c, c10);
    vst1q_f32(c + 4, c11);
    c += ldc;
    vst1q_f32(c, c20);
    vst1q_f32(c + 4, c21);
    c += ldc;
    vst1q_f32(c, c30);
    vst1q_f32(c + 4, c31);
    c += ldc;
    vst1q_f32(c, c40);
    vst1q_f32(c + 4, c41);
    c += ldc;
    vst1q_f32(c, c50);
    vst1q_f32(c + 4, c51);
    c += ldc;
    vst1q_f32(c, c60);
    vst1q_f32(c + 4, c61);
    c += ldc;
    vst1q_f32(c, c70);
    vst1q_f32(c + 4, c71);
}

void NeonFP32Impl::operator()(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i+=8) {
        for (int j = 0; j < n; j+=8) {
            kernel_8x8(a + i, b + j, c + i * n + j, m, n, n, k);
        }
    }
}