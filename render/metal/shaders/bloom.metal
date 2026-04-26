// render/metal/shaders/bloom.metal
//
// Cinematic bloom for the geodesic kernel's HDR output. Three stages,
// all compute:
//
//   bloom_extract      full-res HDR -> 1/4 res bright-only texture
//   bloom_blur_h       1/4 res separable horizontal Gaussian
//   bloom_blur_v       1/4 res separable vertical Gaussian
//
// The blit fragment then samples the blurred 1/4 texture bilinearly and
// mixes it over the tone-mapped full-res HDR for a physically-motivated
// glare/veil effect. Keeps everything on-GPU, no CPU readback.
//
// Two textures are ping-ponged to avoid read/write hazard on the blur
// passes: extract writes A, blur_h reads A writes B, blur_v reads B
// writes A. The app then reads A.

#include <metal_stdlib>
using namespace metal;

// Karis average — downsample 4 samples with a weight that suppresses the
// single-pixel "firefly" highlights that would otherwise bloom into ugly
// blocky artefacts at 1/4 resolution. This is the trick Unreal / COD use.
static inline float3 karis_average(float3 a, float3 b, float3 c, float3 d) {
    const float3 lum_w = float3(0.2126f, 0.7152f, 0.0722f);
    const float la = 1.0f / (1.0f + dot(a, lum_w));
    const float lb = 1.0f / (1.0f + dot(b, lum_w));
    const float lc = 1.0f / (1.0f + dot(c, lum_w));
    const float ld = 1.0f / (1.0f + dot(d, lum_w));
    const float wsum = la + lb + lc + ld;
    return (a * la + b * lb + c * lc + d * ld) / max(wsum, 1e-6f);
}

// Extract: read HDR at 4 neighbour pixels, Karis-downsample, soft-threshold
// to kill the dim pixels while letting very bright ones through smoothly.
kernel void bloom_extract(
    texture2d<float, access::read>  hdr        [[texture(0)]],
    texture2d<float, access::write> dst        [[texture(1)]],
    constant float&                 threshold  [[buffer(0)]],
    uint2                           gid        [[thread_position_in_grid]])
{
    const uint2 dst_dim = uint2(dst.get_width(), dst.get_height());
    if (gid.x >= dst_dim.x || gid.y >= dst_dim.y) return;

    const uint2 src_base = gid * 2u;
    const float3 a = hdr.read(src_base + uint2(0, 0)).rgb;
    const float3 b = hdr.read(src_base + uint2(1, 0)).rgb;
    const float3 c = hdr.read(src_base + uint2(0, 1)).rgb;
    const float3 d = hdr.read(src_base + uint2(1, 1)).rgb;
    float3 avg = karis_average(a, b, c, d);

    const float lum = dot(avg, float3(0.2126f, 0.7152f, 0.0722f));
    // Smooth knee: contribution = lum² / (lum + threshold) — similar to
    // Unreal's bloom extract curve. At `threshold` the gain is 0.5; above
    // it the full HDR magnitude flows through. No hard binary cutoff.
    const float knee = (lum * lum) / max(lum + threshold, 1e-4f);
    avg *= (lum > 1e-4f) ? (knee / lum) : 0.0f;

    dst.write(float4(avg, 1.0f), gid);
}

// 13-tap separable Gaussian (standard σ ≈ 3 at the 1/4-res scale).
// The coefficients pre-convolve two 5-tap kernels for a smoother tail
// without doubling the memory traffic.
constant float kBloomWeights[7] = {
    0.071303f, 0.131514f, 0.189879f, 0.214607f,
    0.189879f, 0.131514f, 0.071303f
};

kernel void bloom_blur_h(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2                           gid [[thread_position_in_grid]])
{
    const uint2 dim = uint2(dst.get_width(), dst.get_height());
    if (gid.x >= dim.x || gid.y >= dim.y) return;
    float3 sum = float3(0.0f);
    for (int i = 0; i < 7; ++i) {
        const int dx = i - 3;
        const int2 pos = int2(gid) + int2(dx, 0);
        const int2 clamped = clamp(pos, int2(0, 0), int2(dim) - int2(1, 1));
        sum += src.read(uint2(clamped)).rgb * kBloomWeights[i];
    }
    dst.write(float4(sum, 1.0f), gid);
}

kernel void bloom_blur_v(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2                           gid [[thread_position_in_grid]])
{
    const uint2 dim = uint2(dst.get_width(), dst.get_height());
    if (gid.x >= dim.x || gid.y >= dim.y) return;
    float3 sum = float3(0.0f);
    for (int i = 0; i < 7; ++i) {
        const int dy = i - 3;
        const int2 pos = int2(gid) + int2(0, dy);
        const int2 clamped = clamp(pos, int2(0, 0), int2(dim) - int2(1, 1));
        sum += src.read(uint2(clamped)).rgb * kBloomWeights[i];
    }
    dst.write(float4(sum, 1.0f), gid);
}
