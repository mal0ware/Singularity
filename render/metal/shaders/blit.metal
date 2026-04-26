// render/metal/shaders/blit.metal
//
// Tone-map + bloom composite pass. Samples the full-res HDR (compute
// kernel output) and the 1/4-res bloom (from bloom.metal), applies ACES
// filmic tone mapping with an exposure multiplier, then writes to the
// sRGB drawable. The drawable format (BGRA8Unorm_sRGB) handles the
// final linear → gamma encode automatically on write.

#include <metal_stdlib>
using namespace metal;

struct BlitVSOut {
    float4 position [[position]];
    float2 uv;
};

vertex BlitVSOut blit_vertex(uint vid [[vertex_id]]) {
    const float2 pos[3] = {
        float2(-1.0f, -1.0f),
        float2(3.0f, -1.0f),
        float2(-1.0f,  3.0f)
    };
    const float2 uv[3] = {
        float2(0.0f, 1.0f),
        float2(2.0f, 1.0f),
        float2(0.0f, -1.0f)
    };
    BlitVSOut o;
    o.position = float4(pos[vid], 0.0f, 1.0f);
    o.uv = uv[vid];
    return o;
}

// Narkowicz ACES filmic approximation — the lean version of the
// full RRT+ODT ACES curve. Good highlight rolloff and skin/warm-tone
// preservation at ~4 ops per channel. Reference:
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
static inline float3 tonemap_aces(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

struct BlitParams {
    float exposure;
    float bloom_strength;
    float pad0;
    float pad1;
};

fragment float4 blit_fragment(
    BlitVSOut                 in     [[stage_in]],
    texture2d<float>          hdr    [[texture(0)]],
    texture2d<float>          bloom  [[texture(1)]],
    constant BlitParams&      p      [[buffer(0)]])
{
    constexpr sampler s_nearest(filter::nearest, address::clamp_to_edge);
    constexpr sampler s_linear (filter::linear,  address::clamp_to_edge);

    float3 c = hdr.sample(s_nearest, in.uv).rgb;
    // Bloom is at 1/4 res — bilinear upsample.
    const float3 b = bloom.sample(s_linear, in.uv).rgb;
    c += b * p.bloom_strength;
    c *= p.exposure;
    return float4(tonemap_aces(c), 1.0f);
}
