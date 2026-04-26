// render/vulkan/shaders/blit.hlsl
//
// HLSL port of render/metal/shaders/blit.metal. Fullscreen-triangle
// vertex + fragment (pixel) shaders. Entry points `main_vs` and
// `main_ps` — compiled separately by DXC into blit.vs.spv and blit.ps.spv.

struct BlitVSOut {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

BlitVSOut main_vs(uint vid : SV_VertexID) {
    const float2 pos[3] = {
        float2(-1.0f, -1.0f),
        float2( 3.0f, -1.0f),
        float2(-1.0f,  3.0f)
    };
    const float2 uv[3] = {
        // Vulkan NDC y is flipped vs. Metal/DX. Account for it here so
        // downstream UVs map top-left = (0, 0) as expected.
        float2(0.0f, 0.0f),
        float2(2.0f, 0.0f),
        float2(0.0f, 2.0f)
    };
    BlitVSOut o;
    o.position = float4(pos[vid], 0.0f, 1.0f);
    o.uv = uv[vid];
    return o;
}

struct BlitParams {
    float exposure;
    float bloom_strength;
    float pad0;
    float pad1;
};

[[vk::binding(0, 0)]] Texture2D<float4>    hdr    : register(t0);
[[vk::binding(1, 0)]] Texture2D<float4>    bloom  : register(t1);
[[vk::binding(2, 0)]] SamplerState         samp   : register(s0);
[[vk::binding(3, 0)]] ConstantBuffer<BlitParams> p : register(b0);

static float3 tonemap_aces(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float4 main_ps(BlitVSOut input) : SV_Target {
    float3 c = hdr.Sample(samp, input.uv).rgb;
    const float3 b = bloom.Sample(samp, input.uv).rgb;
    c += b * p.bloom_strength;
    c *= p.exposure;
    return float4(tonemap_aces(c), 1.0f);
}
