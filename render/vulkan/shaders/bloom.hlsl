// render/vulkan/shaders/bloom.hlsl
//
// HLSL port of render/metal/shaders/bloom.metal. Three compute entry
// points — `extract`, `blur_h`, `blur_v` — compiled separately by DXC
// into three .spv artefacts. Descriptor bindings mirror the Metal
// backend: slot 0 read, slot 1 write, slot 2 (extract only) threshold.

static const float kBloomWeights[7] = {
    0.071303f, 0.131514f, 0.189879f, 0.214607f,
    0.189879f, 0.131514f, 0.071303f
};

[[vk::binding(0, 0)]] Texture2D<float4>    src       : register(t0);
[[vk::binding(1, 0)]] RWTexture2D<float4>  dst       : register(u0);

struct ExtractParams { float threshold; float pad0; float pad1; float pad2; };
[[vk::binding(2, 0)]] ConstantBuffer<ExtractParams> ep : register(b0);

static float3 karis_average(float3 a, float3 b, float3 c, float3 d) {
    const float3 lum_w = float3(0.2126f, 0.7152f, 0.0722f);
    const float la = 1.0f / (1.0f + dot(a, lum_w));
    const float lb = 1.0f / (1.0f + dot(b, lum_w));
    const float lc = 1.0f / (1.0f + dot(c, lum_w));
    const float ld = 1.0f / (1.0f + dot(d, lum_w));
    const float wsum = la + lb + lc + ld;
    return (a * la + b * lb + c * lc + d * ld) / max(wsum, 1e-6f);
}

[numthreads(8, 8, 1)]
void extract(uint2 gid : SV_DispatchThreadID) {
    uint dw, dh;
    dst.GetDimensions(dw, dh);
    if (gid.x >= dw || gid.y >= dh) return;

    const uint2 sb = gid * 2u;
    const float3 a = src.Load(int3(int(sb.x),     int(sb.y),     0)).rgb;
    const float3 b = src.Load(int3(int(sb.x) + 1, int(sb.y),     0)).rgb;
    const float3 c = src.Load(int3(int(sb.x),     int(sb.y) + 1, 0)).rgb;
    const float3 d = src.Load(int3(int(sb.x) + 1, int(sb.y) + 1, 0)).rgb;
    float3 avg = karis_average(a, b, c, d);

    const float lum = dot(avg, float3(0.2126f, 0.7152f, 0.0722f));
    const float knee = (lum * lum) / max(lum + ep.threshold, 1e-4f);
    avg *= (lum > 1e-4f) ? (knee / lum) : 0.0f;

    dst[gid] = float4(avg, 1.0f);
}

[numthreads(8, 8, 1)]
void blur_h(uint2 gid : SV_DispatchThreadID) {
    uint dw, dh;
    dst.GetDimensions(dw, dh);
    if (gid.x >= dw || gid.y >= dh) return;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int i = 0; i < 7; ++i) {
        const int dx = i - 3;
        const int2 pos = int2(gid) + int2(dx, 0);
        const int2 cp  = clamp(pos, int2(0, 0), int2(dw, dh) - int2(1, 1));
        sum += src.Load(int3(cp, 0)).rgb * kBloomWeights[i];
    }
    dst[gid] = float4(sum, 1.0f);
}

[numthreads(8, 8, 1)]
void blur_v(uint2 gid : SV_DispatchThreadID) {
    uint dw, dh;
    dst.GetDimensions(dw, dh);
    if (gid.x >= dw || gid.y >= dh) return;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int i = 0; i < 7; ++i) {
        const int dy = i - 3;
        const int2 pos = int2(gid) + int2(0, dy);
        const int2 cp  = clamp(pos, int2(0, 0), int2(dw, dh) - int2(1, 1));
        sum += src.Load(int3(cp, 0)).rgb * kBloomWeights[i];
    }
    dst[gid] = float4(sum, 1.0f);
}
