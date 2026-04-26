// render/webgpu/shaders/blit.wgsl
//
// WGSL port of render/vulkan/shaders/blit.hlsl. Fullscreen-triangle vertex
// shader + HDR composite + ACES tone map fragment shader. Bindings match
// the Vulkan descriptor-set layout 1:1 so the host resource graph is
// identical across backends.

struct BlitVSOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn main_vs(@builtin(vertex_index) vid: u32) -> BlitVSOut {
    // Fullscreen triangle that covers NDC [-1, 1]² without needing a
    // vertex buffer. vid ∈ {0, 1, 2} → three corners.
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    // WebGPU's clip-space Y is up (same as Metal, same as D3D). The
    // Vulkan port flipped uv here because Vk NDC has Y down; WebGPU
    // doesn't need that flip, so uv maps directly.
    var uvs = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );
    var o: BlitVSOut;
    o.position = vec4f(positions[vid], 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

struct BlitParams {
    exposure: f32,
    bloom_strength: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> p: BlitParams;

fn tonemap_aces(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    // saturate() was added late to WGSL; use clamp for universal support.
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3f(0.0), vec3f(1.0));
}

@fragment
fn main_ps(input: BlitVSOut) -> @location(0) vec4f {
    var col = textureSample(hdr, samp, input.uv).rgb;
    let bl = textureSample(bloom, samp, input.uv).rgb;
    col = col + bl * p.bloom_strength;
    col = col * p.exposure;
    return vec4f(tonemap_aces(col), 1.0);
}
