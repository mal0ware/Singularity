// render/webgpu/shaders/bloom.wgsl
//
// WGSL port of render/vulkan/shaders/bloom.hlsl. Three entry points —
// extract, blur_h, blur_v — all share the same bind-group layout so the
// host can alias a single wgpu::BindGroupLayout across the three
// pipelines, matching the descriptor-set sharing on Vulkan.

const kBloomWeights = array<f32, 7>(
    0.071303, 0.131514, 0.189879, 0.214607,
    0.189879, 0.131514, 0.071303,
);

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

struct ExtractParams {
    threshold: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(2) var<uniform> ep: ExtractParams;

fn karis_average(a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
    let lum_w = vec3f(0.2126, 0.7152, 0.0722);
    let la = 1.0 / (1.0 + dot(a, lum_w));
    let lb = 1.0 / (1.0 + dot(b, lum_w));
    let lc = 1.0 / (1.0 + dot(c, lum_w));
    let ld = 1.0 / (1.0 + dot(d, lum_w));
    let wsum = la + lb + lc + ld;
    return (a * la + b * lb + c * lc + d * ld) / max(wsum, 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn extract(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let sb = gid.xy * 2u;
    let a = textureLoad(src, vec2i(i32(sb.x),     i32(sb.y)),     0).rgb;
    let b = textureLoad(src, vec2i(i32(sb.x) + 1, i32(sb.y)),     0).rgb;
    let c = textureLoad(src, vec2i(i32(sb.x),     i32(sb.y) + 1), 0).rgb;
    let d = textureLoad(src, vec2i(i32(sb.x) + 1, i32(sb.y) + 1), 0).rgb;
    var avg = karis_average(a, b, c, d);

    let lum = dot(avg, vec3f(0.2126, 0.7152, 0.0722));
    let knee = (lum * lum) / max(lum + ep.threshold, 1e-4);
    var scale: f32;
    if (lum > 1e-4) { scale = knee / lum; } else { scale = 0.0; }
    avg = avg * scale;

    textureStore(dst, vec2i(gid.xy), vec4f(avg, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn blur_h(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    var sum = vec3f(0.0);
    let hi = vec2i(i32(dims.x) - 1, i32(dims.y) - 1);
    for (var i: i32 = 0; i < 7; i = i + 1) {
        let dx = i - 3;
        let pos = vec2i(i32(gid.x) + dx, i32(gid.y));
        let cp = clamp(pos, vec2i(0, 0), hi);
        sum = sum + textureLoad(src, cp, 0).rgb * kBloomWeights[i];
    }
    textureStore(dst, vec2i(gid.xy), vec4f(sum, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn blur_v(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    var sum = vec3f(0.0);
    let hi = vec2i(i32(dims.x) - 1, i32(dims.y) - 1);
    for (var i: i32 = 0; i < 7; i = i + 1) {
        let dy = i - 3;
        let pos = vec2i(i32(gid.x), i32(gid.y) + dy);
        let cp = clamp(pos, vec2i(0, 0), hi);
        sum = sum + textureLoad(src, cp, 0).rgb * kBloomWeights[i];
    }
    textureStore(dst, vec2i(gid.xy), vec4f(sum, 1.0));
}
