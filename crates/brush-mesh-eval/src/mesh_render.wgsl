struct Uniforms {
    mvp: mat4x4<f32>,
    // x: 1.0 = sample the atlas texture, 0.0 = vertex colors. yzw unused.
    use_texture: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u.mvp * vec4<f32>(v.pos, 1.0);
    out.uv = v.uv;
    return out;
}

@fragment
fn fs_main(v: VertexOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, v.uv).rgb;
    return vec4<f32>(mix(vec3<f32>(0.5, 0.5, 0.5), tex, u.use_texture.x), 1.0);
}
