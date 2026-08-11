struct Edit {
    x: u32,
    y: u32,
    value: u32,
};

@group(0) @binding(0) var state: texture_storage_2d<r32uint, write>;
@group(0) @binding(1) var<storage, read> edits: array<Edit>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if idx >= arrayLength(&edits) {
        return;
    }
    let e = edits[idx];
    textureStore(state, vec2i(i32(e.x), i32(e.y)), vec4u(e.value, 0u, 0u, 0u));
}
