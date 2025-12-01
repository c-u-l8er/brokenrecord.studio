const std = @import("std");
const vk = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const Allocator = std.mem.Allocator;
const gpu_backend = @import("gpu_backend.zig");

pub const RTCores = struct {
    backend: *gpu_backend.GPUBackend,
    acceleration_structure: ?gpu_backend.BufferHandle,
    scratch_buffer: ?gpu_backend.BufferHandle,
    instance_buffer: ?gpu_backend.BufferHandle,





    pub fn init(backend: *gpu_backend.GPUBackend) RTCores {
        return RTCores{
            .backend = backend,
            .acceleration_structure = null,
            .scratch_buffer = null,
            .instance_buffer = null,
        };
    }

    pub fn deinit(self: *RTCores) void {
        if (self.acceleration_structure) |as| {
            self.backend.destroyBuffer(as);
        }
        if (self.scratch_buffer) |sb| {
            self.backend.destroyBuffer(sb);
        }
        if (self.instance_buffer) |ib| {
            self.backend.destroyBuffer(ib);
        }
    }







    // Private methods
};


```
