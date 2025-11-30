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
    bvh_build_info: BVHBuildInfo,

    pub const BVHBuildInfo = struct {
        primitive_count: u32,
        max_primitive_count: u32,
        first_vertex: u32,
        vertex_offset: u32,
        transform_offset: u32,
        flags: BuildFlags,
    };

    pub const BuildFlags = packed struct {
        allow_update: bool = false,
        allow_compaction: bool = false,
        prefer_fast_trace: bool = true,
        prefer_fast_build: bool = false,
        low_memory: bool = false,
    };

    pub const RayQuery = struct {
        origin: [3]f32,
        direction: [3]f32,
        t_min: f32,
        t_max: f32,
    };

    pub const HitInfo = struct {
        hit: bool,
        t: f32,
        primitive_id: u32,
        instance_id: u32,
        barycentric: [3]f32,
    };

    pub fn init(backend: *gpu_backend.GPUBackend) RTCores {
        return RTCores{
            .backend = backend,
            .acceleration_structure = null,
            .scratch_buffer = null,
            .instance_buffer = null,
            .bvh_build_info = undefined,
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

    pub fn buildBVH(
        self: *RTCores,
        vertices: []const [3]f32,
        indices: []const [3]u32,
        transforms: []const [12]f32
    ) !void {
        // Calculate sizes
        const primitive_count = indices.len;
        const vertex_count = vertices.len;

        // Create vertex buffer
        const vertex_buffer = try self.backend.createBuffer(
            vertex_count * @sizeOf([3]f32),
            .storage
        );
        defer self.backend.destroyBuffer(vertex_buffer);

        // Upload vertex data
        try self.backend.uploadData(
            vertex_buffer,
            std.mem.sliceAsBytes(vertices),
            0
        );

        // Create index buffer
        const index_buffer = try self.backend.createBuffer(
            primitive_count * @sizeOf([3]u32),
            .storage
        );
        defer self.backend.destroyBuffer(index_buffer);

        // Upload index data
        try self.backend.uploadData(
            index_buffer,
            std.mem.sliceAsBytes(indices),
            0
        );

        // Create transform buffer if needed
        if (transforms.len > 0) {
            const transform_buffer = try self.backend.createBuffer(
                transforms.len * @sizeOf([12]f32),
                .storage
            );
            defer self.backend.destroyBuffer(transform_buffer);

            try self.backend.uploadData(
                transform_buffer,
                std.mem.sliceAsBytes(transforms),
                0
            );
        }

        // Build acceleration structure
        try self.buildAccelerationStructure(
            vertex_buffer,
            index_buffer,
            primitive_count
        );
    }

    pub fn rayQuery(self: *const RTCores, ray: RayQuery) !HitInfo {
        if (self.acceleration_structure == null) {
            return HitInfo{
                .hit = false,
                .t = 0,
                .primitive_id = 0,
                .instance_id = 0,
                .barycentric = [_]f32{0, 0, 0},
            };
        }

        // Create ray query shader
        const shader_code = try self.createRayQueryShader();
        defer self.backend.destroyShader(shader_code);

        // Create result buffer
        const result_buffer = try self.backend.createBuffer(@sizeOf(HitInfo), .storage);
        defer self.backend.destroyBuffer(result_buffer);

        // Create uniform buffer for ray parameters
        const uniform_buffer = try self.backend.createBuffer(@sizeOf(RayQuery), .uniform);
        defer self.backend.destroyBuffer(uniform_buffer);

        try self.backend.uploadData(uniform_buffer, std.mem.asBytes(&ray), 0);

        // Dispatch ray query compute shader
        try self.backend.dispatchCompute(
            shader_code,
            [_]u32{1, 1, 1}, // Single ray query
            &[_]gpu_backend.BufferHandle{
                self.acceleration_structure.?,
                uniform_buffer,
                result_buffer,
            },
            &[_]f32{} // No additional uniforms
        );

        // Download result
        var result: HitInfo = undefined;
        try self.backend.downloadData(
            result_buffer,
            std.mem.asBytes(&result),
            0
        );

        return result;
    }

    pub fn batchRayQuery(
        self: *const RTCores,
        rays: []const RayQuery,
        results: []HitInfo
    ) !void {
        if (self.acceleration_structure == null or rays.len != results.len) {
            return error.InvalidParameters;
        }

        // Create ray buffer
        const ray_buffer = try self.backend.createBuffer(
            rays.len * @sizeOf(RayQuery),
            .storage
        );
        defer self.backend.destroyBuffer(ray_buffer);

        try self.backend.uploadData(
            ray_buffer,
            std.mem.sliceAsBytes(rays),
            0
        );

        // Create result buffer
        const result_buffer = try self.backend.createBuffer(
            results.len * @sizeOf(HitInfo),
            .storage
        );
        defer self.backend.destroyBuffer(result_buffer);

        // Create ray query shader for batch processing
        const shader_code = try self.createBatchRayQueryShader();
        defer self.backend.destroyShader(shader_code);

        // Dispatch compute shader
        const workgroups = calculateWorkgroups(rays.len);
        try self.backend.dispatchCompute(
            shader_code,
            workgroups,
            &[_]gpu_backend.BufferHandle{
                self.acceleration_structure.?,
                ray_buffer,
                result_buffer,
            },
            &[_]f32{} // No additional uniforms
        );

        // Download results
        try self.backend.downloadData(
            result_buffer,
            std.mem.sliceAsBytes(results),
            0
        );
    }

    pub fn sphereQuery(
        self: *const RTCores,
        center: [3]f32,
        radius: f32
    ) ![]u32 {
        // Convert sphere query to ray queries against BVH
        // This is a simplified implementation - real RT cores would have sphere tracing

        var hits = std.ArrayList(u32).initCapacity(self.backend.allocator, 16) catch unreachable;
        defer hits.deinit();

        // Sample rays in a sphere pattern
        const ray_count = 64; // Adjust based on quality vs performance needs
        var rays: [ray_count]RayQuery = undefined;
        var ray_results: [ray_count]HitInfo = undefined;

        // Generate rays in spherical distribution
        for (0..ray_count) |i| {
            const phi = 2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / ray_count;
            const theta = std.math.acos(1.0 - 2.0 * @as(f32, @floatFromInt(i)) / ray_count);

            const direction = [_]f32{
                radius * std.math.sin(theta) * std.math.cos(phi),
                radius * std.math.sin(theta) * std.math.sin(phi),
                radius * std.math.cos(theta),
            };

            rays[i] = RayQuery{
                .origin = center,
                .direction = direction,
                .t_min = 0.0,
                .t_max = radius,
            };
        }

        // Execute batch ray query
        try self.batchRayQuery(&rays, &ray_results);

        // Collect unique hit primitive IDs
        var hit_set = std.AutoHashMap(u32, void).init(self.backend.allocator);
        defer hit_set.deinit();

        for (ray_results) |result| {
            if (result.hit) {
                _ = hit_set.put(result.primitive_id, {}) catch {};
            }
        }

        // Convert to array
        const final_hits = try self.backend.allocator.alloc(u32, hit_set.count());
        var i: usize = 0;
        var it = hit_set.iterator();
        while (it.next()) |entry| {
            final_hits[i] = entry.key_ptr.*;
            i += 1;
        }

        return final_hits;
    }

    // Private methods

    fn buildAccelerationStructure(
        self: *RTCores,
        vertex_buffer: gpu_backend.BufferHandle,
        index_buffer: gpu_backend.BufferHandle,
        primitive_count: u32
    ) !void {
        // Build Bottom Level Acceleration Structure (BLAS) using Vulkan RT

        // Get Vulkan device handles from backend
        const vk_device = @as(vk.VkDevice, @ptrFromInt(self.backend.device.vulkan.device));
        const vk_cmd_pool = @as(vk.VkCommandPool, @ptrFromInt(self.backend.command_pool.vulkan));

        // Get vertex and index buffer Vulkan handles
        const vertex_vk_buffer = @as(vk.VkBuffer, @ptrFromInt(vertex_buffer.vulkan));
        const index_vk_buffer = @as(vk.VkBuffer, @ptrFromInt(index_buffer.vulkan));

        // Define geometry for BLAS
        const triangles = vk.VkAccelerationStructureGeometryTrianglesDataKHR{
            .sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = vk.VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData = .{
                .deviceAddress = 0, // Will be set with vkGetBufferDeviceAddress
            },
            .vertexStride = 12, // 3 floats * 4 bytes
            .maxVertex = primitive_count * 3 - 1,
            .indexType = vk.VK_INDEX_TYPE_UINT32,
            .indexData = .{
                .deviceAddress = 0, // Will be set
            },
            .transformData = .{
                .deviceAddress = 0,
            },
        };

        const geometry = vk.VkAccelerationStructureGeometryKHR{
            .sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .flags = vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
            .geometry = .{
                .triangles = triangles,
            },
        };

        const build_range_info = vk.VkAccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = primitive_count,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0,
        };

        // Get build sizes
        var build_info = vk.VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .mode = vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };

        var build_sizes = vk.VkAccelerationStructureBuildSizesInfoKHR{
            .sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
        };

        vk.vkGetAccelerationStructureBuildSizesKHR(
            vk_device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_info,
            &build_range_info.primitiveCount,
            &build_sizes
        );

        // Create BLAS buffer
        self.acceleration_structure = try self.backend.createBuffer(build_sizes.accelerationStructureSize, .storage);

        // Create scratch buffer
        self.scratch_buffer = try self.backend.createBuffer(build_sizes.buildScratchSize, .storage);

        // Get device addresses
        const blas_vk_buffer = @as(vk.VkBuffer, @ptrFromInt(self.acceleration_structure.?.vulkan));
        const scratch_vk_buffer = @as(vk.VkBuffer, @ptrFromInt(self.scratch_buffer.?.vulkan));

        const vertex_address_info = vk.VkBufferDeviceAddressInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = vertex_vk_buffer,
        };
        const vertex_address = vk.vkGetBufferDeviceAddress(vk_device, &vertex_address_info);

        const index_address_info = vk.VkBufferDeviceAddressInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = index_vk_buffer,
        };
        const index_address = vk.vkGetBufferDeviceAddress(vk_device, &index_address_info);

        const scratch_address_info = vk.VkBufferDeviceAddressInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = scratch_vk_buffer,
        };
        const scratch_address = vk.vkGetBufferDeviceAddress(vk_device, &scratch_address_info);

        // Create acceleration structure
        const as_create_info = vk.VkAccelerationStructureCreateInfoKHR{
            .sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = blas_vk_buffer,
            .offset = 0,
            .size = build_sizes.accelerationStructureSize,
            .type = vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        };

        var acceleration_structure: vk.VkAccelerationStructureKHR = undefined;
        _ = vk.vkCreateAccelerationStructureKHR(vk_device, &as_create_info, null, &acceleration_structure);

        // Update build info with addresses
        triangles.vertexData.deviceAddress = vertex_address;
        triangles.indexData.deviceAddress = index_address;

        build_info.dstAccelerationStructure = acceleration_structure;
        build_info.scratchData.deviceAddress = scratch_address;

        // Build the acceleration structure
        const cmd_buffer_allocate_info = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk_cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var cmd_buffer: vk.VkCommandBuffer = undefined;
        _ = vk.vkAllocateCommandBuffers(vk_device, &cmd_buffer_allocate_info, &cmd_buffer);

        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        _ = vk.vkBeginCommandBuffer(cmd_buffer, &begin_info);

        const build_range_infos = [_]vk.VkAccelerationStructureBuildRangeInfoKHR{build_range_info};
        vk.vkCmdBuildAccelerationStructuresKHR(cmd_buffer, 1, &build_info, &build_range_infos);

        // Add memory barrier
        const barrier = vk.VkMemoryBarrier{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };

        vk.vkCmdPipelineBarrier(
            cmd_buffer,
            vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null
        );

        _ = vk.vkEndCommandBuffer(cmd_buffer);

        // Submit and wait
        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd_buffer,
        };

        _ = vk.vkQueueSubmit(@as(vk.VkQueue, @ptrFromInt(self.backend.queue.vulkan)), 1, &submit_info, null);
        _ = vk.vkQueueWaitIdle(@as(vk.VkQueue, @ptrFromInt(self.backend.queue.vulkan)));

        // Free command buffer
        vk.vkFreeCommandBuffers(vk_device, vk_cmd_pool, 1, &cmd_buffer);

        // Store acceleration structure handle (simplified - in real implementation, store in BufferHandle)
        // For now, we'll assume the buffer contains the AS

        self.bvh_build_info = BVHBuildInfo{
            .primitive_count = primitive_count,
            .max_primitive_count = primitive_count,
            .first_vertex = 0,
            .vertex_offset = 0,
            .transform_offset = 0,
            .flags = BuildFlags{},
        };
    }

    fn createRayQueryShader(self: *const RTCores) !gpu_backend.ShaderHandle {
        // SPIR-V code for single ray query
        const spirv_code = [_]u32{
            // This would be actual SPIR-V bytecode
            // For now, return a placeholder
        };

        return try self.backend.createComputeShader(&spirv_code);
    }

    fn createBatchRayQueryShader(self: *const RTCores) !gpu_backend.ShaderHandle {
        // SPIR-V code for batch ray queries
        const spirv_code = [_]u32{
            // This would be actual SPIR-V bytecode
            // For now, return a placeholder
        };

        return try self.backend.createComputeShader(&spirv_code);
    }

    fn calculateWorkgroups(ray_count: usize) [3]u32 {
        const workgroup_size = 64;
        const total_workgroups = (ray_count + workgroup_size - 1) / workgroup_size;
        return [_]u32{
            @intCast(total_workgroups),
            1,
            1,
        };
    }
};

test "RT cores basic functionality" {
    // Test compilation only - actual RT cores require Vulkan RT extension
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Mock backend
    const mock_backend = gpu_backend.GPUBackend{
        .allocator = allocator,
        .vendor = .nvidia,
        .api = .vulkan,
        .capabilities = .{
            .compute_shaders = true,
            .rt_cores = true,
            .tensor_cores = false,
            .cooperative_matrix = false,
            .shader_int8 = false,
            .shader_fp16 = true,
            .max_workgroup_size = 1024,
            .max_compute_workgroups = [_]u32{65535, 65535, 65535},
            .device_name = "RTX 3090",
        },
        .device = .{ .vulkan = .{ .instance = 1, .physical_device = 2, .device = 3 } },
        .queue = .{ .vulkan = 4 },
        .command_pool = .{ .vulkan = 5 },
    };

    var rt_cores = RTCores.init(&mock_backend);
    defer rt_cores.deinit();

    // Test ray query structure
    const ray = RTCores.RayQuery{
        .origin = [_]f32{0, 0, 0},
        .direction = [_]f32{0, 0, 1},
        .t_min = 0.0,
        .t_max = 100.0,
    };

    _ = ray;
    _ = rt_cores;
}
```
