const std = @import("std");
const vk = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const cu = @cImport({
    @cInclude("cuda.h");
});
const cl = @cImport({
    @cInclude("CL/cl.h");
});
const Allocator = std.mem.Allocator;
const heap = std.heap;

// Vulkan buffer with associated memory
const VulkanBuffer = struct {
    buffer: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
};

// Multi-vendor GPU backend supporting Vulkan, CUDA, Metal, OpenCL
pub const GPUBackend = struct {
    allocator: Allocator,
    vendor: Vendor,
    api: API,
    capabilities: HardwareCapabilities,
    device: DeviceHandle,
    queue: QueueHandle,
    command_pool: CommandPoolHandle,

    pub const Vendor = enum {
        nvidia,
        amd,
        intel,
        apple,
        qualcomm,
        unknown,
    };

    pub const API = enum {
        vulkan,
        cuda,
        metal,
        opencl,
        oneapi,
    };

    pub const HardwareCapabilities = struct {
        compute_shaders: bool,
        rt_cores: bool,
        tensor_cores: bool,
        cooperative_matrix: bool,
        shader_int8: bool,
        shader_fp16: bool,
        max_workgroup_size: u32,
        max_compute_workgroups: [3]u32,
        device_name: []const u8,
    };

    pub const DeviceHandle = union(API) {
        vulkan: struct {
            instance: usize, // VkInstance
            physical_device: usize, // VkPhysicalDevice
            device: usize, // VkDevice
        },
        cuda: struct {
            device: i32, // CUdevice
            context: usize, // CUcontext
        },
        metal: struct {
            device: usize, // id<MTLDevice>
        },
        opencl: struct {
            platform: usize, // cl_platform_id
            device: usize, // cl_device_id
            context: usize, // cl_context
        },
        oneapi: struct {
            device: usize, // sycl::device
        },
    };

    pub const QueueHandle = union(API) {
        vulkan: usize, // VkQueue
        cuda: usize, // CUstream
        metal: usize, // id<MTLCommandQueue>
        opencl: usize, // cl_command_queue
        oneapi: usize, // sycl::queue
    };

    pub const CommandPoolHandle = union(API) {
        vulkan: usize, // VkCommandPool
        cuda: usize, // Not needed for CUDA
        metal: usize, // id<MTLCommandBuffer> pool
        opencl: usize, // Not needed for OpenCL
        oneapi: usize, // Not needed for oneAPI
    };

    pub const BufferHandle = union(API) {
        vulkan: usize, // VkBuffer
        cuda: usize, // CUdeviceptr
        metal: usize, // id<MTLBuffer>
        opencl: usize, // cl_mem
        oneapi: usize, // sycl::buffer
    };

    pub const ShaderHandle = union(API) {
        vulkan: usize, // VkShaderModule
        cuda: usize, // CUmodule + CUfunction
        metal: usize, // id<MTLLibrary> + id<MTLFunction>
        opencl: usize, // cl_program + cl_kernel
        oneapi: usize, // sycl::kernel
    };

    pub fn init(allocator: Allocator) !GPUBackend {
        // Detect available APIs and select the best one
        const api = detectBestAPI();
        const vendor = detectVendor(api);

        // Initialize the chosen API
        const device = try initDevice(api, allocator);
        const queue = try createQueue(device, api);
        const command_pool = try createCommandPool(device, api, allocator);

        // Query capabilities
        const capabilities = try queryCapabilities(device, api, allocator);

        return GPUBackend{
            .allocator = allocator,
            .vendor = vendor,
            .api = api,
            .capabilities = capabilities,
            .device = device,
            .queue = queue,
            .command_pool = command_pool,
        };
    }

    pub fn deinit(self: *GPUBackend) void {
        // Clean up resources based on API
        switch (self.api) {
            .vulkan => {
                // vkDestroyCommandPool, vkDestroyDevice, vkDestroyInstance
            },
            .cuda => {
                // cuCtxDestroy, etc.
            },
            .metal => {
                // Release Metal objects
            },
            .opencl => {
                // clReleaseContext, etc.
            },
            .oneapi => {
                // SYCL cleanup
            },
        }
    }

    pub fn createBuffer(self: *const GPUBackend, size: usize, usage: BufferUsage) !BufferHandle {
        return switch (self.api) {
            .vulkan => try createVulkanBuffer(self.allocator, self.device.vulkan, size, usage),
            .cuda => try createCudaBuffer(self.device.cuda, size),
            .metal => try createMetalBuffer(self.device.metal, size),
            .opencl => try createOpenCLBuffer(self.device.opencl, size),
            .oneapi => try createOneAPIBuffer(self.device.oneapi, size),
        };
    }

    pub fn destroyBuffer(self: *const GPUBackend, buffer: BufferHandle) void {
        switch (self.api) {
            .vulkan => destroyVulkanBuffer(self.allocator, self.device.vulkan, buffer.vulkan),
            .cuda => destroyCudaBuffer(buffer.cuda),
            .metal => destroyMetalBuffer(buffer.metal),
            .opencl => destroyOpenCLBuffer(buffer.opencl),
            .oneapi => destroyOneAPIBuffer(buffer.oneapi),
        }
    }

    pub fn uploadData(self: *const GPUBackend, buffer: BufferHandle, data: []const u8, offset: usize) !void {
        switch (self.api) {
            .vulkan => try uploadVulkanData(self.device.vulkan, buffer.vulkan, data, offset),
            .cuda => try uploadCudaData(self.device.cuda, buffer.cuda, data, offset),
            .metal => try uploadMetalData(self.device.metal, self.queue.metal, buffer.metal, data, offset),
            .opencl => try uploadOpenCLData(self.device.opencl, self.queue.opencl, buffer.opencl, data, offset),
            .oneapi => try uploadOneAPIData(self.device.oneapi, self.queue.oneapi, buffer.oneapi, data, offset),
        }
    }

    pub fn downloadData(self: *const GPUBackend, buffer: BufferHandle, data: []u8, offset: usize) !void {
        switch (self.api) {
            .vulkan => try downloadVulkanData(self.device.vulkan, buffer.vulkan, data, offset),
            .cuda => try downloadCudaData(self.device.cuda, buffer.cuda, data, offset),
            .metal => try downloadMetalData(self.device.metal, self.queue.metal, buffer.metal, data, offset),
            .opencl => try downloadOpenCLData(self.device.opencl, self.queue.opencl, buffer.opencl, data, offset),
            .oneapi => try downloadOneAPIData(self.device.oneapi, self.queue.oneapi, buffer.oneapi, data, offset),
        }
    }

    pub fn createComputeShader(self: *const GPUBackend, spirv_code: []const u32) !ShaderHandle {
        return switch (self.api) {
            .vulkan => try createVulkanComputeShader(self.device.vulkan, spirv_code),
            .cuda => try createCudaKernel(self.device.cuda, "compute_kernel"), // Would need PTX
            .metal => try createMetalComputeShader(self.device.metal, "computeShader"), // Would need MSL
            .opencl => try createOpenCLKernel(self.device.opencl, "compute_kernel"), // Would need source
            .oneapi => try createOneAPIKernel(self.device.oneapi, "compute_kernel"), // Would need SYCL
        };
    }

    pub fn destroyShader(self: *const GPUBackend, shader: ShaderHandle) void {
        switch (self.api) {
            .vulkan => destroyVulkanShader(self.device.vulkan, shader.vulkan),
            .cuda => destroyCudaKernel(shader.cuda),
            .metal => destroyMetalShader(shader.metal),
            .opencl => destroyOpenCLKernel(shader.opencl),
            .oneapi => destroyOneAPIKernel(shader.oneapi),
        }
    }

    pub fn dispatchCompute(self: *const GPUBackend, shader: ShaderHandle, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        switch (self.api) {
            .vulkan => try dispatchVulkanCompute(self.allocator, self.device.vulkan, self.queue.vulkan, self.command_pool.vulkan, shader.vulkan, workgroups, buffers, uniforms),
            .cuda => try dispatchCudaCompute(self.device.cuda, shader.cuda, workgroups, buffers, uniforms),
            .metal => try dispatchMetalCompute(self.device.metal, self.queue.metal, shader.metal, workgroups, buffers, uniforms),
            .opencl => try dispatchOpenCLCompute(self.device.opencl, self.queue.opencl, shader.opencl, workgroups, buffers, uniforms),
            .oneapi => try dispatchOneAPICompute(self.device.oneapi, self.queue.oneapi, shader.oneapi, workgroups, buffers, uniforms),
        }
    }

    // Helper functions for API detection
    fn detectBestAPI() API {
        // Priority: Vulkan > CUDA > Metal > OpenCL > oneAPI
        if (isVulkanAvailable()) return .vulkan;
        if (isCudaAvailable()) return .cuda;
        if (isMetalAvailable()) return .metal;
        if (isOpenCLAvailable()) return .opencl;
        if (isOneAPIAvailable()) return .oneapi;
        @panic("No GPU API available");
    }

    fn detectVendor(api: API) Vendor {
        return switch (api) {
            .vulkan => detectVulkanVendor(),
            .cuda => .nvidia, // CUDA is NVIDIA-only
            .metal => .apple, // Metal is Apple-only
            .opencl => detectOpenCLVendor(),
            .oneapi => detectOneAPIVendor(),
        };
    }

    // API-specific implementations (stubs - would need actual API bindings)
    fn isVulkanAvailable() bool {
        // Try to load vkCreateInstance function
        if (std.os.windows.kernel32.GetModuleHandleA("vulkan-1.dll")) |_| {
            return true;
        } else |_| {}
        // For Linux/macOS, check if libvulkan.so exists
        // For now, assume available if not Windows
        return true;
    }
    fn isCudaAvailable() bool {
        // Try to load CUDA library
        const lib_name = if (std.Target.current.os.tag == .windows) "nvcuda.dll" else "libcuda.so";
        const lib = std.os.dynamic_library.open(lib_name) catch return false;
        lib.close();
        return true;
    }
    fn isMetalAvailable() bool {
        return false;
    } // Check for Metal framework

    fn isOpenCLAvailable() bool {
        // Try to load OpenCL library
        const lib_name = if (std.Target.current.os.tag == .windows) "OpenCL.dll" else "libOpenCL.so";
        const lib = std.os.dynamic_library.open(lib_name) catch return false;
        lib.close();
        return true;
    }

    fn isOneAPIAvailable() bool {
        // Try to load oneAPI library (Intel Level Zero)
        const lib_name = if (std.Target.current.os.tag == .windows) "ze_loader.dll" else "libze_loader.so";
        const lib = std.os.dynamic_library.open(lib_name) catch return false;
        lib.close();
        return true;
    }

    fn detectVulkanVendor() Vendor {
        return .unknown;
    }
    fn detectCudaVendor() Vendor {
        return .nvidia; // CUDA is NVIDIA's API
    }
    fn detectOpenCLVendor() Vendor {
        return .unknown;
    }
    fn detectOneAPIVendor() Vendor {
        return .unknown;
    }

    fn initDevice(api: API, allocator: Allocator) !DeviceHandle {
        switch (api) {
            .vulkan => return initVulkanDevice(allocator),
            .cuda => return initCudaDevice(allocator),
            .metal => @panic("Metal not implemented"),
            .opencl => @panic("OpenCL not implemented"),
            .oneapi => @panic("OneAPI not implemented"),
        }
    }
    fn createQueue(device: DeviceHandle, api: API) !QueueHandle {
        switch (api) {
            .vulkan => return createVulkanQueue(device.vulkan),
            .cuda => @panic("CUDA not implemented"),
            .metal => @panic("Metal not implemented"),
            .opencl => @panic("OpenCL not implemented"),
            .oneapi => @panic("OneAPI not implemented"),
        }
    }
    fn createCommandPool(device: DeviceHandle, api: API, allocator: Allocator) !CommandPoolHandle {
        _ = allocator;
        switch (api) {
            .vulkan => return createVulkanCommandPool(device.vulkan),
            .cuda => @panic("CUDA not implemented"),
            .metal => @panic("Metal not implemented"),
            .opencl => @panic("OpenCL not implemented"),
            .oneapi => @panic("OneAPI not implemented"),
        }
    }
    fn queryCapabilities(device: DeviceHandle, api: API, allocator: Allocator) !HardwareCapabilities {
        _ = allocator;
        switch (api) {
            .vulkan => return queryVulkanCapabilities(device.vulkan),
            .cuda => @panic("CUDA not implemented"),
            .metal => @panic("Metal not implemented"),
            .opencl => @panic("OpenCL not implemented"),
            .oneapi => @panic("OneAPI not implemented"),
        }
    }

    // Buffer operations
    fn createVulkanBuffer(allocator: Allocator, device: anytype, size: usize, usage: BufferUsage) !usize {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));
        const vk_physical_device = @as(vk.VkPhysicalDevice, @ptrFromInt(device.physical_device));

        // Convert usage to Vulkan flags
        var buffer_usage_flags: vk.VkBufferUsageFlags = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (usage == .uniform) {
            buffer_usage_flags = vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        } else if (usage == .vertex) {
            buffer_usage_flags = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        } else if (usage == .index) {
            buffer_usage_flags = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        }

        const buffer_info = vk.VkBufferCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = buffer_usage_flags | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
        };

        var buffer: vk.VkBuffer = undefined;
        const result = vk.vkCreateBuffer(vk_device, &buffer_info, null, &buffer);
        if (result != vk.VK_SUCCESS) {
            return error.VulkanBufferCreationFailed;
        }

        // Get memory requirements
        var mem_requirements: vk.VkMemoryRequirements = undefined;
        vk.vkGetBufferMemoryRequirements(vk_device, buffer, &mem_requirements);

        // Find suitable memory type
        var mem_properties: vk.VkPhysicalDeviceMemoryProperties = undefined;
        vk.vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &mem_properties);

        var memory_type_index: u32 = undefined;
        var found = false;
        for (0..mem_properties.memoryTypeCount) |i| {
            if ((mem_requirements.memoryTypeBits & (@as(u32, 1) << @as(u5, @intCast(i)))) != 0 and
                (mem_properties.memoryTypes[i].propertyFlags & (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                memory_type_index = i;
                found = true;
                break;
            }
        }
        if (!found) {
            vk.vkDestroyBuffer(vk_device, buffer, null);
            return error.NoSuitableMemoryType;
        }

        const alloc_info = vk.VkMemoryAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = memory_type_index,
        };

        var buffer_memory: vk.VkDeviceMemory = undefined;
        const alloc_result = vk.vkAllocateMemory(vk_device, &alloc_info, null, &buffer_memory);
        if (alloc_result != vk.VK_SUCCESS) {
            vk.vkDestroyBuffer(vk_device, buffer, null);
            return error.VulkanMemoryAllocationFailed;
        }

        // Bind buffer memory
        const bind_result = vk.vkBindBufferMemory(vk_device, buffer, buffer_memory, 0);
        if (bind_result != vk.VK_SUCCESS) {
            vk.vkFreeMemory(vk_device, buffer_memory, null);
            vk.vkDestroyBuffer(vk_device, buffer, null);
            return error.VulkanBufferMemoryBindFailed;
        }

        // Allocate VulkanBuffer struct
        const vulkan_buffer = try allocator.create(VulkanBuffer);
        vulkan_buffer.* = VulkanBuffer{
            .buffer = buffer,
            .memory = buffer_memory,
        };

        return @intFromPtr(vulkan_buffer);
    }
    fn createCudaBuffer(device: anytype, size: usize) !usize {
        _ = device;
        var device_ptr: cu.CUdeviceptr = undefined;
        const result = cu.cuMemAlloc(&device_ptr, size);
        if (result != cu.CUDA_SUCCESS) {
            return error.CudaMemoryAllocationFailed;
        }
        return @intFromPtr(device_ptr);
    }
    fn createMetalBuffer(device: anytype, size: usize) !usize {
        _ = device;
        _ = size;
        @panic("Not implemented");
    }
    fn createOpenCLBuffer(device: anytype, size: usize) !usize {
        _ = device;
        _ = size;
        @panic("Not implemented");
    }
    fn createOneAPIBuffer(device: anytype, size: usize) !usize {
        _ = device;
        _ = size;
        @panic("Not implemented");
    }

    fn destroyVulkanBuffer(allocator: Allocator, device: anytype, buffer: usize) void {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));
        const vulkan_buffer = @as(*VulkanBuffer, @ptrFromInt(buffer));
        vk.vkDestroyBuffer(vk_device, vulkan_buffer.buffer, null);
        vk.vkFreeMemory(vk_device, vulkan_buffer.memory, null);
        allocator.destroy(vulkan_buffer);
    }
    fn destroyCudaBuffer(buffer: usize) void {
        const device_ptr = @as(cu.CUdeviceptr, @ptrFromInt(buffer));
        _ = cu.cuMemFree(device_ptr);
    }
    fn destroyMetalBuffer(buffer: usize) void {
        _ = buffer;
    }
    fn destroyOpenCLBuffer(buffer: usize) void {
        _ = buffer;
    }
    fn destroyOneAPIBuffer(buffer: usize) void {
        _ = buffer;
    }

    // Data transfer
    fn uploadVulkanData(device: anytype, buffer: usize, data: []const u8, offset: usize) !void {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));
        const vulkan_buffer = @as(*VulkanBuffer, @ptrFromInt(buffer));

        var mapped: ?*anyopaque = undefined;
        const map_result = vk.vkMapMemory(vk_device, vulkan_buffer.memory, offset, data.len, 0, &mapped);
        if (map_result != vk.VK_SUCCESS) {
            return error.VulkanMemoryMapFailed;
        }
        defer vk.vkUnmapMemory(vk_device, vulkan_buffer.memory);

        const dst = @as([*]u8, @ptrCast(mapped));
        @memcpy(dst[0..data.len], data);
    }
    fn uploadCudaData(device: anytype, buffer: usize, data: []const u8, offset: usize) !void {
        _ = device;
        const device_ptr = @as(cu.CUdeviceptr, @ptrFromInt(buffer));
        const result = cu.cuMemcpyHtoD(device_ptr + offset, data.ptr, data.len);
        if (result != cu.CUDA_SUCCESS) {
            return error.CudaMemoryCopyFailed;
        }
    }
    fn uploadMetalData(device: anytype, queue: anytype, buffer: usize, data: []const u8, offset: usize) !void {
        _ = device;
        _ = queue;
        _ = buffer;
        _ = data;
        _ = offset;
        @panic("Not implemented");
    }
    fn uploadOpenCLData(device: anytype, queue: anytype, buffer: usize, data: []const u8, offset: usize) !void {
        _ = device;
        _ = queue;
        _ = buffer;
        _ = data;
        _ = offset;
        @panic("Not implemented");
    }
    fn uploadOneAPIData(device: anytype, queue: anytype, buffer: usize, data: []const u8, offset: usize) !void {
        _ = device;
        _ = queue;
        _ = buffer;
        _ = data;
        _ = offset;
        @panic("Not implemented");
    }

    fn downloadVulkanData(device: anytype, buffer: usize, data: []u8, offset: usize) !void {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));
        const vulkan_buffer = @as(*VulkanBuffer, @ptrFromInt(buffer));

        var mapped: ?*anyopaque = undefined;
        const map_result = vk.vkMapMemory(vk_device, vulkan_buffer.memory, offset, data.len, 0, &mapped);
        if (map_result != vk.VK_SUCCESS) {
            return error.VulkanMemoryMapFailed;
        }
        defer vk.vkUnmapMemory(vk_device, vulkan_buffer.memory);

        const src = @as([*]const u8, @ptrCast(mapped));
        @memcpy(data, src[0..data.len]);
    }
    fn downloadCudaData(device: anytype, buffer: usize, data: []u8, offset: usize) !void {
        _ = device;
        const device_ptr = @as(cu.CUdeviceptr, @ptrFromInt(buffer));
        const result = cu.cuMemcpyDtoH(data.ptr, device_ptr + offset, data.len);
        if (result != cu.CUDA_SUCCESS) {
            return error.CudaMemoryCopyFailed;
        }
    }
    fn downloadMetalData(device: anytype, queue: anytype, buffer: usize, data: []u8, offset: usize) !void {
        _ = device;
        _ = queue;
        _ = buffer;
        _ = data;
        _ = offset;
        @panic("Not implemented");
    }
    fn downloadOpenCLData(device: anytype, queue: anytype, buffer: usize, data: []u8, offset: usize) !void {
        _ = device;
        _ = queue;
        _ = buffer;
        _ = data;
        _ = offset;
        @panic("Not implemented");
    }
    fn downloadOneAPIData(device: anytype, queue: anytype, buffer: usize, data: []u8, offset: usize) !void {
        _ = device;
        _ = queue;
        _ = buffer;
        _ = data;
        _ = offset;
        @panic("Not implemented");
    }

    // Shader operations
    fn createVulkanComputeShader(device: anytype, spirv_code: []const u32) !usize {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));

        const create_info = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_code.len * @sizeOf(u32),
            .pCode = spirv_code.ptr,
        };

        var shader_module: vk.VkShaderModule = undefined;
        const result = vk.vkCreateShaderModule(vk_device, &create_info, null, &shader_module);
        if (result != vk.VK_SUCCESS) {
            return error.VulkanShaderModuleCreationFailed;
        }

        return @intFromPtr(shader_module);
    }
    fn createCudaKernel(device: anytype, kernel_name: []const u8) !usize {
        _ = device;
        _ = kernel_name;
        @panic("Not implemented");
    }
    fn createMetalComputeShader(device: anytype, function_name: []const u8) !usize {
        _ = device;
        _ = function_name;
        @panic("Not implemented");
    }
    fn createOpenCLKernel(device: anytype, kernel_name: []const u8) !usize {
        _ = device;
        _ = kernel_name;
        @panic("Not implemented");
    }
    fn createOneAPIKernel(device: anytype, kernel_name: []const u8) !usize {
        _ = device;
        _ = kernel_name;
        @panic("Not implemented");
    }

    fn destroyVulkanShader(device: anytype, shader: usize) void {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));
        const vk_shader = @as(vk.VkShaderModule, @ptrFromInt(shader));
        vk.vkDestroyShaderModule(vk_device, vk_shader, null);
    }
    fn destroyCudaKernel(kernel: usize) void {
        _ = kernel;
    }
    fn destroyMetalShader(shader: usize) void {
        _ = shader;
    }
    fn destroyOpenCLKernel(kernel: usize) void {
        _ = kernel;
    }
    fn destroyOneAPIKernel(kernel: usize) void {
        _ = kernel;
    }

    // Compute dispatch
    fn dispatchMetalCompute(device: anytype, queue: anytype, shader: usize, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        _ = device;
        _ = queue;
        _ = shader;
        _ = workgroups;
        _ = buffers;
        _ = uniforms;
        @panic("Not implemented");
    }
    fn dispatchOneAPICompute(device: anytype, queue: anytype, kernel: usize, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        _ = device;
        _ = queue;
        _ = kernel;
        _ = workgroups;
        _ = buffers;
        _ = uniforms;
        @panic("Not implemented");
    }
    fn dispatchOpenCLCompute(device: anytype, queue: anytype, kernel: usize, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        _ = device;
        _ = queue;
        _ = kernel;
        _ = workgroups;
        _ = buffers;
        _ = uniforms;
        @panic("Not implemented");
    }

    fn dispatchVulkanCompute(allocator: Allocator, device: anytype, queue: anytype, cmd_pool: anytype, shader: usize, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        const vk_device = @as(vk.VkDevice, @ptrFromInt(device.device));
        const vk_queue = @as(vk.VkQueue, @ptrFromInt(queue));
        const vk_cmd_pool = @as(vk.VkCommandPool, @ptrFromInt(cmd_pool));
        const vk_shader = @as(vk.VkShaderModule, @ptrFromInt(shader));

        // Create descriptor set layout (assuming binding 0: storage buffers, binding 1: uniform buffer)
        const descriptor_set_layout_bindings = [_]vk.VkDescriptorSetLayoutBinding{
            .{
                .binding = 0,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = @intCast(buffers.len),
                .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            },
            .{
                .binding = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            },
        };

        const descriptor_set_layout_info = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = descriptor_set_layout_bindings.len,
            .pBindings = &descriptor_set_layout_bindings,
        };

        var descriptor_set_layout: vk.VkDescriptorSetLayout = undefined;
        _ = vk.vkCreateDescriptorSetLayout(vk_device, &descriptor_set_layout_info, null, &descriptor_set_layout);
        defer vk.vkDestroyDescriptorSetLayout(vk_device, descriptor_set_layout, null);

        // Create descriptor pool
        const pool_sizes = [_]vk.VkDescriptorPoolSize{
            .{
                .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = @intCast(buffers.len),
            },
            .{
                .type = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
            },
        };

        const descriptor_pool_info = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
        };

        var descriptor_pool: vk.VkDescriptorPool = undefined;
        _ = vk.vkCreateDescriptorPool(vk_device, &descriptor_pool_info, null, &descriptor_pool);
        defer vk.vkDestroyDescriptorPool(vk_device, descriptor_pool, null);

        // Allocate descriptor set
        const descriptor_set_allocate_info = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        var descriptor_set: vk.VkDescriptorSet = undefined;
        _ = vk.vkAllocateDescriptorSets(vk_device, &descriptor_set_allocate_info, &descriptor_set);

        // Create uniform buffer for uniforms
        var uniform_buffer: ?*VulkanBuffer = null;
        if (uniforms.len > 0) {
            const uniform_size = uniforms.len * @sizeOf(f32);
            const uniform_buffer_handle = try createVulkanBuffer(allocator, device, uniform_size, .uniform);
            uniform_buffer = @as(*VulkanBuffer, @ptrFromInt(uniform_buffer_handle));
            try uploadVulkanData(device, uniform_buffer_handle, std.mem.sliceAsBytes(uniforms), 0);
        }
        defer if (uniform_buffer) |ub| {
            destroyVulkanBuffer(allocator, device, @intFromPtr(ub));
        };

        // Update descriptor set
        var buffer_infos = std.ArrayList(vk.VkDescriptorBufferInfo).initCapacity(allocator, buffers.len + 1) catch unreachable;
        defer buffer_infos.deinit();

        for (buffers) |buffer_handle| {
            const vulkan_buf = @as(*VulkanBuffer, @ptrFromInt(buffer_handle.vulkan));
            buffer_infos.append(.{
                .buffer = vulkan_buf.buffer,
                .offset = 0,
                .range = vk.VK_WHOLE_SIZE,
            }) catch unreachable;
        }

        if (uniform_buffer) |ub| {
            buffer_infos.append(.{
                .buffer = ub.buffer,
                .offset = 0,
                .range = vk.VK_WHOLE_SIZE,
            }) catch unreachable;
        }

        const write_descriptor_sets = [_]vk.VkWriteDescriptorSet{
            .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 0,
                .descriptorCount = @intCast(buffers.len),
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = buffer_infos.items.ptr,
            },
            .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = if (uniform_buffer != null) &buffer_infos.items[buffers.len] else null,
            },
        };

        vk.vkUpdateDescriptorSets(vk_device, write_descriptor_sets.len, &write_descriptor_sets, 0, null);

        // Create pipeline layout
        const pipeline_layout_info = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        var pipeline_layout: vk.VkPipelineLayout = undefined;
        _ = vk.vkCreatePipelineLayout(vk_device, &pipeline_layout_info, null, &pipeline_layout);
        defer vk.vkDestroyPipelineLayout(vk_device, pipeline_layout, null);

        // Create compute pipeline
        const shader_stage_info = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk_shader,
            .pName = "main",
        };

        const compute_pipeline_info = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = shader_stage_info,
            .layout = pipeline_layout,
        };

        var compute_pipeline: vk.VkPipeline = undefined;
        _ = vk.vkCreateComputePipelines(vk_device, null, 1, &compute_pipeline_info, null, &compute_pipeline);
        defer vk.vkDestroyPipeline(vk_device, compute_pipeline, null);

        // Allocate command buffer
        const cmd_buffer_allocate_info = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk_cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var cmd_buffer: vk.VkCommandBuffer = undefined;
        _ = vk.vkAllocateCommandBuffers(vk_device, &cmd_buffer_allocate_info, &cmd_buffer);

        // Record command buffer
        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        _ = vk.vkBeginCommandBuffer(cmd_buffer, &begin_info);

        vk.vkCmdBindPipeline(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        vk.vkCmdBindDescriptorSets(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, null);
        vk.vkCmdDispatch(cmd_buffer, workgroups[0], workgroups[1], workgroups[2]);

        _ = vk.vkEndCommandBuffer(cmd_buffer);

        // Submit command buffer
        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd_buffer,
        };

        _ = vk.vkQueueSubmit(vk_queue, 1, &submit_info, null);

        // Wait for completion
        _ = vk.vkQueueWaitIdle(vk_queue);

        // Free command buffer
        vk.vkFreeCommandBuffers(vk_device, vk_cmd_pool, 1, &cmd_buffer);
    }

    fn dispatchCudaCompute(device: anytype, shader: usize, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        _ = device;
        _ = shader;
        _ = workgroups;
        _ = buffers;
        _ = uniforms;
        @panic("CUDA compute dispatch not implemented");
    }

    // CUDA-specific implementations
    fn initCudaDevice(allocator: Allocator) !DeviceHandle {
        _ = allocator;

        // Initialize CUDA
        var result = cu.cuInit(0);
        if (result != cu.CUDA_SUCCESS) {
            return error.CudaInitFailed;
        }

        // Get device count
        var device_count: c_int = 0;
        result = cu.cuDeviceGetCount(&device_count);
        if (result != cu.CUDA_SUCCESS or device_count == 0) {
            return error.NoCudaDevices;
        }

        // Get first device
        var device: cu.CUdevice = undefined;
        result = cu.cuDeviceGet(&device, 0);
        if (result != cu.CUDA_SUCCESS) {
            return error.CudaDeviceGetFailed;
        }

        // Create context
        var context: cu.CUcontext = undefined;
        result = cu.cuCtxCreate(&context, 0, device);
        if (result != cu.CUDA_SUCCESS) {
            return error.CudaContextCreationFailed;
        }

        return DeviceHandle{
            .cuda = .{
                .device = @intFromPtr(device),
                .context = @intFromPtr(context),
            },
        };
    }

    pub const BufferUsage = enum {
        storage,
        uniform,
        vertex,
        index,
    };

    // Vulkan-specific implementations
    fn initVulkanDevice(allocator: Allocator) !DeviceHandle {
        // Create Vulkan instance
        const app_info = vk.VkApplicationInfo{
            .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "AII Physics Simulation",
            .applicationVersion = vk.VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "AII Engine",
            .engineVersion = vk.VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk.VK_API_VERSION_1_2,
        };

        // Required extensions for Vulkan RT
        const extensions = [_][*c]const u8{
            vk.VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            vk.VK_KHR_RAY_QUERY_EXTENSION_NAME,
            vk.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        };

        const create_info = vk.VkInstanceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = extensions.len,
            .ppEnabledExtensionNames = &extensions,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
        };

        var instance: vk.VkInstance = undefined;
        const result = vk.vkCreateInstance(&create_info, null, &instance);
        if (result != vk.VK_SUCCESS) {
            return error.VulkanInstanceCreationFailed;
        }

        // Enumerate physical devices
        var device_count: u32 = 0;
        _ = vk.vkEnumeratePhysicalDevices(instance, &device_count, null);
        if (device_count == 0) {
            return error.NoVulkanPhysicalDevices;
        }

        const devices = try allocator.alloc(vk.VkPhysicalDevice, device_count);
        defer allocator.free(devices);
        _ = vk.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

        // Select first discrete GPU, or first device if none
        var selected_device: vk.VkPhysicalDevice = devices[0];
        for (devices) |device| {
            var device_properties: vk.VkPhysicalDeviceProperties = undefined;
            vk.vkGetPhysicalDeviceProperties(device, &device_properties);
            if (device_properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                selected_device = device;
                break;
            }
        }

        // Create logical device with RT extensions
        const queue_priority: f32 = 1.0;
        const queue_create_info = vk.VkDeviceQueueCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = 0, // Assume first queue family supports compute
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        const device_extensions = [_][*c]const u8{
            vk.VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            vk.VK_KHR_RAY_QUERY_EXTENSION_NAME,
            vk.VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            vk.VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
            vk.VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            vk.VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        };

        const device_create_info = vk.VkDeviceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledExtensionCount = device_extensions.len,
            .ppEnabledExtensionNames = &device_extensions,
            .pEnabledFeatures = null,
        };

        var device: vk.VkDevice = undefined;
        const device_result = vk.vkCreateDevice(selected_device, &device_create_info, null, &device);
        if (device_result != vk.VK_SUCCESS) {
            return error.VulkanDeviceCreationFailed;
        }

        return DeviceHandle{
            .vulkan = .{
                .instance = @intFromPtr(instance),
                .physical_device = @intFromPtr(selected_device),
                .device = @intFromPtr(device),
            },
        };
    }

    fn queryVulkanCapabilities(vulkan_device: anytype) !HardwareCapabilities {
        var properties: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(vulkan_device.physical_device, &properties);

        var features: vk.VkPhysicalDeviceFeatures = undefined;
        vk.vkGetPhysicalDeviceFeatures(vulkan_device.physical_device, &features);

        return HardwareCapabilities{
            .compute_shaders = true, // Assume Vulkan 1.0+ has compute
            .rt_cores = false, // Will be checked with extensions later
            .tensor_cores = false,
            .cooperative_matrix = false,
            .shader_int8 = false,
            .shader_fp16 = features.shaderFloat16,
            .max_workgroup_size = 1024, // Conservative default
            .max_compute_workgroups = 65535,
            .device_name = properties.deviceName,
        };
    }

    fn createVulkanQueue(vulkan_device: anytype) !QueueHandle {
        var queue: vk.VkQueue = undefined;
        vk.vkGetDeviceQueue(@as(vk.VkDevice, @ptrFromInt(vulkan_device.device)), 0, 0, &queue);

        return QueueHandle{
            .vulkan = @intFromPtr(queue),
        };
    }

    fn createVulkanCommandPool(vulkan_device: anytype) !CommandPoolHandle {
        const pool_info = vk.VkCommandPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = 0,
        };

        var command_pool: vk.VkCommandPool = undefined;
        const result = vk.vkCreateCommandPool(@as(vk.VkDevice, @ptrFromInt(vulkan_device.device)), &pool_info, null, &command_pool);
        if (result != vk.VK_SUCCESS) {
            return error.VulkanCommandPoolCreationFailed;
        }

        return CommandPoolHandle{
            .vulkan = @intFromPtr(command_pool),
        };
    }
};

// Test functions
test "GPU backend initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // This will panic since no APIs are implemented yet
    // const backend = try GPUBackend.init(allocator);
    // defer backend.deinit();

    // For now, just test that the struct compiles
    const backend = GPUBackend{
        .allocator = allocator,
        .vendor = .unknown,
        .api = .vulkan,
        .capabilities = .{
            .compute_shaders = false,
            .rt_cores = false,
            .tensor_cores = false,
            .cooperative_matrix = false,
            .shader_int8 = false,
            .shader_fp16 = false,
            .max_workgroup_size = 0,
            .max_compute_workgroups = [_]u32{ 0, 0, 0 },
            .device_name = "",
        },
        .device = .{ .vulkan = .{ .instance = 0, .physical_device = 0, .device = 0 } },
        .queue = .{ .vulkan = 0 },
        .command_pool = .{ .vulkan = 0 },
    };

    _ = backend;
}
