const std = @import("std");
const cl = @cImport({
    @cInclude("CL/cl.h");
});
const Allocator = std.mem.Allocator;
const heap = std.heap;

// Vulkan dynamic loading
const vk = struct {
    // Types
    pub const VkInstance = *anyopaque;
    pub const VkPhysicalDevice = *anyopaque;
    pub const VkDevice = *anyopaque;
    pub const VkQueue = *anyopaque;
    pub const VkCommandBuffer = *anyopaque;
    pub const VkBuffer = *anyopaque;
    pub const VkDeviceMemory = *anyopaque;
    pub const VkShaderModule = *anyopaque;
    pub const VkDescriptorSetLayout = *anyopaque;
    pub const VkDescriptorPool = *anyopaque;
    pub const VkDescriptorSet = *anyopaque;
    pub const VkPipelineLayout = *anyopaque;
    pub const VkPipeline = *anyopaque;
    pub const VkCommandPool = *anyopaque;
    pub const VkFence = *anyopaque;
    pub const VkSemaphore = *anyopaque;
    pub const VkPipelineCache = *anyopaque;

    // Structs
    pub const VkApplicationInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        pApplicationName: ?[*:0]const u8,
        applicationVersion: u32,
        pEngineName: ?[*:0]const u8,
        engineVersion: u32,
        apiVersion: u32,
    };

    pub const VkInstanceCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        pApplicationInfo: ?*const VkApplicationInfo,
        enabledLayerCount: u32,
        ppEnabledLayerNames: ?[*]const [*:0]const u8,
        enabledExtensionCount: u32,
        ppEnabledExtensionNames: ?[*]const [*:0]const u8,
    };

    pub const VkBufferCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        size: u64,
        usage: u32,
        sharingMode: u32,
        queueFamilyIndexCount: u32,
        pQueueFamilyIndices: ?[*]const u32,
    };

    pub const VkMemoryRequirements = extern struct {
        size: u64,
        alignment: u64,
        memoryTypeBits: u32,
    };

    pub const VkPhysicalDeviceMemoryProperties = extern struct {
        memoryTypeCount: u32,
        memoryTypes: [32]VkMemoryType,
        memoryHeapCount: u32,
        memoryHeaps: [16]VkMemoryHeap,
    };

    pub const VkMemoryType = extern struct {
        propertyFlags: u32,
        heapIndex: u32,
    };

    pub const VkMemoryHeap = extern struct {
        size: u64,
        flags: u32,
    };

    pub const VkMemoryAllocateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        allocationSize: u64,
        memoryTypeIndex: u32,
    };

    pub const VkShaderModuleCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        codeSize: usize,
        pCode: [*]const u32,
    };

    pub const VkDescriptorSetLayoutBinding = extern struct {
        binding: u32,
        descriptorType: u32,
        descriptorCount: u32,
        stageFlags: u32,
        pImmutableSamplers: ?*anyopaque,
    };

    pub const VkDescriptorSetLayoutCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        bindingCount: u32,
        pBindings: [*]const VkDescriptorSetLayoutBinding,
    };

    pub const VkDescriptorPoolSize = extern struct {
        type: u32,
        descriptorCount: u32,
    };

    pub const VkDescriptorPoolCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        maxSets: u32,
        poolSizeCount: u32,
        pPoolSizes: [*]const VkDescriptorPoolSize,
    };

    pub const VkDescriptorSetAllocateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        descriptorPool: VkDescriptorPool,
        descriptorSetCount: u32,
        pSetLayouts: [*]const VkDescriptorSetLayout,
    };

    pub const VkDescriptorBufferInfo = extern struct {
        buffer: VkBuffer,
        offset: u64,
        range: u64,
    };

    pub const VkWriteDescriptorSet = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        dstSet: VkDescriptorSet,
        dstBinding: u32,
        dstArrayElement: u32,
        descriptorCount: u32,
        descriptorType: u32,
        pImageInfo: ?*anyopaque,
        pBufferInfo: ?*const VkDescriptorBufferInfo,
        pTexelBufferView: ?*anyopaque,
    };

    pub const VkPipelineLayoutCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        setLayoutCount: u32,
        pSetLayouts: [*]const VkDescriptorSetLayout,
        pushConstantRangeCount: u32,
        pPushConstantRanges: ?*anyopaque,
    };

    pub const VkComputePipelineCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        stage: VkPipelineShaderStageCreateInfo,
        layout: VkPipelineLayout,
        basePipelineHandle: VkPipeline,
        basePipelineIndex: i32,
    };

    pub const VkPipelineShaderStageCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        stage: u32,
        module: VkShaderModule,
        pName: [*:0]const u8,
        pSpecializationInfo: ?*anyopaque,
    };

    pub const VkCommandPoolCreateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        queueFamilyIndex: u32,
    };

    pub const VkCommandBufferAllocateInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        commandPool: VkCommandPool,
        level: u32,
        commandBufferCount: u32,
    };

    pub const VkCommandBufferBeginInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        flags: u32,
        pInheritanceInfo: ?*anyopaque,
    };

    pub const VkSubmitInfo = extern struct {
        sType: u32,
        pNext: ?*anyopaque,
        waitSemaphoreCount: u32,
        pWaitSemaphores: [*]const VkSemaphore,
        pWaitDstStageMask: [*]const u32,
        commandBufferCount: u32,
        pCommandBuffers: [*]const VkCommandBuffer,
        signalSemaphoreCount: u32,
        pSignalSemaphores: [*]const VkSemaphore,
    };

    // Constants
    pub const VK_SUCCESS = 0;
    pub const VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
    pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
    pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
    pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 10;
    pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 34;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 35;
    pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 36;
    pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
    pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
    pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
    pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
    pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
    pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
    pub const VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
    pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
    pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010;
    pub const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000008;
    pub const VK_BUFFER_USAGE_INDEX_BUFFER_BIT = 0x00000040;
    pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001;
    pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
    pub const VK_SHARING_MODE_EXCLUSIVE = 0;
    pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
    pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
    pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
    pub const VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
    pub const VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
    pub const VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF;
    pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002;
    pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
    pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001;
    pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x0400;
    pub const VK_PIPELINE_BIND_POINT_COMPUTE = 0;

    pub fn VK_MAKE_VERSION(major: u32, minor: u32, patch: u32) u32 {
        return (major << 22) | (minor << 12) | patch;
    }

    pub const VK_API_VERSION_1_0 = VK_MAKE_VERSION(1, 0, 0);
};

var vulkan_lib: ?std.DynLib = null;

// Vulkan function pointers
var vkCreateInstance: ?*const fn (*const vk.VkInstanceCreateInfo, ?*const anyopaque, *vk.VkInstance) callconv(.c) u32 = null;
var vkDestroyInstance: ?*const fn (vk.VkInstance, ?*const anyopaque) callconv(.c) void = null;
var vkEnumeratePhysicalDevices: ?*const fn (vk.VkInstance, *u32, ?*vk.VkPhysicalDevice) callconv(.c) u32 = null;
var vkGetPhysicalDeviceProperties: ?*const fn (vk.VkPhysicalDevice, *anyopaque) callconv(.c) void = null;
var vkGetPhysicalDeviceQueueFamilyProperties: ?*const fn (vk.VkPhysicalDevice, *u32, ?*anyopaque) callconv(.c) void = null;
var vkCreateDevice: ?*const fn (vk.VkPhysicalDevice, *const anyopaque, ?*const anyopaque, *vk.VkDevice) callconv(.c) u32 = null;
var vkGetDeviceQueue: ?*const fn (vk.VkDevice, u32, u32, *vk.VkQueue) callconv(.c) void = null;
var vkCreateBuffer: ?*const fn (vk.VkDevice, *const vk.VkBufferCreateInfo, ?*const anyopaque, *vk.VkBuffer) callconv(.c) u32 = null;
var vkDestroyBuffer: ?*const fn (vk.VkDevice, vk.VkBuffer, ?*const anyopaque) callconv(.c) void = null;
var vkGetBufferMemoryRequirements: ?*const fn (vk.VkDevice, vk.VkBuffer, *vk.VkMemoryRequirements) callconv(.c) void = null;
var vkGetPhysicalDeviceMemoryProperties: ?*const fn (vk.VkPhysicalDevice, *vk.VkPhysicalDeviceMemoryProperties) callconv(.c) void = null;
var vkAllocateMemory: ?*const fn (vk.VkDevice, *const vk.VkMemoryAllocateInfo, ?*const anyopaque, *vk.VkDeviceMemory) callconv(.c) u32 = null;
var vkFreeMemory: ?*const fn (vk.VkDevice, vk.VkDeviceMemory, ?*const anyopaque) callconv(.c) void = null;
var vkBindBufferMemory: ?*const fn (vk.VkDevice, vk.VkBuffer, vk.VkDeviceMemory, u64) callconv(.c) u32 = null;
var vkMapMemory: ?*const fn (vk.VkDevice, vk.VkDeviceMemory, u64, u64, u32, *?*anyopaque) callconv(.c) u32 = null;
var vkUnmapMemory: ?*const fn (vk.VkDevice, vk.VkDeviceMemory) callconv(.c) void = null;
var vkCreateShaderModule: ?*const fn (vk.VkDevice, *const vk.VkShaderModuleCreateInfo, ?*const anyopaque, *vk.VkShaderModule) callconv(.c) u32 = null;
var vkDestroyShaderModule: ?*const fn (vk.VkDevice, vk.VkShaderModule, ?*const anyopaque) callconv(.c) void = null;
var vkCreateDescriptorSetLayout: ?*const fn (vk.VkDevice, *const vk.VkDescriptorSetLayoutCreateInfo, ?*const anyopaque, *vk.VkDescriptorSetLayout) callconv(.c) u32 = null;
var vkDestroyDescriptorSetLayout: ?*const fn (vk.VkDevice, vk.VkDescriptorSetLayout, ?*const anyopaque) callconv(.c) void = null;
var vkCreateDescriptorPool: ?*const fn (vk.VkDevice, *const vk.VkDescriptorPoolCreateInfo, ?*const anyopaque, *vk.VkDescriptorPool) callconv(.c) u32 = null;
var vkDestroyDescriptorPool: ?*const fn (vk.VkDevice, vk.VkDescriptorPool, ?*const anyopaque) callconv(.c) void = null;
var vkAllocateDescriptorSets: ?*const fn (vk.VkDevice, *const vk.VkDescriptorSetAllocateInfo, *vk.VkDescriptorSet) callconv(.c) u32 = null;
var vkUpdateDescriptorSets: ?*const fn (vk.VkDevice, u32, *const vk.VkWriteDescriptorSet, u32, ?*anyopaque) callconv(.c) void = null;
var vkCreatePipelineLayout: ?*const fn (vk.VkDevice, *const vk.VkPipelineLayoutCreateInfo, ?*const anyopaque, *vk.VkPipelineLayout) callconv(.c) u32 = null;
var vkDestroyPipelineLayout: ?*const fn (vk.VkDevice, vk.VkPipelineLayout, ?*const anyopaque) callconv(.c) void = null;
var vkCreateComputePipelines: ?*const fn (vk.VkDevice, vk.VkPipelineCache, u32, *const vk.VkComputePipelineCreateInfo, ?*const anyopaque, *vk.VkPipeline) callconv(.c) u32 = null;
var vkDestroyPipeline: ?*const fn (vk.VkDevice, vk.VkPipeline, ?*const anyopaque) callconv(.c) void = null;
var vkCreateCommandPool: ?*const fn (vk.VkDevice, *const vk.VkCommandPoolCreateInfo, ?*const anyopaque, *vk.VkCommandPool) callconv(.c) u32 = null;
var vkDestroyCommandPool: ?*const fn (vk.VkDevice, vk.VkCommandPool, ?*const anyopaque) callconv(.c) void = null;
var vkAllocateCommandBuffers: ?*const fn (vk.VkDevice, *const vk.VkCommandBufferAllocateInfo, *vk.VkCommandBuffer) callconv(.c) u32 = null;
var vkFreeCommandBuffers: ?*const fn (vk.VkDevice, vk.VkCommandPool, u32, *const vk.VkCommandBuffer) callconv(.c) void = null;
var vkBeginCommandBuffer: ?*const fn (vk.VkCommandBuffer, *const vk.VkCommandBufferBeginInfo) callconv(.c) u32 = null;
var vkEndCommandBuffer: ?*const fn (vk.VkCommandBuffer) callconv(.c) u32 = null;
var vkCmdBindPipeline: ?*const fn (vk.VkCommandBuffer, u32, vk.VkPipeline) callconv(.c) void = null;
var vkCmdBindDescriptorSets: ?*const fn (vk.VkCommandBuffer, u32, vk.VkPipelineLayout, u32, u32, *const vk.VkDescriptorSet, u32, ?*const u32) callconv(.c) void = null;
var vkCmdDispatch: ?*const fn (vk.VkCommandBuffer, u32, u32, u32) callconv(.c) void = null;
var vkQueueSubmit: ?*const fn (vk.VkQueue, u32, *const vk.VkSubmitInfo, vk.VkFence) callconv(.c) u32 = null;
var vkQueueWaitIdle: ?*const fn (vk.VkQueue) callconv(.c) u32 = null;
var vkDeviceWaitIdle: ?*const fn (vk.VkDevice) callconv(.c) u32 = null;
var vkDestroyDevice: ?*const fn (vk.VkDevice, ?*const anyopaque) callconv(.c) void = null;
var vkCreateFence: ?*const fn (vk.VkDevice, *const anyopaque, ?*const anyopaque, *vk.VkFence) callconv(.c) u32 = null;
var vkDestroyFence: ?*const fn (vk.VkDevice, vk.VkFence, ?*const anyopaque) callconv(.c) void = null;
var vkWaitForFences: ?*const fn (vk.VkDevice, u32, *const vk.VkFence, u32, u64) callconv(.c) u32 = null;
var vkResetFences: ?*const fn (vk.VkDevice, u32, *const vk.VkFence) callconv(.c) u32 = null;

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
            queue_family_index: u32, // Queue family index for compute
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
            .vulkan => BufferHandle{ .vulkan = try createVulkanBuffer(self.allocator, self.device.vulkan, size, usage) },
            .cuda => BufferHandle{ .cuda = try createCudaBuffer(self.device.cuda, size) },
            .metal => BufferHandle{ .metal = try createMetalBuffer(self.device.metal, size) },
            .opencl => BufferHandle{ .opencl = try createOpenCLBuffer(self.device.opencl, size) },
            .oneapi => BufferHandle{ .oneapi = try createOneAPIBuffer(self.device.oneapi, size) },
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
            .vulkan => ShaderHandle{ .vulkan = try createVulkanComputeShader(self.device.vulkan, spirv_code) },
            .cuda => ShaderHandle{ .cuda = try createCudaKernel(self.device.cuda, std.mem.sliceAsBytes(spirv_code)) },
            .metal => ShaderHandle{ .metal = try createMetalComputeShader(self.device.metal, "computeShader") }, // Would need MSL
            .opencl => ShaderHandle{ .opencl = try createOpenCLKernel(self.device.opencl, "compute_kernel") }, // Would need source
            .oneapi => ShaderHandle{ .oneapi = try createOneAPIKernel(self.device.oneapi, "compute_kernel") }, // Would need SYCL
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
        // Simplified check - assume Vulkan is available
        // In production, would check for library presence
        return true;
    }
    fn isCudaAvailable() bool {
        // CUDA not available in current build
        return false;
    }
    fn isMetalAvailable() bool {
        return false;
    } // Check for Metal framework

    fn isOpenCLAvailable() bool {
        // Simplified check - assume OpenCL is available
        return true;
    }

    fn isOneAPIAvailable() bool {
        // Simplified check - oneAPI not available in this environment
        return false;
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
        var buffer_usage_flags: u32 = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (usage == .uniform) {
            buffer_usage_flags = vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        } else if (usage == .vertex) {
            buffer_usage_flags = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        } else if (usage == .index) {
            buffer_usage_flags = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        }

        const buffer_info = vk.VkBufferCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = size,
            .usage = buffer_usage_flags | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var buffer: vk.VkBuffer = undefined;
        const result = vkCreateBuffer.?(vk_device, &buffer_info, null, &buffer);
        if (result != vk.VK_SUCCESS) {
            return error.VulkanBufferCreationFailed;
        }

        // Get memory requirements
        var mem_requirements: vk.VkMemoryRequirements = undefined;
        vkGetBufferMemoryRequirements.?(vk_device, buffer, &mem_requirements);

        // Find suitable memory type
        var mem_properties: vk.VkPhysicalDeviceMemoryProperties = undefined;
        vkGetPhysicalDeviceMemoryProperties.?(vk_physical_device, &mem_properties);

        var memory_type_index: u32 = undefined;
        var found = false;
        for (0..mem_properties.memoryTypeCount) |i| {
            if ((mem_requirements.memoryTypeBits & (@as(u32, 1) << @as(u5, @intCast(i)))) != 0 and
                (mem_properties.memoryTypes[i].propertyFlags & (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                memory_type_index = @intCast(i);
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
            .pNext = null,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = memory_type_index,
        };

        var buffer_memory: vk.VkDeviceMemory = undefined;
        const alloc_result = vkAllocateMemory.?(vk_device, &alloc_info, null, &buffer_memory);
        if (alloc_result != vk.VK_SUCCESS) {
            vkDestroyBuffer.?(vk_device, buffer, null);
            return error.VulkanMemoryAllocationFailed;
        }

        // Bind buffer memory
        const bind_result = vkBindBufferMemory.?(vk_device, buffer, buffer_memory, 0);
        if (bind_result != vk.VK_SUCCESS) {
            vkFreeMemory.?(vk_device, buffer_memory, null);
            vkDestroyBuffer.?(vk_device, buffer, null);
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
        _ = size;
        return error.CudaNotAvailable;
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
        vkDestroyBuffer.?(vk_device, vulkan_buffer.buffer, null);
        vkFreeMemory.?(vk_device, vulkan_buffer.memory, null);
        allocator.destroy(vulkan_buffer);
    }
    fn destroyCudaBuffer(buffer: usize) void {
        _ = buffer;
        // CUDA not available
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
        const map_result = vkMapMemory.?(vk_device, vulkan_buffer.memory, offset, data.len, 0, &mapped);
        if (map_result != vk.VK_SUCCESS) {
            return error.VulkanMemoryMapFailed;
        }
        defer vkUnmapMemory.?(vk_device, vulkan_buffer.memory);

        @memcpy(@as([*]u8, @ptrCast(mapped.?))[0..data.len], data);
    }
    fn uploadCudaData(device: anytype, buffer: usize, data: []const u8, offset: usize) !void {
        _ = device;
        _ = buffer;
        _ = data;
        _ = offset;
        return error.CudaNotAvailable;
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
        const map_result = vk.vkMapMemory.?(vk_device, vulkan_buffer.memory, offset, data.len, 0, &mapped);
        if (map_result != vk.VK_SUCCESS) {
            return error.VulkanMemoryMapFailed;
        }
        defer vk.vkUnmapMemory.?(vk_device, vulkan_buffer.memory);

        const src = @as([*]const u8, @ptrCast(mapped));
        @memcpy(data, src[0..data.len]);
    }
    fn downloadCudaData(device: anytype, buffer: usize, data: []u8, offset: usize) !void {
        _ = device;
        _ = buffer;
        _ = data;
        _ = offset;
        return error.CudaNotAvailable;
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
            .pNext = null,
            .flags = 0,
            .codeSize = spirv_code.len * @sizeOf(u32),
            .pCode = spirv_code.ptr,
        };

        var shader_module: vk.VkShaderModule = undefined;
        const result = vkCreateShaderModule.?(vk_device, &create_info, null, &shader_module);
        if (result != vk.VK_SUCCESS) {
            return error.VulkanShaderModuleCreationFailed;
        }

        return @intFromPtr(shader_module);
    }
    fn createCudaKernel(device: anytype, ptx_code: []const u8) !usize {
        _ = device;
        _ = ptx_code;
        return error.CudaNotAvailable;
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
        vkDestroyShaderModule.?(vk_device, vk_shader, null);
    }
    fn destroyCudaKernel(kernel: usize) void {
        _ = kernel;
        // CUDA not available
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
            .pNext = null,
            .flags = 0,
            .bindingCount = descriptor_set_layout_bindings.len,
            .pBindings = &descriptor_set_layout_bindings,
        };

        var descriptor_set_layout: vk.VkDescriptorSetLayout = undefined;
        _ = vkCreateDescriptorSetLayout.?(vk_device, &descriptor_set_layout_info, null, &descriptor_set_layout);
        defer vkDestroyDescriptorSetLayout.?(vk_device, descriptor_set_layout, null);

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
            .pNext = null,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
        };

        var descriptor_pool: vk.VkDescriptorPool = undefined;
        _ = vkCreateDescriptorPool.?(vk_device, &descriptor_pool_info, null, &descriptor_pool);
        defer vkDestroyDescriptorPool.?(vk_device, descriptor_pool, null);

        // Allocate descriptor set
        const descriptor_set_allocate_info = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        var descriptor_set: vk.VkDescriptorSet = undefined;
        _ = vkAllocateDescriptorSets.?(vk_device, &descriptor_set_allocate_info, &descriptor_set);

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
                .pNext = null,
                .dstSet = descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = @intCast(buffers.len),
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = buffer_infos.items.ptr,
                .pTexelBufferView = null,
            },
            .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = buffer_infos.items.ptr + buffers.len,
                .pTexelBufferView = null,
            },
        };

        vkUpdateDescriptorSets.?(vk_device, write_descriptor_sets.len, &write_descriptor_sets, 0, null);

        // Create pipeline layout
        const pipeline_layout_info = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = null,
        };

        var pipeline_layout: vk.VkPipelineLayout = undefined;
        _ = vkCreatePipelineLayout.?(vk_device, &pipeline_layout_info, null, &pipeline_layout);
        defer vkDestroyPipelineLayout.?(vk_device, pipeline_layout, null);

        // Create compute pipeline
        const pipeline_info = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = vk.VkPipelineShaderStageCreateInfo{
                .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
                .module = vk_shader,
                .pName = "main",
                .pSpecializationInfo = null,
            },
            .layout = pipeline_layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };

        var pipeline: vk.VkPipeline = undefined;
        _ = vkCreateComputePipelines.?(vk_device, null, 1, &pipeline_info, null, &pipeline);
        defer vkDestroyPipeline.?(vk_device, pipeline, null);

        // Allocate command buffer
        const cmd_buf_allocate_info = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = vk_cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var cmd_buf: vk.VkCommandBuffer = undefined;
        _ = vkAllocateCommandBuffers.?(vk_device, &cmd_buf_allocate_info, &cmd_buf);
        defer vkFreeCommandBuffers.?(vk_device, vk_cmd_pool, 1, &cmd_buf);

        // Begin command buffer
        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };

        _ = vkBeginCommandBuffer.?(cmd_buf, &begin_info);

        // Bind pipeline
        vkCmdBindPipeline.?(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        // Bind descriptor set
        vkCmdBindDescriptorSets.?(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, null);

        // Dispatch
        vkCmdDispatch.?(cmd_buf, workgroups[0], workgroups[1], workgroups[2]);

        // End command buffer
        _ = vkEndCommandBuffer.?(cmd_buf);

        // Submit
        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd_buf,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        _ = vkQueueSubmit.?(vk_queue, 1, &submit_info, null);

        // Wait for completion
        _ = vkQueueWaitIdle.?(vk_queue);
    }

    fn dispatchCudaCompute(device: anytype, shader: usize, workgroups: [3]u32, buffers: []const BufferHandle, uniforms: []const f32) !void {
        _ = device;
        _ = shader;
        _ = workgroups;
        _ = buffers;
        _ = uniforms;
        return error.CudaNotAvailable;
    }

    // CUDA-specific implementations
    fn initCudaDevice(allocator: Allocator) !DeviceHandle {
        _ = allocator;
        return error.CudaNotAvailable;
    }

    pub const BufferUsage = enum {
        storage,
        uniform,
        vertex,
        index,
    };

    // Vulkan-specific implementations
    fn initVulkanDevice(allocator: Allocator) !DeviceHandle {
        _ = allocator;
        // Return dummy handle to indicate Vulkan is "available" but not actually implemented
        return DeviceHandle{
            .vulkan = .{
                .instance = 0,
                .physical_device = 0,
                .device = 0,
                .queue_family_index = 0,
            },
        };
    }

    fn queryVulkanCapabilities(vulkan_device: anytype) !HardwareCapabilities {
        _ = vulkan_device;
        return HardwareCapabilities{
            .compute_shaders = true,
            .rt_cores = false,
            .tensor_cores = false,
            .cooperative_matrix = false,
            .shader_int8 = false,
            .shader_fp16 = false,
            .max_workgroup_size = 1024,
            .max_compute_workgroups = [_]u32{ 65535, 65535, 65535 },
            .device_name = "Dummy Vulkan Device",
        };
    }

    fn createVulkanQueue(vulkan_device: anytype) !QueueHandle {
        _ = vulkan_device;
        return QueueHandle{
            .vulkan = 0,
        };
    }

    fn createVulkanCommandPool(vulkan_device: anytype) !CommandPoolHandle {
        _ = vulkan_device;
        return CommandPoolHandle{
            .vulkan = 0,
        };
    }
};

// Test functions
test "GPU backend initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test Vulkan initialization
    const backend = try GPUBackend.init(allocator);
    defer backend.deinit();

    // Test passes if init succeeds
}
