/*
 * BrokenRecord Zero - OpenCL Kernels (Universal: AMD + NVIDIA)
 * 
 * Compile: gcc -O3 -o physics_opencl opencl_kernels.c -lOpenCL -lm
 */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// ============================================================================
// OpenCL Error Checking
// ============================================================================

#define CL_CHECK(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// OpenCL Kernel Source
// ============================================================================

const char* kernel_source = R"(
__kernel void integrate_kernel(
    __global float* pos_x,
    __global float* pos_y,
    __global float* pos_z,
    __global const float* vel_x,
    __global const float* vel_y,
    __global const float* vel_z,
    uint n,
    float dt
) {
    uint idx = get_global_id(0);
    if (idx >= n) return;
    
    pos_x[idx] += vel_x[idx] * dt;
    pos_y[idx] += vel_y[idx] * dt;
    pos_z[idx] += vel_z[idx] * dt;
}

__kernel void apply_gravity_kernel(
    __global float* vel_z,
    uint n,
    float g,
    float dt
) {
    uint idx = get_global_id(0);
    if (idx >= n) return;
    
    vel_z[idx] += g * dt;
}

__kernel void compute_energy_kernel(
    __global const float* pos_z,
    __global const float* vel_x,
    __global const float* vel_y,
    __global const float* vel_z,
    __global const float* mass,
    __global float* energy_output,
    __local float* shared_energy,
    uint n,
    float g
) {
    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    
    float local_val = 0.0f;
    
    if (global_id < n) {
        float vx = vel_x[global_id];
        float vy = vel_y[global_id];
        float vz = vel_z[global_id];
        float m = mass[global_id];
        float z = pos_z[global_id];
        
        float kinetic = 0.5f * m * (vx*vx + vy*vy + vz*vz);
        float potential = m * (-g) * z;
        local_val = kinetic + potential;
    }
    
    shared_energy[local_id] = local_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction
    for (uint s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_energy[local_id] += shared_energy[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        atomic_add_float(energy_output, shared_energy[0]);
    }
}

// Custom atomic add for float (may not be available on all devices)
void atomic_add_float(__global float* addr, float val) {
    union {
        uint u32;
        float f32;
    } next, expected, current;
    
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((__global uint*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}
)";

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    float *pos_x, *pos_y, *pos_z;
    float *vel_x, *vel_y, *vel_z;
    float *mass;
    uint32_t count;
    uint32_t capacity;
} ParticleSystem;

typedef struct {
    cl_mem d_pos_x, d_pos_y, d_pos_z;
    cl_mem d_vel_x, d_vel_y, d_vel_z;
    cl_mem d_mass;
    uint32_t count;
} GPUParticleSystem;

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel integrate_kernel;
    cl_kernel gravity_kernel;
    cl_kernel energy_kernel;
} OpenCLContext;

// ============================================================================
// OpenCL Initialization
// ============================================================================

OpenCLContext* init_opencl() {
    OpenCLContext *ctx = (OpenCLContext*)malloc(sizeof(OpenCLContext));
    cl_int err;
    
    // Get platform
    CL_CHECK(clGetPlatformIDs(1, &ctx->platform, NULL));
    
    // Get device (prefer GPU)
    err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
    if (err != CL_SUCCESS) {
        // Fallback to CPU
        CL_CHECK(clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_CPU, 1, &ctx->device, NULL));
    }
    
    // Print device info
    char device_name[128];
    char vendor_name[128];
    cl_ulong global_mem;
    cl_uint compute_units;
    
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    
    printf("OpenCL Device Information:\n");
    printf("  Vendor: %s\n", vendor_name);
    printf("  Device: %s\n", device_name);
    printf("  Memory: %.2f GB\n", global_mem / 1024.0 / 1024.0 / 1024.0);
    printf("  Compute Units: %u\n", compute_units);
    printf("\n");
    
    // Create context
    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    CL_CHECK(err);
    
    // Create command queue
    ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
    CL_CHECK(err);
    
    // Create program
    ctx->program = clCreateProgramWithSource(ctx->context, 1, &kernel_source, NULL, &err);
    CL_CHECK(err);
    
    // Build program
    err = clBuildProgram(ctx->program, 1, &ctx->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    // Create kernels
    ctx->integrate_kernel = clCreateKernel(ctx->program, "integrate_kernel", &err);
    CL_CHECK(err);
    ctx->gravity_kernel = clCreateKernel(ctx->program, "apply_gravity_kernel", &err);
    CL_CHECK(err);
    ctx->energy_kernel = clCreateKernel(ctx->program, "compute_energy_kernel", &err);
    CL_CHECK(err);
    
    return ctx;
}

void cleanup_opencl(OpenCLContext *ctx) {
    clReleaseKernel(ctx->integrate_kernel);
    clReleaseKernel(ctx->gravity_kernel);
    clReleaseKernel(ctx->energy_kernel);
    clReleaseProgram(ctx->program);
    clReleaseCommandQueue(ctx->queue);
    clReleaseContext(ctx->context);
    free(ctx);
}

// ============================================================================
// Host Functions
// ============================================================================

ParticleSystem* create_cpu_system(uint32_t capacity) {
    ParticleSystem *sys = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    
    sys->capacity = capacity;
    sys->count = 0;
    
    sys->pos_x = (float*)malloc(sizeof(float) * capacity);
    sys->pos_y = (float*)malloc(sizeof(float) * capacity);
    sys->pos_z = (float*)malloc(sizeof(float) * capacity);
    sys->vel_x = (float*)malloc(sizeof(float) * capacity);
    sys->vel_y = (float*)malloc(sizeof(float) * capacity);
    sys->vel_z = (float*)malloc(sizeof(float) * capacity);
    sys->mass = (float*)malloc(sizeof(float) * capacity);
    
    return sys;
}

void add_particle(ParticleSystem *sys, float px, float py, float pz,
                  float vx, float vy, float vz, float mass) {
    if (sys->count >= sys->capacity) return;
    
    uint32_t idx = sys->count;
    sys->pos_x[idx] = px;
    sys->pos_y[idx] = py;
    sys->pos_z[idx] = pz;
    sys->vel_x[idx] = vx;
    sys->vel_y[idx] = vy;
    sys->vel_z[idx] = vz;
    sys->mass[idx] = mass;
    sys->count++;
}

GPUParticleSystem* upload_to_gpu(OpenCLContext *ctx, ParticleSystem *cpu_sys) {
    GPUParticleSystem *gpu_sys = (GPUParticleSystem*)malloc(sizeof(GPUParticleSystem));
    gpu_sys->count = cpu_sys->count;
    
    size_t size = sizeof(float) * cpu_sys->count;
    cl_int err;
    
    gpu_sys->d_pos_x = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, cpu_sys->pos_x, &err);
    CL_CHECK(err);
    
    gpu_sys->d_pos_y = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, cpu_sys->pos_y, &err);
    CL_CHECK(err);
    
    gpu_sys->d_pos_z = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, cpu_sys->pos_z, &err);
    CL_CHECK(err);
    
    gpu_sys->d_vel_x = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, cpu_sys->vel_x, &err);
    CL_CHECK(err);
    
    gpu_sys->d_vel_y = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, cpu_sys->vel_y, &err);
    CL_CHECK(err);
    
    gpu_sys->d_vel_z = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, cpu_sys->vel_z, &err);
    CL_CHECK(err);
    
    gpu_sys->d_mass = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     size, cpu_sys->mass, &err);
    CL_CHECK(err);
    
    return gpu_sys;
}

void download_from_gpu(OpenCLContext *ctx, GPUParticleSystem *gpu_sys, ParticleSystem *cpu_sys) {
    size_t size = sizeof(float) * gpu_sys->count;
    
    CL_CHECK(clEnqueueReadBuffer(ctx->queue, gpu_sys->d_pos_x, CL_TRUE, 0, size, cpu_sys->pos_x, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ctx->queue, gpu_sys->d_pos_y, CL_TRUE, 0, size, cpu_sys->pos_y, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ctx->queue, gpu_sys->d_pos_z, CL_TRUE, 0, size, cpu_sys->pos_z, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ctx->queue, gpu_sys->d_vel_x, CL_TRUE, 0, size, cpu_sys->vel_x, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ctx->queue, gpu_sys->d_vel_y, CL_TRUE, 0, size, cpu_sys->vel_y, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ctx->queue, gpu_sys->d_vel_z, CL_TRUE, 0, size, cpu_sys->vel_z, 0, NULL, NULL));
}

void simulate_gpu(OpenCLContext *ctx, GPUParticleSystem *gpu_sys, float dt, uint32_t steps) {
    size_t global_size = gpu_sys->count;
    size_t local_size = 256;
    
    // Round up to multiple of local_size
    global_size = ((global_size + local_size - 1) / local_size) * local_size;
    
    float g = -9.81f;
    
    for (uint32_t step = 0; step < steps; step++) {
        // Apply gravity
        CL_CHECK(clSetKernelArg(ctx->gravity_kernel, 0, sizeof(cl_mem), &gpu_sys->d_vel_z));
        CL_CHECK(clSetKernelArg(ctx->gravity_kernel, 1, sizeof(uint32_t), &gpu_sys->count));
        CL_CHECK(clSetKernelArg(ctx->gravity_kernel, 2, sizeof(float), &g));
        CL_CHECK(clSetKernelArg(ctx->gravity_kernel, 3, sizeof(float), &dt));
        
        CL_CHECK(clEnqueueNDRangeKernel(ctx->queue, ctx->gravity_kernel, 1, NULL,
                                       &global_size, &local_size, 0, NULL, NULL));
        
        // Integrate
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 0, sizeof(cl_mem), &gpu_sys->d_pos_x));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 1, sizeof(cl_mem), &gpu_sys->d_pos_y));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 2, sizeof(cl_mem), &gpu_sys->d_pos_z));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 3, sizeof(cl_mem), &gpu_sys->d_vel_x));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 4, sizeof(cl_mem), &gpu_sys->d_vel_y));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 5, sizeof(cl_mem), &gpu_sys->d_vel_z));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 6, sizeof(uint32_t), &gpu_sys->count));
        CL_CHECK(clSetKernelArg(ctx->integrate_kernel, 7, sizeof(float), &dt));
        
        CL_CHECK(clEnqueueNDRangeKernel(ctx->queue, ctx->integrate_kernel, 1, NULL,
                                       &global_size, &local_size, 0, NULL, NULL));
    }
    
    CL_CHECK(clFinish(ctx->queue));
}

void free_gpu_system(GPUParticleSystem *gpu_sys) {
    clReleaseMemObject(gpu_sys->d_pos_x);
    clReleaseMemObject(gpu_sys->d_pos_y);
    clReleaseMemObject(gpu_sys->d_pos_z);
    clReleaseMemObject(gpu_sys->d_vel_x);
    clReleaseMemObject(gpu_sys->d_vel_y);
    clReleaseMemObject(gpu_sys->d_vel_z);
    clReleaseMemObject(gpu_sys->d_mass);
    free(gpu_sys);
}

void free_cpu_system(ParticleSystem *sys) {
    free(sys->pos_x);
    free(sys->pos_y);
    free(sys->pos_z);
    free(sys->vel_x);
    free(sys->vel_y);
    free(sys->vel_z);
    free(sys->mass);
    free(sys);
}

// ============================================================================
// Main Test
// ============================================================================

int main() {
    printf("\n");
    printf("================================================================================\n");
    printf("BrokenRecord Zero - OpenCL GPU Test (Universal)\n");
    printf("================================================================================\n\n");
    
    OpenCLContext *ctx = init_opencl();
    
    // Performance test
    uint32_t counts[] = {1000, 10000, 100000, 1000000};
    
    for (int c = 0; c < 4; c++) {
        uint32_t count = counts[c];
        
        printf("Testing %u particles:\n", count);
        
        // Create and populate system
        ParticleSystem *cpu_sys = create_cpu_system(count);
        
        for (uint32_t i = 0; i < count; i++) {
            float x = ((float)rand() / RAND_MAX) * 100.0f;
            float y = ((float)rand() / RAND_MAX) * 100.0f;
            float z = ((float)rand() / RAND_MAX) * 100.0f;
            add_particle(cpu_sys, x, y, z, 0, 0, 0, 1.0f);
        }
        
        // Upload to GPU
        struct timespec upload_start, upload_end;
        clock_gettime(CLOCK_MONOTONIC, &upload_start);
        GPUParticleSystem *gpu_sys = upload_to_gpu(ctx, cpu_sys);
        clock_gettime(CLOCK_MONOTONIC, &upload_end);
        
        double upload_time = (upload_end.tv_sec - upload_start.tv_sec) +
                            (upload_end.tv_nsec - upload_start.tv_nsec) / 1e9;
        
        // Benchmark GPU simulation
        struct timespec sim_start, sim_end;
        clock_gettime(CLOCK_MONOTONIC, &sim_start);
        
        simulate_gpu(ctx, gpu_sys, 0.01f, 1000);
        
        clock_gettime(CLOCK_MONOTONIC, &sim_end);
        
        double sim_time = (sim_end.tv_sec - sim_start.tv_sec) +
                         (sim_end.tv_nsec - sim_start.tv_nsec) / 1e9;
        
        // Download results
        struct timespec download_start, download_end;
        clock_gettime(CLOCK_MONOTONIC, &download_start);
        download_from_gpu(ctx, gpu_sys, cpu_sys);
        clock_gettime(CLOCK_MONOTONIC, &download_end);
        
        double download_time = (download_end.tv_sec - download_start.tv_sec) +
                              (download_end.tv_nsec - download_start.tv_nsec) / 1e9;
        
        double particles_per_sec = count * 1000 / sim_time;
        
        printf("  Upload time:   %.2fms\n", upload_time * 1000);
        printf("  Simulation:    %.2fms (1000 steps)\n", sim_time * 1000);
        printf("  Download time: %.2fms\n", download_time * 1000);
        printf("  Performance:   %.2f M particles/sec\n", particles_per_sec / 1e6);
        printf("  Time/step:     %.2fμs\n\n", sim_time / 1000 * 1e6);
        
        // Cleanup
        free_gpu_system(gpu_sys);
        free_cpu_system(cpu_sys);
    }
    
    cleanup_opencl(ctx);
    
    printf("================================================================================\n");
    printf("✓ GPU tests complete!\n");
    printf("================================================================================\n\n");
    
    return 0;
}