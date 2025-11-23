/*
 * Standalone test for BrokenRecord Zero physics engine
 * Compile: gcc -O3 -march=native -o test_physics test_physics.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef __AVX__
#include <immintrin.h>
#define SIMD_ENABLED 1
#else
#define SIMD_ENABLED 0
#endif

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

// ============================================================================
// System Management
// ============================================================================

ParticleSystem* create_system(uint32_t capacity) {
    ParticleSystem *sys = malloc(sizeof(ParticleSystem));
    if (!sys) return NULL;
    
    sys->capacity = capacity;
    sys->count = 0;
    
    sys->pos_x = malloc(sizeof(float) * capacity);
    sys->pos_y = malloc(sizeof(float) * capacity);
    sys->pos_z = malloc(sizeof(float) * capacity);
    sys->vel_x = malloc(sizeof(float) * capacity);
    sys->vel_y = malloc(sizeof(float) * capacity);
    sys->vel_z = malloc(sizeof(float) * capacity);
    sys->mass = malloc(sizeof(float) * capacity);
    
    if (!sys->pos_x || !sys->pos_y || !sys->pos_z || 
        !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass) {
        free(sys->pos_x);
        free(sys->pos_y);
        free(sys->pos_z);
        free(sys->vel_x);
        free(sys->vel_y);
        free(sys->vel_z);
        free(sys->mass);
        free(sys);
        return NULL;
    }
    
    return sys;
}

void destroy_system(ParticleSystem *sys) {
    if (!sys) return;
    free(sys->pos_x);
    free(sys->pos_y);
    free(sys->pos_z);
    free(sys->vel_x);
    free(sys->vel_y);
    free(sys->vel_z);
    free(sys->mass);
    free(sys);
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

// ============================================================================
// Physics Kernels
// ============================================================================

static void integrate_euler(ParticleSystem *sys, float dt) {
    const uint32_t n = sys->count;
    
#ifdef __AVX__
    // SIMD version
    const uint32_t n_simd = n - (n % 8);
    const __m256 dt_vec = _mm256_set1_ps(dt);
    
    for (uint32_t i = 0; i < n_simd; i += 8) {
        __m256 px = _mm256_loadu_ps(&sys->pos_x[i]);
        __m256 py = _mm256_loadu_ps(&sys->pos_y[i]);
        __m256 pz = _mm256_loadu_ps(&sys->pos_z[i]);
        
        __m256 vx = _mm256_loadu_ps(&sys->vel_x[i]);
        __m256 vy = _mm256_loadu_ps(&sys->vel_y[i]);
        __m256 vz = _mm256_loadu_ps(&sys->vel_z[i]);
        
        px = _mm256_fmadd_ps(vx, dt_vec, px);
        py = _mm256_fmadd_ps(vy, dt_vec, py);
        pz = _mm256_fmadd_ps(vz, dt_vec, pz);
        
        _mm256_storeu_ps(&sys->pos_x[i], px);
        _mm256_storeu_ps(&sys->pos_y[i], py);
        _mm256_storeu_ps(&sys->pos_z[i], pz);
    }
    
    for (uint32_t i = n_simd; i < n; i++) {
        sys->pos_x[i] += sys->vel_x[i] * dt;
        sys->pos_y[i] += sys->vel_y[i] * dt;
        sys->pos_z[i] += sys->vel_z[i] * dt;
    }
#else
    for (uint32_t i = 0; i < n; i++) {
        sys->pos_x[i] += sys->vel_x[i] * dt;
        sys->pos_y[i] += sys->vel_y[i] * dt;
        sys->pos_z[i] += sys->vel_z[i] * dt;
    }
#endif
}

static void apply_gravity(ParticleSystem *sys, float dt, float g) {
    const uint32_t n = sys->count;
    
#ifdef __AVX__
    const uint32_t n_simd = n - (n % 8);
    const __m256 g_dt = _mm256_set1_ps(g * dt);
    
    for (uint32_t i = 0; i < n_simd; i += 8) {
        __m256 vz = _mm256_loadu_ps(&sys->vel_z[i]);
        vz = _mm256_add_ps(vz, g_dt);
        _mm256_storeu_ps(&sys->vel_z[i], vz);
    }
    
    for (uint32_t i = n_simd; i < n; i++) {
        sys->vel_z[i] += g * dt;
    }
#else
    for (uint32_t i = 0; i < n; i++) {
        sys->vel_z[i] += g * dt;
    }
#endif
}

static void simulation_step(ParticleSystem *sys, float dt) {
    apply_gravity(sys, dt, -9.81f);
    integrate_euler(sys, dt);
}

// ============================================================================
// Tests
// ============================================================================

void test_basic_simulation() {
    printf("Test: Basic Simulation\n");
    printf("----------------------\n");
    
    ParticleSystem *sys = create_system(10);
    
    // Add a particle at height 10
    add_particle(sys, 0, 0, 10, 0, 0, 0, 1.0);
    
    printf("Initial: z=%.2f\n", sys->pos_z[0]);
    
    // Simulate for 1 second
    for (int i = 0; i < 100; i++) {
        simulation_step(sys, 0.01f);
    }
    
    printf("After 1s: z=%.2f, vz=%.2f\n", sys->pos_z[0], sys->vel_z[0]);
    printf("Expected: z≈5.1, vz≈-9.81\n\n");
    
    destroy_system(sys);
}

void test_performance() {
    printf("Test: Performance Benchmark\n");
    printf("---------------------------\n");
    
    uint32_t counts[] = {100, 1000, 10000};
    
    for (int c = 0; c < 3; c++) {
        uint32_t count = counts[c];
        ParticleSystem *sys = create_system(count);
        
        // Add random particles
        for (uint32_t i = 0; i < count; i++) {
            float x = (float)rand() / RAND_MAX * 100;
            float y = (float)rand() / RAND_MAX * 100;
            float z = (float)rand() / RAND_MAX * 100;
            add_particle(sys, x, y, z, 0, 0, 0, 1.0);
        }
        
        // Benchmark
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (int i = 0; i < 1000; i++) {
            simulation_step(sys, 0.01f);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
        
        double particles_per_sec = count * 1000 / elapsed;
        
        printf("%u particles × 1000 steps:\n", count);
        printf("  Time: %.2fms\n", elapsed * 1000);
        printf("  Rate: %.2f M particles/sec\n", particles_per_sec / 1e6);
        printf("  Time per step: %.2fμs\n\n", elapsed / 1000 * 1e6);
        
        destroy_system(sys);
    }
}

void test_simd() {
    printf("Test: SIMD Capabilities\n");
    printf("----------------------\n");
    
#ifdef __AVX__
    printf("✓ AVX support detected\n");
    printf("  SIMD width: 8 floats (256-bit)\n");
#elif defined(__SSE__)
    printf("✓ SSE support detected\n");
    printf("  SIMD width: 4 floats (128-bit)\n");
#else
    printf("⚠ No SIMD support\n");
    printf("  Using scalar operations\n");
#endif
    
    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("\n");
    printf("================================================================================\n");
    printf("BrokenRecord Zero - Standalone Test\n");
    printf("================================================================================\n\n");
    
    test_simd();
    test_basic_simulation();
    test_performance();
    
    printf("================================================================================\n");
    printf("All tests complete!\n");
    printf("================================================================================\n\n");
    
    return 0;
}