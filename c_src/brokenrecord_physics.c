#include <erl_nif.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>  // AVX2/AVX-512
#include <omp.h>
#include <math.h>

// ============================================================================
// Data Structures (Struct-of-Arrays for SIMD)
// ============================================================================

typedef struct {
    float* pos_x;
    float* pos_y;
    float* pos_z;
    float* vel_x;
    float* vel_y;
    float* vel_z;
    float* mass;
    char** ids;  // Store particle IDs
    uint32_t count;
    uint32_t capacity;
} ParticleSystem;

static ErlNifResourceType* particle_system_type = NULL;

// ============================================================================
// Resource Management
// ============================================================================

static void particle_system_destructor(ErlNifEnv* env __attribute__((unused)), void* obj) {
    ParticleSystem* sys = (ParticleSystem*)obj;
    if (sys->pos_x) { free(sys->pos_x); }
    if (sys->pos_y) { free(sys->pos_y); }
    if (sys->pos_z) { free(sys->pos_z); }
    if (sys->vel_x) { free(sys->vel_x); }
    if (sys->vel_y) { free(sys->vel_y); }
    if (sys->vel_z) { free(sys->vel_z); }
    if (sys->mass) { free(sys->mass); }
    if (sys->ids) {
        for (uint32_t i = 0; i < sys->count; i++) {
            if (sys->ids[i]) { free(sys->ids[i]); }
        }
        free(sys->ids);
    }
}

static int load(ErlNifEnv* env, void** priv_data __attribute__((unused)), ERL_NIF_TERM load_info __attribute__((unused))) {
    particle_system_type = enif_open_resource_type(
        env, NULL, "particle_system",
        particle_system_destructor,
        ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER,
        NULL
    );
    return particle_system_type == NULL ? -1 : 0;
}

// ============================================================================
// Helper Functions
// ============================================================================

static void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr;
    return (posix_memalign(&ptr, alignment, size) == 0) ? ptr : NULL;
}

static float get_float_from_map(ErlNifEnv* env, ERL_NIF_TERM map, const char* key) {
    ERL_NIF_TERM value;
    double dval;
    if (enif_get_map_value(env, map, enif_make_atom(env, key), &value) &&
        enif_get_double(env, value, &dval)) {
        return (float)dval;
    }
    return 0.0f;
}

static ERL_NIF_TERM get_tuple_elem(ErlNifEnv* env, ERL_NIF_TERM tuple, int index) {
    const ERL_NIF_TERM* arr;
    int arity;
    if (enif_get_tuple(env, tuple, &arity, &arr) && index < arity) {
        return arr[index];
    }
    return enif_make_badarg(env);
}

// ============================================================================
// Physics Kernels (AVX2 SIMD)
// ============================================================================

static void apply_gravity_simd(ParticleSystem* sys, float dt) {
    // DISABLE SIMD - use simple scalar code for debugging
    for (uint32_t i = 0; i < sys->count; i++) {
        sys->vel_y[i] += -0.981f * dt;
    }
}

static void integrate_positions_simd(ParticleSystem* sys, float dt) {
    // DISABLE SIMD - use simple scalar code for debugging
    for (uint32_t i = 0; i < sys->count; i++) {
        sys->pos_x[i] += sys->vel_x[i] * dt;
        sys->pos_y[i] += sys->vel_y[i] * dt;
        sys->pos_z[i] += sys->vel_z[i] * dt;
    }
}

// ============================================================================
// NIF Functions
// ============================================================================

// Create particle system from Elixir state
static ERL_NIF_TERM create_particle_system(ErlNifEnv* env, int argc __attribute__((unused)), const ERL_NIF_TERM argv[]) {
    // argv[0] = %{particles: [...], walls: [...]}
    ERL_NIF_TERM particles_list;
    unsigned int count;
    
    if (!enif_get_map_value(env, argv[0], enif_make_atom(env, "particles"), &particles_list)) {
        return enif_make_badarg(env);
    }
    
    if (!enif_get_list_length(env, particles_list, &count)) {
        return enif_make_badarg(env);
    }
    
    // Allow empty systems
    if (count == 0) {
        // Create empty system
        ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
        sys->count = 0;
        sys->capacity = 0;
        sys->pos_x = NULL;
        sys->pos_y = NULL;
        sys->pos_z = NULL;
        sys->vel_x = NULL;
        sys->vel_y = NULL;
        sys->vel_z = NULL;
        sys->mass = NULL;
        sys->ids = NULL;
        
        ERL_NIF_TERM term = enif_make_resource(env, sys);
        enif_release_resource(sys);
        return term;
    }
    
    // Allocate system
    ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
    sys->count = count;
    sys->capacity = count;
    
    size_t size = count * sizeof(float);
    sys->pos_x = aligned_malloc(size, 32);
    sys->pos_y = aligned_malloc(size, 32);
    sys->pos_z = aligned_malloc(size, 32);
    sys->vel_x = aligned_malloc(size, 32);
    sys->vel_y = aligned_malloc(size, 32);
    sys->vel_z = aligned_malloc(size, 32);
    sys->mass = aligned_malloc(size, 32);
    sys->ids = malloc(count * sizeof(char*));
    
    if (!sys->pos_x || !sys->pos_y || !sys->pos_z ||
        !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass || !sys->ids) {
        enif_release_resource(sys);
        return enif_make_atom(env, "allocation_error");
    }
    
    // Parse particles
    ERL_NIF_TERM head, tail = particles_list;
    uint32_t i = 0;
    
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        // Get position tuple
        ERL_NIF_TERM pos;
        if (!enif_get_map_value(env, head, enif_make_atom(env, "position"), &pos)) {
            enif_release_resource(sys);
            return enif_make_badarg(env);
        }
        
        double px, py, pz;
        enif_get_double(env, get_tuple_elem(env, pos, 0), &px);
        enif_get_double(env, get_tuple_elem(env, pos, 1), &py);
        enif_get_double(env, get_tuple_elem(env, pos, 2), &pz);
        
        // Get velocity tuple
        ERL_NIF_TERM vel;
        if (!enif_get_map_value(env, head, enif_make_atom(env, "velocity"), &vel)) {
            enif_release_resource(sys);
            return enif_make_badarg(env);
        }
        
        double vx, vy, vz;
        enif_get_double(env, get_tuple_elem(env, vel, 0), &vx);
        enif_get_double(env, get_tuple_elem(env, vel, 1), &vy);
        enif_get_double(env, get_tuple_elem(env, vel, 2), &vz);
        
        // Get mass
        double m = get_float_from_map(env, head, "mass");
        
        // Get ID
        ERL_NIF_TERM id_term;
        char id_str[256];
        if (enif_get_map_value(env, head, enif_make_atom(env, "id"), &id_term)) {
            if (enif_get_string(env, id_term, id_str, sizeof(id_str), ERL_NIF_LATIN1) > 0) {
                sys->ids[i] = malloc(strlen(id_str) + 1);
                strcpy(sys->ids[i], id_str);
            } else {
                sys->ids[i] = malloc(8);  // "default"
                strcpy(sys->ids[i], "default");
            }
        } else {
            sys->ids[i] = malloc(8);  // "default"
            strcpy(sys->ids[i], "default");
        }
        
        // Store
        sys->pos_x[i] = (float)px;
        sys->pos_y[i] = (float)py;
        sys->pos_z[i] = (float)pz;
        sys->vel_x[i] = (float)vx;
        sys->vel_y[i] = (float)vy;
        sys->vel_z[i] = (float)vz;
        sys->mass[i] = (float)m;
        i++;
    }
    
    ERL_NIF_TERM term = enif_make_resource(env, sys);
    enif_release_resource(sys);
    return term;
}

// Simulate N steps
static ERL_NIF_TERM native_integrate(ErlNifEnv* env, int argc __attribute__((unused)), const ERL_NIF_TERM argv[]) {
    ParticleSystem* sys;
    double dt;
    int steps;
    int apply_gravity = 1; // Default: apply gravity
    
    if (!enif_get_resource(env, argv[0], particle_system_type, (void**)&sys)) {
        return enif_make_badarg(env);
    }
    
    if (!enif_get_double(env, argv[1], &dt)) {
        return enif_make_badarg(env);
    }
    
    if (!enif_get_int(env, argv[2], &steps)) {
        return enif_make_badarg(env);
    }
    
    // Check if we have a 4th argument (rules) - if present, assume no gravity
    if (argc >= 4) {
        apply_gravity = 0; // Don't apply gravity when rules are provided
    }
    
    // RUN THE ACTUAL PHYSICS!
    for (int step = 0; step < steps; step++) {
        // Debug: Print first particle's state for first few steps
        if (step < 3 && sys->count > 0) {
            printf("DEBUG C: Step %d - apply_gravity: %d, pos_y: %f, vel_y: %f\n",
                   step, apply_gravity, sys->pos_y[0], sys->vel_y[0]);
        }
        
        // DISABLE COLLISIONS FOR NOW - just test integration
        // Check for particle-particle collisions
        /*
        for (uint32_t i = 0; i < sys->count; i++) {
            for (uint32_t j = i + 1; j < sys->count; j++) {
                double dx = sys->pos_x[i] - sys->pos_x[j];
                double dy = sys->pos_y[i] - sys->pos_y[j];
                double dz = sys->pos_z[i] - sys->pos_z[j];
                double dist_sq = dx*dx + dy*dy + dz*dz;
                double min_dist = 0.8f; // Default radius for collision
                
                if (dist_sq < min_dist * min_dist) {
                    // Collision detected - elastic collision
                    double dist = sqrt(dist_sq);
                    if (dist > 0.0) {  // Avoid division by zero
                        double nx = dx / dist;
                        double ny = dy / dist;
                        double nz = dz / dist;
                        
                        double dvx = sys->vel_x[i] - sys->vel_x[j];
                        double dvy = sys->vel_y[i] - sys->vel_y[j];
                        double dvz = sys->vel_z[i] - sys->vel_z[j];
                        
                        double dot_product = dvx * nx + dvy * ny + dvz * nz;
                        
                        if (dot_product < 0) {
                            double impulse = 2.0 * dot_product / (sys->mass[i] + sys->mass[j]);
                            
                            sys->vel_x[i] -= impulse * nx / sys->mass[i];
                            sys->vel_y[i] -= impulse * ny / sys->mass[i];
                            sys->vel_z[i] -= impulse * nz / sys->mass[i];
                            
                            sys->vel_x[j] += impulse * nx / sys->mass[j];
                            sys->vel_y[j] += impulse * ny / sys->mass[j];
                            sys->vel_z[j] += impulse * nz / sys->mass[j];
                        }
                    }
                }
            }
        }
        */
        
        if (apply_gravity) {
            apply_gravity_simd(sys, (float)dt);
        }
        integrate_positions_simd(sys, (float)dt);
    }
    
    // Return the resource (state is mutated in place)
    return argv[0];
}

// Convert back to Elixir format
static ERL_NIF_TERM to_elixir_state(ErlNifEnv* env, int argc __attribute__((unused)), const ERL_NIF_TERM argv[]) {
    ParticleSystem* sys;
    
    if (!enif_get_resource(env, argv[0], particle_system_type, (void**)&sys)) {
        return enif_make_badarg(env);
    }
    
    ERL_NIF_TERM* particles = enif_alloc(sys->count * sizeof(ERL_NIF_TERM));
    
    for (uint32_t i = 0; i < sys->count; i++) {
        ERL_NIF_TERM pos = enif_make_tuple3(env,
            enif_make_double(env, sys->pos_x[i]),
            enif_make_double(env, sys->pos_y[i]),
            enif_make_double(env, sys->pos_z[i])
        );
        
        ERL_NIF_TERM vel = enif_make_tuple3(env,
            enif_make_double(env, sys->vel_x[i]),
            enif_make_double(env, sys->vel_y[i]),
            enif_make_double(env, sys->vel_z[i])
        );
        
        // Get radius from original particle if available
        ERL_NIF_TERM radius_val = enif_make_double(env, 1.0f); // default radius
        // Note: We can't access the original particle list here since we only have the C structure
        // This would need to be passed in separately if needed
        
        ERL_NIF_TERM keys[] = {
            enif_make_atom(env, "position"),
            enif_make_atom(env, "velocity"),
            enif_make_atom(env, "mass"),
            enif_make_atom(env, "radius"),
            enif_make_atom(env, "id")
        };
        
        ERL_NIF_TERM values[] = {
            pos,
            vel,
            enif_make_double(env, sys->mass[i]),
            radius_val,
            enif_make_string_len(env, sys->ids[i] ? sys->ids[i] : "default", sys->ids[i] ? strlen(sys->ids[i]) : 7, ERL_NIF_UTF8)
        };
        
        enif_make_map_from_arrays(env, keys, values, 5, &particles[i]);
    }
    
    ERL_NIF_TERM particles_list = enif_make_list_from_array(env, particles, sys->count);
    enif_free(particles);
    
    // Preserve walls from original state if present
    ERL_NIF_TERM walls_result = enif_make_list(env, 0); // empty list by default
    if (enif_get_map_value(env, argv[0], enif_make_atom(env, "walls"), &walls_result)) {
        // walls_result already contains the walls list
    }
    
    ERL_NIF_TERM keys[] = {enif_make_atom(env, "particles"), enif_make_atom(env, "walls")};
    ERL_NIF_TERM values[] = {particles_list, walls_result};
    ERL_NIF_TERM result;
    enif_make_map_from_arrays(env, keys, values, 2, &result);
    
    return result;
}

// ============================================================================
// Module Registration
// ============================================================================

static ErlNifFunc nif_funcs[] = {
    {"create_particle_system", 1, create_particle_system, 0},
    {"native_integrate", 4, native_integrate, 0},  // Support 4 arguments (system, dt, steps, rules)
    {"to_elixir_state", 1, to_elixir_state, 0}
};

ERL_NIF_INIT(Elixir.BrokenRecord.Zero.NIF, nif_funcs, load, NULL, NULL, NULL)