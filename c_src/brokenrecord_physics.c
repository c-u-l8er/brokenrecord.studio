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
    printf("DEBUG C: create_particle_system called with argc=%d\n", argc);
    
    // argv[0] should be a map with packed binary data
    if (argc < 1) {
        printf("DEBUG C: ERROR - argc < 1\n");
        return enif_make_badarg(env);
    }
    
    // Check if argv[0] is a map
    if (!enif_is_map(env, argv[0])) {
        printf("DEBUG C: ERROR - argv[0] is not a map\n");
        return enif_make_badarg(env);
    }
    
    printf("DEBUG C: argv[0] is a map, checking format\n");
    
    // Check for both packed format (count field) and original format (particles field)
    ERL_NIF_TERM count_term, particles_term;
    int has_count = enif_get_map_value(env, argv[0], enif_make_atom(env, "count"), &count_term);
    int has_particles = enif_get_map_value(env, argv[0], enif_make_atom(env, "particles"), &particles_term);
    
    printf("DEBUG C: has_count=%d, has_particles=%d\n", has_count, has_particles);
    
    unsigned int count;
    
    if (has_count) {
        // Packed format: %{count: N, pos_x: <<...>>, pos_y: <<...>>, ...}
        printf("DEBUG C: detected packed format\n");
        
        if (!enif_get_uint(env, count_term, &count)) {
            printf("DEBUG C: ERROR - failed to get count\n");
            return enif_make_badarg(env);
        }
        
        printf("DEBUG C: particle count: %u\n", count);
    
        // Allow empty systems
        if (count == 0) {
            printf("DEBUG C: creating empty system\n");
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
            printf("DEBUG C: allocation failed\n");
            enif_release_resource(sys);
            return enif_make_atom(env, "allocation_error");
        }
        
        // Extract binary data from packed format
        ERL_NIF_TERM pos_x_bin, pos_y_bin, pos_z_bin;
        ERL_NIF_TERM vel_x_bin, vel_y_bin, vel_z_bin;
        ERL_NIF_TERM mass_bin, ids_bin;
        
        if (!enif_get_map_value(env, argv[0], enif_make_atom(env, "pos_x"), &pos_x_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "pos_y"), &pos_y_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "pos_z"), &pos_z_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "vel_x"), &vel_x_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "vel_y"), &vel_y_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "vel_z"), &vel_z_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "mass"), &mass_bin) ||
            !enif_get_map_value(env, argv[0], enif_make_atom(env, "ids"), &ids_bin)) {
            printf("DEBUG C: ERROR - failed to extract binary fields\n");
            enif_release_resource(sys);
            return enif_make_badarg(env);
        }
        
        printf("DEBUG C: extracted binary fields\n");
        
        // Get binary data
        ErlNifBinary pos_x_binary, pos_y_binary, pos_z_binary;
        ErlNifBinary vel_x_binary, vel_y_binary, vel_z_binary;
        ErlNifBinary mass_binary, ids_binary;
        
        if (!enif_inspect_binary(env, pos_x_bin, &pos_x_binary) ||
            !enif_inspect_binary(env, pos_y_bin, &pos_y_binary) ||
            !enif_inspect_binary(env, pos_z_bin, &pos_z_binary) ||
            !enif_inspect_binary(env, vel_x_bin, &vel_x_binary) ||
            !enif_inspect_binary(env, vel_y_bin, &vel_y_binary) ||
            !enif_inspect_binary(env, vel_z_bin, &vel_z_binary) ||
            !enif_inspect_binary(env, mass_bin, &mass_binary) ||
            !enif_inspect_binary(env, ids_bin, &ids_binary)) {
            printf("DEBUG C: ERROR - failed to get binary data\n");
            enif_release_resource(sys);
            return enif_make_badarg(env);
        }
        
        printf("DEBUG C: got binary data, copying to arrays\n");
        
        // Copy binary data to arrays
        memcpy(sys->pos_x, pos_x_binary.data, pos_x_binary.size);
        memcpy(sys->pos_y, pos_y_binary.data, pos_y_binary.size);
        memcpy(sys->pos_z, pos_z_binary.data, pos_z_binary.size);
        memcpy(sys->vel_x, vel_x_binary.data, vel_x_binary.size);
        memcpy(sys->vel_y, vel_y_binary.data, vel_y_binary.size);
        memcpy(sys->vel_z, vel_z_binary.data, vel_z_binary.size);
        memcpy(sys->mass, mass_binary.data, mass_binary.size);
        
        // Extract IDs from binary format
        printf("DEBUG C: extracting IDs from binary\n");
        size_t ids_offset = 0;
        for (uint32_t i = 0; i < count; i++) {
            if (ids_offset + 4 > ids_binary.size) {
                printf("DEBUG C: ERROR - IDs binary too small\n");
                break;
            }
            
            // Read length prefix (4 bytes)
            uint32_t str_len;
            memcpy(&str_len, ids_binary.data + ids_offset, 4);
            ids_offset += 4;
            
            if (ids_offset + str_len > ids_binary.size) {
                printf("DEBUG C: ERROR - ID string extends beyond binary\n");
                break;
            }
            
            // Allocate and copy string
            sys->ids[i] = malloc(str_len + 1);
            memcpy(sys->ids[i], ids_binary.data + ids_offset, str_len);
            sys->ids[i][str_len] = '\0';
            ids_offset += str_len;
            
            printf("DEBUG C: extracted ID[%u] = '%s'\n", i, sys->ids[i]);
        }
        
        printf("DEBUG C: successfully created particle system with %u particles\n", count);
        
        ERL_NIF_TERM term = enif_make_resource(env, sys);
        enif_release_resource(sys);
        return term;
        
    } else if (has_particles) {
        printf("DEBUG C: detected original format with particles field\n");
        
        // Get the particles list
        ERL_NIF_TERM particles_list;
        if (!enif_get_map_value(env, argv[0], enif_make_atom(env, "particles"), &particles_list)) {
            printf("DEBUG C: ERROR - failed to get particles list\n");
            return enif_make_badarg(env);
        }
        
        // Get length of particles list
        unsigned int particle_count;
        if (!enif_get_list_length(env, particles_list, &particle_count)) {
            printf("DEBUG C: ERROR - failed to get particles list length\n");
            return enif_make_badarg(env);
        }
        
        printf("DEBUG C: particle count: %u\n", particle_count);
        
        // Allow empty systems
        if (particle_count == 0) {
            printf("DEBUG C: creating empty system from particles format\n");
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
        sys->count = particle_count;
        sys->capacity = particle_count;
        
        size_t size = particle_count * sizeof(float);
        sys->pos_x = aligned_malloc(size, 32);
        sys->pos_y = aligned_malloc(size, 32);
        sys->pos_z = aligned_malloc(size, 32);
        sys->vel_x = aligned_malloc(size, 32);
        sys->vel_y = aligned_malloc(size, 32);
        sys->vel_z = aligned_malloc(size, 32);
        sys->mass = aligned_malloc(size, 32);
        sys->ids = malloc(particle_count * sizeof(char*));
        
        if (!sys->pos_x || !sys->pos_y || !sys->pos_z ||
            !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass || !sys->ids) {
            printf("DEBUG C: allocation failed for particles format\n");
            enif_release_resource(sys);
            return enif_make_atom(env, "allocation_error");
        }
        
        // Extract particle data from list
        ERL_NIF_TERM head, tail;
        ERL_NIF_TERM current_list = particles_list;
        
        for (unsigned int i = 0; i < particle_count; i++) {
            if (!enif_get_list_cell(env, current_list, &head, &tail)) {
                printf("DEBUG C: ERROR - failed to get particle %u from list\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract particle fields
            ERL_NIF_TERM pos_term, vel_term, mass_term, id_term;
            if (!enif_get_map_value(env, head, enif_make_atom(env, "position"), &pos_term) ||
                !enif_get_map_value(env, head, enif_make_atom(env, "velocity"), &vel_term) ||
                !enif_get_map_value(env, head, enif_make_atom(env, "mass"), &mass_term) ||
                !enif_get_map_value(env, head, enif_make_atom(env, "id"), &id_term)) {
                printf("DEBUG C: ERROR - failed to extract particle %u fields\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract position tuple
            const ERL_NIF_TERM* pos_arr;
            int pos_arity;
            if (!enif_get_tuple(env, pos_term, &pos_arity, &pos_arr) || pos_arity != 3) {
                printf("DEBUG C: ERROR - invalid position tuple for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            double pos_x, pos_y, pos_z;
            if (!enif_get_double(env, pos_arr[0], &pos_x) ||
                !enif_get_double(env, pos_arr[1], &pos_y) ||
                !enif_get_double(env, pos_arr[2], &pos_z)) {
                printf("DEBUG C: ERROR - failed to get position values for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract velocity tuple
            const ERL_NIF_TERM* vel_arr;
            int vel_arity;
            if (!enif_get_tuple(env, vel_term, &vel_arity, &vel_arr) || vel_arity != 3) {
                printf("DEBUG C: ERROR - invalid velocity tuple for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            double vel_x, vel_y, vel_z;
            if (!enif_get_double(env, vel_arr[0], &vel_x) ||
                !enif_get_double(env, vel_arr[1], &vel_y) ||
                !enif_get_double(env, vel_arr[2], &vel_z)) {
                printf("DEBUG C: ERROR - failed to get velocity values for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract mass
            double mass;
            if (!enif_get_double(env, mass_term, &mass)) {
                printf("DEBUG C: ERROR - failed to get mass for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract ID
            char id_str[256];
            if (!enif_get_string(env, id_term, id_str, sizeof(id_str), ERL_NIF_UTF8)) {
                printf("DEBUG C: ERROR - failed to get ID for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Store in arrays
            sys->pos_x[i] = (float)pos_x;
            sys->pos_y[i] = (float)pos_y;
            sys->pos_z[i] = (float)pos_z;
            sys->vel_x[i] = (float)vel_x;
            sys->vel_y[i] = (float)vel_y;
            sys->vel_z[i] = (float)vel_z;
            sys->mass[i] = (float)mass;
            
            // Allocate and copy ID
            sys->ids[i] = malloc(strlen(id_str) + 1);
            strcpy(sys->ids[i], id_str);
            
            printf("DEBUG C: extracted particle %u: id='%s', pos=(%f,%f,%f), vel=(%f,%f,%f), mass=%f\n",
                   i, sys->ids[i], sys->pos_x[i], sys->pos_y[i], sys->pos_z[i],
                   sys->vel_x[i], sys->vel_y[i], sys->vel_z[i], sys->mass[i]);
            
            current_list = tail;
        }
        
        printf("DEBUG C: successfully created particle system with %u particles from list format\n", particle_count);
        
        ERL_NIF_TERM term = enif_make_resource(env, sys);
        enif_release_resource(sys);
        return term;
        
    } else {
        printf("DEBUG C: ERROR - unsupported format, expected 'count' or 'particles' field\n");
        return enif_make_badarg(env);
    }
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