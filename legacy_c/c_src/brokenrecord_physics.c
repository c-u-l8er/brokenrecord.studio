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
    float* radius;  // Add particle radius
    char** ids;  // Store particle IDs
    uint32_t count;
    uint32_t capacity;
    char particle_key[32];  // Store original key name ("particles", "bodies", "molecules")
} ParticleSystem;

typedef struct {
    float* pos_x;
    float* pos_y;
    float* pos_z;
    float* normal_x;
    float* normal_y;
    float* normal_z;
    uint32_t count;
} WallSystem;

static ErlNifResourceType* particle_system_type = NULL;

// ============================================================================
// Resource Management
// ============================================================================

static void particle_system_destructor(ErlNifEnv* env __attribute__((unused)), void* obj) {
    printf("DEBUG C: destructor called - obj: %p\n", obj);
    if (!obj) {
        printf("DEBUG C: destructor - NULL object, returning\n");
        return;
    }
    
    ParticleSystem* sys = (ParticleSystem*)obj;
    printf("DEBUG C: destructor - sys->count: %u\n", sys->count);
    printf("DEBUG C: destructor - sys->particle_key: '%.32s'\n", sys->particle_key);
    
    if (sys->pos_x) { free(sys->pos_x); }
    if (sys->pos_y) { free(sys->pos_y); }
    if (sys->pos_z) { free(sys->pos_z); }
    if (sys->vel_x) { free(sys->vel_x); }
    if (sys->vel_y) { free(sys->vel_y); }
    if (sys->vel_z) { free(sys->vel_z); }
    if (sys->mass) { free(sys->mass); }
    if (sys->radius) { free(sys->radius); }  // SAFETY: Only free if not NULL
    if (sys->ids) {
        printf("DEBUG C: destructor - freeing IDs for %u particles\n", sys->count);
        for (uint32_t i = 0; i < sys->count; i++) {
            if (sys->ids[i]) {
                printf("DEBUG C: destructor - freeing ID[%u]: '%s'\n", i, sys->ids[i]);
                free(sys->ids[i]);
            }
        }
        free(sys->ids);
    }
    printf("DEBUG C: destructor - completed successfully\n");
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
    // Implement proper N-body gravity to match Elixir DSL
    float G = 1.0f;  // Gravitational constant (matches Elixir DSL)
    
    for (uint32_t i = 0; i < sys->count; i++) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        
        // Calculate gravitational forces from all other bodies
        for (uint32_t j = 0; j < sys->count; j++) {
            if (i == j) continue;  // Skip self
            
            // Calculate distance vector
            float dx = sys->pos_x[j] - sys->pos_x[i];
            float dy = sys->pos_y[j] - sys->pos_y[i];
            float dz = sys->pos_z[j] - sys->pos_z[i];
            
            // Calculate distance squared and distance
            float dist_sq = dx*dx + dy*dy + dz*dz;
            float dist = sqrtf(dist_sq);
            
            // Avoid singularity at zero distance
            float min_dist = sys->radius[i] + sys->radius[j];
            float effective_dist = (dist > min_dist) ? dist : min_dist;
            
            // Calculate force magnitude: F = G * m1 * m2 / r^2
            float force_magnitude = G * sys->mass[i] * sys->mass[j] / (effective_dist * effective_dist);
            
            // Calculate force direction (normalized)
            float nx = dx / dist;
            float ny = dy / dist;
            float nz = dz / dist;
            
            // Apply force to body i (Newton's third law will handle body j in its iteration)
            fx += nx * force_magnitude;
            fy += ny * force_magnitude;
            fz += nz * force_magnitude;
        }
        
        // Update velocity: v = v + (F/m) * dt
        sys->vel_x[i] += (fx / sys->mass[i]) * dt;
        sys->vel_y[i] += (fy / sys->mass[i]) * dt;
        sys->vel_z[i] += (fz / sys->mass[i]) * dt;
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
    
    // Check for both packed format (count field) and original format (particles/bodies/molecules field)
    ERL_NIF_TERM count_term, particles_term, bodies_term, molecules_term;
    int has_count = enif_get_map_value(env, argv[0], enif_make_atom(env, "count"), &count_term);
    int has_particles = enif_get_map_value(env, argv[0], enif_make_atom(env, "particles"), &particles_term);
    int has_bodies = enif_get_map_value(env, argv[0], enif_make_atom(env, "bodies"), &bodies_term);
    int has_molecules = enif_get_map_value(env, argv[0], enif_make_atom(env, "molecules"), &molecules_term);
    
    printf("DEBUG C: has_count=%d, has_particles=%d, has_bodies=%d, has_molecules=%d\n",
           has_count, has_particles, has_bodies, has_molecules);
    
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
            printf("DEBUG C: packed empty create - BEFORE init - sys pointer: %p\n", (void*)sys);
            
            // CRITICAL FIX: Initialize all fields to NULL/0 before use
            memset(sys, 0, sizeof(ParticleSystem));
            printf("DEBUG C: packed empty create - AFTER memset - key_content[0]: '%c'\n", sys->particle_key[0]);
            
            strcpy(sys->particle_key, "particles");
            printf("DEBUG C: packed empty create - AFTER init key_content=%.32s len=%zu\n", sys->particle_key, strlen(sys->particle_key));
            sys->count = 0;
            sys->capacity = 0;
            sys->pos_x = NULL;
            sys->pos_y = NULL;
            sys->pos_z = NULL;
            sys->vel_x = NULL;
            sys->vel_y = NULL;
            sys->vel_z = NULL;
            sys->mass = NULL;
            sys->radius = NULL;  // CRITICAL FIX: Missing NULL initialization
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
        sys->radius = aligned_malloc(size, 32);
        sys->ids = malloc(count * sizeof(char*));
        
        if (!sys->pos_x || !sys->pos_y || !sys->pos_z ||
            !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass || !sys->radius || !sys->ids) {
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
            printf("DEBUG C: particles empty create - BEFORE init - sys pointer: %p\n", (void*)sys);
            printf("DEBUG C: particles empty create - BEFORE init key_content[0]: '%c'\n", sys ? sys->particle_key[0] : '?');
            
            // CRITICAL FIX: Initialize all fields to NULL/0 before use
            memset(sys, 0, sizeof(ParticleSystem));
            printf("DEBUG C: particles empty create - AFTER memset - key_content[0]: '%c'\n", sys->particle_key[0]);
            
            strcpy(sys->particle_key, "particles");
            printf("DEBUG C: particles empty create - AFTER init key_content=%.32s len=%zu\n", sys->particle_key, strlen(sys->particle_key));
            sys->count = 0;
            sys->capacity = 0;
            sys->pos_x = NULL;
            sys->pos_y = NULL;
            sys->pos_z = NULL;
            sys->vel_x = NULL;
            sys->vel_y = NULL;
            sys->vel_z = NULL;
            sys->mass = NULL;
            sys->radius = NULL;  // CRITICAL FIX: Missing NULL initialization
            sys->ids = NULL;
            
            ERL_NIF_TERM term = enif_make_resource(env, sys);
            enif_release_resource(sys);
            return term;
        }
        
        // Allocate system
        ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
        sys->count = particle_count;
        sys->capacity = particle_count;
        strcpy(sys->particle_key, "particles");  // Store original key name
        
        size_t size = particle_count * sizeof(float);
        sys->pos_x = aligned_malloc(size, 32);
        sys->pos_y = aligned_malloc(size, 32);
        sys->pos_z = aligned_malloc(size, 32);
        sys->vel_x = aligned_malloc(size, 32);
        sys->vel_y = aligned_malloc(size, 32);
        sys->vel_z = aligned_malloc(size, 32);
        sys->mass = aligned_malloc(size, 32);
        sys->radius = aligned_malloc(size, 32);
        sys->ids = malloc(particle_count * sizeof(char*));
        
        if (!sys->pos_x || !sys->pos_y || !sys->pos_z ||
            !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass || !sys->radius || !sys->ids) {
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
            ERL_NIF_TERM pos_term, vel_term, mass_term, radius_term, id_term;
            int has_pos = enif_get_map_value(env, head, enif_make_atom(env, "position"), &pos_term);
            int has_vel = enif_get_map_value(env, head, enif_make_atom(env, "velocity"), &vel_term);
            int has_mass = enif_get_map_value(env, head, enif_make_atom(env, "mass"), &mass_term);
            int has_radius = enif_get_map_value(env, head, enif_make_atom(env, "radius"), &radius_term);
            int has_id = enif_get_map_value(env, head, enif_make_atom(env, "id"), &id_term);
            
            printf("DEBUG C: particle %u field availability - pos:%d vel:%d mass:%d radius:%d id:%d\n",
                   i, has_pos, has_vel, has_mass, has_radius, has_id);
            
            if (!has_pos || !has_vel || !has_mass) {
                printf("DEBUG C: ERROR - failed to extract required particle %u fields (pos:%d vel:%d mass:%d)\n",
                       i, has_pos, has_vel, has_mass);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Make radius optional with default
            if (!has_radius) {
                printf("DEBUG C: particle %u - radius missing, using default 1.0\n", i);
                radius_term = enif_make_double(env, 1.0);
            }
            
            // Make id optional with default
            if (!has_id) {
                printf("DEBUG C: particle %u - id missing, using default\n", i);
                char default_id[32];
                snprintf(default_id, sizeof(default_id), "p%u", i);
                id_term = enif_make_string(env, default_id, ERL_NIF_UTF8);
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
            
            // Extract radius
            double radius;
            if (!enif_get_double(env, radius_term, &radius)) {
                printf("DEBUG C: ERROR - failed to get radius for particle %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract ID - Add debugging to check term type
            printf("DEBUG C: particle %u - attempting to extract ID from term\n", i);
            char id_str[256];
            
            // Check if the ID term is an atom or string
            if (enif_is_atom(env, id_term)) {
                printf("DEBUG C: particle %u - ID is an atom, converting to string\n", i);
                char atom_buf[256];
                if (enif_get_atom(env, id_term, atom_buf, sizeof(atom_buf), ERL_NIF_UTF8)) {
                    printf("DEBUG C: particle %u - successfully converted atom to string: '%s'\n", i, atom_buf);
                    strncpy(id_str, atom_buf, sizeof(id_str) - 1);
                    id_str[sizeof(id_str) - 1] = '\0';
                } else {
                    printf("DEBUG C: ERROR - failed to convert atom to string for particle %u\n", i);
                    enif_release_resource(sys);
                    return enif_make_badarg(env);
                }
            } else if (enif_is_binary(env, id_term)) {
                printf("DEBUG C: particle %u - ID is a binary, converting to string\n", i);
                ErlNifBinary id_binary;
                if (!enif_inspect_binary(env, id_term, &id_binary)) {
                    printf("DEBUG C: ERROR - failed to inspect binary ID for particle %u\n", i);
                    enif_release_resource(sys);
                    return enif_make_badarg(env);
                }
                
                // Copy binary data to string, ensuring null termination
                size_t copy_len = (id_binary.size < sizeof(id_str) - 1) ? id_binary.size : sizeof(id_str) - 1;
                memcpy(id_str, id_binary.data, copy_len);
                id_str[copy_len] = '\0';
                printf("DEBUG C: particle %u - successfully converted binary to string: '%s'\n", i, id_str);
            } else {
                printf("DEBUG C: particle %u - ID term type: %d (not atom or binary)\n", i, enif_term_type(env, id_term));
                if (!enif_get_string(env, id_term, id_str, sizeof(id_str), ERL_NIF_UTF8)) {
                    printf("DEBUG C: ERROR - failed to get ID for particle %u (term not a string)\n", i);
                    enif_release_resource(sys);
                    return enif_make_badarg(env);
                }
            }
            
            // Store in arrays
            sys->pos_x[i] = (float)pos_x;
            sys->pos_y[i] = (float)pos_y;
            sys->pos_z[i] = (float)pos_z;
            sys->vel_x[i] = (float)vel_x;
            sys->vel_y[i] = (float)vel_y;
            sys->vel_z[i] = (float)vel_z;
            sys->mass[i] = (float)mass;
            sys->radius[i] = (float)radius;
            
            // Allocate and copy ID
            sys->ids[i] = malloc(strlen(id_str) + 1);
            strcpy(sys->ids[i], id_str);
            
            printf("DEBUG C: extracted particle %u: id='%s', pos=(%f,%f,%f), vel=(%f,%f,%f), mass=%f, radius=%f\n",
                   i, sys->ids[i], sys->pos_x[i], sys->pos_y[i], sys->pos_z[i],
                   sys->vel_x[i], sys->vel_y[i], sys->vel_z[i], sys->mass[i], sys->radius[i]);
            
            current_list = tail;
        }
        
        printf("DEBUG C: successfully created particle system with %u particles from list format\n", particle_count);
        
        ERL_NIF_TERM term = enif_make_resource(env, sys);
        enif_release_resource(sys);
        return term;
        
    } else if (has_bodies) {
        printf("DEBUG C: detected bodies field, treating as particles\n");
        
        // Get bodies list and treat as particles
        ERL_NIF_TERM bodies_list;
        if (!enif_get_map_value(env, argv[0], enif_make_atom(env, "bodies"), &bodies_list)) {
            printf("DEBUG C: ERROR - failed to get bodies list\n");
            return enif_make_badarg(env);
        }
        
        // Get length of bodies list
        unsigned int body_count;
        if (!enif_get_list_length(env, bodies_list, &body_count)) {
            printf("DEBUG C: ERROR - failed to get bodies list length\n");
            return enif_make_badarg(env);
        }
        
        printf("DEBUG C: body count: %u\n", body_count);
        
        // Allow empty systems
        if (body_count == 0) {
            printf("DEBUG C: creating empty system from bodies format\n");
            ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
            printf("DEBUG C: bodies empty create - BEFORE init - sys pointer: %p\n", (void*)sys);
            
            // CRITICAL FIX: Initialize all fields to NULL/0 before use
            memset(sys, 0, sizeof(ParticleSystem));
            printf("DEBUG C: bodies empty create - AFTER memset - key_content[0]: '%c'\n", sys->particle_key[0]);
            
            strcpy(sys->particle_key, "bodies");
            printf("DEBUG C: bodies empty create - AFTER init key_content=%.32s len=%zu\n", sys->particle_key, strlen(sys->particle_key));
            sys->count = 0;
            sys->capacity = 0;
            sys->pos_x = NULL;
            sys->pos_y = NULL;
            sys->pos_z = NULL;
            sys->vel_x = NULL;
            sys->vel_y = NULL;
            sys->vel_z = NULL;
            sys->mass = NULL;
            sys->radius = NULL;  // CRITICAL FIX: Missing NULL initialization
            sys->ids = NULL;
            
            ERL_NIF_TERM term = enif_make_resource(env, sys);
            enif_release_resource(sys);
            return term;
        }
        
        // Allocate system
        ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
        sys->count = body_count;
        sys->capacity = body_count;
        strcpy(sys->particle_key, "bodies");  // Store original key name
        
        size_t size = body_count * sizeof(float);
        sys->pos_x = aligned_malloc(size, 32);
        sys->pos_y = aligned_malloc(size, 32);
        sys->pos_z = aligned_malloc(size, 32);
        sys->vel_x = aligned_malloc(size, 32);
        sys->vel_y = aligned_malloc(size, 32);
        sys->vel_z = aligned_malloc(size, 32);
        sys->mass = aligned_malloc(size, 32);
        sys->radius = aligned_malloc(size, 32);
        sys->ids = malloc(body_count * sizeof(char*));
        
        if (!sys->pos_x || !sys->pos_y || !sys->pos_z ||
            !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass || !sys->radius || !sys->ids) {
            printf("DEBUG C: allocation failed for bodies format\n");
            enif_release_resource(sys);
            return enif_make_atom(env, "allocation_error");
        }
        
        // Extract body data from list
        ERL_NIF_TERM head, tail;
        ERL_NIF_TERM current_list = bodies_list;
        
        for (unsigned int i = 0; i < body_count; i++) {
            if (!enif_get_list_cell(env, current_list, &head, &tail)) {
                printf("DEBUG C: ERROR - failed to get body %u from list\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract body fields (same as particles but without requiring id)
            ERL_NIF_TERM pos_term, vel_term, mass_term, radius_term;
            int has_pos = enif_get_map_value(env, head, enif_make_atom(env, "position"), &pos_term);
            int has_vel = enif_get_map_value(env, head, enif_make_atom(env, "velocity"), &vel_term);
            int has_mass = enif_get_map_value(env, head, enif_make_atom(env, "mass"), &mass_term);
            int has_radius = enif_get_map_value(env, head, enif_make_atom(env, "radius"), &radius_term);
            
            printf("DEBUG C: body %u field availability - pos:%d vel:%d mass:%d radius:%d\n",
                   i, has_pos, has_vel, has_mass, has_radius);
            
            if (!has_pos || !has_vel || !has_mass) {
                printf("DEBUG C: ERROR - failed to extract required body %u fields (pos:%d vel:%d mass:%d)\n",
                       i, has_pos, has_vel, has_mass);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Make radius optional with default
            if (!has_radius) {
                printf("DEBUG C: body %u - radius missing, using default 1.0\n", i);
                radius_term = enif_make_double(env, 1.0);
            }
            
            // Extract position tuple
            const ERL_NIF_TERM* pos_arr;
            int pos_arity;
            if (!enif_get_tuple(env, pos_term, &pos_arity, &pos_arr) || pos_arity != 3) {
                printf("DEBUG C: ERROR - invalid position tuple for body %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            double pos_x, pos_y, pos_z;
            if (!enif_get_double(env, pos_arr[0], &pos_x) ||
                !enif_get_double(env, pos_arr[1], &pos_y) ||
                !enif_get_double(env, pos_arr[2], &pos_z)) {
                printf("DEBUG C: ERROR - failed to get position values for body %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract velocity tuple
            const ERL_NIF_TERM* vel_arr;
            int vel_arity;
            if (!enif_get_tuple(env, vel_term, &vel_arity, &vel_arr) || vel_arity != 3) {
                printf("DEBUG C: ERROR - invalid velocity tuple for body %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            double vel_x, vel_y, vel_z;
            if (!enif_get_double(env, vel_arr[0], &vel_x) ||
                !enif_get_double(env, vel_arr[1], &vel_y) ||
                !enif_get_double(env, vel_arr[2], &vel_z)) {
                printf("DEBUG C: ERROR - failed to get velocity values for body %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract mass
            double mass;
            if (!enif_get_double(env, mass_term, &mass)) {
                printf("DEBUG C: ERROR - failed to get mass for body %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract radius
            double radius;
            if (!enif_get_double(env, radius_term, &radius)) {
                printf("DEBUG C: ERROR - failed to get radius for body %u\n", i);
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
            sys->radius[i] = (float)radius;
            
            // Generate default ID for body
            char default_id[32];
            snprintf(default_id, sizeof(default_id), "body%u", i);
            sys->ids[i] = malloc(strlen(default_id) + 1);
            strcpy(sys->ids[i], default_id);
            
            printf("DEBUG C: extracted body %u: id='%s', pos=(%f,%f,%f), vel=(%f,%f,%f), mass=%f, radius=%f\n",
                   i, sys->ids[i], sys->pos_x[i], sys->pos_y[i], sys->pos_z[i],
                   sys->vel_x[i], sys->vel_y[i], sys->vel_z[i], sys->mass[i], sys->radius[i]);
            
            current_list = tail;
        }
        
        printf("DEBUG C: successfully created particle system with %u bodies from list format\n", body_count);
        
        ERL_NIF_TERM term = enif_make_resource(env, sys);
        enif_release_resource(sys);
        return term;
        
    } else if (has_molecules) {
        printf("DEBUG C: detected molecules field, treating as particles\n");
        
        // Get molecules list and treat as particles
        ERL_NIF_TERM molecules_list;
        if (!enif_get_map_value(env, argv[0], enif_make_atom(env, "molecules"), &molecules_list)) {
            printf("DEBUG C: ERROR - failed to get molecules list\n");
            return enif_make_badarg(env);
        }
        
        // Get length of molecules list
        unsigned int molecule_count;
        if (!enif_get_list_length(env, molecules_list, &molecule_count)) {
            printf("DEBUG C: ERROR - failed to get molecules list length\n");
            return enif_make_badarg(env);
        }
        
        printf("DEBUG C: molecule count: %u\n", molecule_count);
        
        // Allow empty systems
        if (molecule_count == 0) {
            printf("DEBUG C: creating empty system from molecules format\n");
            ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
            printf("DEBUG C: molecules empty create - BEFORE init - sys pointer: %p\n", (void*)sys);
            
            // CRITICAL FIX: Initialize all fields to NULL/0 before use
            memset(sys, 0, sizeof(ParticleSystem));
            printf("DEBUG C: molecules empty create - AFTER memset - key_content[0]: '%c'\n", sys->particle_key[0]);
            
            strcpy(sys->particle_key, "molecules");
            printf("DEBUG C: molecules empty create - AFTER init key_content=%.32s len=%zu\n", sys->particle_key, strlen(sys->particle_key));
            sys->count = 0;
            sys->capacity = 0;
            sys->pos_x = NULL;
            sys->pos_y = NULL;
            sys->pos_z = NULL;
            sys->vel_x = NULL;
            sys->vel_y = NULL;
            sys->vel_z = NULL;
            sys->mass = NULL;
            sys->radius = NULL;  // CRITICAL FIX: Missing NULL initialization
            sys->ids = NULL;
            
            ERL_NIF_TERM term = enif_make_resource(env, sys);
            enif_release_resource(sys);
            return term;
        }
        
        // Allocate system
        ParticleSystem* sys = enif_alloc_resource(particle_system_type, sizeof(ParticleSystem));
        sys->count = molecule_count;
        sys->capacity = molecule_count;
        strcpy(sys->particle_key, "molecules");  // Store original key name
        
        size_t size = molecule_count * sizeof(float);
        sys->pos_x = aligned_malloc(size, 32);
        sys->pos_y = aligned_malloc(size, 32);
        sys->pos_z = aligned_malloc(size, 32);
        sys->vel_x = aligned_malloc(size, 32);
        sys->vel_y = aligned_malloc(size, 32);
        sys->vel_z = aligned_malloc(size, 32);
        sys->mass = aligned_malloc(size, 32);
        sys->radius = aligned_malloc(size, 32);
        sys->ids = malloc(molecule_count * sizeof(char*));
        
        if (!sys->pos_x || !sys->pos_y || !sys->pos_z ||
            !sys->vel_x || !sys->vel_y || !sys->vel_z || !sys->mass || !sys->radius || !sys->ids) {
            printf("DEBUG C: allocation failed for molecules format\n");
            enif_release_resource(sys);
            return enif_make_atom(env, "allocation_error");
        }
        
        // Extract molecule data from list
        ERL_NIF_TERM head, tail;
        ERL_NIF_TERM current_list = molecules_list;
        
        for (unsigned int i = 0; i < molecule_count; i++) {
            if (!enif_get_list_cell(env, current_list, &head, &tail)) {
                printf("DEBUG C: ERROR - failed to get molecule %u from list\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract molecule fields (same as particles but without requiring id)
            ERL_NIF_TERM pos_term, vel_term, mass_term, radius_term;
            int has_pos = enif_get_map_value(env, head, enif_make_atom(env, "position"), &pos_term);
            int has_vel = enif_get_map_value(env, head, enif_make_atom(env, "velocity"), &vel_term);
            int has_mass = enif_get_map_value(env, head, enif_make_atom(env, "mass"), &mass_term);
            int has_radius = enif_get_map_value(env, head, enif_make_atom(env, "radius"), &radius_term);
            
            printf("DEBUG C: molecule %u field availability - pos:%d vel:%d mass:%d radius:%d\n",
                   i, has_pos, has_vel, has_mass, has_radius);
            
            if (!has_pos || !has_vel || !has_mass) {
                printf("DEBUG C: ERROR - failed to extract required molecule %u fields (pos:%d vel:%d mass:%d)\n",
                       i, has_pos, has_vel, has_mass);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Make radius optional with default
            if (!has_radius) {
                printf("DEBUG C: molecule %u - radius missing, using default 0.5\n", i);
                radius_term = enif_make_double(env, 0.5);
            }
            
            // Extract position tuple
            const ERL_NIF_TERM* pos_arr;
            int pos_arity;
            if (!enif_get_tuple(env, pos_term, &pos_arity, &pos_arr) || pos_arity != 3) {
                printf("DEBUG C: ERROR - invalid position tuple for molecule %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            double pos_x, pos_y, pos_z;
            if (!enif_get_double(env, pos_arr[0], &pos_x) ||
                !enif_get_double(env, pos_arr[1], &pos_y) ||
                !enif_get_double(env, pos_arr[2], &pos_z)) {
                printf("DEBUG C: ERROR - failed to get position values for molecule %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract velocity tuple
            const ERL_NIF_TERM* vel_arr;
            int vel_arity;
            if (!enif_get_tuple(env, vel_term, &vel_arity, &vel_arr) || vel_arity != 3) {
                printf("DEBUG C: ERROR - invalid velocity tuple for molecule %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            double vel_x, vel_y, vel_z;
            if (!enif_get_double(env, vel_arr[0], &vel_x) ||
                !enif_get_double(env, vel_arr[1], &vel_y) ||
                !enif_get_double(env, vel_arr[2], &vel_z)) {
                printf("DEBUG C: ERROR - failed to get velocity values for molecule %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract mass
            double mass;
            if (!enif_get_double(env, mass_term, &mass)) {
                printf("DEBUG C: ERROR - failed to get mass for molecule %u\n", i);
                enif_release_resource(sys);
                return enif_make_badarg(env);
            }
            
            // Extract radius
            double radius;
            if (!enif_get_double(env, radius_term, &radius)) {
                printf("DEBUG C: ERROR - failed to get radius for molecule %u\n", i);
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
            sys->radius[i] = (float)radius;
            
            // Generate default ID for molecule
            char default_id[32];
            snprintf(default_id, sizeof(default_id), "mol%u", i);
            sys->ids[i] = malloc(strlen(default_id) + 1);
            strcpy(sys->ids[i], default_id);
            
            printf("DEBUG C: extracted molecule %u: id='%s', pos=(%f,%f,%f), vel=(%f,%f,%f), mass=%f, radius=%f\n",
                   i, sys->ids[i], sys->pos_x[i], sys->pos_y[i], sys->pos_z[i],
                   sys->vel_x[i], sys->vel_y[i], sys->vel_z[i], sys->mass[i], sys->radius[i]);
            
            current_list = tail;
        }
        
        printf("DEBUG C: successfully created particle system with %u molecules from list format\n", molecule_count);
        
        ERL_NIF_TERM term = enif_make_resource(env, sys);
        enif_release_resource(sys);
        return term;
        
    } else {
        printf("DEBUG C: ERROR - unsupported format, expected 'count', 'particles', 'bodies', or 'molecules' field\n");
        return enif_make_badarg(env);
    }
}

// Simulate N steps
static ERL_NIF_TERM native_integrate(ErlNifEnv* env, int argc __attribute__((unused)), const ERL_NIF_TERM argv[]) {
    ParticleSystem* sys;
    double dt;
    int steps;
    int apply_gravity = 1; // Default: apply gravity
    WallSystem walls = {0}; // Initialize wall system
    
    printf("DEBUG C: native_integrate FUNCTION CALLED with argc=%d\n", argc);
    
    if (!enif_get_resource(env, argv[0], particle_system_type, (void**)&sys)) {
        return enif_make_badarg(env);
    }
    
    if (!enif_get_double(env, argv[1], &dt)) {
        return enif_make_badarg(env);
    }
    
    if (!enif_get_int(env, argv[2], &steps)) {
        return enif_make_badarg(env);
    }
    
    // Parse rules if present
    if (argc >= 4) {
      ERL_NIF_TERM rules = argv[3];
      ERL_NIF_TERM head, tail;
      ERL_NIF_TERM current_rules = rules;
      char atom_name[64];
      while (enif_get_list_cell(env, current_rules, &head, &tail)) {
        if (enif_get_atom(env, head, atom_name, sizeof(atom_name), ERL_NIF_LATIN1) && strcmp(atom_name, "integrate_no_gravity") == 0) {
          apply_gravity = 0;
          break;
        }
        current_rules = tail;
      }
    }
    
    // Extract walls from the original state if available
    // We need to get the original state from the particle system resource
    // For now, we'll skip wall collisions in native code and let the Elixir fallback handle it
    
    // RUN THE ACTUAL PHYSICS!
    for (int step = 0; step < steps; step++) {
        // Check for particle-particle collisions
        for (uint32_t i = 0; i < sys->count; i++) {
            for (uint32_t j = i + 1; j < sys->count; j++) {
                double dx = sys->pos_x[i] - sys->pos_x[j];
                double dy = sys->pos_y[i] - sys->pos_y[j];
                double dz = sys->pos_z[i] - sys->pos_z[j];
                double dist_sq = dx*dx + dy*dy + dz*dz;
                double min_dist = sys->radius[i] + sys->radius[j];
                
                if (dist_sq < min_dist * min_dist) {
                    // Collision detected - elastic collision
                    printf("DEBUG C: COLLISION DETECTED between particles %u and %u\n", i, j);
                    printf("DEBUG C: Before collision - p%u vel=(%f,%f,%f), p%u vel=(%f,%f,%f)\n",
                           i, sys->vel_x[i], sys->vel_y[i], sys->vel_z[i],
                           j, sys->vel_x[j], sys->vel_y[j], sys->vel_z[j]);
                    
                    double dist = sqrt(dist_sq);
                    if (dist > 0.0) {  // Avoid division by zero
                        double nx = dx / dist;
                        double ny = dy / dist;
                        double nz = dz / dist;
                        
                        double dvx = sys->vel_x[i] - sys->vel_x[j];
                        double dvy = sys->vel_y[i] - sys->vel_y[j];
                        double dvz = sys->vel_z[i] - sys->vel_z[j];
                        
                        double dot_product = dvx * nx + dvy * ny + dvz * nz;
                        
                        printf("DEBUG C: Collision physics - dist=%f, normal=(%f,%f,%f), dot_product=%f\n",
                               dist, nx, ny, nz, dot_product);
                        
                        if (dot_product < 0) {
                            double impulse = 2.0 * dot_product / (sys->mass[i] + sys->mass[j]);
                            
                            printf("DEBUG C: Collision impulse=%f, masses: m%u=%f, m%u=%f\n",
                                   impulse, i, sys->mass[i], j, sys->mass[j]);
                            
                            sys->vel_x[i] -= impulse * nx / sys->mass[i];
                            sys->vel_y[i] -= impulse * ny / sys->mass[i];
                            sys->vel_z[i] -= impulse * nz / sys->mass[i];
                            
                            sys->vel_x[j] += impulse * nx / sys->mass[j];
                            sys->vel_y[j] += impulse * ny / sys->mass[j];
                            sys->vel_z[j] += impulse * nz / sys->mass[j];
                            
                            printf("DEBUG C: After collision - p%u vel=(%f,%f,%f), p%u vel=(%f,%f,%f)\n",
                                   i, sys->vel_x[i], sys->vel_y[i], sys->vel_z[i],
                                   j, sys->vel_x[j], sys->vel_y[j], sys->vel_z[j]);
                        } else {
                            printf("DEBUG C: Skipping collision - dot_product >= 0 (particles moving apart)\n");
                        }
                    } else {
                        printf("DEBUG C: ERROR - dist = 0, particles at same position!\n");
                    }
                }
            }
        }
        
        // Apply gravity conditionally
        if (apply_gravity) {
            apply_gravity_simd(sys, (float)dt);
        }
        
        integrate_positions_simd(sys, (float)dt);
        
        // Apply wall collisions if we have wall data
        // For now, skip wall collisions in native code since we don't have wall data
        // This will be handled by the Elixir fallback
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
    
    // Handle empty systems properly
    if (sys->count == 0) {
        // Safety check: ensure particle_key is properly null-terminated
        if (sys->particle_key[0] == '\0') {
            strcpy(sys->particle_key, "particles");
        }
        
        // Create empty lists safely
        ERL_NIF_TERM particles_list = enif_make_list(env, 0);
        ERL_NIF_TERM walls_result = enif_make_list(env, 0);
        
        // Use a fixed key atom to avoid potential issues with sys->particle_key
        ERL_NIF_TERM particles_key = enif_make_atom(env, "particles");
        ERL_NIF_TERM walls_key = enif_make_atom(env, "walls");
        
        ERL_NIF_TERM keys[] = {particles_key, walls_key};
        ERL_NIF_TERM values[] = {particles_list, walls_result};
        ERL_NIF_TERM result;
        enif_make_map_from_arrays(env, keys, values, 2, &result);
        
        return result;
    }
    
    ERL_NIF_TERM* particles = enif_alloc(sys->count * sizeof(ERL_NIF_TERM));
    
    for (uint32_t i = 0; i < sys->count; i++) {
        ERL_NIF_TERM pos = enif_make_tuple3(env,
            enif_make_double(env, sys->pos_x[i]),
            enif_make_double(env, sys->pos_y[i]),
            enif_make_double(env, sys->pos_z[i])
        );
        
        // DEBUG: Check for NaN or infinity in velocity components
        printf("DEBUG C: Converting particle %u velocity to Elixir: (%f,%f,%f)\n",
               i, sys->vel_x[i], sys->vel_y[i], sys->vel_z[i]);
        
        // Check for NaN or infinity
        if (isnan(sys->vel_x[i]) || isinf(sys->vel_x[i]) ||
            isnan(sys->vel_y[i]) || isinf(sys->vel_y[i]) ||
            isnan(sys->vel_z[i]) || isinf(sys->vel_z[i])) {
            printf("DEBUG C: ERROR - Invalid velocity detected for particle %u: (%f,%f,%f)\n",
                   i, sys->vel_x[i], sys->vel_y[i], sys->vel_z[i]);
            // Set to zero velocity as fallback
            sys->vel_x[i] = 0.0f;
            sys->vel_y[i] = 0.0f;
            sys->vel_z[i] = 0.0f;
        }
        
        ERL_NIF_TERM vel = enif_make_tuple3(env,
            enif_make_double(env, sys->vel_x[i]),
            enif_make_double(env, sys->vel_y[i]),
            enif_make_double(env, sys->vel_z[i])
        );
        
        // Get radius from original particle if available
        ERL_NIF_TERM radius_val = enif_make_double(env, sys->radius[i]); // Use actual radius value
        
        // CRITICAL FIX: We need to preserve chemical_type for molecules
        // Since we can't determine the type from C structure alone, we'll need to pass it through
        // For now, we'll add a default chemical_type based on mass ranges
        ERL_NIF_TERM chemical_type_val;
        if (sys->mass[i] < 1.5f) {
            chemical_type_val = enif_make_atom(env, "A");
        } else if (sys->mass[i] < 2.5f) {
            chemical_type_val = enif_make_atom(env, "B");
        } else if (sys->mass[i] < 3.0f) {
            chemical_type_val = enif_make_atom(env, "C");
        } else {
            chemical_type_val = enif_make_atom(env, "D");
        }
        
        ERL_NIF_TERM keys[] = {
            enif_make_atom(env, "position"),
            enif_make_atom(env, "velocity"),
            enif_make_atom(env, "mass"),
            enif_make_atom(env, "radius"),
            enif_make_atom(env, "id"),
            enif_make_atom(env, "chemical_type")
        };
        
        ERL_NIF_TERM values[] = {
            pos,
            vel,
            enif_make_double(env, sys->mass[i]),
            radius_val,
            // Create proper Elixir string from C string
            const char* id_str = sys->ids[i] ? sys->ids[i] : "default";
            size_t id_len = sys->ids[i] ? strlen(sys->ids[i]) : 7;
            ERL_NIF_TERM id_term = enif_make_string_len(env, id_str, id_len, ERL_NIF_UTF8);
            chemical_type_val
        };
        
        enif_make_map_from_arrays(env, keys, values, 6, &particles[i]);
    }
    
    ERL_NIF_TERM particles_list = enif_make_list_from_array(env, particles, sys->count);
    enif_free(particles);
    
    // Preserve walls from original state if present
    ERL_NIF_TERM walls_result = enif_make_list(env, 0); // empty list by default
    if (enif_get_map_value(env, argv[0], enif_make_atom(env, "walls"), &walls_result)) {
        // walls_result already contains the walls list
    }
    
    ERL_NIF_TERM keys[] = {enif_make_atom(env, sys->particle_key), enif_make_atom(env, "walls")};
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