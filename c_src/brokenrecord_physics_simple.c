#include <erl_nif.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static ErlNifResourceType* particle_system_type = NULL;

static void particle_system_destructor(ErlNifEnv* env __attribute__((unused)), void* obj) {
    // Simple destructor - do nothing for now
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

// Simple stub that just returns the input
static ERL_NIF_TERM native_integrate(ErlNifEnv* env, int argc __attribute__((unused)), const ERL_NIF_TERM argv[]) {
    printf("DEBUG C: native_integrate called successfully!\n");
    return argv[0];  // Just return the first argument unchanged
}

static ErlNifFunc nif_funcs[] = {
    {"native_integrate", 4, native_integrate, 0},
};

ERL_NIF_INIT(Elixir.BrokenRecord.Zero.NIF, nif_funcs, load, NULL, NULL, NULL)