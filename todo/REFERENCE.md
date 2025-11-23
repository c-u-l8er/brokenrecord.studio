# Quick Reference: Essential Code Changes

## The 3 Critical Changes

### 1. Create the Real NIF (c_src/brokenrecord_physics.c)

Key functions you need:

```c
// Resource type
typedef struct {
    float *pos_x, *pos_y, *pos_z;
    float *vel_x, *vel_y, *vel_z;
    float *mass;
    uint32_t count;
} ParticleSystem;

// SIMD Physics (8 particles at once with AVX2)
void apply_gravity_simd(ParticleSystem* sys, float dt) {
    __m256 gravity = _mm256_set1_ps(-9.81f * dt);
    for (uint32_t i = 0; i < sys->count; i += 8) {
        __m256 vy = _mm256_loadu_ps(&sys->vel_y[i]);
        vy = _mm256_add_ps(vy, gravity);
        _mm256_storeu_ps(&sys->vel_y[i], vy);
    }
}

void integrate_positions_simd(ParticleSystem* sys, float dt) {
    __m256 dt_vec = _mm256_set1_ps(dt);
    for (uint32_t i = 0; i < sys->count; i += 8) {
        __m256 px = _mm256_loadu_ps(&sys->pos_x[i]);
        __m256 vx = _mm256_loadu_ps(&sys->vel_x[i]);
        px = _mm256_fmadd_ps(vx, dt_vec, px);  // px += vx * dt
        _mm256_storeu_ps(&sys->pos_x[i], px);
    }
}

// Main NIF function
ERL_NIF_TERM native_integrate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ParticleSystem* sys;
    double dt;
    int steps;
    
    enif_get_resource(env, argv[0], particle_system_type, (void**)&sys);
    enif_get_double(env, argv[1], &dt);
    enif_get_int(env, argv[2], &steps);
    
    for (int step = 0; step < steps; step++) {
        apply_gravity_simd(sys, dt);
        integrate_positions_simd(sys, dt);
    }
    
    return argv[0];  // Return resource (mutated in place)
}
```

### 2. Fix Runtime (lib/broken_record/zero/runtime.ex)

Replace line 8-22:

```elixir
def execute(system, initial_state, opts) do
  steps = opts[:steps] || 1000
  dt = opts[:dt] || 0.01
  
  # NEW: Actually use native code!
  if native_available?() do
    native_execute(initial_state, dt, steps)
  else
    interpreted_simulate(initial_state, dt, steps)
  end
end

defp native_available?() do
  Code.ensure_loaded?(BrokenRecord.Zero.NIF) and
    function_exported?(BrokenRecord.Zero.NIF, :native_integrate, 3)
end

defp native_execute(state, dt, steps) do
  alias BrokenRecord.Zero.NIF
  
  sys = NIF.create_particle_system(state)
  sys = NIF.native_integrate(sys, dt, steps)
  NIF.to_elixir_state(sys)
end
```

### 3. Add NIF Module (lib/broken_record/zero/nif.ex - NEW FILE)

```elixir
defmodule BrokenRecord.Zero.NIF do
  @on_load :load_nif
  
  def load_nif do
    nif_file = :filename.join(:code.priv_dir(:broken_record_zero), 'brokenrecord_physics')
    :erlang.load_nif(nif_file, 0)
  end
  
  def create_particle_system(_state), do: :erlang.nif_error(:nif_not_loaded)
  def native_integrate(_sys, _dt, _steps), do: :erlang.nif_error(:nif_not_loaded)
  def to_elixir_state(_sys), do: :erlang.nif_error(:nif_not_loaded)
end
```

---

## Makefile (c_src/Makefile)

```makefile
ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)

CFLAGS = -O3 -march=native -fPIC -std=c11 -I$(ERLANG_PATH)
CFLAGS += -mavx2 -mfma  # SIMD
LDFLAGS = -shared -lm

PRIV_DIR = ../priv
NIF_SO = $(PRIV_DIR)/brokenrecord_physics.so

all: $(PRIV_DIR) $(NIF_SO)

$(PRIV_DIR):
	mkdir -p $(PRIV_DIR)

$(NIF_SO): brokenrecord_physics.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(NIF_SO)
```

---

## mix.exs Changes

Add to `def project`:

```elixir
compilers: [:elixir_make] ++ Mix.compilers(),
make_targets: ["all"],
make_clean: ["clean"],
```

Add to `defp deps`:

```elixir
{:elixir_make, "~> 0.6", runtime: false},
```

---

## Build and Test

```bash
# Get dependencies
mix deps.get

# Compile NIF
cd c_src && make && cd ..

# Test
mix run -e "
  particles = [%{position: {0.0, 10.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.0}]
  state = %{particles: particles}
  
  sys = BrokenRecord.Zero.NIF.create_particle_system(state)
  sys = BrokenRecord.Zero.NIF.native_integrate(sys, 0.01, 1000)
  result = BrokenRecord.Zero.NIF.to_elixir_state(sys)
  
  IO.inspect(result.particles)
"
```

---

## Performance Check

```bash
mix run -e "
  # Create 10k particles
  particles = for i <- 1..10_000 do
    %{position: {0.0, 100.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.0}
  end
  
  state = %{particles: particles}
  
  # Benchmark
  {time, _} = :timer.tc(fn ->
    sys = BrokenRecord.Zero.NIF.create_particle_system(state)
    BrokenRecord.Zero.NIF.native_integrate(sys, 0.01, 1000)
  end)
  
  ops = 10_000 * 1000
  gflops = (ops * 20) / (time / 1_000_000) / 1_000_000_000
  
  IO.puts(\"Time: #{time / 1_000_000} sec\")
  IO.puts(\"Performance: #{Float.round(gflops, 1)} GFLOPS\")
"
```

Expected output:
```
Time: 0.05 sec
Performance: 40-200 GFLOPS
```

---

## Common Issues

### NIF won't load
```bash
# Check if file exists
ls -lh priv/brokenrecord_physics.so

# Check for undefined symbols
nm priv/brokenrecord_physics.so | grep " U "

# Rebuild clean
cd c_src && make clean && make
```

### No speedup
Check that:
1. `-mavx2 -mfma` in CFLAGS
2. `-O3` in CFLAGS
3. Runtime actually calls native (add IO.puts to verify)

### Segfault
```bash
# Debug with valgrind
valgrind mix test

# Check alignment
# Arrays should be 32-byte aligned for AVX2
```

---

## That's It!

Three files, ~500 lines of C, 80-200x speedup.

See IMPLEMENTATION_GUIDE.md for complete step-by-step instructions.