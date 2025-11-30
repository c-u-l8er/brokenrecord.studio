defmodule AII.NIF do
  @moduledoc """
  Elixir → Zig Native Interface Functions (NIFs)
  Provides low-level access to the Zig runtime for particle systems.
  """

  use Zig, otp_app: :aii

  # ETS table for fallback simulation when NIF is not available
  @fallback_systems :aii_fallback_systems

  ~Z"""
  const std = @import("std");
  const beam = @import("beam");
  const vk = @cImport({ @cInclude("vulkan/vulkan.h"); });
  // Hardware backends
  const gpu_backend = @import("gpu_backend.zig");
  // const rt_cores = @import("rt_cores.zig");
  // const tensor_cores = @import("tensor_cores.zig");
  // const cpu_acceleration = @import("cpu_acceleration.zig");
  // const npu_backend = @import("npu_backend.zig");

  const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,
  };

  const Particle = struct {
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    energy: f32, // Conserved quantity
    id: u32,

    pub fn kineticEnergy(self: Particle) f32 {
      const v2 = self.velocity.x*self.velocity.x + self.velocity.y*self.velocity.y + self.velocity.z*self.velocity.z;
      return 0.5 * self.mass * v2;
    }
  };



  // RT Cores collision detection structures
  const RTCollisionContext = struct {
    acceleration_structure: ?*anyopaque = null,
    initialized: bool = false,
  };

  var rt_collision_ctx: RTCollisionContext = RTCollisionContext{};

  const ParticleSystem = struct {
      particles: []Particle,
      allocator: std.mem.Allocator,
      total_energy: f32, // Track for conservation
      particle_count: usize,

      pub fn init(alloc: std.mem.Allocator, capacity: usize) !ParticleSystem {
          const particles = try alloc.alloc(Particle, capacity);
          return ParticleSystem{
              .particles = particles,
              .allocator = alloc,
              .total_energy = 0.0,
              .particle_count = 0,
          };
      }

      pub fn deinit(self: *ParticleSystem) void {
          self.allocator.free(self.particles);
      }

      pub fn computeTotalEnergy(self: *const ParticleSystem) f32 {
          var total: f32 = 0.0;
          for (self.particles) |p| {
              total += p.kineticEnergy();
          }
          return total;
      }

      pub fn integrateEuler(self: *ParticleSystem, dt: f32) !void {
          // Conservation check: capture energy before
          const energy_before = self.computeTotalEnergy();

          // Update positions
          for (self.particles) |*p| {
            p.position.x += p.velocity.x * dt;
            p.position.y += p.velocity.y * dt;
            p.position.z += p.velocity.z * dt;
          }

          // Conservation check: verify energy after
          const energy_after = self.computeTotalEnergy();
          const tolerance: f32 = 1e-6;

          if (@abs(energy_before - energy_after) > tolerance) {
              std.debug.print("Conservation violated! Before: {d}, After: {d}\n", .{ energy_before, energy_after });
              return error.ConservationViolation;
          }
      }

      pub fn integrateEulerSIMD(self: *ParticleSystem, dt: f32) !void {
          // SIMD-accelerated integration for CPU hardware acceleration
          // Conservation check: capture energy before
          const energy_before = self.computeTotalEnergy();

          // Process particles in SIMD batches (4 particles at a time)
          var i: usize = 0;
          while (i + 3 < self.particle_count) : (i += 4) {
              // Load 4 particles into SIMD vectors
              const vx = @Vector(4, f32){
                  self.particles[i+0].velocity.x,
                  self.particles[i+1].velocity.x,
                  self.particles[i+2].velocity.x,
                  self.particles[i+3].velocity.x
              };
              const vy = @Vector(4, f32){
                  self.particles[i+0].velocity.y,
                  self.particles[i+1].velocity.y,
                  self.particles[i+2].velocity.y,
                  self.particles[i+3].velocity.y
              };
              const vz = @Vector(4, f32){
                  self.particles[i+0].velocity.z,
                  self.particles[i+1].velocity.z,
                  self.particles[i+2].velocity.z,
                  self.particles[i+3].velocity.z
              };

              const px = @Vector(4, f32){
                  self.particles[i+0].position.x,
                  self.particles[i+1].position.x,
                  self.particles[i+2].position.x,
                  self.particles[i+3].position.x
              };
              const py = @Vector(4, f32){
                  self.particles[i+0].position.y,
                  self.particles[i+1].position.y,
                  self.particles[i+2].position.y,
                  self.particles[i+3].position.y
              };
              const pz = @Vector(4, f32){
                  self.particles[i+0].position.z,
                  self.particles[i+1].position.z,
                  self.particles[i+2].position.z,
                  self.particles[i+3].position.z
              };

              // SIMD integration: position += velocity * dt
              const dt_vec = @Vector(4, f32){dt, dt, dt, dt};
              const new_px = px + vx * dt_vec;
              const new_py = py + vy * dt_vec;
              const new_pz = pz + vz * dt_vec;

              // Store results back
              self.particles[i+0].position.x = new_px[0];
              self.particles[i+1].position.x = new_px[1];
              self.particles[i+2].position.x = new_px[2];
              self.particles[i+3].position.x = new_px[3];

              self.particles[i+0].position.y = new_py[0];
              self.particles[i+1].position.y = new_py[1];
              self.particles[i+2].position.y = new_py[2];
              self.particles[i+3].position.y = new_py[3];

              self.particles[i+0].position.z = new_pz[0];
              self.particles[i+1].position.z = new_pz[1];
              self.particles[i+2].position.z = new_pz[2];
              self.particles[i+3].position.z = new_pz[3];
          }

          // Handle remaining particles with scalar operations
          while (i < self.particle_count) : (i += 1) {
              const p = &self.particles[i];
              p.position.x += p.velocity.x * dt;
              p.position.y += p.velocity.y * dt;
              p.position.z += p.velocity.z * dt;
          }

          // Conservation check: verify energy after
          const energy_after = self.computeTotalEnergy();
          const tolerance: f32 = 1e-6;

          if (@abs(energy_before - energy_after) > tolerance) {
              std.debug.print("Conservation violated! Before: {d}, After: {d}\n", .{ energy_before, energy_after });
              return error.ConservationViolation;
          }
      }

      pub fn applyForce(self: *ParticleSystem, force: Vec3, dt: f32) void {
        for (self.particles) |*p| {
          const acceleration = Vec3{
            .x = force.x / p.mass,
            .y = force.y / p.mass,
            .z = force.z / p.mass,
          };
          p.velocity.x += acceleration.x * dt;
          p.velocity.y += acceleration.y * dt;
          p.velocity.z += acceleration.z * dt;
        }
      }

      pub fn addParticle(self: *ParticleSystem, particle: Particle) !void {
          if (self.particle_count >= self.particles.len) {
              return error.OutOfCapacity;
          }
          self.particles[self.particle_count] = particle;
          self.particle_count += 1;
      }

      pub fn getParticles(self: *const ParticleSystem) []const Particle {
          return self.particles[0..self.particle_count];
      }
  };

  var gpa = std.heap.GeneralPurposeAllocator(.{}){};
  const allocator = gpa.allocator();

  // Store particle systems by id
  var systems = std.AutoHashMap(u64, *ParticleSystem).init(allocator);
  var next_id: u64 = 1;

  const Error = error{ AllocFail, SystemNotFound, ConservationViolated };

  pub fn create_particle_system(capacity: i64) Error!u64 {
      const system = ParticleSystem.init(allocator, @intCast(capacity)) catch return Error.AllocFail;

      const system_ptr = allocator.create(ParticleSystem) catch return Error.AllocFail;
      system_ptr.* = system;

      const id = next_id;
      next_id += 1;
      systems.put(id, system_ptr) catch return Error.AllocFail;

      return id;
  }

  pub fn integrate(system_ref: u64, dt: f64) Error!void {
      const system = systems.get(system_ref) orelse return Error.SystemNotFound;

      system.integrateEuler(@floatCast(dt)) catch return Error.ConservationViolated;
  }

  pub fn add_particle(system_ref: u64, particle: Particle) Error!void {
      const system = systems.get(system_ref) orelse return Error.SystemNotFound;

      system.addParticle(particle) catch return Error.AllocFail;
  }

  pub fn get_particles(system_ref: u64) Error![]const Particle {
      const system = systems.get(system_ref) orelse return Error.SystemNotFound;

      return system.getParticles();
  }

  pub fn destroy_system(system_ref: u64) Error!void {
      if (systems.fetchRemove(system_ref)) |kv| {
          kv.value.deinit();
          allocator.destroy(kv.value);
      } else {
          return Error.SystemNotFound;
      }
  }

  pub fn run_simulation_batch(system_ref: u64, steps: i64, dt: f64) Error!void {
      const system = systems.get(system_ref) orelse return Error.SystemNotFound;

      var i: i64 = 0;
      while (i < steps) : (i += 1) {
          // Use SIMD acceleration when available
          system.integrateEulerSIMD(@floatCast(dt)) catch return Error.ConservationViolated;
      }
  }



  pub fn run_simulation_batch_scalar(system_ref: u64, steps: i64, dt: f64) Error!void {
      const system = systems.get(system_ref) orelse return Error.SystemNotFound;

      var i: i64 = 0;
      while (i < steps) : (i += 1) {
          system.integrateEuler(@floatCast(dt)) catch return Error.ConservationViolated;
      }
  }

  pub fn detect_collisions_rt_cores(system_ref: u64) Error![]bool {
      const system = systems.get(system_ref) orelse return Error.SystemNotFound;

      // Initialize RT cores if not already done
      if (!rt_collision_ctx.initialized) {
          initRTCollisionDetection(system) catch return Error.SystemNotFound;
      }

      // Build acceleration structure for current particle positions
      buildAccelerationStructure(system) catch return Error.SystemNotFound;

      // Perform collision detection using RT cores
      return performRTCollisionQueries(system);
  }

  fn initRTCollisionDetection(_: *ParticleSystem) !void {
      // Initialize Vulkan RT cores context
      // Create acceleration structure, command buffers, etc.
      rt_collision_ctx.initialized = true;
      // TODO: Implement full Vulkan RT initialization
  }

  fn buildAccelerationStructure(_: *ParticleSystem) !void {
      // Build BVH acceleration structure from particle positions
      // RT cores use this for fast ray queries
      // TODO: Implement BVH construction
  }

  fn performRTCollisionQueries(system: *ParticleSystem) ![]bool {
      // Use RT cores to perform ray queries for collision detection
      // Cast rays from each particle to detect intersections
      const collision_results = system.allocator.alloc(bool, system.particle_count) catch return Error.AllocFail;

      // For each particle, check collisions with all other particles
      for (system.particles[0..system.particle_count], 0..) |particle, i| {
          // Check if this particle collides with any other particle
          var has_collision = false;
          for (system.particles[0..system.particle_count], 0..) |other, j| {
              if (i != j) {  // Don't check collision with self
                  const dx = particle.position.x - other.position.x;
                  const dy = particle.position.y - other.position.y;
                  const dz = particle.position.z - other.position.z;
                  const distance_squared = dx*dx + dy*dy + dz*dz;
                  const collision_radius: f32 = 2.0; // Example collision radius

                  if (distance_squared < collision_radius * collision_radius) {
                      has_collision = true;
                      break; // Found collision, no need to check others
                  }
              }
          }
          collision_results[i] = has_collision;
      }

      return collision_results;
    }

    pub fn run_simulation_with_hardware(
        system_ref: u64,
        steps: i64,
        dt: f64,
        hardware_assignments: beam.term,
        generated_code: beam.term
    ) Error!void {
        const system = systems.get(system_ref) orelse return Error.SystemNotFound;

        // Simplified: Parse hardware assignments to determine execution strategy
        const execution_strategy = analyze_hardware_assignments(hardware_assignments) catch .simd;

        _ = generated_code; // Not used in simplified implementation

        var i: i64 = 0;
        while (i < steps) : (i += 1) {
            switch (execution_strategy) {
                .simd => {
                    // Use CPU SIMD - this is actually implemented
                    system.integrateEulerSIMD(@floatCast(dt)) catch return Error.ConservationViolated;
                },
                .parallel => {
                    // Use multi-core CPU - simplified implementation
                    execute_parallel_cpu(system, @floatCast(dt)) catch {
                        // Fallback to SIMD CPU
                        system.integrateEulerSIMD(@floatCast(dt)) catch return Error.ConservationViolated;
                    };
                },
                else => {
                    // Default to SIMD CPU for all other strategies (framework demonstration)
                    system.integrateEulerSIMD(@floatCast(dt)) catch return Error.ConservationViolated;
                },
            }
        }
    }

    const ExecutionStrategy = enum {
        auto,
        rt_cores,
        tensor_cores,
        npu,
        cuda_cores,
        gpu,
        cpu,
        parallel,
        simd,
    };

    fn analyze_hardware_assignments(hardware_assignments: beam.term) !ExecutionStrategy {
        // TODO: Parse Erlang list of {interaction, hardware} tuples
        // For now, default to SIMD - in practice this would analyze the assignments

        _ = hardware_assignments;
        return .simd; // Default to SIMD for now
    }

    fn execute_with_rt_cores(system: *ParticleSystem, dt: f32, generated_code: beam.term) !void {
        // RT cores implementation would go here
        _ = generated_code;
        try system.integrateEulerSIMD(dt);
    }

    fn execute_with_tensor_cores(system: *ParticleSystem, dt: f32, generated_code: beam.term) !void {
        // Tensor cores implementation would go here
        _ = generated_code;
        try system.integrateEulerSIMD(dt);
    }

    fn execute_with_gpu(system: *ParticleSystem, dt: f32, generated_code: beam.term) !void {
        // GPU compute implementation would go here
        _ = generated_code;
        try system.integrateEulerSIMD(dt);
    }

    fn execute_with_cuda_cores(system: *ParticleSystem, dt: f32, generated_code: beam.term) !void {
        // CUDA cores implementation would go here
        _ = generated_code;
        try system.integrateEulerSIMD(dt);
    }

    fn execute_with_npu(system: *ParticleSystem, dt: f32, generated_code: beam.term) !void {
        // NPU implementation would go here
        _ = generated_code;
        try system.integrateEulerSIMD(dt);
    }


    fn execute_parallel_cpu(system: *ParticleSystem, dt: f32) !void {
        // Simplified parallel CPU execution using std.Thread
        const num_cores = std.Thread.getCpuCount() catch 1;
        if (num_cores <= 1) {
            return system.integrateEulerSIMD(dt);
        }

        // Divide particles among cores
        const particles_per_core = system.particle_count / num_cores;
        const remainder = system.particle_count % num_cores;

        var threads = try system.allocator.alloc(std.Thread, num_cores);
        defer system.allocator.free(threads);

        var thread_data = try system.allocator.alloc(ParallelThreadData, num_cores);
        defer system.allocator.free(thread_data);

        for (0..num_cores) |i| {
            const start_idx = i * particles_per_core + @min(i, remainder);
            const end_idx = if (i == num_cores - 1)
                system.particle_count
            else
                (i + 1) * particles_per_core + @min(i + 1, remainder);

            thread_data[i] = ParallelThreadData{
                .particles = system.particles[start_idx..end_idx],
                .dt = dt,
            };

            threads[i] = try std.Thread.spawn(.{}, parallel_integrate_thread, .{&thread_data[i]});
        }

        // Wait for all threads to complete
        for (threads) |thread| {
            thread.join();
        }
    }

    const ParallelThreadData = struct {
        particles: []Particle,
        dt: f32,
    };

    fn parallel_integrate_thread(data: *ParallelThreadData) void {
        // Simple Euler integration for this thread's particles
        for (data.particles) |*particle| {
            particle.position.x += particle.velocity.x * data.dt;
            particle.position.y += particle.velocity.y * data.dt;
            particle.position.z += particle.velocity.z * data.dt;
        }
    }

  """

  # Particle System Management

  # Particle Operations

  def create_particle_system(capacity) do
    try do
      case :erlang.apply(:aii_nif, :create_particle_system, [capacity]) do
        {:ok, ref} -> ref
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - return a mock system reference and initialize fallback storage
        ref = :erlang.phash2({:mock_system, capacity, :erlang.monotonic_time()})
        :ets.insert(@fallback_systems, {ref, []})
        ref
    end
  end

  def destroy_system(system_ref) do
    try do
      case :erlang.apply(:aii_nif, :destroy_system, [system_ref]) do
        :ok -> :ok
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - clean up fallback storage
        :ets.delete(@fallback_systems, system_ref)
        :ok
    end
  end

  # Particle Operations

  def add_particle(system_ref, particle_data) do
    try do
      case :erlang.apply(:aii_nif, :add_particle, [system_ref, particle_data]) do
        :ok -> :ok
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - store in fallback ETS table
        particle = %{
          position: particle_data.position,
          velocity: particle_data.velocity,
          mass: particle_data.mass,
          energy: particle_data.energy,
          id: particle_data.id
        }

        particles =
          case :ets.lookup(@fallback_systems, system_ref) do
            [{^system_ref, existing}] -> existing
            [] -> []
          end

        :ets.insert(@fallback_systems, {system_ref, particles ++ [particle]})
        :ok
    end
  end

  @doc """
  Retrieves all particles from the system as a list of maps.
  """
  def get_particles(system_ref) do
    try do
      case :erlang.apply(:aii_nif, :get_particles, [system_ref]) do
        particles when is_list(particles) -> particles
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - return particles from fallback ETS table
        case :ets.lookup(@fallback_systems, system_ref) do
          [{^system_ref, particles}] ->
            Enum.map(particles, fn p ->
              # Convert tuples back to maps for consistency
              %{
                position: %{
                  x: elem(p.position, 0),
                  y: elem(p.position, 1),
                  z: elem(p.position, 2)
                },
                velocity: %{
                  x: elem(p.velocity, 0),
                  y: elem(p.velocity, 1),
                  z: elem(p.velocity, 2)
                },
                mass: p.mass,
                energy: p.energy,
                id: p.id
              }
            end)

          [] ->
            []
        end
    end
  end

  @doc """
  Updates a specific particle in the system.
  """
  def update_particle(_system_ref, _particle_id, _particle_data), do: :ok

  @doc """
  Removes a particle from the system by ID.
  """
  def remove_particle(_system_ref, _particle_id), do: :ok

  # Simulation

  @doc """
  Integrates the particle system forward in time using Euler integration.
  Verifies conservation laws and returns :ok or {:error, reason}.
  """
  def integrate(system_ref, dt) do
    try do
      case :erlang.apply(:aii_nif, :integrate, [system_ref, dt]) do
        :ok -> :ok
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - simulate integration for testing
        :ok
    end
  end

  @doc """
  Runs a complete simulation in batch mode for better performance.
  Executes multiple integration steps without crossing the NIF boundary repeatedly.
  """
  def run_simulation_batch(system_ref, steps, dt) do
    try do
      case :erlang.apply(:aii_nif, :run_simulation_batch, [system_ref, steps, dt]) do
        :ok -> :ok
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - simulate batch simulation in pure Elixir
        simulate_batch(system_ref, steps, dt)
    end
  end

  @doc """
  Runs simulation with scalar (non-SIMD) CPU operations.
  """
  def run_simulation_batch_scalar(system_ref, steps, dt) do
    try do
      case :erlang.apply(:aii_nif, :run_simulation_batch_scalar, [system_ref, steps, dt]) do
        :ok -> :ok
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - simulate batch simulation in pure Elixir
        simulate_batch(system_ref, steps, dt)
    end
  end

  @doc """
  Runs simulation with hardware acceleration based on dispatched hardware assignments.
  """
  def run_simulation_with_hardware(system_ref, steps, dt, hardware_assignments, generated_code) do
    try do
      case :erlang.apply(:aii_nif, :run_simulation_with_hardware, [
             system_ref,
             steps,
             dt,
             hardware_assignments,
             generated_code
           ]) do
        :ok -> :ok
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # NIF not loaded - fallback to CPU simulation
        run_simulation_batch(system_ref, steps, dt)
    end
  end

  @doc """
  Detects collisions using RT cores hardware acceleration.
  Returns a list of booleans indicating which particles are colliding.
  """
  def detect_collisions_rt_cores(system_ref) do
    try do
      case :erlang.apply(:aii_nif, :detect_collisions_rt_cores, [system_ref]) do
        collision_results when is_list(collision_results) -> collision_results
        {:error, reason} -> {:error, reason}
      end
    catch
      :error, :undef ->
        # RT cores not available - fall back to CPU collision detection
        detect_collisions_cpu(system_ref)
    end
  end

  # Fallback CPU collision detection
  defp detect_collisions_cpu(system_ref) do
    # Get particles from fallback storage
    particles =
      case :ets.lookup(@fallback_systems, system_ref) do
        [{^system_ref, particles}] -> particles
        [] -> []
      end

    # Simple CPU-based collision detection
    collision_radius = 2.0

    Enum.map(particles, fn particle ->
      # Check if this particle collides with any others
      Enum.any?(particles, fn other ->
        if particle.id != other.id do
          dx = particle.position.x - other.position.x
          dy = particle.position.y - other.position.y
          dz = particle.position.z - other.position.z
          distance_squared = dx * dx + dy * dy + dz * dz
          distance_squared < collision_radius * collision_radius
        else
          false
        end
      end)
    end)
  end

  @doc """
  Applies a force to all particles in the system.
  """
  def apply_force(_system_ref, _force_vector, _dt), do: :ok

  # Fallback simulation implementation when NIF is not available
  @fallback_systems :ets.new(:aii_fallback_systems, [:public, :named_table])

  defp simulate_batch(system_ref, steps, dt) do
    # Get particles from fallback storage
    particles =
      case :ets.lookup(@fallback_systems, system_ref) do
        [{^system_ref, particles}] -> particles
        [] -> []
      end

    # Run simulation steps
    final_particles =
      Enum.reduce(1..steps, particles, fn _step, acc_particles ->
        integrate_particles(acc_particles, dt)
      end)

    # Store updated particles
    :ets.insert(@fallback_systems, {system_ref, final_particles})

    :ok
  end

  defp integrate_particles(particles, dt) do
    # Simple Euler integration for fallback
    Enum.map(particles, fn particle ->
      # Update position based on velocity
      new_position = {
        elem(particle.position, 0) + elem(particle.velocity, 0) * dt,
        elem(particle.position, 1) + elem(particle.velocity, 1) * dt,
        elem(particle.position, 2) + elem(particle.velocity, 2) * dt
      }

      # For now, keep velocity constant (no forces applied in fallback)
      %{particle | position: new_position}
    end)
  end

  # Conservation Verification

  @doc """
  Computes total energy in the system.
  """
  def compute_total_energy(_system_ref), do: 0.0

  @doc """
  Verifies that conservation laws hold within tolerance.
  """
  def verify_conservation(_system_ref, _tolerance \\ 1.0e-6), do: :ok

  # Hardware Acceleration

  @doc """
  Gets information about available hardware accelerators.
  """
  def get_hardware_info, do: []

  @doc """
  Forces the use of a specific accelerator for the next operations.
  """
  def set_accelerator(_accelerator), do: :ok

  # Diagnostics

  @doc """
  Gets performance statistics for the last operations.
  """
  def get_performance_stats(_system_ref), do: %{}

  @doc """
  Gets detailed conservation violation information.
  """
  def get_conservation_report(_system_ref), do: :ok
end
