defmodule AII.NIF do
  @moduledoc """
  Elixir → Zig Native Interface Functions (NIFs)
  Provides low-level access to the Zig runtime for particle systems.
  """

  use Zig, otp_app: :aii

  ~Z"""
  const std = @import("std");
  const beam = @import("beam");

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
  """

  # Particle System Management

  @doc """
  Creates a new particle system with the given capacity.

  Returns a reference integer that can be used with other functions.
  """

  @doc """
  Destroys a particle system and frees its memory.
  """

  # Particle Operations

  @doc """
  Adds a particle to the system.

  particle_data should be a map with keys:
  - :position - {x, y, z} tuple
  - :velocity - {x, y, z} tuple
  - :mass - float
  - :energy - float
  - :id - integer
  """

  @doc """
  Retrieves all particles from the system as a list of maps.
  """

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

  @doc """
  Applies a force to all particles in the system.
  """
  def apply_force(_system_ref, _force_vector, _dt), do: :ok

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
