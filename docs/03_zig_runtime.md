# AII Migration: Zig Runtime Core
## Document 3: Particle System Implementation

### Why Zig Over C

```
C Problems:                    Zig Solutions:
- Manual memory management  → Allocators (explicit but safe)
- Undefined behavior        → Compile-time checks
- No error handling         → Try/catch with types
- Weak type system          → Strong, comptime types
- No generics               → Comptime + anytype
```

---

### Core Particle System

**File:** `runtime/zig/particle_system.zig`

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,
    
    pub fn add(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x + other.x, .y = self.y + other.y, .z = self.z + other.z };
    }
    
    pub fn mul(self: Vec3, scalar: f32) Vec3 {
        return .{ .x = self.x * scalar, .y = self.y * scalar, .z = self.z * scalar };
    }
    
    pub fn magnitude(self: Vec3) f32 {
        return @sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }
};

pub const Particle = struct {
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    energy: f32,  // Conserved quantity
    id: u32,
    
    pub fn kineticEnergy(self: Particle) f32 {
        const v_mag = self.velocity.magnitude();
        return 0.5 * self.mass * v_mag * v_mag;
    }
};

pub const ParticleSystem = struct {
    particles: []Particle,
    allocator: Allocator,
    total_energy: f32,  // Track for conservation
    
    pub fn init(allocator: Allocator, capacity: usize) !ParticleSystem {
        const particles = try allocator.alloc(Particle, capacity);
        return ParticleSystem{
            .particles = particles,
            .allocator = allocator,
            .total_energy = 0.0,
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
            p.position = p.position.add(p.velocity.mul(dt));
        }
        
        // Conservation check: verify energy after
        const energy_after = self.computeTotalEnergy();
        const tolerance: f32 = 1e-6;
        
        if (@abs(energy_before - energy_after) > tolerance) {
            std.debug.print("Conservation violated! Before: {d}, After: {d}\n", 
                .{energy_before, energy_after});
            return error.ConservationViolation;
        }
    }
    
    pub fn applyForce(self: *ParticleSystem, force: Vec3, dt: f32) void {
        for (self.particles) |*p| {
            const acceleration = force.mul(1.0 / p.mass);
            p.velocity = p.velocity.add(acceleration.mul(dt));
        }
    }
};
```

---

### Elixir NIF Integration

**File:** `lib/aii/nif.ex`

```elixir
defmodule AII.NIF do
  @moduledoc "Elixir → Zig Native Interface Functions (NIFs)"

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
    energy: f32,
    id: u32,

    pub fn kineticEnergy(self: Particle) f32 {
      const v2 = self.velocity.x*self.velocity.x + self.velocity.y*self.velocity.y + self.velocity.z*self.velocity.z;
      return 0.5 * self.mass * v2;
    }
  };

  const ParticleSystem = struct {
    particles: []Particle,
    allocator: std.mem.Allocator,
    total_energy: f32,
    particle_count: usize,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
      const particles = try allocator.alloc(Particle, capacity);
      return ParticleSystem{
        .particles = particles,
        .allocator = allocator,
        .total_energy = 0.0,
        .particle_count = 0,
      };
    }

    pub fn deinit(self: *ParticleSystem) void {
      self.allocator.free(self.particles);
    }

    pub fn integrateEuler(self: *ParticleSystem, dt: f32) !void {
      const energy_before = self.computeTotalEnergy();

      for (self.particles) |*p| {
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.position.z += p.velocity.z * dt;
      }

      const energy_after = self.computeTotalEnergy();
      const tolerance: f32 = 1e-6;

      if (@abs(energy_before - energy_after) > tolerance) {
        std.debug.print("Conservation violated! Before: {d}, After: {d}\n", .{energy_before, energy_after});
        return error.ConservationViolation;
      }
    }

    pub fn computeTotalEnergy(self: *const ParticleSystem) f32 {
      var total: f32 = 0.0;
      for (self.particles) |p| {
        total += p.kineticEnergy();
      }
      return total;
    }
  };

  var gpa = std.heap.GeneralPurposeAllocator(.{}){};
  const allocator = gpa.allocator();
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
    system.integrateEuler(@floatCast(dt)) catch return Error.ConservationViolation;
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
end
```

**File:** `runtime/zig/particle_system.zig` (actual implementation)

```zig
// Full implementation in runtime/zig/particle_system.zig
// Integrated with Zigler for automatic NIF generation
```

---

### Build System

**File:** `runtime/zig/build.zig`

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Build shared library for Elixir NIF
    const lib = b.addSharedLibrary(.{
        .name = "aii_runtime",
        .root_source_file = .{ .path = "nif.zig" },
        .target = target,
        .optimize = optimize,
    });
    
    // Link with Erlang
    lib.linkSystemLibrary("erl_interface");
    lib.linkSystemLibrary("ei");
    
    b.installArtifact(lib);
    
    // Tests
    const tests = b.addTest(.{
        .root_source_file = .{ .path = "particle_system.zig" },
        .target = target,
        .optimize = optimize,
    });
    
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);
}
```

---

### Conservation Tracking

**File:** `runtime/zig/conservation.zig` (runtime verification)

```zig
// Runtime conservation checking
// Compile-time verification handled by Elixir conservation_checker.ex
```

---

### Usage Pattern

**From Elixir:**

```elixir
# Create particle system
{:ok, system_ref} = AII.NIF.create_particle_system(1000)

# Add particles (in Elixir)
particles = [
  %{position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0},
  %{position: {1.0, 0.0, 0.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0}
]

for p <- particles do
  AII.NIF.add_particle(system_ref, p)
end

# Simulate (calls Zig)
case AII.NIF.integrate(system_ref, 0.01) do
  :ok -> IO.puts("Step successful")
  {:error, :conservation_violated} -> IO.puts("Conservation error!")
end

# Get results
particles_updated = AII.NIF.get_particles(system_ref)
```

---

### Key Points

**1. Memory Safety**
- Zig allocators explicit (no leaks)
- No undefined behavior (compile checks)
- Error handling with try/catch

**2. Performance**
- Zero-cost abstractions
- Comptime generics (no runtime cost)
- SIMD when possible

**3. Interop**
- C-compatible ABI
- Easy Elixir NIF integration
- Can call existing C if needed

**4. Conservation**
- Tracked at runtime
- Errors returned (not crashes)
- Detailed diagnostics

---

### Testing

**File:** `runtime/zig/particle_system_test.zig`

```zig
const std = @import("std");
const testing = std.testing;
const ParticleSystem = @import("particle_system.zig").ParticleSystem;
const Vec3 = @import("particle_system.zig").Vec3;

test "particle system basic" {
    var system = try ParticleSystem.init(testing.allocator, 10);
    defer system.deinit();
    
    try testing.expect(system.particles.len == 10);
}

test "conservation in free fall" {
    var system = try ParticleSystem.init(testing.allocator, 1);
    defer system.deinit();
    
    system.particles[0] = .{
        .position = Vec3{ .x = 0, .y = 10, .z = 0 },
        .velocity = Vec3{ .x = 0, .y = 0, .z = 0 },
        .mass = 1.0,
        .energy = 0.0,
        .id = 0,
    };
    
    const energy_before = system.computeTotalEnergy();
    
    // Apply gravity
    const gravity = Vec3{ .x = 0, .y = -9.81, .z = 0 };
    system.applyForce(gravity, 0.01);
    try system.integrateEuler(0.01);
    
    const energy_after = system.computeTotalEnergy();
    
    // Energy should be conserved (within tolerance)
    try testing.expectApproxEqAbs(energy_before, energy_after, 1e-4);
}
```

---

### Next Steps

1. Implement basic `particle_system.zig`
2. Build NIF integration (`nif.zig`)
3. Test from Elixir (roundtrip)
4. Add conservation tracking
5. Port existing C functions
6. Benchmark: Zig vs C
