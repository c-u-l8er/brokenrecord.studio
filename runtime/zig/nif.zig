const std = @import("std");
const beam = @import("beam");

const Vec3 = struct {
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

    pub fn sub(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x - other.x, .y = self.y - other.y, .z = self.z - other.z };
    }

    pub fn cross(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    pub fn dot(self: Vec3, other: Vec3) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }
};

const Particle = struct {
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    energy: f32, // Conserved quantity
    id: u32,

    pub fn kineticEnergy(self: Particle) f32 {
        const v_mag = self.velocity.magnitude();
        return 0.5 * self.mass * v_mag * v_mag;
    }
};

const ParticleSystem = struct {
    particles: []Particle,
    allocator: std.mem.Allocator,
    total_energy: f32, // Track for conservation
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
            std.debug.print("Conservation violated! Before: {d}, After: {d}\n", .{ energy_before, energy_after });
            return error.ConservationViolation;
        }
    }

    pub fn applyForce(self: *ParticleSystem, force: Vec3, dt: f32) void {
        for (self.particles) |*p| {
            const acceleration = force.mul(1.0 / p.mass);
            p.velocity = p.velocity.add(acceleration.mul(dt));
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

// Store particle systems by reference
var systems = std.AutoHashMap(usize, *ParticleSystem).init(allocator);

pub fn create_particle_system(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    if (argc != 1) return beam.make_error_tuple(env, "expected 1 argument");

    const capacity = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid capacity");

    const system = ParticleSystem.init(allocator, @intCast(capacity)) catch {
        return beam.make_error_tuple(env, "failed to create system");
    };

    const system_ptr = allocator.create(ParticleSystem) catch {
        return beam.make_error_tuple(env, "allocation failed");
    };
    system_ptr.* = system;

    const ref = @intFromPtr(system_ptr);
    systems.put(ref, system_ptr) catch {
        return beam.make_error_tuple(env, "failed to store system");
    };

    return beam.make_i64(env, @intCast(ref));
}

pub fn integrate(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    if (argc != 2) return beam.make_error_tuple(env, "expected 2 arguments");

    const system_ref = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid system");
    const dt = beam.get_f64(env, argv[1]) catch return beam.make_error_tuple(env, "invalid dt");

    const system = systems.get(@intCast(system_ref)) orelse {
        return beam.make_error_tuple(env, "system not found");
    };

    system.integrateEuler(@floatCast(dt)) catch {
        return beam.make_error_tuple(env, "conservation violated");
    };

    return beam.make_atom(env, "ok");
}

pub fn add_particle(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    if (argc != 2) return beam.make_error_tuple(env, "expected 2 arguments");

    const _system_ref = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid system");
    _ = _system_ref;

    // Parse particle data from Elixir map
    const _particle_data = argv[1];
    _ = _particle_data;
    // This would need more implementation to parse the map into a Particle struct

    // For now, stub
    return beam.make_atom(env, "ok");
}

pub fn get_particles(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    if (argc != 1) return beam.make_error_tuple(env, "expected 1 argument");

    const _system_ref = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid system");

    const _system = systems.get(@intCast(_system_ref)) orelse {
        return beam.make_error_tuple(env, "system not found");
    };
    _ = _system;

    // Convert particles to Elixir list
    // This would need implementation to convert Particle structs to Elixir terms

    // For now, return empty list
    return beam.make_list(env, &[_]beam.term{});
}

pub fn destroy_system(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    if (argc != 1) return beam.make_error_tuple(env, "expected 1 argument");

    const system_ref = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid system");

    if (systems.fetchRemove(@intCast(system_ref))) |kv| {
        kv.value.deinit();
        allocator.destroy(kv.value);
        return beam.make_atom(env, "ok");
    } else {
        return beam.make_error_tuple(env, "system not found");
    }
}
