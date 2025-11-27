runtime/zig/particle_system.zig
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
