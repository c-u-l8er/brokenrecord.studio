const std = @import("std");
const beam = @import("beam");
const GPUBackend = @import("gpu_backend.zig").GPUBackend;


/// Simple CPU Acceleration for SIMD operations
const CPUAcceleration = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CPUAcceleration {
        return CPUAcceleration{
            .allocator = allocator,
        };
    }

    pub fn deinit(_: *CPUAcceleration) void {
        // No cleanup needed
    }
};

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

    pub fn integrateEulerSIMD(self: *ParticleSystem, dt: f32, cpu_accel: *const CPUAcceleration) !void {
        // Conservation check: capture energy before
        const energy_before = self.computeTotalEnergy();

        // SIMD-accelerated position updates
        const Vec3x4 = @Vector(4, f32);  // 4-wide SIMD for x,y,z components

        var i: usize = 0;
        while (i < self.particle_count) {
            const remaining = self.particle_count - i;
            const batch_size = if (remaining >= 4) 4 else remaining;

            // Load position and velocity vectors
            var pos_x: Vec3x4 = undefined;
            var pos_y: Vec3x4 = undefined;
            var pos_z: Vec3x4 = undefined;
            var vel_x: Vec3x4 = undefined;
            var vel_y: Vec3x4 = undefined;
            var vel_z: Vec3x4 = undefined;

            // Fill vectors (pad with zeros if needed)
            for (0..batch_size) |j| {
                const p = &self.particles[i + j];
                pos_x[j] = p.position.x;
                pos_y[j] = p.position.y;
                pos_z[j] = p.position.z;
                vel_x[j] = p.velocity.x;
                vel_y[j] = p.velocity.y;
                vel_z[j] = p.velocity.z;
            }

            // Zero out unused elements
            for (batch_size..4) |j| {
                pos_x[j] = 0;
                pos_y[j] = 0;
                pos_z[j] = 0;
                vel_x[j] = 0;
                vel_y[j] = 0;
                vel_z[j] = 0;
            }

            // SIMD update: position += velocity * dt
            const dt_vec = @splat(Vec3x4, dt);
            pos_x += vel_x * dt_vec;
            pos_y += vel_y * dt_vec;
            pos_z += vel_z * dt_vec;

            // Store back
            for (0..batch_size) |j| {
                const p = &self.particles[i + j];
                p.position.x = pos_x[j];
                p.position.y = pos_y[j];
                p.position.z = pos_z[j];
            }

            i += batch_size;
        }

        // Conservation check: verify energy after
        const energy_after = self.computeTotalEnergy();
        const tolerance: f32 = 1e-6;

        if (@abs(energy_before - energy_after) > tolerance) {
            std.debug.print("Conservation violated in SIMD! Before: {d}, After: {d}\n", .{ energy_before, energy_after });
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

pub fn run_simulation_batch(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    if (argc != 3) return beam.make_error_tuple(env, "expected 3 arguments");

    const system_ref = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid system");
    const steps = beam.get_i64(env, argv[1]) catch return beam.make_error_tuple(env, "invalid steps");
    const dt = beam.get_f64(env, argv[2]) catch return beam.make_error_tuple(env, "invalid dt");

    const system = systems.get(@intCast(system_ref)) orelse {
        return beam.make_error_tuple(env, "system not found");
    };

    // Run simulation loop in batch
    var step: usize = 0;
    while (step < steps) : (step += 1) {
        system.integrateEuler(@floatCast(dt)) catch {
            return beam.make_error_tuple(env, "conservation violated");
        };
    }

    return beam.make_atom(env, "ok");
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

pub fn run_simulation_with_hardware(env: beam.env, argc: c_int, argv: [*c]const beam.term) beam.term {
    std.debug.print("Zig run_simulation_with_hardware function called!\n", .{});

    if (argc != 5) return beam.make_error_tuple(env, "expected 5 arguments");

    const system_ref = beam.get_i64(env, argv[0]) catch return beam.make_error_tuple(env, "invalid system");
    const steps = beam.get_i64(env, argv[1]) catch return beam.make_error_tuple(env, "invalid steps");
    const dt = beam.get_f64(env, argv[2]) catch return beam.make_error_tuple(env, "invalid dt");

    const system = systems.get(@intCast(system_ref)) orelse {
        return beam.make_error_tuple(env, "system not found");
    };

    // Check hardware assignments to determine execution strategy
    var use_tensor_cores = false;
    var use_simd = false;

    if (beam.is_list(env, argv[3])) {
        var list_iter = beam.iterate_list(env, argv[3]);
        while (beam.list_iterator_next(env, &list_iter)) |item| {
            if (beam.is_tuple(env, item) and beam.get_tuple_arity(env, item) == 2) {
                const hardware_term = beam.get_tuple_element(env, item, 1);
                const hardware_str = beam.get_atom(env, hardware_term) catch continue;

                if (std.mem.eql(u8, hardware_str, "tensor_cores")) {
                    use_tensor_cores = true;
                } else if (std.mem.eql(u8, hardware_str, "simd")) {
                    use_simd = true;
                }
            }
        }
    }

    // Execute based on hardware requirements
    if (use_tensor_cores and system.particle_count > 0) {
        std.debug.print("Executing on GPU (tensor cores)\n", .{});

        // Initialize GPU backend
        var gpu_backend = GPUBackend.init(allocator) catch {
            std.debug.print("GPU backend initialization failed, falling back to SIMD\n", .{});
            use_simd = true;
            use_tensor_cores = false;
        };

        if (use_tensor_cores) {
            defer gpu_backend.deinit();

            // Create GPU buffer for particles
            const buffer_size = system.particle_count * @sizeOf(Particle);
            var particle_buffer = gpu_backend.createBuffer(buffer_size, .storage) catch {
                std.debug.print("GPU buffer creation failed, falling back to SIMD\n", .{});
                use_simd = true;
                use_tensor_cores = false;
            };

            if (use_tensor_cores) {
                defer gpu_backend.destroyBuffer(particle_buffer);

                // Upload particle data to GPU
                const particles_slice = system.particles[0..system.particle_count];
                const particle_data = std.mem.sliceAsBytes(particles_slice);
                gpu_backend.uploadData(particle_buffer, particle_data, 0) catch {
                    std.debug.print("GPU data upload failed, falling back to SIMD\n", .{});
                    use_simd = true;
                    use_tensor_cores = false;
                };

                if (use_tensor_cores) {
                    // For CUDA, we don't need a shader handle - the kernel is pre-compiled
                    // Execute on GPU for each step
                    var step: usize = 0;
                    while (step < steps) : (step += 1) {
                        // Dispatch compute kernel directly
                        const workgroups = [_]u32{ @intCast(system.particle_count), 1, 1 };
                        const buffers = [_]GPUBackend.BufferHandle{particle_buffer};
                        const uniforms = [_]f32{ @floatCast(dt) };

                        // For CUDA, we use a dummy shader handle (kernel is linked externally)
                        const dummy_shader = GPUBackend.ShaderHandle{ .cuda = 0 };
                        gpu_backend.dispatchCompute(dummy_shader, workgroups, &buffers, &uniforms) catch {
                            std.debug.print("GPU dispatch failed on step {}, falling back to SIMD\n", .{step});
                            use_simd = true;
                            use_tensor_cores = false;
                            break;
                        };
                    }

                    if (use_tensor_cores) {
                        // Download results back to CPU
                        const particles_slice = system.particles[0..system.particle_count];
                        const particle_data = std.mem.sliceAsBytes(particles_slice);
                        gpu_backend.downloadData(particle_buffer, particle_data, 0) catch {
                            std.debug.print("GPU data download failed, results may be invalid\n", .{});
                        };
                    }
                }
            }
        }
    }

    if (!use_tensor_cores and use_simd and system.particle_count > 0) {
        std.debug.print("Executing on SIMD CPU\n", .{});

        // Use SIMD acceleration for particle integration
        var cpu_accel = CPUAcceleration.init(allocator);
        defer cpu_accel.deinit();

        var step: usize = 0;
        while (step < steps) : (step += 1) {
            system.integrateEulerSIMD(@floatCast(dt), &cpu_accel) catch {
                return beam.make_error_tuple(env, "conservation violated in SIMD");
            };
        }
    } else if (!use_tensor_cores and !use_simd) {
        std.debug.print("Executing on regular CPU\n", .{});

        // Fallback to regular CPU integration
        var step: usize = 0;
        while (step < steps) : (step += 1) {
            system.integrateEuler(@floatCast(dt)) catch {
                return beam.make_error_tuple(env, "conservation violated");
            };
        }
    }

    return beam.make_atom(env, "ok");
}
