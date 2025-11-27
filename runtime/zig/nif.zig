runtime/zig/nif.zig
const std = @import("std");
const beam = @import("beam.zig");
const ParticleSystem = @import("particle_system.zig").ParticleSystem;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Store particle systems by reference
var systems = std.AutoHashMap(usize, *ParticleSystem).init(allocator);

export fn create_particle_system(env: beam.Env, argc: c_int, argv: [*c]const beam.Term) beam.Term {
    if (argc != 1) return beam.makeError(env, "expected 1 argument");

    const capacity = beam.getInt(env, argv[0]) catch return beam.makeError(env, "invalid capacity");

    const system = ParticleSystem.init(allocator, @intCast(capacity)) catch {
        return beam.makeError(env, "failed to create system");
    };

    const system_ptr = allocator.create(ParticleSystem) catch {
        return beam.makeError(env, "allocation failed");
    };
    system_ptr.* = system;

    const ref = @intFromPtr(system_ptr);
    systems.put(ref, system_ptr) catch {
        return beam.makeError(env, "failed to store system");
    };

    return beam.makeInt(env, @intCast(ref));
}

export fn integrate(env: beam.Env, argc: c_int, argv: [*c]const beam.Term) beam.Term {
    if (argc != 2) return beam.makeError(env, "expected 2 arguments");

    const system_ref = beam.getInt(env, argv[0]) catch return beam.makeError(env, "invalid system");
    const dt = beam.getFloat(env, argv[1]) catch return beam.makeError(env, "invalid dt");

    const system = systems.get(@intCast(system_ref)) orelse {
        return beam.makeError(env, "system not found");
    };

    system.integrateEuler(@floatCast(dt)) catch {
        return beam.makeError(env, "conservation violated");
    };

    return beam.makeAtom(env, "ok");
}

export fn add_particle(env: beam.Env, argc: c_int, argv: [*c]const beam.Term) beam.Term {
    if (argc != 2) return beam.makeError(env, "expected 2 arguments");

    const system_ref = beam.getInt(env, argv[0]) catch return beam.makeError(env, "invalid system");

    // Parse particle data from Elixir map
    const particle_data = argv[1];
    // This would need more implementation to parse the map into a Particle struct

    // For now, stub
    return beam.makeAtom(env, "ok");
}

export fn get_particles(env: beam.Env, argc: c_int, argv: [*c]const beam.Term) beam.Term {
    if (argc != 1) return beam.makeError(env, "expected 1 argument");

    const system_ref = beam.getInt(env, argv[0]) catch return beam.makeError(env, "invalid system");

    const system = systems.get(@intCast(system_ref)) orelse {
        return beam.makeError(env, "system not found");
    };

    // Convert particles to Elixir list
    // This would need implementation to convert Particle structs to Elixir terms

    // For now, return empty list
    return beam.makeList(env, &[_]beam.Term{});
}

export fn destroy_system(env: beam.Env, argc: c_int, argv: [*c]const beam.Term) beam.Term {
    if (argc != 1) return beam.makeError(env, "expected 1 argument");

    const system_ref = beam.getInt(env, argv[0]) catch return beam.makeError(env, "invalid system");

    if (systems.fetchRemove(@intCast(system_ref))) |kv| {
        kv.value.deinit();
        allocator.destroy(kv.value);
        return beam.makeAtom(env, "ok");
    } else {
        return beam.makeError(env, "system not found");
    }
}
