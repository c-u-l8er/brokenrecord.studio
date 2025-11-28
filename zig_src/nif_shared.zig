const std = @import("std");
const ParticleSystem = @import("particle_system.zig").ParticleSystem;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

pub var systems = std.AutoHashMap(usize, *ParticleSystem).init(allocator);

pub fn add_system(ref: usize, system: *ParticleSystem) !void {
    try systems.put(ref, system);
}

pub fn get_system(ref: usize) ?*ParticleSystem {
    return systems.get(ref);
}

pub fn remove_system(ref: usize) bool {
    return systems.remove(ref);
}
