runtime/zig/conservation.zig
const std = @import("std");

pub fn Conserved(comptime T: type) type {
    return struct {
        value: T,
        initial: T,
        source: []const u8,

        const Self = @This();

        pub fn init(value: T, source: []const u8) Self {
            return Self{
                .value = value,
                .initial = value,
                .source = source,
            };
        }

        pub fn set(self: *Self, new_value: T) void {
            self.value = new_value;
        }

        pub fn delta(self: Self) T {
            return self.value - self.initial;
        }

        pub fn reset(self: *Self) void {
            self.initial = self.value;
        }

        pub fn transfer(from: *Self, to: *Self, amount: T) !void {
            if (from.value < amount) {
                return error.InsufficientValue;
            }
            from.value -= amount;
            to.value += amount;
        }
    };
}

pub const ConservationTracker = struct {
    quantities: std.StringHashMap(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ConservationTracker {
        return ConservationTracker{
            .quantities = std.StringHashMap(f32).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConservationTracker) void {
        self.quantities.deinit();
    }

    pub fn capture(self: *ConservationTracker, name: []const u8, value: f32) !void {
        try self.quantities.put(name, value);
    }

    pub fn verify(self: *ConservationTracker, name: []const u8, current: f32, tolerance: f32) !void {
        const initial = self.quantities.get(name) orelse return error.QuantityNotTracked;

        if (@abs(initial - current) > tolerance) {
            std.debug.print(
                "Conservation violated: {s}\n  Initial: {d}\n  Current: {d}\n  Delta: {d}\n",
                .{name, initial, current, current - initial}
            );
            return error.ConservationViolation;
        }
    }
};

pub fn verifyConserved(before: f32, after: f32, tolerance: f32) bool {
    return @abs(before - after) < tolerance;
}

// Track total energy (assumes particles have energy field)
pub fn computeTotalEnergy(particles: []const anytype) f32 {
    var total: f32 = 0.0;
    for (particles) |p| {
        // Assume particle has energy field that is f32
        total += p.energy;
    }
    return total;
}

// Generic conservation verification
pub fn verifyConservation(
    before: f32,
    after: f32,
    tolerance: f32,
    name: []const u8
) !void {
    if (@abs(before - after) > tolerance) {
        std.debug.print(
            "Conservation violated: {s}\n  Before: {d}\n  After: {d}\n  Error: {d}\n",
            .{name, before, after, @abs(before - after)}
        );
        return error.ConservationViolation;
    }
}
