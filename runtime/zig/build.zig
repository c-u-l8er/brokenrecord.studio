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
