const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build shared library for Elixir NIF
    const nif_module = b.createModule(.{
        .root_source_file = b.path("nif.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "aii_runtime",
        .root_module = nif_module,
        .linkage = .dynamic,
    });

    // Link with Erlang static libraries
    lib.addObjectFile(.{ .cwd_relative = "/usr/lib/erlang/lib/erl_interface-5.5.2/lib/libei.a" });

    b.installArtifact(lib);

    // Tests
    const test_module = b.createModule(.{
        .root_source_file = b.path("particle_system.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);
}
