const std = @import("std");

// Basic types for Erlang/Elixir integration
pub const Env = *opaque {};
pub const Term = usize; // Simplified for stub

// Error type
pub const Error = error{
    BadArg,
    AllocFail,
    NotImplemented,
};

// Basic term constructors
pub fn makeAtom(_env: Env, atom: []const u8) Term {
    _ = _env;
    _ = atom;
    // Stub implementation
    return 0;
}

pub fn makeInt(_env: Env, value: i64) Term {
    _ = _env;
    _ = value;
    // Stub implementation
    return 0;
}

pub fn makeFloat(_env: Env, value: f64) Term {
    _ = _env;
    _ = value;
    // Stub implementation
    return 0;
}

pub fn makeError(_env: Env, message: []const u8) Term {
    _ = _env;
    _ = message;
    // Stub implementation
    return 0;
}

pub fn makeList(_env: Env, terms: []const Term) Term {
    _ = _env;
    _ = terms;
    // Stub implementation
    return 0;
}

// Basic term accessors
pub fn getInt(_env: Env, term: Term) Error!i64 {
    _ = _env;
    _ = term;
    return error.NotImplemented;
}

pub fn getFloat(_env: Env, term: Term) Error!f64 {
    _ = _env;
    _ = term;
    return error.NotImplemented;
}

// NIF function signature type
pub const NifFunc = struct {
    name: []const u8,
    arity: u32,
    func: *const fn (env: Env, argc: i32, argv: [*c]const Term) callconv(.c) Term,
    flags: u32 = 0,
};

// Module initialization (stub)
pub const NifModule = struct {
    name: []const u8,
    funcs: []const NifFunc,
    load: ?*const fn (env: Env, priv_data: [*c]?*anyopaque, load_info: Term) callconv(.c) i32 = null,
    reload: ?*const fn (env: Env, priv_data: [*c]?*anyopaque, load_info: Term) callconv(.c) i32 = null,
    upgrade: ?*const fn (env: Env, priv_data: [*c]?*anyopaque, old_priv_data: [*c]?*anyopaque, load_info: Term) callconv(.c) i32 = null,
    unload: ?*const fn (env: Env, priv_data: [*c]?*anyopaque) callconv(.c) void = null,
};
