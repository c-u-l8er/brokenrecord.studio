defmodule BrokenRecord.Zero do
  @moduledoc """
  Zero-overhead physics compiler.

  Compiles interaction nets → native code at build time.
  Runtime sees only: tight loops + flat arrays + no overhead.

  Architecture:
  1. Parse DSL → Interaction Net IR
  2. Analyze conservation laws
  3. Optimize topology
  4. Generate native kernels
  5. Compile to machine code
  """

  defmacro __using__(_opts) do
    quote do
      import BrokenRecord.Zero.DSL
      Module.register_attribute(__MODULE__, :agents, accumulate: true)
      Module.register_attribute(__MODULE__, :rules, accumulate: true)
      Module.register_attribute(__MODULE__, :compile_opts, accumulate: false)

      @before_compile BrokenRecord.Zero
    end
  end

  defmacro __before_compile__(env) do
    agents = Module.get_attribute(env.module, :agents)
    rules = Module.get_attribute(env.module, :rules)
    opts = Module.get_attribute(env.module, :compile_opts) || []

    # COMPILE-TIME MAGIC HAPPENS HERE
    compiled = BrokenRecord.Zero.Compiler.compile(agents, rules, opts, env.module)

    quote do
      # Runtime module has only the compiled artifacts
      @compiled_system unquote(Macro.escape(compiled))

      def __system__, do: @compiled_system

      # Fast path: pre-compiled native code
      def simulate(initial_state, opts \\ []) do
        BrokenRecord.Zero.Runtime.execute(@compiled_system, initial_state, opts)
      end

      # Expose generated native functions
      unquote(compiled.native_module)
    end
  end
end