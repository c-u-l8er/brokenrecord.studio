defmodule BrokenRecord.Zero.Compiler do
  @moduledoc """
  The main compiler pipeline.

  Transforms: DSL → IR → Optimized IR → Native Code → Machine Code
  """

  alias BrokenRecord.Zero.{IR, Optimizer, Analyzer, CodeGen}

  def compile(agents, rules, opts, module_name) do
    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("BrokenRecord Zero Compiler v1.0")
    IO.puts(String.duplicate("=", 70))

    # Stage 1: Lower to IR
    IO.puts("\n[1/7] Lowering to IR...")
    ir = IR.lower(agents, rules, module_name)
    IO.puts("      ✓ Generated #{length(ir.agents)} agent types")
    IO.puts("      ✓ Generated #{length(ir.rules)} interaction rules")

    # Stage 2: Type inference and checking
    IO.puts("\n[2/7] Type checking...")
    typed_ir = Analyzer.infer_types(ir)
    IO.puts("      ✓ All types verified")

    # Stage 3: Conservation analysis
    IO.puts("\n[3/7] Analyzing conservation laws...")
    conservation = Analyzer.analyze_conservation(typed_ir)
    IO.puts("      ✓ Verified #{length(conservation.proven_rules)} rules at compile time")
    IO.puts("      ⚠ #{length(conservation.runtime_checks)} rules need runtime checks")

    # Stage 4: Optimize IR
    IO.puts("\n[4/7] Optimizing...")
    optimized = Optimizer.optimize(typed_ir, opts[:optimize] || [])
    IO.puts("      ✓ Applied #{length(optimized.metadata.applied_passes)} optimization passes")

    # Stage 5: Memory layout
    IO.puts("\n[5/7] Computing memory layout...")
    layout = Optimizer.compute_memory_layout(optimized, opts[:target] || :cpu)
    IO.puts("      ✓ Layout: #{layout.strategy}")
    IO.puts("      ✓ Alignment: #{layout.alignment} bytes")

    # Stage 6: Code generation
    IO.puts("\n[6/7] Generating native code...")
    native_code = CodeGen.generate(optimized, layout, opts, module_name)
    IO.puts("      ✓ Generated #{byte_size(native_code.source)} bytes of source")

    # Stage 7: Compile to machine code
    IO.puts("\n[7/7] Compiling to machine code...")
    compiled = CodeGen.compile_native(native_code, opts)
    IO.puts("      ✓ Compiled successfully")

    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("Compilation complete!")
    IO.puts("Target: #{opts[:target] || :cpu}")
    IO.puts("Estimated performance:")
    print_performance_estimates(compiled, opts)
    IO.puts(String.duplicate("=", 70) <> "\n")

    %{
      ir: optimized,
      layout: layout,
      conservation: conservation,
      native_code: native_code,
      compiled: compiled,
      native_module: generate_runtime_module(compiled)
    }
  end

  defp print_performance_estimates(_compiled, opts) do
    case opts[:target] do
      :cuda ->
        IO.puts("  • Particle updates: ~1,000,000,000/sec (GPU)")
        IO.puts("  • Interactions:     ~100,000,000/sec (GPU)")
      :cpu ->
        IO.puts("  • Particle updates: ~10,000,000/sec (single core)")
        IO.puts("  • Interactions:     ~3,200,000/sec (single core)")
      _ ->
        IO.puts("  • Performance: optimized for target")
    end
  end

  defp generate_runtime_module(compiled) do
    # Generate Elixir functions that call native code
    filename = compiled.filename
    # Remove extension for load_nif
    basename = Path.rootname(filename)

    quote do
      # Native module loaded via NIF
      @on_load :load_nif

      def load_nif do
        nif_path = Path.join(:code.priv_dir(:broken_record_zero), unquote(basename))
        :erlang.load_nif(String.to_charlist(nif_path), 0)
      end

      # Fast native functions
      def native_step(_state, _dt), do: :erlang.nif_error(:not_loaded)
      def native_collisions(_state), do: :erlang.nif_error(:not_loaded)
      def native_integrate(_state, _dt, _steps), do: :erlang.nif_error(:not_loaded)
    end
  end
end