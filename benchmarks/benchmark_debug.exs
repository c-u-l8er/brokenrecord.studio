defmodule BenchmarkDebug do
  @moduledoc """
  Comprehensive debugging benchmark suite for AII system.

  Isolates performance bottlenecks in different pipeline stages:
  - NIF performance (raw Zig execution)
  - Conservation verification
  - Hardware dispatch
  - Code generation
  - Full pipeline
  - System complexity scaling
  """

  # ============================================================================
  # Test Systems (Simple to Complex)
  # ============================================================================

  defmodule SimpleParticleSystem do
    use AII.DSL

    defagent Particle do
      state(:position, AII.Types.Vec3)
      state(:velocity, AII.Types.Vec3)
      property(:mass, Float, invariant: true)
    end

    definteraction :integrate, accelerator: :auto do
      let p do
        p.position = p.position + p.velocity * 0.01
      end
    end
  end

  defmodule ConservationSystem do
    use AII.DSL

    conserved_quantity(:energy, type: :scalar, law: :sum)

    defagent Particle do
      state(:position, AII.Types.Vec3)
      state(:velocity, AII.Types.Vec3)
      property(:mass, Float, invariant: true)
    end

    definteraction :integrate_with_conservation, accelerator: :auto do
      let p do
        p.position = p.position + p.velocity * 0.01
        conserved(:energy, 0.5 * p.mass * magnitude(p.velocity) ** 2)
      end
    end
  end

  defmodule CollisionSystem do
    use AII.DSL

    defagent Particle do
      state(:position, AII.Types.Vec3)
      state(:velocity, AII.Types.Vec3)
      property(:mass, Float, invariant: true)
    end

    definteraction :detect_collisions, accelerator: :auto do
      let {p1, p2} do
        distance = magnitude(p2.position - p1.position)

        if distance < 2.0 do
          conserved(:collisions, 1)
        end
      end
    end
  end

  # ============================================================================
  # Benchmark Data
  # ============================================================================

  def test_particles(count) do
    Enum.map(1..count, fn i ->
      %{
        position: %{x: i * 2.0, y: 0.0, z: 0.0},
        velocity: %{x: 1.0, y: 0.0, z: 0.0},
        mass: 1.0,
        energy: 0.0,
        id: i
      }
    end)
  end

  # ============================================================================
  # Individual Component Benchmarks
  # ============================================================================

  def benchmark_nif_raw do
    IO.puts("\n🔧 Benchmarking Raw NIF Performance...")

    particles = test_particles(10)

    Benchee.run(
      %{
        "NIF SIMD (10 particles, 100 steps)" => fn ->
          ref = AII.NIF.create_particle_system(20)
          Enum.each(particles, &AII.NIF.add_particle(ref, &1))
          AII.NIF.run_simulation_batch(ref, 100, 0.01)
          AII.NIF.destroy_system(ref)
        end,
        "NIF Scalar (10 particles, 100 steps)" => fn ->
          ref = AII.NIF.create_particle_system(20)
          Enum.each(particles, &AII.NIF.add_particle(ref, &1))
          AII.NIF.run_simulation_batch_scalar(ref, 100, 0.01)
          AII.NIF.destroy_system(ref)
        end
      },
      time: 1,
      memory_time: 0.5,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  def benchmark_conservation_checking do
    IO.puts("\n⚖️  Benchmarking Conservation Verification...")

    agents = SimpleParticleSystem.__agents__()
    interactions = SimpleParticleSystem.__interactions__()

    Benchee.run(
      %{
        "Simple System Conservation" => fn ->
          Enum.each(interactions, fn interaction ->
            AII.verify_conservation(interaction, agents)
          end)
        end
      },
      time: 1,
      memory_time: 0.5,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  def benchmark_hardware_dispatch do
    IO.puts("\n🎯 Benchmarking Hardware Dispatch...")

    interactions = SimpleParticleSystem.__interactions__()

    Benchee.run(
      %{
        "Simple Interaction Dispatch" => fn ->
          Enum.each(interactions, fn interaction ->
            AII.dispatch_interaction(interaction)
          end)
        end
      },
      time: 1,
      memory_time: 0.5,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  def benchmark_code_generation do
    IO.puts("\n📝 Benchmarking Code Generation...")

    interactions = SimpleParticleSystem.__interactions__()

    Benchee.run(
      %{
        "Simple Interaction Code Gen" => fn ->
          Enum.each(interactions, fn interaction ->
            AII.generate_code(interaction, :cpu)
          end)
        end
      },
      time: 1,
      memory_time: 0.5,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  def benchmark_collision_detection do
    IO.puts("\n💥 Benchmarking Collision Detection...")

    particles = test_particles(10)

    Benchee.run(
      %{
        "RT Cores Collision Detection (10 particles)" => fn ->
          AII.detect_collisions(CollisionSystem, particles, 2.0)
        end
      },
      time: 1,
      memory_time: 0.5,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  # ============================================================================
  # Pipeline Stage Benchmarks
  # ============================================================================

  def benchmark_pipeline_stages do
    IO.puts("\n🔬 Benchmarking Pipeline Stages...")

    # Small count for stage analysis
    particles = test_particles(5)

    Benchee.run(
      %{
        "Full AII Pipeline (5 particles, 1 step)" => fn ->
          AII.run_simulation(SimpleParticleSystem, steps: 1, dt: 0.01, particles: particles)
        end,
        "Conservation + Dispatch + CodeGen (cached)" => fn ->
          # This simulates the overhead without NIF execution
          agents = SimpleParticleSystem.__agents__()
          interactions = SimpleParticleSystem.__interactions__()

          # Conservation
          Enum.each(interactions, fn interaction ->
            AII.verify_conservation(interaction, agents)
          end)

          # Dispatch
          Enum.each(interactions, fn interaction ->
            AII.dispatch_interaction(interaction)
          end)

          # Code generation
          Enum.each(interactions, fn interaction ->
            {:ok, hw} = AII.dispatch_interaction(interaction)
            AII.generate_code(interaction, hw)
          end)
        end
      },
      time: 1,
      memory_time: 0.5,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  # ============================================================================
  # Scaling Benchmarks
  # ============================================================================

  def benchmark_scaling do
    IO.puts("\n📈 Benchmarking Scaling Performance...")

    # Test different particle counts
    particle_counts = [1, 5, 10, 25, 50]

    Enum.each(particle_counts, fn count ->
      IO.puts("\n--- Testing with #{count} particles ---")

      particles = test_particles(count)

      Benchee.run(
        %{
          "NIF Raw (#{count} particles, 10 steps)" => fn ->
            ref = AII.NIF.create_particle_system(count + 10)
            Enum.each(particles, &AII.NIF.add_particle(ref, &1))
            AII.NIF.run_simulation_batch(ref, 10, 0.01)
            AII.NIF.destroy_system(ref)
          end,
          "Full Pipeline (#{count} particles, 1 step)" => fn ->
            AII.run_simulation(SimpleParticleSystem, steps: 1, dt: 0.01, particles: particles)
          end
        },
        time: 0.5,
        memory_time: 0.2,
        parallel: 1,
        formatters: [
          {Benchee.Formatters.Console, extended_statistics: true}
        ]
      )
    end)
  end

  # ============================================================================
  # Memory and GC Benchmarks
  # ============================================================================

  def benchmark_memory do
    IO.puts("\n🧠 Benchmarking Memory Usage...")

    particles = test_particles(50)

    Benchee.run(
      %{
        "Memory Usage (50 particles, 100 steps)" => fn ->
          AII.run_simulation(SimpleParticleSystem, steps: 100, dt: 0.01, particles: particles)
        end
      },
      time: 1,
      memory_time: 1,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true}
      ]
    )
  end

  # ============================================================================
  # Main Benchmark Runner
  # ============================================================================

  def run_all do
    IO.puts("🔍 AII Debug Benchmark Suite")
    IO.puts("============================")

    # Start cache agents
    AII.start_cache_agents()

    # Run component benchmarks
    benchmark_nif_raw()
    benchmark_conservation_checking()
    benchmark_hardware_dispatch()
    benchmark_code_generation()
    benchmark_collision_detection()

    # Run pipeline analysis
    benchmark_pipeline_stages()

    # Run scaling tests
    benchmark_scaling()

    # Run memory analysis
    benchmark_memory()

    IO.puts("\n✅ Debug benchmarks complete!")
    IO.puts("📊 Check results above to identify bottlenecks")
  end

  # Quick test for specific issues
  def quick_test do
    IO.puts("⚡ Quick Performance Test")

    # Start cache agents
    AII.start_cache_agents()

    particles = test_particles(10)

    # Test NIF speed
    {time, _} =
      :timer.tc(fn ->
        ref = AII.NIF.create_particle_system(20)
        Enum.each(particles, &AII.NIF.add_particle(ref, &1))
        AII.NIF.run_simulation_batch(ref, 1000, 0.01)
        AII.NIF.destroy_system(ref)
      end)

    IO.puts("NIF Raw (10 particles, 1000 steps): #{time / 1000} ms")

    # Test full pipeline
    {time2, _} =
      :timer.tc(fn ->
        AII.run_simulation(SimpleParticleSystem, steps: 10, dt: 0.01, particles: particles)
      end)

    IO.puts("Full Pipeline (10 particles, 10 steps): #{time2 / 1000} ms")

    IO.puts("Ratio: #{time2 / time}x slower (overhead factor)")
  end
end

# Run the benchmarks
# BenchmarkDebug.run_all()
# Or run quick test:
BenchmarkDebug.quick_test()
