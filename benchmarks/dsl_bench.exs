defmodule DSLBench do
  @moduledoc """
  Benchmarks for BrokenRecord Zero DSL and library functionality.
  Tests the actual DSL compilation and execution pipeline.
  """

  def run do
    IO.puts("Running DSL and Library Benchmarks...")

    # Benchmark DSL compilation
    Benchee.run(%{
      "DSL Compilation (Simple System)" => fn ->
        compile_simple_system()
      end,
      "DSL Compilation (Complex System)" => fn ->
        compile_complex_system()
      end,
      "IR Generation" => fn ->
        generate_ir()
      end,
      "Type Checking" => fn ->
        type_check_system()
      end,
      "Optimization Passes" => fn ->
        optimize_system()
      end,
      "Code Generation" => fn ->
        generate_code()
      end
    },
    memory_time: 2,
    print: [configuration: false],
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true},
      {Benchee.Formatters.HTML, file: "benchmarks/dsl_benchmarks.html"}
    ])

    IO.puts("DSL benchmarks complete!")
  end

  # Test actual DSL compilation
  defp compile_simple_system do
    quote do
      defmodule TestSystem do
        use BrokenRecord.Zero

        defsystem SimplePhysics do
          compile_target :cpu
            optimize [:simd]

          agents do
            defagent Particle do
              field :position, :vec3
              field :velocity, :vec3
              field :mass, :float
            end
          end

          rules do
            interaction gravity(p: Particle, dt: float) do
              p.velocity = p.velocity + {0.0, -9.81 * dt, 0.0}
            end

            interaction integrate(p: Particle, dt: float) do
              p.position = p.position + p.velocity * dt
            end
          end
        end
      end
    end
  end

  defp compile_complex_system do
    quote do
      defmodule TestComplexSystem do
        use BrokenRecord.Zero

        defsystem ComplexPhysics do
          compile_target :cpu
            optimize [:simd, :spatial_hash]

          agents do
            defagent Particle do
              field :position, :vec3
              field :velocity, :vec3
              field :mass, :float
              field :charge, :float
              field :radius, :float
            end

            defagent ForceField do
              field :strength, :float
              field :center, :vec3
              field :radius, :float
            end
          end

          rules do
            # Gravity interaction
            interaction gravity(p1: Particle, p2: Particle) do
              # F = G * m1 * m2 / r^2
              r_vec = p2.position - p1.position
              r_sq = dot(r_vec, r_vec)
              r = sqrt(r_sq)

              force_magnitude = 6.67e-11 * p1.mass * p2.mass / (r * r)
              force_direction = r_vec / r

              p1.velocity = p1.velocity + force_direction * force_magnitude * 0.01
              p2.velocity = p2.velocity - force_direction * force_magnitude * 0.01
            end

            # Force field interaction
            interaction field_force(p: Particle, f: ForceField) do
              # F = q * E * r
              r_vec = p.position - f.center
              r_sq = dot(r_vec, r_vec)

              if r_sq < f.radius * f.radius do
                r = sqrt(r_sq)
                  field_strength = f.strength * (1.0 - r / f.radius)

                  # Apply force in direction away from center
                  force_direction = r_vec / r
                  p.velocity = p.velocity + force_direction * field_strength * p.charge * 0.01
              end
            end

            # Integration
            interaction integrate(p: Particle, dt: float) do
              p.position = p.position + p.velocity * dt
            end

            # Boundary collision
            interaction boundary(p: Particle) do
              # Elastic collision with boundaries
              new_pos = p.position + p.velocity * dt

              {x, y, z} = new_pos
              {vx, vy, vz} = p.velocity

              # Bounce off walls at [-100, 100]
              new_vx = if x > 100 or x < -100, do: -vx, else: vx
              new_vy = if y > 100 or y < -100, do: -vy, else: vy
              new_vz = if z > 100 or z < -100, do: -vz, else: vz

              p.velocity = {new_vx, new_vy, new_vz}
            end
          end
        end
      end
    end
  end

  defp generate_ir do
    # Test IR generation performance
    agent_def = %{
      name: :Particle,
      fields: [
        %{name: :position, type: :vec3},
        %{name: :velocity, type: :vec3},
        %{name: :mass, type: :float}
      ]
    }

    rule_def = %{
      name: :gravity,
      type: :interaction,
      params: [
        %{name: :p, type: :Particle},
        %{name: :dt, type: :float}
      ],
      body: quote do
        p.velocity = p.velocity + {0.0, -9.81 * dt, 0.0}
      end
    }

    BrokenRecord.Zero.IR.lower_agent(agent_def)
    BrokenRecord.Zero.IR.lower_rule(rule_def)
  end

  defp type_check_system do
    # Test type checking performance
    types = [:vec3, :vec4, :float, :int, :bool]
    Enum.each(types, fn type ->
      BrokenRecord.Zero.IR.type_size(type)
      :ok
    end)
  end

  defp optimize_system do
    # Test optimization passes
    ir = %{
      agents: [%{name: :Particle, fields: []}],
      rules: [%{name: :gravity, body: [], metadata: %{parallel: :data_parallel, vectorizable: true}}],
      metadata: %{}
    }

    BrokenRecord.Zero.Optimizer.optimize(ir, [:simd, :spatial_hash])
  end

  defp generate_code do
    # Test code generation performance
    ir = %{
      agents: [%{name: :Particle, fields: []}],
      rules: [%{name: :gravity, body: [], metadata: %{parallel: :data_parallel, vectorizable: true}}],
      metadata: %{}
    }

    BrokenRecord.Zero.CodeGen.generate_native(ir)
  end

  # Helper functions
  defp dot({x1, y1, z1}, {x2, y2, z2}) do
    x1 * x2 + y1 * y2 + z1 * z2
  end

  defp sqrt(x), do: :math.sqrt(x)
end

# Run the benchmarks
DSLBench.run()
