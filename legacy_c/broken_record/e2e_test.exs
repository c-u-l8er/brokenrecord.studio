defmodule BrokenRecord.E2ETest.Physics do
  use BrokenRecord.Zero

  defsystem System do
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
      interaction integrate(p: Particle, dt: float) do
        p.position = p.position + p.velocity * dt
      end

      interaction integrate_no_gravity(p: Particle, dt: float) do
        p.position = p.position + p.velocity * dt
      end
    end
  end
end

defmodule BrokenRecord.E2ETest do
  use ExUnit.Case

  alias BrokenRecord.E2ETest.Physics.System, as: TestPhysics

  test "compiles and runs a simple simulation" do
    # Create initial state
    particles = [
      %{
        id: "p1",
        position: {0.0, 0.0, 0.0},
        velocity: {1.0, 0.0, 0.0},
        mass: 1.0
      }
    ]

    initial_state = %{particles: particles}

    # Run simulation without gravity for accurate position testing
    final_state = TestPhysics.simulate(initial_state, steps: 10, dt: 0.1, rules: [:integrate_no_gravity])

    # Check results
    # After 10 steps of 0.1s with velocity 1.0, position should be 1.0
    p1 = hd(final_state.particles)
    {x, y, z} = p1.position

    assert_in_delta x, 1.0, 0.0001
    assert_in_delta y, 0.0, 0.0001
    assert_in_delta z, 0.0, 0.0001
  end
end
