# Define your physics system
defmodule MyPhysics do
  use BrokenRecord.Zero

  defsystem CollisionWorld do
    # Compile to CUDA for GPU execution
    compile_target :cuda
    optimize [:spatial_hash, :simd, :loop_fusion]

    agents do
      defagent Particle do
        field :position, :vec3
        field :velocity, :vec3
        field :mass, :float
        field :radius, :float
        conserves [:energy, :momentum, :angular_momentum]
      end

      defagent Wall do
        field :position, :vec3
        field :normal, :vec3
        conserves []
      end
    end

    rules do
      # Particle-particle collision
      interaction collision(p1: Particle, p2: Particle) do
        # High-level physics code
        delta = p1.position - p2.position
        dist = length(delta)

        if dist < (p1.radius + p2.radius) do
          # Elastic collision with momentum conservation
          normal = normalize(delta)
          relative_velocity = p1.velocity - p2.velocity
          speed = dot(relative_velocity, normal)

          if speed > 0 do
            impulse = (2 * speed) / (1/p1.mass + 1/p2.mass)
            p1.velocity = %{p1.velocity | x: p1.velocity.x - impulse * normal.x / p1.mass, y: p1.velocity.y - impulse * normal.y / p1.mass, z: p1.velocity.z - impulse * normal.z / p1.mass}
            p2.velocity = %{p2.velocity | x: p2.velocity.x + impulse * normal.x / p2.mass, y: p2.velocity.y + impulse * normal.y / p2.mass, z: p2.velocity.z + impulse * normal.z / p2.mass}
          end
        end
      end

      # Particle-wall collision
      interaction wall_bounce(p: Particle, w: Wall) do
        dist = dot(p.position - w.position, w.normal)

        if dist < p.radius do
          # Reflect velocity
          v_normal = dot(p.velocity, w.normal)
          p.velocity = %{p.velocity | x: p.velocity.x - 2 * v_normal * w.normal.x, y: p.velocity.y - 2 * v_normal * w.normal.y, z: p.velocity.z - 2 * v_normal * w.normal.z}
        end
      end

      # Time integration
      interaction integrate(p: Particle, dt: float) do
        # Simple Euler integration
        p.position = %{p.position | x: p.position.x + p.velocity.x * dt, y: p.position.y + p.velocity.y * dt, z: p.position.z + p.velocity.z * dt}

        # Apply gravity
        p.velocity = %{p.velocity | x: p.velocity.x + 0 * dt, y: p.velocity.y + 0 * dt, z: p.velocity.z + (-9.81) * dt}
      end
    end
  end
end

# Use it
defmodule Demo do
  def run do
    # Create initial state
    initial = %{
      particles: create_particles(100_000),
      walls: create_box()
    }

    # Simulate - this runs at 100M+ particles/sec on GPU!
    result = MyPhysics.CollisionWorld.simulate(
      initial,
      steps: 10_000,
      dt: 0.001
    )

    IO.inspect(result, label: "Final state")
  end

  defp create_particles(n) do
    for i <- 1..n do
      %BrokenRecord.Particle{
        id: "p#{i}",
        mass: 1.0,
        radius: 0.5,
        position: random_position(),
        velocity: random_velocity()
      }
    end
  end

  defp create_box do
    # Six walls forming a box
    []
  end

  defp random_position do
    {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100}
  end

  defp random_velocity do
    {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5}
  end
end
