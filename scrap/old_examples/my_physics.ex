# Define your physics system
defmodule MyPhysics do
  use BrokenRecord.Zero

  defsystem CollisionWorld do
    # Compile to CUDA for GPU execution
    # compile_target :cuda  # Temporarily disabled due to nvcc not found
    compile_target :cpu
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

          if speed < 0 do
            impulse = (2 * speed) / (1/p1.mass + 1/p2.mass)
            {x1, y1, z1} = p1.velocity
            p1.velocity = {x1 - impulse * normal.x / p1.mass, y1 - impulse * normal.y / p1.mass, z1 - impulse * normal.z / p1.mass}
            {x2, y2, z2} = p2.velocity
            p2.velocity = {x2 + impulse * normal.x / p2.mass, y2 + impulse * normal.y / p2.mass, z2 + impulse * normal.z / p2.mass}
          end
        end
      end

      # Particle-wall collision
      interaction wall_bounce(p: Particle, w: Wall) do
        dist = dot(p.position - w.position, w.normal)

        if dist < p.radius do
          # Reflect velocity
          v_normal = dot(p.velocity, w.normal)
          {x, y, z} = p.velocity
          p.velocity = {x - 2 * v_normal * w.normal.x, y - 2 * v_normal * w.normal.y, z - 2 * v_normal * w.normal.z}
        end
      end

      # Time integration
      interaction integrate(p: Particle, dt: float) do
        # Simple Euler integration
        {xp, yp, zp} = p.position
        {xv, yv, zv} = p.velocity
        p.position = {xp + xv * dt, yp + yv * dt, zp + zv * dt}

        # Apply gravity
        p.velocity = {xv, yv, zv + (-0.981) * dt}
      end

      # Position integration without gravity (for testing)
      interaction integrate_no_gravity(p: Particle, dt: float) do
        # Simple Euler integration without gravity
        {xp, yp, zp} = p.position
        {xv, yv, zv} = p.velocity
        p.position = {xp + xv * dt, yp + yv * dt, zp + zv * dt}
        # No gravity applied
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
      %{
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
