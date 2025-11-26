defmodule Examples.GravitySimulation do
  @moduledoc """
  Example: N-body gravity simulation with conservation guarantees.

  Demonstrates:
  - Gravitational force calculations
  - Energy and momentum conservation
  - Performance optimization techniques
  """

  use BrokenRecord.Zero

  defsystem NBodyGravity do
    compile_target :cpu
    optimize [:spatial_hash, :simd, :loop_fusion]

    agents do
      defagent Body do
        field :position, :vec3
        field :velocity, :vec3
        field :mass, :float
        field :radius, :float
        conserves [:energy, :momentum, :angular_momentum]
      end
    end

    rules do
      # Gravitational attraction between bodies
      interaction gravity(b1: Body, b2: Body) do
        # F = G * m1 * m2 / r^2
        # G = 6.67430e-11 (scaled for simulation)
        G = 1.0  # Simplified gravitational constant

        delta = b1.position - b2.position
        dist_sq = dot(delta, delta)
        dist = sqrt(dist_sq)

        # Avoid singularity at zero distance
        min_dist = b1.radius + b2.radius
        effective_dist = max(dist, min_dist)

        # Calculate force magnitude
        force_magnitude = G * b1.mass * b2.mass / (effective_dist * effective_dist)

        # Force direction (normalized)
        force_direction = normalize(delta)

        # Apply forces (Newton's third law)
        force = force_direction * force_magnitude
        b1.velocity = b1.velocity + force / b1.mass * dt
        b2.velocity = b2.velocity - force / b2.mass * dt
      end

      # Integration
      interaction integrate(b: Body, dt: float) do
        b.position = b.position + b.velocity * dt
      end
    end
  end

  @doc """
  Create a solar system-like configuration
  """
  def solar_system do
    bodies = [
      # Sun
      %{
        position: {0.0, 0.0, 0.0},
        velocity: {0.0, -0.03125, 0.0},
        mass: 1000.0,
        radius: 5.0
      },
      # Earth-like planet
      %{
        position: {50.0, 0.0, 0.0},
        velocity: {0.0, 4.5, 0.0},
        mass: 1.0,
        radius: 1.0
      },
      # Mars-like planet
      %{
        position: {80.0, 0.0, 0.0},
        velocity: {0.0, 3.5, 0.0},
        mass: 0.5,
        radius: 0.8
      },
      # Jupiter-like planet
      %{
        position: {150.0, 0.0, 0.0},
        velocity: {0.0, 2.5, 0.0},
        mass: 10.0,
        radius: 3.0
      }
    ]

    %{bodies: bodies}
  end

  @doc """
  Create a galaxy-like configuration with many bodies
  """
  def galaxy(num_bodies \\ 1000) when is_integer(num_bodies) do
    # Central massive body
    central = %{
      position: {0.0, 0.0, 0.0},
      velocity: {0.0, 0.0, 0.0},
      mass: 10000.0,
      radius: 10.0
    }

    # Orbiting bodies in a disk
    orbiting = for i <- 1..num_bodies do
      angle = 2 * :math.pi() * i / num_bodies
      radius = 50.0 + :rand.uniform() * 100.0

      # Circular orbit velocity
      orbital_speed = :math.sqrt(10000.0 / radius)

      %{
        position: {
          radius * :math.cos(angle),
          radius * :math.sin(angle),
          (:rand.uniform() - 0.5) * 10.0  # Small z variation
        },
        velocity: {
          -orbital_speed * :math.sin(angle),
          orbital_speed * :math.cos(angle),
          0.0
        },
        mass: 0.1 + :rand.uniform() * 0.5,
        radius: 0.5
      }
    end

    %{bodies: [central | orbiting]}
  end

  @doc """
  Create a random cluster of bodies
  """
  def cluster(num_bodies \\ 100) when is_integer(num_bodies) do
    bodies = for _i <- 1..num_bodies do
      %{
        position: {
          (:rand.uniform() - 0.5) * 100,
          (:rand.uniform() - 0.5) * 100,
          (:rand.uniform() - 0.5) * 100
        },
        velocity: {
          (:rand.uniform() - 0.5) * 2,
          (:rand.uniform() - 0.5) * 2,
          (:rand.uniform() - 0.5) * 2
        },
        mass: 0.5 + :rand.uniform() * 2.0,
        radius: 0.5 + :rand.uniform() * 1.0
      }
    end

    %{bodies: bodies}
  end

  @doc """
  Run simulation and return final state
  """
  def run_simulation(initial_state, opts) do
    steps = Keyword.get(opts, :steps, 1000)
    dt = Keyword.get(opts, :dt, 0.01)

    NBodyGravity.simulate(initial_state, steps: steps, dt: dt)
  end

  @doc """
  Calculate total energy of the system
  """
  def total_energy(state) do
    bodies = state.bodies

    # Kinetic energy: 0.5 * m * v^2
    kinetic = Enum.reduce(bodies, 0.0, fn body, acc ->
      v_sq = dot(body.velocity, body.velocity)
      acc + 0.5 * body.mass * v_sq
    end)

    # Potential energy: -G * m1 * m2 / r
    potential = Enum.reduce(bodies, 0.0, fn body1, acc1 ->
      Enum.reduce(bodies, acc1, fn body2, acc2 ->
        if body1 != body2 do
          dx = elem(body1.position, 0) - elem(body2.position, 0)
          dy = elem(body1.position, 1) - elem(body2.position, 1)
          dz = elem(body1.position, 2) - elem(body2.position, 2)
          dist_sq = dx*dx + dy*dy + dz*dz
          dist = :math.sqrt(dist_sq)
          if dist > 0.0 do
            acc2 - 1.0 * body1.mass * body2.mass / dist
          else
            acc2
          end
        else
          acc2
        end
      end)
    end) / 2.0  # Divide by 2 to avoid double counting

    kinetic + potential
  end

  @doc """
  Calculate total momentum of the system
  """
  def total_momentum(state) do
    Enum.reduce(state.bodies, {0.0, 0.0, 0.0}, fn body, {px, py, pz} ->
      {vx, vy, vz} = body.velocity
      {
        px + body.mass * vx,
        py + body.mass * vy,
        pz + body.mass * vz
      }
    end)
  end

  @doc """
  Calculate center of mass of the system
  """
  def center_of_mass(state) do
    total_mass = Enum.reduce(state.bodies, 0.0, fn body, acc -> acc + body.mass end)

    if total_mass > 0.0 do
      Enum.reduce(state.bodies, {0.0, 0.0, 0.0}, fn body, {cx, cy, cz} ->
        {x, y, z} = body.position
        {
          cx + body.mass * x / total_mass,
          cy + body.mass * y / total_mass,
          cz + body.mass * z / total_mass
        }
      end)
    else
      {0.0, 0.0, 0.0}
    end
  end

  @doc """
  Verify conservation laws
  """
  def verify_conservation(initial_state, final_state, tolerance \\ 0.01) when is_number(tolerance) do
    initial_energy = total_energy(initial_state)
    final_energy = total_energy(final_state)
    energy_error = abs(final_energy - initial_energy) / abs(initial_energy)

    initial_momentum = total_momentum(initial_state)
    final_momentum = total_momentum(final_state)
    momentum_error = momentum_distance(initial_momentum, final_momentum)

    initial_com = center_of_mass(initial_state)
    final_com = center_of_mass(final_state)
    com_error = distance(initial_com, final_com)

    %{
      energy_conserved: energy_error < tolerance,
      energy_error: energy_error,
      momentum_conserved: momentum_error < tolerance,
      momentum_error: momentum_error,
      center_of_mass_stable: com_error < tolerance,
      com_error: com_error
    }
  end

  # Helper functions
  defp dot({x1, y1, z1}, {x2, y2, z2}) do
    x1 * x2 + y1 * y2 + z1 * z2
  end

  defp distance({x1, y1, z1}, {x2, y2, z2}) do
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    :math.sqrt(dx*dx + dy*dy + dz*dz)
  end

  defp momentum_distance({px1, py1, pz1}, {px2, py2, pz2}) do
    dx = px1 - px2
    dy = py1 - py2
    dz = pz1 - pz2
    :math.sqrt(dx*dx + dy*dy + dz*dz)
  end


end
