
defmodule Examples.Gravity do
  @moduledoc """
  Simple gravity simulation example using AII DSL.
  Demonstrates particle-based physics with conservation laws.
  """

  use AII.DSL

  # Declare conserved quantities
  conserved_quantity :energy, type: :scalar, law: :sum
  conserved_quantity :momentum, type: :vector3, law: :sum

  defagent Particle do
    # Invariant properties
    property :mass, Float, invariant: true

    # Mutable state
    state :position, AII.Types.Vec3
    state :velocity, AII.Types.Vec3

    # Conserved quantities
    state :kinetic_energy, AII.Types.Conserved
    state :momentum, AII.Types.Conserved

    # Derived quantities
    derives :potential_energy, AII.Types.Conserved do
      # For simplicity, potential energy relative to y=0
      AII.Types.Conserved.new(-mass * 9.81 * position.y, "gravity")
    end

    derives :total_energy, AII.Types.Conserved do
      AII.Types.Conserved.new(
        kinetic_energy.value + potential_energy.value,
        "total"
      )
    end

    # This agent conserves energy and momentum
    conserves :energy, :momentum
  end

  definteraction :apply_gravity, accelerator: :cpu do
    # Simple gravity: accelerate downward
    let particle do
      # Gravity acceleration
      gravity = {0.0, -9.81, 0.0}

      # Update velocity (F = ma, a = F/m, but here we directly set acceleration)
      # In a real simulation, we'd integrate properly
      particle.velocity = AII.Types.Vec3.add(particle.velocity, gravity)
    end
  end

  definteraction :integrate_motion, accelerator: :cpu do
    let particle do
      # Euler integration: position += velocity * dt
      # For simplicity, assume dt = 1.0
      dt = 1.0
      particle.position = AII.Types.Vec3.add(
        particle.position,
        AII.Types.Vec3.mul(particle.velocity, dt)
      )

      # Update kinetic energy
      speed_squared = AII.Types.Vec3.dot(particle.velocity, particle.velocity)
      new_ke = 0.5 * particle.mass * speed_squared
      particle.kinetic_energy = AII.Types.Conserved.new(new_ke, "kinetic")

      # Update momentum
      momentum_vec = AII.Types.Vec3.mul(particle.velocity, particle.mass)
      particle.momentum = AII.Types.Conserved.new(momentum_vec, "momentum")
    end
  end

  @doc """
  Creates a simple particle system with one particle.
  """
  def create_simple_system do
    # Create a particle at rest above ground
    particle = %{
      mass: 1.0,
      position: {0.0, 10.0, 0.0},  # 10 units above ground
      velocity: {0.0, 0.0, 0.0},   # Initially at rest
      kinetic_energy: AII.Types.Conserved.new(0.0, "initial"),
      momentum: AII.Types.Conserved.new({0.0, 0.0, 0.0}, "initial")
    }

    [particle]
  end

  @doc """
  Simulates one step of gravity.
  This is a simplified version - in practice this would use the Zig runtime.
  """
  def simulate_step(particles, dt \\ 1.0) do
    particles
    |> apply_gravity(dt)
    |> integrate_motion(dt)
  end

  @doc """
  Applies gravity to all particles for the given time step.
  """
  def apply_gravity(particles, dt) do
    Enum.map(particles, fn particle ->
      # Apply gravity acceleration: velocity += acceleration * dt
      gravity = {0.0, -9.81, 0.0}
      acceleration = gravity
      velocity_change = AII.Types.Vec3.mul(acceleration, dt)
      new_velocity = AII.Types.Vec3.add(particle.velocity, velocity_change)
      %{particle | velocity: new_velocity}
    end)
  end

  @doc """
  Integrates motion for all particles using the given time step.
  """
  def integrate_motion(particles, dt) do
    Enum.map(particles, fn particle ->
      # Update position: position += velocity * dt
      velocity_vec = AII.Types.Vec3.mul(particle.velocity, dt)
      new_position = AII.Types.Vec3.add(particle.position, velocity_vec)

      # Update kinetic energy
      speed_squared = AII.Types.Vec3.dot(particle.velocity, particle.velocity)
      new_ke = 0.5 * particle.mass * speed_squared

      # Update momentum
      momentum_vec = AII.Types.Vec3.mul(particle.velocity, particle.mass)

      %{particle |
        position: new_position,
        kinetic_energy: AII.Types.Conserved.new(new_ke, "kinetic"),
        momentum: AII.Types.Conserved.new(momentum_vec, "momentum")
      }
    end)
  end

  @doc """
  Computes total energy of the system.
  """
  def total_energy(particles) do
    Enum.reduce(particles, 0.0, fn particle, acc ->
      ke = particle.kinetic_energy.value
      {_, y, _} = particle.position
      pe = -particle.mass * 9.81 * y  # Potential energy
      acc + ke + pe
    end)
  end

  @doc """
  Computes total momentum of the system.
  """
  def total_momentum(particles) do
    Enum.reduce(particles, {0.0, 0.0, 0.0}, fn particle, acc ->
      AII.Types.Vec3.add(acc, particle.momentum.value)
    end)
  end

  @doc """
  Runs a simple simulation and checks conservation.
  """
  def run_simulation(steps \\ 10) do
    particles = create_simple_system()

    # Initial state
    initial_energy = total_energy(particles)
    initial_momentum = total_momentum(particles)

    IO.puts("Initial state:")
    IO.puts("  Position: #{inspect(particles |> hd() |> Map.get(:position))}")
    IO.puts("  Velocity: #{inspect(particles |> hd() |> Map.get(:velocity))}")
    IO.puts("  Total Energy: #{initial_energy}")
    IO.puts("  Total Momentum: #{inspect(initial_momentum)}")
    IO.puts("")

    # Simulate
    final_particles = Enum.reduce(1..steps, particles, fn step, acc ->
      IO.puts("Step #{step}:")
      new_particles = simulate_step(acc, 1.0)
      energy = total_energy(new_particles)
      momentum = total_momentum(new_particles)

      pos = new_particles |> hd() |> Map.get(:position)
      vel = new_particles |> hd() |> Map.get(:velocity)

      IO.puts("  Position: #{inspect(pos)}")
      IO.puts("  Velocity: #{inspect(vel)}")
      IO.puts("  Total Energy: #{energy}")
      IO.puts("  Total Momentum: #{inspect(momentum)}")
      IO.puts("")

      new_particles
    end)

    # Final conservation check
    final_energy = total_energy(final_particles)
    final_momentum = total_momentum(final_particles)

    IO.puts("Conservation Check:")
    IO.puts("  Energy conserved: #{abs(initial_energy - final_energy) < 0.01}")
    IO.puts("  Initial: #{initial_energy}, Final: #{final_energy}")

    momentum_conserved = AII.Types.Vec3.magnitude(
      AII.Types.Vec3.sub(initial_momentum, final_momentum)
    ) < 0.01
    IO.puts("  Momentum conserved: #{momentum_conserved}")
    IO.puts("  Initial: #{inspect(initial_momentum)}, Final: #{inspect(final_momentum)}")

    :ok
  end
end
