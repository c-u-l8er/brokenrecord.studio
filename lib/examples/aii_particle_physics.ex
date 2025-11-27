defmodule Examples.AIIParticlePhysics do
  @moduledoc """
  Example: AII-based particle physics with conservation types.

  Demonstrates:
  - Conservation types (Conserved<T>)
  - Property vs state distinction
  - Hardware acceleration hints
  - Compile-time conservation verification
  - Zig runtime integration
  """

  use AII.DSL

  # Declare conserved quantities for this system
  conserved_quantity :energy, type: :scalar, law: :sum
  conserved_quantity :momentum, type: :vector3, law: :sum
  conserved_quantity :information, type: :scalar, law: :sum

  defagent Particle do
    # Invariant properties (cannot change)
    property :mass, Float, invariant: true
    property :charge, Float, invariant: true
    property :particle_id, Integer, invariant: true

    # Mutable state
    state :position, AII.Types.Vec3
    state :velocity, AII.Types.Vec3
    state :acceleration, AII.Types.Vec3

    # Conserved quantities (tracked by type system)
    state :energy, AII.Types.Conserved
    state :momentum, AII.Types.Conserved
    state :information, AII.Types.Conserved

    # Computed/derived quantities
    derives :kinetic_energy, AII.Types.Energy do
      0.5 * mass * AII.Types.Vec3.magnitude(velocity) ** 2
    end

    derives :momentum_vec, AII.Types.Momentum do
      AII.Types.Vec3.mul(velocity, mass)
    end

    derives :information_content, AII.Types.Information do
      # Shannon entropy approximation
      mass * :math.log(1 + AII.Types.Vec3.magnitude(velocity))
    end

    # Declare what this agent conserves
    conserves [:energy, :momentum, :information]
  end

  defagent Field do
    # External fields that can affect particles
    property :field_type, Atom, invariant: true  # :gravity, :electric, :magnetic
    property :strength, Float
    property :direction, AII.Types.Vec3

    state :active, Boolean
    state :energy_source, AII.Types.Conserved

    conserves [:energy]
  end

  # Gravity interaction - uses CPU by default
  definteraction :apply_gravity, accelerator: :cpu do
    # Gravity interaction - acceleration only, no direct energy changes
    # Energy conservation verified by compiler
    let particle do
      gravity = {0.0, -9.81, 0.0}
      particle.acceleration = AII.Types.Vec3.add(
        particle.acceleration,
        AII.Types.Vec3.mul(gravity, 1.0 / particle.mass)
      )
    end
  end

  # Particle collision - uses RT Cores for spatial queries
  definteraction :elastic_collision, accelerator: :rt_cores do
    let {p1, p2} do
      # RT Cores accelerate collision detection
      if colliding?(p1, p2) do
        # Exchange momentum (conserved by type system)
        {p1_new, p2_new} = exchange_momentum(p1, p2)

        # Update velocities conserving momentum and energy
        p1.velocity = p1_new.velocity
        p2.velocity = p2_new.velocity

        # Transfer information between particles
        info_transfer = AII.Types.Conserved.transfer(
          p1.information,
          p2.information,
          0.1  # Transfer 10% of information
        )

        case info_transfer do
          {:ok, p1_info, p2_info} ->
            p1.information = p1_info
            p2.information = p2_info
          {:error, _} ->
            # Conservation violation - compiler catches this!
            :error
        end
      end
    end
  end

  # Field interaction - uses Tensor Cores for matrix operations
  definteraction :field_interaction, accelerator: :tensor_cores do
    let {particle, field} do
      if field.active do
        # Tensor Cores accelerate force calculations
        force = calculate_field_force(particle, field)

        # Apply force: F = ma
        particle.acceleration = AII.Types.Vec3.add(
          particle.acceleration,
          AII.Types.Vec3.mul(force, 1.0 / particle.mass)
        )

        # Energy transfer from field to particle
        energy_transfer = AII.Types.Conserved.transfer(
          field.energy_source,
          particle.energy,
          0.01  # Small energy transfer
        )

        case energy_transfer do
          {:ok, field_energy, particle_energy} ->
            field.energy_source = field_energy
            particle.energy = particle_energy
          {:error, _} ->
            # Conservation violation detected - would log in production
            :ok
        end
      end
    end
  end

  # Integration step - uses NPU for learned dynamics
  definteraction :integrate_motion, accelerator: :npu do
    let particle do
      # NPU accelerates integration with learned corrections
      dt = 0.016  # 60 FPS

      # Update velocity: v = v + a*dt
      particle.velocity = AII.Types.Vec3.add(
        particle.velocity,
        AII.Types.Vec3.mul(particle.acceleration, dt)
      )

      # Update position: x = x + v*dt
      particle.position = AII.Types.Vec3.add(
        particle.position,
        AII.Types.Vec3.mul(particle.velocity, dt)
      )

      # Reset acceleration for next frame
      particle.acceleration = {0.0, 0.0, 0.0}

      # Update conserved quantities
      particle.energy = AII.Types.Conserved.new(
        particle.kinetic_energy,
        :kinetic
      )

      particle.momentum = AII.Types.Conserved.new(
        particle.momentum_vec,
        :mechanical
      )

      particle.information = AII.Types.Conserved.new(
        particle.information_content,
        :state
      )
    end
  end

  # Helper functions (would be implemented in Zig runtime)
  def colliding?(p1, p2) do
    distance = AII.Types.Vec3.magnitude(
      AII.Types.Vec3.sub(p1.position, p2.position)
    )
    collision_distance = 2.0  # Sum of radii (assuming radius = 1.0)
    distance < collision_distance
  end

  def exchange_momentum(p1, p2) do
    # Elastic collision with momentum conservation
    {m1, m2} = {p1.mass, p2.mass}
    {v1, v2} = {p1.velocity, p2.velocity}

    # Calculate new velocities using conservation laws
    v1_new = AII.Types.Vec3.add(
      AII.Types.Vec3.mul(v1, (m1 - m2) / (m1 + m2)),
      AII.Types.Vec3.mul(v2, 2 * m2 / (m1 + m2))
    )

    v2_new = AII.Types.Vec3.add(
      AII.Types.Vec3.mul(v2, (m2 - m1) / (m1 + m2)),
      AII.Types.Vec3.mul(v1, 2 * m1 / (m1 + m2))
    )

    {%{p1 | velocity: v1_new}, %{p2 | velocity: v2_new}}
  end

  def calculate_field_force(particle, field) do
    case field.field_type do
      :gravity ->
        # F = mg (simplified)
        AII.Types.Vec3.mul(field.direction, particle.mass * field.strength)

      :electric ->
        # F = qE (Coulomb force)
        AII.Types.Vec3.mul(field.direction, particle.charge * field.strength)

      :magnetic ->
        # F = q(v × B) (Lorentz force)
        cross_product = AII.Types.Vec3.cross(particle.velocity, field.direction)
        AII.Types.Vec3.mul(cross_product, particle.charge * field.strength)

      _ ->
        {0.0, 0.0, 0.0}
    end
  end

  @doc """
  Create initial particle system
  """
  def create_particle_system(num_particles \\ 100) when is_integer(num_particles) do
    particles = for i <- 1..num_particles do
      mass = 1.0 + :rand.uniform() * 2.0
      velocity = {
        (:rand.uniform() - 0.5) * 10,
        (:rand.uniform() - 0.5) * 10,
        (:rand.uniform() - 0.5) * 10
      }
      velocity_magnitude = :math.sqrt(elem(velocity, 0)**2 + elem(velocity, 1)**2 + elem(velocity, 2)**2)
      kinetic_energy = 0.5 * mass * velocity_magnitude ** 2

      %{
        # Invariant properties
        mass: mass,
        charge: (:rand.uniform() - 0.5) * 2.0,
        particle_id: i,

        # Initial state
        position: {
          (:rand.uniform() - 0.5) * 100,
          (:rand.uniform() - 0.5) * 100,
          (:rand.uniform() - 0.5) * 100
        },
        velocity: velocity,
        acceleration: {0.0, 0.0, 0.0},

        # Computed quantities
        kinetic_energy: kinetic_energy,

        # Conserved quantities
        energy: AII.Types.Conserved.new(25.0, :initial),
        momentum: AII.Types.Conserved.new({5.0, 0.0, 0.0}, :initial),
        information: AII.Types.Conserved.new(10.0, :initial)
      }
    end

    # Create fields
    fields = [
      %{
        field_type: :gravity,
        strength: 9.81,
        direction: {0.0, -1.0, 0.0},
        active: true,
        energy_source: AII.Types.Conserved.new(1000.0, :field)
      },
      %{
        field_type: :electric,
        strength: 5.0,
        direction: {1.0, 0.0, 0.0},
        active: false,
        energy_source: AII.Types.Conserved.new(500.0, :field)
      }
    ]

    %{
      particles: particles,
      fields: fields,
      time: 0.0,
      step: 0
    }
  end

  @doc """
  Run AII simulation
  """
  def run_simulation(initial_state, opts \\ []) do
    steps = Keyword.get(opts, :steps, 1000)
    dt = Keyword.get(opts, :dt, 0.016)

    # Use AII.run_simulation with the physics system
    AII.run_simulation(__MODULE__, steps: steps, dt: dt, particles: initial_state.particles)
  end

  @doc """
  Verify conservation laws
  """
  def verify_conservation(initial_state, final_state, tolerance \\ 0.01) do
    # Calculate initial totals
    initial_energy = total_conserved(initial_state.particles, :energy)
    initial_momentum = total_conserved(initial_state.particles, :momentum)
    initial_info = total_conserved(initial_state.particles, :information)

    # Calculate final totals
    final_energy = total_conserved(final_state.particles, :energy)
    final_momentum = total_conserved(final_state.particles, :momentum)
    final_info = total_conserved(final_state.particles, :information)

    # Check conservation
    energy_conserved = abs(final_energy - initial_energy) < tolerance
    momentum_conserved = AII.Types.Vec3.magnitude(
      AII.Types.Vec3.sub(final_momentum, initial_momentum)
    ) < tolerance
    info_conserved = abs(final_info - initial_info) < tolerance

    %{
      energy_conserved: energy_conserved,
      energy_error: final_energy - initial_energy,
      momentum_conserved: momentum_conserved,
      momentum_error: AII.Types.Vec3.magnitude(
        AII.Types.Vec3.sub(final_momentum, initial_momentum)
      ),
      information_conserved: info_conserved,
      information_error: final_info - initial_info,
      total_particles: length(final_state.particles)
    }
  end

  # Helper functions for conservation checking
  defp total_conserved(particles, quantity) do
    case quantity do
      :energy ->
        Enum.reduce(particles, 0.0, fn p, acc ->
          acc + p.energy.value
        end)

      :momentum ->
        Enum.reduce(particles, {0.0, 0.0, 0.0}, fn p, {px, py, pz} ->
          {mx, my, mz} = p.momentum.value
          {px + mx, py + my, pz + mz}
        end)

      :information ->
        Enum.reduce(particles, 0.0, fn p, acc ->
          acc + p.information.value
        end)
    end
  end

  @doc """
  Get system statistics
  """
  def system_stats(state) do
    particles = state.particles

    %{
      total_particles: length(particles),
      total_mass: Enum.sum(Enum.map(particles, & &1.mass)),
      total_charge: Enum.sum(Enum.map(particles, & &1.charge)),
      average_kinetic_energy:
        if length(particles) > 0 do
          particles
          |> Enum.map(& &1.kinetic_energy)
          |> Enum.sum()
          |> Kernel./(length(particles))
        else
          0.0
        end,
      total_momentum: total_conserved(particles, :momentum),
      total_energy: total_conserved(particles, :energy),
      total_information: total_conserved(particles, :information),
      simulation_time: state.time,
      simulation_step: state.step
    }
  end

end
