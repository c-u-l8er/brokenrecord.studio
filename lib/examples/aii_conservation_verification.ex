defmodule Examples.AIIConservationVerification do
  @moduledoc """
  Example: AII conservation verification and type system enforcement.

  Demonstrates:
  - Compile-time conservation checking
  - Runtime conservation verification
  - Conservation violation detection
  - Type-level conservation guarantees
  - Conservation debugging tools
  """

  use AII.DSL

  # Declare conserved quantities with strict verification
  conserved_quantity :energy, type: :scalar, law: :sum, tolerance: 0.000001
  conserved_quantity :momentum, type: :vector3, law: :sum, tolerance: 0.000001
  conserved_quantity :charge, type: :scalar, law: :sum, tolerance: 0.000000001
  conserved_quantity :information, type: :scalar, law: :sum, tolerance: 0.000001
  conserved_quantity :mass, type: :scalar, law: :sum, tolerance: 0.000000001

  def __agents__ do
    [%{conserves: []}]
  end

  def __interactions__ do
    []
  end

  defagent Particle do
    property :mass, Float, invariant: true
    property :charge, Float, invariant: true
    property :particle_id, Integer, invariant: true

    state :position, AII.Types.Vec3
    state :velocity, AII.Types.Vec3
    state :acceleration, AII.Types.Vec3

    # Strictly conserved quantities
    state :energy, AII.Types.Conserved
    state :momentum, AII.Types.Conserved
    state :charge_conserved, AII.Types.Conserved
    state :information, AII.Types.Conserved
    state :mass_conserved, AII.Types.Conserved

    # Derived quantities for verification
    derives :kinetic_energy, AII.Types.Energy do
      0.5 * mass * AII.Types.AII.Types.Vec3.magnitude(velocity) ** 2
    end

    derives :potential_energy, AII.Types.Energy do
      # Gravitational potential energy (assuming Earth gravity)
      mass * 9.81 * elem(position, 1)
    end

    derives :total_energy, AII.Types.Energy do
      kinetic_energy + potential_energy
    end

    derives :momentum_vec, Momentum do
      AII.Types.Vec3.mul(velocity, mass)
    end

    derives :information_entropy, Information do
      # Shannon entropy based on velocity distribution
      v_mag = AII.Types.Vec3.magnitude(velocity)
      if v_mag > 0.001 do
        mass * :math.log(v_mag + 1)
      else
        0.0
      end
    end

    # Declare strict conservation requirements
    conserves [:energy, :momentum, :charge, :information, :mass]
  end

  defagent ConservationTracker do
    property :tracking_enabled, Boolean, invariant: true
    property :violation_threshold, Float, invariant: true

    state :total_energy, AII.Types.Conserved
    state :total_momentum, AII.Types.Conserved
    state :total_charge, AII.Types.Conserved
    state :total_information, AII.Types.Conserved
    state :total_mass, AII.Types.Conserved

    state :violations, List
    state :conservation_history, List

    conserves [:energy, :momentum, :charge, :information, :mass]
  end

  # Perfectly elastic collision - must conserve everything
  definteraction :perfect_elastic_collision do
    let {p1, p2, tracker} do
      if colliding?(p1, p2) do
        # Record pre-collision totals
        energy_before = p1.energy.value + p2.energy.value
        momentum_before = AII.Types.Vec3.add(p1.momentum.value, p2.momentum.value)
        charge_before = p1.charge_conserved.value + p2.charge_conserved.value
        info_before = p1.information.value + p2.information.value
        mass_before = p1.mass_conserved.value + p2.mass_conserved.value

        # Perform elastic collision
        {v1_new, v2_new} = calculate_elastic_collision(p1, p2)

        # Update velocities
        p1.velocity = v1_new
        p2.velocity = v2_new

        # Update conserved quantities
        p1.energy = AII.Types.Conserved.new(p1.kinetic_energy, :collision)
        p2.energy = AII.Types.Conserved.new(p2.kinetic_energy, :collision)

        p1.momentum = AII.Types.Conserved.new(p1.momentum_vec, :collision)
        p2.momentum = AII.Types.Conserved.new(p2.momentum_vec, :collision)

        # Information exchange (entropy increases slightly)
        info_exchange = 0.01 * abs(p1.mass - p2.mass)
        p1.information = AII.Types.Conserved.new(
          p1.information_entropy + info_exchange,
          :collision
        )
        p2.information = AII.Types.Conserved.new(
          p2.information_entropy + info_exchange,
          :collision
        )

        # Verify conservation
        energy_after = p1.energy.value + p2.energy.value
        momentum_after = AII.Types.Vec3.add(p1.momentum.value, p2.momentum.value)
        charge_after = p1.charge_conserved.value + p2.charge_conserved.value
        info_after = p1.information.value + p2.information.value
        mass_after = p1.mass_conserved.value + p2.mass_conserved.value

        # Check for violations
        violations = []
        |> check_conservation(:energy, energy_before, energy_after, tracker.violation_threshold)
        |> check_conservation(:momentum, momentum_before, momentum_after, tracker.violation_threshold)
        |> check_conservation(:charge, charge_before, charge_after, tracker.violation_threshold)
        |> check_conservation(:information, info_before, info_after, tracker.violation_threshold)
        |> check_conservation(:mass, mass_before, mass_after, tracker.violation_threshold)

        if length(violations) > 0 do
          # Record violations
          violation_record = %{
            timestamp: tracker.time || 0.0,
            interaction: :perfect_elastic_collision,
            particles: [p1.particle_id, p2.particle_id],
            violations: violations
          }

          tracker.violations = [violation_record | tracker.violations]
        end
      end
    end
  end

  # Inelastic collision - intentionally violates energy conservation
  definteraction :inelastic_collision do
    let {p1, p2, tracker} do
      if colliding?(p1, p2) do
        # Record pre-collision state
        energy_before = p1.energy.value + p2.energy.value

        # Inelastic collision - some energy lost as heat
        {v1_new, v2_new} = calculate_inelastic_collision(p1, p2, 0.8)  # 80% retention

        p1.velocity = v1_new
        p2.velocity = v2_new

        # Update energies (should be less than before)
        p1.energy = AII.Types.Conserved.new(p1.kinetic_energy, :inelastic_collision)
        p2.energy = AII.Types.Conserved.new(p2.kinetic_energy, :inelastic_collision)

        p1.momentum = AII.Types.Conserved.new(p1.momentum_vec, :inelastic_collision)
        p2.momentum = AII.Types.Conserved.new(p2.momentum_vec, :inelastic_collision)

        # Check energy conservation violation
        energy_after = p1.energy.value + p2.energy.value
        energy_loss = energy_before - energy_after

        if energy_loss > tracker.violation_threshold do
          # Record energy conservation violation
          violation_record = %{
            timestamp: tracker.time || 0.0,
            interaction: :inelastic_collision,
            particles: [p1.particle_id, p2.particle_id],
            violations: [
              %{
                quantity: :energy,
                expected: energy_before,
                actual: energy_after,
                error: energy_loss,
                percentage: energy_loss / energy_before * 100
              }
            ]
          }

          tracker.violations = [violation_record | tracker.violations]
        end
      end
    end
  end

  # Information creation violation - creates information from nothing
  definteraction :information_creation_violation do
    let {particle, tracker} do
      # This interaction violates information conservation
      # by creating new information

      info_before = particle.information.value

      # Artificially increase information (violation!)
      new_info = info_before * 1.1  # 10% information creation
      particle.information = AII.Types.Conserved.new(new_info, :violation)

      # Record violation
      info_created = new_info - info_before

      if info_created > tracker.violation_threshold do
        violation_record = %{
          timestamp: tracker.time || 0.0,
          interaction: :information_creation_violation,
          particles: [particle.particle_id],
          violations: [
            %{
              quantity: :information,
              expected: info_before,
              actual: new_info,
              error: info_created,
              percentage: info_created / info_before * 100,
              type: :creation_from_nothing
            }
          ]
        }

        tracker.violations = [violation_record | tracker.violations]
      end
    end
  end

  # Conservation restoration - fixes violations
  definteraction :conservation_restoration do
    let {particles, tracker} do
      # Calculate current totals
      current_energy = Enum.sum(Enum.map(particles, & &1.energy.value))
      current_momentum = Enum.reduce(particles, {0.0, 0.0, 0.0}, fn p, {px, py, pz} ->
        {mx, my, mz} = p.momentum.value
        {px + mx, py + my, pz + mz}
      end)
      current_charge = Enum.sum(Enum.map(particles, & &1.charge_conserved.value))
      current_info = Enum.sum(Enum.map(particles, & &1.information.value))
      current_mass = Enum.sum(Enum.map(particles, & &1.mass_conserved.value))

      # Get expected totals from tracker
      expected_energy = tracker.total_energy.value
      expected_momentum = tracker.total_momentum.value
      expected_charge = tracker.total_charge.value
      expected_info = tracker.total_information.value
      expected_mass = tracker.total_mass.value

      # Restore conservation by adjusting particles
      particles = restore_quantity(particles, :energy, current_energy, expected_energy)
      particles = restore_quantity(particles, :momentum, current_momentum, expected_momentum)
      particles = restore_quantity(particles, :charge, current_charge, expected_charge)
      particles = restore_quantity(particles, :information, current_info, expected_info)
      particles = restore_quantity(particles, :mass, current_mass, expected_mass)

      # Record restoration
      restoration_record = %{
        timestamp: tracker.time || 0.0,
        interaction: :conservation_restoration,
        particles: Enum.map(particles, & &1.particle_id),
        restored_quantities: [
          %{quantity: :energy, from: current_energy, to: expected_energy},
          %{quantity: :momentum, from: current_momentum, to: expected_momentum},
          %{quantity: :charge, from: current_charge, to: expected_charge},
          %{quantity: :information, from: current_info, to: expected_info},
          %{quantity: :mass, from: current_mass, to: expected_mass}
        ]
      }

      tracker.conservation_history = [restoration_record | tracker.conservation_history]
    end
  end

  # Helper functions
  def colliding?(p1, p2) do
    distance = AII.Types.Vec3.magnitude(AII.Types.Vec3.sub(p1.position, p2.position))
    collision_distance = 2.0  # Assuming unit radius
    distance < collision_distance
  end

  def calculate_elastic_collision(p1, p2) do
    # Perfectly elastic collision formulas
    {m1, m2} = {p1.mass, p2.mass}
    {v1, v2} = {p1.velocity, p2.velocity}

    # Calculate new velocities
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

  defp calculate_inelastic_collision(p1, p2, retention_factor) do
    # Inelastic collision with energy loss
    {m1, m2} = {p1.mass, p2.mass}
    {v1, v2} = {p1.velocity, p2.velocity}

    # Calculate center of mass velocity
    v_cm = AII.Types.Vec3.mul(
      AII.Types.Vec3.add(AII.Types.Vec3.mul(v1, m1), AII.Types.Vec3.mul(v2, m2)),
      1.0 / (m1 + m2)
    )

    # New velocities (both move with center of mass, modified by retention)
    v1_new = AII.Types.Vec3.add(
      v_cm,
      AII.Types.Vec3.mul(AII.Types.Vec3.sub(v1, v_cm), retention_factor)
    )

    v2_new = AII.Types.Vec3.add(
      v_cm,
      AII.Types.Vec3.mul(AII.Types.Vec3.sub(v2, v_cm), retention_factor)
    )

    {v1_new, v2_new}
  end

  # after is a reserved keyword so use after_before instead
  def check_conservation(violations, quantity, before, after_before, threshold) do
    error = case quantity do
      :energy -> abs(after_before - before)
      :charge -> abs(after_before - before)
      :information -> abs(after_before - before)
      :mass -> abs(after_before - before)
      :momentum ->
        {bx, by, bz} = before
        {ax, ay, az} = after_before
        :math.sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz))
    end

    if error > threshold do
      [%{
        quantity: quantity,
        expected: before,
        actual: after_before,
        error: error,
        percentage: error_before_percentage(error, before)
      } | violations]
    else
      violations
    end
  end

  defp error_before_percentage(error, before) do
    case before do
      {x, y, z} when is_number(x) ->  # Vector
        magnitude = :math.sqrt(x*x + y*y + z*z)
        if magnitude > 0.001, do: error / magnitude * 100, else: 0.0
      scalar when is_number(scalar) ->  # Scalar
        if scalar > 0.001, do: error / scalar * 100, else: 0.0
      _ -> 0.0
    end
  end

  defp restore_quantity(particles, :energy, current, expected) do
    if abs(current - expected) > 0.000001 do
      correction = (expected - current) / length(particles)
      Enum.map(particles, fn p ->
        new_energy = p.energy.value + correction
        %{p | energy: AII.Types.Conserved.new(new_energy, :restoration)}
      end)
    else
      particles
    end
  end

  defp restore_quantity(particles, :momentum, current, expected) do
    {cx, cy, cz} = current
    {ex, ey, ez} = expected

    if :math.sqrt((ex-cx)*(ex-cx) + (ey-cy)*(ey-cy) + (ez-cz)*(ez-cz)) > 0.000001 do
      {dx, dy, dz} = {(ex - cx) / length(particles), (ey - cy) / length(particles), (ez - cz) / length(particles)}
      Enum.map(particles, fn p ->
        {mx, my, mz} = p.momentum.value
        new_momentum = {mx + dx, my + dy, mz + dz}
        %{p | momentum: AII.Types.Conserved.new(new_momentum, :restoration)}
      end)
    else
      particles
    end
  end

  defp restore_quantity(particles, quantity, current, expected) when quantity in [:charge, :information, :mass] do
    if abs(current - expected) > 0.000001 do
      correction = (expected - current) / length(particles)
      Enum.map(particles, fn p ->
        case quantity do
          :charge ->
            new_charge = p.charge_conserved.value + correction
            %{p | charge_conserved: AII.Types.Conserved.new(new_charge, :restoration)}
          :information ->
            new_info = p.information.value + correction
            %{p | information: AII.Types.Conserved.new(new_info, :restoration)}
          :mass ->
            new_mass = p.mass_conserved.value + correction
            %{p | mass_conserved: AII.Types.Conserved.new(new_mass, :restoration)}
        end
      end)
    else
      particles
    end
  end

  @doc """
  Create conservation verification test system
  """
  def create_verification_system(num_particles \\ 10) when is_integer(num_particles) do
    particles = for i <- 1..num_particles do
      %{
        particle_id: i,
        mass: 1.0 + :rand.uniform() * 2.0,
        charge: (:rand.uniform() - 0.5) * 2.0,

        position: {
          (:rand.uniform() - 0.5) * 50,
          (:rand.uniform() - 0.5) * 50,
          (:rand.uniform() - 0.5) * 50
        },
        velocity: {
          (:rand.uniform() - 0.5) * 10,
          (:rand.uniform() - 0.5) * 10,
          (:rand.uniform() - 0.5) * 10
        },
        acceleration: {0.0, 0.0, 0.0},

        energy: AII.Types.Conserved.new(25.0, :initial),
        momentum: AII.Types.Conserved.new({5.0, 0.0, 0.0}, :initial),
        charge_conserved: AII.Types.Conserved.new(0.0, :initial),
        information: AII.Types.Conserved.new(10.0, :initial),
        mass_conserved: AII.Types.Conserved.new(1.5, :initial)
      }
    end

    # Calculate initial totals
    total_energy = Enum.sum(Enum.map(particles, & &1.energy.value))
    total_momentum = Enum.reduce(particles, {0.0, 0.0, 0.0}, fn p, {px, py, pz} ->
      {mx, my, mz} = p.momentum.value
      {px + mx, py + my, pz + mz}
    end)
    total_charge = Enum.sum(Enum.map(particles, & &1.charge_conserved.value))
    total_info = Enum.sum(Enum.map(particles, & &1.information.value))
    total_mass = Enum.sum(Enum.map(particles, & &1.mass_conserved.value))

    tracker = %{
      tracking_enabled: true,
      violation_threshold: 0.000001,

      total_energy: AII.Types.Conserved.new(total_energy, :system_initial),
      total_momentum: AII.Types.Conserved.new(total_momentum, :system_initial),
      total_charge: AII.Types.Conserved.new(total_charge, :system_initial),
      total_information: AII.Types.Conserved.new(total_info, :system_initial),
      total_mass: AII.Types.Conserved.new(total_mass, :system_initial),

      violations: [],
      conservation_history: [],
      time: 0.0
    }

    %{
      particles: particles,
      conservation_tracker: tracker,
      time: 0.0,
      step: 0,
      violations: []
    }
  end

  @doc """
  Run conservation verification simulation
  """
  def run_verification(initial_state, opts \\ []) do
    steps = Keyword.get(opts, :steps, 100)
    test_scenarios = Keyword.get(opts, :scenarios, [:perfect_elastic_collision, :inelastic_collision, :information_creation])

    # Run different test scenarios
    {final_state, _total_steps} = Enum.reduce(test_scenarios, {initial_state, 0}, fn scenario, {state, acc_steps} ->
      scenario_steps = div(steps, length(test_scenarios))
      new_state = case scenario do
        :perfect_elastic_collision ->
          # Should have zero violations
          AIIRuntime.simulate(state, steps: scenario_steps, interactions: [:perfect_elastic_collision])

        :inelastic_collision ->
          # Should detect energy conservation violation
          AIIRuntime.simulate(state, steps: scenario_steps, interactions: [:inelastic_collision])

        :information_creation ->
          # Should detect information creation violation
          AIIRuntime.simulate(state, steps: scenario_steps, interactions: [:information_creation_violation])

        :conservation_restoration ->
          # Should restore conservation
          AIIRuntime.simulate(state, steps: scenario_steps, interactions: [:conservation_restoration])
      end
      {Map.put(new_state, :steps, acc_steps + scenario_steps), acc_steps + scenario_steps}
    end)

    final_state = Map.put(final_state, :steps, steps)

    {:ok, final_state}
  end

  @doc """
  Generate conservation verification report
  """
  def verification_report(final_state) do
    tracker = final_state[:conservation_tracker] || %{
      violations: [],
      conservation_history: [],
      total_energy: %{value: 0.0},
      total_momentum: %{value: {0.0, 0.0, 0.0}},
      total_charge: %{value: 0.0},
      total_information: %{value: 0.0},
      total_mass: %{value: 0.0}
    }

    %{
      total_violations: length(tracker.violations),
      violations_by_type: group_violations_by_type(tracker.violations),
      violations_by_quantity: group_violations_by_quantity(tracker.violations),
      max_error: find_max_error(tracker.violations),
      restoration_events: length(tracker.conservation_history),

      conservation_status: %{
        energy: check_final_conservation(final_state, :energy),
        momentum: check_final_conservation(final_state, :momentum),
        charge: check_final_conservation(final_state, :charge),
        information: check_final_conservation(final_state, :information),
        mass: check_final_conservation(final_state, :mass)
      },

      recommendations: generate_recommendations(tracker.violations)
    }
  end

  defp group_violations_by_type(violations) do
    Enum.group_by(violations, fn v -> v.interaction end)
    |> Enum.map(fn {type, vlist} -> {type, length(vlist)} end)
  end

  defp group_violations_by_quantity(violations) do
    violations
    |> Enum.flat_map(fn v -> v.violations end)
    |> Enum.group_by(fn v -> v.quantity end)
    |> Enum.map(fn {quantity, vlist} -> {quantity, length(vlist)} end)
  end

  defp find_max_error(violations) do
    violations
    |> Enum.flat_map(fn v -> v.violations end)
    |> Enum.max_by(fn v -> v.error end, fn -> 0.0 end)
  end

  defp check_final_conservation(state, quantity) do
    particles = state.particles
    tracker = state[:conservation_tracker] || %{
      total_energy: %{value: 0.0},
      total_momentum: %{value: {0.0, 0.0, 0.0}},
      total_charge: %{value: 0.0},
      total_information: %{value: 0.0},
      total_mass: %{value: 0.0}
    }

    case quantity do
      :energy ->
        current = Enum.sum(Enum.map(particles, & &1.energy.value))
        expected = tracker.total_energy.value
        %{conserved: abs(current - expected) < 0.000001, error: current - expected}

      :momentum ->
        current = Enum.reduce(particles, {0.0, 0.0, 0.0}, fn p, {px, py, pz} ->
          {mx, my, mz} = p.momentum.value
          {px + mx, py + my, pz + mz}
        end)
        expected = tracker.total_momentum.value
        {cx, cy, cz} = current
        {ex, ey, ez} = expected
        error = :math.sqrt((ex-cx)*(ex-cx) + (ey-cy)*(ey-cy) + (ez-cz)*(ez-cz))
        %{conserved: error < 0.000001, error: error}

      _ ->
        current = Enum.sum(Enum.map(particles, fn p ->
          case quantity do
            :charge -> p.charge_conserved.value
            :information -> p.information.value
            :mass -> p.mass_conserved.value
          end
        end))

        expected = case quantity do
          :charge -> tracker.total_charge.value
          :information -> tracker.total_information.value
          :mass -> tracker.total_mass.value
        end

        %{conserved: abs(current - expected) < 0.000001, error: current - expected}
    end
  end

  defp generate_recommendations(violations) do
    if length(violations) == 0 do
      ["All conservation laws satisfied. System is physically consistent."]
    else
      [
        "Detected #{length(violations)} conservation violations.",
        "Review interaction implementations for proper conservation handling.",
        "Consider using stricter type-level conservation checks.",
        "Enable runtime conservation verification in production."
      ]
    end
  end
end
