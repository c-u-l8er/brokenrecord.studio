defmodule Examples.GravitySimulation do
  @moduledoc """
  Example: Gravity simulation with celestial bodies.

  Demonstrates:
  - Gravitational interactions
  - Orbital mechanics
  - Conservation of energy and momentum
  - N-body simulations
  """

  @doc """
  Creates a solar system with Sun, Earth, and Moon.

  Returns a map representing the solar system state.
  """
  def solar_system do
    sun = %{
      name: "Sun",
      # kg
      mass: 1.989e30,
      position: {0.0, 0.0, 0.0},
      velocity: {0.0, 0.0, 0.0},
      # m
      radius: 6.96e8
    }

    earth = %{
      name: "Earth",
      mass: 5.972e24,
      # AU in meters
      position: {1.496e11, 0.0, 0.0},
      # m/s
      velocity: {0.0, 2.978e4, 0.0},
      radius: 6.371e6
    }

    moon = %{
      name: "Moon",
      mass: 7.342e22,
      # Earth-Moon distance
      position: {1.496e11 + 3.844e8, 0.0, 0.0},
      # m/s
      velocity: {0.0, 2.978e4 + 1.022e3, 0.0},
      radius: 1.737e6
    }

    %{
      bodies: [sun, earth, moon],
      time: 0.0,
      # Gravitational constant
      g: 6.67430e-11
    }
  end

  @doc """
  Calculates gravitational force between two bodies
  """
  def gravitational_force(body1, body2, g) do
    dx = body2.position.x - body1.position.x
    dy = body2.position.y - body1.position.y
    dz = body2.position.z - body1.position.z

    distance = :math.sqrt(dx * dx + dy * dy + dz * dz)

    if distance == 0 do
      {0.0, 0.0, 0.0}
    else
      force_magnitude = g * body1.mass * body2.mass / (distance * distance)
      force_x = force_magnitude * dx / distance
      force_y = force_magnitude * dy / distance
      force_z = force_magnitude * dz / distance

      {force_x, force_y, force_z}
    end
  end

  @doc """
  Updates positions and velocities using Euler integration
  """
  def step_simulation(system, dt) do
    updated_bodies =
      Enum.map(system.bodies, fn body ->
        # Calculate total force on this body
        total_force =
          Enum.reduce(system.bodies, {0.0, 0.0, 0.0}, fn other_body, acc ->
            if body != other_body do
              {ax, ay, az} = acc
              {fx, fy, fz} = gravitational_force(body, other_body, system.g)
              {ax + fx, ay + fy, az + fz}
            else
              acc
            end
          end)

        # Update velocity (F = ma => a = F/m)
        ax = elem(total_force, 0) / body.mass
        ay = elem(total_force, 1) / body.mass
        az = elem(total_force, 2) / body.mass

        new_velocity = {
          body.velocity.x + ax * dt,
          body.velocity.y + ay * dt,
          body.velocity.z + az * dt
        }

        # Update position
        new_position = {
          body.position.x + elem(new_velocity, 0) * dt,
          body.position.y + elem(new_velocity, 1) * dt,
          body.position.z + elem(new_velocity, 2) * dt
        }

        %{body | position: new_position, velocity: new_velocity}
      end)

    %{system | bodies: updated_bodies, time: system.time + dt}
  end

  @doc """
  Calculates total energy of the system
  """
  def total_energy(system) do
    kinetic =
      Enum.sum(
        Enum.map(system.bodies, fn b ->
          0.5 * b.mass *
            (b.velocity.x * b.velocity.x + b.velocity.y * b.velocity.y +
               b.velocity.z * b.velocity.z)
        end)
      )

    # Divide by 2 because each pair is counted twice
    potential =
      Enum.sum(
        for b1 <- system.bodies, b2 <- system.bodies, b1 != b2 do
          dx = b2.position.x - b1.position.x
          dy = b2.position.y - b1.position.y
          dz = b2.position.z - b1.position.z
          distance = :math.sqrt(dx * dx + dy * dy + dz * dz)

          if distance > 0 do
            -system.g * b1.mass * b2.mass / distance
          else
            0
          end
        end
      ) / 2

    kinetic + potential
  end
end
