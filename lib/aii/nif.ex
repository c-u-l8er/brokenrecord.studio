defmodule AII.NIF do
  @moduledoc """
  Elixir → Zig Native Interface Functions (NIFs)
  Provides low-level access to the Zig runtime for particle systems.
  """

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:aii), 'aii_runtime')
    :erlang.load_nif(path, 0)
  end

  # Particle System Management

  @doc """
  Creates a new particle system with the given capacity.

  Returns a reference integer that can be used with other functions.
  """
  def create_particle_system(_capacity), do: :erlang.nif_error(:not_loaded)

  @doc """
  Destroys a particle system and frees its memory.
  """
  def destroy_system(_system_ref), do: :erlang.nif_error(:not_loaded)

  # Particle Operations

  @doc """
  Adds a particle to the system.

  particle_data should be a map with keys:
  - :position - {x, y, z} tuple
  - :velocity - {x, y, z} tuple
  - :mass - float
  - :energy - float
  - :id - integer
  """
  def add_particle(_system_ref, _particle_data), do: :erlang.nif_error(:not_loaded)

  @doc """
  Retrieves all particles from the system as a list of maps.
  """
  def get_particles(_system_ref), do: :erlang.nif_error(:not_loaded)

  @doc """
  Updates a specific particle in the system.
  """
  def update_particle(_system_ref, _particle_id, _particle_data), do: :erlang.nif_error(:not_loaded)

  @doc """
  Removes a particle from the system by ID.
  """
  def remove_particle(_system_ref, _particle_id), do: :erlang.nif_error(:not_loaded)

  # Simulation

  @doc """
  Integrates the particle system forward in time using Euler integration.
  Verifies conservation laws and returns :ok or {:error, reason}.
  """
  def integrate(_system_ref, _dt), do: :erlang.nif_error(:not_loaded)

  @doc """
  Applies a force to all particles in the system.
  """
  def apply_force(_system_ref, _force_vector, _dt), do: :erlang.nif_error(:not_loaded)

  # Conservation Verification

  @doc """
  Computes total energy in the system.
  """
  def compute_total_energy(_system_ref), do: :erlang.nif_error(:not_loaded)

  @doc """
  Verifies that conservation laws hold within tolerance.
  """
  def verify_conservation(_system_ref, _tolerance \\ 1.0e-6), do: :erlang.nif_error(:not_loaded)

  # Hardware Acceleration

  @doc """
  Gets information about available hardware accelerators.
  """
  def get_hardware_info, do: :erlang.nif_error(:not_loaded)

  @doc """
  Forces the use of a specific accelerator for the next operations.
  """
  def set_accelerator(_accelerator), do: :erlang.nif_error(:not_loaded)

  # Diagnostics

  @doc """
  Gets performance statistics for the last operations.
  """
  def get_performance_stats(_system_ref), do: :erlang.nif_error(:not_loaded)

  @doc """
  Gets detailed conservation violation information.
  """
  def get_conservation_report(_system_ref), do: :erlang.nif_error(:not_loaded)
end
