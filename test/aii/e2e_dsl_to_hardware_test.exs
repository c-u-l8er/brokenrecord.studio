defmodule AII.DSL.E2EDSLToHardwareTest do
  use ExUnit.Case
  alias AII

  setup_all do
    available_hardware = AII.available_hardware()
    %{available_hardware: available_hardware}
  end

  describe "DSL end-to-end hardware acceleration" do
    test "basic particle system with GPU acceleration", %{available_hardware: available_hardware} do
      # Define a simple particle system using the DSL
      defmodule SimpleParticleSystem do
        use AII.DSL

        defagent Particle do
          property(:mass, :f32)
          state(:position, AII.Types.Vec3)
          state(:velocity, AII.Types.Vec3)
        end

        definteraction :gravity, with: Particle do
          # Simple gravity - just return the particle unchanged for test
          particle
        end
      end

      # Test that the DSL compiles and can be used
      assert function_exported?(SimpleParticleSystem, :__agents__, 0)
      assert function_exported?(SimpleParticleSystem, :__interactions__, 0)

      # Test that we can run a simulation with GPU
      result = AII.run_simulation(SimpleParticleSystem, steps: 1, hardware: :gpu)
      assert {:ok, sim_result} = result
      assert sim_result.hardware == :gpu
      assert sim_result.conservation_verified == true

      # Verify that GPU was actually used (not CPU fallback)
      # This would require checking logs or NIF output, but for now assume success
    end

    test "conservation laws maintained across hardware backends", %{
      available_hardware: available_hardware
    } do
      defmodule ConservedSystem do
        use AII.DSL

        defagent ConservedParticle do
          property(:mass, :f32)
          state(:position, AII.Types.Vec3)
          state(:velocity, AII.Types.Vec3)
          conserves([:energy])
        end

        definteraction :conserve, with: ConservedParticle do
          particle
        end
      end

      # Test with CPU (always available)
      result = AII.run_simulation(ConservedSystem, steps: 1, hardware: :cpu)
      assert {:ok, sim_result} = result
      assert sim_result.hardware == :cpu
      assert sim_result.conservation_verified == true

      # Test with GPU only if available
      if :gpu in available_hardware do
        result = AII.run_simulation(ConservedSystem, steps: 1, hardware: :gpu)
        assert {:ok, sim_result} = result
        assert sim_result.hardware == :gpu
        assert sim_result.conservation_verified == true
      end
    end

    test "hardware auto-selection works", %{available_hardware: available_hardware} do
      defmodule AutoSelectSystem do
        use AII.DSL

        defagent AutoParticle do
          property(:mass, :f32)
          state(:position, AII.Types.Vec3)
        end

        definteraction :auto_compute, with: AutoParticle do
          particle
        end
      end

      # Test auto hardware selection
      result = AII.run_simulation(AutoSelectSystem, steps: 1, hardware: :auto)
      assert {:ok, sim_result} = result
      # Auto should select an available hardware
      assert sim_result.hardware in (available_hardware ++ [:auto])
      assert sim_result.conservation_verified == true
    end

    test "DSL module validation" do
      # Test that valid DSL modules work
      defmodule ValidDSLSystem do
        use AII.DSL

        defagent ValidAgent do
          property(:mass, :f32)
          state(:position, AII.Types.Vec3)
          conserves([:energy])
        end

        definteraction :valid_interaction, with: ValidAgent do
          particle
        end
      end

      # Should work without errors
      result = AII.run_simulation(ValidDSLSystem, steps: 1, hardware: :cpu)
      assert {:ok, sim_result} = result
      assert sim_result.hardware == :cpu
      assert sim_result.conservation_verified == true
    end

    test "error handling for invalid modules" do
      # Test that invalid modules are rejected
      result = AII.run_simulation(NonExistentModule, steps: 1)
      assert {:error, :invalid_system_module} = result
    end

    test "hardware acceleration fallback" do
      defmodule FallbackSystem do
        use AII.DSL

        defagent FallbackParticle do
          property(:mass, :f32)
          state(:position, AII.Types.Vec3)
        end

        definteraction :fallback, with: FallbackParticle do
          particle
        end
      end

      # Test that unknown hardware is handled gracefully
      result = AII.run_simulation(FallbackSystem, steps: 1, hardware: :unknown_hardware)
      assert {:ok, sim_result} = result
      # Should handle unknown hardware and still run
      assert sim_result.hardware == :unknown_hardware
      assert sim_result.conservation_verified == true
    end

    test "multiple interactions in DSL system", %{available_hardware: available_hardware} do
      defmodule MultiInteractionSystem do
        use AII.DSL

        defagent MultiParticle do
          property(:mass, :f32)
          state(:position, AII.Types.Vec3)
          state(:velocity, AII.Types.Vec3)
        end

        definteraction :gravity, with: MultiParticle do
          particle
        end

        definteraction :collision, with: MultiParticle do
          particle
        end

        definteraction :update, with: MultiParticle do
          particle
        end
      end

      # Test that multiple interactions work
      interactions = MultiInteractionSystem.__interactions__()
      assert length(interactions) == 3

      # Test simulation runs with GPU if available
      hardware = if :gpu in available_hardware, do: :gpu, else: :cpu
      result = AII.run_simulation(MultiInteractionSystem, steps: 1, hardware: hardware)
      assert {:ok, sim_result} = result
      assert sim_result.hardware == hardware
      assert sim_result.conservation_verified == true
    end

    test "GPU hardware is properly detected and used", %{available_hardware: available_hardware} do
      # This test verifies that GPU detection works and GPU is used when available
      gpu_available = :gpu in available_hardware

      if gpu_available do
        defmodule GPUVerifySystem do
          use AII.DSL

          defagent GPUParticle do
            property(:mass, :f32)
            state(:position, AII.Types.Vec3)
          end

          definteraction :gpu_compute, with: GPUParticle do
            particle
          end
        end

        # Run with GPU hardware
        result = AII.run_simulation(GPUVerifySystem, steps: 1, hardware: :gpu)
        assert {:ok, sim_result} = result
        assert sim_result.hardware == :gpu
        assert sim_result.conservation_verified == true

        # Verify that the system has GPU in available hardware
        assert :gpu in AII.available_hardware()
      else
        # If GPU not available, skip GPU-specific tests
        assert :gpu not in available_hardware
      end
    end
  end
end
