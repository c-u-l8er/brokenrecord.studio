# BrokenRecord Implementation Guide
## Critical Implementation Details for Atomic/Chemic/Bionic DSL

**Context:** This doc assumes you have read the other AII architecture documents.  
**Goal:** Implement the hierarchical composition system (Agentic → Atomic → Chemic → Bionic).  
**Audience:** LLM coding assistant

---

## 1. Core Type System

### Critical Types to Implement

```elixir
# lib/aii/types.ex

defmodule AII.Types do
  # Conserved value wrapper (CRITICAL - foundation of everything)
  defmodule Conserved do
    @type t(inner) :: %__MODULE__{
      value: inner,
      source: atom(),
      tracked: boolean(),
      lineage: [atom()]  # NEW: Track full provenance chain
    }
    
    defstruct [:value, :source, :tracked, lineage: []]
    
    # ONLY way to create conserved value
    def new(value, source) when is_atom(source) do
      %__MODULE__{
        value: value,
        source: source,
        tracked: true,
        lineage: [source]
      }
    end
    
    # ONLY way to move conserved value
    def transfer(from, to, amount) do
      if from.value < amount do
        {:error, :insufficient_value}
      else
        new_from = %{from | 
          value: from.value - amount,
          lineage: from.lineage ++ [:transfer]
        }
        new_to = %{to | 
          value: to.value + amount,
          lineage: to.lineage ++ from.lineage
        }
        {:ok, new_from, new_to}
      end
    end
    
    # Check if two conserved values have compatible lineage
    def compatible_lineage?(c1, c2) do
      MapSet.intersection(
        MapSet.new(c1.lineage),
        MapSet.new(c2.lineage)
      ) != MapSet.new()
    end
  end
  
  # Particle type (base unit)
  defmodule Particle do
    defstruct [
      :id,
      :position,
      :velocity, 
      :mass,
      :information,  # Conserved<Float>
      :owner,        # :atomic_name or :chemic_name
      :metadata
    ]
  end
end
```

---

## 2. Atomic DSL Implementation

### Critical Macro: `defatomic`

```elixir
# lib/aii/dsl/atomic.ex

defmodule AII.DSL.Atomic do
  defmacro defatomic(name, opts \\ [], do: block) do
    quote do
      defmodule unquote(:"AII.Atomics.#{name |> Macro.camelize()}") do
        use AII.Atomic
        
        Module.put_attribute(__MODULE__, :atomic_name, unquote(name))
        Module.put_attribute(__MODULE__, :atomic_number, unquote(opts[:atomic_number]))
        Module.put_attribute(__MODULE__, :atomic_type, unquote(opts[:type] || :unknown))
        
        unquote(block)
        
        # Auto-generate functions after block is evaluated
        @before_compile AII.Atomic
      end
    end
  end
  
  # Kernel macro - defines transformation logic
  defmacro kernel(do: block) do
    quote do
      def kernel_function(atomic_state, inputs) do
        # Make inputs available in scope
        var!(inputs) = inputs
        var!(atomic) = atomic_state
        
        unquote(block)
      end
    end
  end
  
  # Input declaration
  defmacro input(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :inputs, {
        unquote(name),
        unquote(opts[:type]),
        unquote(opts[:count] || :single)
      })
    end
  end
  
  # State declaration (internal particles)
  defmacro state(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :state_fields, {
        unquote(name),
        unquote(opts[:type]),
        unquote(opts[:conserves] || false)
      })
    end
  end
  
  # Conservation constraint
  defmacro conserves(quantity, do: block) do
    quote do
      Module.put_attribute(__MODULE__, :conservation_laws, {
        unquote(quantity),
        fn inputs, outputs -> unquote(block) end
      })
    end
  end
  
  # Transform block - the actual computation
  defmacro transform(do: block) do
    quote do
      def transform_function(atomic_state, inputs) do
        unquote(block)
      end
    end
  end
  
  # Accelerator hint
  defmacro accelerator(type) do
    quote do
      Module.put_attribute(__MODULE__, :accelerator_hint, unquote(type))
    end
  end
end
```

### Critical: Atomic Behavior Module

```elixir
# lib/aii/atomic.ex

defmodule AII.Atomic do
  @callback execute(atomic_state :: term(), inputs :: map()) :: 
    {:ok, atomic_state :: term(), outputs :: map()} | {:error, term()}
  
  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Atomic
      
      import AII.DSL.Atomic
      
      Module.register_attribute(__MODULE__, :inputs, accumulate: true)
      Module.register_attribute(__MODULE__, :state_fields, accumulate: true)
      Module.register_attribute(__MODULE__, :conservation_laws, accumulate: true)
      Module.register_attribute(__MODULE__, :accelerator_hint, [])
      
      # Default implementation
      def execute(atomic_state, inputs) do
        # 1. Verify input conservation
        :ok = verify_input_conservation(inputs)
        
        # 2. Run kernel
        {updated_state, outputs} = kernel_function(atomic_state, inputs)
        
        # 3. Verify output conservation
        :ok = verify_output_conservation(inputs, outputs)
        
        {:ok, updated_state, outputs}
      end
      
      defp verify_input_conservation(inputs) do
        for {quantity, check_fn} <- @conservation_laws do
          unless check_fn.(inputs, nil) do
            raise AII.ConservationViolation, """
            Atomic #{@atomic_name} input violated conservation of #{quantity}
            """
          end
        end
        :ok
      end
      
      defp verify_output_conservation(inputs, outputs) do
        for {quantity, check_fn} <- @conservation_laws do
          unless check_fn.(inputs, outputs) do
            raise AII.ConservationViolation, """
            Atomic #{@atomic_name} output violated conservation of #{quantity}
            Inputs: #{inspect(inputs)}
            Outputs: #{inspect(outputs)}
            """
          end
        end
        :ok
      end
    end
  end
  
  defmacro __before_compile__(_env) do
    quote do
      # Generate metadata function
      def __atomic_metadata__ do
        %{
          name: @atomic_name,
          atomic_number: @atomic_number,
          type: @atomic_type,
          inputs: @inputs,
          state_fields: @state_fields,
          conservation_laws: @conservation_laws,
          accelerator: @accelerator_hint
        }
      end
    end
  end
end
```

---

## 3. Chemic DSL Implementation

### Critical Macro: `defchemic`

```elixir
# lib/aii/dsl/chemic.ex

defmodule AII.DSL.Chemic do
  defmacro defchemic(name, opts \\ [], do: block) do
    quote do
      defmodule unquote(:"AII.Chemics.#{name |> Macro.camelize()}") do
        use AII.Chemic
        
        Module.put_attribute(__MODULE__, :chemic_name, unquote(name))
        Module.put_attribute(__MODULE__, :element_number, unquote(opts[:element_number]))
        Module.put_attribute(__MODULE__, :element_class, unquote(opts[:class]))
        
        unquote(block)
        
        @before_compile AII.Chemic
      end
    end
  end
  
  # Composition - declare atomics in this chemics
  defmacro composition(do: block) do
    quote do
      def __composition__ do
        unquote(block)
      end
    end
  end
  
  # Atomic declaration within composition
  defmacro atomic(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :atomics, {
        unquote(name),
        unquote(opts[:type])
      })
    end
  end
  
  # Nested chemic
  defmacro chemic(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :sub_chemics, {
        unquote(name),
        unquote(opts[:type])
      })
    end
  end
  
  # Bonds - connections between atmoics
  defmacro bonds(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :bonds, unquote(Macro.escape(block)))
    end
  end
  
  # Bond operator: →
  defmacro left → right do
    quote do
      {unquote(left), unquote(right)}
    end
  end
end
```

### Critical: Chemic Execution Engine

```elixir
# lib/aii/chemic.ex

defmodule AII.Chemic do
  defmacro __using__(_opts) do
    quote do
      import AII.DSL.Chemic
      
      Module.register_attribute(__MODULE__, :atomics, accumulate: true)
      Module.register_attribute(__MODULE__, :sub_chemics, accumulate: true)
      Module.register_attribute(__MODULE__, :bonds, [])
      
      # Execute chemic
      def execute(chemic_state, inputs) do
        # 1. Parse bonds into DAG
        dag = parse_bonds_to_dag(@bonds)
        
        # 2. Topological sort
        execution_order = topological_sort(dag)
        
        # 3. Execute each node in order
        {updated_state, outputs} = 
          Enum.reduce(execution_order, {chemic_state, inputs}, 
            fn node, {state, data} ->
              execute_node(state, data, node)
            end)
        
        # 4. Verify chemic-level conservation
        verify_conservation(inputs, outputs)
        
        {:ok, updated_state, outputs}
      end
      
      defp execute_node(state, data, node_name) do
        # Get atomic module
        {^node_name, atomic_type} =
          Enum.find(@atomics, fn {name, _} -> name == node_name end)

        atomic_module = AII.Atomics[atomic_type]

        # Get atomic state
        atomic_state = Map.get(state.atomics, node_name)

        # Get inputs for this atomic from data flow
        node_inputs = get_node_inputs(data, node_name)

        # Execute atomic
        {:ok, updated_atomic, outputs} =
          atomic_module.execute(atomic_state, node_inputs)

        # Update state
        updated_state =
          put_in(state.atomics[node_name], updated_atomic)

        # Update data flow
        updated_data = Map.put(data, node_name, outputs)

        {updated_state, updated_data}
      end
      
      defp parse_bonds_to_dag(bonds_block) do
        # CRITICAL: Parse AST of bonds block into graph structure
        # bonds_block is like: {:→, meta, [left, right]}
        # Return: %{node => [dependencies]}
        
        bonds_block
        |> extract_arrows()
        |> build_adjacency_list()
      end
      
      defp topological_sort(dag) do
        # CRITICAL: Kahn's algorithm for topological sort
        # Must detect cycles and raise error
        
        AII.Graph.topological_sort(dag)
      rescue
        AII.Graph.CycleDetected ->
          raise AII.ChemicError, "Cycle detected in bonds - DAG required"
      end
    end
  end
end
```

---

## 4. Bionic DSL Implementation

### Critical Macro: `defbionic`

```elixir
# lib/aii/dsl/bionic.ex

defmodule AII.DSL.Bionic do
  defmacro defbionic(name, opts \\ [], do: block) do
    quote do
      defmodule unquote(:"AII.Bionics.#{name |> Macro.camelize()}") do
        use AII.Bionic
        
        Module.put_attribute(__MODULE__, :bionic_name, unquote(name))
        
        unquote(block)
        
        @before_compile AII.Bionic
      end
    end
  end
  
  # DAG structure
  defmacro dag(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :dag_block, unquote(Macro.escape(block)))
    end
  end

  # Node definition
  defmacro node(name, do: block) do
    quote do
      Module.put_attribute(__MODULE__, :nodes, {
        unquote(name),
        unquote(Macro.escape(block))
      })
    end
  end

  # Edges between nodes
  defmacro edges(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :edges, unquote(Macro.escape(block)))
    end
  end
  
  # Input/output declarations
  defmacro inputs(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :bionic_inputs,
        unquote(Macro.escape(block)))
    end
  end

  defmacro outputs(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :bionic_outputs,
        unquote(Macro.escape(block)))
    end
  end
end
```

### Critical: Bionic Execution Engine

```elixir
# lib/aii/bionic.ex

defmodule AII.Bionic do
  defmacro __using__(_opts) do
    quote do
      import AII.DSL.Bionic

      Module.register_attribute(__MODULE__, :nodes, accumulate: true)
      Module.register_attribute(__MODULE__, :edges, [])
      Module.register_attribute(__MODULE__, :dag_block, [])

      # Main entry point
      def run(inputs, opts \\ []) do
        # 1. Build execution plan
        plan = build_execution_plan()

        # 2. Initialize state
        initial_state = initialize_bionic_state()

        # 3. Execute plan
        {final_state, outputs} =
          execute_plan(plan, initial_state, inputs, opts)

        # 4. Verify bionic conservation
        verify_bionic_conservation(inputs, outputs)

        # 5. Return outputs (and trace if requested)
        case opts[:trace] do
          true -> {:ok, outputs, extract_trace(final_state)}
          _ -> {:ok, outputs}
        end
      end
      
      defp build_execution_plan do
        # Parse DAG structure
        dag = parse_dag(@dag_block, @nodes, @edges)
        
        # Topological sort
        AII.Graph.topological_sort(dag)
      end
      
      defp execute_plan(plan, state, inputs, opts) do
        Enum.reduce(plan, {state, inputs}, fn node_name, {st, data} ->
          # Execute node (chemic)
          {updated_st, node_outputs} =
            execute_bionic_node(st, data, node_name, opts)
          
          # Update data flow
          updated_data = Map.put(data, node_name, node_outputs)
          
          # Log if tracing
          if opts[:trace] do
            log_execution(node_name, data, node_outputs)
          end
          
          {updated_st, updated_data}
        end)
      end
      
      defp execute_bionic_node(state, data, node_name, opts) do
        # Get node configuration
        {^node_name, node_config} = 
          Enum.find(@nodes, fn {n, _} -> n == node_name end)
        
        # Extract chemic from node config
        chemic_module = extract_chemic_module(node_config)

        # Get chemic state
        chemic_state = Map.get(state.chemics, node_name)

        # Get inputs for this node
        node_inputs = gather_node_inputs(data, node_config)

        # Execute chemic
        {:ok, updated_chemic, outputs} =
          chemic_module.execute(chemic_state, node_inputs)
        
        # Update state
        updated_state =
          put_in(state.chemics[node_name], updated_chemic)

        {updated_state, outputs}
      end

      defp verify_bionic_conservation(inputs, outputs) do
        # CRITICAL: Overall bionic must not create information
        input_info = AII.Conservation.total_information(inputs)
        output_info = AII.Conservation.total_information(outputs)

        if output_info > input_info + @tolerance do
          raise AII.ConservationViolation, """
          Bionic #{@bionic_name} created information:
            Inputs:  #{input_info}
            Outputs: #{output_info}
            Created: #{output_info - input_info}
          """
        end

        :ok
      end
    end
  end
end
```

---

## 5. Conservation Verification (CRITICAL)

```elixir
# lib/aii/conservation.ex

defmodule AII.Conservation do
  @tolerance 0.0001
  
  # Calculate total information in a data structure
  def total_information(data) when is_map(data) do
    data
    |> Map.values()
    |> Enum.map(&extract_information/1)
    |> Enum.sum()
  end
  
  defp extract_information(%AII.Types.Conserved{value: v}), do: v
  defp extract_information(%AII.Types.Particle{information: info}), 
    do: info.value
  defp extract_information(list) when is_list(list),
    do: Enum.sum(Enum.map(list, &extract_information/1))
  defp extract_information(_), do: 0.0
  
  # Verify conservation between two states
  def verify(before, after_state, opts \\ []) do
    tolerance = opts[:tolerance] || @tolerance
    
    before_info = total_information(before)
    after_info = total_information(after_state)
    
    diff = abs(before_info - after_info)
    
    if diff > tolerance do
      {:error, {:conservation_violated, 
        before: before_info, 
        after: after_info, 
        diff: diff}}
    else
      :ok
    end
  end
  
  # Compile-time symbolic verification (ADVANCED - optional)
  def symbolic_verify(ast) do
    # Analyze AST to prove conservation symbolically
    # This is hard - start with runtime checks
    # Future: constraint solver integration
    :not_implemented
  end
end
```

---

## 6. Graph Utilities (CRITICAL for DAG execution)

```elixir
# lib/aii/graph.ex

defmodule AII.Graph do
  defmodule CycleDetected do
    defexception [:message]
  end
  
  # Kahn's algorithm for topological sort
  def topological_sort(graph) do
    # graph is %{node => [dependencies]}
    
    # Find nodes with no dependencies
    no_deps = find_nodes_with_no_deps(graph)
    
    # Recursive sort
    do_topo_sort(graph, no_deps, [], MapSet.new())
  end
  
  defp do_topo_sort(graph, [], sorted, visited) do
    # Check if all nodes visited
    if MapSet.size(visited) == map_size(graph) do
      Enum.reverse(sorted)
    else
      # Cycle detected - some nodes not reachable
      raise CycleDetected, 
        "Cycle detected in graph - remaining: #{inspect(Map.keys(graph) -- MapSet.to_list(visited))}"
    end
  end
  
  defp do_topo_sort(graph, [node | rest], sorted, visited) do
    if MapSet.member?(visited, node) do
      do_topo_sort(graph, rest, sorted, visited)
    else
      # Add node to sorted list
      new_sorted = [node | sorted]
      new_visited = MapSet.put(visited, node)
      
      # Find newly available nodes
      newly_available = find_newly_available(graph, new_visited)
      
      do_topo_sort(graph, rest ++ newly_available, new_sorted, new_visited)
    end
  end
  
  defp find_nodes_with_no_deps(graph) do
    Enum.filter(graph, fn {_node, deps} ->
      deps == [] or Enum.all?(deps, &is_input?/1)
    end)
    |> Enum.map(fn {node, _} -> node end)
  end
  
  defp find_newly_available(graph, visited) do
    Enum.filter(graph, fn {node, deps} ->
      not MapSet.member?(visited, node) and
      Enum.all?(deps, &MapSet.member?(visited, &1))
    end)
    |> Enum.map(fn {node, _} -> node end)
  end
  
  # Build adjacency list from bonds/edges
  def build_adjacency_list(edges) do
    # edges is list of {from, to} tuples
    Enum.reduce(edges, %{}, fn {from, to}, acc ->
      Map.update(acc, to, [from], fn deps -> [from | deps] end)
    end)
  end
end
```

---

## 7. Hardware Dispatcher Integration

```elixir
# lib/aii/hardware_dispatcher.ex

defmodule AII.HardwareDispatcher do
  # Map atmoic to optimal hardware
  def dispatch(atomic_module) do
    metadata = atomic_module.__atomic_metadata__()
    
    # Check explicit accelerator hint
    case metadata.accelerator do
      :rt_cores -> :rt_cores
      :tensor_cores -> :tensor_cores
      :npu -> :npu
      :cuda_cores -> :cuda_cores
      nil -> infer_accelerator(metadata)
    end
  end
  
  defp infer_accelerator(metadata) do
    # Infer from atomic type and operations
    cond do
      has_spatial_query?(metadata) -> :rt_cores
      has_matrix_op?(metadata) -> :tensor_cores
      has_learned_model?(metadata) -> :npu
      true -> :cuda_cores
    end
  end
  
  # Generate hardware-specific code
  def generate_code(atomic_module, accelerator) do
    case accelerator do
      :rt_cores -> 
        AII.Codegen.Vulkan.generate_ray_query(atomic_module)
      :tensor_cores -> 
        AII.Codegen.Vulkan.generate_tensor_op(atomic_module)
      :npu -> 
        AII.Codegen.NPU.generate_inference(atomic_module)
      :cuda_cores -> 
        AII.Codegen.CUDA.generate_kernel(atomic_module)
    end
  end
end
```

---

## 8. Example Usage (For Testing)

```elixir
# test/examples/simple_bionic_test.exs

defmodule AII.Examples.SimpleBionicTest do
  use ExUnit.Case
  
  # Define a simple atomic
  defatomic :double do
    @atomic_number 1
    
    kernel do
      input :value, type: :float
      
      conserves :information do
        inputs[:value].info == outputs[:result].info
      end
      
      transform do
        result = Conserved.new(
          inputs[:value].value * 2,
          :double
        )
        
        %{result: result}
      end
    end
  end
  
  # Define a chemic
  defchemic :double_twice do
    @element_number 100
    
    composition do
      atomic :first, type: :double
      atomic :second, type: :double
    end
    
    bonds do
      input(:x) → :first
      :first → :second
      :second → output(:result)
    end
  end
  
  # Define a bionic
  defbionic :double_bionic do
    dag do
      node :process do
        chemic :double_twice
        input :number
      end
    end
    
    conserves :information
  end
  
  test "bionic executes with conservation" do
    input = Conserved.new(5.0, :user_input)
    
    {:ok, result} = AII.Bionics.DoubleBionic.run(%{number: input})
    
    # 5 * 2 * 2 = 20
    assert result.result.value == 20.0
    
    # Conservation verified
    assert result.result.info == input.info
  end
end
```

---

## 9. Critical Implementation Order

### Phase 1: Foundation (Week 1-2)
1. **Conserved type** - Must be bulletproof
2. **Particle type** - Base unit
3. **Conservation checker** - verify/2 function
4. **Graph utilities** - Topological sort

### Phase 2: Atmoics (Week 3-4)
1. **defatomic macro** - Parse and generate
2. **Atmoic behavior** - execute/2 
3. **Kernel execution** - Transform with conservation
4. **Test with simple atmoics** (add, multiply)

### Phase 3: Chemics (Week 5-6)
1. **defchemic macro** - Parse bonds
2. **Bond parsing** - AST → DAG
3. **Chemic execution** - execute_node/3
4. **Test with simple chemics**

### Phase 4: Bionics (Week 7-8)
1. **defbionic macro** - Parse DAG
2. **Bionic execution** - run/2
3. **Edge verification** - Conservation at boundaries
4. **Test end-to-end**

### Phase 5: Hardware (Week 9-10)
1. **Hardware dispatcher** - dispatch/1
2. **Code generation stubs** - generate_code/2
3. **Integration with atmoics**
4. **Performance testing**

---

## 10. Critical Edge Cases to Handle

### 1. Parallel Branches in Chemics
```elixir
# This must work:
bonds do
  input(:x) → :branch1
  input(:x) → :branch2
  [:branch1, :branch2] → :combine
  :combine → output(:result)
end

# Conservation: total(branch1 + branch2) == input(x)
```

### 2. Conditional Execution
```elixir
# Atomics can conditionally emit:
transform do
  if condition do
    emit(result1)
  else
    emit(result2)
  end
  # Conservation still verified!
end
```

### 3. Loops (Special Case)
```elixir
# Normally forbidden (DAG only)
# But iterative bionics need special handling:
defbionic :iterative do
  loop max_iterations: 10 do
    node :refine
    until :refine.output.quality > threshold
  end
end
```

### 4. Error Propagation
```elixir
# Conservation violations should:
# 1. Include full trace
# 2. Show exact location
# 3. Provide debugging info
```

---

## 11. Testing Strategy

### Unit Tests
- Each macro independently
- Conservation checker with edge cases
- Graph utilities (cycles, diamonds, etc)

### Integration Tests  
- Simple atomic → chemic → bionic
- Conservation across boundaries
- Error messages quality

### Property Tests
- Conservation always holds
- No information created
- DAG properties maintained

### Performance Tests
- Large chemics (100+ atomics)
- Deep bionics (10+ layers)
- Hardware dispatch overhead

---

## File Structure

```
lib/aii/
├── dsl/
│   ├── atomic.ex            # defatomic macro
│   ├── chemic.ex            # defchemic macro
│   └── bionic.ex            # defbionic macro
├── atomic.ex                # Atomic behavior
├── chemic.ex                # Chemic execution engine
├── bionic.ex                # Bionic execution engine
├── atomics.ex               # Atomic registry
├── chemics.ex               # Chemic registry
├── bionics.ex               # Bionic registry
├── atomic_runtime.ex        # Atomic runtime
├── chemic_runtime.ex        # Chemic runtime
├── bionic_runtime.ex        # Bionic runtime
├── atomic_supervisor.ex     # Atomic supervision
├── chemic_supervisor.ex     # Chemic supervision
├── bionic_supervisor.ex     # Bionic supervision
├── atomic_app.ex            # Atomic application
├── chemic_app.ex            # Chemic application
├── bionic_app.ex            # Bionic application
test/
├── aii/
│   ├── atomic_test.exs      # Atomic tests
│   ├── chemic_test.exs      # Chemic tests
│   └── bionic_test.exs      # Bionic tests
benchmarks/
├── atomic_bench.exs         # Atomic benchmarks
├── chemic_bench.exs         # Chemic benchmarks
└── bionic_bench.exs         # Bionic benchmarks
docs/
├── atomic_guide.md          # Atomic DSL guide
├── chemic_guide.md          # Chemic DSL guide
└── bionic_guide.md          # Bionic DSL guide
examples/
├── atomic_example.exs       # Atomic examples
├── chemic_example.exs       # Chemic examples
└── bionic_example.exs       # Bionic examples
config/
├── atomic_config.exs        # Atomic configuration
├── chemic_config.exs        # Chemic configuration
└── bionic_config.exs        # Bionic configuration
rel/
├── atomic_release.ex        # Atomic release
├── chemic_release.ex        # Chemic release
└── bionic_release.ex        # Bionic release
```

## 12. What NOT to Implement Yet

### Skip These Initially:
1. ❌ Symbolic verification (too complex)
2. ❌ Visual DAG editor (separate project)
3. ❌ Hardware code generation (use stubs)
4. ❌ Distributed execution (later)
5. ❌ Optimize for performance (correctness first)

### Focus On:
1. ✅ Correct macro expansion
2. ✅ Conservation verification
3. ✅ Clear error messages
4. ✅ Clean DSL syntax
5. ✅ End-to-end examples

---

## 13. Success Criteria

### Must Have:
- [ ] Atomics execute with conservation ✓
- [ ] Chemics compose atomics ✓
- [ ] Bionics compose Chemics ✓
- [ ] Conservation verified at all boundaries ✓
- [ ] Clear error messages with traces ✓
- [ ] Example chatbot works ✓

### Should Have:
- [ ] Hardware dispatcher functional
- [ ] Parallel execution in chemics
- [ ] Good performance (>1000 particles/sec)

### Nice to Have:
- [ ] Compile-time optimizations
- [ ] Visual trace output
- [ ] Integration with existing AII runtime

---

## 14. Key Files to Create

```
lib/
├── aii/
│   ├── types.ex                 # Conserved, Particle
│   ├── conservation.ex          # verify/2
│   ├── graph.ex                 # topological_sort/1
│   ├── atomic.ex                # Atmoic behavior
│   ├── chemic.ex                # Chemic behavior
│   ├── bionic.ex                # Bionic behavior
│   ├── hardware_dispatcher.ex   # dispatch/1
│   └── dsl/
│       ├── atomic.ex            # defatomic macro
│       ├── chemic.ex            # defchemic macro
│       └── bionic.ex            # defbionic macro
test/
├── aii/
│   ├── types_test.exs
│   ├── conservation_test.exs
│   ├── graph_test.exs
│   └── integration_test.exs
└── examples/
    ├── simple_atomics_test.exs
    ├── transformer_test.exs
    └── chatbot_test.exs
```

---

## 15. Final Notes for LLM

**Start with:**
1. Implement `AII.Types.Conserved` first - it's the foundation
2. Then `AII.Conservation.verify/2` - you'll need it everywhere
3. Then `AII.Graph.topological_sort/1` - critical for DAG execution
4. Then tackle macros one at a time (Atomic → Chemic → Bionic)

**Remember:**
- Conservation is NON-NEGOTIABLE - raise errors liberally
- Clear error messages > performance
- Test each piece before moving on
- The DSL should feel natural to write

**This is the architecture that makes hallucination impossible.** 🎵⚛️

Get started with `AII.Types.Conserved` - it's literally the atom of everything!
