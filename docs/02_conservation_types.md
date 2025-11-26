# AII Migration: Conservation Type System
## Document 2: Types & Verification

### Core Types Module

**File:** `lib/aii/types.ex`

```elixir
defmodule AII.Types do
  @moduledoc """
  Core types for AII: Conserved<T>, Energy, Momentum, etc.
  These types enforce conservation laws at compile time.
  """
  
  # Conserved wrapper type
  defmodule Conserved do
    @type t(inner) :: %__MODULE__{
      value: inner,
      source: atom(),  # Where this value came from
      tracked: boolean()
    }
    
    defstruct value: 0, source: :unknown, tracked: true
    
    def new(value, source \\ :initial) do
      %__MODULE__{value: value, source: source, tracked: true}
    end
    
    # Can only transfer, never create
    def transfer(from, to, amount) do
      if from.value < amount do
        {:error, :insufficient_value}
      else
        new_from = %{from | value: from.value - amount}
        new_to = %{to | value: to.value + amount}
        {:ok, new_from, new_to}
      end
    end
  end
  
  # Physics types
  defmodule Energy do
    @type t :: Conserved.t(float)
  end
  
  defmodule Momentum do
    @type t :: Conserved.t({float, float, float})
  end
  
  defmodule Information do
    @type t :: Conserved.t(float)
  end
  
  # Geometric types
  defmodule Vec3 do
    @type t :: {float, float, float}
    
    def add({x1, y1, z1}, {x2, y2, z2}), do: {x1 + x2, y1 + y2, z1 + z2}
    def mul({x, y, z}, scalar), do: {x * scalar, y * scalar, z * scalar}
    def magnitude({x, y, z}), do: :math.sqrt(x*x + y*y + z*z)
  end
end
```

---

### Conservation Checker

**File:** `lib/aii/conservation_checker.ex`

```elixir
defmodule AII.ConservationChecker do
  @moduledoc """
  Compile-time verification of conservation laws.
  Analyzes interaction AST to prove conservation.
  """
  
  alias AII.Types.Conserved
  
  def verify(interaction, agent_defs) do
    # Extract conserved quantities from agents
    conserved = extract_conserved_quantities(agent_defs)
    
    # Track each quantity through interaction
    Enum.reduce(conserved, :ok, fn quantity, acc ->
      case track_conservation(interaction.body, quantity) do
        :ok -> acc
        {:error, reason} -> 
          {:error, "#{quantity} not conserved: #{reason}"}
      end
    end)
  end
  
  defp track_conservation(ast, quantity) do
    # Build symbolic expression for quantity before/after
    {before_expr, after_expr} = build_expressions(ast, quantity)
    
    # Check if before == after symbolically
    if symbolically_equal?(before_expr, after_expr) do
      :ok
    else
      # Try to prove with runtime check
      {:needs_runtime_check, before_expr, after_expr}
    end
  end
  
  defp build_expressions(ast, quantity) do
    # Walk AST, accumulate quantity changes
    # Return {total_before, total_after}
  end
  
  defp symbolically_equal?(expr1, expr2) do
    # Simple symbolic equality (can be enhanced)
    normalize(expr1) == normalize(expr2)
  end
  
  defp normalize(expr) do
    # Simplify algebraic expressions
    # e.g., (a + b - b) -> a
  end
end
```

---

### DSL Modifications

**File:** `lib/aii/dsl.ex` (modified from original)

```elixir
defmodule AII.DSL do
  @moduledoc "AII DSL with conservation types"
  
  # New: property (invariant field)
  defmacro property(name, type, opts \\ []) do
    invariant = Keyword.get(opts, :invariant, false)
    
    quote do
      field = %{
        name: unquote(name),
        type: unquote(type),
        invariant: unquote(invariant),
        kind: :property
      }
      Module.put_attribute(__MODULE__, :fields, field)
    end
  end
  
  # New: state (mutable field)
  defmacro state(name, type, opts \\ []) do
    quote do
      field = %{
        name: unquote(name),
        type: unquote(type),
        invariant: false,
        kind: :state
      }
      Module.put_attribute(__MODULE__, :fields, field)
    end
  end
  
  # New: derives (computed field)
  defmacro derives(name, type, do: block) do
    quote do
      derived = %{
        name: unquote(name),
        type: unquote(type),
        computation: unquote(Macro.escape(block)),
        kind: :derived
      }
      Module.put_attribute(__MODULE__, :fields, derived)
    end
  end
  
  # Modified: interaction with hardware hint
  defmacro definteraction(name, opts \\ [], do: block) do
    accelerator = Keyword.get(opts, :accelerator, :auto)
    
    quote do
      interaction = %{
        name: unquote(name),
        body: unquote(Macro.escape(block)),
        accelerator: unquote(accelerator),
        conserved: []  # Filled by checker
      }
      Module.put_attribute(__MODULE__, :interactions, interaction)
    end
  end
  
  # New: conserved_quantity declaration
  defmacro conserved_quantity(name, opts \\ []) do
    quote do
      quantity = %{
        name: unquote(name),
        type: Keyword.get(unquote(opts), :type, :scalar),
        law: Keyword.get(unquote(opts), :law, :sum)
      }
      Module.put_attribute(__MODULE__, :conserved_quantities, quantity)
    end
  end
end
```

---

### Usage Example (For Next Developer)

```elixir
defmodule MyPhysics do
  use AII.DSL
  
  # Declare what's conserved in this system
  conserved_quantity :energy, type: :scalar, law: :sum
  conserved_quantity :momentum, type: :vector3, law: :sum
  
  defagent Particle do
    # Invariant properties (can't change)
    property :mass, Float, invariant: true
    property :charge, Float, invariant: true
    
    # Mutable state
    state :position, Vec3
    state :velocity, Vec3
    
    # Conserved quantities (tracked by type system)
    state :energy, Conserved<Energy>
    state :momentum, Conserved<Momentum>
    
    # Computed/derived quantities
    derives :kinetic_energy, Energy do
      0.5 * mass * Vec3.magnitude(velocity) ** 2
    end
    
    derives :momentum_vec, Momentum do
      Vec3.mul(velocity, mass)
    end
    
    # Declare what this agent conserves
    conserves :energy, :momentum
  end
  
  # Interaction with hardware hint
  definteraction :elastic_collision, accelerator: :rt_cores do
    let {p1, p2} do
      # Compiler verifies energy and momentum conserved here!
      
      # RT cores handle collision detection
      if colliding?(p1, p2) do
        # Exchange momentum (type system ensures conservation)
        {p1_new, p2_new} = exchange_momentum(p1, p2)
        {p1_new, p2_new}
      else
        {p1, p2}
      end
    end
  end
end
```

---

### Compile-Time Checking Flow

```
1. Parse DSL → Extract conserved quantities
                ├─ energy
                ├─ momentum
                └─ information

2. For each interaction:
   ├─ Build symbolic expressions for conserved quantities
   │  └─ total_energy_before = sum(particles, &.energy)
   │     total_energy_after = sum(particles_updated, &.energy)
   │
   ├─ Try to prove: before == after
   │  ├─ Success → No runtime check needed ✓
   │  └─ Failure → Insert runtime check
   │
   └─ Generate code with verification

3. If ANY interaction fails verification:
   └─ COMPILATION ERROR (conservation violated)
```

---

### Runtime Verification (Zig)

**File:** `runtime/zig/conservation.zig`

```zig
const std = @import("std");

pub fn Conserved(comptime T: type) type {
    return struct {
        value: T,
        source: []const u8,
        tracked: bool,
        
        const Self = @This();
        
        pub fn init(value: T, source: []const u8) Self {
            return Self{
                .value = value,
                .source = source,
                .tracked = true,
            };
        }
        
        pub fn transfer(
            from: *Self,
            to: *Self,
            amount: T
        ) !void {
            if (from.value < amount) {
                return error.InsufficientValue;
            }
            from.value -= amount;
            to.value += amount;
        }
    };
}

pub fn verifyConserved(
    before: f32,
    after: f32,
    tolerance: f32
) bool {
    return @abs(before - after) < tolerance;
}

// Track total energy
pub fn computeTotalEnergy(particles: []const Particle) f32 {
    var total: f32 = 0.0;
    for (particles) |p| {
        total += p.energy.value;
    }
    return total;
}
```

---

### Key Implementation Points

**1. Conserved<T> Must Be:**
- Immutable (can only transfer)
- Tracked (compiler knows about it)
- Typed (Energy ≠ Momentum)

**2. Compiler Must:**
- Parse `conserves :energy, :momentum`
- Build symbolic expressions for totals
- Verify: `total_before == total_after`
- Generate runtime checks if needed

**3. Runtime Must:**
- Track conservation in Zig
- Raise errors on violation
- Support transfer operations

**4. Type Errors Look Like:**
```
Error: Conservation violated in interaction 'collide'
  Energy before: 100.0 J
  Energy after: 95.0 J
  Missing: 5.0 J
  
  Suggestion: Check if energy is transferred correctly.
```

---

### Next Steps for Implementation

1. Implement `AII.Types` module (basic types)
2. Add `Conserved<T>` with transfer logic
3. Modify DSL to accept `conserved_quantity`
4. Build basic conservation checker (symbolic)
5. Test with simple example (gravity)
6. Add runtime verification in Zig
7. Connect compile-time + runtime checks
