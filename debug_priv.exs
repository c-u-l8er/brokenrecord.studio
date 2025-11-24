#!/usr/bin/env elixir

IO.puts("Debugging priv_dir...")

priv_dir = :code.priv_dir(:broken_record_zero)
IO.puts("priv_dir: #{inspect(priv_dir)}")
IO.puts("priv_dir type: #{inspect(is_binary(priv_dir))}")

if is_binary(priv_dir) do
  expected_nif_file = :filename.join(priv_dir, "brokenrecord_physics.so")
  IO.puts("expected_nif_file: #{inspect(expected_nif_file)}")
else
  IO.puts("priv_dir is not a binary!")
end
