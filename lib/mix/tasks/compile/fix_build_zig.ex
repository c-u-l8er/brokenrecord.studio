defmodule Mix.Tasks.Compile.FixBuildZig do
  def run(_manifests) do
    build_zig_path = "/tmp/Elixir.AII.NIF/build.zig"

    if File.exists?(build_zig_path) do
      content = File.read!(build_zig_path)

      unless String.contains?(content, "lib.linkSystemLibrary(\"vulkan\");") do
        new_content =
          String.replace(
            content,
            "lib.linker_allow_shlib_undefined = true;",
            "lib.linker_allow_shlib_undefined = true;\n            lib.linkSystemLibrary(\"vulkan\");"
          )

        File.write!(build_zig_path, new_content)
      end
    end

    :ok
  end

  def manifests do
    []
  end

  def clean do
    :ok
  end
end
