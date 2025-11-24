defmodule BrokenRecordZero.MixProject do
  use Mix.Project

  def project do
    [
      app: :broken_record_zero,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      compilers: Mix.compilers(),
      make: [nif_name: "brokenrecord_physics"],
      deps: deps()
    ]
  end

  def aliases do
    [
      nif: fn _args ->
        {_, 0} = System.cmd("make", ["-C", "c_src"], into: IO.stream(:stdio, :line))
        priv_dir = Path.join(Path.dirname(Mix.Project.app_path()), "priv")
        File.cp!("c_src/brokenrecord_physics.so", Path.join(priv_dir, "brokenrecord_physics.so"))
        :ok
      end
    ]
  end

  # Run "mix compile" to compile the C NIF
  def application do
    [
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  defp deps do
    [
      {:benchee, "~> 1.3", only: :dev},
      {:benchee_html, "~> 1.0", only: :dev},
      {:ex_doc, "~> 0.29", only: :dev, runtime: false},
      {:plug_cowboy, "~> 2.6"},
      {:jason, "~> 1.4"},
      {:elixir_make, "~> 0.6", runtime: false}
    ]
  end
end
