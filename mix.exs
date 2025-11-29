defmodule AII.MixProject do
  use Mix.Project

  def project do
    [
      app: :aii,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      compilers: Mix.compilers(),
      deps: deps()
    ]
  end

  # Run "mix compile" to compile Zig NIFs
  def application do
    [
      extra_applications: [:logger, :runtime_tools, :zigler]
    ]
  end

  defp deps do
    [
      {:benchee, "~> 1.3", only: :dev},
      {:benchee_html, "~> 1.0", only: :dev},
      {:ex_doc, "~> 0.29", only: :dev, runtime: false},
      {:plug_cowboy, "~> 2.6"},
      {:jason, "~> 1.4"},
      {:zigler, "~> 0.11"}
    ]
  end
end
