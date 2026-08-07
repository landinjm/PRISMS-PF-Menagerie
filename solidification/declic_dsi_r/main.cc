// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);
  std::string     parameters_filename = cli_options.get_parameters_filename();

  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 2;

  std::vector<FieldAttributes> fields = {FieldAttributes("u"),
                                         FieldAttributes("phi"),
                                         FieldAttributes("xi")};

  SolveBlock explicits(
    0,
    Explicit,
    Initialized,
    {0, 1},
    make_dependency_set(
      fields,
      {"old_1(u)", "grad(old_1(u))", "old_1(phi)", "grad(old_1(phi))", "old_1(xi)"}));
  SolveBlock xi_solve(1,
                      Explicit,
                      Uninitialized,
                      {2},
                      make_dependency_set(fields, {"u", "phi", "grad(phi)"}));

  std::vector<SolveBlock>        solves({explicits, xi_solve});
  UserInputParameters<dim>       user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields,
                                       solves,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  problem.solve();

  return 0;
}
