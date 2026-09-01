// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

#include <prismspf/user_inputs/spatial_discretization.h>

#include <prismspf/utilities/assert.h>

#include <numbers>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prismspf::MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);
  std::string     parameters_filename = cli_options.get_parameters_filename();

  constexpr unsigned int dim    = 3;
  constexpr unsigned int degree = 1;

  std::vector<FieldAttributes> fields = {FieldAttributes("c"),
                                         FieldAttributes("mu"),
                                         FieldAttributes("f_tot")};

  SolveBlock c_block;
  c_block.id               = 0;
  c_block.solve_type       = Explicit;
  c_block.solve_timing     = Initialized;
  c_block.field_indices    = {0};
  c_block.dependencies_rhs = make_dependency_set(fields, {"old_1(c)", "grad(old_1(mu))"});

  SolveBlock mu_block;
  mu_block.id               = 1;
  mu_block.solve_type       = Explicit;
  mu_block.solve_timing     = Uninitialized;
  mu_block.field_indices    = {1};
  mu_block.dependencies_rhs = make_dependency_set(fields, {"c", "grad(c)"});

  SolveBlock pp_block;
  pp_block.id               = 2;
  pp_block.solve_type       = Explicit;
  pp_block.solve_timing     = PostProcess;
  pp_block.field_indices    = {2};
  pp_block.dependencies_rhs = make_dependency_set(fields, {"c", "grad(c)"});

  std::vector<SolveBlock> solve_blocks({c_block, mu_block, pp_block});

  UserInputParameters<dim> user_inputs(parameters_filename);

  /**
   * Determine the length of the domain based on the perturbation wavelength.
   * Additionally, determine a reasonable amount of subdivisions to apply.
   */
  ASSERT(user_inputs.spatial_discretization.mesh_type == TriangulationType::Rectangular,
         "This application only works for rectangular meshes!");
  auto &mesh = user_inputs.spatial_discretization.rectangular_mesh;
  // TODO: Check that y and z lengths are the same
  // Setup logic for cylinder vs plate IC
  // TODO: Assert 3D
  static constexpr double lambda =
    2 * std::numbers::pi / CustomPDE<dim, degree, double>::k;

  const auto         x_length       = 2 * lambda;
  const auto         y_length       = mesh.size[1] - mesh.lower_bound[1];
  const unsigned int x_subdivisions = std::ceil(x_length / y_length);

  mesh.size[0]         = x_length + mesh.lower_bound[0];
  mesh.size[2]         = x_length + mesh.lower_bound[2];
  mesh.subdivisions[0] = x_subdivisions;
  mesh.subdivisions[2] = x_subdivisions;

  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields,
                                         solve_blocks,
                                         user_inputs,
                                         pf_tools,
                                         pde_operator);
  problem.solve();

  return 0;
}
