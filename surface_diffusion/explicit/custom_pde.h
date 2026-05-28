// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/base/config.h>

#include <prismspf/core/pde_operator_base.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , epsilon(get_user_inputs().user_constants.get_double("epsilon"))
  {}

private:
  DEAL_II_ALWAYS_INLINE ScalarValue
  g(ScalarValue phi) const
  {
    return 30.0 * phi * phi * (1.0 - phi) * (1.0 - phi) + reg;
  }

  DEAL_II_ALWAYS_INLINE ScalarValue
  g_prime(ScalarValue phi) const
  {
    return 60.0 * (phi - 1.0) * phi * (2.0 * phi - 1.0);
  }

  DEAL_II_ALWAYS_INLINE ScalarValue
  B(ScalarValue phi) const
  {
    return 18.0 * phi * phi * (1.0 - phi) * (1.0 - phi);
  }

  DEAL_II_ALWAYS_INLINE ScalarValue
  B_prime(ScalarValue phi) const
  {
    return 36.0 * (phi - 1.0) * phi * (2.0 * phi - 1.0);
  }

  DEAL_II_ALWAYS_INLINE ScalarValue
  M(ScalarValue phi) const
  {
    return 2.0 * B(phi);
  }

  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;
    if (index == 0)
      {
        const double r1     = (point - dealii::Point<dim>(30.0, 50.0)).norm();
        const double r2     = (point - dealii::Point<dim>(70.0, 50.0)).norm();
        const double radius = 25.0;

        // phi = 1 inside either circle, 0 outside, smooth tanh interface
        const double phi1 = 0.5 * (1.0 - std::tanh((r1 - radius) / (2.0 * epsilon[0])));
        const double phi2 = 0.5 * (1.0 - std::tanh((r2 - radius) / (2.0 * epsilon[0])));

        const double phi = std::min(phi1 + phi2, 1.0);

        scalar_value = phi;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 0) // phi
      {
        const auto phi     = variable_list.template get_value<Scalar, OldOne>(0);
        const auto mu_grad = variable_list.template get_gradient<Scalar, OldOne>(1);

        const auto dt = ScalarValue(sim_timer.get_timestep());

        variable_list.set_value_term(0, phi);
        variable_list.set_gradient_term(0, -dt / epsilon * M(phi) * mu_grad);
      }
    else if (solve_block_id == 1) // mu
      {
        const auto phi      = variable_list.template get_value<Scalar, Current>(0);
        const auto phi_grad = variable_list.template get_gradient<Scalar, Current>(0);

        const auto g_       = g(phi);
        const auto g_prime_ = g_prime(phi);
        const auto B_prime_ = B_prime(phi);

        const auto term_1 = B_prime_ / (epsilon * g_);
        const auto term_2 = epsilon * phi_grad / g_;
        const auto term_3 = -epsilon * phi_grad.norm_square() * g_prime_ / (g_ * g_);

        variable_list.set_value_term(1, term_1 + term_3);
        variable_list.set_gradient_term(1, term_2);
      }
  }

  ScalarValue epsilon;
  ScalarValue reg = 1.0e-3;
};

PRISMS_PF_END_NAMESPACE
