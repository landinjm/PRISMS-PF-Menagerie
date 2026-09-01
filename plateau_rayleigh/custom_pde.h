// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <random>

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
  {}

  static constexpr number M = 1.0;
  static constexpr number K = 0.2;
  static constexpr number W = 0.4;

  const number delta = 2.0 * std::sqrt(2.0 * K / W);

  static constexpr number R_0 = 5.0;
  static constexpr number A_k = 1.0;
  static constexpr number k   = 0.7 / R_0;

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    using std::abs;
    using std::cos;
    using std::tanh;

    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;

    if (index == 0)
      {
        number sdf = 0.0;
        if constexpr (dim == 3)
          {
            const auto x = point[0];
            const auto y = point[1];
            const auto z = point[2];

            const auto R = R_0 + A_k * cos(k * x) + A_k * cos(k * z);

            const auto y_center = y - 0.5 * mesh_size[1];
            const auto z_center = z - 0.5 * mesh_size[2];

            sdf = sqrt(y_center * y_center) - R;
          }
        scalar_value = 0.5 * (1.0 - tanh(sdf / delta));
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    if (solve_block_id == 0)
      {
        ScalarValue c       = variable_list.template get_value<Scalar, OldOne>(0);
        ScalarGrad  mu_grad = variable_list.template get_gradient<Scalar, OldOne>(1);

        variable_list.set_value_term(0, c);
        variable_list.set_gradient_term(0, -M * sim_timer.get_timestep() * mu_grad);
      }
    else if (solve_block_id == 1)
      {
        ScalarValue c      = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  c_grad = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue f_prime = W * c * (c - 1.0) * (c - 0.5);

        variable_list.set_value_term(1, f_prime);
        variable_list.set_gradient_term(1, K * c_grad);
      }
    else if (solve_block_id == 2)
      {
        ScalarValue c      = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  c_grad = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue F_chem = 0.25 * W * c * c * (1.0 - c) * (1.0 - c);
        ScalarValue F_grad = 0.5 * K * c_grad.norm_square();
        variable_list.set_value_term(2, F_chem + F_grad);
      }
  }
};

PRISMS_PF_END_NAMESPACE
