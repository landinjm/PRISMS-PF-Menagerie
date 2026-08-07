// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>

#include <prismspf/core/pde_operator_base.h>

#include <prismspf/utilities/symmetry.h>

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
  explicit CustomPDE(const UserInputParameters<dim> &_user_inputs,
                     PhaseFieldTools<dim>           &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
  {}

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    using std::cos;
    using std::sqrt;
    using std::tanh;

    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;

    if (index == 0)
      {
        scalar_value = u_0;
      }
    else if (index == 1)
      {
        const auto L_x = mesh_size[0];
        const auto x   = point[0];
        const auto y   = point[1];

        const auto y_interface =
          y_interface_0 +
          0.5 * A * (1.0 + cos(wavenumber * std::numbers::pi * (x - L_x) / L_x));

        scalar_value = -tanh((y - y_interface) / sqrt(2.0));
      }
  }

  void
  compute_rhs(FieldContainer<dim, degree, number> &variable_list,
              const SimulationTimer               &sim_timer,
              unsigned int                         solve_block_id) const override

  {
    using dealii::Utilities::fixed_power;
    using std::sqrt;

    const double dt = sim_timer.get_timestep();

    if (solve_block_id == 0)
      {
        const auto u        = variable_list.template get_value<Scalar, OldOne>(0);
        const auto u_grad   = variable_list.template get_gradient<Scalar, OldOne>(0);
        const auto phi      = variable_list.template get_value<Scalar, OldOne>(1);
        const auto phi_grad = variable_list.template get_gradient<Scalar, OldOne>(1);
        const auto xi       = variable_list.template get_value<Scalar, OldOne>(2);

        const auto n = phi_grad / (phi_grad.norm() + reg);

        const auto a_n = 1.0 + epsilon * Symmetry::cos_theta<4>(n[0], n[1]);

        const auto tau_phi = (1.0 + (1.0 - k) * u) * a_n * a_n;
        const auto tau_u   = ((1.0 + k) - (1.0 - k) * phi) / 2.0;

        const auto q_phi = (1.0 - phi) / 2.0;

        const auto j_at =
          (1.0 / (2.0 * std::numbers::sqrt2)) * (1.0 + (1.0 - k) * u) * xi * n / tau_phi;

        const auto val_term1 = dt * (1.0 + (1.0 - k) * u) * xi / (2.0 * tau_phi * tau_u);
        const auto val_term2 =
          dt * ((1.0 - k) / 2.0) / (tau_u * tau_u) *
          (phi_grad[0] * (D_tilde * ((1.0 - phi) / 2.0) * u_grad[0] + j_at[0]) +
           phi_grad[1] * (D_tilde * ((1.0 - phi) / 2.0) * u_grad[1] + j_at[1]));

        const auto eq_u = u + val_term1 - val_term2;

        const auto eqx_u =
          -1.0 * dt * (D_tilde * ((1.0 - phi) / 2.0) * u_grad + j_at) / tau_u;

        const auto eq_phi = phi + dt * xi / tau_phi;

        variable_list.set_value_term(0, eq_u);
        variable_list.set_gradient_term(0, eqx_u);

        variable_list.set_value_term(1, eq_phi);
      }
    else if (solve_block_id == 1)
      {
        const auto u        = variable_list.template get_value<Scalar, Current>(0);
        const auto phi      = variable_list.template get_value<Scalar, Current>(1);
        const auto phi_grad = variable_list.template get_gradient<Scalar, Current>(1);

        const auto n = phi_grad / (phi_grad.norm() + reg);

        const auto a_n   = 1.0 + epsilon * Symmetry::cos_theta<4>(n[0], n[1]);
        const auto d_a_n = -4.0 * epsilon * Symmetry::sin_theta<4>(n[0], n[1]);

        const auto  y   = variable_list.get_q_point_location()[1];
        ScalarValue t   = sim_timer.get_time();
        const auto  tep = (y - y_0 - V_tilde * t) / l_tilde;

        ScalarGrad aniso;
        aniso[0] = a_n * a_n * phi_grad[0] - a_n * d_a_n * phi_grad[1];
        aniso[1] = a_n * a_n * phi_grad[1] + a_n * d_a_n * phi_grad[0];

        const auto eq_xi =
          phi - (phi * phi * phi) -
          (lambda * (1.0 - phi * phi) * (1.0 - phi * phi) * (u + tep + u_offset));

        const auto eqx_xi = -aniso;

        variable_list.set_value_term(2, eq_xi);
        variable_list.set_gradient_term(2, eqx_xi);
      }
  }

  constexpr static number epsilon       = 0.01;
  constexpr static number k             = 0.1;
  constexpr static number c_0           = 0.46;
  constexpr static number lambda        = 97.581;
  constexpr static number D_tilde       = 61.151;
  constexpr static number V_tilde       = 1.718;
  constexpr static number l_tilde       = 3725.73;
  constexpr static number u_0           = -1.0;
  constexpr static number u_offset      = 0.9;
  constexpr static number y_0           = 5.0;
  constexpr static number reg           = 1.0e-10;
  constexpr static number A             = 12.1905;
  constexpr static number wavenumber    = 6.0;
  constexpr static number y_interface_0 = 379.28;
};

PRISMS_PF_END_NAMESPACE
