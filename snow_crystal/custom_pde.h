// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>

#include <prismspf/core/pde_operator_base.h>

#include <prismspf/utilities/crystal_symmetry.h>

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
    , u_0(get_user_inputs().user_constants.get_double("u_0"))
    , L_sat(get_user_inputs().user_constants.get_double("L_sat"))
    , epsilon_xy(get_user_inputs().user_constants.get_double("epsilon_xy"))
    , epsilon_z(get_user_inputs().user_constants.get_double("epsilon_z"))
    , Gamma(get_user_inputs().user_constants.get_double("Gamma"))
    , lambda(get_user_inputs().user_constants.get_double("lambda"))
    , D_tilde(number(0.6267) * lambda)
    , reg_val(get_user_inputs().user_constants.get_double("reg_val"))
  {}

private:
  /**
   * Here we're going to define some functions that the Demange et al. paper uses.
   * Importantly, we use `DEAL_II_ALWAYS_INLINE` so they get inlined in the code while
   * making it more readable. We also use `dealii::Utilities::fixed_power` to expand
   * integer powers at compile time for maximum performance.
   */
  DEAL_II_ALWAYS_INLINE inline ScalarValue
  f(ScalarValue phi) const
  {
    using dealii::Utilities::fixed_power;
    return -fixed_power<2>(phi) / number(2) + fixed_power<4>(phi) / number(4);
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  f_prime(ScalarValue phi) const
  {
    using dealii::Utilities::fixed_power;
    return -phi + fixed_power<3>(phi);
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  g_prime(ScalarValue phi) const
  {
    using dealii::Utilities::fixed_power;
    return fixed_power<2>(number(1) - fixed_power<2>(phi));
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  q(ScalarValue phi) const
  {
    return number(1) - phi;
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  A(ScalarGrad n) const
  {
    using Symmetries::cos_psi;
    using Symmetries::cos_theta;
    return number(1) + epsilon_xy * cos_theta<6>(n[0], n[1]) +
           epsilon_z * cos_psi<2>(n[0], n[1], n[2]);
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  B(ScalarGrad n) const
  {
    using dealii::Utilities::fixed_power;
    using std::sqrt;
    return sqrt(fixed_power<2>(n[0]) + fixed_power<2>(n[1]) +
                Gamma * fixed_power<2>(n[2]));
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  d_A_d_theta(ScalarGrad n) const
  {
    using Symmetries::sin_theta;
    return -epsilon_xy * number(6) * sin_theta<6>(n[0], n[1]);
  }

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  d_A_d_psi(ScalarGrad n) const
  {
    using Symmetries::sin_psi;
    return -epsilon_z * number(2) * sin_psi<2>(n[0], n[1], n[2]);
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
        // For the supersaturation field, we just set the initial condition
        // to have uniform undercooling everywhere.
        scalar_value = u_0;
      }
    else if (index == 1)
      {
        // For the order parameter, we just place a small seed. Note that
        // the order parameter ranges from -1 to 1 in this model.

        // Center at the origin with some radius
        const dealii::Point<dim> center;
        const double             radius = 5.0;

        // Compute the distance
        double distance = point.distance(center);

        // Apply tanh
        scalar_value = -std::tanh((distance - radius) / std::numbers::sqrt2);
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

    // Explicit u and phi evolution
    if (solve_block_id == 0)
      {
        ScalarValue u        = variable_list.template get_value<Scalar, OldOne>(0);
        ScalarGrad  u_grad   = variable_list.template get_gradient<Scalar, OldOne>(0);
        ScalarValue phi      = variable_list.template get_value<Scalar, OldOne>(1);
        ScalarGrad  phi_grad = variable_list.template get_gradient<Scalar, OldOne>(1);
        ScalarValue xi       = variable_list.template get_value<Scalar, OldOne>(2);

        // Compute the interfacial normal vector for the anisotropy function
        ScalarGrad normal = phi_grad / (phi_grad.norm() + reg_val);

        // Energetic anisotropy
        ScalarValue A_n = A(normal);

        // Kinetic anisotropy
        ScalarValue B_n = B(normal);

        // Order parameter evolution
        ScalarValue d_phi_d_t = xi / fixed_power<2>(A_n);

        // The gradient must consider horizontal and vertial growth preference. As such,
        // we multiply by Gamma. We do this twice because there's two gradient operators.
        phi_grad[2] *= Gamma * Gamma;

        variable_list.set_value_term(0, u - dt * L_sat * B_n * d_phi_d_t / number(2));
        variable_list.set_gradient_term(0, -dt * D_tilde * q(phi) * phi_grad);

        variable_list.set_value_term(1, phi + dt * d_phi_d_t);
      }
    // Explicit xi
    else if (solve_block_id == 1)
      {
        ScalarValue u        = variable_list.template get_value<Scalar, Current>(0);
        ScalarValue phi      = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  phi_grad = variable_list.template get_gradient<Scalar, Current>(1);

        // Compute the interfacial normal vector for the anisotropy function
        ScalarValue phi_grad_norm_2 = phi_grad.norm_square();
        ScalarGrad  normal          = phi_grad / (sqrt(phi_grad_norm_2) + reg_val);

        // Energetic anisotropy
        ScalarValue A_n   = A(normal);
        ScalarValue A_n_2 = fixed_power<2>(A_n);

        // Kinetic anisotropy
        ScalarValue B_n = B(normal);

        // The derivative of the energetic anisotropy with respect to the gradient of phi
        // TODO: Might be able to simplify this
        ScalarValue phi_x_2_and_phi_y_2 = phi_grad_norm_2 - fixed_power<2>(phi_grad[2]);
        ScalarValue sqrt_phi_x_2_and_phi_y_2 = sqrt(phi_x_2_and_phi_y_2);

        ScalarGrad d_theta_d_grad_phi;
        d_theta_d_grad_phi[0] = -phi_grad[1] / (phi_x_2_and_phi_y_2 + reg_val);
        d_theta_d_grad_phi[1] = phi_grad[0] / (phi_x_2_and_phi_y_2 + reg_val);
        d_theta_d_grad_phi[2] = number(0);

        ScalarGrad d_psi_d_grad_phi;
        d_psi_d_grad_phi[0] = phi_grad[0] * phi_grad[2] /
                              (sqrt_phi_x_2_and_phi_y_2 * phi_grad_norm_2 + reg_val);
        d_psi_d_grad_phi[1] = phi_grad[1] * phi_grad[2] /
                              (sqrt_phi_x_2_and_phi_y_2 * phi_grad_norm_2 + reg_val);
        d_psi_d_grad_phi[2] = -sqrt_phi_x_2_and_phi_y_2 / (phi_grad_norm_2 + reg_val);

        ScalarGrad d_A_n_d_grad_phi =
          d_A_d_theta(normal) * d_theta_d_grad_phi + d_A_d_psi(normal) * d_psi_d_grad_phi;

        // The gradient must consider horizontal and vertial growth preference. As such,
        // we multiply by Gamma. We do this twice because there's two gradient operators.
        phi_grad[2] *= Gamma * Gamma;

        // The anisotropy term
        // TODO: Should probably refactor this and above into a helper function
        ScalarGrad anisotropy =
          -number(0.5) * A_n_2 * phi_grad - phi_grad_norm_2 * A_n * d_A_n_d_grad_phi;

        variable_list.set_value_term(2, -f_prime(phi) + lambda * B_n * g_prime(phi) * u);
        variable_list.set_gradient_term(2, anisotropy);
      }
  }

  number u_0;
  number L_sat;
  number epsilon_xy;
  number epsilon_z;
  number Gamma;
  number lambda;
  number D_tilde;
  number reg_val;
};

PRISMS_PF_END_NAMESPACE
