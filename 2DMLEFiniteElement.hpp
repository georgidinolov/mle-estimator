#include <vector>
#include "nlopt.hpp"
#include "PDEDataTypes.hpp"
#include "BivariateSolver.hpp"

class ContinuousProblemData;

class TwoDMLEFiniteElement
{
public:
  TwoDMLEFiniteElement();
  TwoDMLEFiniteElement(std::string data_file_dir,
		       double rho_0,
		       double sigma_x_0,
		       double sigma_y_0,
		       double dx,
		       double rho_for_bases,
		       double sigma_for_bases,
		       double power_for_mollifier,
		       double fraction_for_separation,
		       unsigned number_threads);

  TwoDMLEFiniteElement(std::string data_file_dir,
		       double rho_0,
		       double sigma_x_0,
		       double sigma_y_0,
		       double dx,
		       double rho_for_bases,
		       double sigma_x_for_bases,
		       double sigma_y_for_bases,
		       double power_for_mollifier,
		       double fraction_for_separation,
		       unsigned number_threads);

  TwoDMLEFiniteElement(std::string data_file_dir,
		       double sigma_x_0,
		       double sigma_y_0,
		       double rho_0,
		       double rho_min,
		       double rho_max,
		       unsigned n_rhos,
		       double dx,
		       double sigma_for_bases,
		       double power_for_mollifier,
		       double fraction_for_separation);
  TwoDMLEFiniteElement(const TwoDMLEFiniteElement& rhs);  
  ~TwoDMLEFiniteElement();
  
  void set_data_file(std::string data_file_dir);
  
  // TwoDMLEFiniteElement(std::vector<ContinuousProblemData> data,
  // 		       double sigma_x_0,
  // 		       double sigma_y_0,
  // 		       double rho_0);

  // double negative_log_likelihood(int order);

  // double negative_log_likelihood(int order,
  // 				 double sigma_x,
  // 				 double sigma_y,
  // 				 double rho);

  double negative_log_likelihood_parallel(int order,
  					  double sigma_x,
  					  double sigma_y,
  					  double rho);

  // std::vector<double> negative_log_likelihoods_parallel(int order,
  // 							double sigma_x,
  // 							double sigma_y,
  // 							double rho) const;

  // std::vector<double> solutions_parallel(int order,
  // 					 double sigma_x,
  // 					 double sigma_y,
  // 					 double rho) const;

  // std::vector<ContinuousProblemData> quantized_continuous_data(int order,
  // 							       double sigma_x,
  // 							       double sigma_y,
  // 							       double rho) const;

  std::vector<double> find_mle(int order,
  			       double rel_tol,
  			       double sigma_x,
  			       double sigma_y,
  			       double rho);

  inline const std::vector<int>& get_number_negaive_lls_per_iteration() const 
  {
    return number_negaive_lls_per_iteration_;
  }

private:
  std::vector<BivariateGaussianKernelBasis> bases_;
  static std::vector<BivariateGaussianKernelBasis>* private_bases_;
  std::vector<double> basis_rhos_;
  std::vector<int> number_negaive_lls_per_iteration_;

  std::vector<ContinuousProblemData> data_;
  double sigma_x_;
  double sigma_y_;
  double rho_;
  double log_likelihood_;
  int order_;
  double dx_;

  double operator()(const std::vector<double> &x, 
  		    std::vector<double> &grad);

  double neg_ll_for_optimizer(const std::vector<double> &x, 
  			      std::vector<double> &grad);

  static double wrapper(const std::vector<double> &x, 
  			std::vector<double> &grad,
  			void * data);
};
