#include <vector>
#include "nlopt.hpp"

class ContinuousProblemData;

class TwoDMLEFiniteDifference
{
public:
  TwoDMLEFiniteDifference(std::string data_file_dir,
			  double sigma_x_0,
  			  double sigma_y_0,
  			  double rho_0);
  
  TwoDMLEFiniteDifference(std::vector<ContinuousProblemData> data,
			  double sigma_x_0,
			  double sigma_y_0,
			  double rho_0);

  double negative_log_likelihood(int order);

  double negative_log_likelihood(int order,
				 double sigma_x,
				 double sigma_y,
				 double rho);

  double negative_log_likelihood_parallel(int order,
					  double sigma_x,
					  double sigma_y,
					  double rho) const;

  std::vector<double> negative_log_likelihoods_parallel(int order,
							double sigma_x,
							double sigma_y,
							double rho) const;

  std::vector<double> solutions_parallel(int order,
					 double sigma_x,
					 double sigma_y,
					 double rho) const;

  std::vector<ContinuousProblemData> quantized_continuous_data(int order,
  							       double sigma_x,
  							       double sigma_y,
  							       double rho) const;

  std::vector<double> find_mle(int order,
			       double rel_tol,
			       double sigma_x,
			       double sigma_y,
			       double rho);
private:
  std::vector<ContinuousProblemData> data_;
  double sigma_x_;
  double sigma_y_;
  double rho_;
  double log_likelihood_;
  int order_;

  double operator()(const std::vector<double> &x, 
		    std::vector<double> &grad);

  double neg_ll_for_optimizer(const std::vector<double> &x, 
			      std::vector<double> &grad);

  static double wrapper(const std::vector<double> &x, 
			std::vector<double> &grad,
			void * data);
};
