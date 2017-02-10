#include <vector>
#include "../finite-difference-arpack-version-2/PDEDataTypes.hpp"

class TwoDMLEMethodOfImages
{
public:
  TwoDMLEMethodOfImages(const std::vector<ContinuousProblemData>& data,
			double sigma_x_0,
			double sigma_y_0);

  double negative_log_likelihood(int order);

  double negative_log_likelihood(int order,
				 double sigma_x,
				 double sigma_y);

  std::vector<double> find_mle(int order,
			       double sigma_x,
			       double sigma_y);

private:
  const std::vector<ContinuousProblemData>& data_;
  double sigma_x_;
  double sigma_y_;
  double rho_;
  int order_;
  double log_likelihood_;

  double operator()(const std::vector<double> &x, 
		    std::vector<double> &grad);

  double neg_ll_for_optimizer(const std::vector<double> &x, 
			      std::vector<double> &grad);

  static double wrapper(const std::vector<double> &x, 
			std::vector<double> &grad,
			void * data);
};
