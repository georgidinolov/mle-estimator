#include <vector>
#include <math.h>
#include "2DMLEMethodOfImages.hpp"
#include "1DAdvectionDiffusionSolverImages.hpp"
#include "nlopt.hpp"

TwoDMLEMethodOfImages::
TwoDMLEMethodOfImages(const std::vector<ContinuousProblemData>& data,
		      double sigma_x_0,
		      double sigma_y_0)
  : data_(data),
    sigma_x_(sigma_x_0),
    sigma_y_(sigma_y_0),
    log_likelihood_(0)
{}

double TwoDMLEMethodOfImages::
negative_log_likelihood(int order)
{
  double out = negative_log_likelihood(order,
				       sigma_x_,
				       sigma_y_);
  return out;
}

double TwoDMLEMethodOfImages::
negative_log_likelihood(int order,
			double sigma_x,
			double sigma_y)
{
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;
  double neg_ll = 0;

  for (unsigned i=0; i<data_.size(); ++i) {
    double x_0 = data_[i].get_x_0();
    double y_0 = data_[i].get_y_0();

    double a = data_[i].get_a() - x_0;
    double b = data_[i].get_b() - x_0;
    double c = data_[i].get_c() - y_0;
    double d = data_[i].get_d() - y_0;
    double t = data_[i].get_t();

    double x = data_[i].get_x_T() - x_0;
    double y = data_[i].get_y_T() - y_0;

    OneDAdvectionDiffusionSolverImages x_solver = 
      OneDAdvectionDiffusionSolverImages(0,
					 sigma_x,
					 order,
					 0,
					 a,
					 b);

    OneDAdvectionDiffusionSolverImages y_solver = 
      OneDAdvectionDiffusionSolverImages(0,
					 sigma_y,
					 order,
					 0,
					 c,
					 d);
    neg_ll = neg_ll 
      - log(x_solver.likelihood(t,x))
      - log(y_solver.likelihood(t,y));

  }
  return neg_ll;
}

std::vector<double> TwoDMLEMethodOfImages::
find_mle(int order,
	 double sigma_x,
	 double sigma_y)
{
  order_ = order;
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 2);
  std::vector<double> log_sigma_x_sigma_y = {log(sigma_x), log(sigma_y)};
  
  std::vector<double> lb = {-HUGE_VAL, -HUGE_VAL};
  std::vector<double> ub = {HUGE_VAL, HUGE_VAL};

  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;  
  opt.set_min_objective(TwoDMLEMethodOfImages::wrapper, 
			this);
  opt.optimize(log_sigma_x_sigma_y, minf);


  minf = this->operator()(log_sigma_x_sigma_y, log_sigma_x_sigma_y);
  std::cout << minf << std::endl;
  
  return log_sigma_x_sigma_y;
}

double TwoDMLEMethodOfImages::
operator()(const std::vector<double> &x, 
	   std::vector<double> &grad)
{
  return neg_ll_for_optimizer(x,grad);
}

double TwoDMLEMethodOfImages::
neg_ll_for_optimizer(const std::vector<double> &x, 
		     std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }
  
  double sigma_x = exp(x[0]);
  double sigma_y = exp(x[1]);
  
  return negative_log_likelihood(order_,
				 sigma_x,
				 sigma_y) / (sigma_x * sigma_y);
}

double TwoDMLEMethodOfImages::
wrapper(const std::vector<double> &x, 
	std::vector<double> &grad,
	void * data)
{
  TwoDMLEMethodOfImages * mle = reinterpret_cast<TwoDMLEMethodOfImages*>(data);
  double out = mle->operator()(x,grad);
  std::cout << "neg log-likelihood = " << out << std::endl;
  return out;
}

