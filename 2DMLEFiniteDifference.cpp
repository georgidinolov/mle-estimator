#include <vector>
#include <math.h>
#include <omp.h>
#include "2DMLEFiniteDifference.hpp"
#include "2DHeatEquationFiniteDifferenceSolver.hpp"

TwoDMLEFiniteDifference::
TwoDMLEFiniteDifference(const std::vector<ContinuousProblemData>& data,
			double sigma_x_0,
			double sigma_y_0,
			double rho_0)
  : data_(data),
    sigma_x_(sigma_x_0),
    sigma_y_(sigma_y_0),
    rho_(rho_0),
    log_likelihood_(0)
{}

double TwoDMLEFiniteDifference::
negative_log_likelihood(int order)
{
  double out = negative_log_likelihood(order,
				       sigma_x_,
				       sigma_y_,
				       rho_);
  return out;
}

// calls parallel likelihood
double TwoDMLEFiniteDifference::
negative_log_likelihood(int order,
			double sigma_x,
			double sigma_y,
			double rho)
{
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;
  rho_ = rho;
  double neg_ll = 0;

  std::vector<double> neg_log_likelihoods (data_.size());
  unsigned i;

  for (i=0; i<data_.size(); ++i) {
    const ContinuousProblemData& datum = data_[i];
    std::cout << datum << std::endl;
    double x_0 = data_[i].get_x_0();
    double y_0 = data_[i].get_y_0();
    
    double a = data_[i].get_a() - x_0;
    double b = data_[i].get_b() - x_0;
    double c = data_[i].get_c() - y_0;
    double d = data_[i].get_d() - y_0;
    double t = data_[i].get_t();

    double x = data_[i].get_x_T() - x_0;
    double y = data_[i].get_y_T() - y_0;

    TwoDHeatEquationFiniteDifferenceSolver solver = 
      TwoDHeatEquationFiniteDifferenceSolver(order,
					     rho,
					     sigma_x,
					     sigma_y,
					     a,b,
					     c,d,
					     x,y,
					     t);
    neg_log_likelihoods[i] = -log(solver.likelihood());
  }

  for (unsigned i=0; i<data_.size(); ++i) {
    neg_ll = neg_ll + neg_log_likelihoods[i];
  }

  return neg_ll;
}

double TwoDMLEFiniteDifference::
negative_log_likelihood_parallel(int order,
				 const std::vector<ContinuousProblemData>& data,
				 double sigma_x,
				 double sigma_y,
				 double rho)
{
  double neg_ll = 0;

  std::vector<double> neg_log_likelihoods (data.size());
  unsigned i;
  unsigned size = data.size();

  omp_set_dynamic(0);
  printf("Master construct is executed by thread %d\n",
	 omp_get_thread_num());
  
  printf("There are %d threads\n",
	 omp_get_max_threads());
  
#pragma omp parallel for private(i) shared(data, neg_log_likelihoods)
  for (i=0; i<size; ++i) {
    const ContinuousProblemData& datum = data[i];
    double x_0 = datum.get_x_0();
    double y_0 = datum.get_y_0();
    
    double a = datum.get_a() - x_0;
    double b = datum.get_b() - x_0;
    double c = datum.get_c() - y_0;
    double d = datum.get_d() - y_0;
    double t = datum.get_t();
    
    double x = datum.get_x_T() - x_0;
    double y = datum.get_y_T() - y_0;
    
    TwoDHeatEquationFiniteDifferenceSolver solver = 
	TwoDHeatEquationFiniteDifferenceSolver(order,
					       rho,
					       sigma_x,
					       sigma_y,
					       a,b,
					       c,d,
					       x,y,
					       t);
      neg_log_likelihoods[i] = -log(solver.likelihood());
  }

  for (unsigned i=0; i<data.size(); ++i) {
    neg_ll = neg_ll + neg_log_likelihoods[i];
  }

  return neg_ll;
}

std::vector<double> TwoDMLEFiniteDifference::
find_mle(int order,
	 double sigma_x,
	 double sigma_y,
	 double rho)
{
  order_ = order;
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;
  rho_ = rho;

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 3);

  std::cout << "in find mle " << std::endl;
  opt.set_ftol_rel(1e-2);
  printf("rel tol = %.16e\n",  opt.get_ftol_rel());
  std::cout << "relative tolerance = " << opt.get_ftol_rel()
	    << std::endl;

  std::vector<double> log_sigma_x_sigma_y_rho = {log(sigma_x), 
						 log(sigma_y), 
						 rho};
  
  std::vector<double> lb = {-HUGE_VAL, -HUGE_VAL,-0.95};
  std::vector<double> ub = {HUGE_VAL, HUGE_VAL,0.95};

  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;  
  opt.set_min_objective(TwoDMLEFiniteDifference::wrapper, 
			this);
  opt.optimize(log_sigma_x_sigma_y_rho, minf);

  return log_sigma_x_sigma_y_rho;
}

double TwoDMLEFiniteDifference::
operator()(const std::vector<double> &x, 
	   std::vector<double> &grad)
{
  return neg_ll_for_optimizer(x,grad);
}

double TwoDMLEFiniteDifference::
neg_ll_for_optimizer(const std::vector<double> &x, 
		     std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }
  
  double sigma_x = exp(x[0]);
  double sigma_y = exp(x[1]);
  double rho = x[2];
  
  return negative_log_likelihood_parallel(order_,
					  data_,
					  sigma_x,
					  sigma_y,
					  rho) / (sigma_x * sigma_y);
}

double TwoDMLEFiniteDifference::
wrapper(const std::vector<double> &x, 
	std::vector<double> &grad,
	void * data)
{
  std::cout << "trying sigma_x=" << exp(x[0]) << " ";
  std::cout << "sigma_y=" << exp(x[1]) << " ";
  std::cout << "rho=" << x[2] << std::endl;
  TwoDMLEFiniteDifference * mle = reinterpret_cast<TwoDMLEFiniteDifference*>(data);
  double out = mle->operator()(x,grad);
  std::cout << "neg log-likelihood = " << out << std::endl;
  return out;
}
