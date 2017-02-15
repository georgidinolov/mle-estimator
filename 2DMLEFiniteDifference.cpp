#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <string>
#include "2DMLEFiniteDifference.hpp"
#include "2DHeatEquationFiniteDifferenceSolver.hpp"

namespace{
  inline double logit(double r) {
    return log(r/(1-r));
  }

  inline double logit_2(double rho) {
    return logit((rho+1)/2);
  }

  inline double logit_inv(double alpha) {
    return exp(alpha)/(exp(alpha)+1);
  }

  inline double logit_2_inv(double alpha) {
    return 2 * logit_inv(alpha) - 1;
  }
}

TwoDMLEFiniteDifference::
TwoDMLEFiniteDifference(std::string data_file_dir,
			double sigma_x_0,
			double sigma_y_0,
			double rho_0)
  : data_(std::vector<ContinuousProblemData> (0)),
    sigma_x_(sigma_x_0),
    sigma_y_(sigma_y_0),
    rho_(rho_0),
    log_likelihood_(0)
{
  std::cout << data_file_dir << std::endl;
  std::ifstream data_file (data_file_dir);
  std::string value;

  double x_0;
  double y_0;
  double t;
  double a;
  double x_T;
  double b;
  double c;
  double y_T;
  double d;

  if (data_file.is_open()) {
    std::cout << "FILE IS OPEN" << std::endl;
    // go through header
    for (int i=0; i<12; ++i) {
      if (i < 11) {
	std::getline(data_file, value, ',');
      } else {
	std::getline(data_file, value);
      }
    }

    // first value on the row is sigma_x
    while (std::getline(data_file, value, ',')) {
      double sigma_x = std::stod(value);

      // second value on the row is sigma_y
      std::getline(data_file, value, ',');
      double sigma_y = std::stod(value);

      // third value on the row is rho
      std::getline(data_file, value, ',');
      double rho = std::stod(value);

      // fourth value on the row is x_0
      std::getline(data_file, value, ',');
      x_0 = std::stod(value);

      // fifth value on the row is y_0
      std::getline(data_file, value, ',');
      y_0 = std::stod(value);

      // 6th  value on the row is t
      std::getline(data_file, value, ',');
      t = std::stod(value);

      // 7th value on the row is a
      std::getline(data_file, value, ',');
      a = std::stod(value);

      // 8th value on the row is x_T
      std::getline(data_file, value, ',');
      x_T = std::stod(value);

      // 9th value on the row is b
      std::getline(data_file, value, ',');
      b = std::stod(value);

      // 10th value on the row is c
      std::getline(data_file, value, ',');
      c = std::stod(value);

      // 11th value on the row is x_T
      std::getline(data_file, value, ',');
      y_T = std::stod(value);

      // 12th value on the row is x_T
      // also the last value in this row
      std::getline(data_file, value);
      d = std::stod(value);

      std::cout << "(" << x_T << "," << y_T << ","
		<< x_0 << "," << y_0 << "," << t
		<< a << "," << b << "," << c << "," << d
		<< ")" << std::endl;
      ContinuousProblemData datum = ContinuousProblemData(x_T,
							  y_T,
							  x_0,
							  y_0,
							  t,
							  a,b,c,d);
      data_.push_back(datum);
    }
  }
}

TwoDMLEFiniteDifference::
TwoDMLEFiniteDifference(const std::vector<ContinuousProblemData> data,
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
  const std::vector<ContinuousProblemData>& data = data_;
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;
  rho_ = rho;
  double neg_ll = 0;

  std::vector<double> neg_log_likelihoods (data.size());
  unsigned i;

  for (i=0; i<data.size(); ++i) {
    std::cout << "on data point " << i << std::endl;
    std::cout << data[i] << std::endl;
    
    double x_0 = data[i].get_x_0();
    double y_0 = data[i].get_y_0();
    
    double a = data[i].get_a() - x_0;
    double b = data[i].get_b() - x_0;
    double c = data[i].get_c() - y_0;
    double d = data[i].get_d() - y_0;
    double t = data[i].get_t();

    double x = data[i].get_x_T() - x_0;
    double y = data[i].get_y_T() - y_0;

    TwoDHeatEquationFiniteDifferenceSolver solver = 
      TwoDHeatEquationFiniteDifferenceSolver(order,
					     rho,
					     sigma_x,
					     sigma_y,
					     a,b,
					     c,d,
					     x,y,
					     t);
    const BoundaryIndeces& bi = solver.get_boundary_indeces();
    std::cout << bi << std::endl;
    
    double l = solver.likelihood();
    int order_current = order;
    while (std::signbit(l)) {
     double l_current = l;
     solver.set_order(2*order_current);
     l = solver.likelihood();
     std::cout << "SIGN NEGATIVE: current l ="
	       << l_current << " with new l =" << l
	       << " order = " << 2*order_current << std::endl;
     order_current = 2*order_current;
    }
    
    neg_log_likelihoods[i] = -log(l);
  }

  neg_ll = 0;
  for (unsigned i=0; i<data_.size(); ++i) {
    std::cout << "neg_log_likelihoods[" << i
	      << "] = "
	      << neg_log_likelihoods[i]
	      << std::endl;
    neg_ll = neg_ll + neg_log_likelihoods[i];
  }

  return neg_ll;
}

double TwoDMLEFiniteDifference::
negative_log_likelihood_parallel(int order,
				 double sigma_x,
				 double sigma_y,
				 double rho) const
{
  // const std::vector<ContinuousProblemData>& data = data_;
   std::vector<double> neg_log_likelihoods (data_.size());

   //   TwoDHeatEquationFiniteDifferenceSolver solver;
   std::vector<TwoDHeatEquationFiniteDifferenceSolver> solvers (0);
   double l = 0;
  
   for (unsigned i=0; i<data_.size(); ++i) {
     solvers.
       push_back(TwoDHeatEquationFiniteDifferenceSolver(order,
							rho,
							sigma_x,
							sigma_y,
							data_[i].get_a()-data_[i].get_x_0(),
							data_[i].get_b()-data_[i].get_x_0(),
							data_[i].get_c()-data_[i].get_y_0(),
							data_[i].get_d()-data_[i].get_y_0(),
							data_[i].get_x_T()-data_[i].get_x_0(),
							data_[i].get_y_T()-data_[i].get_y_0(),
							data_[i].get_t()));
   }

  omp_set_dynamic(0);
  unsigned i;
  
#pragma omp parallel private(i,l) shared(solvers, neg_log_likelihoods,sigma_x,sigma_y,rho)
  {
#pragma omp for 
    for (i=0; i<data_.size(); ++i) {

      l = solvers[i].likelihood();

      //      int order_current = order;
      if (std::signbit(l)) {
	// double old_l = l;
	// order_current = order_current * 2;
	// solvers[i].set_order(order_current);
	// l = solvers[i].likelihood();
	std::cout << "SIGN NEGATIVE: current l =" << l << std::endl;
	l = 1;
      }
      neg_log_likelihoods[i] = -log(l);
    }
  }

  double neg_ll = 0;
  for (unsigned i=0; i<data_.size(); ++i) {
    neg_ll = neg_ll + neg_log_likelihoods[i];
  }

  return neg_ll;
}

std::vector<double> TwoDMLEFiniteDifference::
find_mle(int order,
	 double rel_tol,
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
  opt.set_ftol_rel(rel_tol);
  printf("rel tol = %.16e\n",  opt.get_ftol_rel());
  std::cout << "relative tolerance = " << opt.get_ftol_rel()
	    << std::endl;

  std::vector<double> log_sigma_x_sigma_y_rho = {log(sigma_x), 
						 log(sigma_y), 
						 logit_2(rho)};
  
  std::vector<double> lb = {-HUGE_VAL, -HUGE_VAL,-HUGE_VAL};
  std::vector<double> ub = {HUGE_VAL, HUGE_VAL,HUGE_VAL};

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
  double rho = logit_2_inv(x[2]);
  
  return negative_log_likelihood_parallel(order_,
					  sigma_x,
					  sigma_y,
					  rho)
    - x[0] - x[1] - (log(2) + x[2] - 2*log(exp(x[2])+1));
}

double TwoDMLEFiniteDifference::
wrapper(const std::vector<double> &x, 
	std::vector<double> &grad,
	void * data)
{
  printf("Trying sigma_x=%.32f sigma_y=%.32f rho=%.32f\n", 
	 exp(x[0]), 
	 exp(x[1]), 
	 logit_2_inv(x[2]));
  std::cout << "trying sigma_x=" << exp(x[0]) << " ";
  std::cout << "sigma_y=" << exp(x[1]) << " ";
  std::cout << "rho=" << logit_2_inv(x[2]) << std::endl;
  TwoDMLEFiniteDifference * mle = reinterpret_cast<TwoDMLEFiniteDifference*>(data);
  double out = mle->operator()(x,grad);
  std::cout << "neg log-likelihood = " << out << std::endl;
  return out;
}
