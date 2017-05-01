#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <string>
#include "2DMLEFiniteElement.hpp"

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

  inline double round_32(double input) {
    double divisor = 1.0/32.0;
    return divisor * std::round(input/divisor);
  }
}

std::vector<BivariateGaussianKernelBasis>* TwoDMLEFiniteElement::private_bases_;

TwoDMLEFiniteElement::
TwoDMLEFiniteElement()
  : bases_(std::vector<BivariateGaussianKernelBasis> (0)),
    basis_rhos_(std::vector<double> (0)),
    data_(std::vector<ContinuousProblemData> (0)),
    sigma_x_(1),
    sigma_y_(1),
    rho_(0),
    log_likelihood_(0),
    dx_(1.0)
{}

TwoDMLEFiniteElement::
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
		     double fraction_for_separation)
  : bases_(std::vector<BivariateGaussianKernelBasis> (n_rhos)),
    basis_rhos_(std::vector<double> (n_rhos)),
    data_(std::vector<ContinuousProblemData> (0)),
    sigma_x_(sigma_x_0),
    sigma_y_(sigma_y_0),
    rho_(rho_0),
    log_likelihood_(0),
    dx_(dx)
{
  // CREATING BASIS VECTOR START
  double drho = (rho_max-rho_min)/(n_rhos-1);
  std::vector<double>& basis_rhos = basis_rhos_;
  std::vector<BivariateGaussianKernelBasis>& bases = bases_;
#pragma omp parallel for default(shared)
  for (unsigned k=0; k<n_rhos; ++k) {
    double current_rho = rho_min + drho*k;
    printf("on basis %d, with rho = %f\n", k, current_rho);
    if (current_rho <= rho_max && current_rho >= rho_min) {
      basis_rhos[k] = current_rho;
    } else if (current_rho > rho_max) {
      basis_rhos[k] = rho_max;
    } else if (current_rho < rho_min) {
      basis_rhos[k] = rho_min;
    } else {
      basis_rhos[k] = rho_min + (rho_max-rho_min)/2.0;
    }
    
    bases[k] =  BivariateGaussianKernelBasis(dx, 
					     basis_rhos[k],
					     sigma_for_bases,
					     power_for_mollifier, 
					     fraction_for_separation);
    printf("done with basis %d, with rho = %f\n", k, basis_rhos[k]);
  }
  // CREATING BASIS VECTOR END

  // THREADPRIVATE BASES START
#pragma omp threadprivate(private_bases_)
  unsigned i;
#pragma omp parallel default(none) private(i) shared(n_rhos, bases)
  {
    private_bases_ = new std::vector<BivariateGaussianKernelBasis> (n_rhos);
    for (i=0; i<n_rhos; ++i) {
      (*private_bases_)[i] = bases[i];
      if (i==0) {
	printf("Thread %d with address %p\n", omp_get_thread_num(), private_bases_);
      }
    }
  }
  // THREADPRIVATE BASES END

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

      std::cout << "(" << a << "," << x_T << "," << b << ") "
		<< "(" << c << "," << y_T << "," << d << ")"
		<< std::endl;
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

TwoDMLEFiniteElement::
TwoDMLEFiniteElement(const TwoDMLEFiniteElement& rhs)
  : bases_(rhs.bases_),
    basis_rhos_(rhs.basis_rhos_),
    data_(rhs.data_),
    sigma_x_(rhs.sigma_x_),
    sigma_y_(rhs.sigma_y_),
    rho_(rhs.rho_),
    log_likelihood_(rhs.log_likelihood_),
    order_(rhs.order_),
    dx_(rhs.dx_)
{
  // THREADPRIVATE BASES START
#pragma omp threadprivate(private_bases_)
  unsigned i;
  const std::vector<BivariateGaussianKernelBasis>& bases = bases_;
#pragma omp parallel default(none) private(i) shared(bases)
  {
    private_bases_ = new std::vector<BivariateGaussianKernelBasis> (bases.size());
    for (i=0; i<bases.size(); ++i) {
      (*private_bases_)[i] = bases[i];
    }
  }
  // THREADPRIVATE BASES END
}

TwoDMLEFiniteElement::~TwoDMLEFiniteElement()
{
#pragma omp parallel default(none)
  {
    delete private_bases_;
  }
}

void TwoDMLEFiniteElement::set_data_file(std::string data_file_dir)
{
  data_ = std::vector<ContinuousProblemData> (0);
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

      std::cout << "(" << a << "," << x_T << "," << b << ") "
		<< "(" << c << "," << y_T << "," << d << ")"
		<< std::endl;
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

// TwoDMLEFiniteElement::
// TwoDMLEFiniteElement(const std::vector<ContinuousProblemData> data,
// 			double sigma_x_0,
// 			double sigma_y_0,
// 			double rho_0)
//   : data_(data),
//     sigma_x_(sigma_x_0),
//     sigma_y_(sigma_y_0),
//     rho_(rho_0),
//     log_likelihood_(0)
// {}

// double TwoDMLEFiniteElement::
// negative_log_likelihood(int order)
// {
//   double out = negative_log_likelihood(order,
// 				       sigma_x_,
// 				       sigma_y_,
// 				       rho_);
//   return out;
// }

// // calls parallel likelihood
// double TwoDMLEFiniteElement::
// negative_log_likelihood(int order,
// 			double sigma_x,
// 			double sigma_y,
// 			double rho)
// {
//   const std::vector<ContinuousProblemData>& data = data_;
//   sigma_x_ = sigma_x;
//   sigma_y_ = sigma_y;
//   rho_ = rho;
//   double neg_ll = 0;

//   std::vector<double> neg_log_likelihoods (data.size());
//   unsigned i;

//   for (i=0; i<data.size(); ++i) {
//     std::cout << "on data point " << i << std::endl;
//     std::cout << data[i] << std::endl;
    
//     double x_0 = data[i].get_x_0();
//     double y_0 = data[i].get_y_0();
    
//     double a = data[i].get_a() - x_0;
//     double b = data[i].get_b() - x_0;
//     double c = data[i].get_c() - y_0;
//     double d = data[i].get_d() - y_0;
//     double t = data[i].get_t();

//     double x = data[i].get_x_T() - x_0;
//     double y = data[i].get_y_T() - y_0;

//     TwoDHeatEquationFiniteElementSolver solver = 
//       TwoDHeatEquationFiniteElementSolver(order,
// 					     rho,
// 					     sigma_x,
// 					     sigma_y,
// 					     a,b,
// 					     c,d,
// 					     x,y,
// 					     t);
//     const BoundaryIndeces& bi = solver.get_boundary_indeces();
//     std::cout << bi << std::endl;
    
//     double l = solver.likelihood();
//     int order_current = order;
//     while (std::signbit(l)) {
//      double l_current = l;
//      solver.set_order(2*order_current);
//      l = solver.likelihood();
//      std::cout << "SIGN NEGATIVE: current l ="
// 	       << l_current << " with new l =" << l
// 	       << " order = " << 2*order_current << std::endl;
//      order_current = 2*order_current;
//     }
    
//     neg_log_likelihoods[i] = -log(l);
//   }

//   neg_ll = 0;
//   for (unsigned i=0; i<data_.size(); ++i) {
//     std::cout << "neg_log_likelihoods[" << i
// 	      << "] = "
// 	      << neg_log_likelihoods[i]
// 	      << std::endl;
//     neg_ll = neg_ll + neg_log_likelihoods[i];
//   }

//   return neg_ll;
// }

double TwoDMLEFiniteElement::
negative_log_likelihood_parallel(int order,
				 double sigma_x,
				 double sigma_y,
				 double rho) const
{
  unsigned i = 0;
  gsl_vector* input;
  unsigned N = data_.size();
  std::vector<double> likelihoods (N);
  std::vector<double> abs_differences (basis_rhos_.size());
  unsigned k = 0;
  const std::vector<double>& basis_rhos = basis_rhos_;
  std::generate(abs_differences.begin(), 
		abs_differences.end(), [&k, &rho, &basis_rhos]
		{ 
		  double out = std::abs(rho - basis_rhos[k]);
		  k++; 
		  return out;
		});
  
  k=0;
  std::vector<double> abs_differences_indeces (basis_rhos_.size());
  std::generate(abs_differences_indeces.begin(),
		abs_differences_indeces.end(),
		[&k]{ return k++; });
  std::sort(abs_differences_indeces.begin(), 
	    abs_differences_indeces.end(),
    	    [&abs_differences] (unsigned i1, unsigned i2) -> bool
    	    {
    	      return abs_differences[i1] < abs_differences[i2];
    	    });

  k = abs_differences_indeces[0];

  // const std::vector<ContinuousProblemData>& data = data_;
  std::vector<double> neg_log_likelihoods (data_.size());
  const std::vector<ContinuousProblemData>& data = data_;
  double dx_likelihood = 1.0/order;
  double dx = dx_;
  double t = 1;

#pragma omp parallel default(none) private(i, input) shared(N, data, neg_log_likelihoods) firstprivate(k, rho, sigma_x, sigma_y, t, dx, dx_likelihood)
  {
    input = gsl_vector_alloc(2);
    
#pragma omp for
    for (i=0; i<N; ++i) {
      if (i==0) { printf("Thread %d with address %p\n", omp_get_thread_num(), private_bases_); }
      BivariateSolver solver = BivariateSolver(&(*private_bases_)[k],
					       sigma_x,
					       sigma_y,
					       rho,
					       data[i].get_a(),
					       data[i].get_x_0(),
					       data[i].get_b(),
					       data[i].get_c(),
					       data[i].get_y_0(),
					       data[i].get_d(),
					       t, 
					       dx);
      gsl_vector_set(input, 0, data[i].get_x_T());
      gsl_vector_set(input, 1, data[i].get_y_T());
      double likelihood = solver.numerical_likelihood(input, 
						      dx_likelihood);
      // printf("Thread %d: likelihood = %f\n", omp_get_thread_num(), likelihood);
      if (likelihood > 0) {
	neg_log_likelihoods[i] = -log(likelihood);
      } else {
	neg_log_likelihoods[i] = -log(1e-16);
      }
    }

    gsl_vector_free(input);
  }

  double sum_of_elements_parallel = 0;
  for (unsigned i=0; i<neg_log_likelihoods.size(); ++i) {
    sum_of_elements_parallel = sum_of_elements_parallel + 
      neg_log_likelihoods[i];
  }

  // std::for_each(neg_log_likelihoods.begin(), neg_log_likelihoods.end(), 
  // 		[&sum_of_elements_parallel] (double lik) {
  //     sum_of_elements_parallel = sum_of_elements_parallel + lik;
  //   });

  return sum_of_elements_parallel;
}


// std::vector<double> TwoDMLEFiniteElement::
// negative_log_likelihoods_parallel(int order,
// 				 double sigma_x,
// 				 double sigma_y,
// 				 double rho) const
// {
//   // const std::vector<ContinuousProblemData>& data = data_;
//    std::vector<double> neg_log_likelihoods (data_.size());

//    //   TwoDHeatEquationFiniteElementSolver solver;
//    std::vector<TwoDHeatEquationFiniteElementSolver> solvers (0);
//    double l = 0;
  
//    for (unsigned i=0; i<data_.size(); ++i) {
//      solvers.
//        push_back(TwoDHeatEquationFiniteElementSolver(order,
// 							rho,
// 							sigma_x,
// 							sigma_y,
// 							data_[i].get_a()-data_[i].get_x_0(),
// 							data_[i].get_b()-data_[i].get_x_0(),
// 							data_[i].get_c()-data_[i].get_y_0(),
// 							data_[i].get_d()-data_[i].get_y_0(),
// 							data_[i].get_x_T()-data_[i].get_x_0(),
// 							data_[i].get_y_T()-data_[i].get_y_0(),
// 							data_[i].get_t()));
//    }

//   omp_set_dynamic(0);
//   unsigned i;
  
// #pragma omp parallel private(i,l) shared(solvers, neg_log_likelihoods,sigma_x,sigma_y,rho)
//   {
// #pragma omp for 
//     for (i=0; i<data_.size(); ++i) {

//       l = solvers[i].likelihood();

//       neg_log_likelihoods[i] = -log(l);
//     }
//   }

//   return neg_log_likelihoods;
// }

// std::vector<ContinuousProblemData> TwoDMLEFiniteElement::
// quantized_continuous_data(int order,
// 			  double sigma_x,
// 			  double sigma_y,
// 			  double rho) const
// {
//   // const std::vector<ContinuousProblemData>& data = data_;
//    std::vector<ContinuousProblemData> output (data_.size());

//    //   TwoDHeatEquationFiniteElementSolver solver;
//    std::vector<TwoDHeatEquationFiniteElementSolver> solvers (0);
  
//    for (unsigned i=0; i<data_.size(); ++i) {
//      solvers.
//        push_back(TwoDHeatEquationFiniteElementSolver(order,
//    							rho,
//    							sigma_x,
//    							sigma_y,
//    							data_[i].get_a()-data_[i].get_x_0(),
//    							data_[i].get_b()-data_[i].get_x_0(),
//    							data_[i].get_c()-data_[i].get_y_0(),
//    							data_[i].get_d()-data_[i].get_y_0(),
//    							data_[i].get_x_T()-data_[i].get_x_0(),
//    							data_[i].get_y_T()-data_[i].get_y_0(),
//    							data_[i].get_t()));
//      output[i] = solvers[i].get_quantized_continuous_data();
//    }

//   return output;
// }

// std::vector<double> TwoDMLEFiniteElement::
// solutions_parallel(int order,
// 		   double sigma_x,
// 		   double sigma_y,
// 		   double rho) const
// {
//   // const std::vector<ContinuousProblemData>& data = data_;
//    std::vector<double> solutions (data_.size());

//    //   TwoDHeatEquationFiniteElementSolver solver;
//    std::vector<TwoDHeatEquationFiniteElementSolver> solvers (0);
//    double l = 0;
  
//    for (unsigned i=0; i<data_.size(); ++i) {
//      solvers.
//        push_back(TwoDHeatEquationFiniteElementSolver(order,
// 							rho,
// 							sigma_x,
// 							sigma_y,
// 							data_[i].get_a()-data_[i].get_x_0(),
// 							data_[i].get_b()-data_[i].get_x_0(),
// 							data_[i].get_c()-data_[i].get_y_0(),
// 							data_[i].get_d()-data_[i].get_y_0(),
// 							data_[i].get_x_T()-data_[i].get_x_0(),
// 							data_[i].get_y_T()-data_[i].get_y_0(),
// 							data_[i].get_t()));
//    }

//   omp_set_dynamic(0);
//   unsigned i;
  
// #pragma omp parallel private(i,l) shared(solvers, solutions,sigma_x,sigma_y,rho)
//   {
// #pragma omp for 
//     for (i=0; i<data_.size(); ++i) {

//       l = solvers[i].solve();

//       solutions[i] = l;
//     }
//   }

//   return solutions;
// }


std::vector<double> TwoDMLEFiniteElement::
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

  // std::vector<double> log_sigma_x_sigma_y_rho = {log(sigma_x), 
  // 						 log(sigma_y), 
  // 						 logit_2(rho)};
  std::vector<double> log_sigma_x_sigma_y_rho = {sigma_x, 
						 sigma_y, 
						 rho};
  
  std::vector<double> lb = {0.0001, 0.0001, -0.999};
  std::vector<double> ub = {HUGE_VAL, HUGE_VAL,0.999};

  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;  
  opt.set_min_objective(TwoDMLEFiniteElement::wrapper, 
			this);
  opt.optimize(log_sigma_x_sigma_y_rho, minf);

  return log_sigma_x_sigma_y_rho;
}

double TwoDMLEFiniteElement::
operator()(const std::vector<double> &x, 
	   std::vector<double> &grad)
{
  return neg_ll_for_optimizer(x,grad);
}

double TwoDMLEFiniteElement::
neg_ll_for_optimizer(const std::vector<double> &x, 
		     std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }
  
  // double sigma_x = exp(x[0]);
  // double sigma_y = exp(x[1]);
  // double rho = logit_2_inv(x[2]);
  double sigma_x = x[0];
  double sigma_y = x[1];
  double rho = x[2];
  
  return (negative_log_likelihood_parallel(order_,
					   sigma_x,
					   sigma_y,
					   rho));
  //	  - x[0] - x[1] - (log(2) + x[2] - 2*log(exp(x[2])+1)));
}

double TwoDMLEFiniteElement::
wrapper(const std::vector<double> &x, 
	std::vector<double> &grad,
	void * data)
{
  printf("Trying sigma_x=%.32f sigma_y=%.32f rho=%.32f\n", 
	 x[0], 
	 x[1], 
	 x[2]);
  std::cout << "trying sigma_x=" << x[0] << " ";
  std::cout << "sigma_y=" << x[1] << " ";
  std::cout << "rho=" << x[2] << std::endl;
  TwoDMLEFiniteElement * mle = reinterpret_cast<TwoDMLEFiniteElement*>(data);
  double out = mle->operator()(x,grad);
  std::cout << "neg log-likelihood = " << out << std::endl;
  return out;
}
