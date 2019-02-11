#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <string>
#include "2DMLEFiniteElement.hpp"
#include "src/gaussian-interpolator/GaussianInterpolator.hpp"

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

  gsl_vector* find_weights(const std::vector<double>& ys,
			   const std::vector<double>& t_tildes,
			   const std::vector<double>& alphas,
			   const std::vector<double>& lambdas) {

    // This is the design matrix X with dimensions m =
    // size(t_tildes) x (n = size(lambdas) x o = size(alphas))
    unsigned oo = alphas.size();
    unsigned nn = lambdas.size();
    unsigned mm = ys.size();

    double y_array [mm];
    double X_array [mm*nn*oo];

    // filling out the design matrix
    gsl_matrix_view X_view = gsl_matrix_view_array(X_array, mm, nn*oo);
    gsl_vector_view y_view = gsl_vector_view_array(y_array, mm);

    for (unsigned ii=0; ii<mm; ++ii) {
      gsl_vector_set(&y_view.vector, ii, exp(log(ys[ii]) -lambdas[0]*t_tildes[ii]) );

      for (unsigned jj=0; jj<nn; ++jj) {

	for (unsigned kk=0; kk<oo; ++kk) {
	  gsl_matrix_set(&X_view.matrix, ii, jj,
			 exp((lambdas[jj] - lambdas[0])*t_tildes[ii]) *
			 std::pow(t_tildes[ii], alphas[kk]));
	}
      }
    }
    const gsl_matrix* X = &X_view.matrix;

    double work_array [(nn*oo)*(nn*oo)];
    gsl_matrix_view work_view = gsl_matrix_view_array(work_array, nn*oo, nn*oo);

    // X^T * X
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, &work_view.matrix);

    // b = X^T y
    double b_array [nn*oo];
    gsl_vector_view b_view = gsl_vector_view_array(b_array, nn*oo);
    gsl_blas_dgemv(CblasTrans, 1.0, X, &y_view.vector, 0.0, &b_view.vector);

    // b = (X^T * X) weights
    gsl_vector* weights = gsl_vector_alloc(nn*oo);
    gsl_permutation* p = gsl_permutation_alloc(nn*oo);
    int s = 0;

    gsl_linalg_LU_decomp(&work_view.matrix, p, &s);
    gsl_linalg_LU_solve(&work_view.matrix, p, &b_view.vector, weights);

    gsl_permutation_free(p);

    return (weights);
  }

  double find_max(const gsl_vector* weights, 
		  const std::vector<double>& lambdas, 
		  const std::vector<double>& alphas,
		  const std::vector<double>& t_tildes,
		  double small_t) {
    // Find max according to the approximating solution start
    double t_min = 0.0;
    double t_max = 0.0;
    double w1 = gsl_vector_get(weights, 0);
    double w2 = gsl_vector_get(weights, 1);
    if ((w1 > 0.0) & (w2 > 0.0)) {
      t_min = small_t;
      t_max = t_tildes[t_tildes.size()-1];
    } else if ((w1 > 0.0) & (w2 < 0.0)) {
      t_min = -1.0*log(std::abs(w1/w2))/std::abs(lambdas[1]-lambdas[0]);
      if (std::signbit(t_min)) {
	t_min = small_t;
      }
      t_max = t_tildes[t_tildes.size()-1];
    } else if ((w1 < 0.0) & (w2 > 0.0)) {
      t_min = small_t;
      t_max = -1.0*log(std::abs(w1/w2))/std::abs(lambdas[1]-lambdas[0]);
    } else {
      t_min = small_t;
      t_max = t_tildes[t_tildes.size()-1];
    }

    std::vector<double> ts(100);
    unsigned nn = 0;
    double dt = (t_max - t_min)/99;
    std::generate(ts.begin(), ts.end(), [&] () mutable { double out = t_min + dt*nn; nn++; return out; });
    std::sort(ts.begin(), ts.end(), [&lambdas, &alphas, &w1, &w2] (double t1, double t2)->bool {
	double Delta = (lambdas[1]-lambdas[0]);
	double d1 = std::abs(lambdas[0] + alphas[0]/t1 + Delta*w2*exp(Delta*t1)/(w1 + w2*exp(Delta*t1)));
	double d2 = std::abs(lambdas[0] + alphas[0]/t2 + Delta*w2*exp(Delta*t2)/(w1 + w2*exp(Delta*t2)));
	return (d1 < d2); });
    t_max = (ts[0] + ts[1])/2.0; // this is the maximum point for the approximate solution
    return t_max;
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
		     double rho_0,
		     double sigma_x_0,
		     double sigma_y_0,
		     double dx,
		     double rho_for_bases,
		     double sigma_for_bases,
		     double power_for_mollifier,
		     double fraction_for_separation,
		     unsigned number_threads)
  : bases_(std::vector<BivariateGaussianKernelBasis> (1)),
    basis_rhos_(std::vector<double> (1)),
    data_(std::vector<ContinuousProblemData> (0)),
    sigma_x_(sigma_x_0),
    sigma_y_(sigma_y_0),
    rho_(rho_0),
    log_likelihood_(0),
    dx_(dx)
{
  basis_rhos_[0] = rho_for_bases;
  // CREATING BASIS VECTOR START
  std::vector<BivariateGaussianKernelBasis>& bases = bases_;
  bases[0] =  BivariateGaussianKernelBasis(dx,
					   rho_for_bases,
					   sigma_for_bases,
					   power_for_mollifier,
					   fraction_for_separation);
  // CREATING BASIS VECTOR END
  omp_set_num_threads(number_threads);

  // THREADPRIVATE BASES START
#pragma omp threadprivate(private_bases_)

#pragma omp parallel default(none) shared(bases)
  {
    private_bases_ = new std::vector<BivariateGaussianKernelBasis> (1);
    (*private_bases_)[0] = bases[0];
    printf("Thread %d with address %p\n", omp_get_thread_num(), private_bases_);
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
      std::cout << "sigma_x=" << sigma_x
		<< "; sigma_y=" << sigma_y
		<< "; rho=" << rho
		<< std::endl;

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
		     unsigned number_threads)
  : bases_(std::vector<BivariateGaussianKernelBasis> (1)),
    basis_rhos_(std::vector<double> (1)),
    data_(std::vector<ContinuousProblemData> (0)),
    sigma_x_(sigma_x_0),
    sigma_y_(sigma_y_0),
    rho_(rho_0),
    log_likelihood_(0),
    dx_(dx)
{
  basis_rhos_[0] = rho_for_bases;

  omp_set_num_threads(number_threads);

  // CREATING BASIS VECTOR START
  std::vector<BivariateGaussianKernelBasis>& bases = bases_;
  bases[0] =  BivariateGaussianKernelBasis(dx,
					   rho_for_bases,
					   sigma_x_for_bases,
					   sigma_y_for_bases,
					   power_for_mollifier,
					   fraction_for_separation);
  // CREATING BASIS VECTOR END

  // THREADPRIVATE BASES START
#pragma omp threadprivate(private_bases_)
#pragma omp parallel default(none) shared(bases)
  {
    private_bases_ = new std::vector<BivariateGaussianKernelBasis> (1);
    (*private_bases_)[0] = bases[0];
    printf("Thread %d with address %p\n", omp_get_thread_num(), private_bases_);
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
      std::cout << "sigma_x=" << sigma_x
		<< "; sigma_y=" << sigma_y
		<< "; rho=" << rho
		<< std::endl;

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
      std::cout << "sigma_x=" << sigma_x
		<< "; sigma_y=" << sigma_y
		<< "; rho=" << rho
		<< std::endl;

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
      std::cout << "sigma_x=" << sigma_x
		<< "; sigma_y=" << sigma_y
		<< "; rho=" << rho
		<< std::endl;

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
				 double rho)
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
  std::vector<int> bad_likelihoods_indicators (data_.size());
  const std::vector<ContinuousProblemData>& data = data_;
  double dx_likelihood = 1.0/order;
  double dx = dx_;
  double t = 1;

#pragma omp parallel default(none) private(i, input) shared(N, data, neg_log_likelihoods, bad_likelihoods_indicators) firstprivate(k, rho, sigma_x, sigma_y, t, dx, dx_likelihood)
  {
    input = gsl_vector_alloc(2);

#pragma omp for
    for (i=0; i<N; ++i) {
      if (i==0) { printf("Thread %d with address %p\n", omp_get_thread_num(), private_bases_); }

      double Lx = data[i].get_b() - data[i].get_a();
      double Ly = data[i].get_d() - data[i].get_c();

      double x_0_tilde = (data[i].get_x_0() - data[i].get_a())/Lx;
      double y_0_tilde = (data[i].get_y_0() - data[i].get_c())/Ly;

      double x_t_tilde = (data[i].get_x_T() - data[i].get_a())/Lx;
      double y_t_tilde = (data[i].get_y_T() - data[i].get_c())/Ly;

      double tau_x = sigma_x/Lx;
      double tau_y = sigma_y/Ly;

      double sigma_y_tilde = tau_y/tau_x;
      double rho_star = rho;

      double t_tilde = t*tau_x*tau_x;


      if (tau_x < tau_y) {
	x_t_tilde = (data[i].get_y_T() - data[i].get_c())/Ly;
	y_t_tilde = (data[i].get_x_T() - data[i].get_a())/Lx;
	//
	x_0_tilde = (data[i].get_y_0() - data[i].get_c())/Ly;
	y_0_tilde = (data[i].get_x_0() - data[i].get_a())/Lx;
	//
	sigma_y_tilde = tau_x/tau_y;
	//
	t_tilde = t*std::pow(tau_y, 2);
      }


      // flipping x-axis when rho<0
      //
      // x_0 -> -x_0
      // x_T -> -x_T
      //
      // a -> -a
      // b -> -b
      if (std::signbit(rho)) {
	x_0_tilde = (-data[i].get_x_0() - -1.0*data[i].get_b())/Lx;
	x_t_tilde = (-data[i].get_x_T() - -1.0*data[i].get_b())/Lx;

	rho_star = -1.0*rho_star;
      }

      // printf("t_tilde = %f\n", t_tilde);
      BivariateSolver solver = BivariateSolver(&(*private_bases_)[k],
					       1.0,
					       sigma_y_tilde,
					       rho_star,
					       0.0, // a
					       x_0_tilde, // x0
					       1.0, // b
					       0.0, // c
					       y_0_tilde, // y0
					       1.0, // d
					       t_tilde,
					       dx);

      std::vector<double> eigenvalues (5);
      eigenvalues[0] = gsl_vector_get(solver.get_evals(), 0);
      eigenvalues[1] = gsl_vector_get(solver.get_evals(), 1);
      eigenvalues[2] = gsl_vector_get(solver.get_evals(), 2);
      eigenvalues[3] = gsl_vector_get(solver.get_evals(), 3);
      eigenvalues[4] = gsl_vector_get(solver.get_evals(), 4);
      gsl_vector_set(input, 0, x_t_tilde);
      gsl_vector_set(input, 1, y_t_tilde);

      double likelihood = 0.0;

      if (t_tilde >= 0.30) {
	likelihood = solver.numerical_likelihood(input,
						 dx_likelihood)/
	  (Lx*Ly);
      } else {
	likelihood = std::numeric_limits<double>::infinity();
      }

      // printf("Thread %d: likelihood = %f\n", omp_get_thread_num(), likelihood);
      if (!std::signbit(likelihood) &&
	  (std::abs(likelihood) < std::numeric_limits<double>::infinity()) &&
	  !std::isnan(likelihood) &&
	  t_tilde >= 0.30)
	{
	neg_log_likelihoods[i] = -std::log(likelihood);
	bad_likelihoods_indicators[i] = 0;
      } else {
	std::vector<BivariateImageWithTime> small_positions =
	  solver.small_t_image_positions_type_41_symmetric(false);
	double small_t = small_positions[0].get_t();
	double rho_for_small_t = rho_star;

	// for some configurations, the small-time solution doesn't
	// work, in which case small-t is negative. Shrink rho
	// sufficiently s.t. the small-t solution works.  It's
	// always guaranteed to work for sufficiently small rho.
	while (std::signbit(small_t)) {
	  rho_for_small_t = rho_for_small_t * 0.95;
	  solver.set_diffusion_parameters_and_data(1.0,
						   sigma_y_tilde,
						   rho_for_small_t,
						   t_tilde,
						   0.0,
						   x_0_tilde,
						   1.0,
						   0.0,
						   y_0_tilde,
						   1.0);
	  small_positions = solver.small_t_image_positions_type_41_symmetric(false);
	  small_t = small_positions[0].get_t();
	}

	// MATCHING CONSTANTS START
	unsigned number_big_t_points = 3;
	double alpha = 4.0;

	// ------------------ //
	// std::vector<double>::const_iterator first_lambda = eigenvalues.begin();
	// std::vector<double>::const_iterator last_lambda = eigenvalues.begin() + 1;
	std::vector<double> lambdas = std::vector<double> {eigenvalues[0],
							   eigenvalues[1]};
	// ------------------ //

	std::vector<double> t_tildes_small = std::vector<double> {small_t};
	std::vector<double> alphas = std::vector<double> {alpha};

	std::vector<double> ys_small (t_tildes_small.size());
	std::vector<double> log_ys_small (t_tildes_small.size());

	std::generate(log_ys_small.begin(), log_ys_small.end(),
		      [n = 0,
		       &solver,
		       input,
		       &t_tildes_small] () mutable { 
			double out = solver.
			  likelihood_small_t_41_truncated_symmetric(input,
								    t_tildes_small[n],
								    1e-5);
			n++;
			return out;});

	std::generate(ys_small.begin(), ys_small.end(),
		      [n = 0,
		       &log_ys_small] () mutable { double out = exp(log_ys_small[n]);
			n++;
			return out;});	

	std::vector<double> t_tildes = t_tildes_small;
	std::vector<double> ys = ys_small;
	std::vector<double> log_ys = log_ys_small;

	double t_tilde_2 = 0.50;
	if (std::abs(rho_star) <= 0.80) {
	  t_tilde_2 = 0.30;
	}

	while (number_big_t_points > 0) {
	  solver.set_diffusion_parameters_and_data(1.0,
						   sigma_y_tilde,
						   rho_star,
						   t_tilde_2,
						   0.0,
						   x_0_tilde,
						   1.0,
						   0.0,
						   y_0_tilde,
						   1.0);

	  double y2 = solver.numerical_likelihood(input,
						  dx_likelihood);

	  // if (std::isnan(y2)) {
	  //   y2 = 0.0;
	  // }

	  while (std::isnan(std::log(y2))) {
	    if (t_tilde_2 <= 4) {
	      t_tilde_2 = t_tilde_2 + 0.50;
	    } else {
	      t_tilde_2 = t_tilde_2 + 20.0;
	    }
	    solver.set_diffusion_parameters_and_data(1.0,
						     sigma_y_tilde,
						     rho_star,
						     t_tilde_2,
						     0.0,
						     x_0_tilde,
						     1.0,
						     0.0,
						     y_0_tilde,
						     1.0);
	    y2 = solver.numerical_likelihood(input,
					     dx_likelihood);

	    printf("Encountered big t with bad likelihood, trying t_tilde = %f, t_tilde_2 = %f w/ like=%f\n",
		   t_tilde,
		   t_tilde_2,
		   y2);
	  }

	  t_tildes.push_back(t_tilde_2);
	  ys.push_back(y2);

	  t_tilde_2 = t_tilde_2 + 0.50;
	  number_big_t_points--;
	}
	
	gsl_vector* weights = find_weights(ys, t_tildes, alphas, lambdas);
	double t_max = find_max(weights, lambdas, alphas, t_tildes, small_t);
	solver.set_diffusion_parameters_and_data(1.0,
						 sigma_y_tilde,
						 rho_star,
						 t_max,
						 0.0,
						 x_0_tilde,
						 1.0,
						 0.0,
						 y_0_tilde,
						 1.0);

	double galerkin_like = solver.numerical_likelihood(input,
							   dx_likelihood);

	double approx_log_like = alphas[0]*log(t_max) + lambdas[0]*t_max + 
	  log(gsl_vector_get(weights, 0) + gsl_vector_get(weights, 1)*exp((lambdas[1]-lambdas[0]) * t_max));
	
	if (!std::isnan(log(galerkin_like)) &
	    (log(galerkin_like) > approx_log_like) ) {

	  ys[1] = galerkin_like;
	  t_tildes[1] = t_max;

	  gsl_vector_free(weights);
	  weights = find_weights(ys, t_tildes, alphas, lambdas);
	  t_max = find_max(weights, lambdas, alphas, t_tildes, small_t);
	}

	// RHS matching
	double w1 = gsl_vector_get(weights, 0);
	double w2 = gsl_vector_get(weights, 1);
	double Delta = lambdas[1]-lambdas[0];
	double function_val = alphas[0]*log(t_max) + lambdas[0]*t_max + log(w1 + w2*exp(Delta*t_max));
	double first_deriv =  lambdas[0] + alphas[0]/t_max + Delta*w2*exp(Delta*t_max)/(w1 + w2*exp(Delta*t_max));
	double second_deriv = -alphas[0]/(t_max*t_max) + 
	  Delta*Delta*w2*exp(Delta*t_max)/(w1 + w2*exp(Delta*t_max)) -
	  Delta*Delta*w2*w2*exp(2*Delta*t_max)/std::pow(w1 + w2*exp(Delta*t_max), 2.0);
	double deriv_matrix_array [4] = {-1.0/t_max, 1.0/(t_max*t_max), 1.0/(t_max*t_max), -2.0/(t_max*t_max*t_max)};
	gsl_matrix_view deriv_matrix_view = gsl_matrix_view_array(deriv_matrix_array, 2,2);
	
	gsl_vector* gamma_beta_t_max = gsl_vector_alloc(2);
	gsl_vector* b_vector = gsl_vector_alloc(2);
	gsl_vector_set(b_vector, 0, first_deriv);
	gsl_vector_set(b_vector, 1, second_deriv);
	gsl_permutation* p = gsl_permutation_alloc(2);
	int s = 0;

	gsl_linalg_LU_decomp(&deriv_matrix_view.matrix, p, &s);
	gsl_linalg_LU_solve(&deriv_matrix_view.matrix, p, b_vector, gamma_beta_t_max);

	gsl_permutation_free(p);
	gsl_vector_free(b_vector);

	double gamma_t_max = gsl_vector_get(gamma_beta_t_max, 0);
	double beta_t_max = gsl_vector_get(gamma_beta_t_max, 1);
	double log_omega_t_max = function_val + gamma_t_max*log(t_max) + beta_t_max/t_max;
	gsl_vector_free(gamma_beta_t_max);

	// LHS matching 
	double log_omega_t_small = -1.0*log(M_PI*std::sqrt(2)) - 
	  4.5*(log(2.0) +
	       2.0*log(sigma_y_tilde) +
	       log(1-rho_for_small_t*rho_for_small_t));

	std::vector<double> dPdaxs = solver.dPdax(input, dx_likelihood);
	std::vector<double> dPdbxs = solver.dPdbx(input, dx_likelihood);
	std::vector<double> dPdays = solver.dPday(input, dx_likelihood);
	std::vector<double> dPdbys = solver.dPdby(input, dx_likelihood);
	
	std::vector<double> betas_t_small = std::vector<double> (4);
	unsigned nn = 0;
	std::generate(betas_t_small.begin(), 
		      betas_t_small.end(), 
		      [&] () mutable { 
			double beta_t_small = 1.0/(2.0*
						   sigma_y_tilde*
						   sigma_y_tilde*
						   (1.0 - rho*rho))*
			  dPdaxs[nn]*dPdbxs[nn]*dPdays[nn]*dPdbys[nn];
			nn++;
			return beta_t_small;
		      });

	std::sort(betas_t_small.begin(), betas_t_small.end());

	double beta_t_small = betas_t_small[3];
	double gamma_t_small = 4.5;

	double k = 100;
	double log_omega = log_omega_t_small*exp(-k*(t_tilde-small_t)) +
	  log_omega_t_max*(1 - exp(-k*(t_tilde-small_t)));
	double gamma = gamma_t_small*exp(-k*(t_tilde-small_t)) +
	  gamma_t_max*(1 - exp(-k*(t_tilde-small_t)));
	double beta = beta_t_small*exp(-k*(t_tilde-small_t)) +
	  beta_t_max*(1 - exp(-k*(t_tilde-small_t)));
	// MATCH CONSTANTS END

	double matched_sol = log_omega - gamma*log(t_tilde) - beta/t_tilde;
	printf("ys[1]=%f, ys[2]=%f, ys[3]=%f,\n, lambdas[0] = %f, lambdas[1] = %f, small_t = %f, t_tildes_last = %f, Delta = %f, alphas[0] = %f, t_max = %f, w1 = %f, w2 = %f, function_val = %f, log_omega_t_small = %f, log_omega_t_max = %f, log_omega = %f, gamma = %f, beta = %f, matched_sol = %f\n", 
	       ys[0], ys[1], ys[2],
	       lambdas[0],
	       lambdas[1],
	       small_t,
	       t_tildes[t_tildes.size()-1],
	       Delta,
	       alphas[0],
	       t_max,
	       w1, w2,
	       function_val,
	       log_omega_t_small,
	       log_omega_t_max,
	       log_omega,
	       gamma,
	       beta,
	       matched_sol);
	double log_likelihood = matched_sol;

	if (std::isnan(matched_sol)) {
	  log_likelihood = log_ys_small[0];
	}

	neg_log_likelihoods[i] = -log_likelihood;
	bad_likelihoods_indicators[i] = 1;
      }
    }

    gsl_vector_free(input);
  }

  double sum_of_elements_parallel = 0;
  int number_bad_likelihoods = 0;
  std::cout << "neg_lls = ";
  for (unsigned i=0; i<neg_log_likelihoods.size(); ++i) {
    sum_of_elements_parallel = sum_of_elements_parallel +
      neg_log_likelihoods[i];

    std::cout << neg_log_likelihoods[i] << " ";

    number_bad_likelihoods = number_bad_likelihoods +
      bad_likelihoods_indicators[i];
  }
  std::cout << std::endl;
  std::cout << "number bad likelihoods = " << number_bad_likelihoods << std::endl;
  number_negaive_lls_per_iteration_.push_back(number_bad_likelihoods);

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
  std::vector<int> number_negaive_lls_per_iteration_ = std::vector<int> ();
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
  TwoDMLEFiniteElement * mle = reinterpret_cast<TwoDMLEFiniteElement*>(data);
  double out = mle->operator()(x,grad);
  std::cout << "neg log-likelihood = " << out << std::endl;
  return out;
}
