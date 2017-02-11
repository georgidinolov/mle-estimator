#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main() {
  
  unsigned number_data_points = 64;
  std::vector<ContinuousProblemData> data(number_data_points);

  int bm_order = 1000;
  double rho = 0.0;
  double sigma_x = 1.0;
  double sigma_y = 1.0;
  double t = 1;

  for (unsigned i=0; i<number_data_points; ++i) {
    unsigned seed = i;
    
    double x_initial = 0;
    double y_initial = 0;
    if (i!=0) {
      x_initial = data[i-1].get_x_T();
      y_initial = data[i-1].get_y_T();
    }

    BrownianMotion BM = BrownianMotion(bm_order,
				       rho,
				       sigma_x,
				       sigma_y,
				       x_initial,
				       y_initial,
				       t);

    double a = BM.get_a();
    double b = BM.get_b();
    double c = BM.get_c();
    double d = BM.get_d();
    double x_T = BM.get_x_T();
    double y_T = BM.get_y_T();

    std::cout << "(" << a << "," << x_T << "," << b << ") x "
	      << "(" << c << "," << y_T << "," << d << ")"
	      << std::endl;
    
    ContinuousProblemData datum = ContinuousProblemData(x_T,
							y_T,
							x_initial,
							y_initial,
							t,
							a,
							b,
							c,
							d);
    data[i] = datum;
  }

  TwoDMLEFiniteDifference mle_estimator = 
    TwoDMLEFiniteDifference(data,
 			    sigma_x,
 			    sigma_y,
 			    rho);

  double nll = mle_estimator.negative_log_likelihood_parallel(64,
							data,
						     1,	
						     1,		
						     0.0);
  std::cout << "neg log-likelihood = " << nll << std::endl;
  
  std::vector<double> log_sigma_x_sigma_y_rho = 
    mle_estimator.find_mle(128,
  			   1.0,
  			   1.0,
  			   0.0);
  
  // std::cout << "sigma_x = " << exp(log_sigma_x_sigma_y_rho[0]) << "\n";
  // std::cout << "sigma_y = " << exp(log_sigma_x_sigma_y_rho[1]) << "\n";
  // std::cout << "rho = " << log_sigma_x_sigma_y_rho[2] << std::endl;
  
  int discretization_size = 9;
 // double lower_log_sigma = -0.3;
 // double uppper_log_sigma = 0.2;
 // std::vector<double> log_sigma_x_vector(discretization_size+1);
 // std::vector<double> log_sigma_y_vector(discretization_size+1);
 // for (int i=0; i<discretization_size+1; ++i) {
 //   if (i==0) {
 //     log_sigma_x_vector[i] = lower_log_sigma;
 //     log_sigma_y_vector[i] = lower_log_sigma;
 //   } else {
 //     double log_sigma = lower_log_sigma + 
 //       1.0*i*(uppper_log_sigma-lower_log_sigma)/(1.0*discretization_size);
 //     log_sigma_x_vector[i] = log_sigma;
 //     log_sigma_y_vector[i] = log_sigma;
 //   }
 // }

 // std::ofstream likelihood_map;
 // likelihood_map.open("/home/gdinolov/Research/PDE-solvers/documentation/2-D-advection-diffusion/likelihood-map.csv");
 // // header
 // likelihood_map << "log.sigma.x, log.sigma.y, neg.ll\n";
 // for (int i=0; i<discretization_size+1; ++i) {
 //   for (int j=0; j<discretization_size+1; ++j) {
 //     double sigma_x = exp(log_sigma_x_vector[i]);
 //     double sigma_y = exp(log_sigma_y_vector[j]);

 //     std::cout << "On iteration " << (discretization_size+1)*i + (j+1)
 // 	       << " out of " << (discretization_size+1)*(discretization_size+1)
 // 	       << std::endl;

 //     double neg_ll = mle_estimator.negative_log_likelihood(32,
 //     							   sigma_x,
 //     							   sigma_y,
 //     							   rho);

 //     likelihood_map << log_sigma_x_vector[i] 
 // 		    << ","
 // 		    << log_sigma_y_vector[j]
 // 		    << "," 
 // 		    << neg_ll << "\n";
 //   }
 // }
 // likelihood_map.close();
 
 // sigma_x = 1;
 // sigma_y = 1;
 // std::ofstream rho_likelihood_map;
 // rho_likelihood_map.open("/home/gdinolov/Research/PDE-solvers/documentation/2-D-advection-diffusion/rho-likelihood-map.csv");
 // // header
 // rho_likelihood_map << "rho, neg.ll\n";
 // double rho_lower = 0.2;
 // double rho_upper = 0.8;
 // for (int i=0; i<discretization_size+1; ++i) {
 //   double rho_current = rho_lower + 
 //     i*(rho_upper-rho_lower)/(discretization_size+1);

 //   std::cout << "On iteration "
 // 	     << (i+1)
 // 	     << std::endl;

 //   double neg_ll = mle_estimator.negative_log_likelihood(32,
 // 							 sigma_x,
 // 							 sigma_y,
 // 							 rho_current);

 //   rho_likelihood_map << rho_current
 // 		      << ","
 // 		      << neg_ll << "\n";
   
 // }
 // rho_likelihood_map.close();

}
