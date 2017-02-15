#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main(int argc, char *argv[]) {
  if (argc < 7) {
    printf("You must provide input\n");
    printf("The input is: \n load location of data file; int order of numerical accuracy (try 32, 64, or 128 for now); \n relative tolerance for convergence of mle; \n initial guess for sigma_x; \n initial guess for sigma_y; \n initial guess for rho; \n");
    exit(0);
  }

  std::string data_file_dir = argv[1];
  int order = std::stoi(argv[2]);
  double rel_tol = std::stod(argv[3]);
  double sigma_x = std::stod(argv[4]);
  double sigma_y = std::stod(argv[5]);
  double rho = std::stod(argv[6]);
  
  TwoDMLEFiniteDifference mle_estimator = 
    TwoDMLEFiniteDifference(data_file_dir,
 			    sigma_x,
 			    sigma_y,
 			    rho);

   // auto t1 = std::chrono::high_resolution_clock::now();
   // double nll = mle_estimator.negative_log_likelihood_parallel(order,1.94773404105467573543819526094012, 0.36787944117144233402427744294982,0.31666666666666665186369300499791);
   // auto t2 = std::chrono::high_resolution_clock::now();
   // std::cout << "duration = "
   // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
   // 	    << " milliseconds\n";  
   // printf("neg log-likelihood = %.16f\n", nll);
  
   std::vector<double> log_sigma_x_sigma_y_rho = 
    mle_estimator.find_mle(order,
			   rel_tol,
			   sigma_x,
			   sigma_y,
			   rho);
}
