#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main(int argc, char *argv[]) {
  if (argc < 6) {
    printf("You must provide input\n");
    printf("The input is: \n load location of data file; int order of numerical accuracy (try 32, 64, or 128 for now); \n initial guess for sigma_x; \n initial guess for sigma_y; \n initial guess for rho; \n");
    exit(0);
  }

  std::string data_file_dir = argv[1];
  int order = std::stoi(argv[2]);
  double sigma_x = std::stod(argv[3]);
  double sigma_y = std::stod(argv[4]);
  double rho = std::stod(argv[5]);
  
  TwoDMLEFiniteDifference mle_estimator = 
    TwoDMLEFiniteDifference(data_file_dir,
 			    sigma_x,
 			    sigma_y,
 			    rho);

  // auto t1 = std::chrono::high_resolution_clock::now();
  // double nll = mle_estimator.negative_log_likelihood_parallel(70,
  // 						     0.36787944117144233,
  // 						     1.94773404105467573,
  // 						     0.31666666666666665);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "duration = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << " milliseconds\n";  
  // printf("neg log-likelihood = %.16f\n", nll);
  
   std::vector<double> log_sigma_x_sigma_y_rho = 
     mle_estimator.find_mle(order,
			    sigma_x,
			    sigma_y,
			    rho);
}
