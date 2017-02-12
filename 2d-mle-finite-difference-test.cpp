#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("You must provide input\n");
    printf("The input is: \n load location of data file \n");
    exit(0);
  }
  
  double rho = 0.0;
  double sigma_x = 1.0;
  double sigma_y = 1.0;
  double t = 1;

  std::string data_file_dir = argv[1];
  
  TwoDMLEFiniteDifference mle_estimator = 
    TwoDMLEFiniteDifference(data_file_dir,
 			    sigma_x,
 			    sigma_y,
 			    rho);

  auto t1 = std::chrono::high_resolution_clock::now();
  double nll = mle_estimator.negative_log_likelihood_parallel(70,
						     0.36787944117144233,
						     1.94773404105467573,
						     0.31666666666666665);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
  	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  	    << " milliseconds\n";  
  printf("neg log-likelihood = %.16f\n", nll);
  
   std::vector<double> log_sigma_x_sigma_y_rho = 
     mle_estimator.find_mle(128,
   			   1.0,
   			   1.0,
   			   0.0);
}
