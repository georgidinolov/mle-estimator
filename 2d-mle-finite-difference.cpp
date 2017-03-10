#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main(int argc, char *argv[]) {
  if (argc < 8) {
    printf("You must provide input\n");
    printf("The input is: \n data file; \n output file; \n int order of numerical accuracy (try 32, 64, or 128 for now); \n relative tolerance for function during mle estimation (as double); \n initial guess for sigma_x; \n initial guess for sigma_y; \n initial guess for rho; \n");
    exit(0);
  }

  std::string data_file_dir = argv[1];
  std::string output_file_name = argv[2];
  int order = std::stoi(argv[3]);
  double rel_tol = std::stod(argv[4]);
  double sigma_x = std::stod(argv[5]);
  double sigma_y = std::stod(argv[6]);
  double rho = std::stod(argv[7]);

  std::ofstream output_file;
  output_file.open(output_file_name);
  // header
  output_file << "sigma_x, sigma_y, rho\n";
  
  TwoDMLEFiniteDifference mle_estimator = 
    TwoDMLEFiniteDifference(data_file_dir,
 			    sigma_x,
 			    sigma_y,
 			    rho);

   std::vector<double> log_sigma_x_sigma_y_rho = 
     mle_estimator.find_mle(order,
			    rel_tol,
			    sigma_x,
			    sigma_y,
			    rho);

   // output_file << exp(log_sigma_x_sigma_y_rho[0]) << ","
   // 	       << exp(log_sigma_x_sigma_y_rho[1]) << ","
   // 	       << 2* exp(log_sigma_x_sigma_y_rho[2])/
   //   (exp(log_sigma_x_sigma_y_rho[2]) + 1) - 1<< "\n";
   output_file << log_sigma_x_sigma_y_rho[0] << ","
	       << log_sigma_x_sigma_y_rho[1] << ","
	       << log_sigma_x_sigma_y_rho[2] << "\n";
   output_file.close();
}
