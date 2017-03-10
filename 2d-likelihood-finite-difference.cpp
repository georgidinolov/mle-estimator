#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main(int argc, char *argv[]) {
  if (argc < 7) {
    printf("You must provide input\n");
    printf("The input is: \n data file; \n output file; \n int order of numerical accuracy (try 32, 64, or 128 for now); \n value for sigma_x; \n value for sigma_y; \n value for rho; \n");
    exit(0);
  }

  std::string data_file_dir = argv[1];
  std::string output_file_name = argv[2];
  int order = std::stoi(argv[3]);
  double sigma_x = std::stod(argv[4]);
  double sigma_y = std::stod(argv[5]);
  double rho = std::stod(argv[6]);

  std::ofstream output_file;
  output_file.open(output_file_name);
  output_file << std::fixed << std::setprecision(32);
  // header
  output_file << "sigma_x, sigma_y, rho, t, x_0, y_0, a, x_T, b, c, y_T, d, log_likelihood\n";
  
  TwoDMLEFiniteDifference mle_estimator = 
    TwoDMLEFiniteDifference(data_file_dir,
 			    sigma_x,
 			    sigma_y,
 			    rho);

   std::vector<double> neg_log_likelihoods =
     mle_estimator.negative_log_likelihoods_parallel(order,
						    sigma_x,
						    sigma_y,
						    rho);

   std::vector<ContinuousProblemData> quantized_data =
     mle_estimator.quantized_continuous_data(order,
					     sigma_x, 
					     sigma_y, 
					     rho);

   for (unsigned i=0; i<neg_log_likelihoods.size(); ++i) {
     output_file << sigma_x << "," 
		 << sigma_y << ","
		 << rho << ","
		 << quantized_data[i].get_t() << ","
		 << quantized_data[i].get_x_0() << ","
		 << quantized_data[i].get_y_0() << ","
		 << quantized_data[i].get_a() << ","
		 << quantized_data[i].get_x_T() << ","
		 << quantized_data[i].get_b() << ","
		 << quantized_data[i].get_c() << ","
		 << quantized_data[i].get_y_T() << ","
		 << quantized_data[i].get_d() << ","
		 << -1.0*neg_log_likelihoods[i] << "\n";
     std::cout << neg_log_likelihoods[i] << "\n";
   }
   std::cout << std::endl;
   output_file.close();
}
