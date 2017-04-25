#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteElement.hpp"

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
  
  double dx = 5e-3;
  double rho_min = -0.9;
  double rho_max = 0.9;
  unsigned n_rhos = 5;

  TwoDMLEFiniteElement mle_estimator = 
    TwoDMLEFiniteElement(data_file_dir,
			 sigma_x,
			 sigma_y,
			 rho,
			 rho_min, rho_max, n_rhos,
			 dx,
			 0.2,
			 1,
			 0.5);

  // std::cout << "mle_estimator.negative_log_likelihood_parallel(" 
  // 	    << order << ",1,1,0.5) = ";
  // std::cout << mle_estimator.negative_log_likelihood_parallel(order,1,1,0.5)
  // 	    << std::endl;

  // double dr = 0.05;
  // unsigned R = 37;
  // std::vector<double> rhos (R);
  // for (unsigned i=0; i<R; ++i) {
  //   rhos[i] = -0.9 + i*dr;
  //   auto t1 = std::chrono::high_resolution_clock::now();
  //   output_file << sigma_x << "," 
  // 		<< sigma_y << ","
  // 		<< rhos[i] << ","
  // 		<< mle_estimator.negative_log_likelihood_parallel(order,
  // 								  sigma_x,
  // 								  sigma_y,
  // 								  rhos[i])
  // 		<< "\n";
  //   auto t2 = std::chrono::high_resolution_clock::now();
  //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  //   std::cout << "done with rho = " << rhos[i]
  // 	      << "; duration = " << duration
  // 	      << std::endl;
  // }

   std::vector<double> log_sigma_x_sigma_y_rho = 
     mle_estimator.find_mle(order,
   			    rel_tol,
   			    sigma_x,
   			    sigma_y,
   			    rho);
   output_file << log_sigma_x_sigma_y_rho[0] << ","
   	       << log_sigma_x_sigma_y_rho[1] << ","
   	       << log_sigma_x_sigma_y_rho[2] << "\n";
   output_file.close();
}
