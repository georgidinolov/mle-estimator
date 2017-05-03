#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteElement.hpp"

int main(int argc, char *argv[]) {
  if (argc < 9 || argc > 9) {
    printf("You must provide input\n");
    printf("The input is: \n data file list (each file on new line); \noutput directory;\nrelative tolerance for function during mle estimation (as double); \ninitial guess for sigma_x; \ninitial guess for sigma_y; \ninitial guess for rho; \nfile name prefix; \nfile name suffix; \n");
    printf("file names will be PREFIXmle-results-NUMBER_DATA_SET-order-ORDERSUFFIX.csv, stored in the output directory.\n");
    exit(0);
  }

  std::string data_file_list_dir = argv[1];

  std::vector<std::string> data_files (0);
  std::string file;
  std::ifstream data_file_list (data_file_list_dir);

  if (data_file_list.is_open()) {
    while (std::getline(data_file_list, file)) {
      data_files.push_back(file);
    }
  }

  std::string output_file_dir = argv[2];
  double rel_tol = std::stod(argv[3]);
  double sigma_x = std::stod(argv[4]);
  double sigma_y = std::stod(argv[5]);
  double rho = std::stod(argv[6]);
  std::string prefix = argv[7];
  std::string suffix = argv[8];

  double dx = 5e-3;
  double rho_min = 0.60;
  double rho_max = 0.60;
  unsigned n_rhos = 1;

  std::string data_file_dir = data_files[0];
  TwoDMLEFiniteElement mle_estimator = 
    TwoDMLEFiniteElement(data_file_dir,
  			 sigma_x,
  			 sigma_y,
  			 rho,
  			 rho_min, rho_max, n_rhos,
  			 dx,
  			 0.3,
  			 1,
  			 0.5);

  for (unsigned i=227; i<data_files.size(); ++i) {
    data_file_dir = data_files[i];
    
    // order 64 START
    int order = 64;
    std::string output_file_name = output_file_dir + 
      prefix + 
      "mle-results-" + std::to_string(i) + "-order-" + std::to_string(order) 
      + suffix + ".csv";
    mle_estimator.set_data_file(data_file_dir);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<double> log_sigma_x_sigma_y_rho = 
      mle_estimator.find_mle(order,
    			     rel_tol,
    			     sigma_x,
    			     sigma_y,
    			     rho);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "duration = " << duration << std::endl;

    std::ofstream output_file;
    output_file.open(output_file_name);
    // header
    output_file << "sigma_x, sigma_y, rho\n";
    output_file << log_sigma_x_sigma_y_rho[0] << ","
    		<< log_sigma_x_sigma_y_rho[1] << ","
    		<< log_sigma_x_sigma_y_rho[2] << "\n";
    output_file.close();
    // order 64 END

    // order 128 START
    order = 128;
    output_file_name = output_file_dir + 
       prefix + "mle-results-" + 
       std::to_string(i) + "-order-" + std::to_string(order) + suffix + ".csv";
    mle_estimator.set_data_file(data_file_dir);

    log_sigma_x_sigma_y_rho = 
      mle_estimator.find_mle(order,
    			     rel_tol,
    			     sigma_x,
    			     sigma_y,
    			     rho);

    output_file.open(output_file_name);
    // header
    output_file << "sigma_x, sigma_y, rho\n";
    output_file << log_sigma_x_sigma_y_rho[0] << ","
    		<< log_sigma_x_sigma_y_rho[1] << ","
    		<< log_sigma_x_sigma_y_rho[2] << "\n";
    output_file.close();
    // order 128 END

    // order 16 START
    order = 16;
    output_file_name = output_file_dir + 
       prefix + "mle-results-" + 
       std::to_string(i) + "-order-" + std::to_string(order) + suffix + ".csv";
    mle_estimator.set_data_file(data_file_dir);

    log_sigma_x_sigma_y_rho = 
      mle_estimator.find_mle(order,
    			     rel_tol,
    			     sigma_x,
    			     sigma_y,
    			     rho);

    output_file.open(output_file_name);
    // header
    output_file << "sigma_x, sigma_y, rho\n";
    output_file << log_sigma_x_sigma_y_rho[0] << ","
    		<< log_sigma_x_sigma_y_rho[1] << ","
    		<< log_sigma_x_sigma_y_rho[2] << "\n";
    output_file.close();
    // order 16 END
  }

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
}
