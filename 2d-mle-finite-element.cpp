#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteElement.hpp"

int main(int argc, char *argv[]) {
  if (argc < 14 || argc > 14) {
    printf("You must provide input\n");
    printf("The input is: \n data file list (each file on new line); \noutput directory;\nrelative tolerance for function during mle estimation (as double); \ninitial guess for sigmaxx; \ninitial guess for sigma_y; \ninitial guess for rho; \nrho for basis; \nsigma_x for basis; \nsigma_y for basis; \nfile name prefix; \nfile name suffix; \nnumber threads;\nn_dx such that dx = 1.0/n_dx \n");
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
  double rho_basis = std::stod(argv[7]);
  double sigma_x_basis = std::stod(argv[8]);
  double sigma_y_basis = std::stod(argv[9]);
  std::string prefix = argv[10];
  std::string suffix = argv[11];
  unsigned number_threads = std::stoi(argv[12]);
  int n_dx = std::stoi(argv[13]);

  std::cout << "rho_basis = " << rho_basis << "\n";
  std::cout << "sigma_x_basis = " << sigma_x_basis << "\n";
  std::cout << "sigma_y_basis = " << sigma_y_basis << "\n";
  
  double dx = 1.0/n_dx;

  std::string data_file_dir = data_files[0];
  TwoDMLEFiniteElement mle_estimator = 
    TwoDMLEFiniteElement(data_file_dir,
			 rho,
  			 sigma_x,
  			 sigma_y,
			 dx,
  			 rho_basis, // rho
  			 sigma_x_basis, // sigma_x for bases
  			 sigma_y_basis, // sigma_y for bases
  			 1.0, // mollifier
  			 1.0,
			 number_threads); // fraction for separation l
  
  std::cout << "data_files.size() = " << data_files.size() << std::endl;
  for (unsigned i=0; i<data_files.size(); ++i) {
    data_file_dir = data_files[i];

    // // order 16 START
    // int order = 16;
    // std::string output_file_name = output_file_dir + 
    //    prefix + "mle-results-" + 
    //    std::to_string(i) + "-order-" + std::to_string(order) + suffix + ".csv";
    // mle_estimator.set_data_file(data_file_dir);

    // std::vector<double> log_sigma_x_sigma_y_rho = 
    //   mle_estimator.find_mle(order,
    // 			     rel_tol,
    // 			     sigma_x,
    // 			     sigma_y,
    // 			     rho);

    // std::ofstream output_file;
    // output_file.open(output_file_name);
    // // header
    // output_file << "sigma_x, sigma_y, rho\n";
    // output_file << log_sigma_x_sigma_y_rho[0] << ","
    // 		<< log_sigma_x_sigma_y_rho[1] << ","
    // 		<< log_sigma_x_sigma_y_rho[2] << "\n";
    // output_file.close();
    // // order 16 END

    // // order 32 START
    // order = 32;
    // output_file_name = output_file_dir + 
    //    prefix + "mle-results-" + 
    //    std::to_string(i) + "-order-" + std::to_string(order) + suffix + ".csv";
    // mle_estimator.set_data_file(data_file_dir);

    // log_sigma_x_sigma_y_rho = 
    //   mle_estimator.find_mle(order,
    // 			     rel_tol,
    // 			     sigma_x,
    // 			     sigma_y,
    // 			     rho);

    // output_file.open(output_file_name);
    // // header
    // output_file << "sigma_x, sigma_y, rho\n";
    // output_file << log_sigma_x_sigma_y_rho[0] << ","
    // 		<< log_sigma_x_sigma_y_rho[1] << ","
    // 		<< log_sigma_x_sigma_y_rho[2] << "\n";
    // output_file.close();
    // // order 32 END
    
    // // order 64 START
    // order = 64;
    // output_file_name = output_file_dir + 
    //   prefix + 
    //   "mle-results-" + std::to_string(i) + "-order-" + std::to_string(order) 
    //   + suffix + ".csv";
    // mle_estimator.set_data_file(data_file_dir);

    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::vector<double> log_sigma_x_sigma_y_rho = 
    //   mle_estimator.find_mle(order,
    // 			     rel_tol,
    // 			     sigma_x,
    // 			     sigma_y,
    // 			     rho);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "duration = " << duration << std::endl;

    // output_file.open(output_file_name);
    // // header
    // output_file << "sigma_x, sigma_y, rho\n";
    // output_file << log_sigma_x_sigma_y_rho[0] << ","
    // 		<< log_sigma_x_sigma_y_rho[1] << ","
    // 		<< log_sigma_x_sigma_y_rho[2] << "\n";
    // output_file.close();
    // // order 64 END

    // order 64 START
    int order = 64;
    std::string output_file_name = output_file_dir + 
       prefix + "mle-results-" + 
       std::to_string(i) + "-order-" + std::to_string(order) + suffix + ".csv";
    std::string output_file_for_NAs_name = output_file_dir + 
       prefix + "NAs-results-" + 
       std::to_string(i) + "-order-" + std::to_string(order) + suffix + ".csv";
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

    output_file.open(output_file_for_NAs_name);
    // header 
    output_file << "number_NAs_per_iteration\n";
    const std::vector<int>& number_negaive_lls_per_iteration = 
      mle_estimator.get_number_negaive_lls_per_iteration();
    for (const int& num : number_negaive_lls_per_iteration) {
      output_file << num << "\n";
    }
    output_file.close();
    // order 64 END
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
