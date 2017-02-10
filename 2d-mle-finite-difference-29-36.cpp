#include <fstream>
#include <iostream>
#include <vector>
#include "../brownian-motion/2DBrownianMotionPath.hpp"
#include "2DMLEFiniteDifference.hpp"
#include "PDEDataTypes.hpp"

int main() {
  unsigned order = 10000;
  double sigma_x = 1.0;
  double sigma_y = 0.5;
  double rho = 0.4;
  double x_0 = 0;
  double y_0 = 0;
  double t = 1;
  unsigned number_samples = 48;
  unsigned number_observations = 10;


  std::ofstream path_file;
  std::string file_name = "mle-results-29-36.csv";
  path_file.open(file_name);
  // header
  path_file << "sigma_x, sigma_y, rho\n";
  path_file.close();
  
  for (unsigned i=28; i<36; ++i) {
    path_file.open(file_name, std::ios_base::app);
    std::vector<ContinuousProblemData> data(number_observations);

    for (unsigned j=0; j<number_observations; ++j) {

      unsigned seed = i*number_observations + j;
      BrownianMotion BM = BrownianMotion(seed,
					 order,
					 rho,
					 sigma_x,
					 sigma_y,
					 x_0,
					 y_0,
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
							  x_0,
							  y_0,
							  t,
							  a,
							  b,
							  c,
							  d);
      data[j] = datum;
    }

    TwoDMLEFiniteDifference mle_estimator = 
      TwoDMLEFiniteDifference(data,
			      sigma_x,
			      sigma_y,
			      rho);

      std::vector<double> log_sigma_x_sigma_y_rho = 
	mle_estimator.find_mle(32,
			       sigma_x,
			       sigma_y,
			       rho);
      
      path_file << exp(log_sigma_x_sigma_y_rho[0]) << ","
		<< exp(log_sigma_x_sigma_y_rho[1]) << ","
		<< log_sigma_x_sigma_y_rho[2] << "\n";
      path_file.close();
  }
  return 0;
}
