#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

//' @export
// [[Rcpp::export]]
arma::mat cpp_execute_uniformization_advanced(const arma::sp_mat& Q, 
                                              const arma::rowvec& v, // Accepts a dense row vector straight from R
                                              int m, 
                                              double alpha, 
                                              double tolerance) {
  int N = Q.n_rows;
  
  // 1. Build the Uniformized Transition Matrix: P = I + Q / alpha
  arma::sp_mat I;
  I.eye(N, N);
  arma::sp_mat P = I + (Q / alpha);
  
  // 2. Compute the lower truncation bound (m_lower) dynamically up to m
  int m_lower = 0;
  double running_poisson_sum = 0.0;
  
  for (int n = 0; n <= m; ++n) {
    double log_p_n = n * std::log(alpha) - alpha - std::lgamma(n + 1);
    double p_n = std::exp(log_p_n);
    
    running_poisson_sum += p_n;
    if (running_poisson_sum < tolerance) {
      m_lower = n; 
    } else {
      break; 
    }
  }
  
  // 3. Setup dense tracking containers (zero reallocations)
  arma::rowvec v_curr = v;
  arma::rowvec v_next(N);
  arma::rowvec result(N, arma::fill::zeros); // Dense accumulator
  
  // 4. Bare-Metal Execution Loop
  for (int n = 0; n <= m; ++n) {
    
    // Only accumulate if we are in the active Poisson window
    if (n >= m_lower) {
      double log_p_n = n * std::log(alpha) - alpha - std::lgamma(n + 1);
      double p_n = std::exp(log_p_n);
      
      result += p_n * v_curr;
    }
    
    // Super high-speed Dense Vector * Sparse Matrix multiplication
    if (n < m) {
      v_next = v_curr * P;
      v_curr = v_next; // Blazing fast element assignment
    }
  }  
  
  return result;
}