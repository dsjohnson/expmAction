#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

//' @export
// [[Rcpp::export]]
arma::mat cpp_nimble_sparse_uniformization(const arma::vec& nz_values,   // Changing MCMC rates from NIMBLE
                                           const arma::uvec& row_idx,   // Static non-zero row indices
                                           const arma::uvec& col_idx,   // Static non-zero col indices
                                           int N,                       // Matrix dimensions (N x N)
                                           const arma::rowvec& v, 
                                           double t, 
                                           double tolerance) {
  
  // 1. Instantly assemble the sparse matrix using zero-copy batch insertion
  // 'locations' is a 2 x NNZ matrix specifying coordinate mappings
  arma::umat locations(2, nz_values.n_elem);
  locations.row(0) = row_idx.t();
  locations.row(1) = col_idx.t();
  
  arma::sp_mat Q(locations, nz_values, N, N);
  
  // 2. Compute alpha_0 directly from the newly assembled sparse matrix
  arma::vec diag_elements(Q.diag()); // Explicitly instantiate dense vector from the sparse view
  double alpha_0 = arma::max(arma::abs(diag_elements));
  
  // 3. Fallback check for the loop
  double alpha = alpha_0 * t;
  int m = 0;
  double running_right_sum = 0.0;
  while (running_right_sum < (1.0 - tolerance)) {
    double log_p_m = m * std::log(alpha) - alpha - std::lgamma(m + 1);
    running_right_sum += std::exp(log_p_m);
    m++;
    if (m > 1000000) break; 
  }
  
  int m_lower = 0;
  double running_left_sum = 0.0;
  for (int n = 0; n <= m; ++n) {
    double log_p_n = n * std::log(alpha) - alpha - std::lgamma(n + 1);
    running_left_sum += std::exp(log_p_n);
    if (running_left_sum < tolerance) {
      m_lower = n; 
    } else {
      break;
    }
  }
  
  // 4. Uniformization setup
  arma::sp_mat I;
  I.eye(N, N);
  arma::sp_mat P = I + (Q / alpha_0);
  
  arma::rowvec v_curr = v;
  arma::rowvec v_next(N);
  arma::rowvec result(N, arma::fill::zeros); 
  
  // 5. Execution
  for (int n = 0; n <= m; ++n) {
    if (n >= m_lower) {
      double log_p_n = n * std::log(alpha) - alpha - std::lgamma(n + 1);
      double p_n = std::exp(log_p_n);
      result += p_n * v_curr;
    }
    if (n < m) {
      v_next = v_curr * P;
      v_curr = v_next;
    }
  }  
  
  return result;
}