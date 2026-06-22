#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

/**
 * PHASE 1: Parameter Tuning Optimization Kernel
 * Explicitly scoped using 'arma::' namespaces to prevent Rcpp macro collisions.
 */
//' @export
// [[Rcpp::export]]
Rcpp::List cpp_tune_uniformization(const arma::sp_mat& Q, double tolerance = 1e-12) {
    // Explicitly scope the sparse diagonal view template
    arma::spdiagview<double> diag_Q = Q.diag();
    double alpha = 0.0;
    
    // Iterate securely using native Armadillo types
    for (arma::uword i = 0; i < diag_Q.n_elem; ++i) {
        double abs_val = std::abs(diag_Q(i));
        if (abs_val > alpha) {
            alpha = abs_val;
        }
    }
    
    if (alpha == 0.0) {
        return Rcpp::List::create(
            Rcpp::Named("s") = 1,
            Rcpp::Named("m") = 1,
            Rcpp::Named("alpha") = 0.0
        );
    }
    
    // Local tail-mass worker lambda function
    auto compute_m = [](double alpha_tilde, double tol) {
        double tail_sum = std::exp(-alpha_tilde);
        double current_term = tail_sum;
        int m_val = 0;
        
        while ((1.0 - tail_sum) > tol) {
            m_val++;
            current_term = current_term * alpha_tilde / m_val;
            tail_sum += current_term;
            if (m_val > 5000) break; 
        }
        return (m_val < 1) ? 1 : m_val;
    };
    
    int best_s = 1;
    int best_m = compute_m(alpha, tolerance);
    int min_total_operations = best_s * best_m;
    
    int min_s = std::max(1, (int)std::ceil(alpha / 3.5));
    int max_s = std::max(2, (int)std::ceil(alpha * 2.0));
    
    for (int s_cand = min_s; s_cand <= max_s; ++s_cand) {
        double alpha_tilde_cand = alpha / s_cand;
        int m_cand = compute_m(alpha_tilde_cand, tolerance);
        int total_ops = s_cand * m_cand;
        
        if (total_ops < min_total_operations) {
            min_total_operations = total_ops;
            best_s = s_cand;
            best_m = m_cand;
        }
    }
    
    return Rcpp::List::create(
        Rcpp::Named("s") = best_s,
        Rcpp::Named("m") = best_m,
        Rcpp::Named("alpha") = alpha
    );
}

/**
 * PHASE 2: High-Speed Execution Kernel
 */
//' @export
// [[Rcpp::export]]
arma::sp_mat cpp_execute_uniformization(const arma::sp_mat& Q, const arma::sp_mat& x, 
                                        int s, int m, double alpha, 
                                        double tolerance = 1e-12) {
    if (alpha == 0.0) {
        return x;
    }
    
    double alpha_scaled = alpha / s;
    std::vector<double> weights(m + 1);
    weights[0] = std::exp(-alpha_scaled);
    for (int n = 1; n <= m; ++n) {
        weights[n] = weights[n - 1] * alpha_scaled / n;
    }
    
    // Construct identity matrix explicitly scoped
    arma::sp_mat I(Q.n_rows, Q.n_cols);
    I.eye();
    arma::sp_mat P = I + (1.0 / alpha) * Q;
    
    arma::sp_mat v = x; 
    
    for (int step = 0; step < s; ++step) {
        arma::sp_mat curr_P_power = v; 
        arma::sp_mat v_next = weights[0] * curr_P_power;
        
        for (int n = 1; n <= m; ++n) {
            curr_P_power = curr_P_power * P; 
            v_next += weights[n] * curr_P_power;
        }
        
        v = v_next;
        
        if (tolerance > 0.0) {
            arma::sp_mat::iterator it = v.begin();
            arma::sp_mat::iterator it_end = v.end();
            while (it != it_end) {
                if (std::abs(*it) < tolerance) {
                    v(it.row(), it.col()) = 0.0;
                }
                ++it;
            }
            v.clean(0.0); 
        }
    }
    
    return v;
}