#include "poly.h"
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <thread>
#include <mutex>
#include <complex>
#include <cmath>

// Define PI and Complex type
const double PI = acos(-1);
using Complex = std::complex<double>;

class polynomial_impl
{
public:
    std::map<power, coeff, std::greater<>> terms; 
    power highest_degree = 0;
    bool degree_cached = false;

    polynomial_impl()
    {
        terms[0] = 0;
        highest_degree = 0;
        degree_cached = true;
    }

    polynomial_impl(const polynomial_impl &other) = default;

    template <typename Iter>
    polynomial_impl(Iter begin, Iter end)
    {
        for (auto i = begin; i != end; ++i)
        {
            if (i->second != 0)
            {
                terms[i->first] += i->second;
            }
        }
        canonicalize();
    }

    void canonicalize()
    {
        for (auto i = terms.begin(); i != terms.end();)
        {
            if (i->second == 0)
            {
                i = terms.erase(i);
            }
            else
            {
                ++i;
            }
        }
        if (terms.empty())
        {
            terms[0] = 0;
        }
        
        if (!terms.empty()) {
            highest_degree = terms.begin()->first;
        } else {
            highest_degree = 0;
        }
        degree_cached = true;
    }
    
    // Helper functions for FFT implementation
    size_t next_power_of_two(size_t n) const {
        size_t pow_two = 1;
        while (pow_two < n) {
            pow_two <<= 1;
        }
        return pow_two;
    }

    std::vector<Complex> terms_to_coefficients(const std::map<power, coeff, std::greater<>>& terms_map, size_t n) const {
        std::vector<Complex> coeffs(n, 0);
        for (const auto& [p, c] : terms_map) {
            if (p < n)
                coeffs[p] = c;
        }
        return coeffs;
    }

    std::map<power, coeff, std::greater<>> coefficients_to_terms(const std::vector<Complex>& coeffs) const {
        std::map<power, coeff, std::greater<>> result_terms;
        for (size_t i = 0; i < coeffs.size(); ++i) {
            int c = static_cast<int>(std::round(coeffs[i].real()));
            if (c != 0) {
                result_terms[i] = c;
            }
        }
        return result_terms;
    }

    void fft(std::vector<Complex>& a, bool invert, int depth = 0) const {
        size_t n = a.size();
        if (n == 1)
            return;

        // Divide: separate even and odd indices
        std::vector<Complex> a0(n / 2), a1(n / 2);
        for (size_t i = 0; 2 * i < n; ++i) {
            a0[i] = a[2*i];
            a1[i] = a[2*i + 1];
        }

        // Conquer: recursively perform FFT on even and odd parts
        if (depth <= 3) {
            // Only use threads for the top few levels of recursion
            std::thread t1(&polynomial_impl::fft, this, std::ref(a0), invert, depth + 1);
            std::thread t2(&polynomial_impl::fft, this, std::ref(a1), invert, depth + 1);
            t1.join();
            t2.join();
        } else {
            // After reaching a certain depth, switch to sequential execution
            fft(a0, invert, depth + 1);
            fft(a1, invert, depth + 1);
        }

        // Combine: compute the FFT result for the current level
        double angle = 2 * PI / n * (invert ? -1 : 1);
        Complex w(1), wn(std::cos(angle), std::sin(angle));
        for (size_t i = 0; 2 * i < n; ++i) {
            a[i] = a0[i] + w * a1[i];
            a[i + n/2] = a0[i] - w * a1[i];
            if (invert) {
                a[i] /= 2;
                a[i + n/2] /= 2;
            }
            w *= wn;
        }
    }

    std::map<power, coeff, std::greater<>> multiply_fft(
        const std::map<power, coeff, std::greater<>>& lhs_terms,
        const std::map<power, coeff, std::greater<>>& rhs_terms) const {
        
        // Get the maximum degree
        size_t max_power = 0;
        if (!lhs_terms.empty()) max_power = std::max(max_power, lhs_terms.begin()->first);
        if (!rhs_terms.empty()) max_power = std::max(max_power, rhs_terms.begin()->first);
        
        // Determine FFT size
        size_t n = next_power_of_two((max_power + 1) * 2);
        
        // Convert to coefficient vectors
        std::vector<Complex> fa = terms_to_coefficients(lhs_terms, n);
        std::vector<Complex> fb = terms_to_coefficients(rhs_terms, n);
        
        // Perform FFT
        fft(fa, false);
        fft(fb, false);
        
        // Multiply point-wise
        for (size_t i = 0; i < n; ++i) {
            fa[i] *= fb[i];
        }
        
        // Perform inverse FFT
        fft(fa, true);
        
        // Convert back to terms
        return coefficients_to_terms(fa);
    }
};

polynomial::polynomial() : pimpl(std::make_shared<polynomial_impl>()) {}

polynomial::polynomial(const polynomial &other) : pimpl(std::make_shared<polynomial_impl>(*other.pimpl)) {}

polynomial &polynomial::operator=(const polynomial &other)
{
    if (this != &other)
    {
        pimpl = std::make_shared<polynomial_impl>(*other.pimpl);
    }
    return *this;
}

template <typename Iter>
polynomial::polynomial(Iter begin, Iter end) : pimpl(std::make_shared<polynomial_impl>(begin, end)) {}

size_t polynomial::find_degree_of()
{
    if (pimpl->degree_cached) {
        return pimpl->highest_degree;
    }
    
    if (pimpl->terms.empty())
    {
        pimpl->highest_degree = 0;
    } else {
        pimpl->highest_degree = pimpl->terms.begin()->first;
    }
    pimpl->degree_cached = true;
    return pimpl->highest_degree;
}

std::vector<std::pair<power, coeff>> polynomial::canonical_form() const
{
    std::vector<std::pair<power, coeff>> result;
    result.reserve(pimpl->terms.size());
    for (auto [p, c] : pimpl->terms)
    {
        result.emplace_back(p, c);
    }
    if (result.empty())
    {
        result.emplace_back(0, 0);
    }
    return result;
}

// Addition operators remain unchanged
polynomial operator+(const polynomial &lhs, const polynomial &rhs)
{
    if (lhs.pimpl->terms.size() <= 1 && lhs.pimpl->terms.begin()->second == 0) {
        return rhs;
    }
    if (rhs.pimpl->terms.size() <= 1 && rhs.pimpl->terms.begin()->second == 0) {
        return lhs;
    }

    polynomial result = lhs;
    for (auto [p, c] : rhs.pimpl->terms)
    {
        result.pimpl->terms[p] += c;
    }
    result.pimpl->canonicalize();
    return result;
}

polynomial operator+(const polynomial &lhs, int rhs)
{
    if (rhs == 0) return lhs;

    polynomial result = lhs;
    result.pimpl->terms[0] += rhs;
    result.pimpl->canonicalize();
    return result;
}

polynomial operator+(int lhs, const polynomial &rhs)
{
    return rhs + lhs;
}

polynomial operator*(const polynomial &lhs, int rhs)
{
    if (rhs == 0) {
        polynomial result;
        return result;
    }
    if (rhs == 1) return lhs;

    polynomial result;
    result.pimpl->terms.clear();
    for (auto [p, c] : lhs.pimpl->terms)
    {
        result.pimpl->terms[p] = c * rhs;
    }
    result.pimpl->canonicalize();
    return result;
}

polynomial operator*(int lhs, const polynomial &rhs)
{
    return rhs * lhs;
}

struct ThreadData
{
    const std::vector<std::pair<power, coeff>>* lhs_terms;
    const std::vector<std::pair<power, coeff>>* rhs_terms;
    size_t lhs_start, lhs_end;
    size_t rhs_start, rhs_end;
    std::map<power, coeff> local_result;
};

void* multiply_worker(void* arg)
{
    ThreadData* data = static_cast<ThreadData*>(arg);
    const auto& lhs = *(data->lhs_terms);
    const auto& rhs = *(data->rhs_terms);
    
    for (size_t i = data->lhs_start; i < data->lhs_end; ++i) {
        const auto& [p1, c1] = lhs[i];
        
        for (size_t j = data->rhs_start; j < data->rhs_end; ++j) {
            const auto& [p2, c2] = rhs[j];
            data->local_result[p1 + p2] += c1 * c2;
        }
    }
    return nullptr;
}

polynomial operator*(const polynomial &lhs, const polynomial &rhs)
{
    // Fast path for trivial cases
    if (lhs.pimpl->terms.size() <= 1 && lhs.pimpl->terms.begin()->second == 0) {
        return lhs;
    }
    if (rhs.pimpl->terms.size() <= 1 && rhs.pimpl->terms.begin()->second == 0) {
        return rhs;
    }
    
    // Fast path for single-term polynomials
    if (lhs.pimpl->terms.size() == 1) {
        auto [p1, c1] = *lhs.pimpl->terms.begin();
        polynomial result;
        result.pimpl->terms.clear();
        for (auto [p2, c2] : rhs.pimpl->terms) {
            result.pimpl->terms[p1 + p2] = c1 * c2;
        }
        result.pimpl->canonicalize();
        return result;
    }
    if (rhs.pimpl->terms.size() == 1) {
        auto [p2, c2] = *rhs.pimpl->terms.begin();
        polynomial result;
        result.pimpl->terms.clear();
        for (auto [p1, c1] : lhs.pimpl->terms) {
            result.pimpl->terms[p1 + p2] = c1 * c2;
        }
        result.pimpl->canonicalize();
        return result;
    }

    // Calculate maximum degree for both polynomials
    size_t lhs_degree = lhs.pimpl->terms.empty() ? 0 : lhs.pimpl->terms.begin()->first;
    size_t rhs_degree = rhs.pimpl->terms.empty() ? 0 : rhs.pimpl->terms.begin()->first;
    
    // Calculate work size and sparsity
    size_t work_size = lhs.pimpl->terms.size() * rhs.pimpl->terms.size();
    double sparsity = (double)(lhs.pimpl->terms.size() + rhs.pimpl->terms.size()) / 
                      (lhs_degree + rhs_degree + 2);
    
    // Use direct multiplication for small or sparse polynomials
    if (work_size < 1000 || sparsity < 0.1 || 
        lhs.pimpl->terms.size() <= 50 || rhs.pimpl->terms.size() <= 50) {
        
        polynomial result;
        result.pimpl->terms.clear();
        
        for (const auto& [p1, c1] : lhs.pimpl->terms) {
            for (const auto& [p2, c2] : rhs.pimpl->terms) {
                result.pimpl->terms[p1 + p2] += c1 * c2;
            }
        }
        
        result.pimpl->canonicalize();
        return result;
    }
    
    // For large, dense polynomials, use FFT multiplication
    polynomial result;
    result.pimpl->terms = lhs.pimpl->multiply_fft(lhs.pimpl->terms, rhs.pimpl->terms);
    result.pimpl->canonicalize();
    return result;
}

polynomial operator%(const polynomial &a, const polynomial &b)
{
    if (b.pimpl->terms.empty() || (b.pimpl->terms.size() == 1 && b.pimpl->terms.begin()->second == 0)) {
        throw std::runtime_error("Divide by zero polynomial");
    }

    if (a.pimpl->terms.empty() || 
        (a.pimpl->terms.begin()->first < b.pimpl->terms.begin()->first)) {
        return a; // a is already the remainder
    }

    polynomial remainder = a;
    size_t divisor_deg = b.pimpl->terms.begin()->first;
    coeff divisor_lead = b.pimpl->terms.begin()->second;

    std::vector<std::pair<power, coeff>> divisor_terms(b.pimpl->terms.begin(), b.pimpl->terms.end());

    while (!remainder.pimpl->terms.empty() && 
           remainder.pimpl->terms.begin()->first >= divisor_deg && 
           remainder.pimpl->terms.begin()->second != 0)
    {
        power deg_diff = remainder.pimpl->terms.begin()->first - divisor_deg;
        coeff coeff_ratio = remainder.pimpl->terms.begin()->second / divisor_lead;

        for (const auto& [p, c] : divisor_terms) {
            remainder.pimpl->terms[p + deg_diff] -= c * coeff_ratio;
        }
        remainder.pimpl->canonicalize();
    }

    return remainder;
}

void polynomial::print() const {
    auto terms = canonical_form();
    for (const auto& [p, c] : terms) {
        std::cout << c << "x^" << p << " ";
    }
    std::cout << std::endl;
}

template polynomial::polynomial(
    std::vector<std::pair<power, coeff>>::iterator, 
    std::vector<std::pair<power, coeff>>::iterator
);