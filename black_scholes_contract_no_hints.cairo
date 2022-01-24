%lang starknet

%builtins range_check

from starkware.cairo.common.math import abs_value, assert_nn, assert_le, unsigned_div_rem, signed_div_rem, assert_in_range, sqrt
from starkware.cairo.common.math_cmp import is_le, is_in_range
from starkware.cairo.common.pow import pow
from starkware.cairo.common.serialize import serialize_word

# This library uses fixed-point arithmetic with 27-digit precision for accurate
# internal calculations. Extra care must be taken when multiplying/dividing.
# For example, the product x * y must later be divided by UNIT to remove the
# extra factor introduced by multiplication.
const UNIT = 10 ** 27

# sqrt(2*pi) in terms of UNIT.
const SQRT_TWOPI = 2506628274631000543434113024

# Boundaries on the input to std_normal_cdf. This helps overflow and out of
# range errors in internal calculations (like exp()). The "real" values of
# cdf(-5) and cdf(5) are very close to 0 and 1, respectively.
const MIN_CDF_INPUT = -5 * UNIT
const MAX_CDF_INPUT = 5 * UNIT

# Minimum values to avoid divide-by-zero (one second and 0.0001%, respectively).
const MIN_T_ANNUALISED = 31709791983764586496
const MIN_VOLATILITY = UNIT / 10000
const DIV_BOUND = (2 ** 128) / 2

# Returns y, the exponent of x.
# Uses first 50 terms of taylor series expansion centered at 0.
func exp{range_check_ptr}(x) -> (y):
    alloc_locals

    if x == 0:
        return (y=1)
    end

    let (local t2, _) = signed_div_rem(x*x, 2*UNIT, DIV_BOUND)
    let (local t3, _) = signed_div_rem(t2*x, 3*UNIT, DIV_BOUND)
    let (local t4, _) = signed_div_rem(t3*x, 4*UNIT, DIV_BOUND)
    let (local t5, _) = signed_div_rem(t4*x, 5*UNIT, DIV_BOUND)
    let (local t6, _) = signed_div_rem(t5*x, 6*UNIT, DIV_BOUND)
    let (local t7, _) = signed_div_rem(t6*x, 7*UNIT, DIV_BOUND)
    let (local t8, _) = signed_div_rem(t7*x, 8*UNIT, DIV_BOUND)
    let (local t9, _) = signed_div_rem(t8*x, 9*UNIT, DIV_BOUND)
    let (local t10, _) = signed_div_rem(t9*x, 10*UNIT, DIV_BOUND)
    let (local t11, _) = signed_div_rem(t10*x, 11*UNIT, DIV_BOUND)
    let (local t12, _) = signed_div_rem(t11*x, 12*UNIT, DIV_BOUND)
    let (local t13, _) = signed_div_rem(t12*x, 13*UNIT, DIV_BOUND)
    let (local t14, _) = signed_div_rem(t13*x, 14*UNIT, DIV_BOUND)
    let (local t15, _) = signed_div_rem(t14*x, 15*UNIT, DIV_BOUND)
    let (local t16, _) = signed_div_rem(t15*x, 16*UNIT, DIV_BOUND)
    let (local t17, _) = signed_div_rem(t16*x, 17*UNIT, DIV_BOUND)
    let (local t18, _) = signed_div_rem(t17*x, 18*UNIT, DIV_BOUND)
    let (local t19, _) = signed_div_rem(t18*x, 19*UNIT, DIV_BOUND)
    let (local t20, _) = signed_div_rem(t19*x, 20*UNIT, DIV_BOUND)
    let (local t21, _) = signed_div_rem(t20*x, 21*UNIT, DIV_BOUND)
    let (local t22, _) = signed_div_rem(t21*x, 22*UNIT, DIV_BOUND)
    let (local t23, _) = signed_div_rem(t22*x, 23*UNIT, DIV_BOUND)
    let (local t24, _) = signed_div_rem(t23*x, 24*UNIT, DIV_BOUND)
    let (local t25, _) = signed_div_rem(t24*x, 25*UNIT, DIV_BOUND)
    let (local t26, _) = signed_div_rem(t25*x, 26*UNIT, DIV_BOUND)
    let (local t27, _) = signed_div_rem(t26*x, 27*UNIT, DIV_BOUND)
    let (local t28, _) = signed_div_rem(t27*x, 28*UNIT, DIV_BOUND)
    let (local t29, _) = signed_div_rem(t28*x, 29*UNIT, DIV_BOUND)
    let (local t30, _) = signed_div_rem(t29*x, 30*UNIT, DIV_BOUND)
    let (local t31, _) = signed_div_rem(t30*x, 31*UNIT, DIV_BOUND)
    let (local t32, _) = signed_div_rem(t31*x, 32*UNIT, DIV_BOUND)
    let (local t33, _) = signed_div_rem(t32*x, 33*UNIT, DIV_BOUND)
    let (local t34, _) = signed_div_rem(t33*x, 34*UNIT, DIV_BOUND)
    let (local t35, _) = signed_div_rem(t34*x, 35*UNIT, DIV_BOUND)
    let (local t36, _) = signed_div_rem(t35*x, 36*UNIT, DIV_BOUND)
    let (local t37, _) = signed_div_rem(t36*x, 37*UNIT, DIV_BOUND)
    let (local t38, _) = signed_div_rem(t37*x, 38*UNIT, DIV_BOUND)
    let (local t39, _) = signed_div_rem(t38*x, 39*UNIT, DIV_BOUND)
    let (local t40, _) = signed_div_rem(t39*x, 40*UNIT, DIV_BOUND)
    let (local t41, _) = signed_div_rem(t40*x, 41*UNIT, DIV_BOUND)
    let (local t42, _) = signed_div_rem(t41*x, 42*UNIT, DIV_BOUND)
    let (local t43, _) = signed_div_rem(t42*x, 43*UNIT, DIV_BOUND)
    let (local t44, _) = signed_div_rem(t43*x, 44*UNIT, DIV_BOUND)
    let (local t45, _) = signed_div_rem(t44*x, 45*UNIT, DIV_BOUND)
    let (local t46, _) = signed_div_rem(t45*x, 46*UNIT, DIV_BOUND)
    let (local t47, _) = signed_div_rem(t46*x, 47*UNIT, DIV_BOUND)
    let (local t48, _) = signed_div_rem(t47*x, 48*UNIT, DIV_BOUND)
    let (local t49, _) = signed_div_rem(t48*x, 49*UNIT, DIV_BOUND)
    let (local t50, _) = signed_div_rem(t49*x, 50*UNIT, DIV_BOUND)

    let sum = (UNIT + x + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 +
               t12 + t13 + t14 + t15 + t16 + t17 + t18 + t19 + t20 + t21 + t22 +
               t23 + t24 + t25 + t26 + t27 + t28 + t29 + t30 + t31 + t32 + t33 +
               t34 + t35 + t36 + t37 + t38 + t39 + t40 + t41 + t42 + t43 + t44 +
               t45 + t46 + t47 + t48 + t49 + t50)

    return (y=sum)
end

# Used for below ln function approximation.
func msb{range_check_ptr}(x) -> (y):
    let (res) = is_le(x, UNIT)
    if res == 1:
        return (y=0)
    end
    let (div, _) = unsigned_div_rem(x, 2)
    let (rest) = msb(div)
    let ans = 1 + rest
    return (y=ans)
end

# Returns y, the natural logarithm of x.
# Uses numerical approximation (Remez algorithm).
func ln{range_check_ptr}(x) -> (y):
    alloc_locals

    if x == 1:
        return (y=0)
    end

    let (is_frac) = is_le(x, UNIT - 1)
    if is_frac == 1:
        # ln(1/x) = -ln(x)
        let (div, _) = unsigned_div_rem(UNIT, x)
        let (rec) = ln(UNIT * div)
        return (y=-rec)
    end

    let (local b) = msb(x)
    let (divisor) = pow(2, b - 1)
    let (norm, _) = unsigned_div_rem(x, divisor)

    const b1 = -56570851000000000000000000
    const b2 = 447179550000000000000000000
    const b3 = -1469956800000000000000000000
    const b4 = 2821202600000000000000000000
    const b5 = -1741793900000000000000000000

    let (d1, _) = signed_div_rem(b1 * norm, UNIT, DIV_BOUND)
    let (d2, _) = signed_div_rem((b2 + d1) * norm, UNIT, DIV_BOUND)
    let (d3, _) = signed_div_rem((b3 + d2) * norm, UNIT, DIV_BOUND)
    let (d4, _) = signed_div_rem((b4 + d3) * norm, UNIT, DIV_BOUND)
    let d5 = d4 + b5
    let res = d5 + (b - 1) * 693147180559945309417232121
    return (y=res)
end

# Returns y, standard normal distribution at x.
# This computes e^(-x^2/2) / sqrt(2*pi).
func std_normal{range_check_ptr}(x) -> (y):
    let (x_squared_over_two, _) = unsigned_div_rem(x * x, UNIT * 2)
    let (exponent_term) = exp(-x_squared_over_two)
    let (div, _) = unsigned_div_rem(UNIT * exponent_term, SQRT_TWOPI)
    return (y=div)
end

# Returns y, cumulative normal distribution at x.
# Computed using a curve-fitting approximation.
func std_normal_cdf{range_check_ptr}(x) -> (y):
    alloc_locals

    # If input is less than MIN_CDF_INPUT, return 0.
    let (lower) = is_le(x, MIN_CDF_INPUT)
    if lower == 1:
        return (y=0)
    end

    # If input is greater than MAX_CDF_INPUT, return UNIT.
    let (upper) = is_in_range(x, MIN_CDF_INPUT, MAX_CDF_INPUT)
    if upper == 0:
        return (y=UNIT)
    end

    const MUL = 10**7
    const b1 = 3193815
    const b2 = -3565638
    const b3 = 17814780
    const b4 = -18212560
    const b5 = 13302740
    const p = 2316419
    const c2 = 3989423

    let (abs_x)= abs_value(x)
    let (div_t, _) = unsigned_div_rem(p * abs_x, UNIT)
    local t = MUL + div_t

    let (x_squared_over_two, _) = unsigned_div_rem(x * x, UNIT * 2)
    let (exponent_term) = exp(x_squared_over_two)

    let (local b, _) = unsigned_div_rem(UNIT * c2, exponent_term)
    let (local d1, _) = signed_div_rem(b5*MUL, t, DIV_BOUND)
    let (local d2, _) = signed_div_rem((b4 + d1)*MUL, t, DIV_BOUND)
    let (local d3, _) = signed_div_rem((b3 + d2)*MUL, t, DIV_BOUND)
    let (local d4, _) = signed_div_rem((b2 + d3)*MUL, t, DIV_BOUND)
    let (local d5, _) = signed_div_rem((b1 + d4)*MUL, t, DIV_BOUND)
    local prob = b * d5

    let (res) = is_le(x, 0)
    jmp neg if res != 0
    let (local pos_ans, _) = unsigned_div_rem(UNIT * (MUL*MUL - prob), MUL*MUL)
    return (y=pos_ans)

    neg:
    let (local neg_ans, _) = unsigned_div_rem(UNIT * prob, MUL*MUL)
    return (y=neg_ans)
end

# Returns the internal Black-Scholes coefficients.
func d1d2{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (d1, d2):
    alloc_locals

    # Case where t_annualised is too low.
    let (res_t_annualised) = is_le(t_annualised, MIN_T_ANNUALISED - 1)
    if res_t_annualised == 1:
        return d1d2(MIN_T_ANNUALISED, volatility, spot, strike, rate)
    end

    # Case where volatility is too low.
    let (res_volatility) = is_le(volatility, MIN_VOLATILITY - 1)
    if res_volatility == 1:
        return d1d2(t_annualised, MIN_VOLATILITY, spot, strike, rate)
    end

    let (sqrt_t_annualised) = sqrt(UNIT * t_annualised)
    let (local vt_sqrt, _) = unsigned_div_rem(volatility * sqrt_t_annualised, UNIT)
    let (local spot_over_strike, _) = unsigned_div_rem(UNIT * spot, strike)
    let (local log) = ln(spot_over_strike)
    let (local vol2, _) = unsigned_div_rem(volatility * volatility, UNIT * 2)

    local vol2_add = vol2 + rate
    let (local v2t, _) = unsigned_div_rem(vol2_add * t_annualised, UNIT)
    let (local d1, _) = signed_div_rem(UNIT * (log + v2t), vt_sqrt, DIV_BOUND)
    let d2 = d1 - vt_sqrt
    return (d1, d2)
end

# Returns the option's call and put delta value.
@external
func delta{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_delta, put_delta):
    let (d1, _) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (call_delta) = std_normal_cdf(d1)
    let put_delta = call_delta - UNIT
    return (call_delta, put_delta)
end

# Returns the option's gamma value.
@external
func gamma{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (gamma):
    alloc_locals

    let (d1, _) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (local std_normal_d1) = std_normal(d1)

    let (sqrt_t) = sqrt(UNIT * t_annualised)
    let (vol_sqrt_t, _) = unsigned_div_rem(volatility * sqrt_t, UNIT)
    let (spot_mul, _) = unsigned_div_rem(spot * vol_sqrt_t, UNIT)
    let (gamma, _) = unsigned_div_rem(UNIT * std_normal_d1, spot_mul)
    return (gamma)
end

# Returns the option's vega value.
@external
func vega{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (vega):
    alloc_locals

    let (local sqrt_t) = sqrt(UNIT * t_annualised)
    let (d1, _) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (std_normal_d1) = std_normal(d1)
    let (std_normal_d1_spot, _) = unsigned_div_rem(std_normal_d1 * spot, UNIT)
    let (vega, _) = unsigned_div_rem(sqrt_t * std_normal_d1_spot, UNIT)
    return (vega)
end

# Returns the option's call and put rho value.
@external
func rho{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_rho, put_rho):
    alloc_locals

    let (local strike_t, _) = unsigned_div_rem(strike * t_annualised, UNIT)
    let (rt, _) = unsigned_div_rem(rate * t_annualised, UNIT)
    let (exponent_term) = exp(-rt)
    let (lhs, _) = unsigned_div_rem(strike_t * exponent_term, UNIT)

    let (_, local d2) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (local d2_cdf) = std_normal_cdf(d2)
    let (local d2_cdf_neg) = std_normal_cdf(-d2)
    let (local call_rho, _) = unsigned_div_rem(lhs * d2_cdf, UNIT)
    let (local put_rho, _) = unsigned_div_rem(lhs * d2_cdf_neg, UNIT)

    return (call_rho=call_rho, put_rho=-put_rho)
end

# Returns the option's call and put theta value.
@external
func theta{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_theta, put_theta):
    alloc_locals

    let (local d1, local d2) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (local std_norm_d1) = std_normal(d1)
    let (local d2_cdf_pos) = std_normal_cdf(d2)
    let (local d2_cdf_neg) = std_normal_cdf(-d2)

    let (local rt, _) = unsigned_div_rem(rate * t_annualised, UNIT)
    let (local exponent_term) = exp(-rt)
    let (local c1, _) = unsigned_div_rem(strike * rate, UNIT)
    let (local c2, _) = unsigned_div_rem(exponent_term * c1, UNIT)
    let (local c3, _) = unsigned_div_rem(d2_cdf_pos * c2, UNIT)
    let (local p3, _) = unsigned_div_rem(d2_cdf_neg * c2, UNIT)

    let (local sqrt_t) = sqrt(UNIT * t_annualised)
    let (spot_vol, _) = unsigned_div_rem(spot * volatility, UNIT)
    let (local c4, _) = unsigned_div_rem(UNIT * spot_vol, 2 * sqrt_t)
    let (local c5, _) = unsigned_div_rem(std_norm_d1 * c4, UNIT)

    local call_theta_t = -c5 - c3
    local put_theta_t = -c5 + p3

    # Divide by 365 as thetas are per-day.
    let (local call_theta, _) = signed_div_rem(call_theta_t, 365, DIV_BOUND)
    let (local put_theta, _) = signed_div_rem(put_theta_t, 365, DIV_BOUND)

    return (call_theta, put_theta)
end

# Returns the call and put options prices.
@external
func option_prices{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_price, put_price):
    alloc_locals

    let (ann_rate, _) = unsigned_div_rem(rate * t_annualised, UNIT)
    let (exponent_term) = exp(-ann_rate)
    let (local strike_pv, _) = unsigned_div_rem(strike * exponent_term, UNIT)

    let (local d1, local d2) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (local  cdf_d1) = std_normal_cdf(d1)
    let (local cdf_d2) = std_normal_cdf(d2)
    let (local spot_nd1, _) = unsigned_div_rem(spot * cdf_d1, UNIT)
    let (local strike_nd2, _) = unsigned_div_rem(strike_pv * cdf_d2, UNIT)

    local call_price = spot_nd1 - strike_nd2
    local put_price = call_price + strike_pv - spot
    return (call_price, put_price)
end
