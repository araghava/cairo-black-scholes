%builtins output range_check

from starkware.cairo.common.math import abs_value, assert_nn, assert_le, unsigned_div_rem, signed_div_rem
from starkware.cairo.common.math_cmp import is_le, is_in_range
from starkware.cairo.common.serialize import serialize_word

# This library uses fixed-point arithmetic with 27-digit precision for accurate
# internal calculations. Extra care must be taken when multiplying/dividing.
# For example, the product x * y must later be divided by UNIT to remove the
# extra factor introduced by multiplication.
#
# ALL inputs/outputs of this library are in this precision.
const UNIT = 10 ** 27

# sqrt(2*pi) in terms of UNIT.
const SQRT_TWOPI = 2506628274631000543434113024

# Boundaries on the input to std_normal_cdf. This helps overflow and out of
# range errors in internal calculations (like exp()). The "real" values of
# cdf(-5) and cdf(5) are very close to 0 and 1, respectively.
const MIN_CDF_INPUT = -5 * UNIT
const MAX_CDF_INPUT = 5 * UNIT

# Minimum values to avoid divide-by-zero (one second and 0.001%, respectively).
const MIN_T_ANNUALISED = 31709791983764586496
const MIN_VOLATILITY = UNIT / 10000

# Returns y, the exponent of x.
func exp{range_check_ptr}(x) -> (y):
    alloc_locals

    # Use python hint to compute exp.
    local y
    %{
        import math
        from starkware.cairo.common.math_utils import as_int
        value = as_int(ids.x, PRIME)
        ids.y = math.floor(ids.UNIT * math.exp((1.0 * value) / ids.UNIT))
    %}
    return (y)
end

# Returns y, the natural logarithm of x.
func ln{range_check_ptr}(x) -> (y):
    alloc_locals

    # Use python hint to compute ln.
    local y
    local is_positive
    %{
        import math
        value = math.floor(ids.UNIT * math.log(1.0 * ids.x / ids.UNIT))

        # Return value of log can be positive or negative. We can only assign
        # non-negative Cairo memory values, so convert later.
        ids.is_positive = value > 0
        ids.y = value if value > 0 else (PRIME - value)
    %}
    if is_positive == 0:
        return (-y)
    else:
        return (y)
    end
end

# Returns y, the floored square root of x.
func sqrt{range_check_ptr}(x) -> (y):
    alloc_locals

    local y
    %{
        import math
        ids.y = math.floor(math.sqrt(ids.x))
    %}

    return (y)
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
# Computed using a curve-fitting approximation (Hasting's).
func std_normal_cdf{range_check_ptr}(x) -> (y):
    alloc_locals

    let (lower) = is_le(x, MIN_CDF_INPUT)
    if lower == 1:
        return (y=0)
    end

    let (upper) = is_in_range(x, MIN_CDF_INPUT, MAX_CDF_INPUT)
    if upper == 0:
        return (y=UNIT)
    end

    let b1 = 3193815
    let b2 = -3565638
    let b3 = 17814780
    let b4 = -18212560
    let b5 = 13302740
    let p = 2316419
    let c2 = 3989423

    const MUL = 10**7

    let (abs_x) = abs_value(x)
    let (div_t, _) = unsigned_div_rem(p * abs_x, UNIT)
    let t = MUL + div_t

    let (x_squared_over_two, _) = unsigned_div_rem(x * x, UNIT * 2)
    let (exponent_term) = exp(x_squared_over_two)
    let (b, _) = unsigned_div_rem(UNIT * c2, exponent_term)

    let (d1, _) = signed_div_rem(b5*MUL, t, UNIT)
    let (d2, _) = signed_div_rem((b4 + d1)*MUL, t, UNIT)
    let (d3, _) = signed_div_rem((b3 + d2)*MUL, t, UNIT)
    let (d4, _) = signed_div_rem((b2 + d3)*MUL, t, UNIT)
    let (d5, _) = signed_div_rem((b1 + d4)*MUL, t, UNIT)
    local prob = b * d5

    let (res) = is_le(x, 0)
    jmp neg if res != 0
    let (pos_ans, _) = unsigned_div_rem(UNIT * (MUL*MUL - prob), MUL*MUL)
    return (y=pos_ans)

    neg:
    let (neg_ans, _) = unsigned_div_rem(UNIT * prob, MUL*MUL)
    return (y=neg_ans)
end

# Returns the option's call and put delta value.
func delta{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_delta, put_delta):
    let (d1, _) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (call_delta) = std_normal_cdf(d1)
    let put_delta = call_delta - UNIT
    return (call_delta, put_delta)
end

# Returns the option's gamma value.
func gamma{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (gamma):
    let (d1, _) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (std_normal_d1) = std_normal(d1)
    let (sqrt_t) = sqrt(UNIT * t_annualised)
    let (vol_sqrt_t, _) = unsigned_div_rem(volatility * sqrt_t, UNIT)
    let (spot_mul, _) = unsigned_div_rem(spot * vol_sqrt_t, UNIT)
    let (gamma, _) = unsigned_div_rem(UNIT * std_normal_d1, spot_mul)
    return (gamma)
end

# Returns the option's vega value.
func vega{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (vega):
    let (d1, _) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (sqrt_t) = sqrt(UNIT * t_annualised)
    let (std_normal_d1) = std_normal(d1)
    let (std_normal_d1_spot, _) = unsigned_div_rem(std_normal_d1 * spot, UNIT)
    let (vega, _) = unsigned_div_rem(sqrt_t * std_normal_d1_spot, UNIT)
    return (vega)
end

# Returns the option's call and put rho value.
func rho{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_rho, put_rho):
    alloc_locals

    let (_, d2) = d1d2(t_annualised, volatility, spot, strike, rate)
    local nd2 = -d2

    let (d2_cdf) = std_normal_cdf(d2)
    let (d2_cdf_neg) = std_normal_cdf(nd2)

    let (strike_t, _) = unsigned_div_rem(strike * t_annualised, UNIT)
    let (rt, _) = unsigned_div_rem(rate * t_annualised, UNIT)
    let (exponent_term) = exp(-rt)
    let (lhs, _) = unsigned_div_rem(strike_t * exponent_term, UNIT)
    let (call_rho, _) = unsigned_div_rem(lhs * d2_cdf, UNIT)
    let (put_rho, _) = unsigned_div_rem(lhs * d2_cdf_neg, UNIT)

    return (call_rho=call_rho, put_rho=-put_rho)
end

# Returns the option's call and put theta value.
func theta{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_theta, put_theta):
    alloc_locals

    let (d1, d2) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (std_norm_d1) = std_normal(d1)
    let (d2_cdf_pos) = std_normal_cdf(d2)
    let (d2_cdf_neg) = std_normal_cdf(-d2)

    let (rt, _) = unsigned_div_rem(rate * t_annualised, UNIT)
    let (exponent_term) = exp(-rt)
    let (c1, _) = unsigned_div_rem(strike * rate, UNIT)
    let (c2, _) = unsigned_div_rem(exponent_term * c1, UNIT)
    let (c3, _) = unsigned_div_rem(d2_cdf_pos * c2, UNIT)
    let (p3, _) = unsigned_div_rem(d2_cdf_neg * c2, UNIT)

    let (sqrt_t) = sqrt(UNIT * t_annualised)
    let (spot_vol, _) = unsigned_div_rem(spot * volatility, UNIT)
    let (c4, _) = unsigned_div_rem(UNIT * spot_vol, 2 * sqrt_t)
    let (c5, _) = unsigned_div_rem(std_norm_d1 * c4, UNIT)
    let call_theta_t = -c5 - c3
    let put_theta_t = -c5 + p3
    let (call_theta, _) = signed_div_rem(call_theta_t, 365, 10**10*UNIT)
    let (put_theta, _) = signed_div_rem(-c5 + p3, 365, 10**10*UNIT)

    return (call_theta, put_theta)
end

# Returns the internal Black-Scholes coefficients.
func d1d2{range_check_ptr}(
    tAnnualised, volatility, spot, strike, rate) -> (d1, d2):
	alloc_locals

    let (res_tAnnualised) = is_le(tAnnualised, MIN_T_ANNUALISED - 1)
    if res_tAnnualised == 1:
        return d1d2(MIN_T_ANNUALISED, volatility, spot, strike, rate)
    end

    let (res_volatility) = is_le(volatility, MIN_VOLATILITY - 1)
    if res_volatility == 1:
        return d1d2(tAnnualised, MIN_VOLATILITY, spot, strike, rate)
    end

    let (sqrt_tAnnualised) = sqrt(UNIT * tAnnualised)
    let (vt_sqrt, _) = unsigned_div_rem(volatility * sqrt_tAnnualised, UNIT)
    let (spot_over_strike, _) = unsigned_div_rem(UNIT * spot, strike)
    let (log) = ln(spot_over_strike)
    let (vol2, _) = unsigned_div_rem(volatility * volatility, UNIT * 2)
    let vol2_add = vol2 + rate
    let (v2t, _) = unsigned_div_rem(vol2_add * tAnnualised, UNIT)

    let (d1, _) = signed_div_rem(UNIT * (log + v2t), vt_sqrt, 10**10*UNIT)
    let d2 = d1 - vt_sqrt
    return (d1, d2)
end

func option_prices{range_check_ptr}(
    t_annualised, volatility, spot, strike, rate) -> (call_price, put_price):
    alloc_locals

    let (ann_rate, _) = unsigned_div_rem(rate * t_annualised, UNIT)
    let (exponent_term) = exp(-ann_rate)
    let (strike_pv, _) = unsigned_div_rem(strike * exponent_term, UNIT)

    let (d1, d2) = d1d2(t_annualised, volatility, spot, strike, rate)
    let (cdf_d1) = std_normal_cdf(d1)
    let (cdf_d2) = std_normal_cdf(d2)
    let (spot_nd1, _) = unsigned_div_rem(spot * cdf_d1, UNIT)
    let (strike_nd2, _) = unsigned_div_rem(strike_pv * cdf_d2, UNIT)

    let call_price = spot_nd1 - strike_nd2
    let put_price = call_price + strike_pv - spot
    return (call_price, put_price)
end

# Send to SHARP with:
# cairo-sharp submit --source black_scholes.cairo --program_input input.json
func main(output_ptr : felt*, range_check_ptr) -> (output_ptr : felt*, range_check_ptr):
    alloc_locals

    local t_annualised
    local volatility
    local spot
    local strike
    local rate
    %{
        ids.t_annualised = program_input['t_annualised']
        ids.volatility = program_input['volatility']
        ids.spot = program_input['spot']
        ids.strike = program_input['strike']
        ids.rate = program_input['rate']
    %}
    with range_check_ptr:
        let (call_price, _) = option_prices(
            t_annualised, volatility, spot, strike, rate)
    end
    assert [output_ptr] = call_price
    return (output_ptr=output_ptr + 1, range_check_ptr=range_check_ptr)
end
