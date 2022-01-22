import asyncio
import os
import random
import pytest

from py_vollib.black_scholes import black_scholes

from starkware.starknet.compiler.compile import compile_starknet_files
from starkware.starknet.testing.starknet import Starknet

# The path to the contract source code.
CONTRACT_FILE = os.path.join(
    os.path.dirname(__file__), "black_scholes_contract.cairo")

# Precision used in the black scholes Cairo library.
UNIT = 1e27

def get_precise(value):
    return int(UNIT * value)

# Checks accuracy of the option price (within $0.001).
def check_price(got, expected):
  print(got, expected)
  assert(abs(got - expected) < 0.001)

# Returns a random tuple of (t_annualised, volatility, spot, strike, rate)
def get_random_test_input():
    # Random time from 10 minutes to 5 years.
    t_annualised = random.uniform(1/52560.0, 5)
    # Random volatility from 0.01% to 50%.
    volatility = random.uniform(0.0001, 0.5)
    # Random spot price between $0.01 and $1000
    spot = random.uniform(0.01, 10000)
    # Random strike price between $0.01 and $1000
    strike = random.uniform(0.01, 10000)
    # Random interest rate between 0.01% and 50%
    rate = random.uniform(0.0001, 0.5)
    return (t_annualised, volatility, spot, strike, rate)

@pytest.mark.asyncio
async def test_randomized_black_scholes_options_prices():
    # Create a new Starknet class that simulates the StarkNet system.
    starknet = await Starknet.empty()

    # Deploy the contract.
    contract_def = compile_starknet_files(files=[CONTRACT_FILE],
                                          disable_hint_validation=True)
    contract = await starknet.deploy(
        contract_def=contract_def,
    )

    # Number of random tests to run.
    ITERATIONS = 10

    # List of float tuple (t_annualised, volatility, spot, strike, rate).
    test_inputs = []

    # Query the contract for options prices.
    tasks = []
    for i in range(ITERATIONS):
      test_input = get_random_test_input()
      test_inputs.append(test_input)
      tasks.append(contract.option_prices(
          t_annualised=get_precise(test_input[0]),
          volatility=get_precise(test_input[1]),
          spot=get_precise(test_input[2]),
          strike=get_precise(test_input[3]),
          rate=get_precise(test_input[4])).call())

    # Compare call and put prices with the python black scholes library.
    execution_infos = await asyncio.gather(*tasks)
    for i, execution_info in enumerate(execution_infos):
      (got_call, got_put) = (execution_info.result.call_price/UNIT,
                             execution_info.result.put_price/UNIT)

      (exp_call, exp_put) = (
          black_scholes('c', test_inputs[i][2], test_inputs[i][3],
                        test_inputs[i][0], test_inputs[i][4],
                        test_inputs[i][1]),
          black_scholes('p', test_inputs[i][2], test_inputs[i][3],
                        test_inputs[i][0], test_inputs[i][4],
                        test_inputs[i][1]))

      check_price(got_call, exp_call)
      check_price(got_put, exp_put)
