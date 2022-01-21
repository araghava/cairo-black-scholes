# Cairo Black-Scholes Library

Black-Scholes library implemented as a Cairo smart contract.

All inputs, outputs, and internal calculations use 27-digit fixed-point numbers.

## Validation

The pytest deploys the smart contract in a local test environment and queries
for the call and put option pridess with randomly generated test inputs.
Each test input consists of
(expiration time, volatility, spot price, strike price, interest rate).

The results are compared against the options prices as computed by the
[py\_vollib](https://github.com/vollib/py_vollib) library. The test validates
that the error from the Cairo library and the py\_vollib libary is minimal.

## Test Instructions

1. First make sure you can run a basic StarkNet contract unit test (see
   [here](https://www.cairo-lang.org/docs/hello_starknet/unit_tests.html)).
2. Install py\_vollib: ```pip install py_vollib```.
3. Clone this repo and run ```pytest black_scholes_test.py```.
