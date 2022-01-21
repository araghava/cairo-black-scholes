# Cairo Black-Scholes Library

Black-Scholes library implemented as a Cairo smart contract.

All inputs, outputs, and internal calculations use 27-digit fixed-point numbers.

## Library

This library implements the interface described
[here](https://blog.lyra.finance/cairo-developer-grant/). The external functions
exposed allow you to retrieve the Black-Scholes call and put option prices and
the option greeks, given the expiration time, volatility, spot price,
strike price, and interest rate.

## Validation

The pytest deploys the smart contract in a local test environment and queries
for the call and put option prices with randomly generated test inputs.
Each test input consists of
(expiration time, volatility, spot price, strike price, interest rate).

The results are compared against the options prices as computed by the
[py\_vollib](https://github.com/vollib/py_vollib) library. The test validates
that the error from the Cairo library and the py\_vollib libary is minimal.
Option greeks were validated against online tools such as [this
one](https://goodcalculators.com/black-scholes-calculator/).

## Test Instructions

1. First make sure you can run a basic StarkNet contract unit test (see
   [here](https://www.cairo-lang.org/docs/hello_starknet/unit_tests.html)).
2. Install py\_vollib: ```pip install py_vollib```.
3. Clone this repo and run ```pytest black_scholes_test.py```.

## SHARP

You can send a proof to the Shared Prover (SHARP) and check that the computation
was verified on-chain by navigating to the sharp/ directory and running:

```cairo-sharp submit --source black_scholes.cairo --program_input input.json```
```cairo-sharp status {JOB_KEY}```

When the status is "PROCESSED", you can check to see if it is verified on chain.

```cairo-sharp is_verified {FACT} --node_url https://goerli-light.eth.linkpool.io/```
