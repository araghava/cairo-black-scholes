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

### Test Instructions

1. First make sure you can run a basic StarkNet contract unit test (see
   [here](https://www.cairo-lang.org/docs/hello_starknet/unit_tests.html)).
2. Install py\_vollib: ```pip install py_vollib```.
3. Clone this repo and run ```pytest black_scholes_test.py```.

## SHARP

In the sharp/ folder, you can generate a proof of call option price computation
for a given input (```input.json```) using the Shared Prover (SHARP) and check
that the proof gets verified on-chain by running:

```cairo-sharp submit --source black_scholes.cairo --program_input input.json```

```cairo-sharp status {JOB_KEY}```

When the status is "PROCESSED", you can check to see if it is verified on chain.

```cairo-sharp is_verified {FACT} --node_url https://goerli-light.eth.linkpool.io/```

## Example Results

These are the inputs in ```sharp/input.json```. Note that the input file
represents these values as 27-digit fixed-point numbers.
```
t_annualised = 1 year
volatility = 15%
spot = $300
strike = $250
rate = 3%
```

Run the following command in the sharp/ directory:
```
cairo-compile black_scholes.cairo --output black_scholes_compiled.json && \
    cairo-run --program=black_scholes_compiled.json --print_output \
    --layout=small --program_input=input.json
```

The output returns the call option price of $58.82 (divide below by 10^27).
For displaying the other values (put option price and greeks), please modify
'main' in black_scholes.cairo accordingly.
```
Program output:
  58819767434065242445077191149
```
