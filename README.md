# Cairo Black-Scholes Library

Black-Scholes library implemented as a Cairo smart contract
(```black_scholes_contract.cairo```).

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
3. Clone this repo and run ```pytest --capture=no black_scholes_test.py```.

### Sample Test Output

Example output from running the pytest:
```
Input 0:
t_annualised: 1.80990 years
volatility: 13.38042%
spot price: $175.83945
strike price: $210.77366
interest rate: 26.10447%

Result 0:
Computed call price: $45.03733, Expected call price: $45.03732
Computed put price: $0.60770, Expected put price: $0.60769

Input 1:
t_annualised: 2.42301 years
volatility: 4.32787%
spot price: $788.69543
strike price: $209.41947
interest rate: 38.54290%

Result 1:
Computed call price: $706.38983, Expected call price: $706.38983
Computed put price: $0.00000, Expected put price: $0.00000

Input 2:
t_annualised: 1.22038 years
volatility: 32.02755%
spot price: $366.68875
strike price: $513.10870
interest rate: 28.01244%

Result 2:
Computed call price: $52.42110, Expected call price: $52.42097
Computed put price: $50.27003, Expected put price: $50.26990

...
```

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
