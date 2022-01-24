# Cairo Black-Scholes Library

Black-Scholes library implemented as a Cairo smart contract
(```black_scholes_contract.cairo```). The address of the contract on the goerli
testnet is: ```0x004747bafa97f4e2c1491df50540e5dda921ad4a229a3a8e7a75dbf860181ae2```.

No un-whitelisted hints were used; functions like ```ln``` and ```exp``` are
implemented with numerical approximations in native Cairo.

All inputs, outputs, and internal calculations use 27-digit fixed-point numbers.

Example usage of contract:
```
Inputs are:
t_annualised = 1 year
volatility = 15%
spot price = $300
strike price = $250
interest rate = 3%

Option price calculation:
starknet call --address 0x004747bafa97f4e2c1491df50540e5dda921ad4a229a3a8e7a75dbf860181ae2 --abi black_scholes_contract_abi.json  --function option_prices --inputs 1000000000000000000000000000 150000000000000000000000000 300000000000000000000000000000 250000000000000000000000000000 30000000000000000000000000

Results are (call price = $58.82, put price = $1.43):
0xbe0e94e51c07cf860555e499 0x49fd4a0ba906f19c624670b

Delta calculation:
starknet call --address 0x004747bafa97f4e2c1491df50540e5dda921ad4a229a3a8e7a75dbf860181ae2 --abi black_scholes_contract_abi.json  --function delta --inputs 1000000000000000000000000000 150000000000000000000000000 300000000000000000000000000000 250000000000000000000000000000 30000000000000000000000000

Results are (call delta = 0.932, put delta = -0.068):
0x302e63a1bd2e76d922c0000 -0x38480283fd98cf55d40000
```

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
t_annualised: 2.26664 years
volatility: 10.56165%
spot price: $326.08126
strike price: $321.88154
interest rate: 32.53044%

Result 0:
Computed call price: $172.09933, Expected call price: $172.09932
Computed put price: $0.00001, Expected put price: $0.00001

Input 1:
t_annualised: 4.92524 years
volatility: 28.79982%
spot price: $747.82545
strike price: $885.09734
interest rate: 4.57839%

Result 1:
Computed call price: $203.70129, Expected call price: $203.70106
Computed put price: $162.28934, Expected put price: $162.28911

Input 2:
t_annualised: 4.15624 years
volatility: 41.86169%
spot price: $144.70197
strike price: $407.66489
interest rate: 17.34711%

Result 2:
Computed call price: $33.43700, Expected call price: $33.43693
Computed put price: $86.97104, Expected put price: $86.97096

...
```
