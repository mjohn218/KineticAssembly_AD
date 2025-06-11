To run RateGrowth model on your mer of interest please make sure your .pwr file mathes the format of the tetramer example.

Key format points:
In current implementation, only parameters, species, and rules sections matter
Lower case letters within parentheses don't matter either



In the Kinetic_simulation notebook make sure to:
-Play around with rates in block 4, make sure monomer rates are equal to each other, dimer rates are equal to each other, etc; not recommended to exceed 10 for max_val and go too close to 0 for min_val. You could manually create a list of length rn._rxn_count with arbitrary values for rates, the code is designed so that upon finding optimal rates in the optimization notebook, you could test their performance to a set of random rates or to a set of equal rates.
-Play around with runtime in block 6 to make sure you are capturing yield peaking in the graph below; larger complexes need more time to reach 100% yield as there will be more plateu stages.
*yield might jump slightly higher than 100 due if your runtime exceeds ~1e8

In the Optimization notebook make sure to:
- play around with pre-optimization reaction rates in block 4. Once again, you dont want extreme values for the rates. 
- you could play arounf with learning rate, but with low value you might take forever, and with high rate you might never converge to the optimal value because auto differentiation step is too large
- for higher mers the tield is not going to be high; as ooulined in the paper, the homorates cannot avoid trapping thus yield at 1 sec (sim_runtime=1) is not close to 100%.




*There is no point in trying homorates for high mers as you cannot avoid trapping no matter how optimized the rates are