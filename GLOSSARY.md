# skewencoder

Iterative enhanced sampling ("loxodynamics"): a skewencoder neural CV is retrained each iteration and biased with a one-sided wall, driving an MD system from state to state.

## Language

**Loxodynamics kappa**:
The force constant of the one-sided wall placed on the skewencoder CV in each biased iteration; set as `[loxodynamics] kappa`.
_Avoid_: kappa (unqualified), wall kappa

**Contact wall**:
The upper wall on the van der Waals contact gap that keeps all molecular fragments touching; its force constant is fixed and separate from the loxodynamics kappa.
_Avoid_: vdW wall, vdw_constr wall, fragment wall

**Heavy-atom descriptor**:
A distance between two non-hydrogen atoms, named `<El><i><El><j>` (e.g. `C1Cl2`), generated automatically from the starting geometry; the skewencoder input and the state-detection input.
_Avoid_: heavy atom pair CV
