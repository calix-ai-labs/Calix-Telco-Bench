# Calix Telco-Bench judge protocol v1

Use this protocol only for **rubric-based** evaluation of Calix Telco-Bench
generations. The current evaluator does not provide the `response` reference
answers present in some release records to the judge, so this protocol does not
produce reference-grounded accuracy. Score factual correctness,
completeness, terminology/command precision, absence of technically false or
fabricated claims, and operational usefulness as integers 0–5. The total is the
sum (0–25).

Store the exact prompt, provider and immutable model identifier when available,
temperature, retry policy, raw judgments, parsed values, and failures. For a
paper claim, validate a stratified subset with blinded networking SMEs and
report judge–human agreement by route and dimension.
