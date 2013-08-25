Original algorithm based on:
Fast Effective Rule Induction
William W. Cohen
AT&T Bell Laboratories
600 Mountain Avenue Murray Hill, NJ 07974
wcohen@research.att.com

The script (src/ripperk.py) handles two phases, learning and classifying, which are described in more detail below.

============================================
Arguments for python ripperk.py
============================================

	The following parameters are required in all phases:

-e: the execution method (learn|classify)
-a: the attribute file location.
-c: the defining attribute in the attribute file (a.k.a what we are trying to predict).
-t: the training/testing file location.
-m: the model file (machine readable results).
-o: the output file (human readable results).

	The following are arguments are optional during the learn phase:

-k: the number of optimizations (default is 2).
-p: pruning or no pruning [1 or 0] (default is 1).

============================================
Learning
============================================

	The learning part of the script reads in training cases and builds a rule model.

	It is crucial that "learn" is passed as the -e parameter (-e learn)!  The script also requires the attribute file (-a), the defining attribute (what the script is trying to predict) (-c), and the training file (-t).  Optional parameters for the learning phase are the number of optimization passes (-k) and whether to prune or not (-p).

	Once the rule model is built, the script will output a machine-readable data model file (-m) and a human-readable text file (-o).  The machine-readable file will be used during classification.  The human-readable file will contain an IF ... THEN ... ELSE structure with primitive equalities so that the user can easily read the rules.

======================
	Usage
======================

	The following are some example use cases.

Learning the restaurant scenario.
> python ripperk.py -e learn -a "../data/restaurant-attr.txt" -c WillWait -t "../data/restaurant-train.txt" -m "../results/restaurant-model.dat" -o "../results/restaurant-model.txt" -k 2 -p 1

Learning the id scenario.
> python ripperk.py -e learn -a "../data/ids-mixed-attr.txt" -c class -t "../data/ids-mixed-train.txt" -m "../results/ids-model.dat" -o "../results/ids-model.txt" -k 2 -p 1


============================================
Classifying
============================================

	The classifying part of the script reads in test cases and trys to accurately predict them based on the rule model learned in the learning phase.

	It is crucial that "classify" is passed as the -e parameter (-e classify)!  The script also requires the attribute file (-a), the defining attribute (what the script is trying to predict) (-c), the testing file (-t), and the machine-readable data model file created during the learn phase (-m).

	Once the test cases are analyzed, the results will be put into a human-readable text file (-o).

======================
	Usage
======================

	The following are some example use cases.

Classifying test cases for the restaurant scenario.
> python ripperk.py -e classify -a "../data/restaurant-attr.txt" -c WillWait -t "../data/restaurant-test.txt" -m "../results/restaurant-model.dat" -o "../results/restaurant-test-results.txt"

Classifying test cases for the id scenario.
> python ripperk.py -e classify -a "../data/ids-mixed-attr.txt" -c class -t "../data/ids-mixed-test.txt" -m "../results/ids-model.dat" -o "../results/ids-test-results.txt"
