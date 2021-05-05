Run evaluations in directory eval
Store python codes in directory code

1. Generate time series files

Use the Dropbox datasets [1] and generate the time series files for evaluations

$ cd eval
$ ../code/get_tseries.py -i pop1.txt -o Pop1


2. Run evalutation

Use parameter "help" for information

$ ../code/run_evalutation.py help 

See the steps below for the complete evaluation.
Copy the out file and paste in a data sheet.

../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 0 > online_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 1 > svmr_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 2 > svml_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 3 > svms_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 4 > rf_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 5 > knn_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 6 > dct_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 7 > lr_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 8 > vh_pop1.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -c 9 > vs_pop1.out &

grep "Perform " online_pop1.out > out
echo >> out
grep "Perform " svmr_pop1.out >> out
echo >> out
grep "Perform " svml_pop1.out >> out
echo >> out
grep "Perform " svms_pop1.out >> out
echo >> out
grep "Perform " rf_pop1.out >> out
echo >> out
grep "Perform " knn_pop1.out >> out
echo >> out
grep "Perform " dct_pop1.out >> out
echo >> out
grep "Perform " lr_pop1.out >> out
echo >> out
grep "Perform " vh_pop1.out >> out
echo >> out
grep "Perform " vs_pop1.out >> out

../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 0 > online_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 1 > svmr_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 2 > svml_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 3 > svms_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 4 > rf_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 5 > knn_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 6 > dct_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 7 > lr_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 8 > vh_pop1_sweight.out &
../code/run_evaluation_test17.py -l Pop1 -p . -w 5 -s 1 -c 9 > vs_pop1_sweight.out &

grep "Perform " online_pop1_sweight.out > sweight.out
echo >> sweight.out
grep "Perform " svmr_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " svml_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " svms_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " rf_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " knn_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " dct_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " lr_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " vh_pop1_sweight.out >> sweight.out
echo >> sweight.out
grep "Perform " vs_pop1_sweight.out >> sweight.out



