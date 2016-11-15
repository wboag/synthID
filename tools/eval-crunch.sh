#for i in 100 200 500 1000 2000 4000 8000 ; do
#for i in 2000 4000 8000 ; do
for i in 100 1000 ; do
    echo "$i"
    python baselines/eval.py "data/crunch/test/txt/*" "data/predictions/crunch-train-$i/*" "data/crunch/test/tags/*" > results/crunch-$i-train=train-test=test.txt
done
