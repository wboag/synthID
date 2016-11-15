#for i in 100 200 500 1000 2000 4000 8000 ; do
for i in 1000 2000 4000 8000 ; do
#for i in 10 50 ; do
    echo "$i"
    #time python baselines/train_crf.py "data/crunch/train/txt/record-*.txt" "data/crunch/train/tags/record-*.tags" "models/crunch-train-$i-crf.model" --limit $i >/dev/null &
    #mkdir data/predictions/crunch-train-$i
    time python baselines/predict_crf.py "data/crunch/test/txt/record-*.txt" data/predictions/crunch-train-$i models/crunch-train-$i-crf.model &
done
