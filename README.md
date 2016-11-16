# synthID
Project for 6.864 with Hassan, Harini, and Sean.

# usage

    $ git clone https://github.com/wboag/synthID.git
    $ cd synthID

    # optional
    $ virtualenv venv_synthID
    $ source venv_synthID/bin/activate

    $ pip install -r requirements.txt

    $ python synth/synthid.py

    $ mkdir data/predictions

    $ python baseline/train_crf.py "data/all/txt/record-88452-*.txt" "data/all/tags/record-88452-*.tags" models/dummy.crf
    $ python baseline/predict_crf.py "data/all/txt/record-88452-*.txt" data/predictions models/dummy.crf

