# synthID
Project for 6.864 with Hassan, Harini, and Sean.

# setup

    $ git clone https://github.com/wboag/synthID.git
    $ cd synthID

    # optional
    $ virtualenv venv_synthID
    $ source venv_synthID/bin/activate

    $ pip install -r requirements.txt

    # where to store predicted tag files
    $ mkdir data/predictions

    $ python synth/synthid.py

    $ mkdir data/predictions

    $ python deid/crf/train_crf.py "data/all/txt/record-88452-*.txt" "data/all/tags/record-88452-*.tags" models/dummy.crf
    $ python deid/crf/predict_crf.py "data/all/txt/record-88452-*.txt" data/predictions models/dummy.crf

