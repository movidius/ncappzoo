Compile with:

    mvNCCompile -s 12 -w weights.caffemodel network.prototxt -o compiled.graph

Profile with:

    mvNCProfile -s 12 -w weights.caffemodel network.prototxt

Generate inference with:

    cd supporting/
    python3 inferences.py -h
    python3 inferences.py /path/to/movidius/folder

Create the submission zip file

    zip -r submission.zip compiled.graph inferences.csv network.prototxt weights.caffemodel supporting/
