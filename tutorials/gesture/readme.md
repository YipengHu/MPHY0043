# Surgical Gesture and Skill


## Data
This is a tutorial that uses the public [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) for example applications of gesture recognition and skill assessment.

The JIGSAWS dataset can be downloaded at the [linked site](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/). The dataset will be used in this tutorial is the "video" folder and the meta files in each of the "Knot_Tying", "Needle_Passing" and "Suturing" tasks. Once downloaded, create a directory named "data", extract and copy the folder structure into that directory. The resulting directory structure should be:
```
gesture/data/
    -> Knot_Tying/
        -> meta_file_Knot_Tying.txt
        -> readme.text
        -> video/
            -> Knot_Tying_B001_capture1.avi
            -> Knot_Tying_B001_capture2.avi
            -> Knot_Tying_B002_capture1.avi
            ...
    -> Needle_Passing/
        -> meta_file_Needle_Passing.txt
        -> readme.text
        -> video/
            -> Needle_Passing_B001_capture1.avi
            -> Needle_Passing_B001_capture2.avi
            -> Needle_Passing_B002_capture1.avi
            ...
    -> Suturing/
        -> meta_file_Suturing.txt
        -> readme.text
        -> video/
            -> Suturing_B001_capture1.avi
            -> Suturing_B001_capture2.avi
            -> Suturing_B002_capture1.avi
            ...
```

Alternatively, run the provided data script to download and extract the data to the required format:
```bash
python data.py
```
