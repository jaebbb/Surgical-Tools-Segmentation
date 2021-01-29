python3 get_class_names.py input ./
python3 labelme2voc.py input output --labels 'class_names.txt'
mv class_names.txt output/class_names.txt