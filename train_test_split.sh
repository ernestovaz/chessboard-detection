i=0
for file in data/*; do 
  python3 preprocess.py $file
  ((i+=1))
  if (($i == 108))
  then
    mkdir ground_truth/test
    mv ground_truth/pawn ground_truth/test
    mv ground_truth/rook ground_truth/test
    mv ground_truth/king ground_truth/test
    mv ground_truth/queen ground_truth/test
    mv ground_truth/bishop ground_truth/test
    mv ground_truth/knight ground_truth/test
  fi
done

mkdir ground_truth/train
mv ground_truth/pawn ground_truth/train
mv ground_truth/rook ground_truth/train
mv ground_truth/king ground_truth/train
mv ground_truth/queen ground_truth/train
mv ground_truth/bishop ground_truth/train
mv ground_truth/knight ground_truth/train
