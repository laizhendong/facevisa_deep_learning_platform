#python test.py --data ./data/stain.yaml --batch 12 --img 1024 --conf 0.1 --iou 0.5 --task val --device 1 --weights /model/best.pt  --save-txt --verbose
python test.py --data test_data.yaml --task test --weights /model/last.pt --verbose
