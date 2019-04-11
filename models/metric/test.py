
from voc_eval_py3 import voc_eval

#/Users/jiangyy/projects/medical-rib/data/voc2007/ImageSets/test.txt
rec,prec,ap = voc_eval('./darknet/results/{}.txt', '../data/voc2007/Annotations/{}.xml',
                       '../data/voc2007/ImageSets/test.txt', 'comp4_det_test_hurt', '.')

print('rec',rec)
print('prec',prec)
print('ap',ap)