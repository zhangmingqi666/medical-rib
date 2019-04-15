
from voc_eval_py3 import voc_eval

#/Users/jiangyy/projects/medical-rib/data/voc2007/ImageSets/test.txt
rec,prec,ap = voc_eval('/Users/jiangyy/projects/medical-rib/models/darknet/results/{}.txt', '/Users/jiangyy/voc2007.xoy/Annotations/{}.xml',
                       '/Users/jiangyy/voc2007.xoy/ImageSets/trainval.txt', 'hurt', '.')

print('rec',rec)
print('prec',prec)
print('ap',ap)