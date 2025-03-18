
README

Aquest arxiu conté informació sobre:
  -Configuració de les variables globals
  -La estructura d'arxius
  -El funcionament del programa

---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------
CONFIGURACIO:

configuration/configfile.txt es l'arxiu que ens permet configurar totes les variables
globals del programa per no tenir variables "hard coded" i poder configurar tot en un
sol arxiu per poder mantenir coherencia

LLISTA ACTUAL DE VARIABLES:

num_layers=3 // Number of layers and memory allocation
layer=1024 // Number of neurons layer0 (in layer or img_size (dim_x * dim_y))
layer=135 // Number neurons layer1 (hidden layer)
layer=10 // Number neurons layer2 (out layer)
alpha=0.15 //Learning rate (indicates what portion of gradient should be used)
num_training_patterns=1934 // Number of training examples
num_test_patterns=946 // Number of patterns to recognize
dataset_training_path=./datasets/optdigits.tra // Training set path -> all training is in datasets/
dataset_test_path=./datasets/optdigits.csv // Dataset path for test after training
img_dim_x=32 // images size x
img_dim_y=32 //images size y
seed=50 //randomizer seed
num_epochs=10 //número d'iteracions d'entrenament
debug=0 //debugging desactivat

---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Datasets description:
    
optdigits:
Set composet of images of handwritten digits (0-9).
Can be obtained at https://www.kaggle.com/datasets/zwangm/optdigits/code and
it's already in the textual format indicated before ('0' background, '1' number)
    1934 32x32 train images
     946 32x32 test images
    
Dataset values for configfile.txt:
    
num_layers=3
layer=1024
layer=135
layer=10
num_training_patterns=1934
num_test_patterns=946
img_dim_x=32
img_dim_y=32
dataset_training_path=datasets/optdigits/optdigits.tra
dataset_test_path=datasets/optdigits/optdigits.cv
    
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

DEBUGGING:

Com que estem treballant amb C si volem fer debugging per veure on falla el nostre codi pot ser molt complicat, sobretot
perque les eines que existeixen com valgrind i altres no son molt trivials d'utilitzar

Per aixó hem fet un mètode senzill de debugging basat en unna variable en l'arxiu configfile.txt

debug != 1 -> No esta activat el mode debug
debug == 1 -> Esta activat el mode debug

Quan el mode debug esta activat simplement es fan printf() abans de certes crides a funcions importants o
funcions delicades on solen tenir lloc error de Segmentation fault.

Es poden afegir més punts de debug obviament. Simplement cal afegir un if() amb l'opció de debug

if(debug == 1)
  printf("Print Something");
  
Es recomana seguir aquesta sintaxis per mantenir el codi consistent i poder distingir per a que es cada printf()
També permet desactivar-los en qualsevol moment canviant el valor de debug!!!

---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------


HELLOOOOOOO
