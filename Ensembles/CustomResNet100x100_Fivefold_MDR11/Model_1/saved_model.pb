нє7
™э
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8яѕ/
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
В
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
В
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
В
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	А*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ьА
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ґА
valueЂАBІА BЯА
 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer-27
layer_with_weights-17
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"
signatures
 
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
Ч
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
Ч
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
R
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
Ч
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
atrainable_variables
bregularization_losses
c	keras_api
h

dkernel
ebias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
Ч
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
R
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
i

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
А	keras_api
n
Бkernel
	Вbias
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
†
	Зaxis

Иgamma
	Йbeta
Кmoving_mean
Лmoving_variance
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
n
Рkernel
	Сbias
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
†
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
V
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
V
£	variables
§trainable_variables
•regularization_losses
¶	keras_api
V
І	variables
®trainable_variables
©regularization_losses
™	keras_api
V
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
n
ѓkernel
	∞bias
±	variables
≤trainable_variables
≥regularization_losses
і	keras_api
V
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
n
єkernel
	Їbias
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
V
њ	variables
јtrainable_variables
Ѕregularization_losses
¬	keras_api
n
√kernel
	ƒbias
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
И
#0
$1
)2
*3
04
15
26
37
88
99
?10
@11
A12
B13
O14
P15
U16
V17
\18
]19
^20
_21
d22
e23
k24
l25
m26
n27
{28
|29
Б30
В31
И32
Й33
К34
Л35
Р36
С37
Ч38
Ш39
Щ40
Ъ41
ѓ42
∞43
є44
Ї45
√46
ƒ47
§
#0
$1
)2
*3
04
15
86
97
?8
@9
O10
P11
U12
V13
\14
]15
d16
e17
k18
l19
{20
|21
Б22
В23
И24
Й25
Р26
С27
Ч28
Ш29
ѓ30
∞31
є32
Ї33
√34
ƒ35
 
≤
…layers
	variables
trainable_variables
 metrics
 regularization_losses
Ћlayer_metrics
 ћlayer_regularization_losses
Ќnon_trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
≤
ќlayers
%	variables
ѕmetrics
&trainable_variables
'regularization_losses
–layer_metrics
 —layer_regularization_losses
“non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
≤
”layers
+	variables
‘metrics
,trainable_variables
-regularization_losses
’layer_metrics
 ÷layer_regularization_losses
„non_trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33

00
11
 
≤
Ўlayers
4	variables
ўmetrics
5trainable_variables
6regularization_losses
Џlayer_metrics
 џlayer_regularization_losses
№non_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
≤
Ёlayers
:	variables
ёmetrics
;trainable_variables
<regularization_losses
яlayer_metrics
 аlayer_regularization_losses
бnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2
B3

?0
@1
 
≤
вlayers
C	variables
гmetrics
Dtrainable_variables
Eregularization_losses
дlayer_metrics
 еlayer_regularization_losses
жnon_trainable_variables
 
 
 
≤
зlayers
G	variables
иmetrics
Htrainable_variables
Iregularization_losses
йlayer_metrics
 кlayer_regularization_losses
лnon_trainable_variables
 
 
 
≤
мlayers
K	variables
нmetrics
Ltrainable_variables
Mregularization_losses
оlayer_metrics
 пlayer_regularization_losses
рnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
≤
сlayers
Q	variables
тmetrics
Rtrainable_variables
Sregularization_losses
уlayer_metrics
 фlayer_regularization_losses
хnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
≤
цlayers
W	variables
чmetrics
Xtrainable_variables
Yregularization_losses
шlayer_metrics
 щlayer_regularization_losses
ъnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
^2
_3

\0
]1
 
≤
ыlayers
`	variables
ьmetrics
atrainable_variables
bregularization_losses
эlayer_metrics
 юlayer_regularization_losses
€non_trainable_variables
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

d0
e1
 
≤
Аlayers
f	variables
Бmetrics
gtrainable_variables
hregularization_losses
Вlayer_metrics
 Гlayer_regularization_losses
Дnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
m2
n3

k0
l1
 
≤
Еlayers
o	variables
Жmetrics
ptrainable_variables
qregularization_losses
Зlayer_metrics
 Иlayer_regularization_losses
Йnon_trainable_variables
 
 
 
≤
Кlayers
s	variables
Лmetrics
ttrainable_variables
uregularization_losses
Мlayer_metrics
 Нlayer_regularization_losses
Оnon_trainable_variables
 
 
 
≤
Пlayers
w	variables
Рmetrics
xtrainable_variables
yregularization_losses
Сlayer_metrics
 Тlayer_regularization_losses
Уnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
≤
Фlayers
}	variables
Хmetrics
~trainable_variables
regularization_losses
Цlayer_metrics
 Чlayer_regularization_losses
Шnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Б0
В1

Б0
В1
 
µ
Щlayers
Г	variables
Ъmetrics
Дtrainable_variables
Еregularization_losses
Ыlayer_metrics
 Ьlayer_regularization_losses
Эnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_4/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_4/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_4/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_4/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
И0
Й1
К2
Л3

И0
Й1
 
µ
Юlayers
М	variables
Яmetrics
Нtrainable_variables
Оregularization_losses
†layer_metrics
 °layer_regularization_losses
Ґnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_8/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

Р0
С1

Р0
С1
 
µ
£layers
Т	variables
§metrics
Уtrainable_variables
Фregularization_losses
•layer_metrics
 ¶layer_regularization_losses
Іnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ч0
Ш1
Щ2
Ъ3

Ч0
Ш1
 
µ
®layers
Ы	variables
©metrics
Ьtrainable_variables
Эregularization_losses
™layer_metrics
 Ђlayer_regularization_losses
ђnon_trainable_variables
 
 
 
µ
≠layers
Я	variables
Ѓmetrics
†trainable_variables
°regularization_losses
ѓlayer_metrics
 ∞layer_regularization_losses
±non_trainable_variables
 
 
 
µ
≤layers
£	variables
≥metrics
§trainable_variables
•regularization_losses
іlayer_metrics
 µlayer_regularization_losses
ґnon_trainable_variables
 
 
 
µ
Јlayers
І	variables
Єmetrics
®trainable_variables
©regularization_losses
єlayer_metrics
 Їlayer_regularization_losses
їnon_trainable_variables
 
 
 
µ
Љlayers
Ђ	variables
љmetrics
ђtrainable_variables
≠regularization_losses
Њlayer_metrics
 њlayer_regularization_losses
јnon_trainable_variables
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

ѓ0
∞1

ѓ0
∞1
 
µ
Ѕlayers
±	variables
¬metrics
≤trainable_variables
≥regularization_losses
√layer_metrics
 ƒlayer_regularization_losses
≈non_trainable_variables
 
 
 
µ
∆layers
µ	variables
«metrics
ґtrainable_variables
Јregularization_losses
»layer_metrics
 …layer_regularization_losses
 non_trainable_variables
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

є0
Ї1

є0
Ї1
 
µ
Ћlayers
ї	variables
ћmetrics
Љtrainable_variables
љregularization_losses
Ќlayer_metrics
 ќlayer_regularization_losses
ѕnon_trainable_variables
 
 
 
µ
–layers
њ	variables
—metrics
јtrainable_variables
Ѕregularization_losses
“layer_metrics
 ”layer_regularization_losses
‘non_trainable_variables
[Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

√0
ƒ1

√0
ƒ1
 
µ
’layers
≈	variables
÷metrics
∆trainable_variables
«regularization_losses
„layer_metrics
 Ўlayer_regularization_losses
ўnon_trainable_variables
ё
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 
 
 
Z
20
31
A2
B3
^4
_5
m6
n7
К8
Л9
Щ10
Ъ11
 
 
 
 
 
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

^0
_1
 
 
 
 
 
 
 
 
 

m0
n1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

К0
Л1
 
 
 
 
 
 
 
 
 

Щ0
Ъ1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€22*
dtype0*$
shape:€€€€€€€€€22
п
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_139188
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
и
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*=
Tin6
422*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_142112
£
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*<
Tin5
321*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_142268—Њ-
Ѓ
©
6__inference_batch_normalization_2_layer_call_fn_140726

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1368792
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
÷
d
H__inference_activation_1_layer_call_and_return_conditional_losses_137042

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
я
p
__inference_loss_fn_10_141772>
:conv2d_5_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_5_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/addf
IdentityIdentity#conv2d_5/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Д
n
__inference_loss_fn_15_141837<
8conv2d_7_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_7_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/addd
IdentityIdentity!conv2d_7/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
»
К
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_136986

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_136600

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
б
~
)__inference_conv2d_1_layer_call_fn_135592

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1355822
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
х
“
&__inference_model_layer_call_fn_140226

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityИҐStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1387222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
„
m
A__inference_add_1_layer_call_and_return_conditional_losses_140998
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:€€€€€€€€€22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22 :€€€€€€€€€22 :Y U
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/1
С
К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_136437

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
м
”
&__inference_model_layer_call_fn_138400
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./0*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1383012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
Ё
|
'__inference_conv2d_layer_call_fn_135554

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1355442
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_136406

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∞
©
6__inference_batch_normalization_1_layer_call_fn_140523

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1367752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ѓ
©
6__inference_batch_normalization_4_layer_call_fn_141120

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1370902
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
я
p
__inference_loss_fn_16_141850>
:conv2d_8_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_8_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/addf
IdentityIdentity#conv2d_8/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ц
©
6__inference_batch_normalization_1_layer_call_fn_140585

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1358392
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137090

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д 
ђ
D__inference_conv2d_7_layer_call_and_return_conditional_losses_136312

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
ReluЌ
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/addЊ
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
©
6__inference_batch_normalization_3_layer_call_fn_140979

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1362042
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
т
І
4__inference_batch_normalization_layer_call_fn_140332

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1356762
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140554

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140948

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141342

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
х
{
&__inference_dense_layer_call_fn_141471

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1373032
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
‘
b
F__inference_activation_layer_call_and_return_conditional_losses_140615

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
б
~
)__inference_conv2d_7_layer_call_fn_136322

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1363122
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140873

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч
Ђ
C__inference_dense_2_layer_call_and_return_conditional_losses_141620

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidƒ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЉ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140572

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш
”
&__inference_model_layer_call_fn_138821
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityИҐStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1387222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
С
К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141182

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѕ
k
A__inference_add_1_layer_call_and_return_conditional_losses_137028

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:€€€€€€€€€22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22 :€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
Ґ
R
&__inference_add_2_layer_call_fn_141398
inputs_0
inputs_1
identityµ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1372392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22@:€€€€€€€€€22@:Y U
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/1
е$
÷
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140301

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_136204

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ч
G
+__inference_activation_layer_call_fn_140620

inputs
identity≠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1368312
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
з$
Ў
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140770

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
©
6__inference_batch_normalization_4_layer_call_fn_141195

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1364062
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
й
“
&__inference_model_layer_call_fn_140125

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./0*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1383012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
™
І
4__inference_batch_normalization_layer_call_fn_140407

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1366682
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≥Н
у
A__inference_model_layer_call_and_return_conditional_losses_138301

inputs
conv2d_137984
conv2d_137986
conv2d_1_137989
conv2d_1_137991
batch_normalization_137994
batch_normalization_137996
batch_normalization_137998
batch_normalization_138000
conv2d_2_138003
conv2d_2_138005 
batch_normalization_1_138008 
batch_normalization_1_138010 
batch_normalization_1_138012 
batch_normalization_1_138014
conv2d_3_138019
conv2d_3_138021
conv2d_4_138024
conv2d_4_138026 
batch_normalization_2_138029 
batch_normalization_2_138031 
batch_normalization_2_138033 
batch_normalization_2_138035
conv2d_5_138038
conv2d_5_138040 
batch_normalization_3_138043 
batch_normalization_3_138045 
batch_normalization_3_138047 
batch_normalization_3_138049
conv2d_6_138054
conv2d_6_138056
conv2d_7_138059
conv2d_7_138061 
batch_normalization_4_138064 
batch_normalization_4_138066 
batch_normalization_4_138068 
batch_normalization_4_138070
conv2d_8_138073
conv2d_8_138075 
batch_normalization_5_138078 
batch_normalization_5_138080 
batch_normalization_5_138082 
batch_normalization_5_138084
dense_138091
dense_138093
dense_1_138097
dense_1_138099
dense_2_138103
dense_2_138105
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallу
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_137984conv2d_137986*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1355442 
conv2d/StatefulPartitionedCallЮ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_137989conv2d_1_137991*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1355822"
 conv2d_1/StatefulPartitionedCallС
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_137994batch_normalization_137996batch_normalization_137998batch_normalization_138000*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1366682-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_138003conv2d_2_138005*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1357452"
 conv2d_2/StatefulPartitionedCallЯ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_1_138008batch_normalization_1_138010batch_normalization_1_138012batch_normalization_1_138014*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1367572/
-batch_normalization_1/StatefulPartitionedCallИ
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1368172
add/PartitionedCallў
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1368312
activation/PartitionedCallЪ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_3_138019conv2d_3_138021*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1359092"
 conv2d_3/StatefulPartitionedCall†
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_138024conv2d_4_138026*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1359472"
 conv2d_4/StatefulPartitionedCallЯ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_2_138029batch_normalization_2_138031batch_normalization_2_138033batch_normalization_2_138035*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1368792/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_5_138038conv2d_5_138040*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1361102"
 conv2d_5/StatefulPartitionedCallЯ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_3_138043batch_normalization_3_138045batch_normalization_3_138047batch_normalization_3_138049*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1369682/
-batch_normalization_3/StatefulPartitionedCallР
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1370282
add_1/PartitionedCallб
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1370422
activation_1/PartitionedCallЬ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_6_138054conv2d_6_138056*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1362742"
 conv2d_6/StatefulPartitionedCall†
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_138059conv2d_7_138061*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1363122"
 conv2d_7/StatefulPartitionedCallЯ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_4_138064batch_normalization_4_138066batch_normalization_4_138068batch_normalization_4_138070*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1370902/
-batch_normalization_4/StatefulPartitionedCall≠
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_8_138073conv2d_8_138075*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_1364752"
 conv2d_8/StatefulPartitionedCallЯ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_5_138078batch_normalization_5_138080batch_normalization_5_138082batch_normalization_5_138084*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1371792/
-batch_normalization_5/StatefulPartitionedCallР
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1372392
add_2/PartitionedCallб
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1372532
activation_2/PartitionedCallД
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1366182*
(global_average_pooling2d/PartitionedCallЁ
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1372682
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_138091dense_138093*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1373032
dense/StatefulPartitionedCallл
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1373312!
dropout/StatefulPartitionedCallУ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_138097dense_1_138099*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1373762!
dense_1/StatefulPartitionedCallХ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1374042#
!dropout_1/StatefulPartitionedCallФ
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_138103dense_2_138105*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1374492!
dense_2/StatefulPartitionedCallЄ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_137984*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add®
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_137986*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/addЊ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_137989*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/addЃ
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_137991*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/addЊ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_138003*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/addЃ
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_138005*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/addЊ
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_138019*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/addЃ
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_138021*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/addЊ
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_138024*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/addЃ
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_138026*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/addЊ
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_138038*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/addЃ
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_138040*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/addЊ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_138054*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/addЃ
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_138056*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/addЊ
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_138059*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/addЃ
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_138061*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/addЊ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_138073*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/addЃ
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_138075*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/addЃ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_138091*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add¶
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_138093*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_138097* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addђ
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_138099*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addі
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_138103*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЂ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_138105*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addэ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
Ы
I
-__inference_activation_2_layer_call_fn_141408

inputs
identityѓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1372532
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
Н
ђ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_135745

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddЌ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/addЊ
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_136235

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137197

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
’
k
?__inference_add_layer_call_and_return_conditional_losses_140604
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:€€€€€€€€€222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22:€€€€€€€€€22:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/1
б
~
)__inference_conv2d_8_layer_call_fn_136485

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_1364752
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
n
__inference_loss_fn_11_141785<
8conv2d_5_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_5_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/addd
IdentityIdentity!conv2d_5/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
л
U
9__inference_global_average_pooling2d_layer_call_fn_136624

inputs
identityЉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1366182
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
F
*__inference_dropout_1_layer_call_fn_141577

inputs
identity•
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1374092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ф
І
4__inference_batch_normalization_layer_call_fn_140345

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1357072
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140497

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
m
__inference_loss_fn_3_141681<
8conv2d_1_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_1_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/addd
IdentityIdentity!conv2d_1/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
»
К
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140713

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
е$
÷
O__inference_batch_normalization_layer_call_and_return_conditional_losses_135676

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
‘
b
F__inference_activation_layer_call_and_return_conditional_losses_136831

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
б
~
)__inference_conv2d_5_layer_call_fn_136120

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1361102
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
n
__inference_loss_fn_17_141863<
8conv2d_8_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_8_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/addd
IdentityIdentity!conv2d_8/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ќ
i
?__inference_add_layer_call_and_return_conditional_losses_136817

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:€€€€€€€€€222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
ш
©
6__inference_batch_normalization_4_layer_call_fn_141208

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1364372
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ћ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_137409

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Э$
÷
O__inference_batch_normalization_layer_call_and_return_conditional_losses_136668

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
б
~
)__inference_conv2d_4_layer_call_fn_135957

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1359472
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_136897

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ѓ
©
6__inference_batch_normalization_3_layer_call_fn_140904

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1369682
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
©
_
C__inference_flatten_layer_call_and_return_conditional_losses_141414

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
∆
И
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140394

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Н
ђ
D__inference_conv2d_5_layer_call_and_return_conditional_losses_136110

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddЌ
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/addЊ
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ё
o
__inference_loss_fn_4_141694>
:conv2d_2_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/addf
IdentityIdentity#conv2d_2/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ш
©
6__inference_batch_normalization_3_layer_call_fn_140992

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1362352
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д 
ђ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_135582

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
ReluЌ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/addЊ
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141285

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 
a
C__inference_dropout_layer_call_and_return_conditional_losses_141488

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
»К
Ѓ
A__inference_model_layer_call_and_return_conditional_losses_137978
input_1
conv2d_137661
conv2d_137663
conv2d_1_137666
conv2d_1_137668
batch_normalization_137671
batch_normalization_137673
batch_normalization_137675
batch_normalization_137677
conv2d_2_137680
conv2d_2_137682 
batch_normalization_1_137685 
batch_normalization_1_137687 
batch_normalization_1_137689 
batch_normalization_1_137691
conv2d_3_137696
conv2d_3_137698
conv2d_4_137701
conv2d_4_137703 
batch_normalization_2_137706 
batch_normalization_2_137708 
batch_normalization_2_137710 
batch_normalization_2_137712
conv2d_5_137715
conv2d_5_137717 
batch_normalization_3_137720 
batch_normalization_3_137722 
batch_normalization_3_137724 
batch_normalization_3_137726
conv2d_6_137731
conv2d_6_137733
conv2d_7_137736
conv2d_7_137738 
batch_normalization_4_137741 
batch_normalization_4_137743 
batch_normalization_4_137745 
batch_normalization_4_137747
conv2d_8_137750
conv2d_8_137752 
batch_normalization_5_137755 
batch_normalization_5_137757 
batch_normalization_5_137759 
batch_normalization_5_137761
dense_137768
dense_137770
dense_1_137774
dense_1_137776
dense_2_137780
dense_2_137782
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallф
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_137661conv2d_137663*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1355442 
conv2d/StatefulPartitionedCallЮ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_137666conv2d_1_137668*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1355822"
 conv2d_1/StatefulPartitionedCallУ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_137671batch_normalization_137673batch_normalization_137675batch_normalization_137677*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1366862-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_137680conv2d_2_137682*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1357452"
 conv2d_2/StatefulPartitionedCall°
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_1_137685batch_normalization_1_137687batch_normalization_1_137689batch_normalization_1_137691*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1367752/
-batch_normalization_1/StatefulPartitionedCallИ
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1368172
add/PartitionedCallў
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1368312
activation/PartitionedCallЪ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_3_137696conv2d_3_137698*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1359092"
 conv2d_3/StatefulPartitionedCall†
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_137701conv2d_4_137703*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1359472"
 conv2d_4/StatefulPartitionedCall°
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_2_137706batch_normalization_2_137708batch_normalization_2_137710batch_normalization_2_137712*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1368972/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_5_137715conv2d_5_137717*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1361102"
 conv2d_5/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_3_137720batch_normalization_3_137722batch_normalization_3_137724batch_normalization_3_137726*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1369862/
-batch_normalization_3/StatefulPartitionedCallР
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1370282
add_1/PartitionedCallб
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1370422
activation_1/PartitionedCallЬ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_6_137731conv2d_6_137733*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1362742"
 conv2d_6/StatefulPartitionedCall†
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_137736conv2d_7_137738*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1363122"
 conv2d_7/StatefulPartitionedCall°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_4_137741batch_normalization_4_137743batch_normalization_4_137745batch_normalization_4_137747*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1371082/
-batch_normalization_4/StatefulPartitionedCall≠
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_8_137750conv2d_8_137752*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_1364752"
 conv2d_8/StatefulPartitionedCall°
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_5_137755batch_normalization_5_137757batch_normalization_5_137759batch_normalization_5_137761*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1371972/
-batch_normalization_5/StatefulPartitionedCallР
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1372392
add_2/PartitionedCallб
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1372532
activation_2/PartitionedCallД
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1366182*
(global_average_pooling2d/PartitionedCallЁ
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1372682
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_137768dense_137770*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1373032
dense/StatefulPartitionedCall”
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1373362
dropout/PartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_137774dense_1_137776*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1373762!
dense_1/StatefulPartitionedCallџ
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1374092
dropout_1/PartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_137780dense_2_137782*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1374492!
dense_2/StatefulPartitionedCallЄ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_137661*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add®
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_137663*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/addЊ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_137666*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/addЃ
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_137668*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/addЊ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_137680*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/addЃ
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_137682*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/addЊ
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_137696*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/addЃ
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_137698*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/addЊ
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_137701*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/addЃ
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_137703*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/addЊ
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_137715*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/addЃ
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_137717*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/addЊ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_137731*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/addЃ
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_137733*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/addЊ
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_137736*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/addЃ
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_137738*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/addЊ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_137750*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/addЃ
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_137752*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/addЃ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_137768*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add¶
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_137770*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_137774* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addђ
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_137776*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addі
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_137780*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЂ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_137782*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addЈ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
С
К
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_135870

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137108

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Е
c
*__inference_dropout_1_layer_call_fn_141572

inputs
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1374042
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
м
m
__inference_loss_fn_21_141915;
7dense_1_bias_regularizer_square_readvariableop_resource
identityИ’
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_1_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addc
IdentityIdentity dense_1/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ц
©
6__inference_batch_normalization_5_layer_call_fn_141373

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1365692
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
±
o
__inference_loss_fn_20_141902=
9dense_1_kernel_regularizer_square_readvariableop_resource
identityИа
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/adde
IdentityIdentity"dense_1/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ы
m
__inference_loss_fn_18_141876;
7dense_kernel_regularizer_square_readvariableop_resource
identityИў
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addc
IdentityIdentity dense/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
К
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_141562

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
∆
И
O__inference_batch_normalization_layer_call_and_return_conditional_losses_136686

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141089

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∞
©
6__inference_batch_normalization_5_layer_call_fn_141311

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1371972
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_136968

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
÷
d
H__inference_activation_1_layer_call_and_return_conditional_losses_141009

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
б
~
)__inference_conv2d_2_layer_call_fn_135755

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1357452
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
∞
©
6__inference_batch_normalization_4_layer_call_fn_141133

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1371082
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
мћ
ї
!__inference__wrapped_model_135517
input_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource5
1model_batch_normalization_readvariableop_resource7
3model_batch_normalization_readvariableop_1_resourceF
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resourceH
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource7
3model_batch_normalization_1_readvariableop_resource9
5model_batch_normalization_1_readvariableop_1_resourceH
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_3_conv2d_readvariableop_resource2
.model_conv2d_3_biasadd_readvariableop_resource1
-model_conv2d_4_conv2d_readvariableop_resource2
.model_conv2d_4_biasadd_readvariableop_resource7
3model_batch_normalization_2_readvariableop_resource9
5model_batch_normalization_2_readvariableop_1_resourceH
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_5_conv2d_readvariableop_resource2
.model_conv2d_5_biasadd_readvariableop_resource7
3model_batch_normalization_3_readvariableop_resource9
5model_batch_normalization_3_readvariableop_1_resourceH
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_6_conv2d_readvariableop_resource2
.model_conv2d_6_biasadd_readvariableop_resource1
-model_conv2d_7_conv2d_readvariableop_resource2
.model_conv2d_7_biasadd_readvariableop_resource7
3model_batch_normalization_4_readvariableop_resource9
5model_batch_normalization_4_readvariableop_1_resourceH
Dmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_8_conv2d_readvariableop_resource2
.model_conv2d_8_biasadd_readvariableop_resource7
3model_batch_normalization_5_readvariableop_resource9
5model_batch_normalization_5_readvariableop_1_resourceH
Dmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource
identityИЉ
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpЋ
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
model/conv2d/Conv2D≥
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOpЉ
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
model/conv2d/BiasAdd¬
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOpз
model/conv2d_1/Conv2DConv2Dmodel/conv2d/BiasAdd:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
model/conv2d_1/Conv2Dє
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOpƒ
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
model/conv2d_1/BiasAddН
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
model/conv2d_1/Relu¬
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02*
(model/batch_normalization/ReadVariableOp»
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*model/batch_normalization/ReadVariableOp_1х
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpы
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Б
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3!model/conv2d_1/Relu:activations:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2,
*model/batch_normalization/FusedBatchNormV3¬
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOpш
model/conv2d_2/Conv2DConv2D.model/batch_normalization/FusedBatchNormV3:y:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
model/conv2d_2/Conv2Dє
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOpƒ
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
model/conv2d_2/BiasAdd»
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/batch_normalization_1/ReadVariableOpќ
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,model/batch_normalization_1/ReadVariableOp_1ы
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Л
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d_2/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_1/FusedBatchNormV3≤
model/add/addAddV20model/batch_normalization_1/FusedBatchNormV3:y:0model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
model/add/addГ
model/activation/ReluRelumodel/add/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€222
model/activation/Relu¬
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOpн
model/conv2d_3/Conv2DConv2D#model/activation/Relu:activations:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
model/conv2d_3/Conv2Dє
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOpƒ
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/conv2d_3/BiasAddН
model/conv2d_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/conv2d_3/Relu¬
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOpл
model/conv2d_4/Conv2DConv2D!model/conv2d_3/Relu:activations:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
model/conv2d_4/Conv2Dє
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOpƒ
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/conv2d_4/BiasAddН
model/conv2d_4/ReluRelumodel/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/conv2d_4/Relu»
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization_2/ReadVariableOpќ
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,model/batch_normalization_2/ReadVariableOp_1ы
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Н
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3!model/conv2d_4/Relu:activations:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_2/FusedBatchNormV3¬
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02&
$model/conv2d_5/Conv2D/ReadVariableOpъ
model/conv2d_5/Conv2DConv2D0model/batch_normalization_2/FusedBatchNormV3:y:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
model/conv2d_5/Conv2Dє
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_5/BiasAdd/ReadVariableOpƒ
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/conv2d_5/BiasAdd»
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization_3/ReadVariableOpќ
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,model/batch_normalization_3/ReadVariableOp_1ы
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Л
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model/conv2d_5/BiasAdd:output:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_3/FusedBatchNormV3Ї
model/add_1/addAddV20model/batch_normalization_3/FusedBatchNormV3:y:0!model/conv2d_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/add_1/addЙ
model/activation_1/ReluRelumodel/add_1/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model/activation_1/Relu¬
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$model/conv2d_6/Conv2D/ReadVariableOpп
model/conv2d_6/Conv2DConv2D%model/activation_1/Relu:activations:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
model/conv2d_6/Conv2Dє
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_6/BiasAdd/ReadVariableOpƒ
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/conv2d_6/BiasAddН
model/conv2d_6/ReluRelumodel/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/conv2d_6/Relu¬
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_7/Conv2D/ReadVariableOpл
model/conv2d_7/Conv2DConv2D!model/conv2d_6/Relu:activations:0,model/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
model/conv2d_7/Conv2Dє
%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_7/BiasAdd/ReadVariableOpƒ
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/conv2d_7/BiasAddН
model/conv2d_7/ReluRelumodel/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/conv2d_7/Relu»
*model/batch_normalization_4/ReadVariableOpReadVariableOp3model_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_4/ReadVariableOpќ
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOp5model_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_4/ReadVariableOp_1ы
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Н
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3!model/conv2d_7/Relu:activations:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_4/FusedBatchNormV3¬
$model/conv2d_8/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_8/Conv2D/ReadVariableOpъ
model/conv2d_8/Conv2DConv2D0model/batch_normalization_4/FusedBatchNormV3:y:0,model/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
model/conv2d_8/Conv2Dє
%model/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_8/BiasAdd/ReadVariableOpƒ
model/conv2d_8/BiasAddBiasAddmodel/conv2d_8/Conv2D:output:0-model/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/conv2d_8/BiasAdd»
*model/batch_normalization_5/ReadVariableOpReadVariableOp3model_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_5/ReadVariableOpќ
,model/batch_normalization_5/ReadVariableOp_1ReadVariableOp5model_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_5/ReadVariableOp_1ы
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Л
,model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3model/conv2d_8/BiasAdd:output:02model/batch_normalization_5/ReadVariableOp:value:04model/batch_normalization_5/ReadVariableOp_1:value:0Cmodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_5/FusedBatchNormV3Ї
model/add_2/addAddV20model/batch_normalization_5/FusedBatchNormV3:y:0!model/conv2d_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/add_2/addЙ
model/activation_2/ReluRelumodel/add_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model/activation_2/Reluњ
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indicesл
#model/global_average_pooling2d/MeanMean%model/activation_2/Relu:activations:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2%
#model/global_average_pooling2d/Mean{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
model/flatten/ConstЈ
model/flatten/ReshapeReshape,model/global_average_pooling2d/Mean:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model/flatten/Reshape≤
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!model/dense/MatMul/ReadVariableOp∞
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense/MatMul±
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"model/dense/BiasAdd/ReadVariableOp≤
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense/BiasAdd}
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense/ReluП
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dropout/Identityє
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#model/dense_1/MatMul/ReadVariableOpЈ
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense_1/MatMulЈ
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOpЇ
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense_1/BiasAddГ
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense_1/ReluХ
model/dropout_1/IdentityIdentity model/dense_1/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dropout_1/IdentityЄ
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#model/dense_2/MatMul/ReadVariableOpЄ
model/dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/dense_2/MatMulґ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOpє
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/dense_2/BiasAddЛ
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/dense_2/Sigmoidm
IdentityIdentitymodel/dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22:::::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
С
К
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141360

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ѓ
©
6__inference_batch_normalization_1_layer_call_fn_140510

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1367572
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
©
6__inference_batch_normalization_2_layer_call_fn_140801

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1360412
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д 
ђ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_135947

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
ReluЌ
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/addЊ
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ь
©
A__inference_dense_layer_call_and_return_conditional_losses_137303

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Reluј
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addє
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140695

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137179

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_136879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
К
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_136775

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∞
©
6__inference_batch_normalization_3_layer_call_fn_140917

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1369862
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 
a
C__inference_dropout_layer_call_and_return_conditional_losses_137336

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ш
©
6__inference_batch_normalization_2_layer_call_fn_140814

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1360722
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
a
(__inference_dropout_layer_call_fn_141493

inputs
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1373312
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ї
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_136618

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
÷
d
H__inference_activation_2_layer_call_and_return_conditional_losses_141403

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
≈К
≠
A__inference_model_layer_call_and_return_conditional_losses_138722

inputs
conv2d_138405
conv2d_138407
conv2d_1_138410
conv2d_1_138412
batch_normalization_138415
batch_normalization_138417
batch_normalization_138419
batch_normalization_138421
conv2d_2_138424
conv2d_2_138426 
batch_normalization_1_138429 
batch_normalization_1_138431 
batch_normalization_1_138433 
batch_normalization_1_138435
conv2d_3_138440
conv2d_3_138442
conv2d_4_138445
conv2d_4_138447 
batch_normalization_2_138450 
batch_normalization_2_138452 
batch_normalization_2_138454 
batch_normalization_2_138456
conv2d_5_138459
conv2d_5_138461 
batch_normalization_3_138464 
batch_normalization_3_138466 
batch_normalization_3_138468 
batch_normalization_3_138470
conv2d_6_138475
conv2d_6_138477
conv2d_7_138480
conv2d_7_138482 
batch_normalization_4_138485 
batch_normalization_4_138487 
batch_normalization_4_138489 
batch_normalization_4_138491
conv2d_8_138494
conv2d_8_138496 
batch_normalization_5_138499 
batch_normalization_5_138501 
batch_normalization_5_138503 
batch_normalization_5_138505
dense_138512
dense_138514
dense_1_138518
dense_1_138520
dense_2_138524
dense_2_138526
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallу
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_138405conv2d_138407*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1355442 
conv2d/StatefulPartitionedCallЮ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_138410conv2d_1_138412*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1355822"
 conv2d_1/StatefulPartitionedCallУ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_138415batch_normalization_138417batch_normalization_138419batch_normalization_138421*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1366862-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_138424conv2d_2_138426*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1357452"
 conv2d_2/StatefulPartitionedCall°
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_1_138429batch_normalization_1_138431batch_normalization_1_138433batch_normalization_1_138435*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1367752/
-batch_normalization_1/StatefulPartitionedCallИ
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1368172
add/PartitionedCallў
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1368312
activation/PartitionedCallЪ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_3_138440conv2d_3_138442*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1359092"
 conv2d_3/StatefulPartitionedCall†
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_138445conv2d_4_138447*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1359472"
 conv2d_4/StatefulPartitionedCall°
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_2_138450batch_normalization_2_138452batch_normalization_2_138454batch_normalization_2_138456*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1368972/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_5_138459conv2d_5_138461*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1361102"
 conv2d_5/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_3_138464batch_normalization_3_138466batch_normalization_3_138468batch_normalization_3_138470*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1369862/
-batch_normalization_3/StatefulPartitionedCallР
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1370282
add_1/PartitionedCallб
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1370422
activation_1/PartitionedCallЬ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_6_138475conv2d_6_138477*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1362742"
 conv2d_6/StatefulPartitionedCall†
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_138480conv2d_7_138482*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1363122"
 conv2d_7/StatefulPartitionedCall°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_4_138485batch_normalization_4_138487batch_normalization_4_138489batch_normalization_4_138491*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1371082/
-batch_normalization_4/StatefulPartitionedCall≠
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_8_138494conv2d_8_138496*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_1364752"
 conv2d_8/StatefulPartitionedCall°
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_5_138499batch_normalization_5_138501batch_normalization_5_138503batch_normalization_5_138505*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1371972/
-batch_normalization_5/StatefulPartitionedCallР
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1372392
add_2/PartitionedCallб
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1372532
activation_2/PartitionedCallД
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1366182*
(global_average_pooling2d/PartitionedCallЁ
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1372682
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_138512dense_138514*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1373032
dense/StatefulPartitionedCall”
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1373362
dropout/PartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_138518dense_1_138520*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1373762!
dense_1/StatefulPartitionedCallџ
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1374092
dropout_1/PartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_138524dense_2_138526*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1374492!
dense_2/StatefulPartitionedCallЄ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_138405*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add®
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_138407*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/addЊ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_138410*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/addЃ
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_138412*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/addЊ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_138424*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/addЃ
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_138426*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/addЊ
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_138440*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/addЃ
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_138442*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/addЊ
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_138445*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/addЃ
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_138447*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/addЊ
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_138459*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/addЃ
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_138461*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/addЊ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_138475*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/addЃ
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_138477*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/addЊ
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_138480*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/addЃ
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_138482*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/addЊ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_138494*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/addЃ
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_138496*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/addЃ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_138512*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add¶
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_138514*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_138518* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addђ
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_138520*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addі
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_138524*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЂ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_138526*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addЈ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ґН
ф
A__inference_model_layer_call_and_return_conditional_losses_137658
input_1
conv2d_136628
conv2d_136630
conv2d_1_136633
conv2d_1_136635
batch_normalization_136713
batch_normalization_136715
batch_normalization_136717
batch_normalization_136719
conv2d_2_136722
conv2d_2_136724 
batch_normalization_1_136802 
batch_normalization_1_136804 
batch_normalization_1_136806 
batch_normalization_1_136808
conv2d_3_136839
conv2d_3_136841
conv2d_4_136844
conv2d_4_136846 
batch_normalization_2_136924 
batch_normalization_2_136926 
batch_normalization_2_136928 
batch_normalization_2_136930
conv2d_5_136933
conv2d_5_136935 
batch_normalization_3_137013 
batch_normalization_3_137015 
batch_normalization_3_137017 
batch_normalization_3_137019
conv2d_6_137050
conv2d_6_137052
conv2d_7_137055
conv2d_7_137057 
batch_normalization_4_137135 
batch_normalization_4_137137 
batch_normalization_4_137139 
batch_normalization_4_137141
conv2d_8_137144
conv2d_8_137146 
batch_normalization_5_137224 
batch_normalization_5_137226 
batch_normalization_5_137228 
batch_normalization_5_137230
dense_137314
dense_137316
dense_1_137387
dense_1_137389
dense_2_137460
dense_2_137462
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallф
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_136628conv2d_136630*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1355442 
conv2d/StatefulPartitionedCallЮ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_136633conv2d_1_136635*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1355822"
 conv2d_1/StatefulPartitionedCallС
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_136713batch_normalization_136715batch_normalization_136717batch_normalization_136719*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1366682-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_136722conv2d_2_136724*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1357452"
 conv2d_2/StatefulPartitionedCallЯ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_1_136802batch_normalization_1_136804batch_normalization_1_136806batch_normalization_1_136808*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1367572/
-batch_normalization_1/StatefulPartitionedCallИ
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1368172
add/PartitionedCallў
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1368312
activation/PartitionedCallЪ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_3_136839conv2d_3_136841*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1359092"
 conv2d_3/StatefulPartitionedCall†
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_136844conv2d_4_136846*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1359472"
 conv2d_4/StatefulPartitionedCallЯ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_2_136924batch_normalization_2_136926batch_normalization_2_136928batch_normalization_2_136930*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1368792/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_5_136933conv2d_5_136935*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1361102"
 conv2d_5/StatefulPartitionedCallЯ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_3_137013batch_normalization_3_137015batch_normalization_3_137017batch_normalization_3_137019*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1369682/
-batch_normalization_3/StatefulPartitionedCallР
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1370282
add_1/PartitionedCallб
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1370422
activation_1/PartitionedCallЬ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_6_137050conv2d_6_137052*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1362742"
 conv2d_6/StatefulPartitionedCall†
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_137055conv2d_7_137057*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1363122"
 conv2d_7/StatefulPartitionedCallЯ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_4_137135batch_normalization_4_137137batch_normalization_4_137139batch_normalization_4_137141*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1370902/
-batch_normalization_4/StatefulPartitionedCall≠
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_8_137144conv2d_8_137146*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_1364752"
 conv2d_8/StatefulPartitionedCallЯ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_5_137224batch_normalization_5_137226batch_normalization_5_137228batch_normalization_5_137230*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1371792/
-batch_normalization_5/StatefulPartitionedCallР
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1372392
add_2/PartitionedCallб
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1372532
activation_2/PartitionedCallД
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1366182*
(global_average_pooling2d/PartitionedCallЁ
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1372682
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_137314dense_137316*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1373032
dense/StatefulPartitionedCallл
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1373312!
dropout/StatefulPartitionedCallУ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_137387dense_1_137389*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1373762!
dense_1/StatefulPartitionedCallХ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1374042#
!dropout_1/StatefulPartitionedCallФ
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_137460dense_2_137462*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1374492!
dense_2/StatefulPartitionedCallЄ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_136628*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add®
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_136630*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/addЊ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_136633*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/addЃ
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_136635*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/addЊ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_136722*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/addЃ
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_136724*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/addЊ
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_136839*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/addЃ
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_136841*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/addЊ
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_136844*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/addЃ
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_136846*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/addЊ
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_136933*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/addЃ
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_136935*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/addЊ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_137050*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/addЃ
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_137052*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/addЊ
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_137055*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/addЃ
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_137057*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/addЊ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_137144*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/addЃ
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_137146*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/addЃ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_137314*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add¶
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_137316*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_137387* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addђ
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_137389*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addі
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_137460*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЂ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_137462*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addэ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
с
D
(__inference_flatten_layer_call_fn_141419

inputs
identityҐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1372682
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ё
o
__inference_loss_fn_6_141720>
:conv2d_3_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/addf
IdentityIdentity#conv2d_3/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
я
p
__inference_loss_fn_14_141824>
:conv2d_7_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_7_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/addf
IdentityIdentity#conv2d_7/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
х
D
(__inference_dropout_layer_call_fn_141498

inputs
identity£
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1373362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ю
P
$__inference_add_layer_call_fn_140610
inputs_0
inputs_1
identity≥
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1368172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22:€€€€€€€€€22:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/1
Ы
I
-__inference_activation_1_layer_call_fn_141014

inputs
identityѓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1370422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
ѓ
o
__inference_loss_fn_22_141928=
9dense_2_kernel_regularizer_square_readvariableop_resource
identityИя
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/adde
IdentityIdentity"dense_2/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
„
m
A__inference_add_2_layer_call_and_return_conditional_losses_141392
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:€€€€€€€€€22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22@:€€€€€€€€€22@:Y U
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/1
з$
Ў
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_136569

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д 
ђ
D__inference_conv2d_6_layer_call_and_return_conditional_losses_136274

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
ReluЌ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/addЊ
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Я$
Ў
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ѓ
©
6__inference_batch_normalization_5_layer_call_fn_141298

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1371792
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш
©
6__inference_batch_normalization_5_layer_call_fn_141386

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1366002
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_136072

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
n
__inference_loss_fn_13_141811<
8conv2d_6_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_6_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/addd
IdentityIdentity!conv2d_6/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
я
p
__inference_loss_fn_12_141798>
:conv2d_6_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_6_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/addf
IdentityIdentity#conv2d_6/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Г
m
__inference_loss_fn_9_141759<
8conv2d_4_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_4_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/addd
IdentityIdentity!conv2d_4/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ѕ
k
A__inference_add_2_layer_call_and_return_conditional_losses_137239

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:€€€€€€€€€22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22@:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
Є
k
__inference_loss_fn_19_1418899
5dense_bias_regularizer_square_readvariableop_resource
identityИѕ
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/adda
IdentityIdentitydense/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ћ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_141567

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
€
Ђ
C__inference_dense_1_layer_call_and_return_conditional_losses_137376

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relu≈
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addљ
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140788

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
™
m
__inference_loss_fn_0_141642<
8conv2d_kernel_regularizer_square_readvariableop_resource
identityИг
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/addd
IdentityIdentity!conv2d/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
И
b
C__inference_dropout_layer_call_and_return_conditional_losses_137331

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
§†
Т
A__inference_model_layer_call_and_return_conditional_losses_139652

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИҐ7batch_normalization/AssignMovingAvg/AssignSubVariableOpҐ9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpЄ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d/BiasAdd∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpѕ
conv2d_1/Conv2DConv2Dconv2d/BiasAdd:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_1/Relu∞
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpй
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ƒ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization/Constм
)batch_normalization/AssignMovingAvg/sub/xConst*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)batch_normalization/AssignMovingAvg/sub/x£
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0"batch_normalization/Const:output:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subб
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp¬
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg/sub_1Ђ
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul”
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpт
+batch_normalization/AssignMovingAvg_1/sub/xConst*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization/AssignMovingAvg_1/sub/xЂ
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0"batch_normalization/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subз
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpќ
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2-
+batch_normalization/AssignMovingAvg_1/sub_1µ
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mulб
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpа
conv2d_2/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_2/BiasAddґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ќ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_1/Constт
+batch_normalization_1/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_1/AssignMovingAvg/sub/x≠
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subз
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg/sub_1µ
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/mulб
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_1/AssignMovingAvg_1/sub/xµ
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subн
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2/
-batch_normalization_1/AssignMovingAvg_1/sub_1њ
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/mulп
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpЪ
add/addAddV2*batch_normalization_1/FusedBatchNormV3:y:0conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222	
add/addq
activation/ReluReluadd/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€222
activation/Relu∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp’
conv2d_3/Conv2DConv2Dactivation/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_3/Relu∞
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_4/Conv2D/ReadVariableOp”
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_4/Conv2DІ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOpђ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_4/Reluґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1–
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_2/Constт
+batch_normalization_2/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_2/AssignMovingAvg/sub/x≠
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subз
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg/sub_1µ
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/mulб
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_2/AssignMovingAvg_1/sub/xµ
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subн
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_2/AssignMovingAvg_1/sub_1њ
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/mulп
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp∞
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_5/Conv2D/ReadVariableOpв
conv2d_5/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_5/Conv2DІ
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOpђ
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_5/BiasAddґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ќ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_3/Constт
+batch_normalization_3/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_3/AssignMovingAvg/sub/x≠
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0$batch_normalization_3/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/subз
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg/sub_1µ
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/mulб
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_3/AssignMovingAvg_1/sub/xµ
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0$batch_normalization_3/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/subн
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_3/AssignMovingAvg_1/sub_1њ
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/mulп
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ
	add_1/addAddV2*batch_normalization_3/FusedBatchNormV3:y:0conv2d_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
	add_1/addw
activation_1/ReluReluadd_1/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
activation_1/Relu∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_6/Conv2D/ReadVariableOp„
conv2d_6/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_6/Relu∞
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp”
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_7/Conv2DІ
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOpђ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_7/Reluґ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpЉ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1й
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1–
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2(
&batch_normalization_4/FusedBatchNormV3
batch_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_4/Constт
+batch_normalization_4/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_4/AssignMovingAvg/sub/x≠
)batch_normalization_4/AssignMovingAvg/subSub4batch_normalization_4/AssignMovingAvg/sub/x:output:0$batch_normalization_4/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/subз
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_4/AssignMovingAvg/sub_1Sub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_4/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2-
+batch_normalization_4/AssignMovingAvg/sub_1µ
)batch_normalization_4/AssignMovingAvg/mulMul/batch_normalization_4/AssignMovingAvg/sub_1:z:0-batch_normalization_4/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2+
)batch_normalization_4/AssignMovingAvg/mulб
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_4/AssignMovingAvg_1/sub/xµ
+batch_normalization_4/AssignMovingAvg_1/subSub6batch_normalization_4/AssignMovingAvg_1/sub/x:output:0$batch_normalization_4/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/subн
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_4/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2/
-batch_normalization_4/AssignMovingAvg_1/sub_1њ
+batch_normalization_4/AssignMovingAvg_1/mulMul1batch_normalization_4/AssignMovingAvg_1/sub_1:z:0/batch_normalization_4/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2-
+batch_normalization_4/AssignMovingAvg_1/mulп
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp∞
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpв
conv2d_8/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_8/Conv2DІ
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpђ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_8/BiasAddґ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpЉ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1й
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ќ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2(
&batch_normalization_5/FusedBatchNormV3
batch_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_5/Constт
+batch_normalization_5/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_5/AssignMovingAvg/sub/x≠
)batch_normalization_5/AssignMovingAvg/subSub4batch_normalization_5/AssignMovingAvg/sub/x:output:0$batch_normalization_5/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/subз
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_5/AssignMovingAvg/sub_1Sub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_5/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2-
+batch_normalization_5/AssignMovingAvg/sub_1µ
)batch_normalization_5/AssignMovingAvg/mulMul/batch_normalization_5/AssignMovingAvg/sub_1:z:0-batch_normalization_5/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2+
)batch_normalization_5/AssignMovingAvg/mulб
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_5/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_5/AssignMovingAvg_1/sub/xµ
+batch_normalization_5/AssignMovingAvg_1/subSub6batch_normalization_5/AssignMovingAvg_1/sub/x:output:0$batch_normalization_5/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/subн
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_5/AssignMovingAvg_1/sub_1Sub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_5/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2/
-batch_normalization_5/AssignMovingAvg_1/sub_1њ
+batch_normalization_5/AssignMovingAvg_1/mulMul1batch_normalization_5/AssignMovingAvg_1/sub_1:z:0/batch_normalization_5/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2-
+batch_normalization_5/AssignMovingAvg_1/mulп
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpҐ
	add_2/addAddV2*batch_normalization_5/FusedBatchNormV3:y:0conv2d_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
	add_2/addw
activation_2/ReluReluadd_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
activation_2/Relu≥
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices”
global_average_pooling2d/MeanMeanactivation_2/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling2d/Meano
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
flatten/ConstЯ
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
flatten/Reshape†
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstЮ
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeЌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yя
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/GreaterEqualШ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/CastЫ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/dropout/Mul_1І
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_1/MatMul/ReadVariableOpЯ
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/MatMul•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOpҐ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const¶
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape”
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yз
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_1/dropout/GreaterEqualЮ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_1/dropout/Cast£
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_1/dropout/Mul_1¶
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_2/MatMul/ReadVariableOp†
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/Sigmoid–
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/addЅ
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/add÷
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add«
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/add÷
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add«
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/add÷
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add«
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/add÷
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add«
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/add÷
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add«
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/add÷
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add«
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/add÷
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add«
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/add÷
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/add«
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/add∆
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addњ
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addЌ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add≈
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addћ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addƒ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addњ
IdentityIdentitydense_2/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ѕ
k
__inference_loss_fn_1_141655:
6conv2d_bias_regularizer_square_readvariableop_resource
identityИ—
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp6conv2d_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/addb
IdentityIdentityconv2d/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
µi
ќ
__inference__traced_save_142112
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d73d36647a2c44cea1457379dfa22e11/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЅ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*”
value…B∆0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesи
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesз
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *>
dtypes4
2202
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ј
_input_shapes•
Ґ: ::::::::::::::: : :  : : : : : :  : : : : : : @:@:@@:@:@:@:@:@:@@:@:@:@:@:@:	@А:А:
АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:,%(
&
_output_shapes
:@@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:%+!

_output_shapes
:	@А:!,

_output_shapes	
:А:&-"
 
_output_shapes
:
АА:!.

_output_shapes	
:А:%/!

_output_shapes
:	А: 0

_output_shapes
::1

_output_shapes
: 
ё
o
__inference_loss_fn_2_141668>
:conv2d_1_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/addf
IdentityIdentity#conv2d_1/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
»
К
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140891

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
÷
d
H__inference_activation_2_layer_call_and_return_conditional_losses_137253

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
ч
Ђ
C__inference_dense_2_layer_call_and_return_conditional_losses_137449

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidƒ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЉ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э$
÷
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140376

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш
©
6__inference_batch_normalization_1_layer_call_fn_140598

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1358702
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
П
И
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140319

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
™
B__inference_conv2d_layer_call_and_return_conditional_losses_135544

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAdd…
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/addЇ
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140966

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
П
И
O__inference_batch_normalization_layer_call_and_return_conditional_losses_135707

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
÷
—
$__inference_signature_wrapper_139188
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_1355172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ё
o
__inference_loss_fn_8_141746>
:conv2d_4_kernel_regularizer_square_readvariableop_resource
identityИй
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_4_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/addf
IdentityIdentity#conv2d_4/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_136041

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
б
~
)__inference_conv2d_6_layer_call_fn_136284

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1362742
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
К
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_137404

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
}
(__inference_dense_2_layer_call_fn_141629

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1374492
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ
І
4__inference_batch_normalization_layer_call_fn_140420

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1366862
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
©
_
C__inference_flatten_layer_call_and_return_conditional_losses_137268

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Я$
Ў
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_136757

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141164

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
мЉ
Ї
A__inference_model_layer_call_and_return_conditional_losses_140024

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИ™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpЄ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d/BiasAdd∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpѕ
conv2d_1/Conv2DConv2Dconv2d/BiasAdd:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_1/Relu∞
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpй
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1„
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpа
conv2d_2/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_2/BiasAddґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Ъ
add/addAddV2*batch_normalization_1/FusedBatchNormV3:y:0conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222	
add/addq
activation/ReluReluadd/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€222
activation/Relu∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp’
conv2d_3/Conv2DConv2Dactivation/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_3/Relu∞
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_4/Conv2D/ReadVariableOp”
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_4/Conv2DІ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOpђ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_4/Reluґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1г
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3∞
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_5/Conv2D/ReadVariableOpв
conv2d_5/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_5/Conv2DІ
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOpђ
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_5/BiasAddґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Ґ
	add_1/addAddV2*batch_normalization_3/FusedBatchNormV3:y:0conv2d_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
	add_1/addw
activation_1/ReluReluadd_1/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
activation_1/Relu∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_6/Conv2D/ReadVariableOp„
conv2d_6/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_6/Relu∞
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp”
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_7/Conv2DІ
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOpђ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_7/Reluґ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpЉ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1й
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1г
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3∞
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpв
conv2d_8/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_8/Conv2DІ
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpђ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_8/BiasAddґ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpЉ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1й
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3Ґ
	add_2/addAddV2*batch_normalization_5/FusedBatchNormV3:y:0conv2d_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
	add_2/addw
activation_2/ReluReluadd_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
activation_2/Relu≥
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices”
global_average_pooling2d/MeanMeanactivation_2/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling2d/Meano
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
flatten/ConstЯ
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
flatten/Reshape†
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/IdentityІ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_1/MatMul/ReadVariableOpЯ
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/MatMul•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOpҐ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/ReluГ
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_1/Identity¶
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_2/MatMul/ReadVariableOp†
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_2/Sigmoid–
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpЄ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/SquareЫ
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/Constґ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/SumЗ
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d/kernel/Regularizer/mul/xЄ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulЗ
conv2d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/add/xµ
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/add/x:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/addЅ
-conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d/bias/Regularizer/Square/ReadVariableOp¶
conv2d/bias/Regularizer/SquareSquare5conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2 
conv2d/bias/Regularizer/SquareИ
conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
conv2d/bias/Regularizer/ConstЃ
conv2d/bias/Regularizer/SumSum"conv2d/bias/Regularizer/Square:y:0&conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/SumГ
conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
conv2d/bias/Regularizer/mul/x∞
conv2d/bias/Regularizer/mulMul&conv2d/bias/Regularizer/mul/x:output:0$conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/mulГ
conv2d/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv2d/bias/Regularizer/add/x≠
conv2d/bias/Regularizer/addAddV2&conv2d/bias/Regularizer/add/x:output:0conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/bias/Regularizer/add÷
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_1/kernel/Regularizer/SquareЯ
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstЊ
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/SumЛ
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_1/kernel/Regularizer/mul/xј
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulЛ
!conv2d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/add/xљ
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/add/x:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add«
/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_1/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_1/bias/Regularizer/SquareSquare7conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_1/bias/Regularizer/SquareМ
conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_1/bias/Regularizer/Constґ
conv2d_1/bias/Regularizer/SumSum$conv2d_1/bias/Regularizer/Square:y:0(conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/SumЗ
conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_1/bias/Regularizer/mul/xЄ
conv2d_1/bias/Regularizer/mulMul(conv2d_1/bias/Regularizer/mul/x:output:0&conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/mulЗ
conv2d_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_1/bias/Regularizer/add/xµ
conv2d_1/bias/Regularizer/addAddV2(conv2d_1/bias/Regularizer/add/x:output:0!conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_1/bias/Regularizer/add÷
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_2/kernel/Regularizer/SquareЯ
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstЊ
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/SumЛ
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_2/kernel/Regularizer/mul/xј
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЛ
!conv2d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/add/xљ
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/add/x:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add«
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/add÷
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add«
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/add÷
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_4/kernel/Regularizer/SquareЯ
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_4/kernel/Regularizer/ConstЊ
conv2d_4/kernel/Regularizer/SumSum&conv2d_4/kernel/Regularizer/Square:y:0*conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/SumЛ
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_4/kernel/Regularizer/mul/xј
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mulЛ
!conv2d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/add/xљ
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/add/x:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add«
/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_4/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_4/bias/Regularizer/SquareSquare7conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_4/bias/Regularizer/SquareМ
conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_4/bias/Regularizer/Constґ
conv2d_4/bias/Regularizer/SumSum$conv2d_4/bias/Regularizer/Square:y:0(conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/SumЗ
conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_4/bias/Regularizer/mul/xЄ
conv2d_4/bias/Regularizer/mulMul(conv2d_4/bias/Regularizer/mul/x:output:0&conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/mulЗ
conv2d_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_4/bias/Regularizer/add/xµ
conv2d_4/bias/Regularizer/addAddV2(conv2d_4/bias/Regularizer/add/x:output:0!conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_4/bias/Regularizer/add÷
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_5/kernel/Regularizer/SquareЯ
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_5/kernel/Regularizer/ConstЊ
conv2d_5/kernel/Regularizer/SumSum&conv2d_5/kernel/Regularizer/Square:y:0*conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/SumЛ
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_5/kernel/Regularizer/mul/xј
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mulЛ
!conv2d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/add/xљ
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/add/x:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add«
/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_5/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_5/bias/Regularizer/SquareSquare7conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_5/bias/Regularizer/SquareМ
conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_5/bias/Regularizer/Constґ
conv2d_5/bias/Regularizer/SumSum$conv2d_5/bias/Regularizer/Square:y:0(conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/SumЗ
conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_5/bias/Regularizer/mul/xЄ
conv2d_5/bias/Regularizer/mulMul(conv2d_5/bias/Regularizer/mul/x:output:0&conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/mulЗ
conv2d_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_5/bias/Regularizer/add/xµ
conv2d_5/bias/Regularizer/addAddV2(conv2d_5/bias/Regularizer/add/x:output:0!conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_5/bias/Regularizer/add÷
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/SquareЯ
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_6/kernel/Regularizer/ConstЊ
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/SumЛ
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_6/kernel/Regularizer/mul/xј
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mulЛ
!conv2d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/add/xљ
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/add/x:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add«
/conv2d_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_6/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_6/bias/Regularizer/SquareSquare7conv2d_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_6/bias/Regularizer/SquareМ
conv2d_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_6/bias/Regularizer/Constґ
conv2d_6/bias/Regularizer/SumSum$conv2d_6/bias/Regularizer/Square:y:0(conv2d_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/SumЗ
conv2d_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_6/bias/Regularizer/mul/xЄ
conv2d_6/bias/Regularizer/mulMul(conv2d_6/bias/Regularizer/mul/x:output:0&conv2d_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/mulЗ
conv2d_6/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_6/bias/Regularizer/add/xµ
conv2d_6/bias/Regularizer/addAddV2(conv2d_6/bias/Regularizer/add/x:output:0!conv2d_6/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_6/bias/Regularizer/add÷
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_7/kernel/Regularizer/SquareЯ
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_7/kernel/Regularizer/ConstЊ
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/SumЛ
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_7/kernel/Regularizer/mul/xј
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mulЛ
!conv2d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/add/xљ
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/add/x:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add«
/conv2d_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_7/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_7/bias/Regularizer/SquareSquare7conv2d_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_7/bias/Regularizer/SquareМ
conv2d_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_7/bias/Regularizer/Constґ
conv2d_7/bias/Regularizer/SumSum$conv2d_7/bias/Regularizer/Square:y:0(conv2d_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/SumЗ
conv2d_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_7/bias/Regularizer/mul/xЄ
conv2d_7/bias/Regularizer/mulMul(conv2d_7/bias/Regularizer/mul/x:output:0&conv2d_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/mulЗ
conv2d_7/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_7/bias/Regularizer/add/xµ
conv2d_7/bias/Regularizer/addAddV2(conv2d_7/bias/Regularizer/add/x:output:0!conv2d_7/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_7/bias/Regularizer/add÷
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/add«
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/add∆
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addњ
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addЌ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add≈
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addћ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpі
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2#
!dense_2/kernel/Regularizer/SquareХ
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstЇ
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/SumЙ
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_2/kernel/Regularizer/mul/xЉ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mulЙ
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xє
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addƒ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22:::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
Г
m
__inference_loss_fn_5_141707<
8conv2d_2_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_2_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_2/bias/Regularizer/SquareМ
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Constґ
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/SumЗ
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_2/bias/Regularizer/mul/xЄ
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mulЗ
conv2d_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_2/bias/Regularizer/add/xµ
conv2d_2/bias/Regularizer/addAddV2(conv2d_2/bias/Regularizer/add/x:output:0!conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/addd
IdentityIdentity!conv2d_2/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
€
Ђ
C__inference_dense_1_layer_call_and_return_conditional_losses_141541

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relu≈
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addљ
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp™
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_1/bias/Regularizer/SquareК
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const≤
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/SumЕ
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_1/bias/Regularizer/mul/xі
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mulЕ
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x±
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Г
m
__inference_loss_fn_7_141733<
8conv2d_3_bias_regularizer_square_readvariableop_resource
identityИ„
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/addd
IdentityIdentity!conv2d_3/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
”Ќ
з
"__inference__traced_restore_142268
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance&
"assignvariableop_8_conv2d_2_kernel$
 assignvariableop_9_conv2d_2_bias3
/assignvariableop_10_batch_normalization_1_gamma2
.assignvariableop_11_batch_normalization_1_beta9
5assignvariableop_12_batch_normalization_1_moving_mean=
9assignvariableop_13_batch_normalization_1_moving_variance'
#assignvariableop_14_conv2d_3_kernel%
!assignvariableop_15_conv2d_3_bias'
#assignvariableop_16_conv2d_4_kernel%
!assignvariableop_17_conv2d_4_bias3
/assignvariableop_18_batch_normalization_2_gamma2
.assignvariableop_19_batch_normalization_2_beta9
5assignvariableop_20_batch_normalization_2_moving_mean=
9assignvariableop_21_batch_normalization_2_moving_variance'
#assignvariableop_22_conv2d_5_kernel%
!assignvariableop_23_conv2d_5_bias3
/assignvariableop_24_batch_normalization_3_gamma2
.assignvariableop_25_batch_normalization_3_beta9
5assignvariableop_26_batch_normalization_3_moving_mean=
9assignvariableop_27_batch_normalization_3_moving_variance'
#assignvariableop_28_conv2d_6_kernel%
!assignvariableop_29_conv2d_6_bias'
#assignvariableop_30_conv2d_7_kernel%
!assignvariableop_31_conv2d_7_bias3
/assignvariableop_32_batch_normalization_4_gamma2
.assignvariableop_33_batch_normalization_4_beta9
5assignvariableop_34_batch_normalization_4_moving_mean=
9assignvariableop_35_batch_normalization_4_moving_variance'
#assignvariableop_36_conv2d_8_kernel%
!assignvariableop_37_conv2d_8_bias3
/assignvariableop_38_batch_normalization_5_gamma2
.assignvariableop_39_batch_normalization_5_beta9
5assignvariableop_40_batch_normalization_5_moving_mean=
9assignvariableop_41_batch_normalization_5_moving_variance$
 assignvariableop_42_dense_kernel"
assignvariableop_43_dense_bias&
"assignvariableop_44_dense_1_kernel$
 assignvariableop_45_dense_1_bias&
"assignvariableop_46_dense_2_kernel$
 assignvariableop_47_dense_2_bias
identity_49ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1«
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*”
value…B∆0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesо
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*÷
_output_shapes√
ј::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
2202
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityО
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ш
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ґ
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7ђ
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ш
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11І
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13≤
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ь
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_3_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ъ
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_3_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Ь
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_4_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ъ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_4_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_2_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19І
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_2_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21≤
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Ь
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_5_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ъ
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_5_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24®
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_3_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25І
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_3_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ѓ
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_3_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27≤
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_3_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ь
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_6_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Ъ
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_6_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Ь
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_7_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ъ
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_7_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32®
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_4_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33І
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_4_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Ѓ
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_4_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35≤
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_4_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Ь
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_8_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Ъ
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_8_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38®
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_5_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39І
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_5_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Ѓ
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_5_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41≤
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_5_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Щ
AssignVariableOp_42AssignVariableOp assignvariableop_42_dense_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43Ч
AssignVariableOp_43AssignVariableOpassignvariableop_43_dense_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44Ы
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_1_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45Щ
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_1_biasIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46Ы
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_2_kernelIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47Щ
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_2_biasIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpю
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48Л	
Identity_49IdentityIdentity_48:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_49"#
identity_49Identity_49:output:0*„
_input_shapes≈
¬: ::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
∞
©
6__inference_batch_normalization_2_layer_call_fn_140739

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1368972
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з$
Ў
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_135839

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
©
A__inference_dense_layer_call_and_return_conditional_losses_141462

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Reluј
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЃ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2!
dense/kernel/Regularizer/SquareС
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const≤
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЕ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense/kernel/Regularizer/mul/xі
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЕ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x±
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addє
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp§
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
dense/bias/Regularizer/SquareЖ
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const™
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/SumБ
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dense/bias/Regularizer/mul/xђ
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mulБ
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x©
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Н
ђ
D__inference_conv2d_8_layer_call_and_return_conditional_losses_136475

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddЌ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_8/kernel/Regularizer/SquareЯ
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/ConstЊ
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/SumЛ
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_8/kernel/Regularizer/mul/xј
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulЛ
!conv2d_8/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_8/kernel/Regularizer/add/xљ
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/add/x:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/addЊ
/conv2d_8/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_8/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_8/bias/Regularizer/SquareSquare7conv2d_8/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_8/bias/Regularizer/SquareМ
conv2d_8/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_8/bias/Regularizer/Constґ
conv2d_8/bias/Regularizer/SumSum$conv2d_8/bias/Regularizer/Square:y:0(conv2d_8/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/SumЗ
conv2d_8/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_8/bias/Regularizer/mul/xЄ
conv2d_8/bias/Regularizer/mulMul(conv2d_8/bias/Regularizer/mul/x:output:0&conv2d_8/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/mulЗ
conv2d_8/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_8/bias/Regularizer/add/xµ
conv2d_8/bias/Regularizer/addAddV2(conv2d_8/bias/Regularizer/add/x:output:0!conv2d_8/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_8/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
И
b
C__inference_dropout_layer_call_and_return_conditional_losses_141483

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ы
}
(__inference_dense_1_layer_call_fn_141550

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1373762
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д 
ђ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_135909

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
ReluЌ
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpЊ
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_3/kernel/Regularizer/SquareЯ
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/ConstЊ
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/SumЛ
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!conv2d_3/kernel/Regularizer/mul/xј
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mulЛ
!conv2d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/add/xљ
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/add/x:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/addЊ
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOpђ
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_3/bias/Regularizer/SquareМ
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Constґ
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/SumЗ
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
conv2d_3/bias/Regularizer/mul/xЄ
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mulЗ
conv2d_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_3/bias/Regularizer/add/xµ
conv2d_3/bias/Regularizer/addAddV2(conv2d_3/bias/Regularizer/add/x:output:0!conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
к
m
__inference_loss_fn_23_141941;
7dense_2_bias_regularizer_square_readvariableop_resource
identityИ‘
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_2_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp©
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/SquareК
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const≤
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/SumЕ
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_2/bias/Regularizer/mul/xі
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mulЕ
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/x±
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addc
IdentityIdentity dense_2/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ґ
R
&__inference_add_1_layer_call_fn_141004
inputs_0
inputs_1
identityµ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1370282
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22 :€€€€€€€€€22 :Y U
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/1
б
~
)__inference_conv2d_3_layer_call_fn_135919

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1359092
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≤
serving_defaultЮ
C
input_18
serving_default_input_1:0€€€€€€€€€22;
dense_20
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:хв
ЖГ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer-27
layer_with_weights-17
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"
signatures
Џ_default_save_signature
џ__call__
+№&call_and_return_all_conditional_losses"ёъ
_tf_keras_model√ъ{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}], ["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}], ["conv2d_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}], ["conv2d_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}], ["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}], ["conv2d_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}], ["conv2d_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
щ"ц
_tf_keras_input_layer÷{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
 


#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"£	
_tf_keras_layerЙ	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}}
ќ


)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
я__call__
+а&call_and_return_all_conditional_losses"І	
_tf_keras_layerН	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
Х	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
б__call__
+в&call_and_return_all_conditional_losses"њ
_tf_keras_layer•{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
–


8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
г__call__
+д&call_and_return_all_conditional_losses"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
Щ	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
Ф
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
з__call__
+и&call_and_return_all_conditional_losses"Г
_tf_keras_layerй{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 16]}, {"class_name": "TensorShape", "items": [null, 50, 50, 16]}]}
∞
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
й__call__
+к&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
ќ


Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
л__call__
+м&call_and_return_all_conditional_losses"І	
_tf_keras_layerН	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
ќ


Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
н__call__
+о&call_and_return_all_conditional_losses"І	
_tf_keras_layerН	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Щ	
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
atrainable_variables
bregularization_losses
c	keras_api
п__call__
+р&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
–


dkernel
ebias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
с__call__
+т&call_and_return_all_conditional_losses"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Щ	
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ш
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"З
_tf_keras_layerн{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 32]}, {"class_name": "TensorShape", "items": [null, 50, 50, 32]}]}
і
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
ѕ


{kernel
|bias
}	variables
~trainable_variables
regularization_losses
А	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"І	
_tf_keras_layerН	{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
‘

Бkernel
	Вbias
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"І	
_tf_keras_layerН	{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Ґ	
	Зaxis

Иgamma
	Йbeta
Кmoving_mean
Лmoving_variance
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
÷

Рkernel
	Сbias
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
€__call__
+А&call_and_return_all_conditional_losses"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Ґ	
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Ь
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"З
_tf_keras_layerн{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 64]}, {"class_name": "TensorShape", "items": [null, 50, 50, 64]}]}
Є
£	variables
§trainable_variables
•regularization_losses
¶	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
ц
І	variables
®trainable_variables
©regularization_losses
™	keras_api
З__call__
+И&call_and_return_all_conditional_losses"б
_tf_keras_layer«{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
≈
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"∞
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
№
ѓkernel
	∞bias
±	variables
≤trainable_variables
≥regularization_losses
і	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"ѓ
_tf_keras_layerХ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ƒ
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"ѓ
_tf_keras_layerХ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
в
єkernel
	Їbias
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"µ
_tf_keras_layerЫ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
»
њ	variables
јtrainable_variables
Ѕregularization_losses
¬	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
г
√kernel
	ƒbias
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"ґ
_tf_keras_layerЬ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
®
#0
$1
)2
*3
04
15
26
37
88
99
?10
@11
A12
B13
O14
P15
U16
V17
\18
]19
^20
_21
d22
e23
k24
l25
m26
n27
{28
|29
Б30
В31
И32
Й33
К34
Л35
Р36
С37
Ч38
Ш39
Щ40
Ъ41
ѓ42
∞43
є44
Ї45
√46
ƒ47"
trackable_list_wrapper
ƒ
#0
$1
)2
*3
04
15
86
97
?8
@9
O10
P11
U12
V13
\14
]15
d16
e17
k18
l19
{20
|21
Б22
В23
И24
Й25
Р26
С27
Ч28
Ш29
ѓ30
∞31
є32
Ї33
√34
ƒ35"
trackable_list_wrapper
о
Х0
Ц1
Ч2
Ш3
Щ4
Ъ5
Ы6
Ь7
Э8
Ю9
Я10
†11
°12
Ґ13
£14
§15
•16
¶17
І18
®19
©20
™21
Ђ22
ђ23"
trackable_list_wrapper
”
…layers
	variables
trainable_variables
 metrics
 regularization_losses
Ћlayer_metrics
 ћlayer_regularization_losses
Ќnon_trainable_variables
џ__call__
Џ_default_save_signature
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
-
≠serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
µ
ќlayers
%	variables
ѕmetrics
&trainable_variables
'regularization_losses
–layer_metrics
 —layer_regularization_losses
“non_trainable_variables
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
µ
”layers
+	variables
‘metrics
,trainable_variables
-regularization_losses
’layer_metrics
 ÷layer_regularization_losses
„non_trainable_variables
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ўlayers
4	variables
ўmetrics
5trainable_variables
6regularization_losses
Џlayer_metrics
 џlayer_regularization_losses
№non_trainable_variables
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
µ
Ёlayers
:	variables
ёmetrics
;trainable_variables
<regularization_losses
яlayer_metrics
 аlayer_regularization_losses
бnon_trainable_variables
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
вlayers
C	variables
гmetrics
Dtrainable_variables
Eregularization_losses
дlayer_metrics
 еlayer_regularization_losses
жnon_trainable_variables
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
зlayers
G	variables
иmetrics
Htrainable_variables
Iregularization_losses
йlayer_metrics
 кlayer_regularization_losses
лnon_trainable_variables
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
мlayers
K	variables
нmetrics
Ltrainable_variables
Mregularization_losses
оlayer_metrics
 пlayer_regularization_losses
рnon_trainable_variables
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_3/kernel
: 2conv2d_3/bias
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
µ
сlayers
Q	variables
тmetrics
Rtrainable_variables
Sregularization_losses
уlayer_metrics
 фlayer_regularization_losses
хnon_trainable_variables
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_4/kernel
: 2conv2d_4/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
µ
цlayers
W	variables
чmetrics
Xtrainable_variables
Yregularization_losses
шlayer_metrics
 щlayer_regularization_losses
ъnon_trainable_variables
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ыlayers
`	variables
ьmetrics
atrainable_variables
bregularization_losses
эlayer_metrics
 юlayer_regularization_losses
€non_trainable_variables
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_5/kernel
: 2conv2d_5/bias
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
0
Я0
†1"
trackable_list_wrapper
µ
Аlayers
f	variables
Бmetrics
gtrainable_variables
hregularization_losses
Вlayer_metrics
 Гlayer_regularization_losses
Дnon_trainable_variables
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
<
k0
l1
m2
n3"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Еlayers
o	variables
Жmetrics
ptrainable_variables
qregularization_losses
Зlayer_metrics
 Иlayer_regularization_losses
Йnon_trainable_variables
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Кlayers
s	variables
Лmetrics
ttrainable_variables
uregularization_losses
Мlayer_metrics
 Нlayer_regularization_losses
Оnon_trainable_variables
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Пlayers
w	variables
Рmetrics
xtrainable_variables
yregularization_losses
Сlayer_metrics
 Тlayer_regularization_losses
Уnon_trainable_variables
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_6/kernel
:@2conv2d_6/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
0
°0
Ґ1"
trackable_list_wrapper
µ
Фlayers
}	variables
Хmetrics
~trainable_variables
regularization_losses
Цlayer_metrics
 Чlayer_regularization_losses
Шnon_trainable_variables
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
0
Б0
В1"
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
0
£0
§1"
trackable_list_wrapper
Є
Щlayers
Г	variables
Ъmetrics
Дtrainable_variables
Еregularization_losses
Ыlayer_metrics
 Ьlayer_regularization_losses
Эnon_trainable_variables
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
@
И0
Й1
К2
Л3"
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Юlayers
М	variables
Яmetrics
Нtrainable_variables
Оregularization_losses
†layer_metrics
 °layer_regularization_losses
Ґnon_trainable_variables
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_8/kernel
:@2conv2d_8/bias
0
Р0
С1"
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
0
•0
¶1"
trackable_list_wrapper
Є
£layers
Т	variables
§metrics
Уtrainable_variables
Фregularization_losses
•layer_metrics
 ¶layer_regularization_losses
Іnon_trainable_variables
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
@
Ч0
Ш1
Щ2
Ъ3"
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
®layers
Ы	variables
©metrics
Ьtrainable_variables
Эregularization_losses
™layer_metrics
 Ђlayer_regularization_losses
ђnon_trainable_variables
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≠layers
Я	variables
Ѓmetrics
†trainable_variables
°regularization_losses
ѓlayer_metrics
 ∞layer_regularization_losses
±non_trainable_variables
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≤layers
£	variables
≥metrics
§trainable_variables
•regularization_losses
іlayer_metrics
 µlayer_regularization_losses
ґnon_trainable_variables
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Јlayers
І	variables
Єmetrics
®trainable_variables
©regularization_losses
єlayer_metrics
 Їlayer_regularization_losses
їnon_trainable_variables
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Љlayers
Ђ	variables
љmetrics
ђtrainable_variables
≠regularization_losses
Њlayer_metrics
 њlayer_regularization_losses
јnon_trainable_variables
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
:	@А2dense/kernel
:А2
dense/bias
0
ѓ0
∞1"
trackable_list_wrapper
0
ѓ0
∞1"
trackable_list_wrapper
0
І0
®1"
trackable_list_wrapper
Є
Ѕlayers
±	variables
¬metrics
≤trainable_variables
≥regularization_losses
√layer_metrics
 ƒlayer_regularization_losses
≈non_trainable_variables
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
∆layers
µ	variables
«metrics
ґtrainable_variables
Јregularization_losses
»layer_metrics
 …layer_regularization_losses
 non_trainable_variables
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
": 
АА2dense_1/kernel
:А2dense_1/bias
0
є0
Ї1"
trackable_list_wrapper
0
є0
Ї1"
trackable_list_wrapper
0
©0
™1"
trackable_list_wrapper
Є
Ћlayers
ї	variables
ћmetrics
Љtrainable_variables
љregularization_losses
Ќlayer_metrics
 ќlayer_regularization_losses
ѕnon_trainable_variables
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
–layers
њ	variables
—metrics
јtrainable_variables
Ѕregularization_losses
“layer_metrics
 ”layer_regularization_losses
‘non_trainable_variables
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_2/kernel
:2dense_2/bias
0
√0
ƒ1"
trackable_list_wrapper
0
√0
ƒ1"
trackable_list_wrapper
0
Ђ0
ђ1"
trackable_list_wrapper
Є
’layers
≈	variables
÷metrics
∆trainable_variables
«regularization_losses
„layer_metrics
 Ўlayer_regularization_losses
ўnon_trainable_variables
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
z
20
31
A2
B3
^4
_5
m6
n7
К8
Л9
Щ10
Ъ11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Я0
†1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
°0
Ґ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
£0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
•0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
І0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
©0
™1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ђ0
ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
з2д
!__inference__wrapped_model_135517Њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *.Ґ+
)К&
input_1€€€€€€€€€22
ж2г
&__inference_model_layer_call_fn_138400
&__inference_model_layer_call_fn_140125
&__inference_model_layer_call_fn_138821
&__inference_model_layer_call_fn_140226ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
A__inference_model_layer_call_and_return_conditional_losses_139652
A__inference_model_layer_call_and_return_conditional_losses_140024
A__inference_model_layer_call_and_return_conditional_losses_137978
A__inference_model_layer_call_and_return_conditional_losses_137658ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
'__inference_conv2d_layer_call_fn_135554„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
°2Ю
B__inference_conv2d_layer_call_and_return_conditional_losses_135544„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
И2Е
)__inference_conv2d_1_layer_call_fn_135592„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
£2†
D__inference_conv2d_1_layer_call_and_return_conditional_losses_135582„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Т2П
4__inference_batch_normalization_layer_call_fn_140407
4__inference_batch_normalization_layer_call_fn_140420
4__inference_batch_normalization_layer_call_fn_140345
4__inference_batch_normalization_layer_call_fn_140332і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140319
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140376
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140301
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140394і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
И2Е
)__inference_conv2d_2_layer_call_fn_135755„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
£2†
D__inference_conv2d_2_layer_call_and_return_conditional_losses_135745„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ2Ч
6__inference_batch_normalization_1_layer_call_fn_140523
6__inference_batch_normalization_1_layer_call_fn_140585
6__inference_batch_normalization_1_layer_call_fn_140510
6__inference_batch_normalization_1_layer_call_fn_140598і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140572
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140554
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140479
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140497і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
$__inference_add_layer_call_fn_140610Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
й2ж
?__inference_add_layer_call_and_return_conditional_losses_140604Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_activation_layer_call_fn_140620Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_activation_layer_call_and_return_conditional_losses_140615Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
)__inference_conv2d_3_layer_call_fn_135919„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
£2†
D__inference_conv2d_3_layer_call_and_return_conditional_losses_135909„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
И2Е
)__inference_conv2d_4_layer_call_fn_135957„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
£2†
D__inference_conv2d_4_layer_call_and_return_conditional_losses_135947„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ2Ч
6__inference_batch_normalization_2_layer_call_fn_140726
6__inference_batch_normalization_2_layer_call_fn_140814
6__inference_batch_normalization_2_layer_call_fn_140801
6__inference_batch_normalization_2_layer_call_fn_140739і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140788
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140770
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140695
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140713і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
И2Е
)__inference_conv2d_5_layer_call_fn_136120„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
£2†
D__inference_conv2d_5_layer_call_and_return_conditional_losses_136110„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ2Ч
6__inference_batch_normalization_3_layer_call_fn_140979
6__inference_batch_normalization_3_layer_call_fn_140904
6__inference_batch_normalization_3_layer_call_fn_140917
6__inference_batch_normalization_3_layer_call_fn_140992і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140966
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140891
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140873
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140948і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_add_1_layer_call_fn_141004Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_add_1_layer_call_and_return_conditional_losses_140998Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_1_layer_call_fn_141014Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_1_layer_call_and_return_conditional_losses_141009Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
)__inference_conv2d_6_layer_call_fn_136284„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
£2†
D__inference_conv2d_6_layer_call_and_return_conditional_losses_136274„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
И2Е
)__inference_conv2d_7_layer_call_fn_136322„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
£2†
D__inference_conv2d_7_layer_call_and_return_conditional_losses_136312„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ2Ч
6__inference_batch_normalization_4_layer_call_fn_141195
6__inference_batch_normalization_4_layer_call_fn_141133
6__inference_batch_normalization_4_layer_call_fn_141208
6__inference_batch_normalization_4_layer_call_fn_141120і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141182
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141089
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141164
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141107і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
И2Е
)__inference_conv2d_8_layer_call_fn_136485„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
£2†
D__inference_conv2d_8_layer_call_and_return_conditional_losses_136475„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ2Ч
6__inference_batch_normalization_5_layer_call_fn_141298
6__inference_batch_normalization_5_layer_call_fn_141311
6__inference_batch_normalization_5_layer_call_fn_141386
6__inference_batch_normalization_5_layer_call_fn_141373і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141267
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141342
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141360
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141285і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_add_2_layer_call_fn_141398Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_add_2_layer_call_and_return_conditional_losses_141392Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_2_layer_call_fn_141408Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_2_layer_call_and_return_conditional_losses_141403Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
°2Ю
9__inference_global_average_pooling2d_layer_call_fn_136624а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Љ2є
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_136618а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_flatten_layer_call_fn_141419Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_141414Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense_layer_call_fn_141471Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_layer_call_and_return_conditional_losses_141462Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
О2Л
(__inference_dropout_layer_call_fn_141498
(__inference_dropout_layer_call_fn_141493і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_141483
C__inference_dropout_layer_call_and_return_conditional_losses_141488і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
(__inference_dense_1_layer_call_fn_141550Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_1_layer_call_and_return_conditional_losses_141541Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
*__inference_dropout_1_layer_call_fn_141572
*__inference_dropout_1_layer_call_fn_141577і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_1_layer_call_and_return_conditional_losses_141567
E__inference_dropout_1_layer_call_and_return_conditional_losses_141562і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
(__inference_dense_2_layer_call_fn_141629Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_2_layer_call_and_return_conditional_losses_141620Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≥2∞
__inference_loss_fn_0_141642П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_1_141655П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_2_141668П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_3_141681П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_4_141694П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_5_141707П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_6_141720П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_7_141733П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_8_141746П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_9_141759П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_10_141772П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_11_141785П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_12_141798П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_13_141811П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_14_141824П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_15_141837П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_16_141850П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_17_141863П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_18_141876П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_19_141889П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_20_141902П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_21_141915П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_22_141928П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_23_141941П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
3B1
$__inference_signature_wrapper_139188input_1„
!__inference__wrapped_model_135517±B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ8Ґ5
.Ґ+
)К&
input_1€€€€€€€€€22
™ "1™.
,
dense_2!К
dense_2€€€€€€€€€і
H__inference_activation_1_layer_call_and_return_conditional_losses_141009h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ М
-__inference_activation_1_layer_call_fn_141014[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22 
™ " К€€€€€€€€€22 і
H__inference_activation_2_layer_call_and_return_conditional_losses_141403h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22@
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ М
-__inference_activation_2_layer_call_fn_141408[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22@
™ " К€€€€€€€€€22@≤
F__inference_activation_layer_call_and_return_conditional_losses_140615h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ К
+__inference_activation_layer_call_fn_140620[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22
™ " К€€€€€€€€€22б
A__inference_add_1_layer_call_and_return_conditional_losses_140998ЫjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22 
*К'
inputs/1€€€€€€€€€22 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ є
&__inference_add_1_layer_call_fn_141004ОjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22 
*К'
inputs/1€€€€€€€€€22 
™ " К€€€€€€€€€22 б
A__inference_add_2_layer_call_and_return_conditional_losses_141392ЫjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22@
*К'
inputs/1€€€€€€€€€22@
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ є
&__inference_add_2_layer_call_fn_141398ОjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22@
*К'
inputs/1€€€€€€€€€22@
™ " К€€€€€€€€€22@я
?__inference_add_layer_call_and_return_conditional_losses_140604ЫjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22
*К'
inputs/1€€€€€€€€€22
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ Ј
$__inference_add_layer_call_fn_140610ОjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22
*К'
inputs/1€€€€€€€€€22
™ " К€€€€€€€€€22«
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140479r?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ «
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140497r?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ м
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140554Ц?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_140572Ц?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Я
6__inference_batch_normalization_1_layer_call_fn_140510e?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ " К€€€€€€€€€22Я
6__inference_batch_normalization_1_layer_call_fn_140523e?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ " К€€€€€€€€€22ƒ
6__inference_batch_normalization_1_layer_call_fn_140585Й?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
6__inference_batch_normalization_1_layer_call_fn_140598Й?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€«
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140695r\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ «
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140713r\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140770Ц\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_140788Ц\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ Я
6__inference_batch_normalization_2_layer_call_fn_140726e\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ " К€€€€€€€€€22 Я
6__inference_batch_normalization_2_layer_call_fn_140739e\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ " К€€€€€€€€€22 ƒ
6__inference_batch_normalization_2_layer_call_fn_140801Й\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ƒ
6__inference_batch_normalization_2_layer_call_fn_140814Й\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ «
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140873rklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ «
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140891rklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ м
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140948ЦklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ м
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_140966ЦklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ Я
6__inference_batch_normalization_3_layer_call_fn_140904eklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ " К€€€€€€€€€22 Я
6__inference_batch_normalization_3_layer_call_fn_140917eklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ " К€€€€€€€€€22 ƒ
6__inference_batch_normalization_3_layer_call_fn_140979ЙklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ƒ
6__inference_batch_normalization_3_layer_call_fn_140992ЙklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Ћ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141089vИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ Ћ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141107vИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ р
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141164ЪИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ р
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_141182ЪИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ £
6__inference_batch_normalization_4_layer_call_fn_141120iИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ " К€€€€€€€€€22@£
6__inference_batch_normalization_4_layer_call_fn_141133iИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ " К€€€€€€€€€22@»
6__inference_batch_normalization_4_layer_call_fn_141195НИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@»
6__inference_batch_normalization_4_layer_call_fn_141208НИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ћ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141267vЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ Ћ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141285vЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ р
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141342ЪЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ р
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_141360ЪЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ £
6__inference_batch_normalization_5_layer_call_fn_141298iЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ " К€€€€€€€€€22@£
6__inference_batch_normalization_5_layer_call_fn_141311iЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ " К€€€€€€€€€22@»
6__inference_batch_normalization_5_layer_call_fn_141373НЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@»
6__inference_batch_normalization_5_layer_call_fn_141386НЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140301Ц0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140319Ц0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140376r0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ ≈
O__inference_batch_normalization_layer_call_and_return_conditional_losses_140394r0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ ¬
4__inference_batch_normalization_layer_call_fn_140332Й0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€¬
4__inference_batch_normalization_layer_call_fn_140345Й0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Э
4__inference_batch_normalization_layer_call_fn_140407e0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ " К€€€€€€€€€22Э
4__inference_batch_normalization_layer_call_fn_140420e0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ " К€€€€€€€€€22ў
D__inference_conv2d_1_layer_call_and_return_conditional_losses_135582Р)*IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ±
)__inference_conv2d_1_layer_call_fn_135592Г)*IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
D__inference_conv2d_2_layer_call_and_return_conditional_losses_135745Р89IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ±
)__inference_conv2d_2_layer_call_fn_135755Г89IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
D__inference_conv2d_3_layer_call_and_return_conditional_losses_135909РOPIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ±
)__inference_conv2d_3_layer_call_fn_135919ГOPIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ў
D__inference_conv2d_4_layer_call_and_return_conditional_losses_135947РUVIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ±
)__inference_conv2d_4_layer_call_fn_135957ГUVIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ў
D__inference_conv2d_5_layer_call_and_return_conditional_losses_136110РdeIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ±
)__inference_conv2d_5_layer_call_fn_136120ГdeIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ў
D__inference_conv2d_6_layer_call_and_return_conditional_losses_136274Р{|IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ±
)__inference_conv2d_6_layer_call_fn_136284Г{|IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@џ
D__inference_conv2d_7_layer_call_and_return_conditional_losses_136312ТБВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≥
)__inference_conv2d_7_layer_call_fn_136322ЕБВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@џ
D__inference_conv2d_8_layer_call_and_return_conditional_losses_136475ТРСIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≥
)__inference_conv2d_8_layer_call_fn_136485ЕРСIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@„
B__inference_conv2d_layer_call_and_return_conditional_losses_135544Р#$IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
'__inference_conv2d_layer_call_fn_135554Г#$IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€І
C__inference_dense_1_layer_call_and_return_conditional_losses_141541`єЇ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
(__inference_dense_1_layer_call_fn_141550SєЇ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А¶
C__inference_dense_2_layer_call_and_return_conditional_losses_141620_√ƒ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
(__inference_dense_2_layer_call_fn_141629R√ƒ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€§
A__inference_dense_layer_call_and_return_conditional_losses_141462_ѓ∞/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
&__inference_dense_layer_call_fn_141471Rѓ∞/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€АІ
E__inference_dropout_1_layer_call_and_return_conditional_losses_141562^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ І
E__inference_dropout_1_layer_call_and_return_conditional_losses_141567^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dropout_1_layer_call_fn_141572Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€А
*__inference_dropout_1_layer_call_fn_141577Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А•
C__inference_dropout_layer_call_and_return_conditional_losses_141483^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ •
C__inference_dropout_layer_call_and_return_conditional_losses_141488^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
(__inference_dropout_layer_call_fn_141493Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€А}
(__inference_dropout_layer_call_fn_141498Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€АЯ
C__inference_flatten_layer_call_and_return_conditional_losses_141414X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ w
(__inference_flatten_layer_call_fn_141419K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@Ё
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_136618ДRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ і
9__inference_global_average_pooling2d_layer_call_fn_136624wRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€;
__inference_loss_fn_0_141642#Ґ

Ґ 
™ "К <
__inference_loss_fn_10_141772dҐ

Ґ 
™ "К <
__inference_loss_fn_11_141785eҐ

Ґ 
™ "К <
__inference_loss_fn_12_141798{Ґ

Ґ 
™ "К <
__inference_loss_fn_13_141811|Ґ

Ґ 
™ "К =
__inference_loss_fn_14_141824БҐ

Ґ 
™ "К =
__inference_loss_fn_15_141837ВҐ

Ґ 
™ "К =
__inference_loss_fn_16_141850РҐ

Ґ 
™ "К =
__inference_loss_fn_17_141863СҐ

Ґ 
™ "К =
__inference_loss_fn_18_141876ѓҐ

Ґ 
™ "К =
__inference_loss_fn_19_141889∞Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_141655$Ґ

Ґ 
™ "К =
__inference_loss_fn_20_141902єҐ

Ґ 
™ "К =
__inference_loss_fn_21_141915ЇҐ

Ґ 
™ "К =
__inference_loss_fn_22_141928√Ґ

Ґ 
™ "К =
__inference_loss_fn_23_141941ƒҐ

Ґ 
™ "К ;
__inference_loss_fn_2_141668)Ґ

Ґ 
™ "К ;
__inference_loss_fn_3_141681*Ґ

Ґ 
™ "К ;
__inference_loss_fn_4_1416948Ґ

Ґ 
™ "К ;
__inference_loss_fn_5_1417079Ґ

Ґ 
™ "К ;
__inference_loss_fn_6_141720OҐ

Ґ 
™ "К ;
__inference_loss_fn_7_141733PҐ

Ґ 
™ "К ;
__inference_loss_fn_8_141746UҐ

Ґ 
™ "К ;
__inference_loss_fn_9_141759VҐ

Ґ 
™ "К у
A__inference_model_layer_call_and_return_conditional_losses_137658≠B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€22
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ у
A__inference_model_layer_call_and_return_conditional_losses_137978≠B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€22
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ т
A__inference_model_layer_call_and_return_conditional_losses_139652ђB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ т
A__inference_model_layer_call_and_return_conditional_losses_140024ђB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ћ
&__inference_model_layer_call_fn_138400†B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€22
p

 
™ "К€€€€€€€€€Ћ
&__inference_model_layer_call_fn_138821†B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€22
p 

 
™ "К€€€€€€€€€ 
&__inference_model_layer_call_fn_140125ЯB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p

 
™ "К€€€€€€€€€ 
&__inference_model_layer_call_fn_140226ЯB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p 

 
™ "К€€€€€€€€€е
$__inference_signature_wrapper_139188ЉB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒCҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€22"1™.
,
dense_2!К
dense_2€€€€€€€€€