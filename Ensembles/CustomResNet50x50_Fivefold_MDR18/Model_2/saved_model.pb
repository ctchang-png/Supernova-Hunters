јџ3
Њ§
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
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8бс,

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
Ђ
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
: *
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
Ђ
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:@*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
Є
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:@*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:@*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:@*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:@*
dtype0
Є
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:@*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	@*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Цw
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*w
valueїvBєv Bэv
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
layer_with_weights-16
layer-25
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api

,axis
	-gamma
.beta
/moving_mean
0moving_variance
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api

;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
R
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api

Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api

gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
R
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
R
tregularization_losses
utrainable_variables
v	variables
w	keras_api
h

xkernel
ybias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
l

~kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
 regularization_losses
Ёtrainable_variables
Ђ	variables
Ѓ	keras_api
V
Єregularization_losses
Ѕtrainable_variables
І	variables
Ї	keras_api
V
Јregularization_losses
Љtrainable_variables
Њ	variables
Ћ	keras_api
n
Ќkernel
	­bias
Ўregularization_losses
Џtrainable_variables
А	variables
Б	keras_api
n
Вkernel
	Гbias
Дregularization_losses
Еtrainable_variables
Ж	variables
З	keras_api
 

 0
!1
&2
'3
-4
.5
56
67
<8
=9
L10
M11
R12
S13
Y14
Z15
a16
b17
h18
i19
x20
y21
~22
23
24
25
26
27
28
29
Ќ30
­31
В32
Г33
є
 0
!1
&2
'3
-4
.5
/6
07
58
69
<10
=11
>12
?13
L14
M15
R16
S17
Y18
Z19
[20
\21
a22
b23
h24
i25
j26
k27
x28
y29
~30
31
32
33
34
35
36
37
38
39
40
41
Ќ42
­43
В44
Г45
В
Иlayer_metrics
 Йlayer_regularization_losses
regularization_losses
trainable_variables
Кnon_trainable_variables
	variables
Лmetrics
Мlayers
 
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
В
Нlayer_metrics
 Оlayer_regularization_losses
"regularization_losses
#trainable_variables
Пnon_trainable_variables
$	variables
Рmetrics
Сlayers
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
В
Тlayer_metrics
 Уlayer_regularization_losses
(regularization_losses
)trainable_variables
Фnon_trainable_variables
*	variables
Хmetrics
Цlayers
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
/2
03
В
Чlayer_metrics
 Шlayer_regularization_losses
1regularization_losses
2trainable_variables
Щnon_trainable_variables
3	variables
Ъmetrics
Ыlayers
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
В
Ьlayer_metrics
 Эlayer_regularization_losses
7regularization_losses
8trainable_variables
Юnon_trainable_variables
9	variables
Яmetrics
аlayers
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
>2
?3
В
бlayer_metrics
 вlayer_regularization_losses
@regularization_losses
Atrainable_variables
гnon_trainable_variables
B	variables
дmetrics
еlayers
 
 
 
В
жlayer_metrics
 зlayer_regularization_losses
Dregularization_losses
Etrainable_variables
иnon_trainable_variables
F	variables
йmetrics
кlayers
 
 
 
В
лlayer_metrics
 мlayer_regularization_losses
Hregularization_losses
Itrainable_variables
нnon_trainable_variables
J	variables
оmetrics
пlayers
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
В
рlayer_metrics
 сlayer_regularization_losses
Nregularization_losses
Otrainable_variables
тnon_trainable_variables
P	variables
уmetrics
фlayers
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
В
хlayer_metrics
 цlayer_regularization_losses
Tregularization_losses
Utrainable_variables
чnon_trainable_variables
V	variables
шmetrics
щlayers
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

Y0
Z1

Y0
Z1
[2
\3
В
ъlayer_metrics
 ыlayer_regularization_losses
]regularization_losses
^trainable_variables
ьnon_trainable_variables
_	variables
эmetrics
юlayers
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

a0
b1
В
яlayer_metrics
 №layer_regularization_losses
cregularization_losses
dtrainable_variables
ёnon_trainable_variables
e	variables
ђmetrics
ѓlayers
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

h0
i1

h0
i1
j2
k3
В
єlayer_metrics
 ѕlayer_regularization_losses
lregularization_losses
mtrainable_variables
іnon_trainable_variables
n	variables
їmetrics
јlayers
 
 
 
В
љlayer_metrics
 њlayer_regularization_losses
pregularization_losses
qtrainable_variables
ћnon_trainable_variables
r	variables
ќmetrics
§layers
 
 
 
В
ўlayer_metrics
 џlayer_regularization_losses
tregularization_losses
utrainable_variables
non_trainable_variables
v	variables
metrics
layers
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

x0
y1

x0
y1
В
layer_metrics
 layer_regularization_losses
zregularization_losses
{trainable_variables
non_trainable_variables
|	variables
metrics
layers
][
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

~0
1
Е
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
 
hf
VARIABLE_VALUEbatch_normalization_10/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_10/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_10/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_10/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
0
1
2
3
Е
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
][
VARIABLE_VALUEconv2d_17/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_17/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Е
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
 
hf
VARIABLE_VALUEbatch_normalization_11/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_11/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_11/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_11/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
0
1
2
3
Е
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
 
 
 
Е
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
 layers
 
 
 
Е
Ёlayer_metrics
 Ђlayer_regularization_losses
 regularization_losses
Ёtrainable_variables
Ѓnon_trainable_variables
Ђ	variables
Єmetrics
Ѕlayers
 
 
 
Е
Іlayer_metrics
 Їlayer_regularization_losses
Єregularization_losses
Ѕtrainable_variables
Јnon_trainable_variables
І	variables
Љmetrics
Њlayers
 
 
 
Е
Ћlayer_metrics
 Ќlayer_regularization_losses
Јregularization_losses
Љtrainable_variables
­non_trainable_variables
Њ	variables
Ўmetrics
Џlayers
[Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_2/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ќ0
­1

Ќ0
­1
Е
Аlayer_metrics
 Бlayer_regularization_losses
Ўregularization_losses
Џtrainable_variables
Вnon_trainable_variables
А	variables
Гmetrics
Дlayers
[Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_3/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

В0
Г1

В0
Г1
Е
Еlayer_metrics
 Жlayer_regularization_losses
Дregularization_losses
Еtrainable_variables
Зnon_trainable_variables
Ж	variables
Иmetrics
Йlayers
 
 
Z
/0
01
>2
?3
[4
\5
j6
k7
8
9
10
11
 
Ц
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
/0
01
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
>0
?1
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
[0
\1
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
j0
k1
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
0
1
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
0
1
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

serving_default_input_2Placeholder*/
_output_shapes
:џџџџџџџџџ22*
dtype0*$
shape:џџџџџџџџџ22
є
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*:
Tin3
12/*
Tout
2*'
_output_shapes
:џџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_54268
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ч
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*;
Tin4
220*
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
GPU2*0J 8*'
f"R 
__inference__traced_save_56982
Њ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*:
Tin3
12/*
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
GPU2*0J 8**
f%R#
!__inference__traced_restore_57132Ом*
ї
o
__inference_loss_fn_4_56596?
;conv2d_11_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_11_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addg
IdentityIdentity$conv2d_11/kernel/Regularizer/add:z:0*
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
ю
Д
'__inference_model_1_layer_call_fn_53527
input_2
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

unknown_44
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*'
_output_shapes
:џџџџџџџџџ*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_534322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:
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
: 
н
n
__inference_loss_fn_0_56544>
:conv2d_9_kernel_regularizer_square_readvariableop_resource
identityщ
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_9_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addf
IdentityIdentity#conv2d_9/kernel/Regularizer/add:z:0*
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
ј
p
__inference_loss_fn_16_56752?
;conv2d_17_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_17_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addg
IdentityIdentity$conv2d_17/kernel/Regularizer/add:z:0*
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
Цј
ъ
B__inference_model_1_layer_call_and_return_conditional_losses_54693

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityЂ:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpА
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOpО
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_9/Conv2DЇ
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOpЌ
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_9/BiasAddГ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOpд
conv2d_10/Conv2DConv2Dconv2d_9/BiasAdd:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_10/Conv2DЊ
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOpА
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/ReluЖ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOpМ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2(
&batch_normalization_6/FusedBatchNormV3
batch_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_6/Constђ
+batch_normalization_6/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_6/AssignMovingAvg/sub/x­
)batch_normalization_6/AssignMovingAvg/subSub4batch_normalization_6/AssignMovingAvg/sub/x:output:0$batch_normalization_6/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/subч
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_6/AssignMovingAvg/sub_1Sub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_6/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg/sub_1Е
)batch_normalization_6/AssignMovingAvg/mulMul/batch_normalization_6/AssignMovingAvg/sub_1:z:0-batch_normalization_6/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/mulс
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_6/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_6/AssignMovingAvg_1/sub/xЕ
+batch_normalization_6/AssignMovingAvg_1/subSub6batch_normalization_6/AssignMovingAvg_1/sub/x:output:0$batch_normalization_6/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subэ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_6/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2/
-batch_normalization_6/AssignMovingAvg_1/sub_1П
+batch_normalization_6/AssignMovingAvg_1/mulMul1batch_normalization_6/AssignMovingAvg_1/sub_1:z:0/batch_normalization_6/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/mulя
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOpх
conv2d_11/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_11/Conv2DЊ
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOpА
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_11/BiasAddЖ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOpМ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Я
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2(
&batch_normalization_7/FusedBatchNormV3
batch_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_7/Constђ
+batch_normalization_7/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_7/AssignMovingAvg/sub/x­
)batch_normalization_7/AssignMovingAvg/subSub4batch_normalization_7/AssignMovingAvg/sub/x:output:0$batch_normalization_7/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/subч
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_7/AssignMovingAvg/sub_1Sub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_7/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg/sub_1Е
)batch_normalization_7/AssignMovingAvg/mulMul/batch_normalization_7/AssignMovingAvg/sub_1:z:0-batch_normalization_7/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mulс
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_7/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_7/AssignMovingAvg_1/sub/xЕ
+batch_normalization_7/AssignMovingAvg_1/subSub6batch_normalization_7/AssignMovingAvg_1/sub/x:output:0$batch_normalization_7/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subэ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_7/AssignMovingAvg_1/sub_1Sub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_7/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2/
-batch_normalization_7/AssignMovingAvg_1/sub_1П
+batch_normalization_7/AssignMovingAvg_1/mulMul1batch_normalization_7/AssignMovingAvg_1/sub_1:z:0/batch_normalization_7/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mulя
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp 
	add_3/addAddV2*batch_normalization_7/FusedBatchNormV3:y:0conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
	add_3/addw
activation_3/ReluReluadd_3/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
activation_3/ReluГ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpк
conv2d_12/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_12/Conv2DЊ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOpА
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/ReluГ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOpз
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_13/Conv2DЊ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOpА
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/ReluЖ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOpМ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1щ
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2(
&batch_normalization_8/FusedBatchNormV3
batch_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_8/Constђ
+batch_normalization_8/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_8/AssignMovingAvg/sub/x­
)batch_normalization_8/AssignMovingAvg/subSub4batch_normalization_8/AssignMovingAvg/sub/x:output:0$batch_normalization_8/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_8/AssignMovingAvg/subч
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_8/AssignMovingAvg/sub_1Sub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_8/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg/sub_1Е
)batch_normalization_8/AssignMovingAvg/mulMul/batch_normalization_8/AssignMovingAvg/sub_1:z:0-batch_normalization_8/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_8/AssignMovingAvg/mulс
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_8/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_8/AssignMovingAvg_1/sub/xЕ
+batch_normalization_8/AssignMovingAvg_1/subSub6batch_normalization_8/AssignMovingAvg_1/sub/x:output:0$batch_normalization_8/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg_1/subэ
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_8/AssignMovingAvg_1/sub_1Sub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_8/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_8/AssignMovingAvg_1/sub_1П
+batch_normalization_8/AssignMovingAvg_1/mulMul1batch_normalization_8/AssignMovingAvg_1/sub_1:z:0/batch_normalization_8/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg_1/mulя
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_14/Conv2D/ReadVariableOpх
conv2d_14/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_14/Conv2DЊ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOpА
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_14/BiasAddЖ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOpМ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Я
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_14/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2(
&batch_normalization_9/FusedBatchNormV3
batch_normalization_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_9/Constђ
+batch_normalization_9/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_9/AssignMovingAvg/sub/x­
)batch_normalization_9/AssignMovingAvg/subSub4batch_normalization_9/AssignMovingAvg/sub/x:output:0$batch_normalization_9/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/subч
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_9/AssignMovingAvg/sub_1Sub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_9/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg/sub_1Е
)batch_normalization_9/AssignMovingAvg/mulMul/batch_normalization_9/AssignMovingAvg/sub_1:z:0-batch_normalization_9/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/mulс
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp6^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_9/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_9/AssignMovingAvg_1/sub/xЕ
+batch_normalization_9/AssignMovingAvg_1/subSub6batch_normalization_9/AssignMovingAvg_1/sub/x:output:0$batch_normalization_9/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/subэ
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_9/AssignMovingAvg_1/sub_1Sub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_9/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_9/AssignMovingAvg_1/sub_1П
+batch_normalization_9/AssignMovingAvg_1/mulMul1batch_normalization_9/AssignMovingAvg_1/sub_1:z:0/batch_normalization_9/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/mulя
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpЃ
	add_4/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0conv2d_12/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
	add_4/addw
activation_4/ReluReluadd_4/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
activation_4/ReluГ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOpк
conv2d_15/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_15/Conv2DЊ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpА
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/ReluГ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_16/Conv2D/ReadVariableOpз
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_16/Conv2DЊ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOpА
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/ReluЙ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOpП
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1ь
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1з
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2)
'batch_normalization_10/FusedBatchNormV3
batch_normalization_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_10/Constѕ
,batch_normalization_10/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_10/AssignMovingAvg/sub/xВ
*batch_normalization_10/AssignMovingAvg/subSub5batch_normalization_10/AssignMovingAvg/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/subъ
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOpб
,batch_normalization_10/AssignMovingAvg/sub_1Sub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_10/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_10/AssignMovingAvg/sub_1К
*batch_normalization_10/AssignMovingAvg/mulMul0batch_normalization_10/AssignMovingAvg/sub_1:z:0.batch_normalization_10/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_10/AssignMovingAvg/mulш
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_10/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_10/AssignMovingAvg_1/sub/xК
,batch_normalization_10/AssignMovingAvg_1/subSub7batch_normalization_10/AssignMovingAvg_1/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/sub№
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_10/AssignMovingAvg_1/sub_1Sub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_10/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_10/AssignMovingAvg_1/sub_1Ф
,batch_normalization_10/AssignMovingAvg_1/mulMul2batch_normalization_10/AssignMovingAvg_1/sub_1:z:00batch_normalization_10/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_10/AssignMovingAvg_1/mulі
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_17/Conv2D/ReadVariableOpц
conv2d_17/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_17/Conv2DЊ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOpА
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_17/BiasAddЙ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOpП
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ь
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1е
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2)
'batch_normalization_11/FusedBatchNormV3
batch_normalization_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_11/Constѕ
,batch_normalization_11/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_11/AssignMovingAvg/sub/xВ
*batch_normalization_11/AssignMovingAvg/subSub5batch_normalization_11/AssignMovingAvg/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/subъ
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOpб
,batch_normalization_11/AssignMovingAvg/sub_1Sub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_11/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_11/AssignMovingAvg/sub_1К
*batch_normalization_11/AssignMovingAvg/mulMul0batch_normalization_11/AssignMovingAvg/sub_1:z:0.batch_normalization_11/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_11/AssignMovingAvg/mulш
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp7^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_11/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_11/AssignMovingAvg_1/sub/xК
,batch_normalization_11/AssignMovingAvg_1/subSub7batch_normalization_11/AssignMovingAvg_1/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/sub№
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_11/AssignMovingAvg_1/sub_1Sub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_11/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_11/AssignMovingAvg_1/sub_1Ф
,batch_normalization_11/AssignMovingAvg_1/mulMul2batch_normalization_11/AssignMovingAvg_1/sub_1:z:00batch_normalization_11/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_11/AssignMovingAvg_1/mulі
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpЄ
	add_5/addAddV2+batch_normalization_11/FusedBatchNormV3:y:0conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
	add_5/addw
activation_5/ReluReluadd_5/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
activation_5/ReluЗ
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesй
global_average_pooling2d_1/MeanMeanactivation_5/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling2d_1/Means
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
flatten_1/ConstЇ
flatten_1/ReshapeReshape(global_average_pooling2d_1/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten_1/ReshapeІ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЅ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЂ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_2/ReluІ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/Sigmoidж
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЧ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addй
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addЪ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addй
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addЪ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addй
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addЪ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addй
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addЪ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addй
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addЪ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addй
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addЪ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addй
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addЪ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addй
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addЪ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addЬ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addХ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addЬ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addФ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЧ
IdentityIdentitydense_3/Sigmoid:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
: 
ч$
и
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_51702

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
ц$
з
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_51337

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
Ю
j
@__inference_add_4_layer_call_and_return_conditional_losses_52324

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22 :џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs
Ў
n
__inference_loss_fn_18_56778=
9dense_2_kernel_regularizer_square_readvariableop_resource
identityп
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
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
 
Q
%__inference_add_4_layer_call_fn_56012
inputs_0
inputs_1
identityД
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_523242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22 :џџџџџџџџџ22 :Y U
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/1
с
~
)__inference_conv2d_11_layer_call_fn_51051

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_510412
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ГЅ

B__inference_model_1_layer_call_and_return_conditional_losses_55040

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityА
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOpО
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_9/Conv2DЇ
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOpЌ
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_9/BiasAddГ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOpд
conv2d_10/Conv2DConv2Dconv2d_9/BiasAdd:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_10/Conv2DЊ
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOpА
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/ReluЖ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOpМ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3Г
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOpх
conv2d_11/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_11/Conv2DЊ
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOpА
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_11/BiasAddЖ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOpМ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3 
	add_3/addAddV2*batch_normalization_7/FusedBatchNormV3:y:0conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
	add_3/addw
activation_3/ReluReluadd_3/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
activation_3/ReluГ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpк
conv2d_12/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_12/Conv2DЊ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOpА
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/ReluГ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOpз
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_13/Conv2DЊ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOpА
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/ReluЖ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOpМ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1щ
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3Г
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_14/Conv2D/ReadVariableOpх
conv2d_14/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_14/Conv2DЊ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOpА
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_14/BiasAddЖ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOpМ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_14/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3Ѓ
	add_4/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0conv2d_12/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
	add_4/addw
activation_4/ReluReluadd_4/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
activation_4/ReluГ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOpк
conv2d_15/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_15/Conv2DЊ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpА
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/ReluГ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_16/Conv2D/ReadVariableOpз
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_16/Conv2DЊ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOpА
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/ReluЙ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOpП
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1ь
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ъ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3Г
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_17/Conv2D/ReadVariableOpц
conv2d_17/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_17/Conv2DЊ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOpА
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_17/BiasAddЙ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOpП
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ь
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3Є
	add_5/addAddV2+batch_normalization_11/FusedBatchNormV3:y:0conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
	add_5/addw
activation_5/ReluReluadd_5/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
activation_5/ReluЗ
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesй
global_average_pooling2d_1/MeanMeanactivation_5/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling2d_1/Means
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
flatten_1/ConstЇ
flatten_1/ReshapeReshape(global_average_pooling2d_1/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten_1/ReshapeІ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЅ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЂ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_2/ReluІ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/Sigmoidж
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЧ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addй
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addЪ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addй
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addЪ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addй
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addЪ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addй
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addЪ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addй
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addЪ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addй
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addЪ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addй
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addЪ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addй
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addЪ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addЬ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addХ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addЬ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addФ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addg
IdentityIdentitydense_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22:::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:џџџџџџџџџ22
 
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
: 
Ќ
Ј
5__inference_batch_normalization_7_layer_call_fn_55593

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_520532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
ц$
з
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_51135

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
Ч

P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55580

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
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
ї
|
'__inference_dense_3_layer_call_fn_56531

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_526422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

H
,__inference_activation_3_layer_call_fn_55628

inputs
identityЎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_521272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs

H
,__inference_activation_5_layer_call_fn_56416

inputs
identityЎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_525492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs
ј
p
__inference_loss_fn_10_56674?
;conv2d_14_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_14_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addg
IdentityIdentity$conv2d_14/kernel/Regularizer/add:z:0*
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
э
V
:__inference_global_average_pooling2d_1_layer_call_fn_51920

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_519142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў
Ј
5__inference_batch_normalization_9_layer_call_fn_56000

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_522822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
Ю
j
@__inference_add_3_layer_call_and_return_conditional_losses_52113

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
е
c
G__inference_activation_3_layer_call_and_return_conditional_losses_55623

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
Р№
ю
B__inference_model_1_layer_call_and_return_conditional_losses_53432

inputs
conv2d_9_53138
conv2d_9_53140
conv2d_10_53143
conv2d_10_53145
batch_normalization_6_53148
batch_normalization_6_53150
batch_normalization_6_53152
batch_normalization_6_53154
conv2d_11_53157
conv2d_11_53159
batch_normalization_7_53162
batch_normalization_7_53164
batch_normalization_7_53166
batch_normalization_7_53168
conv2d_12_53173
conv2d_12_53175
conv2d_13_53178
conv2d_13_53180
batch_normalization_8_53183
batch_normalization_8_53185
batch_normalization_8_53187
batch_normalization_8_53189
conv2d_14_53192
conv2d_14_53194
batch_normalization_9_53197
batch_normalization_9_53199
batch_normalization_9_53201
batch_normalization_9_53203
conv2d_15_53208
conv2d_15_53210
conv2d_16_53213
conv2d_16_53215 
batch_normalization_10_53218 
batch_normalization_10_53220 
batch_normalization_10_53222 
batch_normalization_10_53224
conv2d_17_53227
conv2d_17_53229 
batch_normalization_11_53232 
batch_normalization_11_53234 
batch_normalization_11_53236 
batch_normalization_11_53238
dense_2_53245
dense_2_53247
dense_3_53250
dense_3_53252
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallњ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_53138conv2d_9_53140*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_508402"
 conv2d_9/StatefulPartitionedCallЂ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_53143conv2d_10_53145*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_508782#
!conv2d_10/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_53148batch_normalization_6_53150batch_normalization_6_53152batch_normalization_6_53154*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_519642/
-batch_normalization_6/StatefulPartitionedCallЏ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_53157conv2d_11_53159*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_510412#
!conv2d_11/StatefulPartitionedCall
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_53162batch_normalization_7_53164batch_normalization_7_53166batch_normalization_7_53168*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_520532/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_521132
add_3/PartitionedCallр
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_521272
activation_3/PartitionedCall
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_53173conv2d_12_53175*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_512052#
!conv2d_12/StatefulPartitionedCallЃ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_53178conv2d_13_53180*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_512432#
!conv2d_13/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_53183batch_normalization_8_53185batch_normalization_8_53187batch_normalization_8_53189*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_521752/
-batch_normalization_8/StatefulPartitionedCallЏ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_53192conv2d_14_53194*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_514062#
!conv2d_14/StatefulPartitionedCall
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_53197batch_normalization_9_53199batch_normalization_9_53201batch_normalization_9_53203*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_522642/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_523242
add_4/PartitionedCallр
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_523382
activation_4/PartitionedCall
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_53208conv2d_15_53210*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_515702#
!conv2d_15/StatefulPartitionedCallЃ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_53213conv2d_16_53215*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_516082#
!conv2d_16/StatefulPartitionedCallЂ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_53218batch_normalization_10_53220batch_normalization_10_53222batch_normalization_10_53224*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5238620
.batch_normalization_10/StatefulPartitionedCallА
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_53227conv2d_17_53229*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_517712#
!conv2d_17/StatefulPartitionedCallЂ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_53232batch_normalization_11_53234batch_normalization_11_53236batch_normalization_11_53238*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5247520
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_525352
add_5/PartitionedCallр
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_525492
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_519142,
*global_average_pooling2d_1/PartitionedCallф
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_525642
flatten_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_53245dense_2_53247*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_525992!
dense_2/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_53250dense_3_53252*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_526422!
dense_3/StatefulPartitionedCallН
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_53138*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/add­
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_53140*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addР
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_53143*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addА
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_53145*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addР
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_53157*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addА
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_53159*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addР
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_53173*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addА
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_53175*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addР
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_53178*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addА
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_53180*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addР
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_53192*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addА
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_53194*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addР
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_53208*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addА
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_53210*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addР
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_53213*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addА
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_53215*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addР
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_53227*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addА
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_53229*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addГ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_53245*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЋ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_53247*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addГ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_53250*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЊ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_53252*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЅ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
: 
$
и
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56097

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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


Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56368

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
нЪ
й
 __inference__wrapped_model_50813
input_23
/model_1_conv2d_9_conv2d_readvariableop_resource4
0model_1_conv2d_9_biasadd_readvariableop_resource4
0model_1_conv2d_10_conv2d_readvariableop_resource5
1model_1_conv2d_10_biasadd_readvariableop_resource9
5model_1_batch_normalization_6_readvariableop_resource;
7model_1_batch_normalization_6_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_11_conv2d_readvariableop_resource5
1model_1_conv2d_11_biasadd_readvariableop_resource9
5model_1_batch_normalization_7_readvariableop_resource;
7model_1_batch_normalization_7_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_12_conv2d_readvariableop_resource5
1model_1_conv2d_12_biasadd_readvariableop_resource4
0model_1_conv2d_13_conv2d_readvariableop_resource5
1model_1_conv2d_13_biasadd_readvariableop_resource9
5model_1_batch_normalization_8_readvariableop_resource;
7model_1_batch_normalization_8_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_14_conv2d_readvariableop_resource5
1model_1_conv2d_14_biasadd_readvariableop_resource9
5model_1_batch_normalization_9_readvariableop_resource;
7model_1_batch_normalization_9_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_15_conv2d_readvariableop_resource5
1model_1_conv2d_15_biasadd_readvariableop_resource4
0model_1_conv2d_16_conv2d_readvariableop_resource5
1model_1_conv2d_16_biasadd_readvariableop_resource:
6model_1_batch_normalization_10_readvariableop_resource<
8model_1_batch_normalization_10_readvariableop_1_resourceK
Gmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_17_conv2d_readvariableop_resource5
1model_1_conv2d_17_biasadd_readvariableop_resource:
6model_1_batch_normalization_11_readvariableop_resource<
8model_1_batch_normalization_11_readvariableop_1_resourceK
Gmodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource2
.model_1_dense_2_matmul_readvariableop_resource3
/model_1_dense_2_biasadd_readvariableop_resource2
.model_1_dense_3_matmul_readvariableop_resource3
/model_1_dense_3_biasadd_readvariableop_resource
identityШ
&model_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/conv2d_9/Conv2D/ReadVariableOpз
model_1/conv2d_9/Conv2DConv2Dinput_2.model_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_1/conv2d_9/Conv2DП
'model_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_9/BiasAdd/ReadVariableOpЬ
model_1/conv2d_9/BiasAddBiasAdd model_1/conv2d_9/Conv2D:output:0/model_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_9/BiasAddЫ
'model_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_10/Conv2D/ReadVariableOpє
model_1/conv2d_10/Conv2DConv2D!model_1/conv2d_9/BiasAdd:output:0/model_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_1/conv2d_10/Conv2DТ
(model_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_10/BiasAdd/ReadVariableOpа
model_1/conv2d_10/BiasAddBiasAdd!model_1/conv2d_10/Conv2D:output:00model_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_10/BiasAdd
model_1/conv2d_10/ReluRelu"model_1/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_10/ReluЮ
,model_1/batch_normalization_6/ReadVariableOpReadVariableOp5model_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_1/batch_normalization_6/ReadVariableOpд
.model_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype020
.model_1/batch_normalization_6/ReadVariableOp_1
=model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$model_1/conv2d_10/Relu:activations:04model_1/batch_normalization_6/ReadVariableOp:value:06model_1/batch_normalization_6/ReadVariableOp_1:value:0Emodel_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_6/FusedBatchNormV3Ы
'model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_11/Conv2D/ReadVariableOp
model_1/conv2d_11/Conv2DConv2D2model_1/batch_normalization_6/FusedBatchNormV3:y:0/model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_1/conv2d_11/Conv2DТ
(model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_11/BiasAdd/ReadVariableOpа
model_1/conv2d_11/BiasAddBiasAdd!model_1/conv2d_11/Conv2D:output:00model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_11/BiasAddЮ
,model_1/batch_normalization_7/ReadVariableOpReadVariableOp5model_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_1/batch_normalization_7/ReadVariableOpд
.model_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype020
.model_1/batch_normalization_7/ReadVariableOp_1
=model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_11/BiasAdd:output:04model_1/batch_normalization_7/ReadVariableOp:value:06model_1/batch_normalization_7/ReadVariableOp_1:value:0Emodel_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_7/FusedBatchNormV3Р
model_1/add_3/addAddV22model_1/batch_normalization_7/FusedBatchNormV3:y:0!model_1/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/add_3/add
model_1/activation_3/ReluRelumodel_1/add_3/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/activation_3/ReluЫ
'model_1/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_1/conv2d_12/Conv2D/ReadVariableOpњ
model_1/conv2d_12/Conv2DConv2D'model_1/activation_3/Relu:activations:0/model_1/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_1/conv2d_12/Conv2DТ
(model_1/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_12/BiasAdd/ReadVariableOpа
model_1/conv2d_12/BiasAddBiasAdd!model_1/conv2d_12/Conv2D:output:00model_1/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_12/BiasAdd
model_1/conv2d_12/ReluRelu"model_1/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_12/ReluЫ
'model_1/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_1/conv2d_13/Conv2D/ReadVariableOpї
model_1/conv2d_13/Conv2DConv2D$model_1/conv2d_12/Relu:activations:0/model_1/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_1/conv2d_13/Conv2DТ
(model_1/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_13/BiasAdd/ReadVariableOpа
model_1/conv2d_13/BiasAddBiasAdd!model_1/conv2d_13/Conv2D:output:00model_1/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_13/BiasAdd
model_1/conv2d_13/ReluRelu"model_1/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_13/ReluЮ
,model_1/batch_normalization_8/ReadVariableOpReadVariableOp5model_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_8/ReadVariableOpд
.model_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_8/ReadVariableOp_1
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$model_1/conv2d_13/Relu:activations:04model_1/batch_normalization_8/ReadVariableOp:value:06model_1/batch_normalization_8/ReadVariableOp_1:value:0Emodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_8/FusedBatchNormV3Ы
'model_1/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_1/conv2d_14/Conv2D/ReadVariableOp
model_1/conv2d_14/Conv2DConv2D2model_1/batch_normalization_8/FusedBatchNormV3:y:0/model_1/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_1/conv2d_14/Conv2DТ
(model_1/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_14/BiasAdd/ReadVariableOpа
model_1/conv2d_14/BiasAddBiasAdd!model_1/conv2d_14/Conv2D:output:00model_1/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_14/BiasAddЮ
,model_1/batch_normalization_9/ReadVariableOpReadVariableOp5model_1_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_9/ReadVariableOpд
.model_1/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_9/ReadVariableOp_1
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_14/BiasAdd:output:04model_1/batch_normalization_9/ReadVariableOp:value:06model_1/batch_normalization_9/ReadVariableOp_1:value:0Emodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_9/FusedBatchNormV3У
model_1/add_4/addAddV22model_1/batch_normalization_9/FusedBatchNormV3:y:0$model_1/conv2d_12/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/add_4/add
model_1/activation_4/ReluRelumodel_1/add_4/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/activation_4/ReluЫ
'model_1/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_1/conv2d_15/Conv2D/ReadVariableOpњ
model_1/conv2d_15/Conv2DConv2D'model_1/activation_4/Relu:activations:0/model_1/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_1/conv2d_15/Conv2DТ
(model_1/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_15/BiasAdd/ReadVariableOpа
model_1/conv2d_15/BiasAddBiasAdd!model_1/conv2d_15/Conv2D:output:00model_1/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_15/BiasAdd
model_1/conv2d_15/ReluRelu"model_1/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_15/ReluЫ
'model_1/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_1/conv2d_16/Conv2D/ReadVariableOpї
model_1/conv2d_16/Conv2DConv2D$model_1/conv2d_15/Relu:activations:0/model_1/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_1/conv2d_16/Conv2DТ
(model_1/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_16/BiasAdd/ReadVariableOpа
model_1/conv2d_16/BiasAddBiasAdd!model_1/conv2d_16/Conv2D:output:00model_1/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_16/BiasAdd
model_1/conv2d_16/ReluRelu"model_1/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_16/Reluб
-model_1/batch_normalization_10/ReadVariableOpReadVariableOp6model_1_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_10/ReadVariableOpз
/model_1/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_10/ReadVariableOp_1
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ђ
/model_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3$model_1/conv2d_16/Relu:activations:05model_1/batch_normalization_10/ReadVariableOp:value:07model_1/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 21
/model_1/batch_normalization_10/FusedBatchNormV3Ы
'model_1/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_1/conv2d_17/Conv2D/ReadVariableOp
model_1/conv2d_17/Conv2DConv2D3model_1/batch_normalization_10/FusedBatchNormV3:y:0/model_1/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_1/conv2d_17/Conv2DТ
(model_1/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_17/BiasAdd/ReadVariableOpа
model_1/conv2d_17/BiasAddBiasAdd!model_1/conv2d_17/Conv2D:output:00model_1/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_17/BiasAddб
-model_1/batch_normalization_11/ReadVariableOpReadVariableOp6model_1_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_11/ReadVariableOpз
/model_1/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_11/ReadVariableOp_1
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1 
/model_1/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_17/BiasAdd:output:05model_1/batch_normalization_11/ReadVariableOp:value:07model_1/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 21
/model_1/batch_normalization_11/FusedBatchNormV3Ф
model_1/add_5/addAddV23model_1/batch_normalization_11/FusedBatchNormV3:y:0$model_1/conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/add_5/add
model_1/activation_5/ReluRelumodel_1/add_5/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/activation_5/ReluЧ
9model_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/global_average_pooling2d_1/Mean/reduction_indicesљ
'model_1/global_average_pooling2d_1/MeanMean'model_1/activation_5/Relu:activations:0Bmodel_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2)
'model_1/global_average_pooling2d_1/Mean
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
model_1/flatten_1/ConstЧ
model_1/flatten_1/ReshapeReshape0model_1/global_average_pooling2d_1/Mean:output:0 model_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_1/flatten_1/ReshapeО
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02'
%model_1/dense_2/MatMul/ReadVariableOpР
model_1/dense_2/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_2/MatMulН
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&model_1/dense_2/BiasAdd/ReadVariableOpТ
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_2/BiasAdd
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_2/ReluО
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%model_1/dense_3/MatMul/ReadVariableOpП
model_1/dense_3/MatMulMatMul"model_1/dense_2/Relu:activations:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_1/dense_3/MatMulМ
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOpС
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_1/dense_3/BiasAdd
model_1/dense_3/SigmoidSigmoid model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_1/dense_3/Sigmoido
IdentityIdentitymodel_1/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22:::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:
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
: 

n
__inference_loss_fn_17_56765=
9conv2d_17_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_17_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/adde
IdentityIdentity"conv2d_17/bias/Regularizer/add:z:0*
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
Ў
Љ
6__inference_batch_normalization_11_layer_call_fn_56306

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_524752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
Ќ
Ј
5__inference_batch_normalization_8_layer_call_fn_55809

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_521752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
і
Ј
5__inference_batch_normalization_6_layer_call_fn_55353

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_510032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
ц$
з
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_51500

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
с
~
)__inference_conv2d_17_layer_call_fn_51781

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_517712
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
В 
Ќ
D__inference_conv2d_15_layer_call_and_return_conditional_losses_51570

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
ReluЯ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addР
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ш

Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56115

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
ј
p
__inference_loss_fn_14_56726?
;conv2d_16_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_16_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addg
IdentityIdentity$conv2d_16/kernel/Regularizer/add:z:0*
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
ц$
з
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55309

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
 
Q
%__inference_add_3_layer_call_fn_55618
inputs_0
inputs_1
identityД
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_521132
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22:џџџџџџџџџ22:Y U
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/1
Ќ
Ј
5__inference_batch_normalization_6_layer_call_fn_55415

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_519642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
Ч

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_52193

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
$
з
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_52175

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
ј
Љ
6__inference_batch_normalization_10_layer_call_fn_56216

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_517332
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
є
Ј
5__inference_batch_normalization_8_layer_call_fn_55734

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_513372
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
ы
l
__inference_loss_fn_19_56791;
7dense_2_bias_regularizer_square_readvariableop_resource
identityе
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_2_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
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

m
__inference_loss_fn_7_56635=
9conv2d_12_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_12_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/adde
IdentityIdentity"conv2d_12/bias/Regularizer/add:z:0*
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
Њ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_56422

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
$
и
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_52386

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
ц$
з
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_50972

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
с
~
)__inference_conv2d_14_layer_call_fn_51416

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_514062
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л
Ќ
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51771

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddЯ
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addР
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ї
o
__inference_loss_fn_6_56622?
;conv2d_12_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_12_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addg
IdentityIdentity$conv2d_12/kernel/Regularizer/add:z:0*
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
Ч

P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_51982

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
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


P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55899

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
ц$
з
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55487

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
і
Њ
B__inference_dense_3_layer_call_and_return_conditional_losses_52642

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
SigmoidФ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addМ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55721

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
$
з
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55562

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
е
c
G__inference_activation_5_layer_call_and_return_conditional_losses_56411

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs
ж
l
@__inference_add_3_layer_call_and_return_conditional_losses_55612
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22:џџџџџџџџџ22:Y U
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/1
і
Њ
B__inference_dense_3_layer_call_and_return_conditional_losses_56522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
SigmoidФ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addМ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ў
Ј
5__inference_batch_normalization_8_layer_call_fn_55822

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_521932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
с
~
)__inference_conv2d_13_layer_call_fn_51253

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_512432
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

m
__inference_loss_fn_5_56609=
9conv2d_11_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_11_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/adde
IdentityIdentity"conv2d_11/bias/Regularizer/add:z:0*
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


P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_51166

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
Ю
j
@__inference_add_5_layer_call_and_return_conditional_losses_52535

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22@:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs
і
Ј
5__inference_batch_normalization_9_layer_call_fn_55925

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515312
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
і
Љ
6__inference_batch_normalization_10_layer_call_fn_56203

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_517022
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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

l
__inference_loss_fn_1_56557<
8conv2d_9_bias_regularizer_square_readvariableop_resource
identityз
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_9_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addd
IdentityIdentity!conv2d_9/bias/Regularizer/add:z:0*
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

m
__inference_loss_fn_3_56583=
9conv2d_10_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_10_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/adde
IdentityIdentity"conv2d_10/bias/Regularizer/add:z:0*
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
Ў
n
__inference_loss_fn_20_56804=
9dense_3_kernel_regularizer_square_readvariableop_resource
identityп
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/adde
IdentityIdentity"dense_3/kernel/Regularizer/add:z:0*
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
В 
Ќ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_51205

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluЯ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addР
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ч

P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55402

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
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
Ў
Ј
5__inference_batch_normalization_7_layer_call_fn_55606

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_520712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
Ш

Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_52404

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
е
c
G__inference_activation_5_layer_call_and_return_conditional_losses_52549

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs
$
з
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_52264

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
і
Ј
5__inference_batch_normalization_8_layer_call_fn_55747

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_513682
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
А
Љ
6__inference_batch_normalization_11_layer_call_fn_56319

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_524932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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


P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_51003

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
$
з
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55956

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
Я№
я
B__inference_model_1_layer_call_and_return_conditional_losses_53132
input_2
conv2d_9_52838
conv2d_9_52840
conv2d_10_52843
conv2d_10_52845
batch_normalization_6_52848
batch_normalization_6_52850
batch_normalization_6_52852
batch_normalization_6_52854
conv2d_11_52857
conv2d_11_52859
batch_normalization_7_52862
batch_normalization_7_52864
batch_normalization_7_52866
batch_normalization_7_52868
conv2d_12_52873
conv2d_12_52875
conv2d_13_52878
conv2d_13_52880
batch_normalization_8_52883
batch_normalization_8_52885
batch_normalization_8_52887
batch_normalization_8_52889
conv2d_14_52892
conv2d_14_52894
batch_normalization_9_52897
batch_normalization_9_52899
batch_normalization_9_52901
batch_normalization_9_52903
conv2d_15_52908
conv2d_15_52910
conv2d_16_52913
conv2d_16_52915 
batch_normalization_10_52918 
batch_normalization_10_52920 
batch_normalization_10_52922 
batch_normalization_10_52924
conv2d_17_52927
conv2d_17_52929 
batch_normalization_11_52932 
batch_normalization_11_52934 
batch_normalization_11_52936 
batch_normalization_11_52938
dense_2_52945
dense_2_52947
dense_3_52950
dense_3_52952
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallћ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_9_52838conv2d_9_52840*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_508402"
 conv2d_9/StatefulPartitionedCallЂ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_52843conv2d_10_52845*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_508782#
!conv2d_10/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_52848batch_normalization_6_52850batch_normalization_6_52852batch_normalization_6_52854*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_519822/
-batch_normalization_6/StatefulPartitionedCallЏ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_52857conv2d_11_52859*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_510412#
!conv2d_11/StatefulPartitionedCall
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_52862batch_normalization_7_52864batch_normalization_7_52866batch_normalization_7_52868*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_520712/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_521132
add_3/PartitionedCallр
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_521272
activation_3/PartitionedCall
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_52873conv2d_12_52875*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_512052#
!conv2d_12/StatefulPartitionedCallЃ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_52878conv2d_13_52880*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_512432#
!conv2d_13/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_52883batch_normalization_8_52885batch_normalization_8_52887batch_normalization_8_52889*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_521932/
-batch_normalization_8/StatefulPartitionedCallЏ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_52892conv2d_14_52894*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_514062#
!conv2d_14/StatefulPartitionedCall
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_52897batch_normalization_9_52899batch_normalization_9_52901batch_normalization_9_52903*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_522822/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_523242
add_4/PartitionedCallр
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_523382
activation_4/PartitionedCall
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_52908conv2d_15_52910*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_515702#
!conv2d_15/StatefulPartitionedCallЃ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_52913conv2d_16_52915*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_516082#
!conv2d_16/StatefulPartitionedCallЄ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_52918batch_normalization_10_52920batch_normalization_10_52922batch_normalization_10_52924*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5240420
.batch_normalization_10/StatefulPartitionedCallА
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_52927conv2d_17_52929*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_517712#
!conv2d_17/StatefulPartitionedCallЄ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_52932batch_normalization_11_52934batch_normalization_11_52936batch_normalization_11_52938*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5249320
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_525352
add_5/PartitionedCallр
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_525492
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_519142,
*global_average_pooling2d_1/PartitionedCallф
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_525642
flatten_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_52945dense_2_52947*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_525992!
dense_2/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_52950dense_3_52952*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_526422!
dense_3/StatefulPartitionedCallН
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_52838*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/add­
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_52840*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addР
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_52843*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addА
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_52845*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addР
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_52857*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addА
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_52859*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addР
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_52873*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addА
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_52875*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addР
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_52878*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addА
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_52880*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addР
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_52892*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addА
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_52894*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addР
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_52908*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addА
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_52910*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addР
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_52913*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addА
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_52915*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addР
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_52927*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addА
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_52929*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addГ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_52945*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЋ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_52947*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addГ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_52950*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЊ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_52952*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЅ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:
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
: 
с
~
)__inference_conv2d_12_layer_call_fn_51215

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_512052
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ж
l
@__inference_add_4_layer_call_and_return_conditional_losses_56006
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22 :џџџџџџџџџ22 :Y U
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/1
Ч

P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_52071

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
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


P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55505

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
$
и
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_52475

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
ж
l
@__inference_add_5_layer_call_and_return_conditional_losses_56400
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22@:џџџџџџџџџ22@:Y U
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/1
ѓ
E
)__inference_flatten_1_layer_call_fn_56427

inputs
identityЃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_525642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
с
~
)__inference_conv2d_15_layer_call_fn_51580

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_515702
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
е
c
G__inference_activation_3_layer_call_and_return_conditional_losses_52127

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
ы
Г
'__inference_model_1_layer_call_fn_55137

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

unknown_44
identityЂStatefulPartitionedCallІ
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
unknown_44*:
Tin3
12/*
Tout
2*'
_output_shapes
:џџџџџџџџџ*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_534322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
: 
В 
Ќ
D__inference_conv2d_16_layer_call_and_return_conditional_losses_51608

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
ReluЯ
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addР
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ў
Љ
6__inference_batch_normalization_10_layer_call_fn_56128

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_523862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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

n
__inference_loss_fn_15_56739=
9conv2d_16_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_16_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/adde
IdentityIdentity"conv2d_16/bias/Regularizer/add:z:0*
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
ї
o
__inference_loss_fn_2_56570?
;conv2d_10_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_10_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addg
IdentityIdentity$conv2d_10/kernel/Regularizer/add:z:0*
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
Ч

P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_52282

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
ї
Г
'__inference_model_1_layer_call_fn_55234

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

unknown_44
identityЂStatefulPartitionedCallВ
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
unknown_44*:
Tin3
12/*
Tout
2*'
_output_shapes
:џџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_538262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
: 
ц$
з
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55881

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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

m
__inference_loss_fn_9_56661=
9conv2d_13_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_13_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/adde
IdentityIdentity"conv2d_13/bias/Regularizer/add:z:0*
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
ч$
и
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56350

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
Ь№
ю
B__inference_model_1_layer_call_and_return_conditional_losses_53826

inputs
conv2d_9_53532
conv2d_9_53534
conv2d_10_53537
conv2d_10_53539
batch_normalization_6_53542
batch_normalization_6_53544
batch_normalization_6_53546
batch_normalization_6_53548
conv2d_11_53551
conv2d_11_53553
batch_normalization_7_53556
batch_normalization_7_53558
batch_normalization_7_53560
batch_normalization_7_53562
conv2d_12_53567
conv2d_12_53569
conv2d_13_53572
conv2d_13_53574
batch_normalization_8_53577
batch_normalization_8_53579
batch_normalization_8_53581
batch_normalization_8_53583
conv2d_14_53586
conv2d_14_53588
batch_normalization_9_53591
batch_normalization_9_53593
batch_normalization_9_53595
batch_normalization_9_53597
conv2d_15_53602
conv2d_15_53604
conv2d_16_53607
conv2d_16_53609 
batch_normalization_10_53612 
batch_normalization_10_53614 
batch_normalization_10_53616 
batch_normalization_10_53618
conv2d_17_53621
conv2d_17_53623 
batch_normalization_11_53626 
batch_normalization_11_53628 
batch_normalization_11_53630 
batch_normalization_11_53632
dense_2_53639
dense_2_53641
dense_3_53644
dense_3_53646
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallњ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_53532conv2d_9_53534*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_508402"
 conv2d_9/StatefulPartitionedCallЂ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_53537conv2d_10_53539*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_508782#
!conv2d_10/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_53542batch_normalization_6_53544batch_normalization_6_53546batch_normalization_6_53548*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_519822/
-batch_normalization_6/StatefulPartitionedCallЏ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_53551conv2d_11_53553*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_510412#
!conv2d_11/StatefulPartitionedCall
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_53556batch_normalization_7_53558batch_normalization_7_53560batch_normalization_7_53562*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_520712/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_521132
add_3/PartitionedCallр
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_521272
activation_3/PartitionedCall
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_53567conv2d_12_53569*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_512052#
!conv2d_12/StatefulPartitionedCallЃ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_53572conv2d_13_53574*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_512432#
!conv2d_13/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_53577batch_normalization_8_53579batch_normalization_8_53581batch_normalization_8_53583*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_521932/
-batch_normalization_8/StatefulPartitionedCallЏ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_53586conv2d_14_53588*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_514062#
!conv2d_14/StatefulPartitionedCall
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_53591batch_normalization_9_53593batch_normalization_9_53595batch_normalization_9_53597*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_522822/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_523242
add_4/PartitionedCallр
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_523382
activation_4/PartitionedCall
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_53602conv2d_15_53604*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_515702#
!conv2d_15/StatefulPartitionedCallЃ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_53607conv2d_16_53609*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_516082#
!conv2d_16/StatefulPartitionedCallЄ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_53612batch_normalization_10_53614batch_normalization_10_53616batch_normalization_10_53618*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5240420
.batch_normalization_10/StatefulPartitionedCallА
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_53621conv2d_17_53623*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_517712#
!conv2d_17/StatefulPartitionedCallЄ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_53626batch_normalization_11_53628batch_normalization_11_53630batch_normalization_11_53632*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5249320
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_525352
add_5/PartitionedCallр
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_525492
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_519142,
*global_average_pooling2d_1/PartitionedCallф
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_525642
flatten_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_53639dense_2_53641*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_525992!
dense_2/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_53644dense_3_53646*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_526422!
dense_3/StatefulPartitionedCallН
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_53532*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/add­
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_53534*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addР
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_53537*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addА
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_53539*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addР
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_53551*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addА
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_53553*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addР
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_53567*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addА
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_53569*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addР
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_53572*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addА
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_53574*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addР
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_53586*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addА
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_53588*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addР
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_53602*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addА
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_53604*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addР
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_53607*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addА
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_53609*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addР
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_53621*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addА
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_53623*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addГ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_53639*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЋ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_53641*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addГ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_53644*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЊ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_53646*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЅ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
: 
і
Ј
5__inference_batch_normalization_7_layer_call_fn_55531

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511662
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
ц$
з
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55703

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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


Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_51896

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
є
Ј
5__inference_batch_normalization_9_layer_call_fn_55912

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515002
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
ј
p
__inference_loss_fn_12_56700?
;conv2d_15_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_15_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addg
IdentityIdentity$conv2d_15/kernel/Regularizer/add:z:0*
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
Ќ
Ј
5__inference_batch_normalization_9_layer_call_fn_55987

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_522642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
с
~
)__inference_conv2d_16_layer_call_fn_51618

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_516082
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

n
__inference_loss_fn_11_56687=
9conv2d_14_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_14_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/adde
IdentityIdentity"conv2d_14/bias/Regularizer/add:z:0*
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


P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_51368

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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
Л
Ќ
D__inference_conv2d_11_layer_call_and_return_conditional_losses_51041

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddЯ
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addР
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ш

Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56293

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
$
и
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56275

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
Ч

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55796

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
В 
Ќ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_51243

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluЯ
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addР
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
п
}
(__inference_conv2d_9_layer_call_fn_50850

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_508402
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
$
з
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55778

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
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
ј
Љ
6__inference_batch_normalization_11_layer_call_fn_56394

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_518962
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
е
c
G__inference_activation_4_layer_call_and_return_conditional_losses_52338

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs
 
Q
%__inference_add_5_layer_call_fn_56406
inputs_0
inputs_1
identityД
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_525352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22@:џџџџџџџџџ22@:Y U
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/1
g

__inference__traced_save_56982
file_prefix.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_21cc48b8a4614b008584778468415924/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameб
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*у
valueйBж.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЙ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Є
_input_shapes
: ::::::::::::::: : :  : : : : : :  : : : : : : @:@:@@:@:@:@:@:@:@@:@:@:@:@:@:	@::	:: 2(
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
:	@:!,

_output_shapes	
::%-!

_output_shapes
:	: .

_output_shapes
::/

_output_shapes
: 
ч$
и
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56172

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
ї
|
'__inference_dense_2_layer_call_fn_56479

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_525992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
љ
Њ
B__inference_dense_2_layer_call_and_return_conditional_losses_52599

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluФ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addН
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
є
Ј
5__inference_batch_normalization_6_layer_call_fn_55340

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_509722
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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


Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_51733

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
є
Ј
5__inference_batch_normalization_7_layer_call_fn_55518

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_511352
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
Л
Ќ
D__inference_conv2d_14_layer_call_and_return_conditional_losses_51406

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddЯ
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addР
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
l
__inference_loss_fn_21_56817;
7dense_3_bias_regularizer_square_readvariableop_resource
identityд
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addc
IdentityIdentity dense_3/bias/Regularizer/add:z:0*
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
$
з
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_51964

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
ч$
и
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_51865

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
д
А
#__inference_signature_wrapper_54268
input_2
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

unknown_44
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*'
_output_shapes
:џџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_508132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:
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
: 
љ
Њ
B__inference_dense_2_layer_call_and_return_conditional_losses_56470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluФ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addН
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
е
c
G__inference_activation_4_layer_call_and_return_conditional_losses_56017

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs
А
Љ
6__inference_batch_normalization_10_layer_call_fn_56141

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_524042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
с
~
)__inference_conv2d_10_layer_call_fn_50888

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_508782
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
КЦ

!__inference__traced_restore_57132
file_prefix$
 assignvariableop_conv2d_9_kernel$
 assignvariableop_1_conv2d_9_bias'
#assignvariableop_2_conv2d_10_kernel%
!assignvariableop_3_conv2d_10_bias2
.assignvariableop_4_batch_normalization_6_gamma1
-assignvariableop_5_batch_normalization_6_beta8
4assignvariableop_6_batch_normalization_6_moving_mean<
8assignvariableop_7_batch_normalization_6_moving_variance'
#assignvariableop_8_conv2d_11_kernel%
!assignvariableop_9_conv2d_11_bias3
/assignvariableop_10_batch_normalization_7_gamma2
.assignvariableop_11_batch_normalization_7_beta9
5assignvariableop_12_batch_normalization_7_moving_mean=
9assignvariableop_13_batch_normalization_7_moving_variance(
$assignvariableop_14_conv2d_12_kernel&
"assignvariableop_15_conv2d_12_bias(
$assignvariableop_16_conv2d_13_kernel&
"assignvariableop_17_conv2d_13_bias3
/assignvariableop_18_batch_normalization_8_gamma2
.assignvariableop_19_batch_normalization_8_beta9
5assignvariableop_20_batch_normalization_8_moving_mean=
9assignvariableop_21_batch_normalization_8_moving_variance(
$assignvariableop_22_conv2d_14_kernel&
"assignvariableop_23_conv2d_14_bias3
/assignvariableop_24_batch_normalization_9_gamma2
.assignvariableop_25_batch_normalization_9_beta9
5assignvariableop_26_batch_normalization_9_moving_mean=
9assignvariableop_27_batch_normalization_9_moving_variance(
$assignvariableop_28_conv2d_15_kernel&
"assignvariableop_29_conv2d_15_bias(
$assignvariableop_30_conv2d_16_kernel&
"assignvariableop_31_conv2d_16_bias4
0assignvariableop_32_batch_normalization_10_gamma3
/assignvariableop_33_batch_normalization_10_beta:
6assignvariableop_34_batch_normalization_10_moving_mean>
:assignvariableop_35_batch_normalization_10_moving_variance(
$assignvariableop_36_conv2d_17_kernel&
"assignvariableop_37_conv2d_17_bias4
0assignvariableop_38_batch_normalization_11_gamma3
/assignvariableop_39_batch_normalization_11_beta:
6assignvariableop_40_batch_normalization_11_moving_mean>
:assignvariableop_41_batch_normalization_11_moving_variance&
"assignvariableop_42_dense_2_kernel$
 assignvariableop_43_dense_2_bias&
"assignvariableop_44_dense_3_kernel$
 assignvariableop_45_dense_3_bias
identity_47ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1з
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*у
valueйBж.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_9_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_9_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_10_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_10_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Є
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_6_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_6_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Њ
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_6_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ў
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_6_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_11_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_11_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Ј
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_7_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ї
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_7_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ў
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_7_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13В
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_7_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_12_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_12_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_13_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_13_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ј
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_8_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ї
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_8_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ў
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_8_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_8_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_14_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_14_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ј
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_9_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ї
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_9_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ў
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_9_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27В
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_9_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_15_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_15_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_16_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_16_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Љ
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_10_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ј
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_10_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Џ
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_10_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Г
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_10_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_17_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_17_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Љ
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_11_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39Ј
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_11_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Џ
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_11_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_11_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_2_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_2_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_3_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_3_biasIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
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
NoOpв
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46п
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*Я
_input_shapesН
К: ::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_45AssignVariableOp_452(
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
: 
Њ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_52564

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
і
Љ
6__inference_batch_normalization_11_layer_call_fn_56381

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_518652
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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

Ћ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_50840

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddЭ
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addО
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55327

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
В 
Ќ
D__inference_conv2d_10_layer_call_and_return_conditional_losses_50878

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluЯ
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addР
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
$
з
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_52053

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
њ
Д
'__inference_model_1_layer_call_fn_53921
input_2
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

unknown_44
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*'
_output_shapes
:џџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_538262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:
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
: 
У№
я
B__inference_model_1_layer_call_and_return_conditional_losses_52835
input_2
conv2d_9_51924
conv2d_9_51926
conv2d_10_51929
conv2d_10_51931
batch_normalization_6_52009
batch_normalization_6_52011
batch_normalization_6_52013
batch_normalization_6_52015
conv2d_11_52018
conv2d_11_52020
batch_normalization_7_52098
batch_normalization_7_52100
batch_normalization_7_52102
batch_normalization_7_52104
conv2d_12_52135
conv2d_12_52137
conv2d_13_52140
conv2d_13_52142
batch_normalization_8_52220
batch_normalization_8_52222
batch_normalization_8_52224
batch_normalization_8_52226
conv2d_14_52229
conv2d_14_52231
batch_normalization_9_52309
batch_normalization_9_52311
batch_normalization_9_52313
batch_normalization_9_52315
conv2d_15_52346
conv2d_15_52348
conv2d_16_52351
conv2d_16_52353 
batch_normalization_10_52431 
batch_normalization_10_52433 
batch_normalization_10_52435 
batch_normalization_10_52437
conv2d_17_52440
conv2d_17_52442 
batch_normalization_11_52520 
batch_normalization_11_52522 
batch_normalization_11_52524 
batch_normalization_11_52526
dense_2_52610
dense_2_52612
dense_3_52653
dense_3_52655
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallћ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_9_51924conv2d_9_51926*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_508402"
 conv2d_9/StatefulPartitionedCallЂ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_51929conv2d_10_51931*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_508782#
!conv2d_10/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_52009batch_normalization_6_52011batch_normalization_6_52013batch_normalization_6_52015*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_519642/
-batch_normalization_6/StatefulPartitionedCallЏ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_52018conv2d_11_52020*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_510412#
!conv2d_11/StatefulPartitionedCall
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_52098batch_normalization_7_52100batch_normalization_7_52102batch_normalization_7_52104*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_520532/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_521132
add_3/PartitionedCallр
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_521272
activation_3/PartitionedCall
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_52135conv2d_12_52137*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_512052#
!conv2d_12/StatefulPartitionedCallЃ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_52140conv2d_13_52142*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_512432#
!conv2d_13/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_52220batch_normalization_8_52222batch_normalization_8_52224batch_normalization_8_52226*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_521752/
-batch_normalization_8/StatefulPartitionedCallЏ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_52229conv2d_14_52231*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_514062#
!conv2d_14/StatefulPartitionedCall
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_52309batch_normalization_9_52311batch_normalization_9_52313batch_normalization_9_52315*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_522642/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_523242
add_4/PartitionedCallр
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_523382
activation_4/PartitionedCall
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_52346conv2d_15_52348*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_515702#
!conv2d_15/StatefulPartitionedCallЃ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_52351conv2d_16_52353*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_516082#
!conv2d_16/StatefulPartitionedCallЂ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_52431batch_normalization_10_52433batch_normalization_10_52435batch_normalization_10_52437*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5238620
.batch_normalization_10/StatefulPartitionedCallА
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_52440conv2d_17_52442*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_517712#
!conv2d_17/StatefulPartitionedCallЂ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_52520batch_normalization_11_52522batch_normalization_11_52524batch_normalization_11_52526*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5247520
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_525352
add_5/PartitionedCallр
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_525492
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_519142,
*global_average_pooling2d_1/PartitionedCallф
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_525642
flatten_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_52610dense_2_52612*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_525992!
dense_2/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_52653dense_3_52655*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_526422!
dense_3/StatefulPartitionedCallН
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_51924*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/add­
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_51926*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addР
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_51929*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addА
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_51931*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addР
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_52018*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addА
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_52020*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addР
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_52135*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addА
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_52137*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addР
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_52140*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addА
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_52142*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addР
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_52229*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addА
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_52231*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addР
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_52346*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addА
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_52348*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addР
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_52351*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addА
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_52353*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addР
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_52440*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addА
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_52442*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addГ
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_52610*
_output_shapes
:	@*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOpД
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_2/kernel/Regularizer/Square
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/ConstК
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_2/kernel/Regularizer/mul/xМ
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul
 dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/add/xЙ
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/addЋ
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_52612*
_output_shapes	
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOpЊ
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_2/bias/Regularizer/Square
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/ConstВ
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_2/bias/Regularizer/mul/xД
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul
dense_2/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_2/bias/Regularizer/add/xБ
dense_2/bias/Regularizer/addAddV2'dense_2/bias/Regularizer/add/x:output:0 dense_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/addГ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_52653*
_output_shapes
:	*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЊ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_52655*
_output_shapes
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЉ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЅ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:
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
: 


P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_51531

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
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


Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56190

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
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
Ш

Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_52493

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
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
ї
o
__inference_loss_fn_8_56648?
;conv2d_13_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_13_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addg
IdentityIdentity$conv2d_13/kernel/Regularizer/add:z:0*
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

n
__inference_loss_fn_13_56713=
9conv2d_15_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_15_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/adde
IdentityIdentity"conv2d_15/bias/Regularizer/add:z:0*
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
$
з
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55384

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
Ў
Ј
5__inference_batch_normalization_6_layer_call_fn_55428

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_519822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
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
Ч

P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55974

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
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

H
,__inference_activation_4_layer_call_fn_56022

inputs
identityЎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_523382
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs
Л
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_51914

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_default
C
input_28
serving_default_input_2:0џџџџџџџџџ22;
dense_30
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
Џѓ
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
layer_with_weights-16
layer-25
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+К&call_and_return_all_conditional_losses
Л__call__
М_default_save_signature"Ьы
_tf_keras_modelБы{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}], ["conv2d_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}], ["conv2d_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}], ["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["global_average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}], ["conv2d_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}], ["conv2d_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}], ["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["global_average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_3", 0, 0]]}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
Ю


 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"Ї	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}}
а


&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
	
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
в


5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
	
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}

Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 16]}, {"class_name": "TensorShape", "items": [null, 50, 50, 16]}]}
Д
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
а


Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
а


Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
	
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
в


akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
	
gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+г&call_and_return_all_conditional_losses
д__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}

pregularization_losses
qtrainable_variables
r	variables
s	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 32]}, {"class_name": "TensorShape", "items": [null, 50, 50, 32]}]}
Д
tregularization_losses
utrainable_variables
v	variables
w	keras_api
+з&call_and_return_all_conditional_losses
и__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
а


xkernel
ybias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
+й&call_and_return_all_conditional_losses
к__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
д


~kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+л&call_and_return_all_conditional_losses
м__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Є	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
и

kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
+п&call_and_return_all_conditional_losses
р__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Є	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}

regularization_losses
trainable_variables
	variables
	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 64]}, {"class_name": "TensorShape", "items": [null, 50, 50, 64]}]}
И
 regularization_losses
Ёtrainable_variables
Ђ	variables
Ѓ	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
њ
Єregularization_losses
Ѕtrainable_variables
І	variables
Ї	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"х
_tf_keras_layerЫ{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Щ
Јregularization_losses
Љtrainable_variables
Њ	variables
Ћ	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"Д
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
р
Ќkernel
	­bias
Ўregularization_losses
Џtrainable_variables
А	variables
Б	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"Г
_tf_keras_layer{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
у
Вkernel
	Гbias
Дregularization_losses
Еtrainable_variables
Ж	variables
З	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"Ж
_tf_keras_layer{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
м
я0
№1
ё2
ђ3
ѓ4
є5
ѕ6
і7
ї8
ј9
љ10
њ11
ћ12
ќ13
§14
ў15
џ16
17
18
19
20
21"
trackable_list_wrapper
А
 0
!1
&2
'3
-4
.5
56
67
<8
=9
L10
M11
R12
S13
Y14
Z15
a16
b17
h18
i19
x20
y21
~22
23
24
25
26
27
28
29
Ќ30
­31
В32
Г33"
trackable_list_wrapper

 0
!1
&2
'3
-4
.5
/6
07
58
69
<10
=11
>12
?13
L14
M15
R16
S17
Y18
Z19
[20
\21
a22
b23
h24
i25
j26
k27
x28
y29
~30
31
32
33
34
35
36
37
38
39
40
41
Ќ42
­43
В44
Г45"
trackable_list_wrapper
г
Иlayer_metrics
 Йlayer_regularization_losses
regularization_losses
trainable_variables
Кnon_trainable_variables
	variables
Лmetrics
Мlayers
Л__call__
М_default_save_signature
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
):'2conv2d_9/kernel
:2conv2d_9/bias
0
я0
№1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
Е
Нlayer_metrics
 Оlayer_regularization_losses
"regularization_losses
#trainable_variables
Пnon_trainable_variables
$	variables
Рmetrics
Сlayers
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_10/kernel
:2conv2d_10/bias
0
ё0
ђ1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
Е
Тlayer_metrics
 Уlayer_regularization_losses
(regularization_losses
)trainable_variables
Фnon_trainable_variables
*	variables
Хmetrics
Цlayers
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
Е
Чlayer_metrics
 Шlayer_regularization_losses
1regularization_losses
2trainable_variables
Щnon_trainable_variables
3	variables
Ъmetrics
Ыlayers
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_11/kernel
:2conv2d_11/bias
0
ѓ0
є1"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
Е
Ьlayer_metrics
 Эlayer_regularization_losses
7regularization_losses
8trainable_variables
Юnon_trainable_variables
9	variables
Яmetrics
аlayers
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
Е
бlayer_metrics
 вlayer_regularization_losses
@regularization_losses
Atrainable_variables
гnon_trainable_variables
B	variables
дmetrics
еlayers
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
жlayer_metrics
 зlayer_regularization_losses
Dregularization_losses
Etrainable_variables
иnon_trainable_variables
F	variables
йmetrics
кlayers
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
лlayer_metrics
 мlayer_regularization_losses
Hregularization_losses
Itrainable_variables
нnon_trainable_variables
J	variables
оmetrics
пlayers
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
0
ѕ0
і1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
Е
рlayer_metrics
 сlayer_regularization_losses
Nregularization_losses
Otrainable_variables
тnon_trainable_variables
P	variables
уmetrics
фlayers
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_13/kernel
: 2conv2d_13/bias
0
ї0
ј1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
Е
хlayer_metrics
 цlayer_regularization_losses
Tregularization_losses
Utrainable_variables
чnon_trainable_variables
V	variables
шmetrics
щlayers
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
Е
ъlayer_metrics
 ыlayer_regularization_losses
]regularization_losses
^trainable_variables
ьnon_trainable_variables
_	variables
эmetrics
юlayers
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_14/kernel
: 2conv2d_14/bias
0
љ0
њ1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
Е
яlayer_metrics
 №layer_regularization_losses
cregularization_losses
dtrainable_variables
ёnon_trainable_variables
e	variables
ђmetrics
ѓlayers
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
<
h0
i1
j2
k3"
trackable_list_wrapper
Е
єlayer_metrics
 ѕlayer_regularization_losses
lregularization_losses
mtrainable_variables
іnon_trainable_variables
n	variables
їmetrics
јlayers
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
љlayer_metrics
 њlayer_regularization_losses
pregularization_losses
qtrainable_variables
ћnon_trainable_variables
r	variables
ќmetrics
§layers
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ўlayer_metrics
 џlayer_regularization_losses
tregularization_losses
utrainable_variables
non_trainable_variables
v	variables
metrics
layers
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_15/kernel
:@2conv2d_15/bias
0
ћ0
ќ1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
zregularization_losses
{trainable_variables
non_trainable_variables
|	variables
metrics
layers
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_16/kernel
:@2conv2d_16/bias
0
§0
ў1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_17/kernel
:@2conv2d_17/bias
0
џ0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_11/gamma
):'@2batch_normalization_11/beta
2:0@ (2"batch_normalization_11/moving_mean
6:4@ (2&batch_normalization_11/moving_variance
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
layers
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
	variables
metrics
 layers
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ёlayer_metrics
 Ђlayer_regularization_losses
 regularization_losses
Ёtrainable_variables
Ѓnon_trainable_variables
Ђ	variables
Єmetrics
Ѕlayers
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Іlayer_metrics
 Їlayer_regularization_losses
Єregularization_losses
Ѕtrainable_variables
Јnon_trainable_variables
І	variables
Љmetrics
Њlayers
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћlayer_metrics
 Ќlayer_regularization_losses
Јregularization_losses
Љtrainable_variables
­non_trainable_variables
Њ	variables
Ўmetrics
Џlayers
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_2/kernel
:2dense_2/bias
0
0
1"
trackable_list_wrapper
0
Ќ0
­1"
trackable_list_wrapper
0
Ќ0
­1"
trackable_list_wrapper
И
Аlayer_metrics
 Бlayer_regularization_losses
Ўregularization_losses
Џtrainable_variables
Вnon_trainable_variables
А	variables
Гmetrics
Дlayers
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_3/kernel
:2dense_3/bias
0
0
1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
И
Еlayer_metrics
 Жlayer_regularization_losses
Дregularization_losses
Еtrainable_variables
Зnon_trainable_variables
Ж	variables
Иmetrics
Йlayers
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
z
/0
01
>2
?3
[4
\5
j6
k7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
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
25"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
я0
№1"
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
ё0
ђ1"
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
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
ѓ0
є1"
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
>0
?1"
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
ѕ0
і1"
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
ї0
ј1"
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
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
љ0
њ1"
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
j0
k1"
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
ћ0
ќ1"
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
§0
ў1"
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
џ0
1"
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
0
1"
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
0
1"
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ж2г
B__inference_model_1_layer_call_and_return_conditional_losses_55040
B__inference_model_1_layer_call_and_return_conditional_losses_54693
B__inference_model_1_layer_call_and_return_conditional_losses_53132
B__inference_model_1_layer_call_and_return_conditional_losses_52835Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
'__inference_model_1_layer_call_fn_53527
'__inference_model_1_layer_call_fn_53921
'__inference_model_1_layer_call_fn_55137
'__inference_model_1_layer_call_fn_55234Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
 __inference__wrapped_model_50813О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_2џџџџџџџџџ22
Ђ2
C__inference_conv2d_9_layer_call_and_return_conditional_losses_50840з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
(__inference_conv2d_9_layer_call_fn_50850з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ѓ2 
D__inference_conv2d_10_layer_call_and_return_conditional_losses_50878з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_conv2d_10_layer_call_fn_50888з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2џ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55402
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55384
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55309
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55327Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_6_layer_call_fn_55415
5__inference_batch_normalization_6_layer_call_fn_55340
5__inference_batch_normalization_6_layer_call_fn_55428
5__inference_batch_normalization_6_layer_call_fn_55353Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ѓ2 
D__inference_conv2d_11_layer_call_and_return_conditional_losses_51041з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_conv2d_11_layer_call_fn_51051з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2џ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55580
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55505
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55487
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55562Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_7_layer_call_fn_55593
5__inference_batch_normalization_7_layer_call_fn_55518
5__inference_batch_normalization_7_layer_call_fn_55606
5__inference_batch_normalization_7_layer_call_fn_55531Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
@__inference_add_3_layer_call_and_return_conditional_losses_55612Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_add_3_layer_call_fn_55618Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_activation_3_layer_call_and_return_conditional_losses_55623Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_activation_3_layer_call_fn_55628Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ2 
D__inference_conv2d_12_layer_call_and_return_conditional_losses_51205з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_conv2d_12_layer_call_fn_51215з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ѓ2 
D__inference_conv2d_13_layer_call_and_return_conditional_losses_51243з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
)__inference_conv2d_13_layer_call_fn_51253з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2џ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55796
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55703
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55721
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55778Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_8_layer_call_fn_55809
5__inference_batch_normalization_8_layer_call_fn_55734
5__inference_batch_normalization_8_layer_call_fn_55747
5__inference_batch_normalization_8_layer_call_fn_55822Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ѓ2 
D__inference_conv2d_14_layer_call_and_return_conditional_losses_51406з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
)__inference_conv2d_14_layer_call_fn_51416з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2џ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55974
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55881
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55899
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55956Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_9_layer_call_fn_55912
5__inference_batch_normalization_9_layer_call_fn_56000
5__inference_batch_normalization_9_layer_call_fn_55925
5__inference_batch_normalization_9_layer_call_fn_55987Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
@__inference_add_4_layer_call_and_return_conditional_losses_56006Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_add_4_layer_call_fn_56012Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_activation_4_layer_call_and_return_conditional_losses_56017Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_activation_4_layer_call_fn_56022Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ2 
D__inference_conv2d_15_layer_call_and_return_conditional_losses_51570з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
)__inference_conv2d_15_layer_call_fn_51580з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Ѓ2 
D__inference_conv2d_16_layer_call_and_return_conditional_losses_51608з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
)__inference_conv2d_16_layer_call_fn_51618з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56115
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56172
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56190
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56097Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
6__inference_batch_normalization_10_layer_call_fn_56141
6__inference_batch_normalization_10_layer_call_fn_56128
6__inference_batch_normalization_10_layer_call_fn_56203
6__inference_batch_normalization_10_layer_call_fn_56216Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ѓ2 
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51771з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
)__inference_conv2d_17_layer_call_fn_51781з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56275
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56350
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56293
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56368Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
6__inference_batch_normalization_11_layer_call_fn_56381
6__inference_batch_normalization_11_layer_call_fn_56319
6__inference_batch_normalization_11_layer_call_fn_56306
6__inference_batch_normalization_11_layer_call_fn_56394Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
@__inference_add_5_layer_call_and_return_conditional_losses_56400Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_add_5_layer_call_fn_56406Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_activation_5_layer_call_and_return_conditional_losses_56411Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_activation_5_layer_call_fn_56416Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Н2К
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_51914р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ђ2
:__inference_global_average_pooling2d_1_layer_call_fn_51920р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
ю2ы
D__inference_flatten_1_layer_call_and_return_conditional_losses_56422Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_flatten_1_layer_call_fn_56427Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_2_layer_call_and_return_conditional_losses_56470Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_2_layer_call_fn_56479Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_3_layer_call_and_return_conditional_losses_56522Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_3_layer_call_fn_56531Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
В2Џ
__inference_loss_fn_0_56544
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_1_56557
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_2_56570
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_3_56583
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_4_56596
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_5_56609
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_6_56622
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_7_56635
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_8_56648
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_9_56661
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_10_56674
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_11_56687
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_12_56700
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_13_56713
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_14_56726
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_15_56739
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_16_56752
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_17_56765
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_18_56778
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_19_56791
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_20_56804
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_21_56817
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
2B0
#__inference_signature_wrapper_54268input_2а
 __inference__wrapped_model_50813Ћ< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ8Ђ5
.Ђ+
)&
input_2џџџџџџџџџ22
Њ "1Њ.
,
dense_3!
dense_3џџџџџџџџџГ
G__inference_activation_3_layer_call_and_return_conditional_losses_55623h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ "-Ђ*
# 
0џџџџџџџџџ22
 
,__inference_activation_3_layer_call_fn_55628[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ " џџџџџџџџџ22Г
G__inference_activation_4_layer_call_and_return_conditional_losses_56017h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 
,__inference_activation_4_layer_call_fn_56022[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22 
Њ " џџџџџџџџџ22 Г
G__inference_activation_5_layer_call_and_return_conditional_losses_56411h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22@
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 
,__inference_activation_5_layer_call_fn_56416[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22@
Њ " џџџџџџџџџ22@р
@__inference_add_3_layer_call_and_return_conditional_losses_55612jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22
*'
inputs/1џџџџџџџџџ22
Њ "-Ђ*
# 
0џџџџџџџџџ22
 И
%__inference_add_3_layer_call_fn_55618jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22
*'
inputs/1џџџџџџџџџ22
Њ " џџџџџџџџџ22р
@__inference_add_4_layer_call_and_return_conditional_losses_56006jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22 
*'
inputs/1џџџџџџџџџ22 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 И
%__inference_add_4_layer_call_fn_56012jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22 
*'
inputs/1џџџџџџџџџ22 
Њ " џџџџџџџџџ22 р
@__inference_add_5_layer_call_and_return_conditional_losses_56400jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22@
*'
inputs/1џџџџџџџџџ22@
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 И
%__inference_add_5_layer_call_fn_56406jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22@
*'
inputs/1џџџџџџџџџ22@
Њ " џџџџџџџџџ22@Ы
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56097v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ы
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56115v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 №
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56172MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 №
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56190MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ѓ
6__inference_batch_normalization_10_layer_call_fn_56128i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ " џџџџџџџџџ22@Ѓ
6__inference_batch_normalization_10_layer_call_fn_56141i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ " џџџџџџџџџ22@Ш
6__inference_batch_normalization_10_layer_call_fn_56203MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ш
6__inference_batch_normalization_10_layer_call_fn_56216MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ы
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56275v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ы
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56293v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 №
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56350MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 №
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56368MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ѓ
6__inference_batch_normalization_11_layer_call_fn_56306i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ " џџџџџџџџџ22@Ѓ
6__inference_batch_normalization_11_layer_call_fn_56319i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ " џџџџџџџџџ22@Ш
6__inference_batch_normalization_11_layer_call_fn_56381MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ш
6__inference_batch_normalization_11_layer_call_fn_56394MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55309-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55327-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55384r-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ц
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_55402r-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22
 У
5__inference_batch_normalization_6_layer_call_fn_55340-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_6_layer_call_fn_55353-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
5__inference_batch_normalization_6_layer_call_fn_55415e-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ " џџџџџџџџџ22
5__inference_batch_normalization_6_layer_call_fn_55428e-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ " џџџџџџџџџ22ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55487<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55505<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55562r<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ц
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_55580r<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22
 У
5__inference_batch_normalization_7_layer_call_fn_55518<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_7_layer_call_fn_55531<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
5__inference_batch_normalization_7_layer_call_fn_55593e<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ " џџџџџџџџџ22
5__inference_batch_normalization_7_layer_call_fn_55606e<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ " џџџџџџџџџ22ы
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55703YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ы
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55721YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ц
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55778rYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ц
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_55796rYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 У
5__inference_batch_normalization_8_layer_call_fn_55734YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ У
5__inference_batch_normalization_8_layer_call_fn_55747YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
5__inference_batch_normalization_8_layer_call_fn_55809eYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ " џџџџџџџџџ22 
5__inference_batch_normalization_8_layer_call_fn_55822eYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ " џџџџџџџџџ22 ы
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55881hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ы
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55899hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ц
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55956rhijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ц
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_55974rhijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 У
5__inference_batch_normalization_9_layer_call_fn_55912hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ У
5__inference_batch_normalization_9_layer_call_fn_55925hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
5__inference_batch_normalization_9_layer_call_fn_55987ehijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ " џџџџџџџџџ22 
5__inference_batch_normalization_9_layer_call_fn_56000ehijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ " џџџџџџџџџ22 й
D__inference_conv2d_10_layer_call_and_return_conditional_losses_50878&'IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_10_layer_call_fn_50888&'IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
D__inference_conv2d_11_layer_call_and_return_conditional_losses_5104156IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_11_layer_call_fn_5105156IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
D__inference_conv2d_12_layer_call_and_return_conditional_losses_51205LMIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Б
)__inference_conv2d_12_layer_call_fn_51215LMIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
D__inference_conv2d_13_layer_call_and_return_conditional_losses_51243RSIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Б
)__inference_conv2d_13_layer_call_fn_51253RSIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
D__inference_conv2d_14_layer_call_and_return_conditional_losses_51406abIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Б
)__inference_conv2d_14_layer_call_fn_51416abIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
D__inference_conv2d_15_layer_call_and_return_conditional_losses_51570xyIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Б
)__inference_conv2d_15_layer_call_fn_51580xyIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@й
D__inference_conv2d_16_layer_call_and_return_conditional_losses_51608~IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Б
)__inference_conv2d_16_layer_call_fn_51618~IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@л
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51771IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Г
)__inference_conv2d_17_layer_call_fn_51781IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@и
C__inference_conv2d_9_layer_call_and_return_conditional_losses_50840 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 А
(__inference_conv2d_9_layer_call_fn_50850 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
B__inference_dense_2_layer_call_and_return_conditional_losses_56470_Ќ­/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 }
'__inference_dense_2_layer_call_fn_56479RЌ­/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЅ
B__inference_dense_3_layer_call_and_return_conditional_losses_56522_ВГ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
'__inference_dense_3_layer_call_fn_56531RВГ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ 
D__inference_flatten_1_layer_call_and_return_conditional_losses_56422X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 x
)__inference_flatten_1_layer_call_fn_56427K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@о
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_51914RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Е
:__inference_global_average_pooling2d_1_layer_call_fn_51920wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ:
__inference_loss_fn_0_56544 Ђ

Ђ 
Њ " ;
__inference_loss_fn_10_56674aЂ

Ђ 
Њ " ;
__inference_loss_fn_11_56687bЂ

Ђ 
Њ " ;
__inference_loss_fn_12_56700xЂ

Ђ 
Њ " ;
__inference_loss_fn_13_56713yЂ

Ђ 
Њ " ;
__inference_loss_fn_14_56726~Ђ

Ђ 
Њ " ;
__inference_loss_fn_15_56739Ђ

Ђ 
Њ " <
__inference_loss_fn_16_56752Ђ

Ђ 
Њ " <
__inference_loss_fn_17_56765Ђ

Ђ 
Њ " <
__inference_loss_fn_18_56778ЌЂ

Ђ 
Њ " <
__inference_loss_fn_19_56791­Ђ

Ђ 
Њ " :
__inference_loss_fn_1_56557!Ђ

Ђ 
Њ " <
__inference_loss_fn_20_56804ВЂ

Ђ 
Њ " <
__inference_loss_fn_21_56817ГЂ

Ђ 
Њ " :
__inference_loss_fn_2_56570&Ђ

Ђ 
Њ " :
__inference_loss_fn_3_56583'Ђ

Ђ 
Њ " :
__inference_loss_fn_4_565965Ђ

Ђ 
Њ " :
__inference_loss_fn_5_566096Ђ

Ђ 
Њ " :
__inference_loss_fn_6_56622LЂ

Ђ 
Њ " :
__inference_loss_fn_7_56635MЂ

Ђ 
Њ " :
__inference_loss_fn_8_56648RЂ

Ђ 
Њ " :
__inference_loss_fn_9_56661SЂ

Ђ 
Њ " ю
B__inference_model_1_layer_call_and_return_conditional_losses_52835Ї< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ю
B__inference_model_1_layer_call_and_return_conditional_losses_53132Ї< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 э
B__inference_model_1_layer_call_and_return_conditional_losses_54693І< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p

 
Њ "%Ђ"

0џџџџџџџџџ
 э
B__inference_model_1_layer_call_and_return_conditional_losses_55040І< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ц
'__inference_model_1_layer_call_fn_53527< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p

 
Њ "џџџџџџџџџЦ
'__inference_model_1_layer_call_fn_53921< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p 

 
Њ "џџџџџџџџџХ
'__inference_model_1_layer_call_fn_55137< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p

 
Њ "џџџџџџџџџХ
'__inference_model_1_layer_call_fn_55234< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p 

 
Њ "џџџџџџџџџо
#__inference_signature_wrapper_54268Ж< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГCЂ@
Ђ 
9Њ6
4
input_2)&
input_2џџџџџџџџџ22"1Њ.
,
dense_3!
dense_3џџџџџџџџџ