4
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
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ѕэ,

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
:*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:*
dtype0

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:*
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
:*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
Є
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:*
dtype0
Є
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:*
dtype0

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
: *
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
: *
dtype0

conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
: *
dtype0

batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_14/gamma

0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
: *
dtype0

batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_14/beta

/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
: *
dtype0

"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_14/moving_mean

6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
: *
dtype0
Є
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_14/moving_variance

:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
: *
dtype0

conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
: *
dtype0

batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_15/gamma

0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
: *
dtype0

batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_15/beta

/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
: *
dtype0

"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_15/moving_mean

6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
: *
dtype0
Є
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_15/moving_variance

:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
: *
dtype0

conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:@*
dtype0

conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:@*
dtype0

batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_16/gamma

0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:@*
dtype0

batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_16/beta

/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:@*
dtype0

"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_16/moving_mean

6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
Є
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_16/moving_variance

:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0

conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:@*
dtype0

batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma

0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0

batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta

/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0

"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean

6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
Є
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance

:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	@*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
иw
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*w
valuewBw Bџv
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
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_24/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_24/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_25/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_25/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_16/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_16/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_16/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_16/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_26/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_26/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_17/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_17/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_17/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_17/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_5/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_3Placeholder*/
_output_shapes
:џџџџџџџџџ22*
dtype0*$
shape:џџџџџџџџџ22

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_23/kernelconv2d_23/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_26/kernelconv2d_26/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*:
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
#__inference_signature_wrapper_82891
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
й
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*;
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
__inference__traced_save_85605
М
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_23/kernelconv2d_23/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_26/kernelconv2d_26/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*:
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
!__inference__traced_restore_85755вч*
ј
Љ
6__inference_batch_normalization_17_layer_call_fn_84942

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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_805192
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
с
~
)__inference_conv2d_25_layer_call_fn_80241

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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_802312
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
ї
|
'__inference_dense_5_layer_call_fn_85154

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
B__inference_dense_5_layer_call_and_return_conditional_losses_812652
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
љ
Њ
B__inference_dense_4_layer_call_and_return_conditional_losses_81222

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
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addН
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addg
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

m
__inference_loss_fn_5_85232=
9conv2d_20_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_20_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/adde
IdentityIdentity"conv2d_20/bias/Regularizer/add:z:0*
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


Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84522

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
$
и
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84579

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


Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_79991

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
ч$
и
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84795

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
аІ

B__inference_model_2_layer_call_and_return_conditional_losses_83663

inputs,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource2
.batch_normalization_14_readvariableop_resource4
0batch_normalization_14_readvariableop_1_resourceC
?batch_normalization_14_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resourceC
?batch_normalization_16_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resourceC
?batch_normalization_17_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityГ
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_18/Conv2D/ReadVariableOpС
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_18/Conv2DЊ
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOpА
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_18/BiasAddГ
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_19/Conv2D/ReadVariableOpе
conv2d_19/Conv2DConv2Dconv2d_18/BiasAdd:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_19/Conv2DЊ
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOpА
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_19/ReluЙ
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_12/ReadVariableOpП
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_12/ReadVariableOp_1ь
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ъ
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_19/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3Г
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpц
conv2d_20/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_20/Conv2DЊ
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOpА
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_20/BiasAddЙ
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_13/ReadVariableOpП
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1ь
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3Ђ
	add_6/addAddV2+batch_normalization_13/FusedBatchNormV3:y:0conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
	add_6/addw
activation_6/ReluReluadd_6/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
activation_6/ReluГ
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpк
conv2d_21/Conv2DConv2Dactivation_6/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_21/Conv2DЊ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOpА
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_21/ReluГ
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_22/Conv2D/ReadVariableOpз
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_22/Conv2DЊ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOpА
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_22/ReluЙ
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_14/ReadVariableOpП
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_14/ReadVariableOp_1ь
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ъ
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_22/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3Г
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_23/Conv2D/ReadVariableOpц
conv2d_23/Conv2DConv2D+batch_normalization_14/FusedBatchNormV3:y:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_23/Conv2DЊ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOpА
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_23/BiasAddЙ
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOpП
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1ь
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_23/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3Є
	add_7/addAddV2+batch_normalization_15/FusedBatchNormV3:y:0conv2d_21/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
	add_7/addw
activation_7/ReluReluadd_7/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
activation_7/ReluГ
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_24/Conv2D/ReadVariableOpк
conv2d_24/Conv2DConv2Dactivation_7/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_24/Conv2DЊ
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOpА
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_24/ReluГ
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_25/Conv2D/ReadVariableOpз
conv2d_25/Conv2DConv2Dconv2d_24/Relu:activations:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_25/Conv2DЊ
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOpА
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_25/ReluЙ
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOpП
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1ь
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ъ
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_25/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3Г
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_26/Conv2D/ReadVariableOpц
conv2d_26/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_26/Conv2DЊ
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOpА
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_26/BiasAddЙ
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOpП
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1ь
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_26/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3Є
	add_8/addAddV2+batch_normalization_17/FusedBatchNormV3:y:0conv2d_24/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
	add_8/addw
activation_8/ReluReluadd_8/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
activation_8/ReluЗ
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_2/Mean/reduction_indicesй
global_average_pooling2d_2/MeanMeanactivation_8/Relu:activations:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling2d_2/Means
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
flatten_2/ConstЇ
flatten_2/ReshapeReshape(global_average_pooling2d_2/Mean:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten_2/ReshapeІ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЅ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЂ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/ReluІ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/Sigmoidй
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addЪ
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/addй
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addЪ
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/addй
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addЪ
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/addй
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addЪ
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/addй
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addЪ
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/addй
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addЪ
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/addй
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addЪ
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/addй
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addЪ
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/addй
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addЪ
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/addЬ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addХ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addЬ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addФ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addg
IdentityIdentitydense_5/Sigmoid:y:0*
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
Ш

Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_81027

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
Ш

Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80905

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
ч$
и
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84504

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
А
Љ
6__inference_batch_normalization_16_layer_call_fn_84764

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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_810272
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


Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80154

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
с
~
)__inference_conv2d_21_layer_call_fn_79838

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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_798282
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
Њ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_85045

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


Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80356

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


Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84916

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
А
Љ
6__inference_batch_normalization_17_layer_call_fn_85017

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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_811162
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
А
Љ
6__inference_batch_normalization_13_layer_call_fn_84229

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
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_806942
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
ѓ
E
)__inference_flatten_2_layer_call_fn_85050

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
D__inference_flatten_2_layer_call_and_return_conditional_losses_811872
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
Нё

B__inference_model_2_layer_call_and_return_conditional_losses_82055

inputs
conv2d_18_81761
conv2d_18_81763
conv2d_19_81766
conv2d_19_81768 
batch_normalization_12_81771 
batch_normalization_12_81773 
batch_normalization_12_81775 
batch_normalization_12_81777
conv2d_20_81780
conv2d_20_81782 
batch_normalization_13_81785 
batch_normalization_13_81787 
batch_normalization_13_81789 
batch_normalization_13_81791
conv2d_21_81796
conv2d_21_81798
conv2d_22_81801
conv2d_22_81803 
batch_normalization_14_81806 
batch_normalization_14_81808 
batch_normalization_14_81810 
batch_normalization_14_81812
conv2d_23_81815
conv2d_23_81817 
batch_normalization_15_81820 
batch_normalization_15_81822 
batch_normalization_15_81824 
batch_normalization_15_81826
conv2d_24_81831
conv2d_24_81833
conv2d_25_81836
conv2d_25_81838 
batch_normalization_16_81841 
batch_normalization_16_81843 
batch_normalization_16_81845 
batch_normalization_16_81847
conv2d_26_81850
conv2d_26_81852 
batch_normalization_17_81855 
batch_normalization_17_81857 
batch_normalization_17_81859 
batch_normalization_17_81861
dense_4_81868
dense_4_81870
dense_5_81873
dense_5_81875
identityЂ.batch_normalization_12/StatefulPartitionedCallЂ.batch_normalization_13/StatefulPartitionedCallЂ.batch_normalization_14/StatefulPartitionedCallЂ.batch_normalization_15/StatefulPartitionedCallЂ.batch_normalization_16/StatefulPartitionedCallЂ.batch_normalization_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallџ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_18_81761conv2d_18_81763*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_794632#
!conv2d_18/StatefulPartitionedCallЃ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_81766conv2d_19_81768*
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_795012#
!conv2d_19/StatefulPartitionedCallЂ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_12_81771batch_normalization_12_81773batch_normalization_12_81775batch_normalization_12_81777*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8058720
.batch_normalization_12/StatefulPartitionedCallА
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_20_81780conv2d_20_81782*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_796642#
!conv2d_20/StatefulPartitionedCallЂ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_13_81785batch_normalization_13_81787batch_normalization_13_81789batch_normalization_13_81791*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_8067620
.batch_normalization_13/StatefulPartitionedCall
add_6/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
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
@__inference_add_6_layer_call_and_return_conditional_losses_807362
add_6/PartitionedCallр
activation_6/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
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
G__inference_activation_6_layer_call_and_return_conditional_losses_807502
activation_6/PartitionedCall
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv2d_21_81796conv2d_21_81798*
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_798282#
!conv2d_21/StatefulPartitionedCallЃ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_81801conv2d_22_81803*
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_798662#
!conv2d_22/StatefulPartitionedCallЂ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_14_81806batch_normalization_14_81808batch_normalization_14_81810batch_normalization_14_81812*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_8079820
.batch_normalization_14/StatefulPartitionedCallА
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_23_81815conv2d_23_81817*
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_800292#
!conv2d_23/StatefulPartitionedCallЂ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_15_81820batch_normalization_15_81822batch_normalization_15_81824batch_normalization_15_81826*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_8088720
.batch_normalization_15/StatefulPartitionedCall
add_7/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*conv2d_21/StatefulPartitionedCall:output:0*
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
@__inference_add_7_layer_call_and_return_conditional_losses_809472
add_7/PartitionedCallр
activation_7/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
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
G__inference_activation_7_layer_call_and_return_conditional_losses_809612
activation_7/PartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_24_81831conv2d_24_81833*
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_801932#
!conv2d_24/StatefulPartitionedCallЃ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_81836conv2d_25_81838*
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_802312#
!conv2d_25/StatefulPartitionedCallЂ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_16_81841batch_normalization_16_81843batch_normalization_16_81845batch_normalization_16_81847*
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_8100920
.batch_normalization_16/StatefulPartitionedCallА
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_26_81850conv2d_26_81852*
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_803942#
!conv2d_26/StatefulPartitionedCallЂ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_17_81855batch_normalization_17_81857batch_normalization_17_81859batch_normalization_17_81861*
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_8109820
.batch_normalization_17/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
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
@__inference_add_8_layer_call_and_return_conditional_losses_811582
add_8/PartitionedCallр
activation_8/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
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
G__inference_activation_8_layer_call_and_return_conditional_losses_811722
activation_8/PartitionedCall
*global_average_pooling2d_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_805372,
*global_average_pooling2d_2/PartitionedCallф
flatten_2/PartitionedCallPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_811872
flatten_2/PartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_81868dense_4_81870*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_812222!
dense_4/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_81873dense_5_81875*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_812652!
dense_5/StatefulPartitionedCallР
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_81761*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addА
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_81763*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/addР
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_81766*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addА
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_81768*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/addР
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_81780*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addА
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_81782*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/addР
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_81796*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addА
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_81798*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/addР
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_81801*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addА
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_81803*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/addР
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_81815*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addА
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_81817*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/addР
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_81831*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addА
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_81833*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/addР
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_81836*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addА
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_81838*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/addР
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_81850*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addА
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_81852*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/addГ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_81868*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЋ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_81870*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addГ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_81873*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЊ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_81875*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЊ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
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


Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80519

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
А
Љ
6__inference_batch_normalization_15_layer_call_fn_84623

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
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_809052
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
$
и
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80798

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
ч$
и
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84326

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
ї
Г
'__inference_model_2_layer_call_fn_83857

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
B__inference_model_2_layer_call_and_return_conditional_losses_824492
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

n
__inference_loss_fn_11_85310=
9conv2d_23_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_23_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/adde
IdentityIdentity"conv2d_23/bias/Regularizer/add:z:0*
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
__inference_loss_fn_0_85167?
;conv2d_18_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_18_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addg
IdentityIdentity$conv2d_18/kernel/Regularizer/add:z:0*
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
і
Њ
B__inference_dense_5_layer_call_and_return_conditional_losses_85145

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
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addМ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/add_
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
ч$
и
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80488

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
с
~
)__inference_conv2d_19_layer_call_fn_79511

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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_795012
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
 
Q
%__inference_add_7_layer_call_fn_84635
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
@__inference_add_7_layer_call_and_return_conditional_losses_809472
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
ы
l
__inference_loss_fn_19_85414;
7dense_4_bias_regularizer_square_readvariableop_resource
identityе
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_4_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addc
IdentityIdentity dense_4/bias/Regularizer/add:z:0*
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_83932

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
е
c
G__inference_activation_6_layer_call_and_return_conditional_losses_80750

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
ж
l
@__inference_add_7_layer_call_and_return_conditional_losses_84629
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
Ш

Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_84025

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
В 
Ќ
D__inference_conv2d_24_layer_call_and_return_conditional_losses_80193

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
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addР
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/add
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
е
c
G__inference_activation_8_layer_call_and_return_conditional_losses_81172

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
Щё

B__inference_model_2_layer_call_and_return_conditional_losses_82449

inputs
conv2d_18_82155
conv2d_18_82157
conv2d_19_82160
conv2d_19_82162 
batch_normalization_12_82165 
batch_normalization_12_82167 
batch_normalization_12_82169 
batch_normalization_12_82171
conv2d_20_82174
conv2d_20_82176 
batch_normalization_13_82179 
batch_normalization_13_82181 
batch_normalization_13_82183 
batch_normalization_13_82185
conv2d_21_82190
conv2d_21_82192
conv2d_22_82195
conv2d_22_82197 
batch_normalization_14_82200 
batch_normalization_14_82202 
batch_normalization_14_82204 
batch_normalization_14_82206
conv2d_23_82209
conv2d_23_82211 
batch_normalization_15_82214 
batch_normalization_15_82216 
batch_normalization_15_82218 
batch_normalization_15_82220
conv2d_24_82225
conv2d_24_82227
conv2d_25_82230
conv2d_25_82232 
batch_normalization_16_82235 
batch_normalization_16_82237 
batch_normalization_16_82239 
batch_normalization_16_82241
conv2d_26_82244
conv2d_26_82246 
batch_normalization_17_82249 
batch_normalization_17_82251 
batch_normalization_17_82253 
batch_normalization_17_82255
dense_4_82262
dense_4_82264
dense_5_82267
dense_5_82269
identityЂ.batch_normalization_12/StatefulPartitionedCallЂ.batch_normalization_13/StatefulPartitionedCallЂ.batch_normalization_14/StatefulPartitionedCallЂ.batch_normalization_15/StatefulPartitionedCallЂ.batch_normalization_16/StatefulPartitionedCallЂ.batch_normalization_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallџ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_18_82155conv2d_18_82157*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_794632#
!conv2d_18/StatefulPartitionedCallЃ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_82160conv2d_19_82162*
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_795012#
!conv2d_19/StatefulPartitionedCallЄ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_12_82165batch_normalization_12_82167batch_normalization_12_82169batch_normalization_12_82171*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8060520
.batch_normalization_12/StatefulPartitionedCallА
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_20_82174conv2d_20_82176*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_796642#
!conv2d_20/StatefulPartitionedCallЄ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_13_82179batch_normalization_13_82181batch_normalization_13_82183batch_normalization_13_82185*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_8069420
.batch_normalization_13/StatefulPartitionedCall
add_6/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
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
@__inference_add_6_layer_call_and_return_conditional_losses_807362
add_6/PartitionedCallр
activation_6/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
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
G__inference_activation_6_layer_call_and_return_conditional_losses_807502
activation_6/PartitionedCall
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv2d_21_82190conv2d_21_82192*
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_798282#
!conv2d_21/StatefulPartitionedCallЃ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_82195conv2d_22_82197*
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_798662#
!conv2d_22/StatefulPartitionedCallЄ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_14_82200batch_normalization_14_82202batch_normalization_14_82204batch_normalization_14_82206*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_8081620
.batch_normalization_14/StatefulPartitionedCallА
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_23_82209conv2d_23_82211*
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_800292#
!conv2d_23/StatefulPartitionedCallЄ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_15_82214batch_normalization_15_82216batch_normalization_15_82218batch_normalization_15_82220*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_8090520
.batch_normalization_15/StatefulPartitionedCall
add_7/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*conv2d_21/StatefulPartitionedCall:output:0*
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
@__inference_add_7_layer_call_and_return_conditional_losses_809472
add_7/PartitionedCallр
activation_7/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
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
G__inference_activation_7_layer_call_and_return_conditional_losses_809612
activation_7/PartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_24_82225conv2d_24_82227*
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_801932#
!conv2d_24/StatefulPartitionedCallЃ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_82230conv2d_25_82232*
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_802312#
!conv2d_25/StatefulPartitionedCallЄ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_16_82235batch_normalization_16_82237batch_normalization_16_82239batch_normalization_16_82241*
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_8102720
.batch_normalization_16/StatefulPartitionedCallА
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_26_82244conv2d_26_82246*
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_803942#
!conv2d_26/StatefulPartitionedCallЄ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_17_82249batch_normalization_17_82251batch_normalization_17_82253batch_normalization_17_82255*
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_8111620
.batch_normalization_17/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
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
@__inference_add_8_layer_call_and_return_conditional_losses_811582
add_8/PartitionedCallр
activation_8/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
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
G__inference_activation_8_layer_call_and_return_conditional_losses_811722
activation_8/PartitionedCall
*global_average_pooling2d_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_805372,
*global_average_pooling2d_2/PartitionedCallф
flatten_2/PartitionedCallPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_811872
flatten_2/PartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_82262dense_4_82264*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_812222!
dense_4/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_82267dense_5_82269*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_812652!
dense_5/StatefulPartitionedCallР
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_82155*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addА
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_82157*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/addР
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_82160*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addА
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_82162*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/addР
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_82174*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addА
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_82176*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/addР
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_82190*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addА
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_82192*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/addР
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_82195*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addА
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_82197*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/addР
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_82209*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addА
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_82211*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/addР
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_82225*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addА
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_82227*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/addР
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_82230*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addА
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_82232*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/addР
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_82244*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addА
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_82246*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/addГ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_82262*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЋ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_82264*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addГ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_82267*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЊ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_82269*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЊ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
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
Ш

Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80605

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
$
и
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80587

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
і
Љ
6__inference_batch_normalization_17_layer_call_fn_84929

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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_804882
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
ч$
и
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80123

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
Єg
Ћ
__inference__traced_save_85605
file_prefix/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
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
value3B1 B+_temp_fadc48efbd28476ca3c836505b7cfde8/part2	
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
SaveV2/shape_and_slicesЫ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop"/device:CPU:0*
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
ї
o
__inference_loss_fn_2_85193?
;conv2d_19_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_19_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addg
IdentityIdentity$conv2d_19/kernel/Regularizer/add:z:0*
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
__inference_loss_fn_17_85388=
9conv2d_26_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_26_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/adde
IdentityIdentity"conv2d_26/bias/Regularizer/add:z:0*
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
__inference_loss_fn_8_85271?
;conv2d_22_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_22_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addg
IdentityIdentity$conv2d_22/kernel/Regularizer/add:z:0*
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
__inference_loss_fn_16_85375?
;conv2d_26_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_26_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addg
IdentityIdentity$conv2d_26/kernel/Regularizer/add:z:0*
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
__inference_loss_fn_13_85336=
9conv2d_24_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_24_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/adde
IdentityIdentity"conv2d_24/bias/Regularizer/add:z:0*
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


Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_83950

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
Ш

Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84991

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
ч$
и
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_79960

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

H
,__inference_activation_6_layer_call_fn_84251

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
G__inference_activation_6_layer_call_and_return_conditional_losses_807502
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
Ў
n
__inference_loss_fn_20_85427=
9dense_5_kernel_regularizer_square_readvariableop_resource
identityп
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_5_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/adde
IdentityIdentity"dense_5/kernel/Regularizer/add:z:0*
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
$
и
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84973

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
А
Љ
6__inference_batch_normalization_14_layer_call_fn_84445

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
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_808162
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
і
Њ
B__inference_dense_5_layer_call_and_return_conditional_losses_81265

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
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addМ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/add_
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
Ю
j
@__inference_add_7_layer_call_and_return_conditional_losses_80947

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
ј
Љ
6__inference_batch_normalization_13_layer_call_fn_84154

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_797892
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
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84344

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

n
__inference_loss_fn_15_85362=
9conv2d_25_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_25_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/adde
IdentityIdentity"conv2d_25/bias/Regularizer/add:z:0*
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_80231

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
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addР
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/add
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
Ю
j
@__inference_add_8_layer_call_and_return_conditional_losses_81158

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

m
__inference_loss_fn_3_85206=
9conv2d_19_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_19_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/adde
IdentityIdentity"conv2d_19/bias/Regularizer/add:z:0*
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
оЦ
І
!__inference__traced_restore_85755
file_prefix%
!assignvariableop_conv2d_18_kernel%
!assignvariableop_1_conv2d_18_bias'
#assignvariableop_2_conv2d_19_kernel%
!assignvariableop_3_conv2d_19_bias3
/assignvariableop_4_batch_normalization_12_gamma2
.assignvariableop_5_batch_normalization_12_beta9
5assignvariableop_6_batch_normalization_12_moving_mean=
9assignvariableop_7_batch_normalization_12_moving_variance'
#assignvariableop_8_conv2d_20_kernel%
!assignvariableop_9_conv2d_20_bias4
0assignvariableop_10_batch_normalization_13_gamma3
/assignvariableop_11_batch_normalization_13_beta:
6assignvariableop_12_batch_normalization_13_moving_mean>
:assignvariableop_13_batch_normalization_13_moving_variance(
$assignvariableop_14_conv2d_21_kernel&
"assignvariableop_15_conv2d_21_bias(
$assignvariableop_16_conv2d_22_kernel&
"assignvariableop_17_conv2d_22_bias4
0assignvariableop_18_batch_normalization_14_gamma3
/assignvariableop_19_batch_normalization_14_beta:
6assignvariableop_20_batch_normalization_14_moving_mean>
:assignvariableop_21_batch_normalization_14_moving_variance(
$assignvariableop_22_conv2d_23_kernel&
"assignvariableop_23_conv2d_23_bias4
0assignvariableop_24_batch_normalization_15_gamma3
/assignvariableop_25_batch_normalization_15_beta:
6assignvariableop_26_batch_normalization_15_moving_mean>
:assignvariableop_27_batch_normalization_15_moving_variance(
$assignvariableop_28_conv2d_24_kernel&
"assignvariableop_29_conv2d_24_bias(
$assignvariableop_30_conv2d_25_kernel&
"assignvariableop_31_conv2d_25_bias4
0assignvariableop_32_batch_normalization_16_gamma3
/assignvariableop_33_batch_normalization_16_beta:
6assignvariableop_34_batch_normalization_16_moving_mean>
:assignvariableop_35_batch_normalization_16_moving_variance(
$assignvariableop_36_conv2d_26_kernel&
"assignvariableop_37_conv2d_26_bias4
0assignvariableop_38_batch_normalization_17_gamma3
/assignvariableop_39_batch_normalization_17_beta:
6assignvariableop_40_batch_normalization_17_moving_mean>
:assignvariableop_41_batch_normalization_17_moving_variance&
"assignvariableop_42_dense_4_kernel$
 assignvariableop_43_dense_4_bias&
"assignvariableop_44_dense_5_kernel$
 assignvariableop_45_dense_5_bias
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

Identity
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_18_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_18_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_19_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_19_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ѕ
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_12_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Є
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_12_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ћ
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_12_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Џ
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_12_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_20_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_20_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Љ
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_13_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ј
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_13_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Џ
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_13_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Г
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_13_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_21_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_21_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_22_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_22_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Љ
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_14_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ј
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_14_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Џ
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_14_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Г
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_14_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_23_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_23_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Љ
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_15_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ј
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_15_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Џ
AssignVariableOp_26AssignVariableOp6assignvariableop_26_batch_normalization_15_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Г
AssignVariableOp_27AssignVariableOp:assignvariableop_27_batch_normalization_15_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_24_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_24_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_25_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_25_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Љ
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_16_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ј
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_16_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Џ
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_16_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Г
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_16_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_26_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_26_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Љ
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_17_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39Ј
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_17_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Џ
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_17_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_17_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_4_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_4_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_5_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_5_biasIdentity_45:output:0*
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


Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_79626

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
с
~
)__inference_conv2d_22_layer_call_fn_79876

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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_798662
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
ј
Љ
6__inference_batch_normalization_14_layer_call_fn_84370

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_799912
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
Њ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_81187

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
Л
Ќ
D__inference_conv2d_26_layer_call_and_return_conditional_losses_80394

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
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addР
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/add~
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


Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84813

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
е
c
G__inference_activation_7_layer_call_and_return_conditional_losses_84640

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
і
Љ
6__inference_batch_normalization_13_layer_call_fn_84141

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_797582
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
А
Љ
6__inference_batch_normalization_12_layer_call_fn_84051

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
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_806052
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
ю
Д
'__inference_model_2_layer_call_fn_82150
input_3
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
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
B__inference_model_2_layer_call_and_return_conditional_losses_820552
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
_user_specified_name	input_3:
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
)__inference_conv2d_20_layer_call_fn_79674

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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_796642
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
Л
q
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_80537

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
 
_user_specified_nameinputs
ј
p
__inference_loss_fn_12_85323?
;conv2d_24_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_24_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addg
IdentityIdentity$conv2d_24/kernel/Regularizer/add:z:0*
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
і
Љ
6__inference_batch_normalization_14_layer_call_fn_84357

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_799602
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
В 
Ќ
D__inference_conv2d_19_layer_call_and_return_conditional_losses_79501

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
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addР
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/add
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
љ
Њ
B__inference_dense_4_layer_call_and_return_conditional_losses_85093

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
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addН
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addg
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
д
А
#__inference_signature_wrapper_82891
input_3
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
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 __inference__wrapped_model_794362
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
_user_specified_name	input_3:
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_79595

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
ч$
и
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84898

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
ч$
и
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_79758

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
Ш

Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84738

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
Ш

Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84597

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
і
Љ
6__inference_batch_normalization_15_layer_call_fn_84535

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_801232
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
э
V
:__inference_global_average_pooling2d_2_layer_call_fn_80543

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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_805372
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
с
~
)__inference_conv2d_26_layer_call_fn_80404

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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_803942
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
ј
Љ
6__inference_batch_normalization_16_layer_call_fn_84839

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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_803562
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
Ю
j
@__inference_add_6_layer_call_and_return_conditional_losses_80736

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
ї
o
__inference_loss_fn_6_85245?
;conv2d_21_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_21_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addg
IdentityIdentity$conv2d_21/kernel/Regularizer/add:z:0*
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

H
,__inference_activation_8_layer_call_fn_85039

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
G__inference_activation_8_layer_call_and_return_conditional_losses_811722
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
Ў
Љ
6__inference_batch_normalization_14_layer_call_fn_84432

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
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_807982
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
ЪЫ
ы
 __inference__wrapped_model_79436
input_34
0model_2_conv2d_18_conv2d_readvariableop_resource5
1model_2_conv2d_18_biasadd_readvariableop_resource4
0model_2_conv2d_19_conv2d_readvariableop_resource5
1model_2_conv2d_19_biasadd_readvariableop_resource:
6model_2_batch_normalization_12_readvariableop_resource<
8model_2_batch_normalization_12_readvariableop_1_resourceK
Gmodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_20_conv2d_readvariableop_resource5
1model_2_conv2d_20_biasadd_readvariableop_resource:
6model_2_batch_normalization_13_readvariableop_resource<
8model_2_batch_normalization_13_readvariableop_1_resourceK
Gmodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_21_conv2d_readvariableop_resource5
1model_2_conv2d_21_biasadd_readvariableop_resource4
0model_2_conv2d_22_conv2d_readvariableop_resource5
1model_2_conv2d_22_biasadd_readvariableop_resource:
6model_2_batch_normalization_14_readvariableop_resource<
8model_2_batch_normalization_14_readvariableop_1_resourceK
Gmodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_23_conv2d_readvariableop_resource5
1model_2_conv2d_23_biasadd_readvariableop_resource:
6model_2_batch_normalization_15_readvariableop_resource<
8model_2_batch_normalization_15_readvariableop_1_resourceK
Gmodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_24_conv2d_readvariableop_resource5
1model_2_conv2d_24_biasadd_readvariableop_resource4
0model_2_conv2d_25_conv2d_readvariableop_resource5
1model_2_conv2d_25_biasadd_readvariableop_resource:
6model_2_batch_normalization_16_readvariableop_resource<
8model_2_batch_normalization_16_readvariableop_1_resourceK
Gmodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource4
0model_2_conv2d_26_conv2d_readvariableop_resource5
1model_2_conv2d_26_biasadd_readvariableop_resource:
6model_2_batch_normalization_17_readvariableop_resource<
8model_2_batch_normalization_17_readvariableop_1_resourceK
Gmodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceM
Imodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource2
.model_2_dense_4_matmul_readvariableop_resource3
/model_2_dense_4_biasadd_readvariableop_resource2
.model_2_dense_5_matmul_readvariableop_resource3
/model_2_dense_5_biasadd_readvariableop_resource
identityЫ
'model_2/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_2/conv2d_18/Conv2D/ReadVariableOpк
model_2/conv2d_18/Conv2DConv2Dinput_3/model_2/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_2/conv2d_18/Conv2DТ
(model_2/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/conv2d_18/BiasAdd/ReadVariableOpа
model_2/conv2d_18/BiasAddBiasAdd!model_2/conv2d_18/Conv2D:output:00model_2/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_2/conv2d_18/BiasAddЫ
'model_2/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_2/conv2d_19/Conv2D/ReadVariableOpѕ
model_2/conv2d_19/Conv2DConv2D"model_2/conv2d_18/BiasAdd:output:0/model_2/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_2/conv2d_19/Conv2DТ
(model_2/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/conv2d_19/BiasAdd/ReadVariableOpа
model_2/conv2d_19/BiasAddBiasAdd!model_2/conv2d_19/Conv2D:output:00model_2/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_2/conv2d_19/BiasAdd
model_2/conv2d_19/ReluRelu"model_2/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_2/conv2d_19/Reluб
-model_2/batch_normalization_12/ReadVariableOpReadVariableOp6model_2_batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_2/batch_normalization_12/ReadVariableOpз
/model_2/batch_normalization_12/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_2/batch_normalization_12/ReadVariableOp_1
>model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
@model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ђ
/model_2/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3$model_2/conv2d_19/Relu:activations:05model_2/batch_normalization_12/ReadVariableOp:value:07model_2/batch_normalization_12/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 21
/model_2/batch_normalization_12/FusedBatchNormV3Ы
'model_2/conv2d_20/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_2/conv2d_20/Conv2D/ReadVariableOp
model_2/conv2d_20/Conv2DConv2D3model_2/batch_normalization_12/FusedBatchNormV3:y:0/model_2/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_2/conv2d_20/Conv2DТ
(model_2/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/conv2d_20/BiasAdd/ReadVariableOpа
model_2/conv2d_20/BiasAddBiasAdd!model_2/conv2d_20/Conv2D:output:00model_2/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_2/conv2d_20/BiasAddб
-model_2/batch_normalization_13/ReadVariableOpReadVariableOp6model_2_batch_normalization_13_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_2/batch_normalization_13/ReadVariableOpз
/model_2/batch_normalization_13/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_2/batch_normalization_13/ReadVariableOp_1
>model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
@model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1 
/model_2/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_20/BiasAdd:output:05model_2/batch_normalization_13/ReadVariableOp:value:07model_2/batch_normalization_13/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 21
/model_2/batch_normalization_13/FusedBatchNormV3Т
model_2/add_6/addAddV23model_2/batch_normalization_13/FusedBatchNormV3:y:0"model_2/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_2/add_6/add
model_2/activation_6/ReluRelumodel_2/add_6/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_2/activation_6/ReluЫ
'model_2/conv2d_21/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_2/conv2d_21/Conv2D/ReadVariableOpњ
model_2/conv2d_21/Conv2DConv2D'model_2/activation_6/Relu:activations:0/model_2/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_2/conv2d_21/Conv2DТ
(model_2/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_21/BiasAdd/ReadVariableOpа
model_2/conv2d_21/BiasAddBiasAdd!model_2/conv2d_21/Conv2D:output:00model_2/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/conv2d_21/BiasAdd
model_2/conv2d_21/ReluRelu"model_2/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/conv2d_21/ReluЫ
'model_2/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_22/Conv2D/ReadVariableOpї
model_2/conv2d_22/Conv2DConv2D$model_2/conv2d_21/Relu:activations:0/model_2/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_2/conv2d_22/Conv2DТ
(model_2/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_22/BiasAdd/ReadVariableOpа
model_2/conv2d_22/BiasAddBiasAdd!model_2/conv2d_22/Conv2D:output:00model_2/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/conv2d_22/BiasAdd
model_2/conv2d_22/ReluRelu"model_2/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/conv2d_22/Reluб
-model_2/batch_normalization_14/ReadVariableOpReadVariableOp6model_2_batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_14/ReadVariableOpз
/model_2/batch_normalization_14/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_14/ReadVariableOp_1
>model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp
@model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Ђ
/model_2/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3$model_2/conv2d_22/Relu:activations:05model_2/batch_normalization_14/ReadVariableOp:value:07model_2/batch_normalization_14/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 21
/model_2/batch_normalization_14/FusedBatchNormV3Ы
'model_2/conv2d_23/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_2/conv2d_23/Conv2D/ReadVariableOp
model_2/conv2d_23/Conv2DConv2D3model_2/batch_normalization_14/FusedBatchNormV3:y:0/model_2/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_2/conv2d_23/Conv2DТ
(model_2/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_2/conv2d_23/BiasAdd/ReadVariableOpа
model_2/conv2d_23/BiasAddBiasAdd!model_2/conv2d_23/Conv2D:output:00model_2/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/conv2d_23/BiasAddб
-model_2/batch_normalization_15/ReadVariableOpReadVariableOp6model_2_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_2/batch_normalization_15/ReadVariableOpз
/model_2/batch_normalization_15/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_2/batch_normalization_15/ReadVariableOp_1
>model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp
@model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1 
/model_2/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_23/BiasAdd:output:05model_2/batch_normalization_15/ReadVariableOp:value:07model_2/batch_normalization_15/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 21
/model_2/batch_normalization_15/FusedBatchNormV3Ф
model_2/add_7/addAddV23model_2/batch_normalization_15/FusedBatchNormV3:y:0$model_2/conv2d_21/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/add_7/add
model_2/activation_7/ReluRelumodel_2/add_7/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_2/activation_7/ReluЫ
'model_2/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_2/conv2d_24/Conv2D/ReadVariableOpњ
model_2/conv2d_24/Conv2DConv2D'model_2/activation_7/Relu:activations:0/model_2/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_2/conv2d_24/Conv2DТ
(model_2/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_2/conv2d_24/BiasAdd/ReadVariableOpа
model_2/conv2d_24/BiasAddBiasAdd!model_2/conv2d_24/Conv2D:output:00model_2/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/conv2d_24/BiasAdd
model_2/conv2d_24/ReluRelu"model_2/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/conv2d_24/ReluЫ
'model_2/conv2d_25/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_2/conv2d_25/Conv2D/ReadVariableOpї
model_2/conv2d_25/Conv2DConv2D$model_2/conv2d_24/Relu:activations:0/model_2/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_2/conv2d_25/Conv2DТ
(model_2/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_2/conv2d_25/BiasAdd/ReadVariableOpа
model_2/conv2d_25/BiasAddBiasAdd!model_2/conv2d_25/Conv2D:output:00model_2/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/conv2d_25/BiasAdd
model_2/conv2d_25/ReluRelu"model_2/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/conv2d_25/Reluб
-model_2/batch_normalization_16/ReadVariableOpReadVariableOp6model_2_batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_2/batch_normalization_16/ReadVariableOpз
/model_2/batch_normalization_16/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_2/batch_normalization_16/ReadVariableOp_1
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Ђ
/model_2/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3$model_2/conv2d_25/Relu:activations:05model_2/batch_normalization_16/ReadVariableOp:value:07model_2/batch_normalization_16/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 21
/model_2/batch_normalization_16/FusedBatchNormV3Ы
'model_2/conv2d_26/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_2/conv2d_26/Conv2D/ReadVariableOp
model_2/conv2d_26/Conv2DConv2D3model_2/batch_normalization_16/FusedBatchNormV3:y:0/model_2/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_2/conv2d_26/Conv2DТ
(model_2/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_2/conv2d_26/BiasAdd/ReadVariableOpа
model_2/conv2d_26/BiasAddBiasAdd!model_2/conv2d_26/Conv2D:output:00model_2/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/conv2d_26/BiasAddб
-model_2/batch_normalization_17/ReadVariableOpReadVariableOp6model_2_batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_2/batch_normalization_17/ReadVariableOpз
/model_2/batch_normalization_17/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_2/batch_normalization_17/ReadVariableOp_1
>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp
@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1 
/model_2/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_26/BiasAdd:output:05model_2/batch_normalization_17/ReadVariableOp:value:07model_2/batch_normalization_17/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 21
/model_2/batch_normalization_17/FusedBatchNormV3Ф
model_2/add_8/addAddV23model_2/batch_normalization_17/FusedBatchNormV3:y:0$model_2/conv2d_24/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/add_8/add
model_2/activation_8/ReluRelumodel_2/add_8/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_2/activation_8/ReluЧ
9model_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_2/global_average_pooling2d_2/Mean/reduction_indicesљ
'model_2/global_average_pooling2d_2/MeanMean'model_2/activation_8/Relu:activations:0Bmodel_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2)
'model_2/global_average_pooling2d_2/Mean
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
model_2/flatten_2/ConstЧ
model_2/flatten_2/ReshapeReshape0model_2/global_average_pooling2d_2/Mean:output:0 model_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_2/flatten_2/ReshapeО
%model_2/dense_4/MatMul/ReadVariableOpReadVariableOp.model_2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02'
%model_2/dense_4/MatMul/ReadVariableOpР
model_2/dense_4/MatMulMatMul"model_2/flatten_2/Reshape:output:0-model_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_2/dense_4/MatMulН
&model_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&model_2/dense_4/BiasAdd/ReadVariableOpТ
model_2/dense_4/BiasAddBiasAdd model_2/dense_4/MatMul:product:0.model_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_2/dense_4/BiasAdd
model_2/dense_4/ReluRelu model_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_2/dense_4/ReluО
%model_2/dense_5/MatMul/ReadVariableOpReadVariableOp.model_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%model_2/dense_5/MatMul/ReadVariableOpП
model_2/dense_5/MatMulMatMul"model_2/dense_4/Relu:activations:0-model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_5/MatMulМ
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_2/dense_5/BiasAdd/ReadVariableOpС
model_2/dense_5/BiasAddBiasAdd model_2/dense_5/MatMul:product:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_5/BiasAdd
model_2/dense_5/SigmoidSigmoid model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_5/Sigmoido
IdentityIdentitymodel_2/dense_5/Sigmoid:y:0*
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
_user_specified_name	input_3:
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_81098

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
$
и
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_84007

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
ж
l
@__inference_add_6_layer_call_and_return_conditional_losses_84235
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
е
c
G__inference_activation_6_layer_call_and_return_conditional_losses_84246

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
с
~
)__inference_conv2d_24_layer_call_fn_80203

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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_801932
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
ч$
и
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80325

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
ј
p
__inference_loss_fn_10_85297?
;conv2d_23_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_23_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addg
IdentityIdentity$conv2d_23/kernel/Regularizer/add:z:0*
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
яћ

B__inference_model_2_layer_call_and_return_conditional_losses_83316

inputs,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource2
.batch_normalization_14_readvariableop_resource4
0batch_normalization_14_readvariableop_1_resourceC
?batch_normalization_14_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resourceC
?batch_normalization_16_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resourceC
?batch_normalization_17_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityЂ:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_18/Conv2D/ReadVariableOpС
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_18/Conv2DЊ
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOpА
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_18/BiasAddГ
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_19/Conv2D/ReadVariableOpе
conv2d_19/Conv2DConv2Dconv2d_18/BiasAdd:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_19/Conv2DЊ
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOpА
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_19/ReluЙ
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_12/ReadVariableOpП
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_12/ReadVariableOp_1ь
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1з
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_19/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2)
'batch_normalization_12/FusedBatchNormV3
batch_normalization_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_12/Constѕ
,batch_normalization_12/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_12/AssignMovingAvg/sub/xВ
*batch_normalization_12/AssignMovingAvg/subSub5batch_normalization_12/AssignMovingAvg/sub/x:output:0%batch_normalization_12/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_12/AssignMovingAvg/subъ
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOpб
,batch_normalization_12/AssignMovingAvg/sub_1Sub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_12/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg/sub_1К
*batch_normalization_12/AssignMovingAvg/mulMul0batch_normalization_12/AssignMovingAvg/sub_1:z:0.batch_normalization_12/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/mulш
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_12/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_12/AssignMovingAvg_1/sub/xК
,batch_normalization_12/AssignMovingAvg_1/subSub7batch_normalization_12/AssignMovingAvg_1/sub/x:output:0%batch_normalization_12/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_12/AssignMovingAvg_1/sub№
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_12/AssignMovingAvg_1/sub_1Sub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_12/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:20
.batch_normalization_12/AssignMovingAvg_1/sub_1Ф
,batch_normalization_12/AssignMovingAvg_1/mulMul2batch_normalization_12/AssignMovingAvg_1/sub_1:z:00batch_normalization_12/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/mulі
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpц
conv2d_20/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_20/Conv2DЊ
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOpА
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_20/BiasAddЙ
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_13/ReadVariableOpП
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1ь
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1е
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2)
'batch_normalization_13/FusedBatchNormV3
batch_normalization_13/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_13/Constѕ
,batch_normalization_13/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_13/AssignMovingAvg/sub/xВ
*batch_normalization_13/AssignMovingAvg/subSub5batch_normalization_13/AssignMovingAvg/sub/x:output:0%batch_normalization_13/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_13/AssignMovingAvg/subъ
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOpб
,batch_normalization_13/AssignMovingAvg/sub_1Sub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_13/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg/sub_1К
*batch_normalization_13/AssignMovingAvg/mulMul0batch_normalization_13/AssignMovingAvg/sub_1:z:0.batch_normalization_13/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/mulш
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp7^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_13/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_13/AssignMovingAvg_1/sub/xК
,batch_normalization_13/AssignMovingAvg_1/subSub7batch_normalization_13/AssignMovingAvg_1/sub/x:output:0%batch_normalization_13/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_13/AssignMovingAvg_1/sub№
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_13/AssignMovingAvg_1/sub_1Sub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_13/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:20
.batch_normalization_13/AssignMovingAvg_1/sub_1Ф
,batch_normalization_13/AssignMovingAvg_1/mulMul2batch_normalization_13/AssignMovingAvg_1/sub_1:z:00batch_normalization_13/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/mulі
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpЂ
	add_6/addAddV2+batch_normalization_13/FusedBatchNormV3:y:0conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
	add_6/addw
activation_6/ReluReluadd_6/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
activation_6/ReluГ
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpк
conv2d_21/Conv2DConv2Dactivation_6/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_21/Conv2DЊ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOpА
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_21/ReluГ
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_22/Conv2D/ReadVariableOpз
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_22/Conv2DЊ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOpА
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_22/ReluЙ
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_14/ReadVariableOpП
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_14/ReadVariableOp_1ь
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1з
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_22/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2)
'batch_normalization_14/FusedBatchNormV3
batch_normalization_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_14/Constѕ
,batch_normalization_14/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_14/AssignMovingAvg/sub/xВ
*batch_normalization_14/AssignMovingAvg/subSub5batch_normalization_14/AssignMovingAvg/sub/x:output:0%batch_normalization_14/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_14/AssignMovingAvg/subъ
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_14/AssignMovingAvg/ReadVariableOpб
,batch_normalization_14/AssignMovingAvg/sub_1Sub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_14/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_14/AssignMovingAvg/sub_1К
*batch_normalization_14/AssignMovingAvg/mulMul0batch_normalization_14/AssignMovingAvg/sub_1:z:0.batch_normalization_14/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_14/AssignMovingAvg/mulш
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp7^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_14/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_14/AssignMovingAvg_1/sub/xК
,batch_normalization_14/AssignMovingAvg_1/subSub7batch_normalization_14/AssignMovingAvg_1/sub/x:output:0%batch_normalization_14/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_14/AssignMovingAvg_1/sub№
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_14/AssignMovingAvg_1/sub_1Sub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_14/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_14/AssignMovingAvg_1/sub_1Ф
,batch_normalization_14/AssignMovingAvg_1/mulMul2batch_normalization_14/AssignMovingAvg_1/sub_1:z:00batch_normalization_14/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_14/AssignMovingAvg_1/mulі
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_23/Conv2D/ReadVariableOpц
conv2d_23/Conv2DConv2D+batch_normalization_14/FusedBatchNormV3:y:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_23/Conv2DЊ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOpА
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_23/BiasAddЙ
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOpП
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1ь
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1е
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_23/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2)
'batch_normalization_15/FusedBatchNormV3
batch_normalization_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_15/Constѕ
,batch_normalization_15/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_15/AssignMovingAvg/sub/xВ
*batch_normalization_15/AssignMovingAvg/subSub5batch_normalization_15/AssignMovingAvg/sub/x:output:0%batch_normalization_15/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_15/AssignMovingAvg/subъ
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOpб
,batch_normalization_15/AssignMovingAvg/sub_1Sub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_15/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg/sub_1К
*batch_normalization_15/AssignMovingAvg/mulMul0batch_normalization_15/AssignMovingAvg/sub_1:z:0.batch_normalization_15/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_15/AssignMovingAvg/mulш
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp7^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_15/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_15/AssignMovingAvg_1/sub/xК
,batch_normalization_15/AssignMovingAvg_1/subSub7batch_normalization_15/AssignMovingAvg_1/sub/x:output:0%batch_normalization_15/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg_1/sub№
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_15/AssignMovingAvg_1/sub_1Sub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_15/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_15/AssignMovingAvg_1/sub_1Ф
,batch_normalization_15/AssignMovingAvg_1/mulMul2batch_normalization_15/AssignMovingAvg_1/sub_1:z:00batch_normalization_15/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg_1/mulі
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpЄ
	add_7/addAddV2+batch_normalization_15/FusedBatchNormV3:y:0conv2d_21/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
	add_7/addw
activation_7/ReluReluadd_7/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
activation_7/ReluГ
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_24/Conv2D/ReadVariableOpк
conv2d_24/Conv2DConv2Dactivation_7/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_24/Conv2DЊ
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOpА
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_24/ReluГ
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_25/Conv2D/ReadVariableOpз
conv2d_25/Conv2DConv2Dconv2d_24/Relu:activations:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_25/Conv2DЊ
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOpА
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_25/ReluЙ
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOpП
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1ь
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1з
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_25/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2)
'batch_normalization_16/FusedBatchNormV3
batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_16/Constѕ
,batch_normalization_16/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_16/AssignMovingAvg/sub/xВ
*batch_normalization_16/AssignMovingAvg/subSub5batch_normalization_16/AssignMovingAvg/sub/x:output:0%batch_normalization_16/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_16/AssignMovingAvg/subъ
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOpб
,batch_normalization_16/AssignMovingAvg/sub_1Sub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_16/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_16/AssignMovingAvg/sub_1К
*batch_normalization_16/AssignMovingAvg/mulMul0batch_normalization_16/AssignMovingAvg/sub_1:z:0.batch_normalization_16/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_16/AssignMovingAvg/mulш
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp7^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_16/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_16/AssignMovingAvg_1/sub/xК
,batch_normalization_16/AssignMovingAvg_1/subSub7batch_normalization_16/AssignMovingAvg_1/sub/x:output:0%batch_normalization_16/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_16/AssignMovingAvg_1/sub№
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_16/AssignMovingAvg_1/sub_1Sub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_16/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_16/AssignMovingAvg_1/sub_1Ф
,batch_normalization_16/AssignMovingAvg_1/mulMul2batch_normalization_16/AssignMovingAvg_1/sub_1:z:00batch_normalization_16/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_16/AssignMovingAvg_1/mulі
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_26/Conv2D/ReadVariableOpц
conv2d_26/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_26/Conv2DЊ
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOpА
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_26/BiasAddЙ
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOpП
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1ь
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1е
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_26/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2)
'batch_normalization_17/FusedBatchNormV3
batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_17/Constѕ
,batch_normalization_17/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_17/AssignMovingAvg/sub/xВ
*batch_normalization_17/AssignMovingAvg/subSub5batch_normalization_17/AssignMovingAvg/sub/x:output:0%batch_normalization_17/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_17/AssignMovingAvg/subъ
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOpб
,batch_normalization_17/AssignMovingAvg/sub_1Sub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_17/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg/sub_1К
*batch_normalization_17/AssignMovingAvg/mulMul0batch_normalization_17/AssignMovingAvg/sub_1:z:0.batch_normalization_17/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_17/AssignMovingAvg/mulш
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp7^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_17/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_17/AssignMovingAvg_1/sub/xК
,batch_normalization_17/AssignMovingAvg_1/subSub7batch_normalization_17/AssignMovingAvg_1/sub/x:output:0%batch_normalization_17/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_17/AssignMovingAvg_1/sub№
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_17/AssignMovingAvg_1/sub_1Sub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_17/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_17/AssignMovingAvg_1/sub_1Ф
,batch_normalization_17/AssignMovingAvg_1/mulMul2batch_normalization_17/AssignMovingAvg_1/sub_1:z:00batch_normalization_17/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg_1/mulі
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpЄ
	add_8/addAddV2+batch_normalization_17/FusedBatchNormV3:y:0conv2d_24/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
	add_8/addw
activation_8/ReluReluadd_8/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
activation_8/ReluЗ
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_2/Mean/reduction_indicesй
global_average_pooling2d_2/MeanMeanactivation_8/Relu:activations:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling2d_2/Means
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
flatten_2/ConstЇ
flatten_2/ReshapeReshape(global_average_pooling2d_2/Mean:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten_2/ReshapeІ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЅ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЂ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/ReluІ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/Sigmoidй
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addЪ
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/addй
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addЪ
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/addй
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addЪ
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/addй
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addЪ
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/addй
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addЪ
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/addй
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addЪ
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/addй
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addЪ
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/addй
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addЪ
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/addй
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addЪ
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/addЬ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addХ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addЬ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addФ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЯ
IdentityIdentitydense_5/Sigmoid:y:0;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_14/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp:W S
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
Ў
Љ
6__inference_batch_normalization_17_layer_call_fn_85004

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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_810982
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
ј
Љ
6__inference_batch_normalization_12_layer_call_fn_83976

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_796262
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
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84128

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
Ш

Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80816

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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_79828

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
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addР
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/add
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
щ
l
__inference_loss_fn_21_85440;
7dense_5_bias_regularizer_square_readvariableop_resource
identityд
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_5_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addc
IdentityIdentity dense_5/bias/Regularizer/add:z:0*
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
__inference_loss_fn_7_85258=
9conv2d_21_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_21_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/adde
IdentityIdentity"conv2d_21/bias/Regularizer/add:z:0*
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
Ш

Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_81116

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
ж
l
@__inference_add_8_layer_call_and_return_conditional_losses_85023
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
е
c
G__inference_activation_7_layer_call_and_return_conditional_losses_80961

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
$
и
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84401

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

m
__inference_loss_fn_1_85180=
9conv2d_18_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_18_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/adde
IdentityIdentity"conv2d_18/bias/Regularizer/add:z:0*
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
Ш

Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80694

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
с
~
)__inference_conv2d_23_layer_call_fn_80039

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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_800292
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

H
,__inference_activation_7_layer_call_fn_84645

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
G__inference_activation_7_layer_call_and_return_conditional_losses_809612
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
$
и
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80676

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

m
__inference_loss_fn_9_85284=
9conv2d_22_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_22_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/adde
IdentityIdentity"conv2d_22/bias/Regularizer/add:z:0*
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
Л
Ќ
D__inference_conv2d_20_layer_call_and_return_conditional_losses_79664

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
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addР
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/add~
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
В 
Ќ
D__inference_conv2d_22_layer_call_and_return_conditional_losses_79866

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
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addР
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/add
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
Л
Ќ
D__inference_conv2d_18_layer_call_and_return_conditional_losses_79463

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
BiasAddЯ
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addР
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/add~
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
ј
Љ
6__inference_batch_normalization_15_layer_call_fn_84548

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_801542
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
$
и
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_81009

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
$
и
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84185

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
Ьё

B__inference_model_2_layer_call_and_return_conditional_losses_81755
input_3
conv2d_18_81461
conv2d_18_81463
conv2d_19_81466
conv2d_19_81468 
batch_normalization_12_81471 
batch_normalization_12_81473 
batch_normalization_12_81475 
batch_normalization_12_81477
conv2d_20_81480
conv2d_20_81482 
batch_normalization_13_81485 
batch_normalization_13_81487 
batch_normalization_13_81489 
batch_normalization_13_81491
conv2d_21_81496
conv2d_21_81498
conv2d_22_81501
conv2d_22_81503 
batch_normalization_14_81506 
batch_normalization_14_81508 
batch_normalization_14_81510 
batch_normalization_14_81512
conv2d_23_81515
conv2d_23_81517 
batch_normalization_15_81520 
batch_normalization_15_81522 
batch_normalization_15_81524 
batch_normalization_15_81526
conv2d_24_81531
conv2d_24_81533
conv2d_25_81536
conv2d_25_81538 
batch_normalization_16_81541 
batch_normalization_16_81543 
batch_normalization_16_81545 
batch_normalization_16_81547
conv2d_26_81550
conv2d_26_81552 
batch_normalization_17_81555 
batch_normalization_17_81557 
batch_normalization_17_81559 
batch_normalization_17_81561
dense_4_81568
dense_4_81570
dense_5_81573
dense_5_81575
identityЂ.batch_normalization_12/StatefulPartitionedCallЂ.batch_normalization_13/StatefulPartitionedCallЂ.batch_normalization_14/StatefulPartitionedCallЂ.batch_normalization_15/StatefulPartitionedCallЂ.batch_normalization_16/StatefulPartitionedCallЂ.batch_normalization_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCall
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_18_81461conv2d_18_81463*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_794632#
!conv2d_18/StatefulPartitionedCallЃ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_81466conv2d_19_81468*
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_795012#
!conv2d_19/StatefulPartitionedCallЄ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_12_81471batch_normalization_12_81473batch_normalization_12_81475batch_normalization_12_81477*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8060520
.batch_normalization_12/StatefulPartitionedCallА
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_20_81480conv2d_20_81482*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_796642#
!conv2d_20/StatefulPartitionedCallЄ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_13_81485batch_normalization_13_81487batch_normalization_13_81489batch_normalization_13_81491*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_8069420
.batch_normalization_13/StatefulPartitionedCall
add_6/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
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
@__inference_add_6_layer_call_and_return_conditional_losses_807362
add_6/PartitionedCallр
activation_6/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
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
G__inference_activation_6_layer_call_and_return_conditional_losses_807502
activation_6/PartitionedCall
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv2d_21_81496conv2d_21_81498*
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_798282#
!conv2d_21/StatefulPartitionedCallЃ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_81501conv2d_22_81503*
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_798662#
!conv2d_22/StatefulPartitionedCallЄ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_14_81506batch_normalization_14_81508batch_normalization_14_81510batch_normalization_14_81512*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_8081620
.batch_normalization_14/StatefulPartitionedCallА
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_23_81515conv2d_23_81517*
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_800292#
!conv2d_23/StatefulPartitionedCallЄ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_15_81520batch_normalization_15_81522batch_normalization_15_81524batch_normalization_15_81526*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_8090520
.batch_normalization_15/StatefulPartitionedCall
add_7/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*conv2d_21/StatefulPartitionedCall:output:0*
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
@__inference_add_7_layer_call_and_return_conditional_losses_809472
add_7/PartitionedCallр
activation_7/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
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
G__inference_activation_7_layer_call_and_return_conditional_losses_809612
activation_7/PartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_24_81531conv2d_24_81533*
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_801932#
!conv2d_24/StatefulPartitionedCallЃ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_81536conv2d_25_81538*
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_802312#
!conv2d_25/StatefulPartitionedCallЄ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_16_81541batch_normalization_16_81543batch_normalization_16_81545batch_normalization_16_81547*
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_8102720
.batch_normalization_16/StatefulPartitionedCallА
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_26_81550conv2d_26_81552*
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_803942#
!conv2d_26/StatefulPartitionedCallЄ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_17_81555batch_normalization_17_81557batch_normalization_17_81559batch_normalization_17_81561*
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_8111620
.batch_normalization_17/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
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
@__inference_add_8_layer_call_and_return_conditional_losses_811582
add_8/PartitionedCallр
activation_8/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
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
G__inference_activation_8_layer_call_and_return_conditional_losses_811722
activation_8/PartitionedCall
*global_average_pooling2d_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_805372,
*global_average_pooling2d_2/PartitionedCallф
flatten_2/PartitionedCallPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_811872
flatten_2/PartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_81568dense_4_81570*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_812222!
dense_4/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_81573dense_5_81575*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_812652!
dense_5/StatefulPartitionedCallР
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_81461*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addА
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_81463*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/addР
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_81466*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addА
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_81468*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/addР
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_81480*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addА
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_81482*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/addР
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_81496*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addА
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_81498*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/addР
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_81501*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addА
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_81503*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/addР
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_81515*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addА
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_81517*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/addР
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_81531*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addА
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_81533*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/addР
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_81536*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addА
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_81538*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/addР
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_81550*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addА
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_81552*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/addГ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_81568*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЋ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_81570*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addГ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_81573*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЊ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_81575*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЊ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_3:
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
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84110

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
Ш

Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84203

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
 
Q
%__inference_add_8_layer_call_fn_85029
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
@__inference_add_8_layer_call_and_return_conditional_losses_811582
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
с
~
)__inference_conv2d_18_layer_call_fn_79473

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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_794632
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
ї
|
'__inference_dense_4_layer_call_fn_85102

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
B__inference_dense_4_layer_call_and_return_conditional_losses_812222
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
ј
p
__inference_loss_fn_14_85349?
;conv2d_25_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_25_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addg
IdentityIdentity$conv2d_25/kernel/Regularizer/add:z:0*
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
%__inference_add_6_layer_call_fn_84241
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
@__inference_add_6_layer_call_and_return_conditional_losses_807362
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
Ш

Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84419

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
і
Љ
6__inference_batch_normalization_16_layer_call_fn_84826

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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_803252
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
Л
Ќ
D__inference_conv2d_23_layer_call_and_return_conditional_losses_80029

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
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addР
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/add~
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
е
c
G__inference_activation_8_layer_call_and_return_conditional_losses_85034

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
ы
Г
'__inference_model_2_layer_call_fn_83760

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
B__inference_model_2_layer_call_and_return_conditional_losses_820552
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
$
и
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80887

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
Рё

B__inference_model_2_layer_call_and_return_conditional_losses_81458
input_3
conv2d_18_80547
conv2d_18_80549
conv2d_19_80552
conv2d_19_80554 
batch_normalization_12_80632 
batch_normalization_12_80634 
batch_normalization_12_80636 
batch_normalization_12_80638
conv2d_20_80641
conv2d_20_80643 
batch_normalization_13_80721 
batch_normalization_13_80723 
batch_normalization_13_80725 
batch_normalization_13_80727
conv2d_21_80758
conv2d_21_80760
conv2d_22_80763
conv2d_22_80765 
batch_normalization_14_80843 
batch_normalization_14_80845 
batch_normalization_14_80847 
batch_normalization_14_80849
conv2d_23_80852
conv2d_23_80854 
batch_normalization_15_80932 
batch_normalization_15_80934 
batch_normalization_15_80936 
batch_normalization_15_80938
conv2d_24_80969
conv2d_24_80971
conv2d_25_80974
conv2d_25_80976 
batch_normalization_16_81054 
batch_normalization_16_81056 
batch_normalization_16_81058 
batch_normalization_16_81060
conv2d_26_81063
conv2d_26_81065 
batch_normalization_17_81143 
batch_normalization_17_81145 
batch_normalization_17_81147 
batch_normalization_17_81149
dense_4_81233
dense_4_81235
dense_5_81276
dense_5_81278
identityЂ.batch_normalization_12/StatefulPartitionedCallЂ.batch_normalization_13/StatefulPartitionedCallЂ.batch_normalization_14/StatefulPartitionedCallЂ.batch_normalization_15/StatefulPartitionedCallЂ.batch_normalization_16/StatefulPartitionedCallЂ.batch_normalization_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂ!conv2d_24/StatefulPartitionedCallЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_26/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCall
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_18_80547conv2d_18_80549*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_794632#
!conv2d_18/StatefulPartitionedCallЃ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_80552conv2d_19_80554*
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_795012#
!conv2d_19/StatefulPartitionedCallЂ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_12_80632batch_normalization_12_80634batch_normalization_12_80636batch_normalization_12_80638*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8058720
.batch_normalization_12/StatefulPartitionedCallА
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_20_80641conv2d_20_80643*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_796642#
!conv2d_20/StatefulPartitionedCallЂ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_13_80721batch_normalization_13_80723batch_normalization_13_80725batch_normalization_13_80727*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_8067620
.batch_normalization_13/StatefulPartitionedCall
add_6/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
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
@__inference_add_6_layer_call_and_return_conditional_losses_807362
add_6/PartitionedCallр
activation_6/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
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
G__inference_activation_6_layer_call_and_return_conditional_losses_807502
activation_6/PartitionedCall
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv2d_21_80758conv2d_21_80760*
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_798282#
!conv2d_21/StatefulPartitionedCallЃ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_80763conv2d_22_80765*
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_798662#
!conv2d_22/StatefulPartitionedCallЂ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_14_80843batch_normalization_14_80845batch_normalization_14_80847batch_normalization_14_80849*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_8079820
.batch_normalization_14/StatefulPartitionedCallА
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0conv2d_23_80852conv2d_23_80854*
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_800292#
!conv2d_23/StatefulPartitionedCallЂ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_15_80932batch_normalization_15_80934batch_normalization_15_80936batch_normalization_15_80938*
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
GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_8088720
.batch_normalization_15/StatefulPartitionedCall
add_7/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*conv2d_21/StatefulPartitionedCall:output:0*
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
@__inference_add_7_layer_call_and_return_conditional_losses_809472
add_7/PartitionedCallр
activation_7/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
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
G__inference_activation_7_layer_call_and_return_conditional_losses_809612
activation_7/PartitionedCall
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_24_80969conv2d_24_80971*
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_801932#
!conv2d_24/StatefulPartitionedCallЃ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_80974conv2d_25_80976*
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_802312#
!conv2d_25/StatefulPartitionedCallЂ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_16_81054batch_normalization_16_81056batch_normalization_16_81058batch_normalization_16_81060*
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_8100920
.batch_normalization_16/StatefulPartitionedCallА
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_26_81063conv2d_26_81065*
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_803942#
!conv2d_26/StatefulPartitionedCallЂ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_17_81143batch_normalization_17_81145batch_normalization_17_81147batch_normalization_17_81149*
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_8109820
.batch_normalization_17/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
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
@__inference_add_8_layer_call_and_return_conditional_losses_811582
add_8/PartitionedCallр
activation_8/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
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
G__inference_activation_8_layer_call_and_return_conditional_losses_811722
activation_8/PartitionedCall
*global_average_pooling2d_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_805372,
*global_average_pooling2d_2/PartitionedCallф
flatten_2/PartitionedCallPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_811872
flatten_2/PartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_81233dense_4_81235*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_812222!
dense_4/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_81276dense_5_81278*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_812652!
dense_5/StatefulPartitionedCallР
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_80547*&
_output_shapes
:*
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_18/kernel/Regularizer/SquareЁ
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/ConstТ
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_18/kernel/Regularizer/mul/xФ
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_18/kernel/Regularizer/add/xС
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/addА
0conv2d_18/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_80549*
_output_shapes
:*
dtype022
0conv2d_18/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_18/bias/Regularizer/SquareSquare8conv2d_18/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_18/bias/Regularizer/Square
 conv2d_18/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_18/bias/Regularizer/ConstК
conv2d_18/bias/Regularizer/SumSum%conv2d_18/bias/Regularizer/Square:y:0)conv2d_18/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/Sum
 conv2d_18/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_18/bias/Regularizer/mul/xМ
conv2d_18/bias/Regularizer/mulMul)conv2d_18/bias/Regularizer/mul/x:output:0'conv2d_18/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/mul
 conv2d_18/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_18/bias/Regularizer/add/xЙ
conv2d_18/bias/Regularizer/addAddV2)conv2d_18/bias/Regularizer/add/x:output:0"conv2d_18/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_18/bias/Regularizer/addР
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_80552*&
_output_shapes
:*
dtype024
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_19/kernel/Regularizer/SquareЁ
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_19/kernel/Regularizer/ConstТ
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/Sum
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_19/kernel/Regularizer/mul/xФ
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/mul
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_19/kernel/Regularizer/add/xС
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_19/kernel/Regularizer/addА
0conv2d_19/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_19_80554*
_output_shapes
:*
dtype022
0conv2d_19/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_19/bias/Regularizer/SquareSquare8conv2d_19/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_19/bias/Regularizer/Square
 conv2d_19/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_19/bias/Regularizer/ConstК
conv2d_19/bias/Regularizer/SumSum%conv2d_19/bias/Regularizer/Square:y:0)conv2d_19/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/Sum
 conv2d_19/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_19/bias/Regularizer/mul/xМ
conv2d_19/bias/Regularizer/mulMul)conv2d_19/bias/Regularizer/mul/x:output:0'conv2d_19/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/mul
 conv2d_19/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_19/bias/Regularizer/add/xЙ
conv2d_19/bias/Regularizer/addAddV2)conv2d_19/bias/Regularizer/add/x:output:0"conv2d_19/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_19/bias/Regularizer/addР
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_80641*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addА
0conv2d_20/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_80643*
_output_shapes
:*
dtype022
0conv2d_20/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_20/bias/Regularizer/SquareSquare8conv2d_20/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_20/bias/Regularizer/Square
 conv2d_20/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_20/bias/Regularizer/ConstК
conv2d_20/bias/Regularizer/SumSum%conv2d_20/bias/Regularizer/Square:y:0)conv2d_20/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/Sum
 conv2d_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_20/bias/Regularizer/mul/xМ
conv2d_20/bias/Regularizer/mulMul)conv2d_20/bias/Regularizer/mul/x:output:0'conv2d_20/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/mul
 conv2d_20/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_20/bias/Regularizer/add/xЙ
conv2d_20/bias/Regularizer/addAddV2)conv2d_20/bias/Regularizer/add/x:output:0"conv2d_20/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_20/bias/Regularizer/addР
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_80758*&
_output_shapes
: *
dtype024
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_21/kernel/Regularizer/SquareЁ
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_21/kernel/Regularizer/ConstТ
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/Sum
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_21/kernel/Regularizer/mul/xФ
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/mul
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_21/kernel/Regularizer/add/xС
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_21/kernel/Regularizer/addА
0conv2d_21/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_21_80760*
_output_shapes
: *
dtype022
0conv2d_21/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_21/bias/Regularizer/SquareSquare8conv2d_21/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_21/bias/Regularizer/Square
 conv2d_21/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_21/bias/Regularizer/ConstК
conv2d_21/bias/Regularizer/SumSum%conv2d_21/bias/Regularizer/Square:y:0)conv2d_21/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/Sum
 conv2d_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_21/bias/Regularizer/mul/xМ
conv2d_21/bias/Regularizer/mulMul)conv2d_21/bias/Regularizer/mul/x:output:0'conv2d_21/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/mul
 conv2d_21/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_21/bias/Regularizer/add/xЙ
conv2d_21/bias/Regularizer/addAddV2)conv2d_21/bias/Regularizer/add/x:output:0"conv2d_21/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_21/bias/Regularizer/addР
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_80763*&
_output_shapes
:  *
dtype024
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_22/kernel/Regularizer/SquareЁ
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_22/kernel/Regularizer/ConstТ
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/Sum
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_22/kernel/Regularizer/mul/xФ
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/mul
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_22/kernel/Regularizer/add/xС
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_22/kernel/Regularizer/addА
0conv2d_22/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_80765*
_output_shapes
: *
dtype022
0conv2d_22/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_22/bias/Regularizer/SquareSquare8conv2d_22/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_22/bias/Regularizer/Square
 conv2d_22/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_22/bias/Regularizer/ConstК
conv2d_22/bias/Regularizer/SumSum%conv2d_22/bias/Regularizer/Square:y:0)conv2d_22/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/Sum
 conv2d_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_22/bias/Regularizer/mul/xМ
conv2d_22/bias/Regularizer/mulMul)conv2d_22/bias/Regularizer/mul/x:output:0'conv2d_22/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/mul
 conv2d_22/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_22/bias/Regularizer/add/xЙ
conv2d_22/bias/Regularizer/addAddV2)conv2d_22/bias/Regularizer/add/x:output:0"conv2d_22/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_22/bias/Regularizer/addР
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_80852*&
_output_shapes
:  *
dtype024
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_23/kernel/Regularizer/SquareЁ
"conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_23/kernel/Regularizer/ConstТ
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/Sum
"conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_23/kernel/Regularizer/mul/xФ
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/mul
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_23/kernel/Regularizer/add/xС
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_23/kernel/Regularizer/addА
0conv2d_23/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_80854*
_output_shapes
: *
dtype022
0conv2d_23/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_23/bias/Regularizer/SquareSquare8conv2d_23/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_23/bias/Regularizer/Square
 conv2d_23/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_23/bias/Regularizer/ConstК
conv2d_23/bias/Regularizer/SumSum%conv2d_23/bias/Regularizer/Square:y:0)conv2d_23/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/Sum
 conv2d_23/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_23/bias/Regularizer/mul/xМ
conv2d_23/bias/Regularizer/mulMul)conv2d_23/bias/Regularizer/mul/x:output:0'conv2d_23/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/mul
 conv2d_23/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_23/bias/Regularizer/add/xЙ
conv2d_23/bias/Regularizer/addAddV2)conv2d_23/bias/Regularizer/add/x:output:0"conv2d_23/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_23/bias/Regularizer/addР
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_80969*&
_output_shapes
: @*
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_24/kernel/Regularizer/SquareЁ
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/ConstТ
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_24/kernel/Regularizer/mul/xФ
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_24/kernel/Regularizer/add/xС
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/addА
0conv2d_24/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_80971*
_output_shapes
:@*
dtype022
0conv2d_24/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_24/bias/Regularizer/SquareSquare8conv2d_24/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_24/bias/Regularizer/Square
 conv2d_24/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_24/bias/Regularizer/ConstК
conv2d_24/bias/Regularizer/SumSum%conv2d_24/bias/Regularizer/Square:y:0)conv2d_24/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/Sum
 conv2d_24/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_24/bias/Regularizer/mul/xМ
conv2d_24/bias/Regularizer/mulMul)conv2d_24/bias/Regularizer/mul/x:output:0'conv2d_24/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/mul
 conv2d_24/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_24/bias/Regularizer/add/xЙ
conv2d_24/bias/Regularizer/addAddV2)conv2d_24/bias/Regularizer/add/x:output:0"conv2d_24/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_24/bias/Regularizer/addР
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_80974*&
_output_shapes
:@@*
dtype024
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_25/kernel/Regularizer/SquareЁ
"conv2d_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_25/kernel/Regularizer/ConstТ
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/Sum
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_25/kernel/Regularizer/mul/xФ
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/mul
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_25/kernel/Regularizer/add/xС
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_25/kernel/Regularizer/addА
0conv2d_25/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_25_80976*
_output_shapes
:@*
dtype022
0conv2d_25/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_25/bias/Regularizer/SquareSquare8conv2d_25/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_25/bias/Regularizer/Square
 conv2d_25/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_25/bias/Regularizer/ConstК
conv2d_25/bias/Regularizer/SumSum%conv2d_25/bias/Regularizer/Square:y:0)conv2d_25/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/Sum
 conv2d_25/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_25/bias/Regularizer/mul/xМ
conv2d_25/bias/Regularizer/mulMul)conv2d_25/bias/Regularizer/mul/x:output:0'conv2d_25/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/mul
 conv2d_25/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_25/bias/Regularizer/add/xЙ
conv2d_25/bias/Regularizer/addAddV2)conv2d_25/bias/Regularizer/add/x:output:0"conv2d_25/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_25/bias/Regularizer/addР
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_81063*&
_output_shapes
:@@*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_26/kernel/Regularizer/SquareЁ
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstТ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_26/kernel/Regularizer/mul/xФ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mul
"conv2d_26/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_26/kernel/Regularizer/add/xС
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/addА
0conv2d_26/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_81065*
_output_shapes
:@*
dtype022
0conv2d_26/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_26/bias/Regularizer/SquareSquare8conv2d_26/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_26/bias/Regularizer/Square
 conv2d_26/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_26/bias/Regularizer/ConstК
conv2d_26/bias/Regularizer/SumSum%conv2d_26/bias/Regularizer/Square:y:0)conv2d_26/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/Sum
 conv2d_26/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_26/bias/Regularizer/mul/xМ
conv2d_26/bias/Regularizer/mulMul)conv2d_26/bias/Regularizer/mul/x:output:0'conv2d_26/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/mul
 conv2d_26/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_26/bias/Regularizer/add/xЙ
conv2d_26/bias/Regularizer/addAddV2)conv2d_26/bias/Regularizer/add/x:output:0"conv2d_26/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_26/bias/Regularizer/addГ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_81233*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЋ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_81235*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addГ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_81276*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЊ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_81278*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЊ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*ш
_input_shapesж
г:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_3:
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84720

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
Ў
Љ
6__inference_batch_normalization_13_layer_call_fn_84216

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
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_806762
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
і
Љ
6__inference_batch_normalization_12_layer_call_fn_83963

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_795952
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
ї
o
__inference_loss_fn_4_85219?
;conv2d_20_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_20_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_20/kernel/Regularizer/SquareЁ
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/ConstТ
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/Sum
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_20/kernel/Regularizer/mul/xФ
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_20/kernel/Regularizer/add/xС
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/addg
IdentityIdentity$conv2d_20/kernel/Regularizer/add:z:0*
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
њ
Д
'__inference_model_2_layer_call_fn_82544
input_3
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
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
B__inference_model_2_layer_call_and_return_conditional_losses_824492
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
_user_specified_name	input_3:
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
Ў
Љ
6__inference_batch_normalization_12_layer_call_fn_84038

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
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_805872
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


Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_79789

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
Ў
Љ
6__inference_batch_normalization_15_layer_call_fn_84610

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
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_808872
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
Ў
Љ
6__inference_batch_normalization_16_layer_call_fn_84751

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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_810092
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
Ў
n
__inference_loss_fn_18_85401=
9dense_4_kernel_regularizer_square_readvariableop_resource
identityп
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpД
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/adde
IdentityIdentity"dense_4/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: "ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_default
C
input_38
serving_default_input_3:0џџџџџџџџџ22;
dense_50
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
Яѓ
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
М_default_save_signature"ьы
_tf_keras_modelбы{"class_name": "Model", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["conv2d_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}], ["conv2d_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_21", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["conv2d_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_14", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_23", "inbound_nodes": [[["batch_normalization_14", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_23", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}], ["conv2d_21", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "name": "add_8", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}], ["conv2d_24", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["add_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_2", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["global_average_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["conv2d_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}], ["conv2d_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_21", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["conv2d_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_14", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_23", "inbound_nodes": [[["batch_normalization_14", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_23", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}], ["conv2d_21", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "name": "add_8", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}], ["conv2d_24", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["add_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_2", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["global_average_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_5", 0, 0]]}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
а


 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}}
а


&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
	
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
Т__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
в


5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
	
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
Ц__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}

Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 16]}, {"class_name": "TensorShape", "items": [null, 50, 50, 16]}]}
Д
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
а


Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
а


Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
	
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
а__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
в


akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
	
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
д__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}

pregularization_losses
qtrainable_variables
r	variables
s	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 32]}, {"class_name": "TensorShape", "items": [null, 50, 50, 32]}]}
Д
tregularization_losses
utrainable_variables
v	variables
w	keras_api
+з&call_and_return_all_conditional_losses
и__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}
а


xkernel
ybias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
+й&call_and_return_all_conditional_losses
к__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
д


~kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+л&call_and_return_all_conditional_losses
м__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
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
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
и

kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
+п&call_and_return_all_conditional_losses
р__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
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
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}

regularization_losses
trainable_variables
	variables
	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 64]}, {"class_name": "TensorShape", "items": [null, 50, 50, 64]}]}
И
 regularization_losses
Ёtrainable_variables
Ђ	variables
Ѓ	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}}
њ
Єregularization_losses
Ѕtrainable_variables
І	variables
Ї	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"х
_tf_keras_layerЫ{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Щ
Јregularization_losses
Љtrainable_variables
Њ	variables
Ћ	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"Д
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
р
Ќkernel
	­bias
Ўregularization_losses
Џtrainable_variables
А	variables
Б	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"Г
_tf_keras_layer{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
у
Вkernel
	Гbias
Дregularization_losses
Еtrainable_variables
Ж	variables
З	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"Ж
_tf_keras_layer{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
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
*:(2conv2d_18/kernel
:2conv2d_18/bias
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
*:(2conv2d_19/kernel
:2conv2d_19/bias
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
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
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
*:(2conv2d_20/kernel
:2conv2d_20/bias
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
*:(2batch_normalization_13/gamma
):'2batch_normalization_13/beta
2:0 (2"batch_normalization_13/moving_mean
6:4 (2&batch_normalization_13/moving_variance
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
*:( 2conv2d_21/kernel
: 2conv2d_21/bias
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
*:(  2conv2d_22/kernel
: 2conv2d_22/bias
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
*:( 2batch_normalization_14/gamma
):' 2batch_normalization_14/beta
2:0  (2"batch_normalization_14/moving_mean
6:4  (2&batch_normalization_14/moving_variance
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
*:(  2conv2d_23/kernel
: 2conv2d_23/bias
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
*:( 2batch_normalization_15/gamma
):' 2batch_normalization_15/beta
2:0  (2"batch_normalization_15/moving_mean
6:4  (2&batch_normalization_15/moving_variance
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
*:( @2conv2d_24/kernel
:@2conv2d_24/bias
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
*:(@@2conv2d_25/kernel
:@2conv2d_25/bias
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
*:(@2batch_normalization_16/gamma
):'@2batch_normalization_16/beta
2:0@ (2"batch_normalization_16/moving_mean
6:4@ (2&batch_normalization_16/moving_variance
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
*:(@@2conv2d_26/kernel
:@2conv2d_26/bias
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
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
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
!:	@2dense_4/kernel
:2dense_4/bias
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
!:	2dense_5/kernel
:2dense_5/bias
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
B__inference_model_2_layer_call_and_return_conditional_losses_83316
B__inference_model_2_layer_call_and_return_conditional_losses_83663
B__inference_model_2_layer_call_and_return_conditional_losses_81458
B__inference_model_2_layer_call_and_return_conditional_losses_81755Р
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
'__inference_model_2_layer_call_fn_83857
'__inference_model_2_layer_call_fn_82150
'__inference_model_2_layer_call_fn_82544
'__inference_model_2_layer_call_fn_83760Р
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
 __inference__wrapped_model_79436О
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
input_3џџџџџџџџџ22
Ѓ2 
D__inference_conv2d_18_layer_call_and_return_conditional_losses_79463з
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
2
)__inference_conv2d_18_layer_call_fn_79473з
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_79501з
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
)__inference_conv2d_19_layer_call_fn_79511з
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
2
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_83950
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_83932
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_84025
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_84007Д
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
6__inference_batch_normalization_12_layer_call_fn_84038
6__inference_batch_normalization_12_layer_call_fn_84051
6__inference_batch_normalization_12_layer_call_fn_83963
6__inference_batch_normalization_12_layer_call_fn_83976Д
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_79664з
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
)__inference_conv2d_20_layer_call_fn_79674з
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
2
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84128
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84185
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84203
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84110Д
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
6__inference_batch_normalization_13_layer_call_fn_84216
6__inference_batch_normalization_13_layer_call_fn_84229
6__inference_batch_normalization_13_layer_call_fn_84141
6__inference_batch_normalization_13_layer_call_fn_84154Д
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
@__inference_add_6_layer_call_and_return_conditional_losses_84235Ђ
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
%__inference_add_6_layer_call_fn_84241Ђ
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
G__inference_activation_6_layer_call_and_return_conditional_losses_84246Ђ
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
,__inference_activation_6_layer_call_fn_84251Ђ
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_79828з
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
)__inference_conv2d_21_layer_call_fn_79838з
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_79866з
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
)__inference_conv2d_22_layer_call_fn_79876з
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
2
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84401
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84344
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84326
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84419Д
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
6__inference_batch_normalization_14_layer_call_fn_84357
6__inference_batch_normalization_14_layer_call_fn_84432
6__inference_batch_normalization_14_layer_call_fn_84370
6__inference_batch_normalization_14_layer_call_fn_84445Д
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_80029з
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
)__inference_conv2d_23_layer_call_fn_80039з
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
2
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84522
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84504
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84597
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84579Д
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
6__inference_batch_normalization_15_layer_call_fn_84610
6__inference_batch_normalization_15_layer_call_fn_84548
6__inference_batch_normalization_15_layer_call_fn_84535
6__inference_batch_normalization_15_layer_call_fn_84623Д
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
@__inference_add_7_layer_call_and_return_conditional_losses_84629Ђ
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
%__inference_add_7_layer_call_fn_84635Ђ
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
G__inference_activation_7_layer_call_and_return_conditional_losses_84640Ђ
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
,__inference_activation_7_layer_call_fn_84645Ђ
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_80193з
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
)__inference_conv2d_24_layer_call_fn_80203з
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_80231з
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
)__inference_conv2d_25_layer_call_fn_80241з
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84738
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84720
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84795
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84813Д
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
6__inference_batch_normalization_16_layer_call_fn_84826
6__inference_batch_normalization_16_layer_call_fn_84764
6__inference_batch_normalization_16_layer_call_fn_84839
6__inference_batch_normalization_16_layer_call_fn_84751Д
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_80394з
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
)__inference_conv2d_26_layer_call_fn_80404з
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84916
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84991
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84898
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84973Д
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
6__inference_batch_normalization_17_layer_call_fn_85004
6__inference_batch_normalization_17_layer_call_fn_85017
6__inference_batch_normalization_17_layer_call_fn_84929
6__inference_batch_normalization_17_layer_call_fn_84942Д
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
@__inference_add_8_layer_call_and_return_conditional_losses_85023Ђ
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
%__inference_add_8_layer_call_fn_85029Ђ
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
G__inference_activation_8_layer_call_and_return_conditional_losses_85034Ђ
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
,__inference_activation_8_layer_call_fn_85039Ђ
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_80537р
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
:__inference_global_average_pooling2d_2_layer_call_fn_80543р
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_85045Ђ
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
)__inference_flatten_2_layer_call_fn_85050Ђ
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
B__inference_dense_4_layer_call_and_return_conditional_losses_85093Ђ
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
'__inference_dense_4_layer_call_fn_85102Ђ
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
B__inference_dense_5_layer_call_and_return_conditional_losses_85145Ђ
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
'__inference_dense_5_layer_call_fn_85154Ђ
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
__inference_loss_fn_0_85167
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
__inference_loss_fn_1_85180
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
__inference_loss_fn_2_85193
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
__inference_loss_fn_3_85206
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
__inference_loss_fn_4_85219
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
__inference_loss_fn_5_85232
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
__inference_loss_fn_6_85245
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
__inference_loss_fn_7_85258
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
__inference_loss_fn_8_85271
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
__inference_loss_fn_9_85284
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
__inference_loss_fn_10_85297
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
__inference_loss_fn_11_85310
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
__inference_loss_fn_12_85323
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
__inference_loss_fn_13_85336
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
__inference_loss_fn_14_85349
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
__inference_loss_fn_15_85362
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
__inference_loss_fn_16_85375
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
__inference_loss_fn_17_85388
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
__inference_loss_fn_18_85401
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
__inference_loss_fn_19_85414
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
__inference_loss_fn_20_85427
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
__inference_loss_fn_21_85440
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
#__inference_signature_wrapper_82891input_3а
 __inference__wrapped_model_79436Ћ< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ8Ђ5
.Ђ+
)&
input_3џџџџџџџџџ22
Њ "1Њ.
,
dense_5!
dense_5џџџџџџџџџГ
G__inference_activation_6_layer_call_and_return_conditional_losses_84246h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ "-Ђ*
# 
0џџџџџџџџџ22
 
,__inference_activation_6_layer_call_fn_84251[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ " џџџџџџџџџ22Г
G__inference_activation_7_layer_call_and_return_conditional_losses_84640h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 
,__inference_activation_7_layer_call_fn_84645[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22 
Њ " џџџџџџџџџ22 Г
G__inference_activation_8_layer_call_and_return_conditional_losses_85034h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22@
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 
,__inference_activation_8_layer_call_fn_85039[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22@
Њ " џџџџџџџџџ22@р
@__inference_add_6_layer_call_and_return_conditional_losses_84235jЂg
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
%__inference_add_6_layer_call_fn_84241jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22
*'
inputs/1џџџџџџџџџ22
Њ " џџџџџџџџџ22р
@__inference_add_7_layer_call_and_return_conditional_losses_84629jЂg
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
%__inference_add_7_layer_call_fn_84635jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22 
*'
inputs/1џџџџџџџџџ22 
Њ " џџџџџџџџџ22 р
@__inference_add_8_layer_call_and_return_conditional_losses_85023jЂg
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
%__inference_add_8_layer_call_fn_85029jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22@
*'
inputs/1џџџџџџџџџ22@
Њ " џџџџџџџџџ22@ь
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_83932-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_83950-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_84007r-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ч
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_84025r-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ф
6__inference_batch_normalization_12_layer_call_fn_83963-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
6__inference_batch_normalization_12_layer_call_fn_83976-./0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
6__inference_batch_normalization_12_layer_call_fn_84038e-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ " џџџџџџџџџ22
6__inference_batch_normalization_12_layer_call_fn_84051e-./0;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ " џџџџџџџџџ22ь
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84110<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84128<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84185r<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ч
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_84203r<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ф
6__inference_batch_normalization_13_layer_call_fn_84141<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
6__inference_batch_normalization_13_layer_call_fn_84154<=>?MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
6__inference_batch_normalization_13_layer_call_fn_84216e<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ " џџџџџџџџџ22
6__inference_batch_normalization_13_layer_call_fn_84229e<=>?;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ " џџџџџџџџџ22ь
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84326YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ь
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84344YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ч
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84401rYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ч
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_84419rYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ф
6__inference_batch_normalization_14_layer_call_fn_84357YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ф
6__inference_batch_normalization_14_layer_call_fn_84370YZ[\MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
6__inference_batch_normalization_14_layer_call_fn_84432eYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ " џџџџџџџџџ22 
6__inference_batch_normalization_14_layer_call_fn_84445eYZ[\;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ " џџџџџџџџџ22 ь
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84504hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ь
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84522hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ч
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84579rhijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ч
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_84597rhijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ф
6__inference_batch_normalization_15_layer_call_fn_84535hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ф
6__inference_batch_normalization_15_layer_call_fn_84548hijkMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
6__inference_batch_normalization_15_layer_call_fn_84610ehijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ " џџџџџџџџџ22 
6__inference_batch_normalization_15_layer_call_fn_84623ehijk;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ " џџџџџџџџџ22 Ы
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84720v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ы
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84738v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 №
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84795MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 №
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_84813MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ѓ
6__inference_batch_normalization_16_layer_call_fn_84751i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ " џџџџџџџџџ22@Ѓ
6__inference_batch_normalization_16_layer_call_fn_84764i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ " џџџџџџџџџ22@Ш
6__inference_batch_normalization_16_layer_call_fn_84826MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ш
6__inference_batch_normalization_16_layer_call_fn_84839MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@№
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84898MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 №
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84916MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ы
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84973v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ы
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_84991v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ш
6__inference_batch_normalization_17_layer_call_fn_84929MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ш
6__inference_batch_normalization_17_layer_call_fn_84942MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ѓ
6__inference_batch_normalization_17_layer_call_fn_85004i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ " џџџџџџџџџ22@Ѓ
6__inference_batch_normalization_17_layer_call_fn_85017i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ " џџџџџџџџџ22@й
D__inference_conv2d_18_layer_call_and_return_conditional_losses_79463 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_18_layer_call_fn_79473 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
D__inference_conv2d_19_layer_call_and_return_conditional_losses_79501&'IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_19_layer_call_fn_79511&'IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
D__inference_conv2d_20_layer_call_and_return_conditional_losses_7966456IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_20_layer_call_fn_7967456IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
D__inference_conv2d_21_layer_call_and_return_conditional_losses_79828LMIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Б
)__inference_conv2d_21_layer_call_fn_79838LMIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
D__inference_conv2d_22_layer_call_and_return_conditional_losses_79866RSIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Б
)__inference_conv2d_22_layer_call_fn_79876RSIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
D__inference_conv2d_23_layer_call_and_return_conditional_losses_80029abIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Б
)__inference_conv2d_23_layer_call_fn_80039abIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ й
D__inference_conv2d_24_layer_call_and_return_conditional_losses_80193xyIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Б
)__inference_conv2d_24_layer_call_fn_80203xyIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@й
D__inference_conv2d_25_layer_call_and_return_conditional_losses_80231~IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Б
)__inference_conv2d_25_layer_call_fn_80241~IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@л
D__inference_conv2d_26_layer_call_and_return_conditional_losses_80394IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Г
)__inference_conv2d_26_layer_call_fn_80404IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ѕ
B__inference_dense_4_layer_call_and_return_conditional_losses_85093_Ќ­/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 }
'__inference_dense_4_layer_call_fn_85102RЌ­/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЅ
B__inference_dense_5_layer_call_and_return_conditional_losses_85145_ВГ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
'__inference_dense_5_layer_call_fn_85154RВГ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ 
D__inference_flatten_2_layer_call_and_return_conditional_losses_85045X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 x
)__inference_flatten_2_layer_call_fn_85050K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@о
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_80537RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Е
:__inference_global_average_pooling2d_2_layer_call_fn_80543wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ:
__inference_loss_fn_0_85167 Ђ

Ђ 
Њ " ;
__inference_loss_fn_10_85297aЂ

Ђ 
Њ " ;
__inference_loss_fn_11_85310bЂ

Ђ 
Њ " ;
__inference_loss_fn_12_85323xЂ

Ђ 
Њ " ;
__inference_loss_fn_13_85336yЂ

Ђ 
Њ " ;
__inference_loss_fn_14_85349~Ђ

Ђ 
Њ " ;
__inference_loss_fn_15_85362Ђ

Ђ 
Њ " <
__inference_loss_fn_16_85375Ђ

Ђ 
Њ " <
__inference_loss_fn_17_85388Ђ

Ђ 
Њ " <
__inference_loss_fn_18_85401ЌЂ

Ђ 
Њ " <
__inference_loss_fn_19_85414­Ђ

Ђ 
Њ " :
__inference_loss_fn_1_85180!Ђ

Ђ 
Њ " <
__inference_loss_fn_20_85427ВЂ

Ђ 
Њ " <
__inference_loss_fn_21_85440ГЂ

Ђ 
Њ " :
__inference_loss_fn_2_85193&Ђ

Ђ 
Њ " :
__inference_loss_fn_3_85206'Ђ

Ђ 
Њ " :
__inference_loss_fn_4_852195Ђ

Ђ 
Њ " :
__inference_loss_fn_5_852326Ђ

Ђ 
Њ " :
__inference_loss_fn_6_85245LЂ

Ђ 
Њ " :
__inference_loss_fn_7_85258MЂ

Ђ 
Њ " :
__inference_loss_fn_8_85271RЂ

Ђ 
Њ " :
__inference_loss_fn_9_85284SЂ

Ђ 
Њ " ю
B__inference_model_2_layer_call_and_return_conditional_losses_81458Ї< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_3џџџџџџџџџ22
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ю
B__inference_model_2_layer_call_and_return_conditional_losses_81755Ї< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_3џџџџџџџџџ22
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 э
B__inference_model_2_layer_call_and_return_conditional_losses_83316І< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
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
B__inference_model_2_layer_call_and_return_conditional_losses_83663І< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
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
'__inference_model_2_layer_call_fn_82150< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_3џџџџџџџџџ22
p

 
Њ "џџџџџџџџџЦ
'__inference_model_2_layer_call_fn_82544< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ@Ђ=
6Ђ3
)&
input_3џџџџџџџџџ22
p 

 
Њ "џџџџџџџџџХ
'__inference_model_2_layer_call_fn_83760< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p

 
Њ "џџџџџџџџџХ
'__inference_model_2_layer_call_fn_83857< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p 

 
Њ "џџџџџџџџџо
#__inference_signature_wrapper_82891Ж< !&'-./056<=>?LMRSYZ[\abhijkxy~Ќ­ВГCЂ@
Ђ 
9Њ6
4
input_3)&
input_3џџџџџџџџџ22"1Њ.
,
dense_5!
dense_5џџџџџџџџџ